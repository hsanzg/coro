//! Unwinding support for coroutines.
//!
//! Most subroutines in this module are based on the great [blog post]
//! on faster exception handling by Alisa Sireneva; her [`lithium`]
//! crate was also a source of inspiration.
//!
//! [blog post]: https://purplesyringa.moe/blog/bringing-faster-exceptions-to-rust/
//! [`lithium`]: https://docs.rs/lithium/latest/lithium/
#[cfg_attr(panic = "unwind", path = "itanium.rs")]
#[cfg_attr(not(panic = "unwind"), path = "abort.rs")]
mod imp;

pub use self::imp::*;
use crate::stack::StackPtr;
use alloc::boxed::Box;
use core::any::Any;
use core::fmt::{Display, Formatter};
use core::hint::{cold_path, unreachable_unchecked};
use core::mem::ManuallyDrop;
use core::ptr;

/// The exception data associated with a Rust panic.
type PanicPayload = Box<dyn Any + Send>;

/// The cause of a stack unwinding operation.
pub enum Unwind {
  /// Via [`start_unwind`].
  Forced,
  /// A Rust panic or manual unwind.
  Panic(PanicPayload),
}

#[must_use]
#[repr(transparent)]
pub struct Completion(*mut ());

impl Completion {
  const UNWIND: usize = 1;

  pub fn into_raw(self) -> usize {
    ManuallyDrop::new(self).0.expose_provenance()
  }

  #[inline(always)]
  pub unsafe fn from_raw(raw: usize) -> Self {
    Self(ptr::with_exposed_provenance_mut(raw))
  }

  pub fn is_ok(&self) -> bool {
    self.0.addr() & Self::UNWIND == 0
  }

  pub fn is_suspended(&self) -> bool {
    self.0.is_null()
  }

  /// # Safety
  ///
  /// Assumes that `self.is_ok()` is `false`.
  #[cold]
  #[inline(never)]
  pub unsafe fn maybe_rethrow(self) {
    debug_assert!(!self.is_ok());
    let ptr = ManuallyDrop::new(self)
      .0
      .map_addr(|a| a & !Self::UNWIND)
      .cast::<PanicPayload>();
    if !ptr.is_null() {
      cold_path();
      let payload = *unsafe { Box::from_raw(ptr) };
      resume(payload);
    }
  }

  #[inline]
  pub fn ret_addr(self) -> *mut () {
    let ptr = self.0;
    if self.is_ok() {
      core::mem::forget(self);
      ptr
    } else {
      unsafe { self.maybe_rethrow() };
      unsafe { unreachable_unchecked() };
    }
  }
}

impl From<CatchResult<StackPtr>> for Completion {
  fn from(res: CatchResult<StackPtr>) -> Self {
    Self(match res {
      Ok(ret_addr) => {
        debug_assert_eq!(ret_addr.addr().get() & Self::UNWIND, 0);
        ret_addr.as_ptr()
      }
      Err(Unwind::Forced) => ptr::without_provenance_mut(Self::UNWIND),
      Err(Unwind::Panic(payload)) => {
        cold_path();
        const {
          assert!(align_of::<PanicPayload>() > Self::UNWIND);
        }
        Box::into_raw(Box::new(payload))
          .cast::<()>()
          .map_addr(|a| a | Self::UNWIND)
      }
    })
  }
}

impl Drop for Completion {
  fn drop(&mut self) {
    let ptr = self.0;
    if !self.is_ok() {
      let ptr = ptr.map_addr(|a| a & !Self::UNWIND).cast::<PanicPayload>();
      if !ptr.is_null() {
        cold_path();
        drop(unsafe { Box::from_raw(ptr) });
      }
    }
  }
}

pub type CatchResult<R> = Result<R, Unwind>;

pub fn catch<F: FnOnce() -> R, R>(f: F) -> CatchResult<R> {
  union Data<F, T> {
    f: ManuallyDrop<F>,
    ok: ManuallyDrop<T>,
    err: ManuallyDrop<Unwind>,
  }

  let mut data = Data {
    f: ManuallyDrop::new(f),
  };

  #[inline]
  fn do_call<F: FnOnce() -> R, R>(data: *mut Data<F, R>) {
    let data = unsafe { &mut *data };
    let f = unsafe { ManuallyDrop::take(&mut data.f) };
    data.ok = ManuallyDrop::new(f());
  }

  #[inline]
  #[rustc_nounwind]
  fn do_catch<F: FnOnce() -> T, T>(data: *mut Data<F, T>, payload: *mut u8) {
    let err = if unsafe { is_unwind_exception(payload) } {
      Unwind::Forced
    } else {
      cold_path();
      Unwind::Panic(unsafe { __rust_panic_cleanup(payload) })
    };
    unsafe { (*data).err = ManuallyDrop::new(err) };
  }

  if unsafe { !core::intrinsics::catch_unwind(do_call::<F, R>, &raw mut data, do_catch::<F, R>) } {
    Ok(ManuallyDrop::into_inner(unsafe { data.ok }))
  } else {
    Err(ManuallyDrop::into_inner(unsafe { data.err }))
  }
}

unsafe extern "Rust" {
  #[rustc_std_internal_symbol]
  fn __rust_panic_cleanup(payload: *mut u8) -> Box<dyn Any + Send + 'static>;

  #[rustc_std_internal_symbol]
  safe fn __rust_start_panic(payload: &mut dyn alloc::panicking::PanicPayload) -> u32;
}

pub fn resume(payload: Box<dyn Any + Send>) -> ! {
  struct RewrapBox(Box<dyn Any + Send>);

  impl alloc::panicking::PanicPayload for RewrapBox {
    fn take_box(&mut self) -> Box<dyn Any + Send> {
      core::mem::replace(&mut self.0, Box::new(()))
    }

    fn get(&mut self) -> &(dyn Any + Send) {
      &*self.0
    }
  }

  impl Display for RewrapBox {
    fn fmt(&self, _f: &mut Formatter<'_>) -> core::fmt::Result {
      unreachable!()
    }
  }

  let _code = __rust_start_panic(&mut RewrapBox(payload));
  // todo: write `code`.
  abort("failed to initiate panic");
}

#[cold]
pub fn abort(msg: &'static str) -> ! {
  core::panicking::panic_nounwind_nobacktrace(msg);
}
