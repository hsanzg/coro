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
use core::hint::{cold_path, unreachable_unchecked};
use core::mem::ManuallyDrop;
use core::ptr;

pub type CatchResult<R> = Result<R, Exception>;

pub fn catch<F: FnOnce() -> R, R>(f: F) -> CatchResult<R> {
  // https://github.com/rust-lang/rust/blob/df87fa6604e33760d4703cccc777a2281bf15a2d/library/std/src/panicking.rs#L498
  union Data<F, T> {
    f: ManuallyDrop<F>,
    ok: ManuallyDrop<T>,
    ex: Exception,
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
  fn do_catch<F: FnOnce() -> T, T>(data: *mut Data<F, T>, ex: *mut u8) {
    unsafe { (*data).ex = ex.cast() };
  }

  if unsafe { !core::intrinsics::catch_unwind(do_call, &raw mut data, do_catch) } {
    Ok(ManuallyDrop::into_inner(unsafe { data.ok }))
  } else {
    Err(unsafe { data.ex })
  }
}

#[must_use]
#[repr(transparent)]
pub struct Completion(*mut ());

impl Completion {
  const UNWOUND: usize = 1;

  pub fn into_raw(self) -> usize {
    ManuallyDrop::new(self).0.expose_provenance()
  }

  #[inline(always)]
  pub unsafe fn from_raw(raw: usize) -> Self {
    Self(ptr::with_exposed_provenance_mut(raw))
  }

  pub fn is_ok(&self) -> bool {
    self.0.addr() & Self::UNWOUND == 0
  }

  pub fn is_suspended(&self) -> bool {
    self.0.is_null()
  }

  fn exception(&self) -> Exception {
    self.0.map_addr(|a| a & !Self::UNWOUND).cast()
  }

  /// # Safety
  ///
  /// Assumes that `self.is_ok()` is `false`.
  #[cold]
  #[inline(never)]
  pub unsafe fn maybe_rethrow(self) {
    debug_assert!(!self.is_ok());
    let ex = ManuallyDrop::new(self).exception();
    if unsafe { !is_coro_unwind(ex) } {
      cold_path();
      unsafe { rethrow(ex) };
    }
  }

  #[inline]
  pub fn ret_addr(self) -> *mut () {
    if self.is_ok() {
      ManuallyDrop::new(self).0
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
        debug_assert_eq!(ret_addr.addr().get() & Self::UNWOUND, 0);
        ret_addr.as_ptr()
      }
      Err(ex) => {
        // todo: add comment about double-word alignment of exception types.
        ex.cast::<()>().map_addr(|a| a | Self::UNWOUND)
      }
    })
  }
}

impl Drop for Completion {
  fn drop(&mut self) {
    if !self.is_ok() {
      let ex = self.exception();
      if unsafe { !is_coro_unwind(ex) } {
        cold_path();
        unsafe { destroy(ex) };
      }
    }
  }
}

#[cold]
pub fn abort(msg: &'static str) -> ! {
  core::panicking::panic_nounwind_nobacktrace(msg);
}
