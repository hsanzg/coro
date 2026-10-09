#[cfg_attr(not(windows), path = "itanium.rs")]
mod imp;

pub use self::imp::*;
use alloc::boxed::Box;
use alloc::panicking::PanicPayload;
use core::any::Any;
use core::fmt::Display;
use core::hint::cold_path;
use core::mem::ManuallyDrop;
use core::process::abort_immediate;

pub enum Exception {
  Panic(Box<dyn Any + Send + 'static>),
  ForcedUnwind,
}

pub fn catch<T, F: FnOnce() -> T>(f: F) -> Result<T, Exception> {
  union Data<F, T> {
    f: ManuallyDrop<F>,
    ok: ManuallyDrop<T>,
    err: ManuallyDrop<Exception>,
  }

  let mut data = Data {
    f: ManuallyDrop::new(f),
  };

  #[inline]
  fn do_call<F: FnOnce() -> T, T>(data: *mut Data<F, T>) {
    let data = unsafe { &mut *data };
    let f = unsafe { ManuallyDrop::take(&mut data.f) };
    data.ok = ManuallyDrop::new(f());
  }

  #[inline]
  #[rustc_nounwind]
  fn do_catch<F: FnOnce() -> T, T>(data: *mut Data<F, T>, payload: *mut u8) {
    let err = if unsafe { is_unwind_exception(payload) } {
      Exception::ForcedUnwind
    } else {
      cold_path();
      Exception::Panic(unsafe { __rust_panic_cleanup(payload) })
    };
    unsafe { (*data).err = ManuallyDrop::new(err) };
  }

  if unsafe { !core::intrinsics::catch_unwind(do_call::<F, T>, &raw mut data, do_catch::<F, T>) } {
    Ok(ManuallyDrop::into_inner(unsafe { data.ok }))
  } else {
    Err(ManuallyDrop::into_inner(unsafe { data.err }))
  }
}

unsafe extern "Rust" {
  #[rustc_std_internal_symbol]
  fn __rust_panic_cleanup(payload: *mut u8) -> Box<dyn Any + Send + 'static>;

  #[rustc_std_internal_symbol]
  safe fn __rust_start_panic(payload: &mut dyn PanicPayload) -> u32;
}

pub fn resume(payload: Box<dyn Any + Send>) -> ! {
  struct RewrapBox(Box<dyn Any + Send>);

  impl PanicPayload for RewrapBox {
    fn take_box(&mut self) -> Box<dyn Any + Send> {
      core::mem::replace(&mut self.0, Box::new(()))
    }

    fn get(&mut self) -> &(dyn Any + Send) {
      &*self.0
    }
  }

  impl Display for RewrapBox {
    fn fmt(&self, _f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      unreachable!()
    }
  }

  let _code = __rust_start_panic(&mut RewrapBox(payload));
  // todo: write `code`.
  abort("failed to initiate panic");
}

#[cold]
pub fn abort(_message: &str) -> ! {
  abort_immediate();
}
