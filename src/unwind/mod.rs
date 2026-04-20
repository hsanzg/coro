#[cfg_attr(not(windows), path = "itanium.rs")]
mod imp;

pub use self::imp::*;
use std::mem::ManuallyDrop;

pub fn catch_unwind<R, F: FnOnce() -> R>(f: F) -> Result<R, ()> {
  union Data<F, R> {
    f: ManuallyDrop<F>,
    r: ManuallyDrop<R>,
  }
  let mut data = Data {
    f: ManuallyDrop::new(f),
  };

  fn do_call<F: FnOnce() -> R, R>(data: *mut u8) {
    unsafe {
      let data = data as *mut Data<F, R>;
      let data = &mut *data;
      let f = ManuallyDrop::take(&mut data.f);
      data.r = ManuallyDrop::new(f());
    }
  }

  fn do_catch<F: FnOnce() -> R, R>(_data: *mut u8, _payload: *mut u8) {}

  let data_ptr = (&raw mut data) as *mut u8;
  if unsafe { core::intrinsics::catch_unwind(do_call::<F, R>, data_ptr, do_catch::<F, R>) == 0 } {
    Ok(ManuallyDrop::into_inner(unsafe { data.r }))
  } else {
    Err(())
  }
}
