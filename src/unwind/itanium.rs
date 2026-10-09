#![allow(non_camel_case_types)]

use crate::CALLER;
use crate::unwind::abort;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ptr;
use core::ptr::NonNull;

use crate::stack::StackPtr;
use uw::{_Unwind_Exception, _Unwind_Exception_Class, _Unwind_RaiseException, _Unwind_Reason_Code};

// the great [blog post] on faster exception handling in Rust
// by A. Sireneva; her `lithium` crate was also a source of
// inspiration.
// The reason for not using is that I personally wanted to
// learn about unwinding mechanisms; in the future I might
// replace them if some form of control transfers becomes
// too convoluted.

// todo: rename to `CORO_DROP`?
const CORO_UNWIND: _Unwind_Exception_Class = u64::from_ne_bytes(*b"CORORUST");

pub extern "sysv64-unwind" fn start_unwind(_sp: StackPtr, caller: NonNull<*const ()>) -> ! {
  CALLER.set(unsafe { caller.read() });

  extern "C" fn cleanup(_code: _Unwind_Reason_Code, _ex: *mut _Unwind_Exception) {}

  #[thread_local]
  static EX: UnsafeCell<MaybeUninit<_Unwind_Exception>> = UnsafeCell::new(MaybeUninit::uninit());
  unsafe {
    EX.get().write(MaybeUninit::new(_Unwind_Exception {
      exception_class: CORO_UNWIND,
      exception_cleanup: Some(cleanup),
      private: [ptr::null(); _],
    }))
  };

  let ex_ptr = EX.get().cast();
  let _code = unsafe { _Unwind_RaiseException(ex_ptr) };
  // todo: write `code`.
  abort("_Unwind_RaiseException failed")
}

pub unsafe fn is_unwind_exception(ptr: *mut u8) -> bool {
  let ex = ptr as *mut _Unwind_Exception;
  unsafe { (*ex).exception_class == CORO_UNWIND }
}
