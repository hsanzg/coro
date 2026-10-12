#![allow(non_camel_case_types)]

use crate::unwind::abort;
use core::cell::Cell;
use core::mem::MaybeUninit;
use core::ptr;
use uw::{_Unwind_Exception, _Unwind_Exception_Class, _Unwind_RaiseException, _Unwind_Reason_Code};

/// Identifies coroutine unwind exceptions, as opposed to Rust panics
/// and foreign exceptions.
const CORO_UW: _Unwind_Exception_Class = u64::from_ne_bytes(*b"CORORUST");

/// Unwinds the stack of the active coroutine.
pub extern "sysv64-unwind" fn start_unwind() -> ! {
  // What follows is essentially a version of `uw::__rust_start_panic`
  // that avoids heap allocation; see https://github.com/rust-lang/rust/blob/main/library/panic_unwind/src/gcc.rs.

  #[cold]
  extern "C" fn cleanup(_code: _Unwind_Reason_Code, _ex: *mut _Unwind_Exception) {
    abort("coro exception was caught and discarded by foreign code");
  }

  #[thread_local]
  static EX: Cell<MaybeUninit<_Unwind_Exception>> = Cell::new(MaybeUninit::uninit());

  EX.set(MaybeUninit::new(_Unwind_Exception {
    exception_class: CORO_UW,
    exception_cleanup: Some(cleanup),
    private: [ptr::null(); _],
  }));

  let ex_ptr = EX.as_ptr().cast();
  let code = unsafe { _Unwind_RaiseException(ex_ptr) };
  abort(match code {
    _Unwind_Reason_Code::_URC_END_OF_STACK => "no handler caught the coroutine unwind",
    _Unwind_Reason_Code::_URC_FATAL_PHASE1_ERROR => "coroutine unwind failed during search phase",
    _Unwind_Reason_Code::_URC_FATAL_PHASE2_ERROR => "coroutine unwind failed during cleanup phase",
    _ => "coroutine unwind failed for an unknown reason",
  })
}

/// Does `ptr` point to an exception raised by [`start_unwind`]?
///
/// # Safety
///
/// `ptr` must point to a valid [`_Unwind_Exception`].
pub unsafe fn is_unwind_exception(ptr: *mut u8) -> bool {
  let ex = ptr as *mut _Unwind_Exception;
  unsafe { (*ex).exception_class == CORO_UW }
}
