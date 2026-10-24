#![allow(non_camel_case_types)]

use crate::unwind::abort;
use core::cell::Cell;
use core::mem::MaybeUninit;
use core::ptr;
use uw::{
  _Unwind_DeleteException, _Unwind_Exception, _Unwind_Exception_Class, _Unwind_RaiseException,
  _Unwind_Reason_Code,
};

pub type Exception = *mut _Unwind_Exception;

/// Identifies coroutine unwind exceptions, as opposed to Rust panics
/// and foreign exceptions.
const CORO_UW: _Unwind_Exception_Class = u64::from_ne_bytes(*b"CORORUST");

/// Unwinds the stack of the active coroutine.
pub extern "sysv64-unwind" fn start_unwind() -> ! {
  // What follows is essentially a version of `uw::__rust_start_panic`
  // that avoids heap allocation; see https://github.com/rust-lang/rust/blob/main/library/panic_unwind/src/gcc.rs.

  #[cold]
  extern "C" fn cleanup(_code: _Unwind_Reason_Code, _ex: *mut _Unwind_Exception) {
    abort("coroutine unwind exception was caught and discarded");
  }

  #[thread_local]
  static EX: Cell<MaybeUninit<_Unwind_Exception>> = Cell::new(MaybeUninit::uninit());

  EX.set(MaybeUninit::new(_Unwind_Exception {
    exception_class: CORO_UW,
    exception_cleanup: Some(cleanup),
    private: [ptr::null(); _],
  }));
  unsafe { rethrow(EX.as_ptr().cast()) }
}

/// Returns whether `ex` was raised by [`start_unwind`].
///
/// # Safety
///
/// `ex` must be an exception that has not been destroyed.
pub unsafe fn is_coro_unwind(ex: Exception) -> bool {
  unsafe { (*ex).exception_class == CORO_UW }
}

/// Raises `ex` again from the current stack frame.
///
/// # Safety
///
/// `ex` must be an exception that has not been destroyed.
pub unsafe fn rethrow(ex: Exception) -> ! {
  let code = unsafe { _Unwind_RaiseException(ex) };
  abort(match code {
    _Unwind_Reason_Code::_URC_END_OF_STACK => "no handler caught the exception",
    _Unwind_Reason_Code::_URC_FATAL_PHASE1_ERROR => "unwind failed during search phase",
    _Unwind_Reason_Code::_URC_FATAL_PHASE2_ERROR => "unwind failed during cleanup phase",
    _ => "unwind failed for an unknown reason",
  })
}

/// Destroys the given exception without propagating it.
///
/// # Safety
///
/// `ex` must be an exception that has not been destroyed.
pub unsafe fn destroy(ex: Exception) {
  unsafe { _Unwind_DeleteException(ex) };
}
