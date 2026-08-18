#![allow(non_camel_case_types)]

use crate::ACTIVE;
use core::cell::UnsafeCell;
use core::hint::cold_path;
use core::mem::MaybeUninit;
use core::ptr;
// Taken from https://github.com/rust-lang/rust/blob/main/library/unwind/src/libunwind.rs.

// the great [blog post] on faster exception handling in Rust
// by A. Sireneva; her `lithium` crate was also a source of
// inspiration.
// The reason for not using is that I personally wanted to
// learn about unwinding mechanisms; in the future I might
// replace them if some form of control transfers becomes
// too convoluted.
type c_int = core::ffi::c_int;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum _Unwind_Reason_Code {
  _URC_NO_REASON = 0,
  _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
  _URC_FATAL_PHASE2_ERROR = 2,
  _URC_FATAL_PHASE1_ERROR = 3,
  _URC_NORMAL_STOP = 4,
  _URC_END_OF_STACK = 5,
  _URC_HANDLER_FOUND = 6,
  _URC_INSTALL_CONTEXT = 7,
  _URC_CONTINUE_UNWIND = 8,
  _URC_FAILURE = 9,
}

type _Unwind_Exception_Class = u64;
type _Unwind_Word = *const u8;
type _Unwind_Action = c_int;
type _Unwind_Exception_Cleanup_Fn =
  Option<extern "C" fn(_Unwind_Reason_Code, *mut _Unwind_Exception)>;

const UNWINDER_PRIVATE_DATA_SIZE: usize = 2;

// todo: rename to `CORO_DROP`?
const CORO_UNWIND: _Unwind_Exception_Class = u64::from_ne_bytes(*b"CORORUST");

#[repr(C)]
pub struct _Unwind_Exception {
  exception_class: _Unwind_Exception_Class,
  exception_cleanup: _Unwind_Exception_Cleanup_Fn,
  private: [_Unwind_Word; UNWINDER_PRIVATE_DATA_SIZE],
}

pub enum _Unwind_Context {}

unsafe extern "C-unwind" {
  fn _Unwind_RaiseException(exception: *mut _Unwind_Exception) -> _Unwind_Reason_Code;
}

pub extern "C-unwind" fn start_unwind() -> ! {
  extern "C" fn cleanup(_code: _Unwind_Reason_Code, _ex: *mut _Unwind_Exception) {}

  #[thread_local]
  static EXCEPTION: UnsafeCell<MaybeUninit<_Unwind_Exception>> =
    UnsafeCell::new(MaybeUninit::uninit());

  unsafe {
    EXCEPTION.get().write(MaybeUninit::new(_Unwind_Exception {
      exception_class: CORO_UNWIND,
      exception_cleanup: Some(cleanup),
      private: [ptr::null(); UNWINDER_PRIVATE_DATA_SIZE],
    }))
  };

  let ex_ptr = EXCEPTION.get().cast();
  // panic!("manual unwind");
  let res = unsafe { _Unwind_RaiseException(ex_ptr) };
  // todo: abort better than panic.
  panic!("_Unwind_RaiseException failed: {res:?}");
}

// See Section 1.6.2.
const _UA_SEARCH_PHASE: _Unwind_Action = 1;
const _UA_CLEANUP_PHASE: _Unwind_Action = 2;

pub unsafe extern "C" fn __coro_trampoline_personality(
  version: c_int,
  actions: _Unwind_Action,
  ex_class: _Unwind_Exception_Class,
  _ex: *mut _Unwind_Exception,
  _ctx: *mut _Unwind_Context,
) -> _Unwind_Reason_Code {
  if version != 1 {
    cold_path();
    return _Unwind_Reason_Code::_URC_FATAL_PHASE1_ERROR;
  }
  // extern crate std;
  // std::println!("tramp_personality: actions={actions:x}, ex_class={ex_class:x}");
  // Poison the active coroutine as soon as possible, before the second phase starts.
  if actions & _UA_SEARCH_PHASE != 0
  /* && ex_class != CORO_UNWIND*/
  {
    // todo: poison the active coroutine.
    // todo: Move to some `cleanup_on_panic` global function in `lib.rs`.
    let cur = ACTIVE.get();
    if cur.is_null() {
      cold_path();
      return _Unwind_Reason_Code::_URC_FATAL_PHASE2_ERROR;
    }
    let cur = unsafe { &mut *cur };
    // cur.poisoned = true;
    // std::println!("poisoned active {:?}", &raw const cur);
  }
  _Unwind_Reason_Code::_URC_CONTINUE_UNWIND
}
