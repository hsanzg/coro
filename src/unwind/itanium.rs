#![allow(non_camel_case_types)]

use crate::{ACTIVE, arch, finished};
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

const UNWINDER_PRIVATE_DATA_SIZE: usize = 2; // minimum across all supported targets.

// todo: rename to `CORO_DROP`?
const CORO_UNWIND: _Unwind_Exception_Class = u64::from_ne_bytes(*b"CORORUST");

#[repr(C)]
#[cfg_attr(target_pointer_width = "16", repr(align(4)))]
#[cfg_attr(target_pointer_width = "32", repr(align(8)))]
#[cfg_attr(target_pointer_width = "64", repr(align(16)))]
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
  /*extern crate std;
  std::panic::resume_unwind(std::boxed::Box::new(()));*/
  // todo: add `#[rustc_nounwind]`.
  extern "C" fn cleanup(_code: _Unwind_Reason_Code, _ex: *mut _Unwind_Exception) {}

  #[thread_local]
  static EXCEPTION: UnsafeCell<MaybeUninit<_Unwind_Exception>> =
    UnsafeCell::new(MaybeUninit::uninit());

  unsafe {
    EXCEPTION.get().write(MaybeUninit::new(_Unwind_Exception {
      exception_class: CORO_UNWIND,
      exception_cleanup: Some(cleanup),
      private: [ptr::null(); _],
    }))
  };

  let ex_ptr = EXCEPTION.get().cast();
  let res = unsafe { _Unwind_RaiseException(ex_ptr) };
  extern "C" fn handle_failure(code: _Unwind_Reason_Code) -> ! {
    panic!("_Unwind_RaiseException failed: {code:?}"); // aborts due to `extern "C"`.
  }
  handle_failure(res);
}

pub fn backtrace() {
  extern crate std;
  #[repr(C)]
  #[derive(Default)]
  struct dwarf_eh_bases {
    tbase: _Unwind_Word,
    dbase: _Unwind_Word,
    func: _Unwind_Word,
  }
  unsafe extern "C" {
    fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> usize;
    fn _Unwind_GetIPInfo(ctx: *mut _Unwind_Context, ip_before: *mut core::ffi::c_int);
    fn _Unwind_GetCFA(ctx: *mut _Unwind_Context) -> _Unwind_Word;
    fn _Unwind_Find_FDE(pc: *const core::ffi::c_void, bases: *mut dwarf_eh_bases) -> _Unwind_Word;
    fn _Unwind_GetLanguageSpecificData(ctx: *mut _Unwind_Context) -> _Unwind_Word;
    fn _Unwind_GetRegionStart(ctx: *mut _Unwind_Context) -> _Unwind_Word;
    fn _Unwind_GetTextRelBase(ctx: *mut _Unwind_Context) -> _Unwind_Word;
    fn _Unwind_GetDataRelBase(ctx: *mut _Unwind_Context) -> _Unwind_Word;
    fn _Unwind_Backtrace(
      trace: unsafe extern "C" fn(
        ctx: *mut _Unwind_Context,
        arg: *mut core::ffi::c_void,
      ) -> _Unwind_Reason_Code,
      arg: *mut core::ffi::c_void,
    ) -> _Unwind_Reason_Code;
  }
  unsafe extern "C" fn callback(
    ctx: *mut _Unwind_Context,
    _arg: *mut core::ffi::c_void,
  ) -> _Unwind_Reason_Code {
    let ip = unsafe { _Unwind_GetIP(ctx) };
    let mut ip_before = 0;
    unsafe { _Unwind_GetIPInfo(ctx, &raw mut ip_before) };
    let cfa = unsafe { _Unwind_GetCFA(ctx) };
    let mut eh_bases = dwarf_eh_bases::default();
    let fde = unsafe { _Unwind_Find_FDE(ptr::without_provenance(ip), &raw mut eh_bases) };
    let lsda = unsafe { _Unwind_GetLanguageSpecificData(ctx) };
    let func_start = unsafe { _Unwind_GetRegionStart(ctx) };
    // let text_start = unsafe { _Unwind_GetTextRelBase(ctx) };
    // let data_start = unsafe { _Unwind_GetDataRelBase(ctx) };
    std::println!(
      "ctx={ctx:p}, ip={ip:#x} (before={ip_before:#x}), cfa={cfa:#x?}, fde={fde:x?} (func={:x?}), lsda={lsda:x?}, func={func_start:?}", //  text={text_start:?}, data={data_start:?}
      eh_bases.func
    );
    _Unwind_Reason_Code::_URC_NO_REASON
  }
  unsafe {
    let res = _Unwind_Backtrace(callback, ptr::null_mut());
    core::assert_matches!(
      res,
      _Unwind_Reason_Code::_URC_END_OF_STACK | _Unwind_Reason_Code::_URC_NO_REASON
    );
  }
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
  extern crate std;
  std::println!("tramp_personality: actions={actions:x}, ex_class={ex_class:x}");
  std::println!("  active at this point is {:?}", ACTIVE.get());
  // Poison the active coroutine as soon as possible, before the second phase starts.
  if actions & _UA_SEARCH_PHASE != 0 && ex_class != CORO_UNWIND {
    // todo: poison the active coroutine.
    // todo: Move to some `cleanup_on_panic` global function in `lib.rs`.
    let cur = ACTIVE.get();
    if cur.is_null() {
      cold_path();
      return _Unwind_Reason_Code::_URC_FATAL_PHASE2_ERROR;
    }

    // todo: experimental; untested.
    let cur = unsafe { &mut *cur };
    unsafe { arch::poison(cur, crate::panicked) };

    // cur.poisoned = true;
    std::println!(
      "poisoned active {:?} (={:?}?)",
      ACTIVE.get(),
      ptr::from_mut(cur)
    );
  }
  _Unwind_Reason_Code::_URC_CONTINUE_UNWIND
}
