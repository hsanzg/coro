use crate::stack::StackPtr;
use crate::{Bottom, Control};
use core::arch::{asm, global_asm, naked_asm};
use core::mem::offset_of;
use core::panic::AssertUnwindSafe;
use core::ptr;
use core::ptr::NonNull;

// Invariant: return address always stored at `control.sp`.

#[inline(always)]
pub unsafe fn resume_coro(target: &mut Control) {
  unsafe {
    asm!(
      // Preserve the current contents of the reserved register `rbx`.
      // Register `rbp` is also reserved, but we push its contents to
      // the stack only after the `call` instruction below transfers
      // control to `trampoline` or `suspend_coro`. In this way the
      // top of the stack will contain a suitably arranged frame, as
      // specified in Section 3.2.2 of the System V AMD64 ABI.
      // todo: Well, that's not entirely true, since we do not align
      //  the stack to 16 bytes prior to the `call`. But that's no
      //  issue, since `trampoline` and `suspend_coro` can handle
      //  a misaligned frame, and debuggers that use the chain of
      //  frame pointers do not really care about the misalignment
      //  either.
      // todo: to be correct, maybe we should do a `sub rsp, 8`.
      "push rbx",
      ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size 0
      "call [rsi]",
      // The coroutine has returned control to us; all that remains
      // is to restore the previous contents of `rbx`.
      "pop rbx",
      in("rdi") target,
      in("rsi") target.stack_ptr.as_ptr(),
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      options(may_unwind),
    )
  }
}

pub fn initial_sp<R, F>(bottom: NonNull<Bottom<R, F>>) -> StackPtr
where
  F: FnOnce() -> R + Unpin,
{
  unsafe { bottom.byte_sub(8).cast() }
}

pub fn first_inst_addr<R, F>() -> *const ()
where
  F: FnOnce() -> R + Unpin,
{
  // See `first_tramp` for the reason behind the 1-byte offset.
  unsafe { (first_tramp::<R, F> as *const ()).byte_add(1) }
}

pub fn prepare_stack<R, F>(bottom: NonNull<Bottom<R, F>>)
where
  F: FnOnce() -> R + Unpin,
{
  let pad = unsafe { bottom.byte_add(8).cast() };
  unsafe { pad.write(ptr::null::<()>()) };
  // let ret_addr = unsafe { bottom.byte_add(8).cast() };
  // unsafe { ret_addr.write(resume_coro as *const ()) };
  let tramp_ret_addr = unsafe { bottom.byte_sub(8).cast() };
  unsafe { tramp_ret_addr.write(first_inst_addr::<R, F>()) };
}

// Define an LSDA for the personality function of the trampoline function.
// See https://www.airs.com/blog/archives/464 for an explanation of the format.
// Also see https://maskray.me/blog/2020-12-12-c++-exception-handling-abi.
/*global_asm!(
  r#".section .gcc_except_table, "a", @progbits"#,
  ".p2align 2, 0x0", // todo: why?
  ".globl __coro_tramp_lsda",
  ".type __coro_tramp_lsda, @object",
  "__coro_tramp_lsda:",
  // The personality function does not set up a landing pad,
  // so there's no need to set a base offset for it.
  ".byte 0xff", // LPStart = DW_EH_PE_omit
  // Leave the type table empty.
  ".byte 0xff", // TType = DW_EH_PE_omit
  // Define the encoding and length of the call-site table.
  ".byte 0x01", // DW_EH_PE_uleb128
  ".uleb128 .Lcs_table_end - .Lcs_table_start",
  // Declare the call-site table entry.
  ".Lcs_table_start:",
  ".uleb128 0",          // start instruction.
  ".uleb128 0xffffffff", // length of the instruction sequence.
  ".uleb128 0",          // there's no landing pad.
  // The action to take is to clean up and continue unwinding.
  ".byte 0", // 1 + action table index. // todo: zero is a special case meaning "cleanup".
  ".Lcs_table_end:",
);*/

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn first_tramp<R, F: FnOnce() -> R + Unpin>(
  _active: &mut Control,
  _coro_sp: StackPtr,
) -> ! {
  naked_asm!(
    ".cfi_startproc", // simple
    // ".cfi_personality 0x1b, {personality_fn}", // 0x1b = DW_EH_PE_pcrel | DW_EH_PE_sdata4
    // ".cfi_lsda 0x1b, __coro_tramp_lsda",
    // todo: This seems to cause the unwinder to use the unwind info
    //  corresponding the instruction _following_ the return address;
    //  check that all relevant `asm!` blocks do not end with a `ret`
    //  or a `jmp` not followed by e.g. a dummy `nop` or `ud2`.
    // ".cfi_signal_frame", // hack to allow backtrace generation with gdb.

    // Place this frame on the parent stack.
    // DW_CFA_def_cfa_expression DW_FORM_exprloc(len=5, DW_OP_breg6/rbp 0 DW_OP_deref DW_OP_plus_uconst 24)
    // In theory we could use `DW_OP_reg6` instead of `DW_OP_breg6 +0`.
    // Alas, gdb doesn't accept the former operation when it appears
    // within `DW_CFA_def_cfa_expression`.
    ".cfi_escape 0x0f, 5, 0x76, 0, 0x06, 0x23, 24",
    /*// LLVM doesn't support `DW_CFA_val_offset`. // todo: is this still up to date? Record LLVM version.
    // DW_CFA_val_expression rsp, DW_FORM_block(len=2, DW_OP_plus_uconst 8)
    // ".cfi_escape 0x16, 0x06, 2, 0x23, 8", // on entry to the `asm!` block of `resume_coro`.*/
    ".cfi_offset rbx, -8",
    ".cfi_offset rip, -16",
    ".cfi_offset rbp, -24",

    // LLDB subtracts 1 from the return address of a frame before
    // resolving the function symbol that it points to. The return
    // address of the main `trampoline` function is special in that
    // it points to the beginning of `first_tramp` instead of the
    // instruction following the `jmp` below. (This is because we
    // also use that field to store the initial resumption address
    // of the coroutine.) We include this `nop` to correct for that
    // 1-byte adjustment, and point the initial address one byte
    // after it in `prepare_stack`.
    "nop",

    // Preserve the current contents of `rbp`.
    "push rbp",
    // Record the parent stack pointer value.
    "mov [rdi + {sp_off}], rsp",
    // Switch to the new stack.
    "lea rbp, [rdi + {sp_off}]",
    "mov rsp, rsi",
    // Bounce into the architecture-independent trampoline.
    ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size, 0
    "jmp {second_tramp}", // todo: replace by call and skip the manual stack adjustment in `initial_sp`?
    "ud2",
    ".cfi_endproc",
    sp_off = const offset_of!(Control, stack_ptr),
    // personality_fn = sym crate::unwind::__coro_trampoline_personality,
    second_tramp = sym crate::trampoline::<R, F>,
  )
}

#[inline(never)]
pub unsafe fn suspend_coro(active: &mut Control) {
  unsafe {
    asm!(
      // Preserve the current contents of the reserved registers
      // on the coroutine stack.
      "push rbx",
      "push rbp",
      // Also record the resumption program address.
      "lea rax, [rip + 2f]",
      "push rax",
      // Switch to the parent stack.
      "mov [rdi + {sp_off}], rsp",
      "mov rsp, rsi",
      // Restore the previous contents of `rbp`.
      "pop rbp",
      // Return control to `resume_coro`.
      ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size 0
      "ret",
      "2:",
      // Preserve the current contents of the `rbp` register. After
      // this step, the top of the parent stack has a valid frame.
      "push rbp",
      // Record the stack pointer value of the parent program.
      "mov [rdi + {sp_off}], rsp",
      // Switch to the coroutine stack. (The contents of `rsi` are
      // not the same as they were on entry to this assembly macro,
      // since they were updated by `resume_coro`.)
      "lea rsp, [rsi + 8]", // skip resumption address.
      // Restore the previous contents of the reserved registers.
      "pop rbp",
      "pop rbx",
      in("rdi") active,
      in("rsi") active.stack_ptr.as_ptr(),
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      sp_off = const offset_of!(Control, stack_ptr),
      options(may_unwind),
    )
  };
}

#[inline(always)]
pub unsafe fn return_control(active: &mut Control) -> ! {
  unsafe {
    asm!(
      // Mark the coroutine as finished (but non-panicking).
      // todo: improve strategy; it should suffice to have a nonnull `ret_val` pointer.
      "mov [rdi + {sp_off}], rsp",
      // Switch to the parent stack.
      "mov rsp, rsi",
      // Restore the previous contents of `rbp`.
      "pop rbp",
      // Return control to `resume_coro`.
      "ret",
      /*"ud2",*/
      in("rdi") active,
      in("rsi") active.stack_ptr.as_ptr(),
      sp_off = const offset_of!(Control, stack_ptr),
      options(/*may_unwind,*/ noreturn)
    )
  }
}

#[inline]
pub unsafe fn force_unwind2(target: &mut Control) {
  let mut target_ref = AssertUnwindSafe(target);
  let res = crate::unwind::catch_unwind(move || unsafe {
    let target: &mut _ = *target_ref;
    asm!(
      // Create a frame on the parent stack.
      "push rbx",
      "lea rax, [rip + 2f]",
      "push rax",
      "push rbp",
      // Record the current stack pointer value.
      "mov [rdi + {sp_off}], rsp",
      // Switch to the coroutine stack.
      "mov rsp, rsi",
      // Restore the previous contents of the reserved registers.
      "pop rax",
      "pop rbp",
      "pop rbx",
      // Begin the unwinding.
      // todo: Say that this is to set the return address of `start_unwind` to the original resumption point of the coroutine.
      ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size 0
      "push rax",
      "jmp {start_unwind}",
      "2:",
      // It may be the case that the coroutine program catches the exception
      // itself, by using `catch_unwind`. This function must work properly
      // in that case as well. This section of call is jumped-to from
      // `suspend_coro`, so it suffices to restore the contents of `rbx`,
      // just as in `resume_coro`.
      "pop rbx",
      in("rdi") target,
      in("rsi") target.stack_ptr.as_ptr(),
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      sp_off = const offset_of!(Control, stack_ptr),
      start_unwind = sym crate::unwind::start_unwind,
      options(may_unwind)
    );
  });
  // todo: it may be the case that the coroutine caught the unwind. Do not
  //  complain in that case, but check what happens if dropping coroutine
  //  suspends.
  // todo: if the coroutine suspends when it is being dropped,
  //       this code seems to get called. Replace `debug_assert`
  //       by an assertion, and make panic message clearer with
  //       something like "Did the coroutine suspend while being unwound?"
  assert!(res.is_err(), "coroutine did not unwind");
}

/*pub unsafe extern "sysv64-unwind" fn transfer_control(target: &mut Control) {
  unsafe {
    asm!(
      "mov [rsp - 8], rbx",
      // ".cfi_adjust_cfa_offset 8",
      // ".cfi_rel_offset rbx, -8",
      "mov [rsp - 16], rbp",
      // ".cfi_rel_offset rbp, -16",
      // ".cfi_adjust_cfa_offset 8",
      // ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size, 0

      "mov rax, [rdi + {sp_off}]",
      "mov [rdi + {sp_off}], rsp",
      "mov rsp, rax",

      "mov rax, [rdi + {ip_off}]",
      "lea rsi, [rip + 2f]",
      "mov [rdi + {ip_off}], rsi",
      "jmp rax",

      "mov rbp, [rsp - 16]",
      "mov rbx, [rsp - 8]",
      "call {unwind}",

      "2:",
      "mov rbp, [rsp - 16]",
      // ".cfi_restore rbp",
      // ".cfi_adjust_cfa_offset -8",
      "mov rbx, [rsp - 8]",
      // ".cfi_adjust_cfa_offset -8",
      // ".cfi_restore rbx",
      in("rdi") target, lateout("rsi") _,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      sp_off = const offset_of!(Control, stack_ptr),
      ip_off = const offset_of!(Control, inst_ptr),
      unwind = sym crate::unwind::start_unwind,
      options(may_unwind)
    )
  };
  // extern crate std;
  // std::println!("returning from transfer_control");
}

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn transfer_control2(_target: &mut Control) {
  #[inline(always)]
  unsafe extern "sysv64-unwind" fn inner(target: &mut Control) {
    unsafe {
      asm!(
        "mov rax, [rdi + {sp_off}]",
        "mov [rdi + {sp_off}], rsp",
        "mov rsp, rax",

        "mov rax, [rdi + {ip_off}]",
        "lea rsi, [rip + 2f]",
        "mov [rdi + {ip_off}], rsi",

        "jmp rax",

        "call {unwind}",
        "2:",
        in("rdi") target,
        clobber_abi("sysv64"),
        lateout("r12") _, lateout("r13") _,
        lateout("r14") _, lateout("r15") _,
        sp_off = const offset_of!(Control, stack_ptr),
        ip_off = const offset_of!(Control, inst_ptr),
        unwind = sym crate::unwind::start_unwind,
        options(may_unwind)
      )
    }
  }
  naked_asm!(
    ".cfi_startproc",
    "mov [rsp - 16], rbp",
    ".cfi_rel_offset rbp, -16",
    "mov [rsp - 8], rbx",
    ".cfi_rel_offset rbx, -8",
    "sub rsp, 24", // mod 16 = 8
    ".cfi_adjust_cfa_offset 24",
    "call {inner}",
    "mov rbp, [rsp + 8]",
    ".cfi_restore rbp",
    "mov rbx, [rsp + 16]",
    ".cfi_restore rbx",
    "add rsp, 24",
    ".cfi_adjust_cfa_offset -24",
    "ret",
    ".cfi_endproc",
    inner = sym inner,
  )
}

/// Returns the value that the stack pointer should have after
/// [transferring control] to the coroutine associated with the
/// metadata object pointed to by `bottom` (which need not be
/// deferrenceable).
///
/// [transferring control]: transfer_control
pub fn initial_sp2<R, F>(bottom: NonNull<Bottom<R, F>>) -> StackPtr
where
  F: FnOnce() -> R + Unpin,
{
  // todo: finish comment. Explain why stack pointer is pointed
  //  to where it is, and prove that location is aligned to a
  //  16-byte boundary.
  // SAFETY: This is similar to `&raw const bottom.ctrl.inst_ptr`,
  //  but it does not require
  // unsafe { bottom.byte_add(offset_of!(Control, inst_ptr)) }.cast()
  let ret_addr = unsafe { bottom.byte_sub(8) }.cast::<*const u8>();
  unsafe { ret_addr.write((first_frame as *const u8).byte_add(1)) }
  ret_addr.cast()
}

// todo: this is now wrong; rename to `first_coro_frame`.
/// The initial sequence of instructions executed by a coroutine
/// after it is activated for the first time.
///
/// See [`crate::trampoline`] for some ABI-related considerations.
#[unsafe(naked)]
pub extern "sysv64-unwind" fn first_frame() -> ! {
  // todo: finish comment. Reference `initial_sp` and make clear
  //  that the 16control record is actually part of the first frame
  //  in the coroutine's program stack. That way we get linkage
  //  between the different stacks of coroutines, allowing unwinding
  //  by regular means.
  // The only direct entry point to this function is through the `jmp`
  // instruction in `transfer_control`. So at this point `rdi` points
  // to the control record of a coroutine.

  // Conveniently, since that control record appears as the first field
  // of the `Bottom` structure of the coroutine at the bottom of its stack,
  // and `Bottom` is annotated with `#[repr(C)]`, `rdi` is also the
  // address of the `Bottom` structure. This sets up things perfectly
  // to call the architecture-independent trampoline function. But
  // there's still some work left.
  naked_asm!(
    ".cfi_startproc",
    ".cfi_personality 0x1b, {personality_fn}", // DW_EH_PE_pcrel | DW_EH_PE_sdata4
    ".cfi_signal_frame", // hack to allow backtrace generation with gdb.
    ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size, 0
    ".cfi_def_cfa rsp, 16",
    ".cfi_rel_offset rsp, 0",
    // DW_CFA_expression DW_OP_reg6 (rbp) DW_FORM_block(len, DW_OP_lit16 DW_OP_minus DW_OP_deref DW_OP_plus_uconst 8 DW_OP_deref)
    // ".cfi_escape 0x10, 0x66, 6, 0x40, 0x1c, 0x06, 0x23, 0x08, 0x06",
    // DW_CFA_expression DW_OP_reg3 (rbx) DW_FORM_block(len, DW_OP_lit16 DW_OP_minus DW_OP_deref DW_OP_deref)
    // ".cfi_escape 0x10, 0x63, 4, 0x40, 0x1c, 0x06, 0x06",
    // Bounce off to the second, architecture-independent trampoline.
    // "call {second_tramp}",
    "nop",
    "ud2",
    ".cfi_endproc",
    // second_tramp = sym crate::trampoline::<R, F>,
    personality_fn = sym crate::unwind::__coro_trampoline_personality
  )
}

/// Jumps to the instruction immediately after the [`resume`] call that
/// last activated the current [coroutine]. This function is a version
/// of [`transfer_control`] for the case when the coroutine will never
/// get activated again (hence the `!` return type): it does not bother
/// to save the caller-saved register contents, it writes the address
/// `_ret_val_addr` of the coroutine's return value to the `stack_ptr`
/// field of the control record, and it clears the instruction pointer
/// in the control record. (See [`Control`] for more information.)
///
/// # Safety
///
/// This function must be called from within a coroutine that shall
/// never be resumed again. Pointer `_ret_val_addr` must be valid
/// for reads.
///
/// [`resume`]: crate::Coro::resume
/// [coroutine]: crate::Coro
#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn return_control2<R>(
  _control: &mut Control,    // in `rdi`.
  _ret_val_addr: NonNull<R>, // in `rsi`.
) -> ! {
  naked_asm!(
    // Switch back to the previous stack.
    "mov rsp, [rdi + {sp_off}]",
    // Record the coroutine's return value location.
    "mov [rdi + {sp_off}], rsi",
    // Mark the coroutine as finished and return to the previous
    // computation.
    "mov rsi, [rdi + {ip_off}]",
    "mov qword ptr [rdi + {ip_off}], 0",
    "jmp rsi",
    sp_off = const offset_of!(Control, stack_ptr),
    ip_off = const offset_of!(Control, inst_ptr),
  )
}

///
/// # Safety
///
/// The coroutine must be currently [paused]; its computation must
/// have started but not be completely finished.
///
/// [paused]: crate::Coro::is_paused
// todo: note that caller is responsible for marking the coroutine as finished, if necessary.
pub unsafe fn force_unwind(control: &mut Control) {
  // Transfer control back to the `call {unwind}` instruction in
  // `transfer_control`, by moving the resumption address of the
  // coroutine (local label `2` in that function) five bytes back
  // (the length of the `call {unwind}` instruction).
  control.inst_ptr = unsafe {
    control.inst_ptr.byte_sub(5 /*15*/)
  };
  let resume_addr = control.stack_ptr.cast::<*const ()>();
  unsafe { resume_addr.write(resume_addr.read().byte_sub(5)) };

  // crate::ACTIVE.set(&raw mut *control);
  let mut control_ref = AssertUnwindSafe(control);
  let res = crate::unwind::catch_unwind(move || unsafe { resume_coro(*control_ref) });
  debug_assert!(res.is_err(), "coroutine did not unwind");
}*/

// Rust translation of the client request macros in `valgrind.h`. See
// https://sourceware.org/git/?p=valgrind.git;a=blob;f=include/valgrind.h.in;h=ed7d8c68e24b896e912fb19e17cc906730a82a4c;hb=refs/heads/master#l409.

#[cfg(feature = "valgrind")]
fn valgrind_do_client_request(default: usize, args: &[usize; 6]) -> usize {
  let result: usize;
  unsafe {
    asm!(
      "rol rdi, 3; rol rdi, 13",
      "rol rdi, 61; rol rdi, 51",
      "xchg rbx, rbx",
      inlateout("rdx") default => result,
      in("rax") args,
      // The sum of the rotation offsets is 128, so
      // the original contents of `rdi` are left intact.
    )
  };
  result
}

#[cfg(feature = "valgrind")]
use crate::stack::ValgrindStackId;

#[cfg(feature = "valgrind")]
pub fn valgrind_register_stack(start: *mut u8, end: *mut u8) -> ValgrindStackId {
  let res = valgrind_do_client_request(0, &[0x1501, start as _, end as _, 0, 0, 0]);
  ValgrindStackId(res as _)
}

#[cfg(feature = "valgrind")]
pub fn valgrind_deregister_stack(id: ValgrindStackId) {
  valgrind_do_client_request(0, &[0x1502, id.0 as _, 0, 0, 0, 0]);
}
