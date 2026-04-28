use crate::stack::StackPtr;
use crate::{Bottom, Control};
use core::arch::{asm, naked_asm};
use core::mem::offset_of;
use core::panic::AssertUnwindSafe;
use core::ptr::NonNull;

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn transfer_control(_target: &mut Control) {
  extern "sysv64-unwind" fn unwind() -> ! {
    crate::unwind::start_unwind()
  }
  naked_asm!(
    ".cfi_startproc",
    "sub rsp, 56", // mod 16 = 8.
    ".cfi_adjust_cfa_offset 56",
    "mov [rsp], rbp",
    ".cfi_rel_offset rbp, 0",
    "mov [rsp + 8], rbx",
    ".cfi_rel_offset rbx, 8",
    "mov [rsp + 16], r12",
    ".cfi_rel_offset r12, 16",
    "mov [rsp + 24], r13",
    ".cfi_rel_offset r13, 24",
    "mov [rsp + 32], r14",
    ".cfi_rel_offset r14, 32",
    "mov [rsp + 40], r15",
    ".cfi_rel_offset r15, 40",

    "mov rax, [rdi + {sp_off}]",
    "mov [rdi + {sp_off}], rsp",
    "mov rsp, rax",

    "mov rax, [rdi + {ip_off}]",
    "lea rsi, [rip + 2f]",
    "mov [rdi + {ip_off}], rsi",

    "jmp rax",

    "call {unwind}",

    "2:",
    "mov rbp, [rsp]",
    ".cfi_restore rbp",
    "mov rbx, [rsp + 8]",
    ".cfi_restore rbx",
    "mov r12, [rsp + 16]",
    ".cfi_restore r12",
    "mov r13, [rsp + 24]",
    ".cfi_restore r13",
    "mov r14, [rsp + 32]",
    ".cfi_restore r14",
    "mov r15, [rsp + 40]",
    ".cfi_restore r15",
    "add rsp, 56",
    ".cfi_adjust_cfa_offset -56",
    "ret",
    ".cfi_endproc",
    sp_off = const offset_of!(Control, stack_ptr),
    ip_off = const offset_of!(Control, inst_ptr),
    unwind = sym unwind,
  )
}

/// Resumes execution of a computation at the point where it was last
/// suspended. This function saves the context of the current program
/// and restores the old context from the given control record.
pub unsafe extern "sysv64-unwind" fn transfer_control2(target: &mut Control) {
  extern "sysv64-unwind" fn unwind() -> ! {
    // todo: remove the indirection and call directly from assembly?
    //  That will require adding `extern "C-unwind"` to `start_unwind`.
    crate::unwind::start_unwind();
  }
  // The transfer of control consists of three steps: saving the previous
  // register contents, replacing the program stack, and jumping to the
  // location following the place where the action was last suspended.
  // There is no need to preserve the contents of _all_ registers; rather,
  // it suffices to save and restore the contents of every register that
  // the resumed computation _could_ possibly destroy. All transfers of
  // control are carried out by this function, which uses the System V
  // ABI for the x86-64 architecture. (The "-unwind" variant allows the
  // following `asm!` block to unwind the stack if necessary.) The new
  // computation may destroy the contents of any _caller-saved_ register
  // under this ABI, but it must restore any other state (including the
  // stack and frame pointers) just before it reaches a suspension point
  // or terminates. The `clobber_abi` option below tells the compiler to
  // insert all necessary clobber constraints to save and eventually
  // restore the contents of every caller-saved register that may change
  // during the new computation. (This assumes that the new computation
  // jumps back to the current program; `transfer_control` will never
  // return otherwise.)
  unsafe {
    asm!(
      // Save the contents of every reserved, callee-saved register
      // in the current stack. The ABI clobber list does not include
      // the frame pointer `bp` nor `bx` (which LLVM uses internally
      // as a "base pointer" for functions with complex stack frames).
      // todo: note that Rust guarantees that the stack is aligned to
      //  a 16-byte boundary on entry to an `asm!` block.
      // todo: can we use the red zone and avoid modifying the sp?
      //  this will require adding the sub below, right before calling
      //  the `force_unwind` subroutine, since otherwise that would
      //  overwrite the saved contents of rbx and rbp.
      "sub rsp, 16",
      ".cfi_adjust_cfa_offset 16",
      "mov [rsp + 8], rbx",
      ".cfi_rel_offset rbx, 8",
      "mov [rsp], rbp",
      ".cfi_rel_offset rbp, 0",
      // Swap the current stack pointer (`sp`) with the pointer address
      // in the control record (`target.stack_ptr`). The `initial_sp`
      // function ensures that the stack pointer is aligned during the
      // first activation of a coroutine. (See Section 3.2.3 of the
      // System V ABI for details.) We store the current stack pointer
      // in the control record so that a future call to `suspend` can
      // go back to the current stack.
      "mov rax, [rdi + {sp_off}]",
      "mov [rdi + {sp_off}], rsp",
      "mov rsp, rax",
      // Fetch the resumption point where we are to jump.
      "mov rax, [rdi + {ip_off}]",
      // The new computation may transfer control back to the current
      // program by jumping to local label `2`; let us compute its
      // effective address and store it in the control structure.
      "lea rsi, [rip + 2f]",
      "mov [rdi + {ip_off}], rsi",
      // Resume the computation.
      "jmp rax",
      // A landing pad for unwinding; see `force_unwind` below.
      "call {unwind}",
      "2:",
      // At this point we have come back to the original stack. It
      // remains to restore the contents of all manually-preserved
      // registers.
      "mov rbp, [rsp]",
      ".cfi_restore rbp",
      "mov rbx, [rsp + 8]",
      ".cfi_restore rbx",
      "add rsp, 16",
      ".cfi_adjust_cfa_offset -16",
      in("rdi") target,
      sp_off = const offset_of!(Control, stack_ptr),
      ip_off = const offset_of!(Control, inst_ptr),
      unwind = sym unwind,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _,
      lateout("r14") _, lateout("r15") _,
      // todo: remove these, since they aren't callee-preserved.
      lateout("rdi") _, lateout("rsi") _,
      options(may_unwind)
    )
  }
}

/// Returns the value that the stack pointer should have after
/// [transferring control] to the coroutine associated with the
/// metadata object pointed to by `bottom` (which need not be
/// deferrenceable).
///
/// [transferring control]: transfer_control
pub fn initial_sp<R, F>(bottom: NonNull<Bottom<F>>) -> StackPtr
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
  unsafe { ret_addr.write((trampoline::<R, F> as *const u8).byte_add(1)) }
  ret_addr.cast()
}

/// The initial sequence of instructions executed by a coroutine
/// after it is activated for the first time.
///
/// See [`crate::trampoline`] for some ABI-related considerations.
#[unsafe(naked)]
pub extern "sysv64-unwind" fn trampoline<R, F>(_bottom: NonNull<Bottom<F>>) -> !
where
  F: FnOnce() -> R + Unpin,
{
  // todo: finish comment. Reference `initial_sp` and make clear
  //  that the control record is actually part of the first frame
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
    ".cfi_def_cfa_offset 16",
    ".cfi_offset rsp, -16",
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
pub unsafe extern "sysv64-unwind" fn return_control<R>(
  _control: &mut Control,    // in `rdi`.
  _ret_val_addr: NonNull<R>, // in `rsi`.
) -> ! {
  naked_asm!(
    // Switch back to the previous stack.
    "mov rsp, [rdi + {stack_ptr_off}]",
    // Record the coroutine's return value location.
    "mov [rdi + {stack_ptr_off}], rsi",
    // Mark the coroutine as finished and return to the previous
    // computation.
    "mov rsi, [rdi + {inst_ptr_off}]",
    "mov qword ptr [rdi + {inst_ptr_off}], 0",
    "jmp rsi",
    stack_ptr_off = const offset_of!(Control, stack_ptr),
    inst_ptr_off = const offset_of!(Control, inst_ptr),
  )
}

///
/// # Safety
///
/// The coroutine must be currently [paused]; its computation must
/// have started but not be completely finished.
///
/// [paused]: crate::Coro::is_paused
pub unsafe fn force_unwind(control: &mut Control) {
  // Transfer control back to the `call {unwind}` instruction in
  // `transfer_control`, by moving the resumption address of the
  // coroutine (local label `2` in that function) five bytes back
  // (the length of the `call {unwind}` instruction).
  control.inst_ptr = unsafe {
    control.inst_ptr.byte_sub(5 /*14*/)
  };
  let mut control_ref = AssertUnwindSafe(control);
  let res = crate::unwind::catch_unwind(move || unsafe { transfer_control(*control_ref) });
  debug_assert!(res.is_err(), "coroutine did not unwind");
}
