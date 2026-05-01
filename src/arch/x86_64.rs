use crate::stack::StackPtr;
use crate::{Bottom, Control};
use core::arch::{asm, naked_asm};
use core::mem::offset_of;
use core::panic::AssertUnwindSafe;
use core::ptr::NonNull;

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn transfer_control(_target: &mut Control) {
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
