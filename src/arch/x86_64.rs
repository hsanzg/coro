use crate::{Bottom, Control};
use std::arch::{asm, naked_asm};
use std::mem::offset_of;
use std::pin::Pin;
use std::ptr::NonNull;

pub unsafe extern "sysv64-unwind" fn transfer_control(target: &mut Control) {
  unsafe {
    asm!(
      // Save additional caller-preserved registers in the current stack.
      "sub rsp, 16",
      ".cfi_adjust_cfa_offset 16",
      "mov [rsp + 8], rbx",
      ".cfi_rel_offset rbx, 8",
      "mov [rsp], rbp",
      ".cfi_rel_offset rbp, 0",
      //
      "mov rax, [rdi + {sp_off}]",
      "mov [rdi + {sp_off}], rsp",
      "mov rsp, rax",
      //
      "mov rax, [rdi + {ip_off}]",
      "lea rsi, [rip + 2f]",
      "mov [rdi + {ip_off}], rsi",
      "jmp rax",
      //
      "2:",
      "mov rbp, [rsp]",
      ".cfi_restore rbp",
      "mov rbx, [rsp + 8]",
      ".cfi_restore rbx",
      "add rsp, 16",
      ".cfi_adjust_cfa_offset -16",
      inlateout("rdi") target => _,
      lateout("rax") _, lateout("rsi") _,
      sp_off = const offset_of!(Control, stack_ptr),
      ip_off = const offset_of!(Control, inst_ptr),
      clobber_abi("sysv64"),
      options(may_unwind)
    )
  }
}

/*pub unsafe extern "sysv64-unwind" fn suspend(active: NonNull<Control>) {
  todo!()
}*/

/*pub unsafe extern "sysv64-unwind" fn transfer_control(control: Pin<&mut Control>) {
  unsafe {
    asm!(
      "sub rsp, 32",
      ".cfi_adjust_cfa_offset 32",
      "mov [rsp + 24], rbx",
      ".cfi_rel_offset rbx, 24",
      // todo: check if doing the offset 24 write and then the 16 is a bit faster.
      "mov [rsp + 16], rbp",
      ".cfi_rel_offset rbp, 16",
      // (Leave [rsp] and [rsp+8] empty for a future `transfer_control` call to
      // place a stack pointer and a return address value there, respectively.)

      "mov rax, [rdi + {stack_ptr_off}]",
      // Record the old value of `sp` both in the control record
      // and at the top of the new stack.
      "mov [rdi + {stack_ptr_off}], rsp",
      "mov [rax], rsp",
      // Switch to the new stack.
      "mov rsp, rax",
      // Notes: If I push something here to the new stack, then
      // it would affect the following instructions:
      // - the first instruction of the trampoline,
      // - the instruction following label `2` below, when returning
      //   to a suspended coroutine.
      ".cfi_remember_state",
      ".cfi_def_cfa rsp, 16",
      ".cfi_offset rip, -8", // todo: we have not yet written the return address, reorder code or delay.
      ".cfi_offset rsp, -16",
      "",
      "mov rax, [rdi + {inst_ptr_off}]",
      "lea rsi, [rip + 2f]",
      "mov [rdi + {inst_ptr_off}], rsi",
      "mov [rsp + 8], rsi",
      "jmp rax",
      "2:",
      ".cfi_restore_state",
      // Note that this code does not run when a coroutine
      // completely finishes execution; it instead returns
      // control to its final caller via `return_control`.
      // Otherwise the following instructions would read
      // out of bounds stuff from the control record.
      "mov rbp, [rsp + 16]",
      ".cfi_restore rbp",
      "mov rbx, [rsp + 24]",
      ".cfi_restore rbx",
      "add rsp, 32",
      ".cfi_adjust_cfa_offset -32",
      // todo: document safety of `into_inner_unchecked`.
      inlateout("rdi") Pin::into_inner_unchecked(control) => _,
      lateout("rax") _, lateout("rsi") _,
      // todo: rename to `csp_off` and `cip_off`?
      stack_ptr_off = const offset_of!(Control, stack_ptr),
      inst_ptr_off = const offset_of!(Control, inst_ptr),
      clobber_abi("system"),
      options(may_unwind)
    )
  }
}*/

#[unsafe(naked)]
pub extern "sysv64-unwind" fn trampoline<R, F>(
  _control: NonNull<Control>, // in `rdi`.
  _ret_addr: NonNull<()>,     // in `rsi`.
) -> !
where
  F: FnOnce() -> R + Unpin,
{
  naked_asm!(
    ".cfi_startproc",
    ".cfi_offset rsp, -16",
    // ".cfi_offset rsp, -16",
    // "sub rsp, 16",
    // ".cfi_adjust_cfa_offset 16",
    // "mov [rsp], rbp",
    // ".cfi_offset rbp, -24",
    // "mov rbp, rsp",
    // ".cfi_def_cfa_register rbp",
    "sub rsp, 8",
    ".cfi_adjust_cfa_offset 8",
    "call {second_tramp}",
    ".cfi_endproc",
    second_tramp = sym crate::trampoline::<R, F>
  )
}

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn return_control<R>(
  _control: &mut Control,    // in `rdi`.
  _ret_val_addr: NonNull<R>, // in `rsi`.
) -> ! {
  naked_asm!(
    "mov rsp, [rdi + {stack_ptr_off}]",
    "mov [rdi + {stack_ptr_off}], rsi",
    "mov rsi, [rdi + {inst_ptr_off}]",
    "mov qword ptr [rdi + {inst_ptr_off}], 0",
    "jmp rsi",
    stack_ptr_off = const offset_of!(Control, stack_ptr),
    inst_ptr_off = const offset_of!(Control, inst_ptr),
  )
}

pub fn initial_sp<F>(bottom: NonNull<Bottom<F>>) -> NonNull<()> {
  unsafe { bottom.byte_add(offset_of!(Control, inst_ptr)) }.cast()
}
