use crate::Control;
use core::arch::asm;
use core::mem::offset_of;
use core::ptr::NonNull;

/// Returns the current stack pointer value.
///
/// We do not support any platform where the stack may occupy the first page
/// of memory (also known as the "zero page"), hence the stack pointer is
/// always [nonnull].
///
/// [nonnull]: core::ptr::null
#[naked]
pub extern "system" fn stack_ptr() -> NonNull<u8> {
    unsafe { asm!("mov rax, rsp", "ret", options(noreturn)) }
}

/// Resumes execution of a computation at the point where it was last suspended.
/// More precisely, this function saves the program's current context and
/// restores the old context in the given control structure.
///
/// # Safety
///
/// The control record must refer to a previously saved context.
pub unsafe extern "system" fn transfer_control(record: &mut Control) {
    // The transfer of control consists of three parts: saving the previous
    // register contents, replacing the program stack, and jumping to the
    // location following the place where the action was last suspended.
    // Note that we do not need to preserve the contents of _all_ registers;
    // rather, it suffices to save and restore the contents of every register
    // that the resumed computation could _possibly_ destroy. All transfers
    // of control are carried out by the present function, which uses the
    // "system" ABI calling convention. (The "system" ABI name encompasses
    // the System V and Windows ABIs for the x86-64 architecture.) The new
    // computation may destroy the contents of any _callee-saved_ register
    // under this ABI, but it must restore any other state (including the
    // stack and frame pointers) just before it reaches a suspension point
    // or terminates. The `clobber_abi` option below tells the compiler
    // to insert all necessary clobber constraints to save and eventually
    // restore the contents of every callee-saved register that may change
    // during the new computation. (This assumes that the new computation
    // eventually jumps back to the current program; `transfer_control`
    // will never return otherwise.)

    // The `rax` and `rdx` registers are call-clobbered under the "system" ABI,
    // so they are a good choice for temporary storage of intermediate results.
    // Similarly, `rdi` contains the location of the control structure when
    // following version 1.0 of the System V ABI (see Section 3.2.3); we will
    // use the same register to save a `mov` on non-Windows systems.
    asm!(
        // Swap the current stack pointer (`sp`) with the pointer address
        // in the control record (`record.stack_ptr`).
        "mov rax, rsp",
        "mov rsp, [rdi + {stack_ptr_offset}]",
        "mov [rdi + {stack_ptr_offset}], rax",
        // Fetch the resumption point where we are to jump.
        "mov rax, [rdi + {instr_ptr_offset}]",
        // The new computation may transfer control back to the current program
        // by jumping to local label `2`; let us compute its effective address
        // and store it in the control structure.
        "lea rdx, [rip + 2f]",
        "mov [rdi + {instr_ptr_offset}], rdx",
        // Resume the computation.
        "jmp rax",
        "2:",
        in("rdi") record,
        stack_ptr_offset = const offset_of!(Control, stack_ptr),
        instr_ptr_offset = const offset_of!(Control, instr_ptr),
        clobber_abi("system")
    );
}
