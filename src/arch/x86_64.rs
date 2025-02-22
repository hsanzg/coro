use crate::Control;
use core::arch::asm;
use core::mem::offset_of;
use core::ptr::{Alignment, NonNull};

/// Returns the current stack pointer value.
///
/// We do not support any platform where the stack may occupy the first page
/// of memory (also known as the "zero page"), hence the stack pointer is
/// always [nonnull].
///
/// [nonnull]: core::ptr::null
// #[naked]
pub extern "system" fn stack_ptr() -> NonNull<u8> {
    // The `rax` register is to contain the first return value under the
    // System V and Windows ABI calling conventions for the x86-64 architecture.
    // todo: replace the current implementation by the following line, and mark this function as
    //       #[naked] once https://github.com/rust-lang/rust/pull/127853#issuecomment-2257323333
    //       is resolved.
    // unsafe { asm!("mov rax, rsp", "ret", options(noreturn)) }
    let mut sp: *mut u8;
    unsafe {
        asm!(
            "mov {sp}, rsp",
            sp = lateout(reg) sp,
            options(nomem, nostack)
        )
    };
    unsafe { NonNull::new_unchecked(sp) }
}

/// Saves the contents of every reserved, callee-saved register under the
/// current ABI.
#[cfg(target_family = "unix")]
macro_rules! save_additional_context {
    () => {
        // The System V ABI defines a 128-byte red zone above the top of the
        // current stack, so there's no need to increase the stack pointer.
        // Still, we need to update the CFA and indicate that the `rbp` and
        // `rbx` old register contents are present in the stack.
        "mov [rsp - 16], rbp
         mov [rsp - 8], rbx
         #.cfi_def_cfa_offset 16
         .cfi_rel_offset rbp, -16
         .cfi_rel_offset rbx, -8"
    };
}

/// Undoes the effect of [`save_additional_context`].
#[cfg(target_family = "unix")]
macro_rules! restore_additional_context {
    () => {
        "mov rbp, [rsp - 16]
         mov rbx, [rsp - 8]"
    };
}

#[cfg(target_family = "windows")]
macro_rules! save_additional_context {
    () => {
        // Under the Windows ABI, the contents above the top of the stack may
        // be overwritten by an interrupt handler. Update the stack pointer.
        "push rbx
         push rbp"
    };
}

#[cfg(target_family = "windows")]
macro_rules! restore_additional_context {
    () => {
        "pop rbp
         pop rbx"
    };
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
        // Preserve the contents of the `rbx` and `rbp` registers manually. The
        // former is not included in the clobber list because LLVM uses it
        // internally; similarly, Rust does not allow us to specify the frame
        // pointer as an input nor output.
        save_additional_context!(),
        // Swap the current stack pointer (`sp`) with the pointer address
        // in the control record (`record.stack_ptr`). The stack setup section
        // of code in `Coro::new` ensures that the stack pointer is properly
        // aligned during the first coroutine activation. (See Section 3.2.3
        // of the System V ABI for details.)
        "mov rax, rsp",
        "mov rsp, [rdi + {stack_ptr_offset}]",
        "mov [rdi + {stack_ptr_offset}], rax",
        "mov rbp, rdi", // todo: can we avoid this renaming? Probably not, because
                        //       we need a callee-preserved register.
        // Define the Canonical Frame Address (CFA) of the current computation,
        // namely the value of the stack pointer just before we jump the control
        // to the new program. The `DW_CFA_def_cfa_expression` (`0x0F`) call
        // frame instruction takes a single `DW_FORM_exprloc` value consisting
        // of a LEB128-encoded length (3) and a DWARF expression that tells the
        // unwinder how to compute the current CFA. (See Section 2.5 of the
        // DWARF Version 5 specification for details.) At this point the value
        // of the `rdi` register is the location of the control record, whose
        // `stack_ptr` field contains the desired CFA value. Push the offset of
        // that field (`stack_ptr_offset`) and `DW_OP_breg5` (`0x75`, which
        // corresponds to the `rdi` register on the x86-64 architecture) onto
        // the DWARF stack. Next, apply the `DW_OP_deref` (`0x06`) operation to
        // read the field contents.
        // todo: replace breg5 (rdi) by breg6 (rbp) in comment above.
        ".cfi_escape 0x0F, 3, 0x76, {stack_ptr_offset}, 0x06",
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
        // todo: Do we need to restore the original CFA?
        // At this point we have come back to the original stack. It remains to
        // restore the contents of all manually-preserved registers.
        restore_additional_context!(),
        in("rdi") record,
        stack_ptr_offset = const offset_of!(Control, stack_ptr),
        instr_ptr_offset = const offset_of!(Control, instr_ptr),
        clobber_abi("system"), // includes `rax` and `rdx`.
        // The `clobber_abi("system")` clobber set is currently missing some
        // registers; specify them manually.
        lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
    );
}

/// Returns to the instruction immediately after the [`resume`] call that
/// activated the current [coroutine]. This function is a special version
/// of [`transfer_control`] for the case when the coroutine will never get
/// activated again (hence the `!` return type); for example, it does not
/// save the previous _callee-saved_ register contents, nor does it store
/// the current instruction pointer in the coroutine's [control record].
/// It is important to note that a call to one function cannot be replaced
/// by one to the other, however, because `return_control` is the only to
/// "mark" the coroutine as finished. (The [`Coro::is_finished`] method
/// has further details; in short, this function resets the instruction
/// pointer to zero and writes the address `ret_val_addr` of the coroutine's
/// return value to the stack pointer field.)
///
/// # Safety
///
/// This function must be called from within a coroutine that shall never be
/// resumed again.
///
/// [`resume`]: crate::Coro::resume
/// [coroutine]: crate::Coro
/// [control record]: Control
/// [`Coro::is_finished`]: crate::Coro::is_finished
pub unsafe extern "system" fn return_control<R>(
    record: &mut Control,
    ret_val_addr: NonNull<R>,
) -> ! {
    asm!(
        "mov rsp, [rdi + {stack_ptr_offset}]",
        "mov [rdi + {stack_ptr_offset}], rdx",
        "mov rax, [rdi + {instr_ptr_offset}]",
        "mov qword ptr [rdi + {instr_ptr_offset}], 0",
        // Jump to the last caller (which takes care of restoring the CFA).
        "jmp rax",
        in("rdi") record,
        in("rdx") ret_val_addr.as_ptr(),
        stack_ptr_offset = const offset_of!(Control, stack_ptr),
        instr_ptr_offset = const offset_of!(Control, instr_ptr),
        options(noreturn)
    )
}

/// The ABI-required minimum alignment of a stack frame.
pub const STACK_FRAME_ALIGN: Alignment = unsafe { Alignment::new_unchecked(16) };
