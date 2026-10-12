use crate::stack::StackPtr;
use crate::unwind::Completion;
use crate::{Caller, PoisonTarget};
use core::arch::{asm, naked_asm};
use core::hint::unlikely;
use core::ptr::NonNull;

// Byte offsets of various fields from the bottom of a coroutine stack.
const INIT_SP_OFF: isize = -8; // initial stack pointer.
const CALLER_REF_OFF: isize = -16; // caller chain pointer.
const PREV_FRAME_PTR_OFF: isize = -16; // link to last caller's frame.
const TRAMP_PTR_OFF: isize = -24; // trampoline jump address.

/// Returns the initial stack pointer of the coroutine whose stack
/// starts at `bot`; that is, the value the stack pointer will have
/// when control is first [transferred] to the coroutine.
///
/// [transferred]: resume_coro
unsafe fn initial_sp(bot: StackPtr) -> StackPtr {
  unsafe { bot.byte_offset(INIT_SP_OFF) }
}

/// Restores the coroutine stack starting at `bot` to its initial
/// state, so that the next call to [`resume_coro`] on it enters
/// [`init_coro`] and jumps to `tramp_fn`.
///
/// For convenience, the [initial stack pointer](initial_sp) of
/// the coroutine is returned.
pub unsafe fn prepare_stack(bot: StackPtr, tramp_fn: *const ()) -> StackPtr {
  // The `init_coro` subroutine obeys a custom calling convention
  // where its *own* address appears at `rsp` immediately after the
  // call.
  // todo: improve wording.
  let ip = unsafe { bot.byte_offset(INIT_SP_OFF).cast() };
  unsafe { ip.write(init_coro as *const ()) };
  // Write the address of the trampoline function immediately
  // above the bottommost frame of the coroutine. The address
  // will be read and jumped to by `init_coro`.
  let tp = unsafe { bot.byte_offset(TRAMP_PTR_OFF) }.cast();
  unsafe { tp.write(tramp_fn) };
  // Leave the caller reference field uninitialized.
  unsafe { initial_sp(bot) }
}

// todo: document.
unsafe fn prev_frame_ptr_loc(bot: StackPtr) -> NonNull<StackPtr> {
  unsafe { bot.byte_offset(PREV_FRAME_PTR_OFF) }.cast()
}

/// Transfers control to the coroutine whose stack starts at `bot`,
/// resuming it where it was last suspended (or starting it, if it
/// has not yet begun execution). This subroutine returns once the
/// coroutine transfers control back via [`suspend_coro`] or
/// [`return_control`].
#[inline(always)]
pub unsafe fn resume_coro(sp: &mut StackPtr, bot: StackPtr, caller: *mut Caller) -> Completion {
  let (comp, sus_sp);
  unsafe {
    asm!(
      // Preserve the contents of the reserved register `rbx`;
      // `init_coro` and `suspend_coro` take care of the rest.
      "push rbx",
      // Read the resumption program address from the top of the
      // coroutine stack, and jump to it. Note that `suspend_coro`
      // and `init_coro` read `rdi` and `rsi` in order to update
      // the thread-local `caller`, and to store a pointer to the
      // caller's frame at the bottom of the coroutine's stack.
      "call [rax]",
      // The coroutine has returned control to us. At this point `rsi`
      // contains the new stack pointer of the coroutine; and `rax` is
      // the tagged pointer that tells whether the coroutine finished
      // completely or not, and if its stack was unwound forcibly or
      // due to a panic. All that remains is to restore the contents
      // of `rbx`, since `rbp` was already restored by `suspend_coro`
      // or `return_control`.
      "pop rbx",
      in("rdi") prev_frame_ptr_loc(bot).as_ptr(),
      inlateout("rsi") caller => sus_sp,
      inlateout("rax") (*sp).as_ptr() => comp,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      options(may_unwind),
    )
  };
  *sp = unsafe { NonNull::new_unchecked(sus_sp) };
  unsafe { Completion::from_raw(comp) }
}

/// The initial sequence of instructions executed by a coroutine
/// after it is activated for the first time or after being poisoned.
///
/// See [`crate::trampoline`] for some ABI-related considerations.
#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn init_coro(
  _prev_frame_ptr: StackPtr,
  _caller: &mut Caller,
) -> ! {
  naked_asm!(
    ".cfi_startproc simple",
    // Place this frame on the caller's stack.
    // DW_CFA_def_cfa_expression DW_FORM_exprloc(len=5, DW_OP_breg6/rbp 0 DW_OP_deref DW_OP_plus_uconst 24)
    ".cfi_escape 0x0f, 5, 0x76, 0, 0x06, 0x23, 24",
    ".cfi_offset rbx, -8",
    ".cfi_offset rip, -16",
    ".cfi_offset rbp, -24",
    // Preserve the contents of `rbp`. After this step, the top of
    // the caller stack has a valid frame to which we will link next.
    "push rbp",
    // Record the stack pointer of the caller program
    // - in the thread-local `caller` variable; and
    "mov [rsi], rsp",
    // - at the bottom of the coroutine stack.
    "mov [rdi], rsp",
    // todo: put comment above.
    // debuggers and profilers without DWARF CFI support can
    // can walk from the coroutine's back to the caller's stack.
    // the caller's frame pointer at the bottom
    // of the coroutine's stack.

    // The latter is required only to form a valid chain of
    // frame pointers for debuggers that do not support DWARF
    // CFI information; see `init_coro` for details.

    // Switch to the coroutine stack.
    "mov rsp, rdi", // skip initial resumption address.
    "mov rbp, rdi",
    // Prepare the argument for the trampoline.
    "add rdi, {fn_off}",
    // Bounce into the architecture-independent trampoline.
    // The stack pointer is properly aligned for a `sysv64` call,
    // since `rsp - CALLER_REF_OFFSET` (the bottom of the stack)
    // is 16-byte aligned by construction (see `Coro::with_stack`).
    "call [rsp + {tramp_off}]",
    "ud2",
    ".cfi_endproc",
    // These two offsets are relative to `_prev_frame_ptr`.
    fn_off = const -CALLER_REF_OFF,
    tramp_off = const TRAMP_PTR_OFF - CALLER_REF_OFF,
  )
}

/// Jumps to the instruction immediately after the [`resume_coro`]
/// call that last activated the current coroutine.
#[inline(always)]
pub unsafe fn suspend_coro(caller: Caller) {
  unsafe {
    asm!(
      // Preserve the contents of the reserved registers on the
      // coroutine stack.
      "push rbx",
      "push rbp",
      // Record the resumption program address.
      "lea rax, [rip + 2f]",
      "push rax",
      "xor eax, eax", // the coroutine is not done yet.
      // Switch to the caller stack.
      "mov rsi, rsp", // needed by `resume_coro`.
      "mov rsp, rdi",
      // Restore the previous value of `rbp`.
      "pop rbp",
      // Return control to `resume_coro`, which takes care of
      // restoring the remaining registers.
      "ret",
      // The coroutine has been resumed by `resume_coro`.
      "2:",
      // Preserve the caller's `rbp` value.
      "push rbp",
      // Record the stack pointer of the caller program.
      "mov [rsi], rsp",
      "mov [rdi], rsp",
      // Switch to the coroutine stack.
      "lea rsp, [rax + 8]", // skip resumption program address.
      // Restore the contents of the reserved registers.
      "pop rbp",
      "pop rbx",
      in("rdi") caller,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      options(may_unwind),
    )
  };
}

/// A variant of [`suspend_coro`] for the case when the current
/// coroutine will never be activated again (unless it is first
/// [poisoned](poison)).
pub unsafe fn return_control(caller: StackPtr, final_sp: StackPtr, comp: Completion) -> ! {
  unsafe {
    asm!(
      // Switch to the caller stack.
      "mov rsp, rdi",
      // Restore the previous contents of `rbp`.
      "pop rbp",
      // Return control to `resume_coro`.
      "ret",
      in("rdi") caller.as_ptr(),
      in("rsi") final_sp.as_ptr(), // needed by `resume_coro`.
      in("rax") comp.into_raw(), // ditto.
      options(noreturn)
    )
  }
}

/// Unwinds the stack of a [paused] coroutine, running the destructors
/// of all live objects in it. After this operation, the coroutine can
/// be resumed or unwound again only after being [poisoned].
///
/// Note that a poisoned coroutine is never paused, and so its stack
/// cannot be unwound.
///
/// [paused]: crate::Coro::is_paused
/// [poisoned]: poison
pub unsafe fn force_unwind(sp: &mut StackPtr, bot: StackPtr, caller: *mut Caller) -> Completion {
  let (comp, final_sp);
  unsafe {
    asm!(
      // Emulate the "resumption" part of `resume_coro`.
      "push rbx",
      "lea rbx, [rip + 2f]",
      "push rbx",
      // And the "resumption" part of `suspend_coro`.
      "push rbp",
      "mov [rsi], rsp",
      "mov [rdi], rsp",
      // Switch to the coroutine stack.
      "mov rsp, rax",
      "pop rax", // the previous resumption address.
      "pop rbp",
      "pop rbx",
      // Begin the unwinding.
      "push rax", // preserve the resumption address.
      "jmp {start_unwind}",
      // The trampoline function catches the exception itself,
      // and returns control normally to the following section
      // of code. Emulate the "suspension" part of `resume_coro`,
      // i.e. those instructions after the resumption `call`.
      "2:",
      "pop rbx",
      in("rdi") prev_frame_ptr_loc(bot).as_ptr(),
      inlateout("rsi") caller => final_sp,
      inlateout("rax") (*sp).as_ptr() => comp,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      options(may_unwind),
      start_unwind = sym crate::unwind::start_unwind,
    )
  };
  *sp = unsafe { NonNull::new_unchecked(final_sp) };
  unsafe { Completion::from_raw(comp) }
}

/// Restores the stack of the coroutine whose stack starts at `bot`
/// to its initial state, but with [`poison_trampoline`] instead of
/// the original trampoline and `target_fn` instead of the original
/// target function. Returns the new stack pointer of the coroutine.
///
/// A poisoned coroutine can return control back only by panicking
/// or [forcibly unwinding] its stack.
///
/// [`poison_trampoline`]: crate::poison_trampoline
/// [forcibly unwinding]: force_unwind
#[must_use]
pub unsafe fn poison(bot: StackPtr, target_fn: PoisonTarget) -> StackPtr {
  unsafe { bot.cast().write(target_fn) };
  unsafe { prepare_stack(bot, crate::poison_trampoline as *const _) }
}

/// Returns the target function of a poisoned coroutine,
/// or [`None`] if the coroutine is not currently poisoned.
pub unsafe fn poison_target(cont_sp: StackPtr) -> Option<PoisonTarget> {
  ({
    let ip = unsafe { cont_sp.cast().read() };
    init_coro as *const () == ip
  } && {
    // Is `init_coro` going to jump the alternative trampoline?
    let tp = unsafe { cont_sp.byte_offset(TRAMP_PTR_OFF - INIT_SP_OFF) };
    crate::poison_trampoline as *const () == unsafe { tp.cast().read() }
  })
  .then(|| {
    // Read the landing point of that trampoline.
    let bot = unsafe { cont_sp.byte_offset(-INIT_SP_OFF) };
    unsafe { bot.cast().read() }
  })
}

/// Returns whether the (suspended) coroutine whose stack
/// pointer is currently `cont_sp` has begun execution.
///
/// This is the architecture-specific part of [`Coro::has_started`].
///
/// [`Coro::has_started`]: crate::Coro::has_started
pub unsafe fn has_started(cont_sp: StackPtr) -> bool {
  // Either the instruction pointer has moved, or the coroutine is poisoned.
  let ip: *const () = unsafe { cont_sp.cast().read() };
  if ip != init_coro as *const _ {
    return true;
  }
  let tp = unsafe { cont_sp.byte_offset(TRAMP_PTR_OFF - INIT_SP_OFF) };
  unlikely(crate::poison_trampoline as *const () == unsafe { tp.cast().read() })
}

// Rust translation of the client request macros in `valgrind.h`. See
// https://sourceware.org/git/?p=valgrind.git;a=blob;f=include/valgrind.h.in;h=ed7d8c68e24b896e912fb19e17cc906730a82a4c;hb=refs/heads/master#l409.

#[cfg(feature = "valgrind")]
fn valgrind_client_req(default: usize, args: &[usize; 6]) -> usize {
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
pub fn valgrind_register_stack(start: *mut (), end: *mut ()) -> ValgrindStackId {
  let res = valgrind_client_req(0, &[0x1501, start as _, end as _, 0, 0, 0]);
  ValgrindStackId(res as _)
}

#[cfg(feature = "valgrind")]
pub fn valgrind_deregister_stack(id: ValgrindStackId) {
  valgrind_client_req(0, &[0x1502, id.0 as _, 0, 0, 0, 0]);
}
