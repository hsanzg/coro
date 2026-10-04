use crate::CALLER;
use crate::stack::StackPtr;
use core::arch::{asm, naked_asm};
use core::ptr::NonNull;

// Byte offsets of various fields from the bottom of a coroutine stack.
const INIT_SP_OFF: isize = -8; // initial stack pointer.
const CALLER_REF_OFF: isize = -16; // caller chain pointer.
const TRAMP_PTR_OFF: isize = -24; // trampoline jump address.

pub unsafe fn prepare_stack(bot: StackPtr, tramp_fn: *const ()) {
  // Write the address of the trampoline function immediately
  // above the bottommost frame of the coroutine. The address
  // will be read and jumped to by `init_coro`.
  let ptr = unsafe { bot.byte_offset(TRAMP_PTR_OFF) }.cast();
  unsafe { ptr.write(tramp_fn) };
}

#[inline(always)]
pub unsafe fn initial_sp(bot: StackPtr) -> StackPtr {
  // The `init_coro` subroutine obeys a custom calling convention
  // where its *own* address appears at `rsp` immediately after the
  // call.
  // todo: improve wording.
  unsafe { bot.byte_offset(INIT_SP_OFF) }
}

#[inline(always)]
unsafe fn caller_ref_loc(bot: StackPtr) -> NonNull<()> {
  unsafe { bot.byte_offset(CALLER_REF_OFF) }
}

pub unsafe fn resume_coro(sp: &mut StackPtr, bot: StackPtr) -> *mut () {
  let sus_sp;
  let ret_addr;
  let caller_ref = unsafe { caller_ref_loc(bot) };
  unsafe {
    asm!(
      // Preserve the contents of the reserved register `rbx`;
      // `init_coro` and `suspend_coro` take care of the rest.
      "push rbx",
      // Link the caller to its own caller.
      "push rdx",
      // Read the resumption program address from the top of the
      // coroutine stack, and jump to it. Note that `suspend_coro`
      // and `init_coro` read `rsi` in order to update the "caller
      // reference" at the bottom of the coroutine's stack.
      "call [rdi]",
      // The coroutine has returned control to us. At this point `rdx`
      // contains the new stack pointer of the coroutine; and if it
      // finished via `return_control`, `rax` contains the address of
      // its return value (otherwise `rax` is null). All that remains
      // is to restore the contents of `rbx`, since `rbp` was already
      // restored by `suspend_coro` or `return_control`.
      "pop rbx", // todo: is this more efficient than `add rsp, 8`?
      "pop rbx",
      in("rdi") (*sp).as_ptr(),
      in("rsi") caller_ref.as_ptr(),
      inlateout("rdx") CALLER.get() => sus_sp,
      lateout("rax") ret_addr,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      options(may_unwind),
    )
  };
  *sp = unsafe { NonNull::new_unchecked(sus_sp) };
  ret_addr
}

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn init_coro(_init_sp: StackPtr, _caller_ref: StackPtr) -> ! {
  // Invariant: `_init_sp - INIT_SP_OFF == _caller_ref - CALLER_REF_OFF`.
  naked_asm!(
    ".cfi_startproc simple",
    // todo: add CFI information.
    // Preserve the contents of `rbp`. After this step, the top of
    // the caller stack has a valid frame to which we will soon link.
    "push rbp",
    // Record the stack pointer of the caller program at the bottom
    // of the coroutine stack.
    "mov [rsi], rsp",
    // Prepare the first argument for the trampoline.
    "mov rdi, rsp",
    // Switch to the coroutine stack.
    "mov rsp, rsi", // skip caller reference/frame pointer.
    "mov rbp, rsp",
    // Prepare the second argument for the trampoline.
    "add rsi, {fn_off}",
    // Bounce into the architecture-independent trampoline.
    "jmp [rsp + {tramp_off}]",
    "ud2",
    ".cfi_endproc",
    // These two offsets are relative to `caller_ref`.
    fn_off = const -CALLER_REF_OFF,
    tramp_off = const TRAMP_PTR_OFF - CALLER_REF_OFF,
  )
}

pub unsafe fn suspend_coro(caller: StackPtr) {
  let caller_ref: *const StackPtr;
  unsafe {
    asm!(
      // Preserve the contents of the reserved registers on the
      // coroutine stack.
      "push rbx",
      "push rbp",
      // Record the resumption program address.
      "lea rax, [rip + 2f]",
      "push rax",
      "xor rax, rax", // the coroutine is not done yet.
      // Switch to the caller stack.
      "mov rdx, rsp",
      "mov rsp, rdi",
      // Restore the previous value of `rbp`.
      "pop rbp",
      // Return control to `resume_coro`, which takes care of
      // restoring the remaining registers.
      "ret",
      // The coroutine has been resumed.
      "2:",
      // Preserve the caller's `rbp` value.
      "push rbp",
      // Record the stack pointer of the caller program.
      "mov [rsi], rsp",
      // Switch to the coroutine stack.
      "lea rsp, [rdx + 8]", // skip resumption program address.
      // Restore the contents of the reserved registers.
      "pop rbp",
      "pop rbx",
      in("rdi") caller.as_ptr(),
      lateout("rsi") caller_ref,
      clobber_abi("sysv64"),
      lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
      options(may_unwind),
    )
  };
  let new_caller = unsafe { caller_ref.read() };
  CALLER.set(new_caller.as_ptr());
}

pub unsafe fn return_control(caller: StackPtr, ret_val: NonNull<()>) -> ! {
  unsafe {
    asm!(
      // Switch to the caller stack.
      "mov rsp, rdi",
      // Restore the previous contents of `rbp`.
      "pop rbp",
      // Return control to `resume_coro`.
      "ret",
      in("rdi") caller.as_ptr(),
      in("rax") ret_val.as_ptr(),
      options(noreturn)
    )
  }
}

pub unsafe fn force_unwind(sp: StackPtr, bot: StackPtr) {
  let res = crate::unwind::catch_unwind(move || {
    let caller_ref = unsafe { caller_ref_loc(bot) };
    unsafe {
      asm!(
        // Emulate the "resumption" part of `resume_coro`.
        "push rbx",
        "push rdx",
        "lea rax, [rip + 2f]",
        "push rax",
        // And the "resumption" part of `suspend_coro`.
        "push rbp",
        "mov [rsi], rsp",
        // Switch to the coroutine stack.
        "mov rsp, rdi",
        "pop rax", // the previous return address.
        "pop rbp",
        "pop rbx",
        // Begin the unwinding.
        "push rax", // preserve the return address.
        "jmp {start_unwind}",
        // If the coroutine program catches the exception itself,
        // `suspend_coro` will return control to the following
        // section of code. Emulate the "suspension" part of
        // `resume_coro`, i.e. those instructions after `call`.
        "2:",
        "pop rbx", // todo: is this more efficient than `add rsp, 8`?
        "pop rbx",
        in("rdi") sp.as_ptr(),
        in("rsi") caller_ref.as_ptr(),
        clobber_abi("sysv64"),
        lateout("r12") _, lateout("r13") _, lateout("r14") _, lateout("r15") _,
        options(may_unwind),
        start_unwind = sym crate::unwind::start_unwind,
      )
    };
  });
  assert!(res.is_err(), "coroutine did not unwind");
}

#[unsafe(naked)]
unsafe extern "sysv64-unwind" fn poison_trampoline(_sp: StackPtr, _caller_ref: StackPtr) {
  naked_asm!(
    // Emulate the "resumption" part of `suspend_coro`.
    ".cfi_startproc simple",
    "push rbp",
    "mov [rsi], rsp",
    // Switch to the coroutine stack.
    "lea rsp, [rdi + 8]", // skip resumption address.
    // Restore the contents of the reserved registers.
    "pop rbp",
    "pop rbx",
    // Jump to the target function, instead of returning normally.
    // todo: see how to handle `CALLER.set()`.
    "jmp [rdi - 8]",
    ".cfi_endproc",
  )
}

pub unsafe fn poison(sp: StackPtr, target_fn: extern "sysv64-unwind" fn() -> !) {
  // Overwrite the instruction pointer of the coroutine.
  unsafe { sp.cast().write(poison_trampoline) };
  unsafe { sp.byte_sub(8).cast().write(target_fn) };
}

pub unsafe fn poison_target(cont_sp: StackPtr) -> Option<NonNull<()>> {
  let ip: *const () = unsafe { cont_sp.cast().read() };
  (ip == poison_trampoline as *const _).then(|| unsafe { cont_sp.byte_sub(8).cast().read() })
}

// Old implementation:

/*// Invariant: return address always stored at `control.sp`.

#[inline(always)]
pub unsafe fn resume_coro(target: &mut Control) {
  unsafe {
    asm!(
      // Preserve the current contents of the reserved register `rbx`.
      // Register `rbp` is also reserved, but we push its contents to
      // the stack only after the `call` instruction below transfers
      // control to `trampoline` or `suspend_coro`. In this way the
      // top of the coroutine stack will contain a suitably arranged
      // frame, as specified in Section 3.2.2 of the System V AMD64 ABI.
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

/*pub const fn initial_sp<R, F>(bottom: NonNull<Bottom<R, F>>) -> StackPtr
where
  F: FnOnce() -> R + Unpin,
{
  unsafe { bottom.byte_sub(8).cast() }
}*/

pub const fn first_inst_addr() -> NonNull<Code>
/*where
  F: FnOnce() -> R + Unpin,*/
{
  // todo: replace by `FnPtr::as_ptr` if we can avoid the 1-byte offset.
  // See `first_tramp` for the reason behind the 1-byte offset.
  unsafe { fn_ptr!(first_tramp).byte_add(1) }
  // (first_tramp/*::<R, F>*/ as *const ()).byte_add(1)
}

// const FIRST_INST_ADDR: NonNull<Code> = unsafe { fn_ptr!(first_tramp).byte_add(1) };

/*pub fn prepare_stack_old<R, F>(bottom: NonNull<Bottom<R, F>>)
where
  F: FnOnce() -> R + Unpin,
{
  let pad = unsafe { bottom.byte_add(8).cast() };
  unsafe { pad.write(ptr::null::<()>()) };
  // let ret_addr = unsafe { bottom.byte_add(8).cast() };
  // unsafe { ret_addr.write(resume_coro as *const ()) };
  let tramp_ret_addr = unsafe { bottom.byte_sub(8).cast() };
  unsafe { tramp_ret_addr.write(first_inst_addr::<R, F>()) };
}*/

pub unsafe fn prepare_stack(
  stack: &mut Stack,
  fn_layout: Layout,
  trampoline_addr: NonNull<Code>,
) -> (NonNull<()>, NonNull<Control>, StackPtr) {
  let mut top = stack.bottom();
  let fn_ptr = unsafe { push(&mut top, fn_layout) };
  let ctrl_ptr = unsafe { push(&mut top, Layout::new::<Control>()) }.cast();
  // todo: write address of arch trampoline as the return address of
  //       the first frame.
  // todo: write address of second trampoline on top of that address;
  //       it is to be read by `first_tramp` and be placed on the
  //       appropriate register for `trampoline::<R, F>` to accept it
  //       correctly.

  // Arrange things so that the frame corresponding to `first_tramp`
  // has the field `stack_ptr` of the coroutine's control record as
  // the "previous `rbp` value" slot. (See the System V AMD64 ABI.)
  // In this way the bottommost frame of the coroutine always points
  // to the topmost frame of the program that last resumed it. The
  // return address of this first frame will be invalid, but that
  // does not seem to prevent `libunwind` nor `libgcc` to proceed
  // normally through it during unwinding.
  let mut push_fn_ptr = |ptr: NonNull<Code>| {
    let loc = unsafe { push(&mut top, Layout::for_value(&ptr)) }.cast();
    unsafe { loc.write(ptr) };
    loc
  };
  let sp = push_fn_ptr(first_inst_addr()).cast();
  push_fn_ptr(trampoline_addr);
  push_fn_ptr(unsafe { NonNull::new_unchecked(fn_ptr.as_ptr() as *mut _) });
  (fn_ptr, ctrl_ptr, sp)
}

// Define an LSDA for the personality function of the trampoline function.
// See https://www.airs.com/blog/archives/464 for an explanation of the format.
// Also see https://maskray.me/blog/2020-12-12-c++-exception-handling-abi.
global_asm!(
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
);

#[unsafe(naked)]
pub unsafe extern "sysv64-unwind" fn first_tramp(_active: &mut Control, _coro_sp: StackPtr) -> ! {
  naked_asm!(
    ".cfi_startproc simple", // not simple
    ".cfi_personality 0x1b, {personality_fn}", // 0x1b = DW_EH_PE_pcrel | DW_EH_PE_sdata4
    ".cfi_lsda 0x1b, __coro_tramp_lsda",
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
    // Load the address of the coroutine function from its stack,
    // and pass it as the second argument to the arch-independent
    // trampoline.
    "mov rsi, [rdi - 16]",
    // Bounce into the architecture-independent trampoline.
    ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size, 0
    // "jmp {second_tramp}",
    "jmp [rsp - 8]",
    "ud2",
    ".cfi_endproc",
    sp_off = const offset_of!(Control, stack_ptr),
    //caller_off = const offset_of!(Control, caller),
    personality_fn = sym crate::unwind::__coro_trampoline_personality,
    // second_tramp = sym crate::trampoline::<R, F>,
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

#[unsafe(naked)]
unsafe extern "sysv64-unwind" fn on_poison_resume() {
  naked_asm!(
    // todo: Note that there's no need to have correct CFI directives,
    //       because the final `jmp` "eliminates" `on_resume` from
    //       the call stack. (It makes it seem as if the original
    //       caller of `suspend` within the coroutine had called
    //       `dst`.) But they would be nice to have anyway.
    ".cfi_startproc simple",
    // Preserve the current contents of the `rbp` register. After
    // this step, the top of the parent stack has a valid frame.
    "push rbp",
    // Record the stack pointer value of the parent program.
    "mov [rdi + {sp_off}], rsp",
    // Switch to the coroutine stack.
    // todo: document that at this point `rsi` contains the control record location.
    "lea rsp, [rsi - 8]",
    // Load the new and the original resumption addresses.
    "pop rsi", // new.
    "pop rax", // original.
    // Restore the previous contents of the reserved registers.
    "pop rbp",
    "pop rbx",
    // Jump to the new destination (instead of returning, like `suspend`),
    // making sure to preserve the chain of call frames through `rax`.
    ".cfi_escape 0x2e, 0", // DW_CFA_GNU_args_size 0
    "push rsi",
    "jmp rax",
    ".cfi_endproc",
    sp_off = const offset_of!(Control, stack_ptr),
  );
}

// todo: rename to `prime`? Come up with better name.
// todo: note that coroutine must be suspended.
pub unsafe fn poison(ctrl: &mut Control, dst: extern "C-unwind" fn() -> !) {
  // todo: document safety; the stack pointer of a suspended coroutine
  //       is always 8-byte aligned.
  unsafe { ctrl.stack_ptr.cast().write(fn_ptr!(on_poison_resume)) };
  extern crate std;
  std::println!(
    "poison: overwrote ip at {:?} by {:?}",
    ctrl.stack_ptr,
    on_poison_resume as *const ()
  );

  let dst_ptr = unsafe { ctrl.stack_ptr.cast().byte_sub(1) };
  unsafe { dst_ptr.write(fn_ptr!(dst)) };
  // unsafe { ctrl.stack_ptr.cast().write(fn_ptr!(on_resume)) };
  // let dst_ptr = unsafe { ctrl.stack_ptr.cast().byte_sub(1) };
  // unsafe { dst_ptr.write(dst) };
}

pub fn poison_dst(ctrl: &Control) -> Option<NonNull<Code>> {
  extern crate std;
  std::println!(
    "checking whether poisoned @ {:?} (well it has {:?})",
    ctrl.stack_ptr,
    unsafe { ctrl.stack_ptr.cast::<*const Code>().read() }
  );
  ptr::eq(
    unsafe { ctrl.stack_ptr.cast().read() },
    unsafe { fn_ptr!(on_poison_resume) }.as_ptr(),
  )
  .then(|| {
    let dest_addr = unsafe { ctrl.stack_ptr.cast().byte_sub(1) };
    std::println!("  yes! dest address is {:?}", unsafe { dest_addr.read() });
    unsafe { dest_addr.read() }
  })
}*/

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
