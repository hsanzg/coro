// The following doc comment is kept in sync with the README.md file.
// Please execute `cargo sync-readme` after modifying its contents.
//! This crate provides an implementation of first-class stackful
//! asymmetric coroutines.
//!
//! The general notion of _coroutines_, first discussed in the published
//! literature by M. E. Conway \[_CACM_ **6** (1963), 396–408], extends
//! the concept of subroutines by allowing them to share and pass data
//! and control back and forth. An _asymmetric_ coroutine suspends the
//! execution of its program by invoking a [`suspend`] function, which
//! returns control to the caller. Invoking the [`resume`] method on a
//! coroutine resumes execution of its program immediately after the
//! point where it was last suspended.
//!
//! The Rust language has built-in support for asymmetric coroutines,
//! but one cannot easily suspend their execution from within a nested
//! function call. One way to solve this issue is to maintain a separate
//! program stack for each coroutine, at the expense of some extra memory
//! space. The resulting coroutine implementation is said to be _stackful_.
//!
//! The crux of this crate is the [`Coro`] type, which implements the
//! built-in [`core::ops::Coroutine`] trait using a stackful approach.
//! Perhaps its most interesting feature is that it allows a coroutine
//! to return control without the need to pass a "yielder" object around.
//! (Instead, a thread-local variable keeps track of the currently active
//! coroutine.)
//!
//! The paper "Revisiting Coroutines" by A. L. de Moura and R. Ierusalimschy
//! \[_ACM TOPLAS_ **31** (2009), 1–31] formally defines the terms "stackful",
//! "first-class" and "asymmetric" used above.
//!
//! `coro` incorporates ideas from two other implementations of stackful
//! coroutines: [`corosensei`] by Amanieu d'Antras, and [`libfringe`] by edef.
//! Preliminary experiments seem to indicate that the three libraries can
//! transfer control between coroutines with comparable efficiency, but
//! `coro` is still inferior with respect to platform support. I mainly
//! put together this library for my own education, so significant
//! improvements are likely possible.
//!
//! [`resume`]: Coro::resume
//! [`corosensei`]: https://github.com/Amanieu/corosensei
//! [`libfringe`]: https://github.com/edef1c/libfringe

#![feature(asm_unwind)]
#![feature(core_intrinsics)] // todo: document purpose.
#![feature(std_internals)] // for `PanicPayload`.
#![feature(coroutine_trait)]
#![feature(rustc_attrs)] // todo: document purpose.
#![feature(thread_local)]
#![feature(likely_unlikely)]
#![feature(abort_immediate)]
#![feature(ptr_metadata)] // todo: can we remove?
#![cfg_attr(all(panic = "unwind", target_arch = "x86_64"), feature(panic_unwind))]
#![expect(
  internal_features,
  reason = "we unwind coroutine stacks by throwing a non-Rust exception type, and need `core::intrinsics::catch_unwind` to catch it"
)]
#![cfg_attr(all(test, feature = "stack_alloc"), feature(pointer_is_aligned_to))]
#![no_std]

#[cfg(all(panic = "unwind", target_arch = "x86_64"))]
extern crate unwind as uw;

#[cfg(any(panic = "unwind", feature = "stack_alloc"))]
extern crate alloc;

mod arch;
pub mod stack;
mod unwind;

use crate::stack::{GrowthDir, Stack, StackPtr};
use crate::unwind::{CatchResult, Completion};
use core::cell::Cell;
use core::hint::cold_path;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr;
use core::ptr::NonNull;
#[allow(unused_imports)] // false positive depending on platform.
use coro_macros::abi;

type Caller = *const ();

/// The stack pointer of the program that last resumed the currently
/// active coroutine, or null if no coroutine is active.
#[thread_local]
static CALLER: Cell<Caller> = Cell::new(ptr::null());

/// Returns whether this subroutine is being called from
/// within a [coroutine](Coro) or the main program.
pub fn in_coro() -> bool {
  !CALLER.get().is_null()
}

/// The default size in bytes of the [program stack] of a [coroutine].
/// We currently use Rust's [default stack size] for a thread, 2 MiB.
///
/// [program stack]: Stack
/// [coroutine]: Coro
/// [default stack size]: https://doc.rust-lang.org/std/thread/#stack-size
pub const DEFAULT_STACK_SIZE: usize = 1 << 21;

/// A first-class stackful asymmetric coroutine.
///
/// The [crate-level documentation] discusses this type in more detail.
///
/// [crate-level documentation]: crate
pub struct Coro<R> {
  /// The value of the stack pointer just before the last suspension
  /// of the coroutine.
  ///
  /// Special cases:
  /// - if the coroutine has not been activated yet, the pointer
  ///   returned by [`arch::prepare_stack`];
  /// - if the coroutine has finished normally or its stack has been
  ///   unwound (forcibly or because of a panic), the pointer returned
  ///   by [`arch::poison`].
  cont_sp: StackPtr,
  /// The bottom of the stack, as returned by after pushing the coroutine function to the stack.
  stack_bot: StackPtr,
  /// A type-erased pointer to a generic function for destroying
  /// the coroutine function given `stack_bot`, provided that
  /// the coroutine has not started execution.
  /// calling this function does not require external knowledge about
  /// the type of the function, only its location in memory.
  drop_fn: unsafe fn(StackPtr),
  /// The program stack of the coroutine.
  stack: ManuallyDrop<Stack>,
  /// This object acts as though it points to the return value
  /// of the coroutine.
  // todo: talk about covariance.
  _phantom: PhantomData<*mut R>,
}

// todo: document.
const fn fn_ptr<F>(bot: StackPtr) -> NonNull<F> {
  match stack::GROWTH_DIR {
    GrowthDir::Upwards => unsafe { bot.cast().sub(1) },
    GrowthDir::Downwards => bot.cast(),
  }
}

impl<R> Coro<R> {
  // It is important that the coroutine function object is not address-sensitive,
  // because it needs to be moved to the coroutine's stack, and that is why we
  // require its type `F` to implement `Unpin`. It follows that `Coro` is `Unpin`.
  // Another way to say this: for ergonomic reasons, we would like `Coro` to be
  // `Unpin`, i.e. not address-sensitive.

  /// Creates a new coroutine that is to execute the given [`FnOnce`]
  /// instance on a separate [stack] of size [`DEFAULT_STACK_SIZE`].
  ///
  /// The program stack is taken from a common thread-local pool of
  /// memory; the coroutine's destructor returns it to that same pool.
  ///
  /// [stack]: Stack
  #[cfg(feature = "common_pool")]
  #[must_use = "method creates a coroutine but does not begin its execution"]
  pub fn new<F: FnOnce() -> R + Unpin>(f: F) -> Self {
    // SAFETY: We have exclusive access to the thread-local pool.
    let stack = unsafe { stack::borrow_common_pool(|pool| pool.take(DEFAULT_STACK_SIZE)) };
    Self::with_stack(f, stack)
  }

  /// Creates a new coroutine that is to execute the given [`FnOnce`]
  /// instance on the provided program stack.
  ///
  /// If the `common_pool` feature is enabled, [`Coro::drop`] returns
  /// the stack to the thread-local storage pool when the coroutine is
  /// no longer needed. Otherwise, if feature `alloc` is enabled, you
  /// may use [`drop_and_take_stack`] to reclaim the stack before the
  /// coroutine's destructor deallocates it.
  ///
  /// [`drop_and_take_stack`]: Self::drop_and_take_stack
  #[must_use = "method creates a coroutine but does not begin its execution"]
  pub fn with_stack<F: FnOnce() -> R + Unpin>(f: F, stack: Stack) -> Self {
    #[repr(C)]
    #[cfg_attr(target_arch = "x86_64", repr(align(16)))]
    union Fn<F> {
      normal: ManuallyDrop<F>,
      poison: PoisonTarget,
    }

    assert!(size_of::<Fn<F>>() <= stack.size().get());

    // Make a bitwise copy of `f` on the new stack, in order to pin
    // it to a fixed memory location. The actual stack contents appear
    // on top of this copy.
    // SAFETY: The stack is at least one page long, so it can definitely
    //  fit `f` without overflowing.
    // todo: actually check that this doesn't overflow.
    let mut bot = stack.bottom();
    let fn_ptr = unsafe { stack::push(&mut bot) };
    unsafe {
      // SAFETY: `stack::push` returns a pointer valid for writes.
      fn_ptr.write(Fn {
        normal: ManuallyDrop::new(f),
      })
    };

    unsafe fn drop_fn<F>(bot: StackPtr) {
      // SAFETY: `drop_` is the only caller of this function,
      //  and it promises that `fn_ptr` is valid for dropping.
      unsafe { crate::fn_ptr::<F>(bot).drop_in_place() };
    }

    Self {
      // Arrange things so that the coroutine begins its execution at
      // the first instruction of the trampoline function defined below.
      // Actually, in some architectures the coroutine must first bounce
      // off a smaller trampoline before reaching `crate::trampoline` to
      // e.g. have good backtrace information.
      cont_sp: unsafe { arch::prepare_stack(bot, trampoline::<R, F> as *const _) },
      stack_bot: bot,
      drop_fn: drop_fn::<F>,
      stack: ManuallyDrop::new(stack),
      _phantom: PhantomData,
    }
  }

  #[must_use]
  fn poison_target(&self) -> Option<PoisonTarget> {
    unsafe { arch::poison_target(self.cont_sp) }
  }

  /// Checks whether the coroutine has begun execution.
  #[must_use]
  pub fn has_started(&self) -> bool {
    unsafe { arch::has_started(self.cont_sp) }
  }

  // todo: find better name? `is_suspended`? not good enough, since
  //  this method returns `true` for a coroutine that has not yet
  //  begun execution.
  #[must_use]
  pub fn is_paused(&self) -> bool {
    !self.has_finished() && !self.has_panicked()
  }

  /// Checks if the coroutine has completely finished its execution
  /// without panicking. In that case, calling [`resume`] on the
  /// coroutine will cause a panic.
  #[must_use]
  #[expect(
    unpredictable_function_pointer_comparisons,
    reason = "set and read from same codegen"
  )]
  pub fn has_finished(&self) -> bool {
    self.poison_target() == Some(finished)
  }

  /// Checks if the coroutine has panicked via the [`panic!`] macro
  /// or a similar standard library mechanism. In that case, calling
  /// [`resume`] on the coroutine will cause a panic.
  ///
  /// [`panic!`]: panic
  /// [`resume`]: Self::resume
  #[must_use]
  #[expect(
    unpredictable_function_pointer_comparisons,
    reason = "set and read from same codegen"
  )]
  pub fn has_panicked(&self) -> bool {
    self.poison_target() == Some(panicked)
  }

  /// Resumes execution of this coroutine's program where it last left off.
  ///
  /// This method is equivalent to [`Coroutine::resume`],
  /// but it does not require the coroutine to be pinned.
  ///
  /// # Panics
  ///
  /// This function panics if the coroutine has already finished
  /// its execution (either [normally] or by [panicking]).
  ///
  /// [normally]: Self::has_finished
  /// [panicking]: Self::has_panicked
  pub fn resume(&mut self) -> CoroutineState<(), R> {
    let prev_caller = CALLER.get();
    // SAFETY: The coroutine has exclusive access to its stack at this point.
    let comp = unsafe { arch::resume_coro(&mut self.cont_sp, self.stack_bot, CALLER.as_ptr()) };
    CALLER.set(prev_caller);

    if comp.is_suspended() {
      core::mem::forget(comp); // no panic payload to destroy.
      CoroutineState::Yielded(())
    } else {
      // Read the return address of the coroutine program,
      // or resume unwinding if the coroutine panicked.
      let ret_addr = comp.ret_addr().cast::<R>();
      // The computation finished normally. Copy the return value to
      // the current stack, because the stack of the coroutine might
      // be about to get dropped.
      // SAFETY: todo.
      CoroutineState::Complete(unsafe { ret_addr.read() })
    }
  }

  /// Consumes the (suspended) coroutine, and returns its program stack.
  ///
  /// The stack is filled with undefined information, but it can be
  /// passed to [`Coro::with_stack`] to construct a new coroutine.
  #[cfg(all(feature = "stack_alloc", not(feature = "common_pool")))]
  pub unsafe fn drop_and_take_stack(mut self) -> Stack {
    // SAFETY: We forget `self` immediately after calling `drop_`.
    let stack = unsafe { self.inner_drop() };
    core::mem::forget(self);
    stack
  }
}

impl<R> Coroutine for Coro<R> {
  type Yield = ();
  type Return = R;

  fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
    self.get_mut().resume()
  }
}

// The default ABI of Rust does not define the set of registers a caller
// needs to save before invoking a function. For this reason, the first
// call to `Coro::resume` jumps through the following trampoline function
// with an explicit calling convention.
//
// Note that `arch::resume_coro` expects this function to follow the
// System V ABI on all x86-64 systems, including those running Windows.
#[cfg_attr(target_arch = "x86_64", abi("sysv64-unwind"))]
extern "C-unwind" fn trampoline<R, F>(bot: NonNull<()>) -> !
where
  F: FnOnce() -> R + Unpin,
{
  // SAFETY: `arch::resume_coro` passes the address of the target
  //  function at the bottom of the stack of a coroutine. We have
  //  exclusive access to the stack, so it is sound to convert `fn_ptr`
  //  into a unique reference. To see why, notice that calling `Coro::resume`
  //  is the only way to reach this trampoline function; that function
  //  calls `arch::resume_coro`, which borrows the `cont_sp` field
  //  of the `Coro` mutably and jumps to the trampoline from assembly
  //  code.
  let f = unsafe { fn_ptr(bot).as_mut() };
  // SAFETY: The destructor of `Coro` drops the `F` instance at
  //  `fn_ptr` only if the coroutine is never activated (that is,
  //  if this trampoline function is never invoked). And this
  //  trampoline is called only once, hence there's no risk of
  //  moving out `f` from the coroutine twice.
  let f: F = unsafe { ManuallyDrop::take(f) };

  #[repr(align(2))] // see `unwind::Result`.
  struct RetVal<R>(R);

  let mut ret_val = MaybeUninit::uninit();
  let res = unwind::catch(|| {
    // Execute the actual coroutine's program.
    let ret_val = ret_val.write(RetVal(f()));
    NonNull::from_mut(ret_val).cast::<()>()
  });

  // At this point the coroutine has almost finished its execution.
  // Variable `f` has been destroyed, because an `FnOnce` consumes
  // itself when executed.

  unsafe fn return_control(res: CatchResult<NonNull<()>>, bot: NonNull<()>) -> ! {
    let poison_target = if res.is_ok() { finished } else { panicked };
    let new_sp = unsafe { arch::poison(bot, poison_target) };
    let comp = Completion::from(res);

    // The coroutine caller may have changed since its first resumption.
    let caller = unsafe { NonNull::new_unchecked(CALLER.get().cast_mut()) };
    unsafe { arch::return_control(caller, new_sp, comp) }
  }
  unsafe { return_control(res, bot) };
}

/// Transfers control back to the program that last resumed the active
/// coroutine.
///
/// # Panics
///
/// This function panics if it is called from outside a coroutine.
#[inline(always)]
pub fn suspend() {
  let caller = CALLER.get();
  if caller.is_null() {
    #[cold]
    #[inline(never)]
    fn suspend_from_outside() -> ! {
      panic!("cannot suspend from outside a coroutine")
    }
    suspend_from_outside();
  }
  unsafe { arch::suspend_coro(caller) };
}

#[cfg_attr(target_arch = "x86_64", abi("sysv64-unwind"))]
#[allow(improper_ctypes_definitions, reason = "written and read from Rust")]
extern "C-unwind" fn poison_trampoline(target_loc: NonNull<PoisonTarget>) -> ! {
  (unsafe { target_loc.read() })();
}

/// The target function of a [poisoned coroutine].
///
/// [poisoned coroutine]: Coro::poison_target
type PoisonTarget = fn() -> !;

#[cold]
#[inline(never)]
fn finished() -> ! {
  panic!("coroutine resumed after completion")
}

#[cold]
#[inline(never)]
fn panicked() -> ! {
  panic!("coroutine resumed after panicking")
}

type DropReturn = cfg_select! {
  feature = "common_pool" => (),
  feature = "stack_alloc" => Stack,
  _ => (),
};

impl<R> Coro<R> {
  /// # Safety
  ///
  /// This function is sound to call only once with the same coroutine.
  /// Such coroutine may not be used again, and that includes calling
  /// its destructor. Thus, the only safe way to proceed after calling
  /// `drop_(&mut coro)` is to do `core::mem::forget(coro)`.
  // #[cfg_attr(feature = "common_pool", inline(always))]
  unsafe fn inner_drop(&mut self) -> DropReturn {
    if self.has_started() {
      if self.is_paused() {
        // The coroutine was left in progress; unwind its stack manually.
        let prev_caller = CALLER.get();
        // SAFETY: We just tested that the coroutine is paused.
        let comp =
          unsafe { arch::force_unwind(&mut self.cont_sp, self.stack_bot, CALLER.as_ptr()) };
        CALLER.set(prev_caller);

        if comp.is_ok() {
          // Either (a) the coroutine suspended itself when being
          // dropped; or (b) some frame between the last suspension
          // point and the trampoline caught the exception thrown
          // to unwind the stack, and forgot to clean it up.
          panic!("coroutine did not unwind");
        }
        // SAFETY: todo.
        unsafe { comp.maybe_rethrow() };
      }
    } else {
      // The coroutine was never activated, so `self.func`
      // still holds the coroutine's program. Drop it now.
      cold_path();
      unsafe { (self.drop_fn)(self.stack_bot) };
    }
    // SAFETY: `self.stack` is not used again.
    let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
    cfg_select! {
      feature = "common_pool" => {
        // SAFETY: We have exclusive access to the thread-local pool.
        unsafe { stack::borrow_common_pool(|pool| pool.give(stack)) };
      }
      feature = "stack_alloc" => stack,
      _ => {}
    }
  }
}

impl<R> Drop for Coro<R> {
  // todo: complete with details: (or document these in `drop`)
  /// first unwinds the stack if necessary, destroys the coroutine.
  ///
  /// if unwinding is disabled, then the live variables in the stack
  /// are leaked.
  fn drop(&mut self) {
    // SAFETY: This is the only exception to the rule, since we are
    // already inside `Coro`'s destructor.
    unsafe { self.inner_drop() };
  }
}
