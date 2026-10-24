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
//! ## Example
//!
//! As a typical example of the use of coroutines, let us consider
//! the problem of determining whether two ordered trees are similar.
//! Formally speaking, a _tree_ is a set of nodes that consists of a
//! _root_ together with $m\ge0$ disjoint trees. If the relative order
//! of the $m$ subtrees is important, we say that the tree is _ordered_.
//! Let $T$ and $T'$ be ordered trees with subtrees $T_1,\dots,T_m$
//! and $T_1',\dots,T_{m'}'$, respectively. Then $T$ and $T'$ are
//! said to be _similar_ if $m=m'$ and the subtrees $T_i$ and $T'_i$
//! are similar for all $1\le i\le m$.
//!
//! ```
//! #![feature(coroutine_trait)]
//!
//! # #[cfg(feature = "common_pool")] {
//! use core::cell::Cell;
//! use core::ops::CoroutineState;
//! use coro::{suspend, Coro};
//!
//! struct Node(Vec<Node>);
//!
//! fn similar(t: &Node, tp: &Node) -> bool {
//!   let (m, mp) = (Cell::new(0), Cell::new(0));
//!
//!   fn visit(root: &Node, m: &Cell<usize>) {
//!     m.set(root.0.len());
//!     suspend();
//!     for child in &root.0 {
//!       visit(child, m);
//!     }
//!   }
//!
//!   let mut t_coro = Coro::new(|| visit(t, &m));
//!   let mut tp_coro = Coro::new(|| visit(tp, &mp));
//!
//!   loop {
//!     match (t_coro.resume(), tp_coro.resume()) {
//!       (CoroutineState::Complete(_), CoroutineState::Complete(_)) => return true,
//!       (CoroutineState::Yielded(_), CoroutineState::Yielded(_)) if m == mp => {},
//!       _ => return false,
//!     }
//!   }
//! }
//! # }
//! ```
//!
//! [`resume`]: Coro::resume
//! [`corosensei`]: https://github.com/Amanieu/corosensei
//! [`libfringe`]: https://github.com/edef1c/libfringe

#![feature(asm_unwind)]
#![feature(coroutine_trait)]
#![feature(core_intrinsics)] // `catch_unwind`.
#![feature(panic_internals)] // `panic_nounwind`.
#![feature(rustc_attrs)] // `rustc_nounwind`.
#![feature(thread_local)]
#![feature(macro_attr)] // `arch::trampoline`
#![cfg_attr(panic = "unwind", feature(panic_unwind))]
#![expect(
  internal_features,
  reason = "we unwind coroutine stacks by throwing a non-Rust exception type, and need `core::intrinsics::catch_unwind` to catch it"
)]
#![cfg_attr(all(test, feature = "stack_alloc"), feature(pointer_is_aligned_to))]
#![no_std]

#[cfg(panic = "unwind")]
extern crate unwind as uw;

#[cfg(feature = "stack_alloc")]
extern crate alloc;

mod arch;
pub mod stack;
mod unwind;

use crate::stack::{Stack, StackPtr};
use crate::unwind::{CatchResult, Completion};
use core::cell::Cell;
use core::hint::cold_path;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr;
use core::ptr::NonNull;

type Caller = *const ();

/// The stack pointer of the program that last resumed the currently
/// active coroutine, or null if no coroutine is active.
#[thread_local]
static CALLER: Cell<Caller> = Cell::new(ptr::null());

/// Returns whether this subroutine is being called from within
/// a [coroutine](Coro) or the main program.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "common_pool")] {
/// use coro::{in_coro, Coro};
///
/// assert!(!in_coro());
///
/// let mut coro = Coro::new(|| assert!(in_coro()));
/// coro.resume();
/// # }
/// ```
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
  /// The bottom location of the deepest stack frame of the coroutine's
  /// program. For reasons given below, this frame appears immediately
  /// above [a copy] of the coroutine's function object. It follows
  /// that `stack_bot != stack.bottom()` in general.
  ///
  /// [a copy]: CoroFn
  stack_bot: StackPtr,
  /// A subroutine that, given `stack_bot`, destroys the function
  /// object of the coroutine. The subroutine is safe to call only
  /// if the coroutine has not and will never begin execution.
  ///
  /// The subroutine itself is always [`CoroFn::raw_drop`], which
  /// is generic over the function type `F`. Storing a pointer to it
  /// allows [`Coro`] to not depend on `F`, thus avoiding needless
  /// monomorphization. The disadvantage is that the size of each
  /// coroutine increases by that of a pointer variable.
  drop_fn: unsafe fn(StackPtr),
  /// The program stack of the coroutine.
  stack: ManuallyDrop<Stack>,
  /// Make [`Coro`] covariant over its return type.
  _phantom: PhantomData<fn() -> R>,
}

/// A record at the bottom of a coroutine stack containing a copy
/// of its function object or, if the coroutine has been poisoned,
/// the address of its [new target].
///
/// For ergonomic reasons, [`Coro`] should ideally be [`Unpin`] so
/// that it can be moved freely in memory. A coroutine program needs
/// its function object to remain valid at the same memory location
/// during its lifetime, so it cannot be stored in [`Coro`] itself.
/// We instead copy the object to the bottom of the coroutine's stack,
/// where its address is fixed. For this move to be sound, `F` must
/// be [`Unpin`].
///
/// [new target]: Coro::poison_target
// Insist that all fields start at offset 0 from the beginning of the union.
#[repr(C)]
// Make sure that the stack frame on top of this record is
// 16-byte-aligned on x86-64, as the System V ABI dictates.
#[cfg_attr(target_arch = "x86_64", repr(align(16)))]
union CoroFn<F: Unpin> {
  /// The coroutine program; valid until it begins execution.
  normal: ManuallyDrop<F>,
  /// The new program of the coroutine after it becomes [poisoned].
  ///
  /// # Safety
  ///
  /// We want [`poison_trampoline`] to be capable of determining
  /// this target from the coroutine's `stack_bot` pointer alone,
  /// without knowledge of `F`. For that reason, this field holds
  /// the target only when stacks grow [downward]. Otherwise
  /// [`arch::poison`] places the target address just before the
  /// end of the union, so it must instead be accessed through a
  /// raw pointer. (This is safe, because a union may contain
  /// arbitrary data.) In the latter case, this field merely
  /// ensures that the union is sufficiently large and aligned
  /// to hold a [`PoisonTarget`] instance.
  ///
  /// [poisoned]: Coro::poison_target
  /// [downward]: stack::GrowthDir::Downwards
  poison: PoisonTarget,
}

impl<F: Unpin> CoroFn<F> {
  /// Returns a pointer to the function object of a [coroutine]
  /// whose `stack_bot` field equals `bot`.
  ///
  /// [coroutine]: Coro
  unsafe fn addr(bot: StackPtr) -> NonNull<Self> {
    // SAFETY: The function object occupies all locations
    //  within the stack allocation immediately below `bot`,
    //  or so the caller says.
    unsafe { stack::adjacent(bot) }
  }

  /// Destroys the function object of a [coroutine] whose `stack_bot`
  /// field equals `bot`.
  ///
  /// # Safety
  ///
  /// This operation is safe to apply only to coroutines that have
  /// never [started] execution and never will.
  ///
  /// [coroutine]: Coro
  /// [started]: Coro::has_started
  unsafe fn raw_drop(bot: StackPtr) {
    // SAFETY: The coroutine wasn't started, so the record at
    //  the bottom of its stack still contains its program.
    let f = unsafe { &mut Self::addr(bot).as_mut().normal };
    // SAFETY: The caller promises not to access `f` again.
    unsafe { ManuallyDrop::drop(f) };
  }
}

impl<R> Coro<R> {
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
  /// no longer needed. Otherwise, if feature `stack_alloc` is enabled,
  /// you may use [`drop_and_take_stack`] to reclaim the stack before
  /// the coroutine's destructor deallocates it.
  ///
  /// [`drop_and_take_stack`]: Self::drop_and_take_stack
  #[must_use = "method creates a coroutine but does not begin its execution"]
  pub fn with_stack<F: FnOnce() -> R + Unpin>(f: F, stack: Stack) -> Self {
    // Make a bitwise copy of `f` on the new stack, in order to pin
    // it to a fixed memory location. The actual stack contents will
    // appear on top of this copy. See `CoroFn` for details.
    let mut bot = stack.bottom();
    // SAFETY: `bot` is derived from the stack allocation.
    let f_ptr = unsafe { stack.push(&mut bot) };
    // SAFETY: `Stack::push` returns a pointer valid for writes.
    unsafe {
      f_ptr.write(CoroFn {
        normal: ManuallyDrop::new(f),
      })
    };

    Self {
      // Arrange things so that the coroutine begins its execution at
      // the first instruction of the trampoline function defined below.
      // Actually, in some architectures the coroutine must first bounce
      // off a smaller trampoline before reaching `crate::trampoline` in
      // order to support stack unwinding. See the `arch` module for
      // details.
      cont_sp: unsafe { arch::prepare_stack(bot, trampoline::<R, F>) },
      stack_bot: bot,
      drop_fn: CoroFn::<F>::raw_drop,
      stack: ManuallyDrop::new(stack),
      _phantom: PhantomData,
    }
  }

  /// Checks whether the coroutine has begun execution.
  ///
  /// # Example
  ///
  /// ```
  /// # #[cfg(feature = "common_pool")] {
  /// use coro::{suspend, Coro};
  ///
  /// let mut coro = Coro::new(|| suspend());
  /// assert!(!coro.has_started());
  ///
  /// coro.resume(); // suspended.
  /// assert!(coro.has_started());
  ///
  /// coro.resume(); // finished.
  /// assert!(coro.has_started());
  /// # }
  /// ```
  #[must_use]
  pub fn has_started(&self) -> bool {
    // SAFETY: `cont_sp` is the current stack pointer of this coroutine.
    unsafe { arch::has_started(self.cont_sp) }
  }

  /// If this coroutine has been poisoned, returns the function to
  /// which [`Coro::resume`] will jump. Otherwise returns [`None`].
  ///
  /// When a coroutine can no longer run normally (e.g. because its
  /// program finished normally or due to a panic), [`arch::poison`]
  /// rearranges its stack so that resuming the coroutine calls a
  /// target function that never returns normally. This saves us
  /// from having to check for invalid states on every call to
  /// [`Coro::resume`].
  #[must_use]
  fn poison_target(&self) -> Option<PoisonTarget> {
    // SAFETY: `cont_sp` is the current stack pointer of this coroutine.
    unsafe { arch::poison_target(self.cont_sp) }
  }

  /// Checks if the coroutine has completely finished its execution
  /// without panicking.
  ///
  /// # Example
  ///
  /// ```
  /// # #[cfg(feature = "common_pool")] {
  /// use coro::Coro;
  ///
  /// let mut coro = Coro::new(|| {});
  /// assert!(!coro.has_finished());
  ///
  /// coro.resume();
  /// assert!(coro.has_finished());
  /// # }
  /// ```
  #[must_use]
  #[expect(
    unpredictable_function_pointer_comparisons,
    reason = "set and read from the same codegen"
  )]
  pub fn has_finished(&self) -> bool {
    self.poison_target() == Some(finished)
  }

  /// Checks if the coroutine has panicked via the [`panic!`] macro
  /// or a similar standard library mechanism.
  ///
  /// # Example
  ///
  /// ```
  /// # #[cfg(all(feature = "common_pool", panic = "unwind"))] {
  /// # extern crate std;
  /// use std::panic::AssertUnwindSafe;
  /// use coro::Coro;
  ///
  /// let mut coro = Coro::new(|| panic!());
  /// assert!(!coro.has_panicked());
  ///
  /// let mut coro_ref = AssertUnwindSafe(&mut coro);
  /// std::panic::catch_unwind(move || {
  ///   coro_ref.resume();
  /// });
  /// assert!(coro.has_panicked());
  /// # }
  /// ```
  ///
  /// [`panic!`]: panic
  #[must_use]
  #[expect(
    unpredictable_function_pointer_comparisons,
    reason = "set and read from the same codegen"
  )]
  pub fn has_panicked(&self) -> bool {
    self.poison_target() == Some(panicked)
  }

  /// Returns whether this coroutine can be resumed by calling
  /// [`Coro::resume`]; that is, whether its program has either
  /// not yet [begun execution] or is suspended at one of its
  /// [suspension points].
  ///
  /// # Example
  ///
  /// ```
  /// # #[cfg(feature = "common_pool")] {
  /// use coro::{suspend, Coro};
  ///
  /// let mut coro = Coro::new(|| suspend());
  /// assert!(coro.is_resumable());
  ///
  /// coro.resume();
  /// assert!(coro.is_resumable());
  ///
  /// coro.resume();
  /// assert!(!coro.is_resumable());
  /// # }
  /// ```
  ///
  /// [begun execution]: Coro::has_started
  /// [suspension points]: suspend
  #[must_use]
  pub fn is_resumable(&self) -> bool {
    !self.has_finished() && !self.has_panicked()
  }

  // Control transfers.

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
    // SAFETY: The coroutine has exclusive access to its stack.
    let comp = unsafe { arch::resume_coro(&mut self.cont_sp, self.stack_bot, CALLER.as_ptr()) };
    // The coroutine's computation has transferred control back.
    CALLER.set(prev_caller);
    if comp.is_suspended() {
      core::mem::forget(comp); // there's no panic payload to destroy.
      CoroutineState::Yielded(())
    } else {
      // Find out where the coroutine program stored its return value,
      // or resume unwinding if it panicked.
      let ret_addr = comp.ret_addr().cast::<R>();
      // The computation finished normally. Move the return value to
      // the current stack, because the stack of the coroutine might
      // be about to get dropped.
      // SAFETY: The trampoline function leaves the return value at
      //  `ret_addr` as a `ManuallyDrop<R>` object that has not been
      //  dropped. Thus, reading once from `ret_addr` is valid.
      CoroutineState::Complete(unsafe { ret_addr.read() })
    }
  }
}

impl<R> Coroutine for Coro<R> {
  type Yield = ();
  type Return = R;

  fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
    self.get_mut().resume()
  }
}

/// The routine that takes control on the first [resumption] of a
/// coroutine, and that immediately invokes the coroutine's program.
///
/// The coroutine program is a function of type `F` that follows the
/// default ABI of Rust, which does not define the set of registers
/// that one needs to save before calling it. That's why the first
/// call to [`Coro::resume`] must go through this trampoline with
/// an explicit calling convention (set by [`arch::trampoline`]).
///
/// [resumption]: Coro::resume
#[arch::trampoline]
unsafe fn trampoline<R, F>(bot: NonNull<()>) -> !
where
  F: FnOnce() -> R + Unpin,
{
  // Take ownership of the function object, which `Coro::with_stack`
  // placed at the bottom of the stack. The original object must not
  // be accessed again; `Coro::inner_drop` is careful not to drop it
  // after the coroutine has begun execution.
  // SAFETY: `bot` is the `stack_bot` value of the active coroutine.
  let f_ptr = unsafe { CoroFn::<F>::addr(bot) }.as_ptr();
  // Read the `normal` field of `*f_ptr` rather than the whole union,
  // to avoid some copying when `PoisonTarget` is bigger than `F`.
  // SAFETY: This is a field projection on a valid pointer.
  let f_ptr = unsafe { &raw const (*f_ptr).normal };
  // SAFETY: This trampoline is the only subroutine that reads the
  //  contents of `f_ptr`, and it runs at most once per coroutine.
  let f = ManuallyDrop::into_inner(unsafe { f_ptr.read() });

  /// The return value of the coroutine, to be read by [`Coro::resume`].
  #[repr(align(2))] // required by `Completion::from`.
  struct RetVal<R>(R);

  // Since the present subroutine cannot return normally, its frame
  // is never popped off the stack. So we can conveniently store the
  // return value in a local variable. This gives the last program
  // that activated the coroutine a chance to move the return value
  // from there into its own stack before the coroutine is activated
  // again (at which point the poison target may overwrite its frame).
  let mut ret_val = MaybeUninit::uninit();
  let res = unwind::catch(|| {
    // Execute the actual coroutine's program.
    NonNull::from_mut(ret_val.write(RetVal(f()))).cast()
  });

  fn return_control(res: CatchResult<NonNull<()>>, bot: NonNull<()>) -> ! {
    // Force any future attempt to resume this coroutine to trigger
    // an immediate panic instead of returning back here. The panic
    // message depends on how the coroutine finished execution.
    let poison_target = if res.is_ok() { finished } else { panicked };
    // SAFETY: The function object is no longer alive, so we may store
    //  the address of the poison target in the bottom `CoroFn` record.
    let new_sp = unsafe { arch::poison(bot, poison_target) };
    // SAFETY: The return address, if any, is wyde-aligned.
    let comp = unsafe { Completion::from(res) };

    // Before transferring control to us, `arch::resume_coro` saved
    // the stack pointer of its caller in `CALLER`. Load and pass
    // that value to `arch::return_control`, so that it can switch
    // back to the caller's stack.
    // SAFETY: This coroutine is active, so its caller is nonnull.
    let caller = unsafe { NonNull::new_unchecked(CALLER.get().cast_mut()) };
    // SAFETY: We have poisoned the coroutine already.
    unsafe { arch::return_control(caller, new_sp, comp) }
  }
  return_control(res, bot)
}

/// Transfers control back to the program that last resumed the active
/// coroutine.
///
/// # Panics
///
/// This function panics if it is called from outside a coroutine.
#[doc(alias = "yield")]
#[inline(always)]
pub fn suspend() {
  let caller = CALLER.get();
  if caller.is_null() {
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn suspend_from_outside() -> ! {
      panic!("cannot suspend from outside a coroutine")
    }
    suspend_from_outside();
  }
  // SAFETY: `caller` is the stack pointer a coroutine or the main
  //  program had before transferring control to the active coroutine;
  //  see `arch::resume_coro`. That caller is currently suspended, so
  //  we are allowed to pass control back to it.
  unsafe { arch::suspend_coro(caller) };
}

// Poisoning.

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

/// The routine that takes control whenever a [poisoned] coroutine
/// is [resumed].
///
/// [poisoned]: Coro::poison_target
/// [resumed]: Coro::resume
#[arch::trampoline]
unsafe fn poison_trampoline(bot: NonNull<()>) -> ! {
  // SAFETY: `arch::poison` stores the address of the new target at
  //  whichever end of the coroutine's `CoroFn` lies next to `bot`,
  //  namely the beginning when stacks grow downward, and the end
  //  when they grow upward.
  let target_ptr = unsafe { stack::adjacent::<PoisonTarget>(bot) };
  // SAFETY: The pointer is valid for reads, because
  //  1. it is aligned: the size of `CoroFn<F>` is a multiple
  //     of the alignment of `PoisonTarget`; and
  //  2. we have exclusive access to the coroutine.
  (unsafe { target_ptr.read() })();
}

// Destruction.

/// The value returned by [`Coro::inner_drop`].
// todo: Move inside the following `impl` block once Rust allows
//  declaring associated types in inherent impls.
type DropReturn = cfg_select! {
  // The stack is returned to the common pool.
  feature = "common_pool" => (),
  // The caller must dispose of the stack.
  feature = "stack_alloc" => Stack,
  // The stack is deallocated immediately.
  _ => (),
};

impl<R> Coro<R> {
  /// Destroys this coroutine and releases its stack.
  ///
  /// # Safety
  ///
  /// This method is sound to call at most once on a given coroutine.
  /// The coroutine must not be used in any way afterward, and that
  /// includes dropping it. Thus, the only way to proceed after
  /// `coro.inner_drop()` is to pass `coro` to [`core::mem::forget`].
  unsafe fn inner_drop(&mut self) -> DropReturn {
    if !self.has_started() {
      // The coroutine never started, so its program still lies intact
      // at the bottom of its stack (see `CoroFn`). Drop it now.
      cold_path();
      // SAFETY: The coroutine has never run, and the caller promises
      //  that it never will.
      unsafe { (self.drop_fn)(self.stack_bot) };
    } else if self.is_resumable() {
      // The coroutine was left in progress.
      cfg_select! {
        // Unwind its stack manually.
        panic = "unwind" => {
          let prev_caller = CALLER.get();
          // SAFETY: We just tested that the coroutine is paused.
          let comp =
            unsafe { arch::force_unwind(&mut self.cont_sp, self.stack_bot, CALLER.as_ptr()) };
          CALLER.set(prev_caller);

          if comp.is_ok() {
            // Unwinding didn't reach the trampoline. This can happen
            // if the coroutine suspends itself while being dropped, or
            // if some frame catches the unwinding exception and forgets
            // to clean it up. (Any attempt to call the cleanup function
            // of the exception aborts the process.)
            #[cold]
            #[inline(never)]
            #[track_caller]
            fn did_not_unwind() -> ! {
              panic!("coroutine did not unwind")
            }
            did_not_unwind();
          }
          // SAFETY: The completion was not successful, as just checked.
          unsafe { comp.maybe_rethrow() };
        }
        // We have no way to run the destructors of the objects on
        // the stack of the coroutine, so we leak them. Since some
        // of those objects may be pinned, their memory cannot be
        // reused; thus we must also leak the stack itself.
        _ => return,
      }
    }
    // SAFETY: `self.stack` contains no live objects, and the caller
    //  promises that the coroutine will not use it again.
    let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
    cfg_select! {
      feature = "common_pool" => {
        // SAFETY: We have exclusive access to the thread-local pool.
        unsafe { stack::borrow_common_pool(|pool| pool.give(stack)) };
      }
      feature = "stack_alloc" => stack,
      _ => drop(stack),
    }
  }

  /// Consumes the (suspended) coroutine, and returns its program stack.
  ///
  /// The stack is filled with undefined information, but it can be
  /// passed to [`Coro::with_stack`] to construct a new coroutine.
  ///
  /// # Example
  ///
  /// ```
  /// use coro::{suspend, Coro};
  /// use coro::stack::Stack;
  ///
  /// let stack = Stack::allocate(1 << 13); // 8 kilobytes.
  /// let mut coro = Coro::with_stack(|| suspend(), stack);
  /// coro.resume();
  ///
  /// // Unwind the stack of the coroutine, and reuse it as the stack
  /// // of a new coroutine.
  /// let stack = coro.drop_and_take_stack();
  /// let mut coro = Coro::with_stack(|| {}, stack);
  /// coro.resume();
  /// ```
  #[cfg(all(
    feature = "stack_alloc",
    not(feature = "common_pool"),
    panic = "unwind"
  ))]
  #[must_use = "if you don't need the stack, drop the coroutine normally"]
  pub fn drop_and_take_stack(mut self) -> Stack {
    // SAFETY: We forget `self` immediately after calling `inner_drop`.
    let stack = unsafe { self.inner_drop() };
    core::mem::forget(self);
    stack
  }
}

impl<R> Drop for Coro<R> {
  /// Destroys the coroutine and [disposes] of its stack.
  ///
  /// If the coroutine has [never started], its program is dropped.
  /// Otherwise, if it has [not completely finished] execution and
  /// the [panic strategy] `coro` was compiled with is
  /// - `unwind`, its stack is unwound as though the coroutine were
  ///   to panic at its current [suspension point];
  /// - not `unwind`, the stack and its contents are leaked.
  ///
  /// [disposes]: Self::with_stack
  /// [never started]: Self::has_started
  /// [not completely finished]: Self::is_resumable
  /// [panic strategy]: https://doc.rust-lang.org/reference/panic.html#panic-strategy
  /// [suspension point]: suspend
  fn drop(&mut self) {
    // SAFETY: This method runs at most once, and the coroutine
    //  cannot be used after it runs.
    unsafe { self.inner_drop() };
  }
}
