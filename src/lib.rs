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
//! coroutines: [`corosensei`] by A. d'Antras, and [`libfringe`] by edef1c.
//! Preliminary experiments seem to indicate that the three libraries can
//! transfer control between coroutines with comparable efficiency, but
//! `coro` is still inferior with respect to platform support. I mainly
//! put together this library for my own education, so significant
//! improvements are likely possible.

#![feature(asm_unwind)]
#![feature(coroutine_trait)]
#![feature(thread_local)]
#![feature(adt_const_params)]
#![feature(core_intrinsics)]
#![feature(likely_unlikely)]
#![cfg_attr(test, feature(pointer_is_aligned_to))]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod arch;
#[cfg(feature = "std")]
mod os;
pub mod stack;
pub mod unwind;

use crate::stack::{Stack, StackPtr};
use core::cell::Cell;
use core::hint::likely;
use core::marker::PhantomPinned;
use core::mem::ManuallyDrop;
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr;
use core::ptr::NonNull;
use std::println;

// The big picture:
// A coroutine stores a nonnull pointer to a "continuation", which contains
// the stack pointer and instruction pointer of its last suspension point,
// or if it has not started yet, its entry point. When active, the continuation
// stores the stack pointer and instruction pointer of its last caller,
// as well as a `caller` pointer to the continuation of that caller (which
// contains arbitrary data). The continuation sits on top of the function
// being executed by the coroutine (a pointer, or more generally, an object
// implementing `FnOnce() -> R + Unpin`). A `Coro` stores a pointer to this
// continuation, and the metadata of its stack (a base pointer + a size).
// Importantly, a `Coro` is not address-sensitive, because its fields do
// not require pinning; remember that its function is separately pinned
// on its stack.
//
// A static thread-local `ACTIVE` pointer links to the continuation of
// the active coroutine; it is null when the main program is running.
//
// Control transfers:
// The first transfer of control into a coroutine is a bit special,
// because we must go through a trampoline function to properly read
// and call the coroutine's function. The trampoline receives the address
// of the continuation in the coroutine's stack, from which it can
// determine the location of the function pointer. (Remember that the
// continuation sits on top of this pointer.)
//
// When a coroutine calls the `suspend` method, it similarly goes through
// the architecture-specific `transfer_control` function (which takes both
// the continuation of the currently-active coroutine, present in `ACTIVE`,
// and the coroutine to which we are supposed to transfer control to, namely
// `ACTIVE->caller`). The current stack pointer and instruction pointer
// are saved on the coroutine's continuation, and the `ACTIVE` value is
// replaced by `ACTIVE->caller`.

/// The control record of a [coroutine], used to save or restore the
/// context needed by the coroutine during a [transfer of control].
///
/// If the coroutine has not completely finished its execution, this
/// object is essentially a one-shot continuation of its computation
/// (when suspended) or that of its last caller (when active). See
/// C. Bruggeman, O. Waddell, and R. Kent Dybvig, _SIGPLAN Notices_
/// **31**, 5 (May 1996), 99--107.
///
/// [coroutine]: Coro
/// [transfer of control]: arch::transfer_control
#[cfg_attr(target_arch = "x86_64", repr(C, align(16)))]
#[derive(Debug)]
pub(crate) struct Control {
  /// The value of the stack pointer just before a transfer of control
  /// involving the coroutine associated with this control record.
  ///
  /// Special cases:
  /// - if the coroutine has not been activated yet, a pointer to the
  ///   bottom of its stack;
  /// - if the coroutine has finished normally, the address of its
  ///   return value on its own stack;
  /// - if the coroutine has panicked, a dangling pointer not derived
  ///   from the coroutine's stack allocation.
  stack_ptr: StackPtr,
  /// The value of the instruction pointer just before a transfer of
  /// control involving the coroutine associated with this control
  /// record.
  ///
  /// Special cases:
  /// - if the coroutine has not been activated yet, a pointer to the
  ///   first instruction of a [trampoline function] that immediately
  ///   bounces to the coroutine's program;
  /// - if the coroutine has finished normally, a [null pointer];
  /// - if the coroutine has panicked, an undefined nonzero address.
  ///
  /// [trampoline function]: trampoline
  /// [null pointer]: ptr::null
  inst_ptr: *const (),
  /// If the coroutine is active, the address of the control record
  /// of the coroutine that last resumed it (with `ptr::null_mut()`
  /// representing the main program). Otherwise undefined.
  caller: *mut Control,
  _pin: PhantomPinned,
}

impl Control {
  pub const SIZE: usize = size_of::<Self>();

  /*pub fn set_parent(self: Pin<&mut Self>, addr: *mut Control) {
    // SAFETY:
    let mut pinned_parent = unsafe { self.map_unchecked_mut(|this| &mut this.parent) };
    *pinned_parent = addr;
  }*/
  /*fn has_finished(&self) -> bool {
    self.inst_ptr.is_null()
  }*/
}

/// The control record of the currently active coroutine, if any.
#[thread_local]
static ACTIVE: Cell<*mut Control> = Cell::new(ptr::null_mut());

#[repr(C)]
struct Bottom<F> {
  ctrl: Control,
  func: ManuallyDrop<F>,
}

// It is important that the coroutine function object is not address-sensitive,
// because it needs to be moved to the coroutine's stack, and that is why we
// require its type `F` to implement `Unpin`. It follows that `Coro` is `Unpin`.
// Another way to say this: for ergonomic reasons, we would like `Coro` to be
// `Unpin`, i.e. not address-sensitive.

/// A first-class stackful asymmetric coroutine.
///
/// The [crate-level documentation] discusses this type in more detail.
///
/// [crate-level documentation]: crate
pub struct Coro<R, F: FnOnce() -> R + Unpin> {
  /// This record is stored at the bottom of the coroutine's stack.
  bottom: NonNull<Bottom<F>>,
  /// The program stack of the coroutine.
  #[cfg(feature = "std")]
  stack: ManuallyDrop<Stack>,
}

impl<R, F: FnOnce() -> R + Unpin> Coro<R, F> {
  pub fn new(f: F) -> Self {
    let stack = Stack::allocate(1 << 21);
    Self::with_stack(f, stack)
  }

  pub fn with_stack(f: F, stack: Stack) -> Self {
    let mut stack = ManuallyDrop::new(stack);
    let sp = stack.bottom();
    let (bottom, _) = unsafe {
      stack.push(sp, |loc| Bottom {
        ctrl: Control {
          stack_ptr: arch::initial_sp(loc),
          inst_ptr: arch::trampoline::<R, F> as *const _,
          caller: ptr::null_mut(),
          _pin: PhantomPinned,
        },
        func: ManuallyDrop::new(f),
      })
    };
    /*let (f_ptr, sp) = unsafe { stack.push(sp, |_| ManuallyDrop::new(f)) };
    let (control, _) = unsafe {
      stack.push(sp, |loc| Control {
        stack_ptr: arch::initial_sp(loc),
        inst_ptr: arch::trampoline::<R, F> as *const _,
        parent: ptr::null_mut(),
        _pin: PhantomPinned,
      })
    };
    println!("f_ptr={f_ptr:?}, control={control:?}");*/
    Self { bottom, stack }
  }

  /// Returns an exclusive reference to the control record of this coroutine.
  ///
  /// # Safety
  ///
  ///
  #[inline]
  fn control(&self) -> &Control {
    // SAFETY:
    &unsafe { self.bottom.as_ref() }.ctrl
  }

  /// Returns a mutable reference to the control record of this coroutine.
  ///
  /// # Safety
  ///
  // todo: make unsafe, because caller promises not to move.
  #[inline]
  fn control_mut(&mut self) -> &mut Control {
    // SAFETY:
    &mut unsafe { self.bottom.as_mut() }.ctrl
  }

  #[inline]
  fn func(&mut self) -> &mut ManuallyDrop<F> {
    &mut unsafe { self.bottom.as_mut() }.func
    /*let mut f_ptr = next_slot(self.stack.bottom());
    // SAFETY:
    unsafe { f_ptr.as_mut() }*/
  }

  /// Checks whether the coroutine has begun execution.
  #[must_use]
  #[inline]
  pub fn has_started(&self) -> bool {
    self.control().inst_ptr != arch::trampoline::<R, F> as *const _
  }

  // todo: find better name?
  #[must_use]
  pub fn is_paused(&self) -> bool {
    let ctrl = self.control();
    !ctrl.inst_ptr.is_null() && self.stack.contains(ctrl.stack_ptr)
  }

  /// Checks if the coroutine has completely finished its execution
  /// without panicking.
  #[must_use]
  #[inline]
  pub fn has_finished(&self) -> bool {
    self.control().inst_ptr.is_null()
  }

  /// Checks if the coroutine has panicked via the [`panic!`] macro
  /// or a similar standard library mechanism. In that case, calling
  /// [`resume`] on the coroutine will cause a panic.
  ///
  /// [`panic!`]: panic
  /// [`resume`]: Self::resume
  #[must_use]
  pub fn has_panicked(&self) -> bool {
    !self.stack.contains(self.control().stack_ptr)
  }

  /// Resumes execution of this coroutine's program where it last left off.
  ///
  /// This method is essentially equivalent to [`Coroutine::resume`],
  /// but it does not require the coroutine to be pinned.
  ///
  /// # Panics
  ///
  /// This function panics if the coroutine has already finished its
  /// execution (either [normally] or by [panicking]).
  ///
  /// [normally]: Self::has_finished
  /// [panicking]: Self::has_panicked
  pub fn resume(&mut self) -> CoroutineState<(), R> {
    // Ensure that the coroutine's computation is only partially complete
    // before we attempt to re-activate it.
    assert!(!self.has_finished(), "coroutine resumed after completion");
    assert!(!self.has_panicked(), "coroutine resumed after panicking");
    // Push ourselves to the stack of active coroutines.
    let ctrl = self.control_mut();
    ctrl.caller = ACTIVE.replace(ctrl);
    println!(
      "resume: transferring control to {:?} (sp={:?}, ip={:?})",
      ptr::from_mut(ctrl),
      ctrl.stack_ptr,
      ctrl.inst_ptr
    );
    // SAFETY: The coroutine has exclusive access to its stack at this point.
    unsafe { arch::transfer_control(ctrl) };
    // The coroutine's computation has transferred control back.
    if ctrl.inst_ptr.is_null() {
      // The computation finished. Copy the return value to the current stack,
      // because the stack of the coroutine might be about to get dropped.
      let ret_val_addr = ctrl.stack_ptr.cast();
      // SAFETY: The `trampoline` function links the `stack_ptr` control field
      //         to the return value of the coroutine, which was declared to be
      //         of type `ManuallyDrop<R>` and was _not_ manually dropped. Thus
      //         it is valid to read the value from `ret_val_addr` once, even
      //         after the coroutine terminates.
      let ret_val = unsafe { ret_val_addr.read() };
      CoroutineState::Complete(ret_val)
    } else {
      CoroutineState::Yielded(())
    }
  }

  // todo: add cfg for non common stack pool.
  unsafe fn take_stack(mut self) -> Stack {
    unsafe { ManuallyDrop::take(&mut self.stack) }
  }
}

impl<R, F: FnOnce() -> R + Unpin> Coroutine for Coro<R, F> {
  type Yield = ();
  type Return = R;

  fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
    self.get_mut().resume()
  }
}

// The default ABI of Rust does not define the set of registers a caller
// needs to save before invoking a function. For this reason, the first
// call to `Coro::resume` jumps through the following trampoline function
// with an explicit calling convention. But how is the trampoline to read
// its destination `f`? There are (at least) three ways to proceed:
// a) The coroutine pushes the address of `f` to its stack when created,
//    and the trampoline reads it from there.

// Note that `arch::transfer_control` expects this trampoline function to
// follow the System V ABI on all x86-64 systems, including those running
// Windows.
#[cfg_attr(all(windows, target_arch = "x86_64"), abi("sysv64-unwind"))]
extern "C-unwind" fn trampoline<R, F>(mut bottom: NonNull<Bottom<F>>) -> !
where
  F: FnOnce() -> R + Unpin,
{
  /*let control_bot = match GROWTH_DIR {
    GrowthDir::Upwards => control_ptr,
    GrowthDir::Downwards => unsafe { control_ptr.add(1) },
  };
  let mut f_ptr = next_slot::<_, { GROWTH_DIR.rev() }>(control_bot.cast());
  println!("reading f_ptr={f_ptr:?}");
  let f: &mut ManuallyDrop<F> = unsafe { f_ptr.as_mut() };*/

  // SAFETY: `arch::transfer_control` passes the address of the control
  //         record at the bottom of the stack of a coroutine, in the
  //         register for the `bottom` argument, according to the current ABI.
  let f = &mut unsafe { bottom.as_mut() }.func;
  // SAFETY: The destructor of `Coro` drops the `func` field only if the
  //         coroutine is never activated (that is, if this trampoline
  //         function is never invoked). And this trampoline is called
  //         only once, so there's no risk to move out `func` from the
  //         coroutine twice.
  let f = unsafe { ManuallyDrop::take(f) };
  // Execute the actual coroutine's program.
  let ret_val = ManuallyDrop::new(f());
  //
  let control = unsafe { &mut *ACTIVE.get() };
  ACTIVE.set(control.caller);
  unsafe { arch::return_control(control, NonNull::from(&ret_val)) }
}

/// Transfers control back to the program that last resumed the active
/// coroutine.
///
/// # Panics
///
/// This function panics if it is called from outside a coroutine.
pub fn suspend() {
  match unsafe { ACTIVE.get().as_mut() } {
    Some(control) => {
      println!(
        "suspend: transferring control back, reading from {:?} (sp={:?}, ip={:?})",
        ptr::from_mut(control),
        control.stack_ptr,
        control.inst_ptr
      );
      // Pop the current coroutine from the stack.
      ACTIVE.set(control.caller);
      unsafe { arch::transfer_control(control) };
    }
    None => panic!("cannot suspend from outside a coroutine"),
  }
}

impl<R, F: FnOnce() -> R + Unpin> Drop for Coro<R, F> {
  fn drop(&mut self) {
    if likely(self.has_started()) {
      // We need to unwind the coroutine's stack manually if it was left in progress.
      if self.is_paused() {
        let ctrl = self.control_mut();
        println!("coroutine has not panicked, unwinding... control={ctrl:?}");
        unsafe { arch::force_unwind(ctrl) };
      }
    } else {
      // The coroutine was never activated, so `self.func`
      // still holds the coroutine's program. Drop it now.
      unsafe { ManuallyDrop::drop(self.func()) };
    }
    #[cfg(feature = "std")]
    stack::COMMON_POOL.with_borrow_mut(|pool| {
      // SAFETY: `self.stack` is not used again.
      let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
      pool.give(stack);
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::println;

  #[test]
  fn start_suspend_resume() {
    let mut coro = Coro::new(|| {
      println!("hello world");
      suspend();
      println!("hello again");
    });
    coro.resume();
    coro.resume();
  }

  #[test]
  fn resume_via_trait_method() {
    let mut coro = Coro::new(|| {
      suspend();
      0xfefe
    });
    assert_eq!(Pin::new(&mut coro).resume(()), CoroutineState::Yielded(()));
    assert_eq!(
      Pin::new(&mut coro).resume(()),
      CoroutineState::Complete(0xfefe)
    );
  }

  #[test]
  fn triply_nested_coroutines() {
    let mut inner = Coro::new(|| {
      suspend();
      "inner"
    });
    let mut middle = Coro::new(|| {
      assert_eq!(inner.resume(), CoroutineState::Yielded(()));
      assert!(inner.has_started());
      assert!(!inner.has_finished());
      suspend();
      assert!(inner.has_started());
      assert!(!inner.has_finished());
      assert_eq!(inner.resume(), CoroutineState::Complete("inner"));
      assert!(inner.has_started());
      assert!(inner.has_finished());
      suspend();
      "middle"
    });
    let mut outer = Coro::new(|| {
      assert_eq!(middle.resume(), CoroutineState::Yielded(()));
      assert!(middle.has_started());
      assert!(!middle.has_finished());
      assert_eq!(middle.resume(), CoroutineState::Yielded(()));
      assert!(middle.has_started());
      assert!(!middle.has_finished());
      assert_eq!(middle.resume(), CoroutineState::Complete("middle"));
      assert!(middle.has_started());
      assert!(middle.has_finished());
      "outer"
    });
    assert_eq!(outer.resume(), CoroutineState::Complete("outer"));
    assert!(outer.has_started());
    assert!(outer.has_finished());
  }

  #[test]
  fn many_control_transfers() {
    const COUNT: u32 = 100;
    let mut coro = Coro::new(|| {
      for _ in 0..COUNT {
        suspend();
      }
    });
    for _ in 0..COUNT {
      assert_eq!(coro.resume(), CoroutineState::Yielded(()));
    }
    assert_eq!(coro.resume(), CoroutineState::Complete(()));
    assert!(coro.has_finished());
  }

  #[test]
  fn panic() {
    let mut coro = Coro::new(|| panic!("payload"));
    let res = std::panic::catch_unwind(move || coro.resume());
    assert_eq!(
      res.expect_err("coro must panic").downcast_ref(),
      Some(&"payload")
    );
    // todo: check `has_finished` et al.
  }

  #[test]
  fn inner_panic() {
    let mut inner = Coro::new(|| {
      suspend();
      panic!("inner");
    });
    let mut outer = Coro::new(move || {
      inner.resume();
      inner.resume(); // panics.
    });
    let res = std::panic::catch_unwind(move || outer.resume());
    assert_eq!(
      res.expect_err("inner coro must panic").downcast_ref(),
      Some(&"inner")
    );
    // todo: check `has_finished` et al.
  }

  #[test]
  fn panic_after_many_suspends_from_recursive_fn() {
    const COUNTDOWN: u32 = 1;
    let mut coro = Coro::new(|| {
      fn tick(i: u32) {
        if i == 0 {
          panic!("boom");
        }
        suspend();
        tick(i - 1);
      }
      tick(COUNTDOWN);
    });
    for _ in 0..COUNTDOWN {
      coro.resume();
    }
    let res = std::panic::catch_unwind(move || coro.resume());
    assert_eq!(
      res.expect_err("coro must panic").downcast_ref(),
      Some(&"boom")
    );
  }

  pub fn other_place<R, F: FnOnce() -> R + Unpin>(coro: &mut Coro<R, F>) {
    coro.resume();
  }

  // todo: remove or clean up this test:
  #[test]
  fn troubleshoot() {
    let mut coro = Coro::new(|| {
      println!("started");
      suspend();
      println!("resumed");
      suspend();
    });
    coro.resume();
    println!("back in main, resuming");
    other_place(&mut coro);
    coro.resume(); // todo: test unwind by removing.
  }
}
