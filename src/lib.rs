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
//!
//! [`resume`]: Coro::resume
//! [`corosensei`]: https://github.com/Amanieu/corosensei
//! [`libfringe`]: https://github.com/edef1c/libfringe

#![feature(asm_unwind)]
#![feature(core_intrinsics)]
#![feature(coroutine_trait)]
#![feature(rustc_attrs)]
#![feature(thread_local)]
#![feature(fn_static)]
#![feature(ptr_alignment_type)]
#![expect(
  internal_features,
  reason = "we unwind coroutine stacks by throwing a non-Rust exception type, and need `catch_unwind` to catch it"
)]
#![cfg_attr(all(test, feature = "stack_alloc"), feature(pointer_is_aligned_to))]
#![no_std]
#[cfg(feature = "stack_alloc")]
extern crate alloc;

mod arch;
pub mod stack;
mod unwind;

use crate::stack::{Stack, StackPtr};
use core::alloc::Layout;
use core::cell::Cell;
use core::hint::cold_path;
use core::marker::{PhantomData, PhantomPinned};
use core::mem::ManuallyDrop;
use core::ops::{Code, Coroutine, CoroutineState, FnPtr};
use core::panic::AssertUnwindSafe;
use core::pin::Pin;
use core::ptr::NonNull;
use core::{mem, ptr};
#[allow(unused_imports)] // false positive depending on platform.
use coro_macros::abi;

macro_rules! fn_ptr {
  ($f:expr) => {
    NonNull::new_unchecked($f as *const () as *mut Code)
  };
}

pub(crate) use fn_ptr;

struct Continuation {}

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
#[repr(C)]
#[derive(Debug)]
struct Control {
  // todo: document ABI requirement: `stack_ptr` must be first field.
  /// The value of the stack pointer just before a transfer of control
  /// involving the coroutine associated with this control record.
  ///
  /// Special cases:
  /// - if the coroutine has not been activated yet, a pointer to the
  ///   bottom of its stack;
  /// - if the coroutine has finished normally, the address of its // todo: undefined pointer to its own stack, or zero? think carefully.
  ///   return value on its own stack;
  /// - if the coroutine has panicked, a dangling pointer not derived
  ///   from the coroutine's stack allocation. // todo: no longer depend on this; cleanup fn sets inst_ptr instead.
  stack_ptr: StackPtr,
  /*#[cfg(target_arch = "x86_64")]
  _pad: *const (), // todo: take advantage of `caller` field being undefined when inactive, and place the initial resumption address there (that is, point `stack_ptr` to it).*/
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
  // inst_ptr: *const (),
  /// If the coroutine is active, the address of the control record
  /// of the coroutine that last resumed it (with `ptr::null_mut()`
  /// representing the main program). Otherwise undefined.
  caller: *mut Control,
  ret_val: *const (),
  // poisoned: bool,
  _pin: PhantomPinned, // todo: remove.
}

impl Control {
  #[inline(always)]
  fn inst_ptr(&self) -> *const Code {
    unsafe { self.stack_ptr.cast().read() }
  }
}

/// The control record of the currently active coroutine, if any.
#[thread_local]
static ACTIVE: Cell<*mut Control> = Cell::new(ptr::null_mut());

// Triggered on panics and manual unwinds of the coroutine stack.
struct ActivateOnDrop(*mut Control);
impl Drop for ActivateOnDrop {
  fn drop(&mut self) {
    extern crate std;
    std::println!(
      "setting ACTIVE back to {:?}, was {:?}",
      self.0,
      ACTIVE.get()
    );
    ACTIVE.set(self.0);
  }
}

// todo: this is incorrect; the bottom taken by the `FnOnce` instance.
/*/// The [`resume`] and [`suspend`] operations employ the bottommost
/// [`Control::SIZE`] bytes of a coroutine's stack to record status
/// information during transfers of control. (Here the meaning of
/// "bottommost" depends on the [stack growth convention] on the
/// current platform.) The actual stack contents appear on top of
/// the control record.

/// Control structures and additional information placed at the
/// bottom of a coroutine's program stack.
///
/// The actual stack contents appear on top of this structure.
#[deprecated]
#[repr(C)]
#[cfg_attr(target_arch = "x86_64", repr(C, align(16)))]
struct Bottom<R, F: FnOnce() -> R> {
  // todo: document ABI requirement: `ctrl` must be first field.
  /// A control record used to store status information during
  /// transfers of control that involve this coroutine.
  ctrl: Control,
  func: ManuallyDrop<F>,
}*/

// It is important that the coroutine function object is not address-sensitive,
// because it needs to be moved to the coroutine's stack, and that is why we
// require its type `F` to implement `Unpin`. It follows that `Coro` is `Unpin`.
// Another way to say this: for ergonomic reasons, we would like `Coro` to be
// `Unpin`, i.e. not address-sensitive.

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
/// # Safety
///
// todo: talk about unwinding mechanism, and that, at the time a paused
//  non-finished coroutine is to be dropped, no frames between the
//  last suspension point and the trampoline may catch the foreign
//  exception used to unwind the stack. Failure to do so will result
//  in undefined behavior. Other uses of `catch_unwind` are allowed.
// todo: the previous todo is partly incorrect, because we now handle
//       intermediate catches correctly ourselves by panicking.
///
/// [crate-level documentation]: crate
pub struct Coro<R> {
  ctrl: NonNull<Control>,
  // bottom: NonNull<Bottom<R, F>>,
  // todo: finish comment.
  /// The program stack of the coroutine.
  stack: ManuallyDrop<Stack>,
  fn_ptr: AssertUnwindSafe<NonNull<Code>>,
  drop_fn: unsafe fn(NonNull<Code>),
  _phantom: PhantomData<*mut R>,
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
  /// no longer needed. Otherwise, if feature `alloc` is enabled, you
  /// may use [`drop_and_take_stack`] to reclaim the stack before the
  /// coroutine's destructor deallocates it.
  ///
  /// [`drop_and_take_stack`]: Self::drop_and_take_stack
  #[must_use = "method creates a coroutine but does not begin its execution"]
  pub fn with_stack<F: FnOnce() -> R + Unpin>(f: F, stack: Stack) -> Self {
    let mut stack = ManuallyDrop::new(stack);
    let (f_ptr, ctrl, sp) = unsafe {
      arch::prepare_stack(
        &mut stack,
        Layout::for_value(&f),
        fn_ptr!(trampoline::<R, F>),
      )
    };
    unsafe { f_ptr.cast().write(ManuallyDrop::new(f)) };
    unsafe {
      ctrl.write(Control {
        stack_ptr: sp,
        //_pad: ptr::null_mut(),
        caller: ptr::null_mut(),
        ret_val: ptr::null_mut(),
        _pin: PhantomPinned,
      })
    };

    /*let bottom = unsafe {
      stack.push(&mut sp, |loc| Bottom {
        ctrl: Control {
          stack_ptr: arch::initial_sp(loc),
          #[cfg(target_arch = "x86_64")]
          _pad: ptr::null(),
          // inst_ptr: arch::first_tramp::<R, F> as *const _,
          // inst_ptr: trampoline::<R, F> as *const _,
          ret_val: ptr::null_mut(),
          caller: ptr::null_mut(),
          // poisoned: false,
          _pin: PhantomPinned,
        },
        // ret_val: ptr::null_mut(),
        // todo: finish comment.
        // Make a bitwise copy of the `FnOnce` instance on the new stack, in
        // order to pin it to a fixed memory location.
        // that its location in memory remains stable across context switches.
        //

        // order to pin it in a stable place across control transfers.
        // (Leaving it in the `Coro` is not an option, because the coroutine is
        // `Unpin` and thus free to be moved around in memory.)
        func: ManuallyDrop::new(f),
      })
    };
    arch::prepare_stack_old(bottom);
    // todo: move this function outside this constructor?*/
    unsafe fn drop_fn<F>(ptr: NonNull<Code>) {
      unsafe { ptr.cast::<F>().drop_in_place() };
    }

    Self {
      // ctrl: unsafe { bottom.byte_add(mem::offset_of!(Bottom<R, F>, ctrl)) }.cast(),
      ctrl,
      stack,
      fn_ptr: AssertUnwindSafe(unsafe { NonNull::new_unchecked(f_ptr.as_ptr() as *mut _) }),
      drop_fn: drop_fn::<F>,
      _phantom: PhantomData,
    }
  }

  /// Returns an exclusive reference to the control record of this coroutine.
  #[inline]
  fn control(&self) -> &Control {
    unsafe { self.ctrl.as_ref() }
  }

  /// Returns a mutable reference to the control record of this coroutine.
  ///
  /// # Safety
  ///
  // todo: make unsafe, because caller promises not to move.
  #[inline]
  fn control_mut(&mut self) -> &mut Control {
    unsafe { self.ctrl.as_mut() }
  }

  /*/// Returns a mutable reference to the [`FnOnce`] instance,
  /// placed in its program stack.
  #[inline]
  fn func(&mut self) -> &mut ManuallyDrop<F> {
    &mut unsafe { self.bottom.as_mut() }.func
  }*/

  /*unsafe fn bottom_mut<F: FnOnce() -> R + Unpin>(&mut self) -> &mut Bottom<R, F> {
    unsafe {
      self
        .ctrl
        .byte_sub(mem::offset_of!(Bottom<R, F>, ctrl))
        .cast()
        .as_mut()
    }
  }*/

  /// Checks whether the coroutine has begun execution.
  #[must_use]
  #[inline]
  pub fn has_started(&self) -> bool {
    !ptr::eq(self.control().inst_ptr(), arch::first_inst_addr().as_ptr())
    // self.has_panicked() || self.control().inst_ptr() != arch::first_inst_addr::<R, F>()
  }

  // todo: find better name? `is_suspended`? not good enough, since
  //  this method returns `true` for a coroutine that has not yet
  //  begun execution.
  #[must_use]
  pub fn is_paused(&self) -> bool {
    let ctrl = self.control();
    ctrl.ret_val.is_null() && self.stack.contains(ctrl.stack_ptr)
  }

  /// Checks if the coroutine has completely finished its execution
  /// without panicking.
  #[must_use]
  #[inline]
  pub fn has_finished(&self) -> bool {
    arch::poison_dst(self.control())
      .is_some_and(|d| ptr::eq(d.as_ptr(), unsafe { fn_ptr!(finished) }.as_ptr()))
    // !self.control().ret_val.is_null()
    // self.control().inst_ptr().is_null()
  }

  /// Checks if the coroutine has panicked via the [`panic!`] macro
  /// or a similar standard library mechanism. In that case, calling
  /// [`resume`] on the coroutine will cause a panic.
  ///
  /// [`panic!`]: panic
  /// [`resume`]: Self::resume
  #[must_use]
  pub fn has_panicked(&self) -> bool {
    extern crate std;
    std::println!(
      "checking whether {:?} (with sp={:?}) has panicked; active is {:?}",
      self.ctrl,
      self.control().stack_ptr,
      ACTIVE.get()
    );
    arch::poison_dst(self.control())
      .is_some_and(|d| ptr::eq(d.as_ptr(), unsafe { fn_ptr!(panicked) }.as_ptr()))
    // self.control().inst_ptr() == panicked as *const ()
    // !self.stack.contains(self.control().stack_ptr)
  }

  // todo: remove.
  /*pub fn is_poisoned(&self) -> bool {
    self.control().poisoned
  }*/

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
    //assert!(!self.has_finished(), "coroutine resumed after completion");
    //assert!(!self.has_panicked(), "coroutine resumed after panicking");
    // Push ourselves to the stack of active coroutines.
    let ctrl = self.control_mut();
    //ctrl.caller = ACTIVE.replace(ctrl);
    extern crate std;
    std::println!(
      "resume: set active to {:?}, with caller {:?}",
      ptr::from_mut(ctrl),
      ctrl.caller
    );
    let act_parent = ActivateOnDrop(ACTIVE.replace(ctrl));
    // SAFETY: The coroutine has exclusive access to its stack at this point.
    unsafe { arch::resume_coro(ctrl) };
    drop(act_parent);
    // The coroutine's computation has transferred control back.
    if ctrl.ret_val.is_null() {
      CoroutineState::Yielded(())
    } else {
      // The computation finished. Copy the return value to the current stack,
      // because the stack of the coroutine might be about to get dropped.
      // let ret_val_addr = NonNull::new(ctrl.ret_val).unwrap().cast();
      // SAFETY: The `trampoline` function links the `stack_ptr` control field
      //         to the return value of the coroutine, which was declared to be
      //         of type `ManuallyDrop<R>` and was _not_ manually dropped. Thus
      //         it is valid to read the value from `ret_val_addr` once, even
      //         after the coroutine terminates.
      let ret_val = unsafe { ctrl.ret_val.cast::<R>().read() };
      unsafe { arch::poison(self.control_mut(), finished) };
      CoroutineState::Complete(ret_val)
    }
  }

  /// Consumes the (suspended) coroutine, and returns its program stack.
  ///
  /// The stack is filled with undefined information, but it can be
  /// passed to [`Coro::with_stack`] to construct a new coroutine.
  ///
  // todo: complete with details: (or document these in `drop`)
  /// first unwinds the stack if necessary, destroys the coroutine,
  /// and then returns the stack.
  #[cfg(all(feature = "stack_alloc", not(feature = "common_pool")))]
  pub unsafe fn drop_and_take_stack(mut self) -> Stack {
    // SAFETY: We forget `self` immediately after calling `drop_`.
    let stack = unsafe { drop_(&mut self) };
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
// with an explicit calling convention. But how is the trampoline to read
// its destination `f`? There are (at least) three ways to proceed:
// a) The coroutine pushes the address of `f` to its stack when created,
//    and the trampoline reads it from there.

// Note that `arch::transfer_control` expects this trampoline function to
// follow the System V ABI on all x86-64 systems, including those running
// Windows.
#[cfg_attr(target_arch = "x86_64", abi("sysv64-unwind"))]
extern "C-unwind" fn trampoline<R, F>(
  ctrl: &mut Control,
  f: &mut ManuallyDrop<F>, /*mut bottom: NonNull<Bottom<R, F>>*/
) -> !
where
  F: FnOnce() -> R + Unpin,
{
  debug_assert_eq!(ACTIVE.get(), ptr::from_mut(ctrl));
  // SAFETY: `arch::transfer_control` passes the address of the control
  //  record at the bottom of the stack of a coroutine, in the register
  //  for the `bottom` argument, according to the current ABI. We have
  //  exclusive access to the `Bottom` object, so it is sound to convert
  //  `bottom` into a unique reference. To see why, notice that calling
  //  `Coro::resume` is the only way to reach this trampoline function:
  //  that function calls `arch::transfer_control`, which borrows the
  //  control recrd mutably and jumps to the trampoline from assembly
  //  code.
  //let f = &mut unsafe { bottom.as_mut() }.func;
  // SAFETY: The destructor of `Coro` drops the `func` field only if the
  //  coroutine is never activated (that is, if this trampoline function
  //  is never invoked). And this trampoline is called only once, hence
  //  there's no risk to move out `func` from the coroutine twice.
  let f = unsafe { ManuallyDrop::take(f) };
  // Execute the actual coroutine's program.
  let ret_val = ManuallyDrop::new(f());

  // let ctrl = &mut unsafe { bottom.as_mut() }.ctrl;
  debug_assert_eq!(ACTIVE.get(), ptr::from_mut(ctrl));
  // ACTIVE.set(ctrl.caller);
  ctrl.ret_val = (&raw const ret_val).cast();
  unsafe { arch::return_control(ctrl) }
}

/// Transfers control back to the program that last resumed the active
/// coroutine.
///
/// # Panics
///
/// This function panics if it is called from outside a coroutine.
// todo: note that for safety reasons, it is not possible to call `suspend`
//  from a coroutine whose stack is being unwound as a result of a panic or
//  a drop. Attempting to do so will abort the process. This protects against
//  cases of calling `suspend` from a `drop` method, or after stopping the
//  unwinding process using `catch_unwind`. Failure to do so could result
//  in memory unsafety caused by re-entering the active coroutine.
#[inline(always)]
pub fn suspend() {
  if let Some(control) = unsafe { ACTIVE.get().as_mut() } {
    /*if control.poisoned {
      cold_path();
      suspend_during_unwind();
    }*/
    extern crate std;
    std::println!(
      "suspend: transferring control back, reading from {:?} (sp={:?}, ip={:?})",
      ptr::from_mut(control),
      control.stack_ptr,
      control.inst_ptr()
    );
    // Pop the current coroutine from the stack.
    // ACTIVE.set(control.caller);
    unsafe { arch::suspend_coro(control) };
  } else {
    cold_path();
    suspend_from_outside();
  };
}

/*#[inline(never)]
fn suspend_during_unwind() -> ! {
  panic!("cannot suspend unwinding coroutine");
}*/

#[cold]
#[inline(never)]
extern "C-unwind" fn finished() -> ! {
  panic!("coroutine resumed after completion")
}

#[cold]
#[inline(never)]
extern "C-unwind" fn panicked() -> ! {
  panic!("coroutine resumed after panicking")
}

#[cold]
#[inline(never)]
fn suspend_from_outside() -> ! {
  panic!("cannot suspend from outside a coroutine")
}

type DropReturn = cfg_select! {
  feature = "common_pool" => (),
  feature = "stack_alloc" => Stack,
  _ => (),
};

///
/// # Safety
///
/// This function is sound to call only once with the same coroutine.
/// Such coroutine may not be used again, and that includes calling
/// its `drop` method. Thus the only safe way to proceed after calling
/// `drop_(&mut coro)` is to do `core::mem::forget(coro)`.
#[cfg_attr(feature = "common_pool", inline(always))]
unsafe fn drop_<R>(coro: &mut Coro<R>) -> DropReturn {
  if coro.has_started() {
    // We need to unwind the coroutine's stack manually if it was left in progress.
    if coro.is_paused() {
      let ctrl = coro.control_mut();
      /*extern crate std;
      std::println!(
        "coroutine has not panicked, unwinding... control @ {:?}={ctrl:?}",
        ptr::from_mut(ctrl)
      );*/
      let _active = ActivateOnDrop(ACTIVE.replace(ctrl));
      unsafe { arch::force_unwind2(ctrl) };
      // ACTIVE.set(ctrl.caller);
    }
  } else {
    cold_path();
    // The coroutine was never activated, so `self.func`
    // still holds the coroutine's program. Drop it now.
    unsafe { (coro.drop_fn)(coro.fn_ptr.0) };
    // unsafe { ManuallyDrop::drop(coro.func()) };
  }
  // SAFETY: `self.stack` is not used again.
  let stack = unsafe { ManuallyDrop::take(&mut coro.stack) };
  cfg_select! {
    feature = "common_pool" => {
      // SAFETY: We have exclusive access to the thread-local pool.
      unsafe { stack::borrow_common_pool(|pool| pool.give(stack)) };
    }
    feature = "stack_alloc" => stack,
    _ => {}
  }
}

impl<R> Drop for Coro<R> {
  fn drop(&mut self) {
    // SAFETY: This is the only exception to the rule, since we are
    // already inside `Coro`'s destructor.
    unsafe { drop_(self) };
  }
}
