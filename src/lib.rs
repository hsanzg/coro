#![feature(asm_unwind)]
#![feature(coroutine_trait)]
#![feature(thread_local)]
#![feature(adt_const_params)]
#![feature(core_intrinsics)]
#![feature(box_as_ptr)]
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

use crate::stack::{GROWTH_DIR, GrowthDir, Stack, StackPtr, next_slot};
use core::cell::Cell;
use core::mem::ManuallyDrop;
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr;
use std::alloc::Layout;
use std::arch::asm;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::{MaybeUninit, offset_of};
use std::println;
use std::ptr::NonNull;

// The big picture:
// A coroutine stores a nonnull pointer to a "continuation", which contains
// the stack pointer and instruction pointer of its last suspension point,
// or if it has not started yet, its entry point. When active, the continuation
// stores the stack pointer and instruction pointer of its last caller,
// as well as a `caller` pointer to the continuation of that caller (which
// contains arbitrary data). Importantly, all continuations of a coroutine
// are stored in its own stack. The bottommost continuation is special,
// because it sits on top of the function being executed by the coroutine
// (a pointer, or more generally, an object implementing `FnOnce() -> R + Unpin`).
// A `Coro` stores a pointer to this bottommost continuation, and the
// metadata of its stack (a base pointer + a size). Importantly, a `Coro`
// is not address-sensitive, because its fields do not require pinning;
// remember that its function is separately pinned on its stack.
//
// todo: Should we make each coroutine have exactly one continuation
//  sitting at the bottom of its stack? I think this is compatible with
//  DWARF unwinding. If so, update this big comment.
//
// A static thread-local `ACTIVE` pointer links to the topmost continuation
// of the active coroutine; it is null when the main program is running.
//
// Control transfers:
// The first transfer of control into a coroutine is a bit special,
// because we must go through a trampoline function to properly read
// and call the coroutine's function. The trampoline receives the address
// of the first continuation in the coroutine's stack, from which it can
// determine the location of the function pointer. (Remember that the
// bottommost continuation sits on top of this pointer.)
//
// When a coroutine calls the `suspend` method, it similarly goes through
// the architecture-specific `transfer_control` function (which takes both
// the continuation of the currently-active coroutine, present in `ACTIVE`,
// and the coroutine to which we are supposed to transfer control to, namely
// `ACTIVE->caller`). The current stack pointer and instruction pointer
// todo: is this true? Would it suffice to pass a single continuation?
// are saved on the coroutine's continuation, and the `ACTIVE` value is
// replaced by `ACTIVE->caller`.
//
// It follows that a coroutine has at most two continuations in its
// stack, one of them sitting at its bottom.

// todo: mention that when the coroutine is inactive, this object acts as
//  a continuation. When it is active, it acts as a continuation of its
//  last caller.
/// The control record of a [coroutine], used to save or restore the
/// context needed by the coroutine during a [transfer of control].
///
/// [coroutine]: Coro
/// [transfer of control]: arch::transfer_control
// todo: document that alignment needed to satisfy System V x86-64 ABI
//  on first call to the architecture-specific trampoline.
// todo: rename to `FrameBase` or `FirstFrame` or `Activation`?
#[repr(C, align(16))]
pub(crate) struct Control {
  stack_ptr: StackPtr,
  inst_ptr: *const (),
  parent: *mut Control,
  _pin: PhantomPinned,
}

impl Control {
  pub const SIZE: usize = size_of::<Self>();

  /*pub fn set_parent(self: Pin<&mut Self>, addr: *mut Control) {
    // SAFETY:
    let mut pinned_parent = unsafe { self.map_unchecked_mut(|this| &mut this.parent) };
    *pinned_parent = addr;
  }*/
  fn has_finished(&self) -> bool {
    self.inst_ptr.is_null()
  }
}

#[repr(C)]
struct Bottom<F> {
  ctrl: Control,
  func: ManuallyDrop<F>,
}

#[thread_local]
static ACTIVE: Cell<*mut Control> = Cell::new(ptr::null_mut());

pub struct Coro<R, F: FnOnce() -> R + Unpin> {
  bottom: NonNull<Bottom<F>>,
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
          parent: ptr::null_mut(),
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

  fn control(&self) -> &Control {
    // SAFETY:
    &unsafe { self.bottom.as_ref() }.ctrl
  }

  fn control_mut(&mut self) -> &mut Control {
    // SAFETY:
    &mut unsafe { self.bottom.as_mut() }.ctrl
    // unsafe { Pin::new_unchecked(self.control.as_mut()) }
  }

  fn func(&mut self) -> &mut ManuallyDrop<F> {
    &mut unsafe { self.bottom.as_mut() }.func
    /*let mut f_ptr = next_slot(self.stack.bottom());
    // SAFETY:
    unsafe { f_ptr.as_mut() }*/
  }

  #[must_use]
  pub fn has_started(&self) -> bool {
    self.control().inst_ptr != arch::trampoline::<R, F> as *const _
  }

  #[must_use]
  pub fn has_finished(&self) -> bool {
    self.control().has_finished()
  }

  pub fn resume(&mut self) -> CoroutineState<(), R> {
    assert!(!self.has_finished(), "coroutine resumed after completion");
    let control = self.control_mut();
    // let control_ptr = addr_of_pin_mut(&control);
    // control.as_mut().set_parent(ACTIVE.replace(control_ptr));
    control.parent = ACTIVE.replace(control);
    println!(
      "resume: transferring control to {:?} (sp={:?}, ip={:?})",
      ptr::from_mut(control),
      control.stack_ptr,
      control.inst_ptr
    );
    unsafe { arch::transfer_control(control) };
    if control.has_finished() {
      let ret_val_addr = control.stack_ptr.cast();
      let ret_val = unsafe { ret_val_addr.read() };
      CoroutineState::Complete(ret_val)
    } else {
      CoroutineState::Yielded(())
    }
  }
}

impl<R, F: FnOnce() -> R + Unpin> Coroutine for Coro<R, F> {
  type Yield = ();
  type Return = R;

  fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
    self.get_mut().resume()
  }
}

extern "sysv64-unwind" fn trampoline<R, F>(mut bottom: NonNull<Bottom<F>>) -> !
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
  let f = &mut unsafe { bottom.as_mut() }.func;
  let f = unsafe { ManuallyDrop::take(f) };
  let ret_val = ManuallyDrop::new(f());
  let control = unsafe { &mut *ACTIVE.get() };
  ACTIVE.set(control.parent);
  unsafe { arch::return_control(control, NonNull::from(&ret_val)) }
}

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
      ACTIVE.set(control.parent);
      unsafe { arch::transfer_control(control) };
    }
    None => panic!("cannot suspend from outside a coroutine"),
  }
}

impl<R, F: FnOnce() -> R + Unpin> Drop for Coro<R, F> {
  fn drop(&mut self) {
    if self.has_started() {
      let Self { stack, bottom } = self;
      let control = &mut unsafe { bottom.as_mut() }.ctrl;
      if !control.has_finished() {
        let has_panicked = !stack.contains(control.stack_ptr);
        if !has_panicked {
          println!("coroutine has not panicked, unwinding... control={bottom:?}");
          unsafe { arch::force_unwind(control) };
        }
      }
    } else {
      // The coroutine was never activated, so `self.func`
      // still holds the coroutine's program. Drop it now.
      unsafe { ManuallyDrop::drop(self.func()) };
    }
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
