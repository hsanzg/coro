// The following doc comment is kept in sync with the README.md file. Please
// run the `cargo sync-readme` command after modifying the comment contents.
//! This crate provides an implementation of stackful, first-class asymmetric
//! coroutines in the Rust language.
//!
//! The general notion of _coroutines_, first discussed in the published
//! literature by M. E. Conway \[_CACM_ **6** (1963), 396–408], extends the
//! concept of subroutines by allowing them to share and pass data and control
//! back and forth. A coroutine suspends execution of its program by invoking
//! the [`yield`] function, which returns control to the caller. Invoking the
//! [`resume`] method on a coroutine resumes execution of its program
//! immediately after the point where it was last suspended.
//!
//! The Rust language has built-in support for coroutines, but one cannot easily
//! suspend their execution from within a nested function call. One way to solve
//! this issue is to maintain a separate program stack for each coroutine. This
//! approach takes more memory space, but one can employ pooling techniques to
//! diminish the need for large allocations and liberations.
//!
//! The crux of this crate is the [`Coro`] type, which implements the built-in
//! [`core::ops::Coroutine`] trait. The most noteworthy feature of this library
//! is that it allows a coroutine to return control without the need to pass
//! a "yielder" object around. (It's easy to achieve this by carefully aligning
//! the program stack of a coroutine, so that we can deduce the location of a
//! control record from the current stack pointer value alone.) This approach
//! has the following disadvantage: the compiler can no longer prove that
//! [`yield`] is always called from a coroutine. Nevertheless, the function
//! will trigger a panic in the rare event that this invariant does not hold.
//! It may be of interest to note that this faulty-call detection mechanism can
//! slow down the action, because it needs to update a thread-local variable
//! during each transfer of control. Experienced programmers can disable the
//! `safe_yield` feature to skip the check, but we should mention that this
//! can lead to undefined behavior in purely safe code. Another potential
//! disadvantage of our implementation is that the program stacks of all
//! coroutines must have the [same size].
//!
//! See the paper "Revisiting Coroutines" by A. L. de Moura and R. Ierusalimschy
//! \[_ACM TOPLAS_ **31** (2009), 1–31] for the definitions of "stackful",
//! "first-class" and "asymmetric" used above.
//!
//! This library incorporates ideas from two other implementations of stackful
//! coroutines: [`corosensei`] by A. d'Antras, and [`libfringe`] by edef1c.
//! Preliminary experiments seem to indicate that the three approaches can
//! transfer control between coroutines with comparable efficiency, but `coro`
//! is inferior with respect to platform support and stack unwinding. I put
//! together this library for my own education, so significant improvements
//! are possible.
//!
//! # Example
//!
//! As a [typical example] where coroutines are useful, we will write a program
//! that tests the similarity of two ordered trees. Formally speaking, a _tree_
//! is a finite set of nodes that consists of a _root_ together with $m\ge0$
//! disjoint trees. If the relative order of the $m$ subtrees is important, we
//! say that the tree is an _ordered tree_. Let the subtrees of ordered trees
//! $T$ and $T\'$ be respectively $T_1,\dots,T_m$ and $T\'\_1,\dots,T\'_{m\'}$.
//! Then $T$ and $T\'$ are said to be _similar_ if $m=m\'$ and the subtrees
//! $T_i$ and $T\'_i$ are similar for all $1\le i\le m$.
//!
//! ```
//! # #![feature(coroutine_trait)]
//! use std::cell::Cell;
//! use std::ops::{Coroutine, CoroutineState};
//! use std::pin::Pin;
//! use coro::{Coro, yield_};
//! # #[cfg(not(feature = "std"))]
//! # use coro::stack::StackPool;
//!
//! #[derive(Clone)]
//! struct Node {
//!     children: Vec<Node>,
//! }
//!
//! fn visit(root: &Node, m: &Cell<usize>) {
//!     m.set(root.children.len());
//!     yield_();
//!     for ch in &root.children {
//!         visit(ch, m);
//!     }
//! }
//!
//! fn similar(first: &Node, second: &Node) -> bool {
//!     let m = Cell::new(0);
//!     let m_prime = Cell::new(0);
//!     # #[cfg(feature = "std")]
//!     let mut first_coro = Coro::new(|| visit(first, &m));
//!     # #[cfg(feature = "std")]
//!     let mut second_coro = Coro::new(|| visit(second, &m_prime));
//!     # // Alternate implementation for use in `no_std` environments.
//!     # #[cfg(not(feature = "std"))]
//!     let mut pool = StackPool::new();
//!     # #[cfg(not(feature = "std"))]
//!     # let mut first_coro = Coro::with_stack_from(&mut pool, || visit(first, &m));
//!     # #[cfg(not(feature = "std"))]
//!     # let mut second_coro = Coro::with_stack_from(&mut pool, || visit(second, &m_prime));
//!     loop {
//!         match (
//!             Pin::new(&mut first_coro).resume(()),
//!             Pin::new(&mut second_coro).resume(()),
//!         ) {
//!             (CoroutineState::Complete(_), CoroutineState::Complete(_)) => return true,
//!             (CoroutineState::Yielded(_), CoroutineState::Yielded(_)) if m == m_prime => {}
//!             _ => return false,
//!         };
//!     }
//! }
//! # let first_tree = Node {
//! #     children: vec![
//! #         Node {
//! #             children: vec![Node { children: vec![] }],
//! #         },
//! #         Node {
//! #             children: vec![
//! #                 Node { children: vec![] },
//! #                 Node {
//! #                     children: vec![Node { children: vec![] }],
//! #                 },
//! #                 Node { children: vec![] },
//! #             ],
//! #         },
//! #     ],
//! # };
//! # let second_tree = first_tree.clone();
//! # assert!(similar(&first_tree, &second_tree));
//! #
//! # let second_tree = Node {
//! #     children: vec![
//! #         Node {
//! #             children: vec![Node { children: vec![] }],
//! #         },
//! #         Node {
//! #             children: vec![
//! #                 Node { children: vec![] },
//! #                 Node {
//! #                     children: vec![Node { children: vec![] }],
//! #                 },
//! #                 Node { children: vec![] },
//! #                 Node { children: vec![] },
//! #             ],
//! #         },
//! #     ],
//! # };
//! # assert!(!similar(&first_tree, &second_tree));
//! ```
//!
//! [`yield`]: yield_
//! [`resume`]: Coro::resume
//! [same size]: STACK_SIZE
//! [`libfringe`]: https://github.com/edef1c/libfringe
//! [`corosensei`]: https://github.com/Amanieu/corosensei
//! [typical example]: https://research.swtch.com/coro

#![feature(ptr_alignment_type)]
#![feature(coroutine_trait)]
#![feature(ptr_as_uninit)]
#![feature(ptr_mask)]
#![feature(ptr_sub_ptr)]
#![feature(strict_provenance)]
#![feature(nonzero_ops)]
#![feature(pointer_is_aligned_to)]
#![feature(naked_functions)]
#![feature(asm_const)]
#![cfg_attr(not(feature = "std"), no_std)]

mod arch;
pub(crate) mod os;
pub mod stack;

use crate::arch::STACK_FRAME_ALIGN;
#[cfg(feature = "std")]
use crate::stack::COMMON_POOL;
use crate::stack::{align_alloc, Stack, StackOrientation, StackPool};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::num::NonZeroUsize;
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr::{Alignment, NonNull};

/// The control record of a [coroutine], used to store and restore the context
/// needed by the coroutine during a [transfer of control].
///
/// [coroutine]: Coro
/// [transfer of control]: arch::transfer_control
struct Control {
    /// If the coroutine has not been activated yet, the address of a
    /// [trampoline function] that immediately bounces to its associated
    /// coroutine. If the coroutine is active, the location of the first
    /// instruction to execute when it reaches a suspension point or terminates.
    /// Else (namely if the coroutine has completely finished its execution)
    /// a [null pointer].
    ///
    /// This field is of use when [resuming] a coroutine that has not [finished]
    /// its execution, because it does so at the point where it last left off.
    ///
    /// [trampoline function]: Coro::new
    /// [null pointer]: core::ptr::null
    /// [resuming]: Coro::resume
    /// [finished]: Coro::is_finished
    instr_ptr: *const u8,
    /// Analogous to [`instr_ptr`], the value of the stack pointer register
    /// just before a transfer of control involving the coroutine associated
    /// with this control record. Just after the coroutine has finished
    /// successfully, this field links to its return value.
    ///
    /// [`instr_ptr`]: Self::instr_ptr
    stack_ptr: NonNull<u8>,
}

impl Control {
    /// The fixed number of cells at the bottom of a [program stack] reserved
    /// for special use by the [`resume`] and [`yield`] functions, which perform
    /// transfer of control between coroutines. For this reason it is undefined
    /// behavior for a [coroutine] to write onto the first `SIZE` bytes (in
    /// the [direction] of stack growth) starting at the [bottom] of its
    /// program stack.
    ///
    /// [program stack]: Stack
    /// [`resume`]: Coro::resume
    /// [`yield`]: yield_
    /// [coroutine]: Coro
    /// [direction]: StackOrientation
    /// [bottom]: Stack::bottom
    /// [starting address]: Stack::start
    /// [size]: Stack::size
    /// [page size]: os::page_size
    pub const SIZE: usize = size_of::<Self>();
}

/// A stackful, first-class asymmetric coroutine.
///
/// The [crate-level documentation] discusses this type in more detail.
///
/// [crate-level documentation]: crate
pub struct Coro<'a, Return> {
    /// The program stack of the coroutine, which [`drop`] returns to the
    /// [thread-local storage pool] when the coroutine is no longer needed.
    ///
    /// [`drop`]: Self::drop
    /// [thread-local storage pool]: COMMON_POOL
    #[cfg(feature = "std")]
    stack: ManuallyDrop<Stack>,
    /// The program stack of the coroutine, which is freed when the coroutine
    /// is dropped. Alternatively, [`StackPool::give_from`] returns the stack
    /// to a particular pool of available storage once the coroutine has
    /// completely finished its execution.
    ///
    /// [`StackPool::give_from`]: stack::StackPool::give_from
    #[cfg(not(feature = "std"))]
    stack: Stack,
    /// A coroutine acts as though it contains a reference to the [`FnOnce`]
    /// instance in its [program stack].
    ///
    /// [program stack]: Stack
    phantom: PhantomData<&'a fn() -> Return>,
}

/// The fixed size in bytes of the [program stack] of a [coroutine].
/// We currently use Rust's [default stack size] for a thread, 2 MiB.
///
/// [program stack]: Stack
/// [coroutine]: Coro
/// [default stack size]: std::thread#stack-size
// todo: this value should be configurable through an environment variable.
pub const STACK_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1 << 21) };

/// The minimum alignment requirement for the [program stack] of a [coroutine].
///
/// This quantity must match the [size] in bytes of any program stack, because
/// some internal routines need to locate the start of the current program stack
/// from the stack pointer value alone.
///
/// [program stack]: Stack
/// [coroutine]: Coro
/// [size]: STACK_SIZE
pub const STACK_ALIGN: Alignment = unsafe { Alignment::new_unchecked(STACK_SIZE.get()) };

impl<'a, Return> Coro<'a, Return> {
    /// Creates a coroutine that is to run the provided [`FnOnce`] instance on
    /// an independent [program stack] (of size [`STACK_SIZE`]) from a
    /// thread-local [pool of memory].
    ///
    /// [program stack]: Stack
    /// [pool of memory]: COMMON_POOL
    #[cfg(feature = "std")]
    #[must_use = "method creates a coroutine but does not begin its execution"]
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce() -> Return + 'a,
    {
        COMMON_POOL.with_borrow_mut(|pool| Self::with_stack_from(pool, f))
    }

    /// Creates a new coroutine that is to run the provided [`FnOnce`] instance
    /// on an independent [program stack] from a pool of available storage.
    ///
    /// If the `std` feature is enabled, the coroutine's destructor returns
    /// the stack to a thread-local [pool of memory]; otherwise the stack
    /// is dropped. One can also return the stack to a different pool of
    /// available storage by calling [`StackPool::give_from`], provided that
    /// the coroutine has completely finished its execution.
    ///
    /// [program stack]: Stack
    /// [pool of memory]: COMMON_POOL
    #[allow(rustdoc::broken_intra_doc_links)]
    #[must_use = "method creates a coroutine but does not begin its execution"]
    pub fn with_stack_from<F>(pool: &mut StackPool, f: F) -> Self
    where
        F: FnOnce() -> Return + 'a,
    {
        // Before the first call to `resume` invokes `f`, it needs to save the
        // old context at the top of the current program stack. The default ABI
        // in Rust does not define the set of callee-saved registers, so we must
        // jump through a trampoline function with an explicit calling convention
        // immediately after `arch::transfer_control` activates the coroutine
        // for the first time.
        extern "system" fn trampoline<'a, F, Return>() -> !
        where
            F: FnOnce() -> Return + 'a,
        {
            // Make a bitwise copy of the `FnOnce` instance on the new stack,
            // because that's the only way to call it safely.
            let stack_start = unsafe { stack_start() };
            let stack_bottom = match StackOrientation::current() {
                // SAFETY:
                StackOrientation::Upwards => unsafe { stack_start.byte_add(Control::SIZE) },
                StackOrientation::Downwards => unsafe {
                    stack_start.byte_add(STACK_SIZE.get() - Control::SIZE)
                },
            };
            let f_ptr = align_alloc::<F>(stack_bottom, Alignment::of::<F>());
            let f = unsafe { f_ptr.read() };
            // We are ready to execute the user's `FnOnce` instance.
            let ret_value = ManuallyDrop::new(f());
            // At this point the coroutine has almost finished its execution,
            // and `f` has been destroyed. (The latter observation is important,
            // because there's no guarantee that the local variables in scope
            // will be dropped before `return_control` is invoked below.) It
            // remains to restore the stack pointer to the value present in the
            // control record, and return to the last caller of `resume`. Any
            // attempt to resume the coroutine ever again will trigger a panic.
            // SAFETY: We have exclusive access to the stack.
            let control = unsafe { current_control() };
            unsafe { arch::return_control(control, NonNull::from(&ret_value)) }
        }
        // Allocate a stack for the coroutine.
        let stack = pool.take(STACK_SIZE, STACK_ALIGN);
        // Place `f` at the bottom of the new stack, and initialize the control
        // record as follows: First arrange things so that the coroutine begins
        // its execution at the first instruction of the trampoline function.
        // Then set the value of the stack pointer to the first appropriate
        // location above the copied `f` in the stack growth direction.
        let (f_ptr, stack_ptr) = stack.align_alloc(stack.bottom());
        // SAFETY: We know that `f_ptr` is aligned and points to a cell of the
        //         stack. Also, the present function is the only one that can
        //         access the stack at this moment.
        unsafe { f_ptr.write(f) };
        // Immediately before the first transfer of control to the coroutine,
        // its stack pointer needs to point to the beginning of the first frame
        // (which is associated with the trampoline function). The purpose of
        // the following `align_alloc` call is to respect the frame alignment
        // restrictions set by the ABI calling convention.
        let stack_ptr = align_alloc::<()>(stack_ptr, STACK_FRAME_ALIGN);
        // SAFETY: We do not support any platform where the stack may occupy
        //         the first page of memory (also known as the "zero page").
        //         Since `STACK_FRAME_ALIGN` is much smaller than the page size,
        //         this implies that the stack pointer cannot become zero after
        //         alignment.
        let stack_ptr = unsafe { NonNull::new_unchecked(stack_ptr) }.cast();
        // SAFETY: Again, we have exclusive access to the control record at the
        //         bottom of the stack.
        let control = unsafe { stack.control().as_uninit_mut() };
        control.write(Control {
            instr_ptr: trampoline::<F, Return> as *const _,
            stack_ptr,
        });
        // If the `std` feature is enabled, we must be careful not to drop the
        // stack once the coroutine is destroyed. Instead, we need to return it
        // to the thread-local pool of available storage.
        #[cfg(feature = "std")]
        let stack = ManuallyDrop::new(stack);
        Self {
            stack,
            phantom: PhantomData,
        }
    }

    /// Returns a reference to the control record associated with this coroutine.
    fn control(&self) -> &Control {
        // SAFETY: The control record was initialized in `Self::new`.
        unsafe { self.stack.control().as_ref() }
    }

    /// Returns an exclusive reference to the control record associated with
    /// this coroutine.
    fn control_mut(&mut self) -> &mut Control {
        // SAFETY: The control record was initialized in `Self::new`, and we
        //         have exclusive access to the stack.
        unsafe { self.stack.control().as_mut() }
    }

    /// Checks if the coroutine has completely finished its execution.
    pub fn is_finished(&self) -> bool {
        // The coroutine clears the `instr_ptr` field of the control record
        // when the trampoline function exits via `arch::return_control`.
        self.control().instr_ptr.is_null()
    }
}

#[cfg(feature = "std")]
impl<'a, Return> Drop for Coro<'a, Return> {
    fn drop(&mut self) {
        COMMON_POOL.with_borrow_mut(|pool| {
            // SAFETY: The `self.stack` variable is not used again.
            let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
            pool.give(stack);
        });
    }
}

impl<'a, Return> Coroutine for Coro<'a, Return> {
    type Yield = ();
    type Return = Return;

    fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
        // todo: document safety properties.
        let coro = self.get_mut();
        assert!(!coro.is_finished(), "coroutine resumed after completion");
        {
            let control = coro.control_mut();
            unsafe { arch::transfer_control(control) };
        }
        if coro.is_finished() {
            let ret_val_addr = coro.control().stack_ptr.cast();
            // Make a bitwise copy of the return value, so that it is safe to
            // destroy the coroutine stack.
            let ret_val = unsafe { ret_val_addr.read() };
            CoroutineState::Complete(ret_val)
        } else {
            CoroutineState::Yielded(())
        }
    }
}

/// Transfers control back to the computation that last resumed the active
/// coroutine.
///
/// # Panics
///
/// This function must be called from within a coroutine.
///
/// see the next section.
// todo: finish documentation.
///
/// # Safety
pub fn yield_() {
    // todo: check that we are within the active coroutine.
    let control = unsafe { current_control() };
    unsafe { arch::transfer_control(control) };
}

/// Returns the lowest location within the [current stack] of a [coroutine],
/// including its guard page.
///
/// # Safety
///
/// This function must be called from within a coroutine.
///
/// [current stack]: Stack
/// [coroutine]: Coro
unsafe fn stack_start() -> NonNull<u8> {
    arch::stack_ptr()
        // This transformation is the `NonNull`-equivalent of `pointer::mask`.
        // SAFETY: The stack does not occupy the first page of memory by
        //         assumption (see `arch::stack_ptr`), hence its starting
        //         location is positive.
        .map_addr(|a| NonZeroUsize::new_unchecked(a.get() & STACK_ALIGN.mask()))
}

/// Returns a reference to the control record of the active coroutine.
///
/// # Safety
///
/// This function must be called from within a coroutine. Moreover, the caller
/// must ensure that the control record is not accessed (read or written)
/// through any other pointer while the returned reference exists.
unsafe fn current_control<'a>() -> &'a mut Control {
    // SAFETY: The caller ensures that the current program stack is associated
    //         with the active coroutine.
    let stack_start = stack_start().cast();
    let mut control_ptr = match StackOrientation::current() {
        StackOrientation::Upwards => stack_start,
        // SAFETY: The program stack of the coroutine consists of `STACK_SIZE`
        //         bytes, a quantity bigger than `Control::SIZE`.
        StackOrientation::Downwards => stack_start.byte_add(STACK_SIZE.get() - Control::SIZE),
    };
    // SAFETY: This method is called by a coroutine whose control record was
    //         initialized in `Control::new`. The pointer is aligned because
    //         the page size is a multiple of `Control::SIZE`. Also, the caller
    //         guarantees that we have exclusive access to `control_ptr`'s
    //         contents.
    control_ptr.as_mut()
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use crate::{yield_, Coro};
    use std::ops::{Coroutine, CoroutineState};
    use std::pin::Pin;

    #[test]
    fn simple() {
        let mut coro = Coro::new(|| {
            eprintln!("[c] activated for the first time");
            yield_();
            eprintln!("[c] resumed fine, exiting");
        });
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => {
                eprintln!("[m] yielded once")
            }
            CoroutineState::Complete(_) => panic!("completed before expected"),
        }
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => panic!("yielded when not expected"),
            CoroutineState::Complete(_) => {
                eprintln!("[m] completed!")
            }
        }
    }

    #[test]
    fn return_value() {
        let mut coro = Coro::new(|| 12345);
        assert_eq!(
            Pin::new(&mut coro).resume(()),
            CoroutineState::Complete(12345)
        );
    }

    #[test]
    fn lots_of_yields() {
        const ITER_COUNT: usize = 10_000_000;
        let mut coro = Coro::new(|| {
            let mut res = 0;
            for i in 0..ITER_COUNT {
                res |= core::hint::black_box(i);
                yield_();
            }
            res
        });
        let mut res = 0;
        for i in 0..ITER_COUNT {
            res |= i;
            assert_eq!(Pin::new(&mut coro).resume(()), CoroutineState::Yielded(()));
        }
        assert_eq!(
            Pin::new(&mut coro).resume(()),
            CoroutineState::Complete(res)
        );
    }
}
