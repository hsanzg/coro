// The following doc comment is kept in sync with the README.md file. Please
// run the `cargo sync-readme` command after modifying the comment contents.
//! This crate provides stackful, first-class asymmetric coroutines.
// todo: improve crate docs; include the following:
// The solution taken by other stackful coroutine libraries is to provide
// a `Yielder` object passed to the `FnOnce` instance of the coroutine,
// but we like the flexibility of suspending the execution of a coroutine
// without carrying any object everywhere.

#![feature(coroutine_trait)]
#![feature(non_null_convenience)]
#![feature(ptr_alignment_type)]
#![feature(nonzero_ops)]
#![feature(pointer_is_aligned)]
#![feature(naked_functions)]
#![feature(ptr_mask)]
#![feature(ptr_as_uninit)]
#![feature(strict_provenance)]
#![feature(asm_const)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub(crate) mod arch;
pub(crate) mod os;
pub(crate) mod stack;

#[cfg(feature = "std")]
use crate::stack::StackPool;
#[cfg(not(feature = "std"))]
pub use crate::stack::StackPool;
use crate::stack::{push_loc, Stack, StackOrientation};
#[cfg(feature = "std")]
use core::cell::RefCell;
use core::marker::PhantomData;
use core::mem::{size_of, ManuallyDrop};
use core::num::NonZeroUsize;
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr::{Alignment, NonNull};

#[cfg(feature = "std")]
thread_local! {
    /// A per-thread pool of available program stacks.
    static STACK_POOL: RefCell<StackPool> = const { RefCell::new(StackPool::new()) };
}

/// The control record of a [coroutine], used to store and restore the context
/// needed by the coroutine during a [transfer of control].
///
/// Any bit pattern is valid for this type; in particular, it is safe to
/// [transmute] an initialized array of `size_of::<Control>()` bytes into a
/// `Control`. The [`current_control`] function takes advantage of this fact.
///
/// [coroutine]: Coroutine
/// [transfer of control]: arch::transfer_control
/// [transmute]: core::mem::transmute
#[repr(C)] // The order of fields and the alignment of the struct are important.
struct Control {
    /// If the coroutine has not been activated yet, the address of a
    /// [trampoline function] that immediately bounces to its associated
    /// coroutine. If the coroutine is active, the location of the first
    /// instruction to execute when it reaches a suspension point or terminates.
    /// Otherwise the location immediately after the point where the coroutine
    /// was last suspended.
    ///
    /// This field is of use when [resuming] a coroutine that has not [finished]
    /// its execution, because it does so at the point where it last left off.
    ///
    /// [`trampoline` function]: Coro::new
    /// [resuming]: Coro::resume
    /// [finished]: Coro::is_finished
    instr_ptr: *const u8,
    /// Analogous to [`instr_ptr`], the value of the stack pointer register
    /// just before a transfer of control involving the coroutine associated
    /// with this control record.
    ///
    /// [`instr_ptr`]: Self::instr_ptr
    stack_ptr: *const u8,
    /// Whether the coroutine has completely finished its execution.
    finished: Finished,
}

impl Control {
    /// The fixed number of cells at the bottom of a [program stack] reserved
    /// for special use by the [`resume`] and [`yield`] functions, which perform
    /// transfer of control between coroutines. For this reason it is undefined
    /// behavior for a [coroutine] to write onto the first `SIZE` bytes (in
    /// the [direction] of stack growth) starting at the [bottom] of its
    /// program stack.
    ///
    /// This value is such that the initial stack pointer address meets the
    /// ABI-required minimum alignment requirements in the current architecture.
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

/// Indicates whether a [coroutine] has completed its computations or not.
///
/// This type is very much like a wrapper around a `bool` variable, except that
/// it can hold any _initialized_ bit pattern. This relaxed representation helps
/// us to check that the [current program stack] belongs to a [coroutine], and
/// hence that its [bottommost] cells of memory represent a valid [control record].
/// This detection mechanism is not foolproof, however, because the original
/// program stack (which is not associated with any coroutine) may contain
/// a [valid] `Finished` value in the place supposedly reserved for the
/// `finished` field of a [`Control`] object. The good news is that the bit
/// representations of the [`YES`] and [`NO`] constants were chosen completely
/// at random, so this event ought to occur very rarely. The bad news is that
/// this "probing" technique can easily lead to undefined behavior: Even if the
/// [`Control`] type accepts any bit pattern, the relevant [`Control::SIZE`]-byte
/// block of consecutive memory locations must all be within the bounds of a
/// single allocated object. (As the documentation of [`ptr::read`] states, each
/// stack-allocated variable is considered to be a separate allocated object in
/// Rust.) The `coro` library cannot uphold this invariant in general, but the
/// danger is low in practice: Calling [`yield`] from outside a coroutine will
/// almost always lead to a panic (the desired outcome!) or a memory-protection
/// fault (a particular manifestation of undefined behavior) on most platforms.
/// If we were to skip the check, the effects of a faulty call to [`yield`]
/// would be extremely difficult to debug. For example, the program may jump
/// the control to any location in memory. The key fact is that proper use of
/// the library will not lead to any memory safety violations. We hope that
/// future static code analysis tools will allow us to verify that every call
/// to [`yield`] in a program appears within the sections of code exclusively
/// executed by coroutines. This would eliminate the need for the fragile check
/// just explained.
///
/// The size of this type is platform-dependent, because it takes care of
/// bringing the [initial stack pointer] value of a [coroutine] into alignment.
///
/// [coroutine]: Coro
/// [current program stack]: Stack
/// [bottommost]: Control::SIZE
/// [control record]: Control
/// [valid]: Self::check_valid
/// [`YES`]: Self::YES
/// [`NO`]: Self::NO
/// [`ptr::read`]: core::ptr::read
/// [`yield`]: yield_
/// [initial stack pointer]: Stack::initial_ptr
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Finished(usize);

impl Finished {
    // todo: replace constants by randomly-chosen values.
    /// The coroutine has completely finished its execution.
    pub const YES: Self = Self(0xF00);
    /// The coroutine either has not yet started execution, is active, or has
    /// reached a suspension point.
    pub const NO: Self = Self(0xF01);

    /// Panics if this object does not represent one of the two possible
    /// termination states of a [coroutine]. Otherwise does nothing.
    ///
    /// [coroutine]: Coro
    pub fn check_valid(self) {
        assert!(
            self == Self::YES || self == Self::NO,
            "invalid coroutine termination state {:#X}; was `yield` called from outside a coroutine?",
            self.0
        )
    }
}

/// A stackful, first-class asymmetric coroutine.
///
/// The general notion of _coroutines_, first discussed in the published
/// literature by M. E. Conway \[_CACM_ **6** (1963), 396--408\], extends the
/// concept of subroutines by allowing them to share and pass data and control
/// back and forth. A coroutine suspends execution of its program by invoking
/// the function [`yield`], which returns control to the caller. Invoking
/// the [`resume`] method on a coroutine resumes execution of its program
/// immediately after the point where it was last suspended.
///
/// The Rust language has built-in support for coroutines, but one cannot easily
/// suspend their execution from within a nested function call. One way to solve
/// this problem is to maintain a separate [program stack] for each coroutine.
/// This approach usually takes more memory space; we employ [pooling techniques]
/// to diminish the need for large allocations and liberations.
///
/// See the paper "Revisiting Coroutines" by A. L. de Moura and R. Ierusalimschy
/// \[_ACM TOPLAS_ **31** (2009), 1--31] for the definitions of "stackful",
/// "first-class" and "asymmetric" used above.
///
/// [`yield`]: yield_
/// [`resume`]: Coroutine::resume
/// [program stack]: Stack
/// [pooling techniques]: StackPool
pub struct Coro<'f> {
    /// The program stack of the coroutine, which [`drop`] returns to the
    /// [thread-local storage pool] when the coroutine is no longer needed.
    ///
    /// [`drop`]: Self::drop
    /// [thread-local storage pool]: STACK_POOL
    #[cfg(feature = "std")]
    stack: ManuallyDrop<Stack>,
    /// The program stack of the coroutine, which is freed when the coroutine
    /// is dropped. Alternatively, [`StackPool::give_from`] returns the program
    /// stack to a pool of available storage once the coroutine has completely
    /// finished its execution.
    #[cfg(not(feature = "std"))]
    stack: Stack,
    /// A coroutine acts as though it contains a reference to the [`FnOnce`]
    /// instance in the [control record] of its [program stack].
    ///
    /// [control record]: Control
    /// [program stack]: Stack
    phantom: PhantomData<&'f ()>,
}

impl<'f> Coro<'f> {
    // todo: The stack size should be configurable through an environment variable.
    /// The fixed size in bytes of the [program stack] for a coroutine.
    /// We currently use Rust's [default stack size] for a thread, 2 MiB.
    ///
    /// [program stack]: Stack
    /// [default stack size]: std::thread#stack-size
    const STACK_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1 << 21) };
    /// The minimum alignment requirement for the [program stack] of a coroutine.
    /// Functions such as [`current_control`] require that this quantity is at
    /// least [`STACK_SIZE`], because they need to locate the [start] of the
    /// current program stack from the stack pointer value alone.
    ///
    /// [program stack]: Stack
    /// [`STACK_SIZE`]: Self::STACK_SIZE
    /// [start]: Stack::start
    const STACK_ALIGN: Alignment = unsafe { Alignment::new_unchecked(Self::STACK_SIZE.get()) };

    /// Creates a new coroutine that is to run the provided [`FnOnce`] instance
    /// on an independent [program stack] from a thread-local [pool of memory].
    ///
    /// [program stack]: Stack
    /// [pool of memory]: STACK_POOL
    #[cfg(feature = "std")]
    #[must_use = "method creates a coroutine but does not begin its execution"]
    pub fn new<F: FnOnce() + 'f>(f: F) -> Self {
        STACK_POOL.with_borrow_mut(|pool| Self::with_stack_from(pool, f))
    }

    /// Creates a new coroutine that is to run the provided [`FnOnce`] instance
    /// on an independent [program stack] from the pool of available storage.
    ///
    /// [program stack]: Stack
    #[must_use = "method creates a coroutine but does not begin its execution"]
    pub fn with_stack_from<F: FnOnce() + 'f>(pool: &mut StackPool, f: F) -> Self {
        // Before we can activate the coroutine (and invoke `f`) upon the first
        // call to `resume`, we need to save the old context at the top of the
        // current program stack. The default ABI in Rust does not define the
        // set of callee-saved registers, so we must jump through a trampoline
        // function with an explicit calling convention immediately after the
        // function `arch::transfer_control` activates the coroutine for the
        // first time.
        extern "system" fn trampoline<F: FnOnce()>() -> ! {
            // todo: outdated; update comment.
            // Read the address of the `FnOnce` instance from the control record
            // within the program stack of the coroutine. Then make a bitwise
            // copy of the coroutine on this new stack, because that's the only
            // way to call it (safely).
            let stack_start = unsafe { stack_start() };
            let bottom_ptr = match StackOrientation::current() {
                StackOrientation::Upwards => unsafe { stack_start.byte_add(Control::SIZE) },
                StackOrientation::Downwards => unsafe {
                    stack_start.byte_add(Coro::STACK_SIZE.get() - Control::SIZE)
                },
            };
            let (_, f_ptr) = push_loc::<F>(bottom_ptr);
            let f = unsafe { f_ptr.read() };

            /*let coro_fn: F = {
                // SAFETY: We have exclusive access to the stack, because the
                //         owner of this coroutine just transferred control to
                //         this function and is therefore suspended.
                let control = unsafe { current_control() };
                // Take the closure out of the control record, so that method
                // `is_finished` can distinguish between the case in which the
                // coroutine has not yet started, and the case in which it has
                // finished its execution.
                let fn_ptr = control
                    .coro_fn
                    .take()
                    .expect("closure pointer should be nonnull on first activation")
                    .cast();
                // Read the closure from `fn_ptr` without moving it.
                // SAFETY: `Coro::new` takes ownership of the closure, and it
                //         must return before the coroutine can be activated.
                //         Hence `trampoline` is the only function that can
                //         reference the closure at this point.
                unsafe { fn_ptr.read() }
                // Destroy the exclusive reference to `control` prior to calling
                // the closure, because `yield` also requires exclusive access
                // to the control record.
            };*/
            // Execute the provided closure in the stack of the coroutine.
            f();
            // At this point the coroutine has finished its execution, and `f`
            // has been destroyed. (The latter observation is important, because
            // there's no guarantee that the local variables in scope will be
            // dropped before `return_control` is invoked.) It remains to
            // restore the stack pointer to the value present in the control
            // record, and return to the last caller of `resume`. Any attempt
            // to resume the coroutine ever again will trigger a panic.
            // SAFETY: We have exclusive access to the stack.
            let control = unsafe { current_control() };
            unsafe { arch::return_control(control) }
        }
        // Allocate a stack for the coroutine, place `f` at the bottom, and
        // initialize the control record as follows: First set the value of
        // the stack pointer to the location one byte past the end of the
        // copied `FnOnce` instance (in the stack growth direction). Next,
        // arrange things so that the coroutine begins its execution at the
        // first instruction of the `trampoline` function.
        let stack = pool.take(Self::STACK_SIZE, Self::STACK_ALIGN);
        let stack_ptr = stack.initial_ptr();
        let (stack_ptr, f_ptr) = push_loc::<F>(stack_ptr);
        // SAFETY: todo
        unsafe { f_ptr.write(f) };

        // SAFETY: We have exclusive access to the entire stack, including
        //         its control record.
        let control = unsafe { stack.control().as_uninit_mut() };
        control.write(Control {
            stack_ptr: stack_ptr.as_ptr(),
            // SAFETY: Function pointers cannot be null.
            instr_ptr: trampoline::<F> as *const _,
            finished: Finished::NO,
        });
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

    /*/// Checks if the coroutine has completely finished its execution.
    fn is_finished(&self) -> bool {
        // The stack of a coroutine is empty if and only if it has either not
        // yet started execution or it has already terminated. To distinguish
        // between the two cases, we can exploit the fact that the trampoline
        // function clears the `coro_fn` field of the control record during
        // the first activation of the coroutine. Of course, the stack is empty
        // if the address in `control.stack_ptr` points to the item one byte
        // past the end of the control structure in the stack growth direction;
        // see `Stack::initial_ptr` for details.
        let control = self.control();
        control.coro_fn.is_none() && self.stack.initial_ptr() == control.stack_ptr
    }*/
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
        StackOrientation::Downwards => stack_start
            .byte_add(Coro::STACK_SIZE.get())
            .byte_sub(Control::SIZE),
    };
    // SAFETY: This method is called by a coroutine whose control record was
    //         initialized in `Control::new`. The pointer is aligned because
    //         the page size is a multiple of `Control::SIZE`. Also, the caller
    //         guarantees that we have exclusive access to `control_ptr`'s
    //         contents.
    control_ptr.as_mut()
}

unsafe fn stack_start() -> NonNull<u8> {
    arch::stack_ptr()
        // This transformation is the `NonNull`-equivalent of `pointer::mask`.
        // SAFETY: The stack does not occupy the first page of memory by
        //         assumption (see `arch::stack_ptr`), hence its starting
        //         location is positive.
        .map_addr(|a| NonZeroUsize::new_unchecked(a.get() & Coro::STACK_ALIGN.mask()))
}

#[cfg(feature = "std")]
impl<'f> Drop for Coro<'f> {
    fn drop(&mut self) {
        STACK_POOL.with_borrow_mut(|pool| {
            // SAFETY: The `self.stack` variable is not used again.
            let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
            pool.give(stack);
        });
    }
}

impl<'f> Coroutine for Coro<'f> {
    type Yield = ();
    type Return = ();

    fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
        // todo: document + add safety comments
        let coro = self.get_mut();
        {
            let control = coro.control_mut();
            unsafe { arch::transfer_control(control) };
        }
        let control = coro.control_mut();
        if control.finished == Finished::YES {
            CoroutineState::Complete(())
        } else {
            CoroutineState::Yielded(())
        }
    }
}

/// "save and restore the context needed by each coroutine"
pub fn yield_() {
    // todo: this is going to be really complicated, but fail if this function
    //       was not called from within a coroutine. Otherwise we would need
    //       to mark this function as "unsafe".
    // todo: this can actually lead to undefined behavior. Document extensively.
    let control = unsafe { current_control() };
    unsafe { arch::transfer_control(control) };
}

// todo: Convert to integration tests.
// todo: Test unwinding. (This is a big one.)
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    // todo: Make most test use a standalone stack pool, and test Coro::new
    //       separately.

    #[test]
    #[cfg(feature = "std")]
    fn simple() {
        eprintln!("[c] creating new");
        let mut coro = Coro::new(|| {
            eprintln!("[c] first point");
            yield_();
            eprintln!("[c] resumed fine, returning");
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

    /*#[test]
    fn foo() {
        let mut coro = Coro::new(|| {
            eprintln!("[c] started");
            yield_();
            eprintln!("[c] first resume");
            yield_();
            eprintln!("[c] second resume");
        });
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => eprintln!("[m] yielded once"),
            CoroutineState::Complete(_) => panic!("completed before"),
        };
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => eprintln!("[m] yielded twice"),
            CoroutineState::Complete(_) => panic!("completed before"),
        }
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => panic!("[m] yielded thrice"),
            CoroutineState::Complete(_) => eprintln!("completed"),
        }
    }*/

    /*#[test]
    fn foo() {
        struct Baz<'a, F> {
            f_ptr: *const u8,
            ph: PhantomData<&'a F>,
        }

        impl<'a, F: FnMut() + 'a> Baz<'a, F> {
            fn new(f: F) -> Self {
                let f = ManuallyDrop::new(f);
                Self {
                    f_ptr: &f as *const _ as *const u8,
                    ph: PhantomData,
                }
            }

            fn call(self) {
                let foo = self.f_ptr as *const F;
                let f = unsafe { foo.read() };
                f();
            }
        }

        let x = String::from("hello");
        let test = move || assert!(x == "hello");
        let baz = Baz::new(test);
        baz.call();
        // test();
    }*/
}
