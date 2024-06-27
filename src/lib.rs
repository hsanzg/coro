use std::arch::asm;
use std::cell::RefCell;
use std::io::{Error, Result};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;
use std::ptr::{addr_of, null_mut, Alignment, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};

use smallvec::SmallVec;

macro_rules! bail_if {
    ($e:expr) => {
        if $e {
            return Err(Error::last_os_error());
        }
    };
}

/// Returns the default page size set by current platform, measured in bytes.
fn page_size() -> NonZeroUsize {
    // We try to remember the page size value across function invocations.
    // This optimization lets us avoid making a system call in the common
    // case, which gives a significant speedup. Mara Bos has explained the
    // idea underlying the following implementation of the lazy one-time
    // initialization pattern on pages 35--36 of her book _Rust Atomics and
    // Locks: Low-Level Concurrency in Practice_ (O'Reilly, 2023). Note
    // that the size of a page is a positive quantity, hence we can use
    // zero as a placeholder value.
    static VALUE: AtomicUsize = AtomicUsize::new(0);
    let value = match VALUE.load(Ordering::Relaxed) {
        0 => {
            let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
            assert!(value.is_power_of_two(), "page size must be a power of 2");
            VALUE.store(value, Ordering::Relaxed);
            value
        }
        v => v,
    };
    // SAFETY: Any power of 2 is positive.
    unsafe { NonZeroUsize::new_unchecked(value) }
}

/// The direction by which the stack pointer changes after a `push` instruction.
enum StackOrientation {
    /// A program stack grows towards _higher_ memory addresses.
    /// In other words, a `push` instruction _increases_ the stack pointer.
    Upwards,
    /// A program stack grows towards _lower_ memory addresses.
    /// In other words, a `push` instruction _decreases_ the stack pointer.
    Downwards,
}

impl StackOrientation {
    /// The orientation of every program stack in the current platform.
    #[cfg(target_arch = "x86_64")]
    pub fn current() -> Self {
        Self::Downwards
    }
}

/// The program stack of a [coroutine], used to store data and control structures.
/// More precisely, it is a set of contiguous memory pages where a coroutine is
/// free to store any data. There are exceptions to this statement, however:
/// 1. The [`resume`] and [`yield`] operations employ the first
///    [`STATUS_AREA_SIZE`] bytes of a coroutine's stack to record status
///    information during transfers of control. Here the meaning of "first"
///    depends on the [stack growth direction] in the present platform.
/// 2. To protect against overflows, we protect the last page of a program stack.
///    In this way, the process does not have permission to read from, write to,
///    or execute any memory location within this _guard page_; any attempt to
///    do so will cause a protection fault. For this reason, some [authors] do
///    not include the guard page as part of a program stack. We will _not_
///    follow this convention.
///
/// [coroutine]: `Coro`
/// [`resume`]: `Coro::resume`
/// [`yield`]: `yield_`
/// [`STATUS_AREA_SIZE`]: `Self::STATUS_AREA_SIZE`
/// [stack growth direction]: `StackOrientation`
/// [authors]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
struct Stack {
    /// The lowest address of a cell in the stack.
    base: NonNull<u8>,
    /// The number of bytes occupied by the stack.
    size: usize,
}

impl Stack {
    /// The fixed number of cells at the bottom of a program stack reserved for
    /// special use by the [`resume`] and [`yield`] functions, which perform
    /// transfer of control between coroutines. It is undefined behavior for
    /// a [coroutine] to write onto the first `STATUS_AREA_SIZE` locations
    /// (in the [direction] of stack growth) starting at the [first address]
    /// of its program stack.
    ///
    /// [`resume`]: `Coro::resume`
    /// [`yield`]: `yield_`
    /// [coroutine]: `Coro`
    /// [direction]: `StackOrientation`
    /// [first address]: `Stack::first`
    pub const STATUS_AREA_SIZE: usize = 16; // todo: replace by sizeof(CoroStatus)

    /// Allocates space for a new program stack of size at least `min_size`,
    /// with its [base address] being a multiple of `align`.
    ///
    /// [base address]: `Self::base`
    pub fn new(min_size: NonZeroUsize, align: Alignment) -> Result<Self> {
        // The size of the stack must be a multiple of the page size, because
        // we need to align the guard page to a page boundary. It also needs
        // to be larger than the page size, because otherwise the stack would
        // consist of a single guard page with no space available for data and
        // control structures.
        let page_size = page_size();
        let size = min_size
            // SAFETY: There is no known platform whose default page size is
            //         the largest power of 2 that fits in a `usize`.
            .max(unsafe {
                let two = NonZeroUsize::new_unchecked(2);
                page_size.unchecked_mul(two)
            })
            .get()
            .next_multiple_of(page_size.get());
        // The latest version of the POSIX standard at the time of writing,
        // namely IEEE Std. 1003.1-2024, doesn't give us a portable way to
        // create a memory mapping whose base address is a multiple of some
        // value greater than `page_size`. For this reason, we allocate just
        // enough storage to guarantee that the reserved area of memory has
        // a block of `size` locations starting at a multiple of `align`.
        let alloc_size = if align.as_usize() <= page_size.get() {
            size
        } else {
            size + align.as_usize()
        };
        // Reserve a `alloc_size`-byte memory area with read and write permission
        // only. We pass the `MAP_STACK` flag to indicate that we will use the
        // region to store a program stack. On success, the kernel guarantees
        // that the `base` address is page-aligned.
        let base = unsafe {
            libc::mmap(
                null_mut(),
                alloc_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_STACK,
                -1,
                0,
            )
        };
        bail_if!(base == libc::MAP_FAILED);
        // Locate the aligned `size`-byte block that will hold the stack. Next,
        // free all other pages in the mapping that would otherwise go unused.
        // This step can cause up to two TLB shootdowns, but stack allocations
        // are quite infrequent thanks to memory pooling (see `StackPool`).
        let offset = base.align_offset(align.as_usize());
        if offset > 0 {
            // SAFETY: Note that `offset` can be positive only if `align` is
            //         greater than `page_size`; in that case the page size
            //         divides `align`, so it also divides `offset`.
            bail_if!(unsafe { libc::munmap(base, offset) } == -1);
        }
        let base = unsafe { base.add(offset) };
        let tail_base = unsafe { base.add(size) };
        let tail_size = alloc_size - offset - size;
        if tail_size > 0 {
            // SAFETY: Since `base` and `size` are multiples of `page_size`,
            //         so is `tail_base`.
            bail_if!(unsafe { libc::munmap(tail_base, tail_size) } == -1);
        }
        // Make the last page of the stack inaccessible to the process.
        // The purpose of this step is to detect the situation when the stack
        // grows too large, and the running program attempts to access a memory
        // location within this "overflow" area (called the _guard page_). The
        // operating system will arrange things so that a memory-protection
        // interrupt will occur on any such attempt.
        let guard_base = match StackOrientation::current() {
            // SAFETY: `size` is at least `page_size`, so the result is between
            //         the starting address `base` of a `size`-byte mapping and
            //         one byte past it. Also, overflow is impossible in every
            //         supported architecture.
            StackOrientation::Upwards => unsafe { tail_base.sub(page_size.get()) },
            StackOrientation::Downwards => base,
        };
        bail_if!(unsafe { libc::mprotect(guard_base, page_size.get(), libc::PROT_NONE) } == -1);
        Ok(Self {
            // SAFETY: `base` is nonnull, because `mmap` completed successfully.
            base: unsafe { NonNull::new_unchecked(base.cast()) },
            size,
        })
    }

    /// Returns the base address of the stack.
    pub fn base(&self) -> NonNull<u8> {
        self.base
    }

    /// Returns the address where the item at the bottom of the stack appears
    /// in memory. This value depends on the [stack growth direction] in the
    /// present platform, and a [coroutine] is to access the contents of its
    /// program stack [`STATUS_AREA_SIZE`] bytes past the returned location
    /// in this direction.
    ///
    /// [stack growth direction]: `StackOrientation`
    /// [coroutine]: `Coro`
    /// [`STATUS_AREA_SIZE`]: `Self::STATUS_AREA_SIZE`
    pub fn first(&self) -> NonNull<u8> {
        match StackOrientation::current() {
            StackOrientation::Upwards => self.base,
            StackOrientation::Downwards => unsafe { self.base.add(self.size - 1) },
        }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        // Delete the mappings for this stack's address range.
        assert_eq!(
            unsafe { libc::munmap(self.base.cast().as_ptr(), self.size) },
            0, // indicates success.
            "failed to deallocate program stack"
        );
    }
}

/// A pool of available storage for use as the program stacks of new coroutines.
struct StackPool {
    /// The list of available program stacks.
    stacks: SmallVec<Stack, 4>,
}

impl StackPool {
    /// Creates a new storage pool.
    pub const fn new() -> Self {
        Self {
            stacks: SmallVec::new(),
        }
    }

    /// Reserves a program stack that can hold at least `min_size` bytes,
    /// and whose starting location is a multiple of `align`.
    pub fn take(&mut self, min_size: NonZeroUsize, align: Alignment) -> Result<Stack> {
        // Find the position of an appropriate available stack.
        if let Some(ix) = self
            .stacks
            .iter()
            .position(|s| s.size >= min_size.get() && s.base.is_aligned_to(align.as_usize()))
        {
            // Success: Remove `s` from the pool and hand it off to the caller.
            Ok(self.stacks.swap_remove(ix))
        } else {
            // Failure: Create a new stack of adequate size.
            Stack::new(min_size, align)
        }
    }

    /// Returns a stack to the pool of available storage.
    ///
    /// Note that this pool need not have allocated the given stack.
    pub fn give(&mut self, stack: Stack) {
        self.stacks.push(stack);
    }
}

thread_local! {
    /// A per-thread pool of available program stacks.
    static STACK_POOL: RefCell<StackPool> = const { RefCell::new(StackPool::new()) };
}

// idea: if all stacks had the same alignment, and it was a power of 2, then each
// coroutine could know where its stack starts and ends without any information
// apart from the current value of the stack pointer (assuming it has not overflowed
// nor underflowed). This would let us store metadata such as the caller's
// instruction and stack pointer in the "header" of the coroutine stack. We
// can then use this information to implement `yield` very efficiently, without
// the need of an external (nonprogram) stack in the heap.

/// Platform-specific functionality for the `x86_64` architecture.
#[cfg(target_arch = "x86_64")]
mod arch {
    use crate::coro::CoroStatus;

    unsafe fn current_status() -> CoroStatus {
        // todo: we should return a mutable reference, not a copy of the status.
        //       I'm having trouble expresssing the lifetime of the status object.
        todo!()
    }
}

#[repr(C)]
struct CoroStatus {
    ret_instr: *const extern "C" fn(),
    stack_pointer: *mut u8,
}

/// A stackful, first-class asymmetric coroutine.
///
/// The general notion of _coroutines_, first discussed in the published
/// literature by M. E. Conway \[_CACM_, **6** (1963), 396--408\], extends the
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
/// [`yield`]: `yield_`
/// [`resume`]: `Coroutine::resume`
/// [program stack]: `Stack`
/// [pooling techniques]: `StackPool`
pub struct Coro {
    stack: ManuallyDrop<Stack>,
}

impl Coro {
    // todo: document https://doc.rust-lang.org/std/thread/#stack-size
    // todo: make size configurable by getting it from an environment variable.
    const STACK_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1 << 21) }; // 2 KiB
    const STACK_ALIGN: Alignment = unsafe { Alignment::new_unchecked(Self::STACK_SIZE.get()) };

    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(),
    {
        extern "C" fn trampoline() {
            // todo: prepare stack
            // todo: call the given function
            // todo: restore the stack
            // note: all the previous steps must be done in assembly, because
            //       an asm! statement must preserve the original stack pointer.
        }
        /*extern "C" fn trampoline<F>()
        where
            F: FnOnce(),
        {
            unsafe { asm!("call rdi") };
            // let f_ptr = unsafe { ptr::read(raw_ptr as *const F) };
            // todo: switch stack pointer to correct place within `f_ptr`.
            // loop {}
        }*/

        let stack = STACK_POOL
            .with_borrow_mut(|pool| pool.take(Self::STACK_SIZE, Self::STACK_ALIGN))
            .expect("failed to create new stack");
        let status = unsafe { stack.base().cast::<CoroStatus>().as_mut() };
        status.ret_instr = addr_of!(trampoline);
        status.stack_pointer = stack.first().as_ptr();
        // let f_ptr = (f as *const _) as usize;
        Self {
            stack: ManuallyDrop::new(stack),
            //f_ptr: unsafe { std::mem::transmute(f_addr) },
        }
    }

    fn status_mut(&mut self) -> &mut CoroStatus {
        unsafe { self.stack.base().cast().as_mut() }
    }
}

impl Drop for Coro {
    fn drop(&mut self) {
        eprintln!("giving back stack to pool");
        STACK_POOL.with_borrow_mut(|pool| {
            // todo: review safety.
            // todo: should we replace `take` by `into_inner`? Be extremely
            //       confident that we are not adding the stack multiple times
            //       to the available storage pool.
            let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
            pool.give(stack);
        });
    }
}

impl Coroutine for Coro {
    // todo: support coroutine inputs, yield values and return values.
    type Yield = ();
    type Return = ();

    fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
        let coro = self.get_mut();
        let status = coro.status_mut();
        unsafe {
            asm!(
                // todo: Overwrite rdi and rsi with current sp and ip;
                //       do we need auxiliary variables?
            "mov sp, rdi",
                // todo: replace by regular jump, per recommendation of libfringe.
                //       Can we do this? We need to restore the SP and IP of the
                //       last caller to the function on procedure exit. This
                //       requires returning to the trampoline, which would then
                //       restore this necessary state.
            "call rsi",
            inout("rdi") status.stack_pointer, inout("rsi") status.ret_instr,
            clobber_abi("C")
            );
        }
        todo!()
    }
}

/// "save and restore the context needed by each coroutine"
pub fn yield_() {
    // todo: this is going to be really complicated, but fail if this function
    //       was not called from within a coroutine. Otherwise we would need
    //       to mark this function as "unsafe".
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
    }
}
