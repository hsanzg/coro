use smallvec::SmallVec;
use std::alloc::Layout;
use std::arch::{asm, global_asm};
use std::cell::RefCell;
use std::io::{Error, Result};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::{Coroutine, CoroutineState, Deref};
use std::os::raw::c_void;
use std::pin::Pin;
use std::ptr::{addr_of, null_mut, Alignment, NonNull};
use std::slice::SliceIndex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::{io, ptr};

macro_rules! bail_if {
    ($e:expr) => {
        if $e {
            return Err(Error::last_os_error());
        }
    };
}

/// Returns the default page size in the current platform, measured in bytes.
fn page_size() -> usize {
    // We try to remember the page size value across function invocations.
    // This optimization lets us avoid making a system call in the common
    // case, which gives a significant speedup. The following implementation
    // of the lazy one-time initialization pattern is due to M. Bos; see
    // https://marabos.nl/atomics/atomics.html#example-racy-init for details.
    static VALUE: AtomicUsize = AtomicUsize::new(0);
    match VALUE.load(Ordering::Relaxed) {
        0 => {
            let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
            match VALUE.compare_exchange(0, value, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => value,
                Err(p) => p,
            }
        }
        value => value,
    }
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
    /// The orientation of a program stack in the current platform.
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "wasm32"
    ))]
    pub fn current() -> Self {
        Self::Downwards
    }
}

// idea: if all stacks had the same size, and it was a power of 2, then each
// coroutine could know where its stack starts and ends without any information
// apart from the current value of the stack pointer (assuming it has not overflowed
// nor underflowed). This would let us store metadata such as the caller's
// instruction and stack pointer in the "header" of the coroutine stack. We
// can then use this information to implement `yield` very efficiently, without
// the need of an external (nonprogram) stack in the heap.
// todo: We would then need to speed up `page_size`; benchmark cost of `OnceLock`.
//       Are there any safe alternatives that would let us speed it up? Thread
//       locals could work at the expense of a minimally greater amount of memory.
//       I still don't have a good mental model of the cost of thread locals either,
//       however.

/// The program stack of a coroutine, used to store data and control structures.
/// More precisely, it is a set of contiguous memory pages where a coroutine is
/// free to store any data. There are exceptions to this statement, however:
/// 1. The [`resume`] and [`yield`] operations employ the first eight bytes of
///    a coroutine's stack to record status information during transfers of
///    control. Here the meaning of "first" depends on the [stack growth direction]
///    in the present platform.
/// 2. To protect against overflows, we protect the last page of a program stack.
///    In this way, the process does not have permission to read from, write to,
///    or execute any memory location within this _guard page_; any attempt to
///    do so will cause a protection fault. For this reason, some [authors] do
///    not include the guard page as part of a program stack. We will _not_
///    follow this convention.
///
/// [`resume`]: `Coro::resume`
/// [`yield`]: `yield_`
/// [stack growth direction]: `StackOrientation`
/// [authors]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
struct Stack {
    /// The lowest address of a cell in the stack.
    base: NonNull<u8>,
    /// The number of bytes occupied by the stack, which is always
    /// a multiple of [`page_size()`](`page_size`).
    size: usize,
}

impl Stack {
    /// Allocates space for a new program stack.
    pub fn new(min_size: NonZeroUsize) -> Result<Self> {
        // The size of the stack must be a multiple of the page size,
        // because we need to align the guard page to a page boundary.
        let page_size = page_size();
        let size = min_size
            .get()
            .checked_next_multiple_of(page_size)
            .expect("`usize` cannot represent page-aligned stack size");
        // Reserve a `size`-byte memory area with read and write permission only.
        // We pass the `MAP_STACK` flag to indicate that we will use the region
        // to store a program stack. On success, the kernel guarantees that the
        // `base` address is page-aligned.
        let base = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_STACK,
                -1,
                0,
            )
        };
        bail_if!(base == libc::MAP_FAILED);
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
            StackOrientation::Upwards => unsafe { base.add(size).sub(page_size) },
            StackOrientation::Downwards => base,
        };
        bail_if!(unsafe { libc::mprotect(guard_base, page_size, libc::PROT_NONE) } == -1);
        Ok(Self {
            base: unsafe { NonNull::new_unchecked(base.cast()) },
            size,
        })
    }

    /// Returns the base address of the stack.
    fn base(&self) -> NonNull<u8> {
        self.base
    }

    fn first(&self) -> NonNull<u8> {
        match StackOrientation::current() {
            StackOrientation::Upwards => self.base,
            StackOrientation::Downwards => unsafe { self.base.add(self.size - 1) },
        }
    }

    fn end(&self) -> NonNull<u8> {
        unsafe { self.base.add(self.size) }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.size) };
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

    /// Reserves a program stack that can hold at least `min_size` bytes.
    pub fn take(&mut self, min_size: NonZeroUsize) -> Stack {
        // Find the position of an appropriate available stack.
        if let Some(ix) = self.stacks.iter().position(|s| s.size >= min_size.get()) {
            // Success: Remove `s` from the pool and hand it off to the caller.
            self.stacks.swap_remove(ix)
        } else {
            // Failure: Create a new stack of adequate size.
            Stack::new(min_size).expect("failed to create new stack")
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
    // todo: rename to GLOBAL_POOL?
    static STACK_POOL: RefCell<StackPool> = const { RefCell::new(StackPool::new()) };
    // todo: is this the most appropriate type? Can we encode pinning here?
    // todo: get rid of this thread local by keeping parent coroutine information
    //       in the first part of the current coroutine stack. In fact, we only
    //       need some parts of it (which address?)
    static ACTIVE: RefCell<SmallVec<NonNull<Coro>, 4>> = const { RefCell::new(SmallVec::new() )};
}

/// Platform-specific functionality for the `x86_64` architecture.
#[cfg(target_arch = "x86_64")]
mod arch {}

/// "An independent process that shares and pass data and control
/// back and forth."
pub struct Coro {
    state: CoroutineState<(), ()>,
    stack: ManuallyDrop<Stack>,
    f_ptr: fn(),
}

/*extern "C" resume_trampoline() {

}*/

global_asm!(
    r#"
.global resume_trampoline
resume_trampoline:
    mov rsp, rsi
    push rsi
    call rdi
    pop rsp
"#
);

unsafe extern "C" fn resume_trampoline() {}

impl Coro {
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(),
    {
        /*extern "C" fn trampoline<F>()
        where
            F: FnOnce(),
        {
            unsafe { asm!("call rdi") };
            // let f_ptr = unsafe { ptr::read(raw_ptr as *const F) };
            // todo: switch stack pointer to correct place within `f_ptr`.
            // loop {}
        }*/

        let stack = STACK_POOL.with_borrow_mut(|pool| pool.take(NonZeroUsize::new(0x100).unwrap()));
        let f_addr = addr_of!(f);
        // let f_ptr = (f as *const _) as usize;
        Self {
            state: CoroutineState::Yielded(()),
            stack: ManuallyDrop::new(stack),
            f_ptr: unsafe { std::mem::transmute(f_addr) },
        }
    }
}

impl Drop for Coro {
    fn drop(&mut self) {
        eprintln!("giving back stack to pool");
        STACK_POOL.with_borrow_mut(|pool| {
            // todo: review safety.
            let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
            pool.give(stack);
        });
    }
}

impl Coroutine for Coro {
    type Yield = ();
    type Return = ();

    fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
        ACTIVE.with_borrow_mut(|active| {
            active.push(NonNull::from(self.deref()));
            eprintln!("pushed coro to active stack");
        });
        //let current_addr = &*self as *const Self as usize;
        let f_ptr = (&*self).f_ptr;
        let stack_ptr = (&*self).stack.end().as_ptr();
        eprintln!("calling coro f_ptr {f_ptr:?} with stack @ {stack_ptr:?}");

        //let f_ptr = self.f_ptr;
        // The stack pointer is guaranteed to be suitably aligned.
        unsafe {
            asm!("call {tramp}", tramp = sym resume_trampoline, in("rdi") f_ptr, in("rsi") stack_ptr, clobber_abi("C"))
        };
        // todo: to see if the coroutine is done, we can check the stack pointer.
        CoroutineState::Complete(())
    }
}

pub fn yield_() {
    ACTIVE.with_borrow_mut(|active| {
        // todo: replace by expect.
        if let Some(active) = active.pop() {
            let active = unsafe { active.as_ref() };
            todo!("yield active coroutine")
        } else {
            // "from the main program"
            panic!("cannot yield while not running within a coroutine");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foo() {
        let mut coro = Coro::new(|| {
            eprintln!("started");
            yield_();
            eprintln!("first resume");
            yield_();
            eprintln!("second resume");
        });
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => eprintln!("yielded once"),
            CoroutineState::Complete(_) => panic!("completed before"),
        };
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => eprintln!("yielded twice"),
            CoroutineState::Complete(_) => panic!("completed before"),
        }
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => panic!("yielded thrice"),
            CoroutineState::Complete(_) => eprintln!("completed"),
        }
    }
}
