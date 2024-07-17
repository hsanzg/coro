// The following doc comment is kept in sync with the README.md file. Please
// run the `cargo sync-readme` command after modifying the comment contents.
//! This crate provides stackful, first-class asymmetric coroutines.

#![feature(coroutine_trait)]
#![feature(non_null_convenience)]
#![feature(ptr_alignment_type)]
#![feature(nonzero_ops)]
#![feature(pointer_is_aligned)]
#![feature(naked_functions)]
#![feature(ptr_mask)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
use core::cell::RefCell;
use core::mem::{size_of, ManuallyDrop};
use core::num::NonZeroUsize;
use core::ops::{Coroutine, CoroutineState};
use core::pin::Pin;
use core::ptr;
use core::ptr::{addr_of, null_mut, Alignment, NonNull};
use core::sync::atomic::{AtomicUsize, Ordering};

use smallvec::SmallVec;

/// Returns an object that represents the last error reported by a system call
/// or a library function (if any).
#[cfg(feature = "std")]
fn last_os_error() -> std::io::Error {
    std::io::Error::last_os_error()
}

/// Returns the number of the last error reported by a system call or a library
/// function (if any), as found in variable `errno`.
#[cfg(not(feature = "std"))]
fn last_os_error() -> i32 {
    unsafe { *libc::__errno_location() }
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
    // SAFETY: Any real power of 2 is positive.
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
/// 1. The [`resume`] and [`yield`] operations employ the bottommost
///    [`CONTROL_BLOCK_SIZE`] bytes of a coroutine's stack to record status
///    information during transfers of control. Here the meaning of "bottom"
///    depends on the [stack growth direction] in the present platform. The
///    first stack frame starts on top of the [control record].
/// 2. To protect against overflows, we protect the last page of a program stack.
///    In this way the process does not have permission to read from, write to,
///    or execute any memory location within this _guard page_; any attempt to
///    do so will cause a protection fault. Some [authors] do not include the
///    guard page as part of a program stack, but we will _not_ follow this
///    convention when doing size calculations.
///
/// [coroutine]: Coro
/// [`resume`]: Coro::resume
/// [`yield`]: yield_
/// [`CONTROL_BLOCK_SIZE`]: Self::CONTROL_BLOCK_SIZE
/// [control record]: Control
/// [stack growth direction]: StackOrientation
/// [authors]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
struct Stack {
    /// The lowest address of a memory cell in the stack.
    ///
    /// It is important to note that this field does not necessarily point to
    /// the [bottom] of the stack.
    ///
    /// [bottom]: Self::bottom
    start: NonNull<u8>,
    /// The number of bytes occupied by the stack.
    size: usize,
}

impl Stack {
    /// The fixed number of cells at the bottom of a program stack reserved for
    /// special use by the [`resume`] and [`yield`] functions, which perform
    /// transfer of control between coroutines. For this reason it is undefined
    /// behavior for a [coroutine] to write onto the first `CONTROL_BLOCK_SIZE`
    /// bytes (in the [direction] of stack growth) starting at the [bottom] of
    /// its program stack.
    ///
    /// This value is such that the initial stack pointer address meets the
    /// ABI-required minimum alignment requirements in the current platform.
    ///
    /// # Panics
    ///
    /// This function panics if it fails to allocate enough memory for the
    /// stack, or if it cannot protect the guard page against access by
    /// the current process.
    ///
    /// [`resume`]: Coro::resume
    /// [`yield`]: yield_
    /// [coroutine]: Coro
    /// [direction]: StackOrientation
    /// [bottom]: Self::bottom
    /// [starting address]: Stack::start
    /// [size]: Self::size
    /// [page size]: page_size
    pub const CONTROL_BLOCK_SIZE: usize = size_of::<Control>();

    /// Allocates space for a new program stack of size at least `min_size`,
    /// with its [starting address] being a multiple of `align`.
    ///
    /// [starting address]: Self::start
    pub fn new(min_size: NonZeroUsize, align: Alignment) -> Self {
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
        // namely IEEE Std. 1003.1-2024, does not give us a portable way to
        // create a memory mapping whose lowest location is a multiple of some
        // value greater than `page_size`. For this reason, we allocate just
        // enough storage to guarantee that the reserved area of memory has
        // a block of `size` locations starting at a multiple of `align`.
        let alloc_size = if align.as_usize() <= page_size.get() {
            size
        } else {
            size + align.as_usize()
        };
        // Reserve an `alloc_size`-byte memory area with read and write permission
        // only. We pass the `MAP_STACK` flag to indicate that we will use the
        // region to store a program stack. On success, the kernel guarantees
        // that the `start` address is page-aligned.
        let start = unsafe {
            libc::mmap(
                null_mut(),
                alloc_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_STACK,
                -1,
                0,
            )
        };
        assert_ne!(
            start,
            libc::MAP_FAILED,
            "stack mapping creation failed: {}",
            last_os_error()
        );
        // Locate the aligned `size`-byte block that will hold the stack. Next,
        // free all other pages in the mapping that would otherwise go unused.
        // This step can cause up to two TLB shootdowns, but stack allocations
        // are quite infrequent thanks to memory pooling (see `StackPool`).
        let offset = start.align_offset(align.as_usize());
        if offset > 0 {
            // SAFETY: Note that `offset` can be positive only if `align` is
            //         greater than `page_size`; in that case the page size
            //         divides `align`, so it also divides `offset`.
            assert_ne!(
                unsafe { libc::munmap(start, offset) },
                -1,
                "failed to trim the front of the stack mapping: {}",
                last_os_error()
            );
        }
        let start = unsafe { start.add(offset) };
        let tail_start = unsafe { start.byte_add(size) };
        let tail_size = alloc_size - offset - size;
        if tail_size > 0 {
            // SAFETY: Since `start` and `size` are multiples of `page_size`,
            //         so is `tail_start`.
            assert_ne!(
                unsafe { libc::munmap(tail_start, tail_size) },
                -1,
                "failed to trim the back of the stack mapping: {}",
                last_os_error()
            );
        }
        // Make the last page of the stack inaccessible to the process.
        // The purpose of this step is to detect the situation when the stack
        // grows too large, and the running program attempts to access a memory
        // location within this "overflow" area (called the _guard page_). The
        // operating system will arrange things so that a memory-protection
        // interrupt will occur on any such attempt.
        let guard_start = match StackOrientation::current() {
            // SAFETY: `size` is at least `page_size`, so the result is between
            //         the starting address `start` of a `size`-byte mapping and
            //         one byte past it.
            StackOrientation::Upwards => unsafe { tail_start.byte_sub(page_size.get()) },
            StackOrientation::Downwards => start,
        };
        assert_ne!(
            unsafe { libc::mprotect(guard_start, page_size.get(), libc::PROT_NONE) },
            -1,
            "could not apply memory protection to guard page: {}",
            last_os_error()
        );
        Self {
            // SAFETY: `start` is nonnull, because `mmap` completed successfully.
            start: unsafe { NonNull::new_unchecked(start.cast()) },
            size,
        }
    }

    /// Returns the lowest location within the stack, including the guard page.
    pub fn start(&self) -> NonNull<u8> {
        self.start
    }

    /// Returns the bottom boundary of the stack in memory, which depends on
    /// the [stack growth direction] in the present platform. More precisely,
    /// if the stack grows [upwards], this method returns the address of the
    /// item that would appear at the bottom of the stack. Otherwise (namely
    /// if the stack grows [downwards]), the returned address is the location
    /// one byte past the bottom of the stack. In any case a [coroutine] is
    /// to access the contents of its program stack [`CONTROL_BLOCK_SIZE`]
    /// bytes past this address, in the stack growth direction.
    ///
    /// This concept should not to be confused with the [starting location]
    /// of the stack.
    ///
    /// [stack growth direction]: StackOrientation
    /// [upwards]: StackOrientation::Upwards
    /// [downwards]: StackOrientation::Downwards
    /// [coroutine]: Coro
    /// [`CONTROL_BLOCK_SIZE`]: Self::CONTROL_BLOCK_SIZE
    /// [starting location]: Self::start
    pub fn bottom(&self) -> NonNull<u8> {
        match StackOrientation::current() {
            StackOrientation::Upwards => self.start,
            // SAFETY: `start` is the first location of a `size`-byte region.
            StackOrientation::Downwards => unsafe { self.start.byte_add(self.size) },
        }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        // Delete the mappings for this stack's address range.
        assert_eq!(
            unsafe { libc::munmap(self.start.cast().as_ptr(), self.size) },
            0, // indicates success.
            "failed to remove program stack mapping: {}",
            last_os_error()
        );
    }
}

/// A pool of available storage for use as the program stacks of new coroutines.
pub struct StackPool(
    /// The list of available program stacks.
    SmallVec<Stack, 3>,
);

impl StackPool {
    /// Creates a new storage pool.
    pub const fn new() -> Self {
        Self(SmallVec::new())
    }

    /// Reserves a program stack that can hold at least `min_size` bytes,
    /// and whose starting location is a multiple of `align`.
    pub(crate) fn take(&mut self, min_size: NonZeroUsize, align: Alignment) -> Stack {
        // Find the position of an appropriate available stack.
        if let Some(ix) = self
            .0
            .iter()
            .position(|s| s.size >= min_size.get() && s.start.is_aligned_to(align.as_usize()))
        {
            // Success: Remove `s` from the pool and hand it off to the caller.
            self.0.swap_remove(ix)
        } else {
            // Failure: Create a new stack of adequate size.
            Stack::new(min_size, align)
        }
    }

    /// Returns a stack to the pool of available storage.
    ///
    /// Note that this pool need not have allocated the given stack.
    pub(crate) fn give(&mut self, stack: Stack) {
        self.0.push(stack);
    }
}

#[cfg(feature = "std")]
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

/// The control record of a [coroutine].
///
/// [coroutine]: Coroutine
#[repr(C)] // the order of fields is important
struct Control {
    resume_addr: *const u8,
    // todo: document
    _padding: u64,
    stack_ptr: *mut u8,
}

/// Platform-specific functionality for the `x86_64` architecture.
#[cfg(target_arch = "x86_64")]
mod arch {
    use crate::Control;
    use core::arch::asm;
    /*/// Returns the ABI-required minimum alignment of a stack frame in bytes.
    ///
    /// todo: clarify alignment requirement is on the base pointer, not on
    ///       starting address.
    pub const fn frame_alignment() -> Alignment {
        // See Section 3.2.2 of
        unsafe { Alignment::new_unchecked(16) }
    }*/

    /*pub unsafe fn current_status() -> &'static mut CoroStatus {
        // todo: we should return a mutable reference, not a copy of the status.
        //       I'm having trouble expressing the lifetime of the status object.
        todo!()
    }*/

    #[naked]
    pub extern "sysv64" fn stack_ptr() -> *mut u8 {
        unsafe { asm!("mov rax, rsp", "ret", options(noreturn)) }
    }

    // todo: remove
    #[naked]
    extern "sysv64" fn frame_pointer() -> *mut u8 {
        unsafe { asm!("mov rax, rbp", "ret", options(noreturn)) }
    }

    pub unsafe fn transfer_control(record: &mut Control) {
        #[cfg(feature = "std")]
        eprintln!(
            "[arch] transferring control: current sp={:?}, new sp={:?}, rp={:?}, rbp={:?}",
            stack_ptr(),
            record.stack_ptr,
            record.resume_addr,
            frame_pointer()
        );
        asm!(
            // Swap the current stack pointer with the address at $rdi.
            "mov rax, rsp",
            "mov rsp, [rdi]", // rsp <- record.stack_ptr
            "mov rbp, rsp", // also set frame base pointer.
                            // todo: only do so in debug mode.
            "mov [rdi], rax", // record.stack_ptr <- rax
            // Fetch the callee resumption point stored at $rsi, and store
            // the caller resumption point there.
            "mov rax, [rsi]",    // rax <- record.resume_addr
            "lea rdx, [rip+2f]", // rdx <- #2
            "mov [rsi], rdx",    // record.resume_addr <- rdx
            // Jump to the coroutine.
            "jmp rax",
            // If the coroutine yields or returns, it will do so to this point.
            // `yield` guarantees that the contents in the coroutine's registers
            // are not clobbered. Detect type of exit (return or yield) by
            // signaling via register.
            "2:",
            // todo: choose better registers.
            inout("rdi") &mut record.stack_ptr as *mut _ => _,
            inout("rsi") &mut record.resume_addr as *mut _ => _,
            // todo: Handle clobbers on Windows.
            clobber_abi("sysv64")
        );
        #[cfg(feature = "std")]
        eprintln!(
            "[arch] resumed or returned from coroutine: current sp={:?}, coro final sp={:?}, last rp={:?}",
            stack_ptr(),
            record.stack_ptr,
            record.resume_addr
        );
    }

    #[cfg(test)]
    mod tests {
        use crate::Stack;

        #[test]
        fn alignment_requirements() {
            // todo: cite System V 64.
            assert_eq!(Stack::CONTROL_BLOCK_SIZE % 16, 0);
        }
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
pub struct Coro {
    #[cfg(feature = "std")]
    stack: ManuallyDrop<Stack>,
    #[cfg(not(feature = "std"))]
    stack: Stack,
}

impl Coro {
    // todo: document https://doc.rust-lang.org/std/thread/#stack-size
    // todo: make size configurable by getting it from an environment variable.
    const STACK_SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1 << 21) }; // 2 KiB
    const STACK_ALIGN: Alignment = unsafe { Alignment::new_unchecked(Self::STACK_SIZE.get()) };

    #[cfg(feature = "std")]
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(),
    {
        STACK_POOL.with_borrow_mut(|pool| Self::with_stack_from(pool, f))
    }

    #[must_use = "must call `resume` to start the coroutine"] // todo: improve message and add it to new
    pub fn with_stack_from<F>(pool: &mut StackPool, f: F) -> Self
    where
        F: FnOnce(),
    {
        // todo: document that we need trampoline function to support our custom
        //       calling convention, because Rust has no defined ABI yet.
        extern "sysv64" fn trampoline<F: FnOnce()>() -> ! {
            let mut control_ptr = unsafe { current_control() };
            let control = unsafe { control_ptr.as_mut() };

            // todo: replace *mut F by *const F.
            let f_ptr = control_ptr.cast::<*const F>();
            let f_ptr = match StackOrientation::current() {
                StackOrientation::Upwards => unsafe { f_ptr.byte_add(Stack::CONTROL_BLOCK_SIZE) },
                StackOrientation::Downwards => unsafe { f_ptr.sub(1) },
            };
            #[cfg(feature = "std")]
            eprintln!(
                "[trampolining] status: control={:?}, sp={:?}, rp={:?}, f_ptr={f_ptr:?}",
                control_ptr, control.stack_ptr, control.resume_addr
            );
            // todo: document that initialized by `new`
            // This step makes a copy, but it's only executed once per coroutine.
            // todo: do we need to do anything with respect to dropping `F`. There
            //       is a quite delicate interaction with `new` here.
            let f = unsafe { f_ptr.read().read() };
            f();

            // todo: document that current value at top of stack is the return
            //       address of the last caller to the resume.
            let mut control_ptr = unsafe { current_control() };
            let control = unsafe { control_ptr.as_mut() };
            #[cfg(feature = "std")]
            eprintln!(
                "[returning] status: sp={:?}, rp={:?}",
                control.stack_ptr, control.resume_addr
            );
            loop {
                // Yield immediately, coroutine has completed its execution.
                // This is a rare case, so we do not optimize for it.
                // todo: follow rust built-in coroutines behavior and panic after
                //       the first transfer of control.
                unsafe {
                    arch::transfer_control(control);
                }
            }
        }
        let stack = pool.take(Self::STACK_SIZE, Self::STACK_ALIGN);
        #[cfg(feature = "std")]
        eprintln!(
            "[new] took stack start={:?}, size={:?}, bottom={:?}",
            stack.start(),
            stack.size,
            stack.bottom()
        );
        let control_ptr = match StackOrientation::current() {
            StackOrientation::Upwards => stack.bottom(),
            StackOrientation::Downwards => unsafe {
                stack.bottom().byte_sub(Stack::CONTROL_BLOCK_SIZE)
            },
        };
        // todo: review safety of this operation
        // todo: mention that initial stack pointer is aligned to nearest 16-byte boundary in x86-64, due to control size constraints.
        let control = unsafe { control_ptr.cast::<Control>().as_mut() };
        control.stack_ptr = match StackOrientation::current() {
            StackOrientation::Upwards => unsafe { control_ptr.byte_add(Stack::CONTROL_BLOCK_SIZE) },
            StackOrientation::Downwards => control_ptr,
        }
        .as_ptr();
        control.resume_addr = trampoline::<F> as *const u8;
        // Pass the original function pointer via the new coroutine stack.
        // todo: assert that *const F and *const u8 have same size and alignment
        //       requirements. Add verification as SAFETY notes, or (even better)
        //       improve.
        // let f_ptr = unsafe { status.stack_pointer.cast::<*const F>().sub(1) };
        // let f_arg_ptr = unsafe { control.stack_ptr.cast::<*const F>() };
        let f_arg_ptr = unsafe { stack.bottom().cast::<*const F>() };
        let f_arg_ptr = match StackOrientation::current() {
            StackOrientation::Upwards => unsafe { f_arg_ptr.byte_add(Stack::CONTROL_BLOCK_SIZE) },
            StackOrientation::Downwards => unsafe {
                f_arg_ptr.byte_sub(Stack::CONTROL_BLOCK_SIZE).sub(1)
            },
        };
        assert!(f_arg_ptr.is_aligned(), "f_arg is aligned");
        unsafe {
            f_arg_ptr.write(&f);
        }
        /*unsafe {
            // todo: remove, only for debugging purposes.
            control
                .stack_ptr
                .cast::<usize>()
                .sub(1)
                .write(0xDEADDEADDEADDEAD);
        }*/
        #[cfg(feature = "std")]
        eprintln!(
            "[new] status: sp={:?}, rp={:?}, wrote f_arg={:?} to {:?}",
            control.stack_ptr,
            control.resume_addr,
            unsafe { f_arg_ptr.read() },
            // &f as *const F,
            f_arg_ptr,
        );
        #[cfg(feature = "std")]
        let stack = ManuallyDrop::new(stack);
        Self { stack }
    }

    fn control_mut(&mut self) -> &mut Control {
        let bottom_ptr = self.stack.bottom();
        let control_ptr = match StackOrientation::current() {
            StackOrientation::Upwards => bottom_ptr,
            StackOrientation::Downwards => unsafe {
                bottom_ptr.byte_sub(Stack::CONTROL_BLOCK_SIZE)
            },
        };
        unsafe { control_ptr.cast().as_mut() }
    }
}

unsafe fn current_control() -> NonNull<Control> {
    let sp = arch::stack_ptr();
    let start =
        NonNull::new(unsafe { sp.mask(Coro::STACK_ALIGN.mask()) }).expect("null stack pointer");
    match StackOrientation::current() {
        StackOrientation::Upwards => start,
        StackOrientation::Downwards => unsafe {
            start
                .byte_add(Coro::STACK_SIZE.get())
                .byte_sub(Stack::CONTROL_BLOCK_SIZE)
        },
    }
    .cast()
}

#[cfg(feature = "std")]
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
        let control = coro.control_mut();
        #[cfg(feature = "std")]
        eprintln!(
            "[resuming] status: sp={:?}, rp={:?}",
            control.stack_ptr, control.resume_addr
        );
        // todo: we could probably do away with a single pointer to the status
        //       struct, and then use offsets.
        unsafe { arch::transfer_control(control) };
        /*unsafe {
            asm!(
                // Swap the current stack pointer with the address at $rdi.
                "mov rax, rsp",
                "mov rsp, [rdi]",
                "mov [rdi], rax",
                // Fetch the callee resumption point stored at $rsi, and store
                // the caller resumption point at $rsi.
                "mov rax, [rsi]",
                "lea rbx, [rip+2f]",
                "mov [rsi], rbx",
                // Jump to the coroutine.
                "jmp rax",
                // If the coroutine yields or returns, it will do so to this
                // point. `yield` guarantees that no registers are clobbered,
                // and it restores the stack pointer (saving it at $rdi) and
                // stores the resumption point at $rsi (if any).
                "2:", // resumption point
                /*// todo: Overwrite rdi and rsi with current sp and ip;
                //       do we need auxiliary variables?
                "mov rsp, [rdi]",
                // todo: replace by regular jump, per recommendation of libfringe.
                //       Can we do this? We need to restore the SP and IP of the
                //       last caller to the function on procedure exit. This
                //       requires returning to the trampoline, which would then
                //       restore this necessary state.
                "call rsi",
                "2:",*/
                inout("rdi") status.stack_pointer, inout("rsi") status.ret_instr,
                clobber_abi("C")
            );
        };*/
        // todo: To determine if yielded or not, check if `status.ret_instr`
        //       was overwritten or not (if yes, then yielded; otherwise returned
        //       normally via `ret` instruction).
        CoroutineState::Yielded(())
    }
}

/// "save and restore the context needed by each coroutine"
pub fn yield_() {
    // todo: this is going to be really complicated, but fail if this function
    //       was not called from within a coroutine. Otherwise we would need
    //       to mark this function as "unsafe".
    // let status = unsafe { arch::current_status() };
    let mut control_ptr = unsafe { current_control() };
    let control = unsafe { control_ptr.as_mut() };
    #[cfg(feature = "std")]
    eprintln!(
        "[yielding] status: sp={:?}, rp={:?}",
        control.stack_ptr, control.resume_addr
    );
    unsafe { arch::transfer_control(control) };
    /*unsafe {
        asm!(
            // Swap the current stack pointer with the address at $rdi.
            "mov rax, rsp",
            "mov rsp, [rdi]",
            "mov [rdi], rax",
            // Read the caller resumption point stored at $rsi, and store
            // the coroutine resumption point at $rsi.
            "mov rax, [rsi]",
            "lea rbx, [rip+2f]",
            "mov [rsi], rbx",
            // Jump to caller.
            "jmp rax",
            "2:", // resumption point
            inout("rdi") status.stack_pointer, inout("rsi") status.ret_instr,
            clobber_abi("C")
        )
    };*/
}

#[cfg(test)]
mod tests {
    use super::*;

    // todo: make most test use a standalone stack pool, and test Coro::new
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
}
