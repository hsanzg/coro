use crate::os::{last_os_error, page_size};
use crate::{Control, Coro};
use core::mem::size_of;
use core::num::NonZeroUsize;
use core::ptr::{null_mut, Alignment, NonNull};
use smallvec::SmallVec;

/// The direction by which the stack pointer changes after a `push` instruction.
pub enum StackOrientation {
    /// A program stack grows towards _higher_ memory addresses.
    /// In other words, a `push` instruction _increases_ the stack pointer.
    Upwards,
    /// A program stack grows towards _lower_ memory addresses.
    /// In other words, a `push` instruction _decreases_ the stack pointer.
    Downwards,
}

impl StackOrientation {
    /// The orientation of every program stack in the current arch.
    #[cfg(target_arch = "x86_64")]
    pub const fn current() -> Self {
        Self::Downwards
    }
}

/// The program stack of a [coroutine], used to store data and control structures.
/// More precisely, it is a set of contiguous memory pages where a coroutine is
/// free to store any data. There are exceptions to this statement, however:
/// 1. The [`resume`] and [`yield`] operations employ the bottommost
///    [`CONTROL_BLOCK_SIZE`] bytes of a coroutine's stack to record status
///    information during transfers of control. Here the meaning of "bottom"
///    depends on the [stack growth direction] in the present arch. The
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
pub struct Stack {
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
    /// ABI-required minimum alignment requirements in the current architecture.
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
    pub(crate) const CONTROL_BLOCK_SIZE: usize = size_of::<Control>();

    /// Allocates space for a new program stack of size at least `min_size`,
    /// with its [starting address] being a multiple of `align`.
    ///
    /// [starting address]: Self::start
    pub(crate) fn new(min_size: NonZeroUsize, align: Alignment) -> Self {
        // The size of the stack must be a multiple of the page size, because
        // we need to align the guard page to a page boundary. It also needs
        // to be larger than the page size, because otherwise the stack would
        // consist of a single guard page with no space available for data and
        // control structures. Signal an error in the highly unlikely event
        // that the control record cannot be placed at the bottom of the stack.
        let page_size = page_size();
        debug_assert!(
            Self::CONTROL_BLOCK_SIZE <= page_size.get(),
            "control record at the bottom of stack must be aligned"
        );
        let size = min_size
            // SAFETY: There is no known architecture whose default page size is
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
    pub(crate) const fn start(&self) -> NonNull<u8> {
        self.start
    }

    /// Returns the bottom boundary of the stack in memory, which depends on the
    /// [stack growth direction] in the present architecture. More precisely,
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
    pub(crate) const fn bottom(&self) -> NonNull<u8> {
        match StackOrientation::current() {
            StackOrientation::Upwards => self.start,
            // SAFETY: `start` is the first location of a `size`-byte region.
            StackOrientation::Downwards => unsafe { self.start.byte_add(self.size) },
        }
    }

    /// Returns the location of the control record within this stack.
    ///
    /// The pointer is properly aligned: the stack size is a multiple of
    /// the page size, which is itself a multiple of [`CONTROL_BLOCK_SIZE`].
    /// Care must be taken when dereferencing the pointer, however, because
    /// the record contents might be uninitialized.
    ///
    /// [`CONTROL_BLOCK_SIZE`]: Self::CONTROL_BLOCK_SIZE
    pub(crate) const fn control(&self) -> NonNull<Control> {
        match StackOrientation::current() {
            StackOrientation::Upwards => self.start,
            // SAFETY: The control structure starts `CONTROL_BLOCK_SIZE` bytes
            //         before the end of a memory region of much greater size.
            StackOrientation::Downwards => unsafe {
                self.start.byte_add(self.size - Self::CONTROL_BLOCK_SIZE)
            },
        }
        .cast()
    }

    /*/// Returns an exclusive reference to the control structure at the bottom
    /// of this program stack.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the control record is not accessed (read or
    /// written) through any other pointer while the returned reference exists.
    pub unsafe fn control_mut<'a>(&self) -> &'a mut MaybeUninit<Control> {
        let ptr = match StackOrientation::current() {
            StackOrientation::Upwards => self.start,
            // SAFETY: The control structure starts `CONTROL_BLOCK_SIZE` bytes
            //         before the end of a memory region of much greater size.
            StackOrientation::Downwards => unsafe {
                self.start.byte_add(self.size - Self::CONTROL_BLOCK_SIZE)
            },
        };
        // SAFETY: the pointer is aligned because the page size is a multiple
        //         of `CONTROL_BLOCK_SIZE`. Also, the caller guarantees that
        //         we have exclusive access to `ptr`'s contents.
        unsafe { ptr.cast().as_uninit_mut() }
    }*/
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
    pub fn give(&mut self, stack: Stack) {
        self.0.push(stack);
    }

    /// Consumes the given coroutine, returning its program stack to the pool of
    /// available storage.
    ///
    /// # Panics
    ///
    /// This function panics if the coroutine has not finished execution yet.
    #[cfg(not(feature = "std"))]
    pub fn give_from(&mut self, coro: Coro) {
        assert!(
            coro.is_finished(),
            "cannot take stack of suspended coroutine"
        );
        self.0.push(coro.stack);
    }
}
