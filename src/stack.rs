use crate::os::{last_os_error, page_size};
use crate::Control;
use core::any::type_name;
#[cfg(feature = "std")]
use core::cell::RefCell;
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
    /// The orientation of every program stack in the current platform.
    #[cfg(target_arch = "x86_64")]
    pub const fn current() -> Self {
        Self::Downwards
    }
}

/// The program stack of a [coroutine], used to store data and control structures.
/// More precisely, it is a set of contiguous memory pages where a coroutine is
/// free to store any data. There are exceptions to this statement, however:
/// 1. The [`resume`] and [`yield`] operations employ the bottommost
///    [`Control::SIZE`] bytes of a coroutine's stack to record status
///    information during transfers of control. Here the meaning of "bottom"
///    depends on the [stack growth direction] in the present arch. The actual
///    stack contents appear on top of the [control record].
/// 2. To guard against overflows, we protect the last page of a program stack.
///    In this way the process does not have permission to read from, write to,
///    or execute any memory location within this _guard page_; any attempt to
///    do so will cause a protection fault. Some [authors] do not include the
///    guard page as part of a program stack, but we will _not_ follow this
///    convention when doing size calculations.
///
/// [coroutine]: crate::Coro
/// [`resume`]: core::ops::Coroutine::resume
/// [`yield`]: crate::yield_
/// [control record]: Control
/// [stack growth direction]: StackOrientation
/// [authors]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
#[allow(rustdoc::private_intra_doc_links)]
pub struct Stack {
    /// The lowest address of a memory cell in the stack.
    ///
    /// It is important to note that this field does not necessarily point to
    /// the [bottom] of the stack.
    ///
    /// [bottom]: Self::bottom
    start: NonNull<u8>,
    /// The number of bytes occupied by the stack.
    size: NonZeroUsize,
}

impl Stack {
    /// Allocates space for a new program stack of size at least `min_size`,
    /// with its [starting address] being a multiple of `align`.
    ///
    /// # Panics
    ///
    /// This function panics if it fails to allocate enough memory for the
    /// stack, or if it cannot protect the guard page against access by
    /// the current process.
    ///
    /// # Safety
    ///
    /// This function assumes that the smallest multiple of `align` and the
    /// [page size] greater than or equal to `size` is at most [`isize::MAX`].
    ///
    /// [starting address]: Self::start
    /// [page size]: page_size
    pub(crate) fn new(min_size: NonZeroUsize, align: Alignment) -> Self {
        // The size of the stack must be a multiple of the page size, because
        // we need to align the guard page to a page boundary. It also needs
        // to be larger than the page size, because otherwise the stack would
        // consist of a single guard page with no space available for data and
        // control structures. Signal an error in the highly unlikely event
        // that the control record cannot be placed at the bottom of the stack.
        let page_size = page_size();
        debug_assert!(
            Control::SIZE <= page_size.get(),
            "control record at the bottom of stack must be aligned"
        );
        let size = min_size
            // SAFETY: There is no known architecture whose default page size is
            //         the largest power of 2 that fits in a `usize`.
            .max(unsafe {
                let two = NonZeroUsize::new_unchecked(2);
                page_size.unchecked_mul(two)
            });
        let size = next_multiple_of(size, page_size);
        // The latest version of the POSIX standard at the time of writing,
        // namely IEEE Std. 1003.1-2024, does not give us a portable way to
        // create a memory mapping whose lowest location is a multiple of some
        // value greater than `page_size`. For this reason, we allocate just
        // enough storage to guarantee that the reserved area of memory has
        // a block of `size` locations starting at a multiple of `align`.
        let alloc_size = if align.as_usize() <= page_size.get() {
            size
        } else {
            // SAFETY: The caller ensures that this addition cannot overflow.
            unsafe { size.unchecked_add(align.as_usize()) }
        };
        // Reserve an `alloc_size`-byte area with read and write permission
        // only. We pass the `MAP_STACK` flag to indicate that we will use the
        // region to store a program stack. On success, the kernel guarantees
        // that the `start` address is page-aligned.
        let start = unsafe {
            libc::mmap(
                null_mut(),
                alloc_size.get(),
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
        let tail_start = unsafe { start.byte_add(size.get()) };
        let tail_size = alloc_size.get() - offset - size.get();
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

    /// Returns the location of the control record within this stack.
    ///
    /// The pointer is properly aligned: the stack size is a multiple of
    /// the page size, which is itself a multiple of [`Control::SIZE`].
    /// Care must be taken when dereferencing the pointer, however, because
    /// the record contents might be uninitialized.
    pub(crate) const fn control(&self) -> NonNull<Control> {
        match StackOrientation::current() {
            StackOrientation::Upwards => self.start,
            // SAFETY: The control structure starts `Control::SIZE` bytes before
            //         the end of a memory region of much greater size.
            StackOrientation::Downwards => unsafe {
                self.start.byte_add(self.size.get() - Control::SIZE)
            },
        }
        .cast()
    }

    /// Returns the top boundary of the control record within this stack, which
    /// depends on the [stack growth direction] in the present architecture.
    /// More precisely, if the stack grows [upwards], this method returns the
    /// address of the item that would appear at the bottom of the stack (on
    /// top of the control record). Otherwise (namely if the stack grows
    /// [downwards]), the returned address is the location one byte past the
    /// bottom of the stack. In any case a [coroutine] is not to access the
    /// [`Control::SIZE`] bytes past this address, in the opposite direction
    /// of stack growth.
    ///
    /// This concept should not be confused with the [starting location] of
    /// the stack.
    ///
    /// [stack growth direction]: StackOrientation
    /// [upwards]: StackOrientation::Upwards
    /// [downwards]: StackOrientation::Downwards
    /// [coroutine]: Coro
    /// [starting location]: Self::start
    pub(crate) const fn bottom(&self) -> NonNull<u8> {
        // SAFETY: The control structure occupies less than a page, so we can
        //         fit at least one byte in the stack.
        match StackOrientation::current() {
            StackOrientation::Upwards => unsafe { self.start.byte_add(Control::SIZE) },
            StackOrientation::Downwards => unsafe {
                self.start.byte_add(self.size.get() - Control::SIZE)
            },
        }
    }

    /// Returns two pointers to the beginning and end of the first block within
    /// the stack lying _above_ `ptr` that may contain a value of type `T`. The
    /// block is a set of `size_of::<T>()` contiguous memory locations starting
    /// at a multiple of `align_of::<T>()`, and it does not intersect any cell
    /// of memory [below] `ptr`. Note that the two locations coincide if the
    /// stack grows [downwards]; otherwise the first one is `size_of::<T>()`
    /// bytes smaller than the second.
    ///
    /// This operation is always safe, but using the resulting pointers is not.
    /// For example, the first pointer might not link to an initialized value
    /// of type `T`.
    ///
    /// # Panics
    ///
    /// This function panics if there is not enough room to store a value of
    /// `T` above `ptr` in the stack, and with support for aligned accesses.
    ///
    /// [below]: StackOrientation
    /// [downwards]: StackOrientation::Downwards
    pub(crate) fn align_alloc<T>(&self, ptr: NonNull<u8>) -> (NonNull<T>, NonNull<u8>) {
        let aligned = align_alloc::<T>(ptr, Alignment::of::<T>());
        // Check that the relevant `sizeof::<T>()`-byte block of consecutive
        // memory locations is within the bounds of the stack region. Note that
        // all pointers involved have the same provenance, so it makes sense to
        // compare their addresses.
        assert!(
            match StackOrientation::current() {
                StackOrientation::Upwards => {
                    // SAFETY: `start` is the first location of a `size`-byte stack.
                    let stack_end = unsafe { self.start.byte_add(self.size.get()) };
                    aligned.wrapping_add(1).cast() <= stack_end.as_ptr()
                }
                StackOrientation::Downwards => self.start.as_ptr() <= aligned.cast(),
            },
            "aligned value of type `{}` above {:?} would overflow the stack",
            type_name::<T>(),
            ptr
        );
        // SAFETY: The pointer refers to a location within the stack.
        let aligned = unsafe { NonNull::new_unchecked(aligned) };
        let end = match StackOrientation::current() {
            // SAFETY: In the worst case, the result of `add` is one byte past
            //         the end of the stack.
            StackOrientation::Upwards => unsafe { aligned.add(1) },
            StackOrientation::Downwards => aligned,
        };
        (aligned, end.cast())
    }
}

/// Returns the address of the first block _above_ `ptr` (in the [direction]
/// of stack growth) that may contain a value of type `T`. The block is a set
/// of `size_of::<T>()` contiguous memory locations starting at a multiple of
/// `align_of::<T>()`.
///
/// # Safety
///
/// This operation is always safe, but using the resulting pointer is not. For
/// example, the resulting pointer might not link to an initialized value of
/// type `T`. Offsetting the pointer might cause overflow, underflow, or return
/// the [zero address].
///
/// [direction]: StackOrientation
/// [zero address]: ptr::null
pub(crate) fn align_alloc<T>(ptr: NonNull<u8>, align: Alignment) -> *mut T {
    let align_mask = align.as_usize() - 1;
    ptr.as_ptr()
        .map_addr(|a| match StackOrientation::current() {
            StackOrientation::Upwards => (a + align_mask) & !align_mask,
            // The size of a value is always a multiple of its alignment, except
            // if `T` is zero-sized; the following code also handles this case.
            StackOrientation::Downwards => (a & !align_mask) - size_of::<T>(),
        })
        .cast()
}

impl Drop for Stack {
    fn drop(&mut self) {
        // Delete the mappings for this stack's address range.
        assert_eq!(
            unsafe { libc::munmap(self.start.cast().as_ptr(), self.size.get()) },
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
            .position(|s| s.size >= min_size && s.start.is_aligned_to(align.as_usize()))
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

    /// Consumes the given coroutine, returning its program stack to the pool of
    /// available storage.
    ///
    /// # Panics
    ///
    /// This function panics if the coroutine has not finished execution yet.
    #[cfg(not(feature = "std"))]
    pub fn give_from<F>(&mut self, coro: crate::Coro<F>) {
        assert!(
            coro.is_finished(),
            "cannot take stack of suspended coroutine"
        );
        self.0.push(coro.stack);
        // Section 10.8 of [_The Rust Language Reference_ (4d292b6)] states
        // that only the remaining fields of `Coro` are dropped after a partial
        // move; this property prevents a double-free of the stack instance.
        //
        // [_The Rust Language Reference_ (4d292b6)]: https://doc.rust-lang.org/reference/
    }
}

impl Default for StackPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
thread_local! {
    /// A per-thread pool of available program stacks.
    pub static COMMON_POOL: RefCell<StackPool> = const { RefCell::new(StackPool::new()) };
}

/// Calculates the smallest value greater than or equal to `lhs` that is a
/// multiple of `rhs`.
///
/// This function is the [`NonZeroUsize`] version of [`usize::next_multiple_of`].
///
/// # Panics
///
/// This function will panic on overflow.
fn next_multiple_of(lhs: NonZeroUsize, rhs: NonZeroUsize) -> NonZeroUsize {
    match lhs.get().next_multiple_of(rhs.get()) {
        // If overflow checks are enabled, `usize::next_multiple_of` already
        // panics by itself. Otherwise it wraps to zero on overflow, because
        // zero is the trivial multiple of every integer.
        0 => panic!("least multiple of {rhs} greater than or equal to {lhs} overflows a `usize`"),
        r => unsafe { NonZeroUsize::new_unchecked(r) },
    }
}
