//! Program stacks for coroutines, and other related utilities.

#[cfg_attr(unix, path = "unix.rs")]
mod os;

use core::num::NonZero;
use core::num::NonZeroUsize;
use core::ptr::NonNull;

/// Describes how the stack pointer changes after a `push` instruction.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum GrowthDir {
  /// A program stack grows towards _higher_ memory addresses;
  /// that is, a `push` instruction _increases_ the stack pointer.
  Upwards,
  /// A program stack grows towards _lower_ memory addresses;
  /// that is, a `push` instruction _decreases_ the stack pointer.
  Downwards,
}

/// The [program stack] growth convention on the current platform.
///
/// [program stack]: Stack
pub const GROWTH_DIR: GrowthDir = const { GrowthDir::Downwards };

/// If [program stacks] are set to grow [towards zero], the address of
/// the top entry in a stack. Otherwise (namely if program stacks grow
/// [upwards]), the location following the topmost entry of a stack.
///
/// [program stacks]: Stack
/// [towards zero]: GrowthDir::Downwards
/// [upwards]: GrowthDir::Upwards
pub type StackPtr = NonNull<()>;

/// A set of contiguous memory pages where a [coroutine] is free to
/// store any data.
///
/// The page table systems of almost all modern computers support
/// memory protection via permission bits. In such computers, to
/// guard against overflows, the topmost page of a program stack
/// is _protected_. (Here the meaning of "topmost" depends on the
/// [stack growth convention] on the current platform.) In this way
/// the process does not have permission to read from nor write to
/// any memory location within this [_guard page_]; any attempt to
/// do so causes a protection fault. This special page is not part
/// of the stack, but we must be careful to consider it when doing
/// size calculations.
///
/// [coroutine]: crate::Coro
/// [stack growth convention]: GROWTH_DIR
/// [_guard page_]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
pub struct Stack {
  base: NonNull<()>,
  size: NonZeroUsize, // bytes.
  #[cfg(feature = "valgrind")]
  valgrind_id: ValgrindStackId,
}

impl Stack {
  /// Assigns an area of memory to a new program stack.
  ///
  /// # Safety
  ///
  /// The following conditions must hold before calling this method:
  /// - Locations `base` through `base + size - 1`, inclusive, belong
  ///   to the same [allocation];
  /// - The allocation remains valid for the entire lifetime of the
  ///   returned [`Stack`].
  ///
  /// Moreover, on architectures that support memory protection,
  /// - The pointer `base` is aligned with the beginning of a page;
  /// - The system's page size evenly divides `size`;
  /// - The topmost page of the stack (with respect to the stack's
  ///   [growth direction]) is inaccessible by the current process;
  /// - The current process also lacks permission to execute from
  ///   any other page belonging to the stack.
  ///
  /// Failure to satisfy these requirements results in undefined
  /// behavior, which may manifest itself in the form of memory
  /// corruption and/or a stack overflow.
  ///
  /// # Notes
  ///
  /// This method does not initialize the stack contents; the caller
  /// is responsible for ensuring that any data written to the stack
  /// is properly initialized before being read. In this respect the
  /// [`MaybeUninit`] type and the [`ptr::write`] function are often
  /// useful.
  ///
  /// [allocation]: https://doc.rust-lang.org/std/ptr/index.html#allocation
  /// [growth direction]: GROWTH_DIR
  /// [`MaybeUninit`]: core::mem::MaybeUninit
  /// [`ptr::write`]: core::ptr::write
  #[cfg(not(feature = "stack_alloc"))]
  pub unsafe fn from_raw(base: NonNull<()>, size: NonZeroUsize) -> Self {
    Self { base, size }
  }

  /// Reserves space for a new program stack of size at least
  /// `min_size` bytes.
  ///
  /// # Panics
  ///
  /// This method panics if it fails to reserve enough memory for
  /// the stack, or if it cannot protect the guard page against
  /// access by the current process.
  ///
  /// # Example
  ///
  /// Allocate an 8-kilobyte program stack:
  ///
  /// ```
  /// use coro::stack::Stack;
  ///
  /// let min_size = 1 << 13;
  /// let stack = Stack::allocate(min_size);
  ///
  /// assert!(stack.size().get() >= min_size);
  /// ```
  #[cfg(feature = "stack_alloc")]
  #[must_use]
  pub fn allocate(min_size: usize) -> Self {
    // The size of the stack must be a multiple of the page size, since
    // we need to align the guard page to a page boundary. It also needs
    // to be larger than the page size, because otherwise the stack would
    // consist of a single guard page with no space available for data.
    let page_size = os::page_size();
    let size = min_size
      .max(2 * page_size.get())
      .checked_next_multiple_of(page_size.get())
      .expect("stack size overflows a `usize` after page alignment");
    // SAFETY: We have `size >= 2 * page_size > 0`.
    let size = unsafe { NonZeroUsize::new_unchecked(size) };
    let base = os::alloc_stack(size);
    Self {
      base,
      size,
      #[cfg(feature = "valgrind")]
      valgrind_id: crate::arch::valgrind_register_stack(
        base.as_ptr(),
        // SAFETY: The stack covers the memory range `base..base+size`.
        unsafe { base.byte_add(size.get()) }.as_ptr(),
      ),
    }
  }

  /// Returns the smallest memory location within the stack area,
  /// including its guard page. This address is aligned with the
  /// beginning of a page.
  ///
  /// It is important to note that the returned pointer does not
  /// necessarily point to the [bottom] of the stack; this occurs
  /// only if program stacks grow [upwards] on the current platform.
  ///
  /// [bottom]: Self::bottom
  /// [upwards]: GrowthDir::Upwards
  pub fn base(&self) -> NonNull<()> {
    self.base
  }

  /// Returns the number of bytes occupied by the stack, including
  /// its guard page. This value is a multiple of the system's
  /// page size.
  pub fn size(&self) -> NonZeroUsize {
    self.size
  }

  /// If the stack grows [towards zero], returns the location one
  /// byte below the bottom of the stack. Otherwise (namely if the
  /// stack grows [upwards]) it gives the address of the item that
  /// would appear at the bottom of the stack. In both cases the
  /// address is aligned with the beginning of a page.
  ///
  /// This concept should not be confused with the [base address]
  /// of the stack.
  ///
  /// [upwards]: GrowthDir::Upwards
  /// [downwards]: GrowthDir::Downwards
  /// [base address]: Self::base
  pub fn bottom(&self) -> NonNull<()> {
    match GROWTH_DIR {
      GrowthDir::Upwards => self.base,
      // SAFETY: The last byte of the stack is at address `base+size-1`.
      GrowthDir::Downwards => unsafe { self.base.byte_add(self.size.get()) },
    }
  }
}

#[cfg(feature = "stack_alloc")]
impl Drop for Stack {
  fn drop(&mut self) {
    #[cfg(feature = "valgrind")]
    crate::arch::valgrind_deregister_stack(self.valgrind_id);
    // SAFETY: This stack was allocated by `os::alloc_stack`.
    unsafe { os::free_stack(self.base(), self.size) };
  }
}

/// Stacks can be allocated and deallocated by different threads.
unsafe impl Send for Stack {}

/// Reserves space for an aligned `T` immediately [above] `sp`.
///
/// More precisely, moves the stack pointer and returns an aligned
/// pointer to the reserved stack slot.
///
/// # Safety
///
/// Let `c..d` be the memory range of the allocation that `sp` is
/// derived from. The conditions `c <= next_multiple(sp, a) <= b - s`
/// and `c <= prev_multiple(sp, a) - s <= d - s` must hold if stacks
/// grow [upwards] and [downwards], respectively. Here `s` and `a`
/// are shorthand for `size_of::<T>()` and `align_of::<T>()`, resp.
///
/// [above]: GrowthDir
pub(crate) unsafe fn push<T>(sp: &mut StackPtr) -> NonNull<T> {
  let align_mask = align_of::<T>() - 1;
  let loc = sp.map_addr(|a| {
    let a = match GROWTH_DIR {
      GrowthDir::Upwards => (a.get() + align_mask) & !align_mask,
      GrowthDir::Downwards => (a.get() & !align_mask) - size_of::<T>(),
    };
    // SAFETY: The caller promises that rounding `sp` up, or rounding
    //  it down and subtracting `size_of::<T>()`, keeps it within its
    //  allocation. In particular, it cannot become zero nor overflow
    //  a `usize`.
    unsafe { NonZero::new_unchecked(a) }
  });
  // Advance the stack pointer.
  *sp = match GROWTH_DIR {
    // SAFETY: See above.
    GrowthDir::Upwards => unsafe { loc.byte_add(size_of::<T>()) },
    GrowthDir::Downwards => loc,
  };
  loc.cast()
}

/// A pool of available storage for use as the program stacks
/// of [coroutines].
///
/// [coroutines]: crate::Coro
#[cfg(feature = "stack_alloc")]
#[derive(Default)]
pub struct StackPool(
  /// The list of available (currently unused) program stacks.
  alloc::vec::Vec<Stack>,
);

#[cfg(feature = "stack_alloc")]
impl StackPool {
  /// Creates a new storage pool.
  pub const fn new() -> Self {
    Self(alloc::vec::Vec::new())
  }

  /// Reserves a program stack that can hold at least `min_size` bytes.
  ///
  /// # Example
  ///
  /// ```
  /// use coro::stack::StackPool;
  ///
  /// let mut pool = StackPool::new();
  /// let min_size = 1 << 13; // 8 kilobytes.
  /// let stack = pool.take(min_size);
  ///
  /// assert!(stack.size().get() >= min_size);
  /// ```
  #[must_use]
  pub fn take(&mut self, min_size: usize) -> Stack {
    self
      .0
      .iter()
      // Find an appropriate available stack.
      .position(|s| s.size.get() >= min_size)
      .map(|ix|
        // Success: Remove it from the pool and hand it off to the caller.
        self.0.swap_remove(ix))
      .unwrap_or_else(||
        // Failure: Create a new stack of adequate size.
        Stack::allocate(min_size))
  }

  /// Returns a stack back to the pool of available storage.
  ///
  /// Note that this pool need not have allocated the given stack.
  ///
  /// # Example
  ///
  /// ```
  /// use coro::stack::StackPool;
  ///
  /// let mut pool = StackPool::new();
  /// let stack = pool.take(1);
  /// pool.give(stack);
  /// ```
  pub fn give(&mut self, stack: Stack) {
    self.0.push(stack);
  }
}

/// Acquires a mutable reference to the thread-local common pool
/// of available program stacks.
///
/// # Safety
///
/// This function is sound to call only if there are no active
/// references (mutable or not) to the thread-local common pool.
/// In effect, this means that `f` must not call this function.
#[cfg(feature = "common_pool")]
pub(crate) unsafe fn borrow_common_pool<R, F>(f: F) -> R
where
  F: FnOnce(&mut StackPool) -> R,
{
  use core::cell::UnsafeCell;
  #[thread_local]
  static POOL: UnsafeCell<StackPool> = UnsafeCell::new(StackPool::new());
  // SAFETY: The caller upholds the aliasing rules.
  f(unsafe { &mut *POOL.get() })
}

/// A stack identifier to be referenced in Valgrind client requests.
#[cfg(feature = "valgrind")]
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct ValgrindStackId(pub libc::c_uint);

// The implementation of `valgrind_stack_{,de}register` is
// architecture-specific, hence it appears in `crate::arch`.

#[cfg(test)]
mod stack_tests {
  use super::*;

  #[test]
  #[cfg(not(feature = "stack_alloc"))]
  fn from_raw() {
    extern crate alloc;
    use alloc::alloc::*;

    let size = NonZeroUsize::new(1 << 12).unwrap();
    let page_size = os::page_size().get();
    let layout = Layout::from_size_align(size.get(), page_size).unwrap();
    let Some(base) = NonNull::new(unsafe { alloc(layout) }) else {
      handle_alloc_error(layout);
    };
    let stack = unsafe { Stack::from_raw(base, size) };

    assert_eq!(stack.base(), base);
    assert_eq!(stack.size(), size);

    drop(stack);
    unsafe { dealloc(base.as_ptr(), layout) };
  }

  #[test]
  #[cfg(feature = "stack_alloc")]
  fn alloc_stack_is_page_aligned() {
    let stack = Stack::allocate(1);
    let page_size = os::page_size().get();
    assert!(stack.base().is_aligned_to(page_size));
    assert!(stack.size().get().is_multiple_of(page_size));
    assert!(stack.bottom().is_aligned_to(page_size));
  }

  #[test]
  fn push_() {
    let stack = Stack::allocate(10);
    let mut sp = stack.bottom();
    let offset = |ptr: StackPtr| unsafe { ptr.byte_offset_from(stack.bottom()) };

    let upwards = GROWTH_DIR == GrowthDir::Upwards;

    let ptr = unsafe { push::<u8>(&mut sp) };
    assert_eq!(offset(sp), if upwards { 1 } else { -1 });
    assert_eq!(offset(ptr.cast()), if upwards { 0 } else { -1 });

    let ptr = unsafe { push::<u16>(&mut sp) };
    assert_eq!(offset(sp), if upwards { 4 } else { -4 });
    assert_eq!(offset(ptr.cast()), if upwards { 2 } else { -4 });

    let ptr = unsafe { push::<u32>(&mut sp) };
    assert_eq!(offset(sp), if upwards { 8 } else { -8 });
    assert_eq!(offset(ptr.cast()), if upwards { 4 } else { -8 });
  }
}
