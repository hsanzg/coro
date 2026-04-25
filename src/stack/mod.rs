//! Program stacks for coroutines, and other related utilities.

/// Platform-specific functionality for the current operating system.
#[cfg_attr(unix, path = "unix.rs")]
mod os;

#[cfg(feature = "stack_alloc")]
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::num::NonZero;
use core::num::NonZeroUsize;
use core::ptr::NonNull;

/// Describes how the stack pointer changes after a `push` instruction.
#[derive(PartialEq, Eq, Clone, Copy, Debug, core::marker::ConstParamTy)]
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
pub type StackPtr = NonNull<u8>;

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
/// [`resume`]: crate::Coro::resume
/// [`suspend`]: crate::suspend
/// [coroutine]: crate::Coro
/// [stack growth direction]: GROWTH_DIR
/// [_guard page_]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
pub struct Stack {
  base: NonNull<u8>,
  size: NonZeroUsize, // bytes.
}

impl Stack {
  // todo: finish doc comment.
  /// Assigns an area of memory to a new program stack.
  ///
  /// # Safety
  ///
  /// The following invariants must hold before calling this method:
  /// 1. All addresses in the range `[base, base+size)` must point to
  ///   the same (contiguous) allocation;
  /// 2. The allocation remains valid for the entire lifetime of the
  ///   returned [`Stack`], and it is not aliased mutably elsewhere; // todo: aliased in general, mutably or not. but rewrite somehow to say that coroutines may alias it.
  /// 3. On architectures that support memory protection, the topmost
  ///    page of the allocation (with respect to the stack growth direction)
  ///    must not be accessible by the current process;
  /// 4. The remaining pages are readable and writable but not executable
  ///    by the current process;
  /// 5. `size` is a multiple of the system's page size.
  /// 6. `base` must be aligned to a page boundary.
  ///
  /// Failure to satisfy these requirements may lead to undefined
  /// behavior, including memory corruption or failure to detect
  /// stack overflows.
  ///
  /// # Notes
  ///
  /// This method does not initialize the stack contents; the caller
  /// is responsible for ensuring that any data written to the stack
  /// is properly initialized before being read.
  /// The [`MaybeUninit`] type and the [`ptr::write`] subroutine may be of help in this respect.
  ///
  /// a (possibly partly-uninitialized) region of memory that can be used
  /// as a program stack. The topmost page of this region must not be
  /// accessible by the current process.
  ///
  ///
  #[cfg(not(feature = "stack_alloc"))]
  pub unsafe fn from_raw(base: NonNull<u8>, size: NonZeroUsize) -> Self {
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
    Self { base, size }
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
  pub fn base(&self) -> NonNull<u8> {
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
  pub fn bottom(&self) -> NonNull<u8> {
    match GROWTH_DIR {
      GrowthDir::Upwards => self.base,
      // SAFETY: The last byte of the stack is at address `base + size - 1`.
      GrowthDir::Downwards => unsafe { self.base.add(self.size.get()) },
    }
  }

  /// Returns whether `addr` refers to a location within this stack
  /// or its guard page. The [bottom location] of the stack is part
  /// of the stack if and only if program stacks grow [upwards] on
  /// the [current platform].
  ///
  /// [bottom location]: Self::bottom
  /// [upwards]: GrowthDir::Upwards
  /// [current platform]: GROWTH_DIR
  pub fn contains(&self, addr: NonNull<u8>) -> bool {
    let offset = addr.addr().get().wrapping_sub(self.base.addr().get());
    offset < self.size.get()
  }

  // the reason for taking a closure is to allow for objects to depend
  // on their own address; note that for address sensitivity to be safe,
  // care must be taken.
  pub(crate) unsafe fn push<T, F: FnOnce(NonNull<T>) -> T>(
    &mut self,
    sp: &mut StackPtr,
    f: F,
  ) -> NonNull<T> {
    let loc = next_slot(*sp);
    unsafe { loc.write(f(loc)) };
    *sp = match GROWTH_DIR {
      GrowthDir::Upwards => unsafe { loc.add(1) },
      GrowthDir::Downwards => loc,
    }
    .cast();
    loc
  }
}

pub(crate) fn next_slot<T>(sp: StackPtr) -> NonNull<T> {
  let align_mask = align_of::<T>() - 1;
  sp.map_addr(|a| {
    let addr = match GROWTH_DIR {
      GrowthDir::Upwards => (a.get() + align_mask) & !align_mask,
      GrowthDir::Downwards => (a.get() & !align_mask) - size_of::<T>(),
    };
    // SAFETY:
    // todo: add proof of correctness.
    unsafe { NonZero::new_unchecked(addr) }
  })
  .cast()
}

#[cfg(feature = "stack_alloc")]
impl Drop for Stack {
  fn drop(&mut self) {
    os::free_stack(self.base, self.size);
  }
}

/// A pool of available storage for use as the program stacks
/// of [coroutines].
///
/// [coroutines]: crate::Coro
#[cfg(feature = "stack_alloc")]
#[derive(Default)]
pub struct StackPool(
  /// The list of available (currently unused) program stacks.
  Vec<Stack>,
);

#[cfg(feature = "stack_alloc")]
impl StackPool {
  /// Creates a new storage pool.
  pub const fn new() -> Self {
    Self(Vec::new())
  }

  /// Reserves a program stack that can hold at least `min_size` bytes.
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
  pub fn give(&mut self, stack: Stack) {
    self.0.push(stack);
  }
}

/// Returns a mutable reference to the thread-local common pool
/// of available program stacks.
///
/// # Safety
///
/// This function is sound to call only if there are no active
/// references (mutable or not) to the thread-local common pool.
#[cfg(feature = "common_pool")]
pub(crate) unsafe fn common_pool() -> &'static mut StackPool {
  #[thread_local]
  static POOL: UnsafeCell<StackPool> = const { UnsafeCell::new(StackPool::new()) };
  // SAFETY: The caller upholds the aliasing rules.
  unsafe { &mut *POOL.get() }
}

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
  #[cfg(feature = "stack_alloc")]
  fn contains() {
    let stack = Stack::allocate(1 << 12);
    assert!(stack.contains(stack.base()));
    let last_byte = unsafe { stack.base().add(stack.size().get() - 1) };
    assert!(stack.contains(last_byte));
    assert_eq!(
      stack.contains(stack.bottom()),
      GROWTH_DIR == GrowthDir::Upwards
    );
  }
}

#[cfg(all(test, feature = "stack_alloc"))]
mod pool_tests {
  use super::*;

  #[test]
  fn take_from_pool() {
    let mut pool = StackPool::new();
    let min_size = 1 << 13;
    let stack = pool.take(min_size);
    assert!(stack.size.get() >= min_size);
  }

  #[test]
  fn give_back_to_pool() {
    let mut pool = StackPool::new();
    let stack = pool.take(1);
    pool.give(stack);
  }
}
