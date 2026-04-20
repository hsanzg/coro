use crate::Control;
#[cfg(feature = "std")]
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ptr::NonNull;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::pin::Pin;
use std::thread_local;

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
pub type StackPtr = NonNull<()>;

/// A set of contiguous memory pages where a [coroutine] is free to
/// store any data.
///
// todo: this is incorrect; the bottom taken by the `FnOnce` instance.
/// The [`resume`] and [`suspend`] operations employ the bottommost
/// [`Control::SIZE`] bytes of a coroutine's stack to record status
/// information during transfers of control. (Here the meaning of
/// "bottommost" depends on the [stack growth convention] on the
/// current platform.) The actual stack contents appear on top of
/// the control record.
///
// todo: say that only in architectures that support memory protection.
/// To guard against overflows, the topmost page of a program stack
/// is _protected_. (Again, the meaning of "topmost" depends on the
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
  // todo: Finish doc comment.
  /// The lowest address of any memory cell in the stack.
  ///
  /// this is not necessarily the
  ///
  /// It is important to note that this field does not
  /// necessarily point to the [bottom] of the stack.
  ///
  /// [bottom]: Self::bottom
  base: NonNull<()>,
  /// The number of bytes occupied by the stack, including its
  /// guard page.
  size: NonZeroUsize,
}

impl Stack {
  /// Assigns an area of memory to a new program stack.
  ///
  /// # Safety
  ///
  /// The following invariants must hold before calling this method:
  /// 1. All addresses in the range `[base, base+size)` must point to
  ///   the same (contiguous) allocation;
  /// 2. The allocation remains valid for the entire lifetime of the
  ///   returned [`Stack`], and it is not aliased mutably elsewhere;
  /// 3. The topmost page of the allocation (with respect to the stack
  ///   growth direction) must not be accessible by the current process;
  /// 4. The remaining pages are readable and writable but not executable
  ///    by the current process, on architectures that support this feature;
  /// 5. `size` is a multiple of the system's page size.
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
  /// The `MaybeUninit` type may be of help.
  ///
  /// a (possibly partly-uninitialized) region of memory that can be used
  /// as a program stack. The topmost page of this region must not be
  /// accessible by the current process.
  ///
  ///
  #[cfg(not(feature = "std"))]
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
  #[cfg(feature = "std")]
  #[must_use]
  pub fn allocate(min_size: usize) -> Self {
    // The size of the stack must be a multiple of the page size, since
    // we need to align the guard page to a page boundary. It also needs
    // to be larger than the page size, because otherwise the stack would
    // consist of a single guard page with no space available for data.
    let page_size = crate::os::page_size();
    let size = min_size
      .max(2 * page_size.get())
      .checked_next_multiple_of(page_size.get())
      .expect("stack size overflows a `usize` after page alignment");
    // SAFETY: We have `size >= 2 * page_size > 0`.
    let size = unsafe { NonZeroUsize::new_unchecked(size) };
    let base = crate::os::alloc_stack(size);
    Self { base, size }
  }

  /// If the stack grows [towards zero], returns the location one
  /// byte below the bottom of the stack. Otherwise (namely if the
  /// stack grows [upwards]) it gives the address of the item that
  /// would appear at the bottom of the stack.
  ///
  /// This concept should not be confused with the [base address]
  /// of the stack.
  ///
  /// [upwards]: GrowthDir::Upwards
  /// [downwards]: GrowthDir::Downwards
  /// [base address]: Self::base
  pub(crate) fn bottom(&self) -> NonNull<()> {
    match GROWTH_DIR {
      GrowthDir::Upwards => unsafe { self.base.byte_add(Control::SIZE) },
      // SAFETY: The last location of the stack is `base + size - 1`.
      GrowthDir::Downwards => unsafe { self.base.byte_add(self.size.get() - Control::SIZE) },
    }
  }

  /// Returns a pointer that refers to the location `count` bytes
  /// above the [bottom] of the stack, following the [stack growth
  /// direction] on the current platform.
  ///
  /// # Safety
  ///
  /// This operation is always safe, but using the resulting pointer
  /// is not. For example, the pointer might link to an uninitialized
  /// position in the stack. Further offsetting operations might cause
  /// overflow, underflow, or produce the [zero address].
  ///
  /// # Panics
  ///
  /// This method panics if the result is not a memory location within
  /// the stack area.
  ///
  /// [bottom]: Self::bottom
  /// [stack growth direction]: GROWTH_DIR
  /// [zero address]: core::ptr::null
  // todo: is this method needed?
  pub fn byte_offset(&self, count: usize) -> NonNull<()> {
    assert!(
      count <= self.size.get(),
      "offset {count} overflows the stack"
    );
    match GROWTH_DIR {
      GrowthDir::Upwards => unsafe { self.base.byte_add(count) },
      GrowthDir::Downwards => unsafe { self.bottom().byte_sub(count) },
    }
  }

  /// Returns whether `addr` refers to a location within the stack
  /// or its guard page. The [bottom location] of the stack is part
  /// of the stack if and only if program stacks grow [upwards] on
  /// the [current platform].
  ///
  /// [bottom location]: Self::bottom
  /// [upwards]: GrowthDir::Upwards
  /// [current platform]: GROWTH_DIR
  // todo: is this method needed?
  pub fn contains(&self, addr: NonNull<()>) -> bool {
    let offset = addr.addr().get().wrapping_sub(self.base.addr().get());
    offset < self.size.get()
  }

  // the reason for taking a closure is to allow for objects to depend
  // on their own address; note that for address sensitivity to be safe,
  // care must be taken.
  pub unsafe fn push<T, F: FnOnce(NonNull<T>) -> T>(
    &mut self,
    sp: StackPtr, // todo: take as `&mut StackPtr`.
    f: F,
  ) -> (NonNull<T>, StackPtr) {
    let loc = next_slot(sp);
    unsafe { loc.write(f(loc)) };
    let sp = match GROWTH_DIR {
      GrowthDir::Upwards => unsafe { loc.add(1) },
      GrowthDir::Downwards => loc,
    };
    (loc, sp.cast())
  }

  /*fn control_addr(&self) -> NonNull<MaybeUninit<Control>> {
    match GROWTH_DIR {
      GrowthDir::Upwards => self.base,
      GrowthDir::Downwards => self.bottom(),
    }
    .cast()
  }*/

  /*/// Returns a reference to the control record in this stack.
  pub(crate) unsafe fn control(&self) -> Pin<&MaybeUninit<Control>> {
    // SAFETY:
    let control = unsafe { self.control_addr().cast().as_ref() };
    unsafe { Pin::new_unchecked(control) }
  }

  /// Returns a mutable reference to the control record in this stack.
  pub(crate) unsafe fn control_mut(&mut self) -> Pin<&mut MaybeUninit<Control>> {
    // SAFETY:
    // todo: document that caller guarantees that we have exclusive
    //  access to the control record.
    let control = unsafe { self.control_addr().cast().as_mut() };
    unsafe { Pin::new_unchecked(control) }
  }*/
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

#[cfg(feature = "std")]
impl Drop for Stack {
  fn drop(&mut self) {
    crate::os::free_stack(self.base, self.size);
  }
}

/// A pool of available storage for use as the program stacks
/// of [coroutines].
///
/// [coroutines]: crate::Coro
#[cfg(feature = "std")]
#[derive(Default)]
pub struct StackPool(Vec<Stack>);

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
thread_local! {
  /// A per-thread pool of available program stacks.
  pub(crate) static COMMON_POOL: RefCell<StackPool> = const { RefCell::new(StackPool::new()) };
}

#[cfg(all(test, feature = "alloc"))]
mod stack_tests {
  use super::*;
  use std::ptr;

  #[cfg(feature = "std")]
  fn alloc_stack(min_size: usize) -> ((), Stack) {
    ((), Stack::allocate(min_size))
  }

  #[cfg(not(feature = "std"))]
  fn alloc_stack(size: usize) -> (Box<[u8]>, Stack) {
    let reg = alloc::boxed::Box::new_uninit_slice(size);
    let stack = unsafe { Stack::new_unchecked(reg.as_ptr(), size) };
    (reg, stack)
  }

  #[test]
  fn new() {
    let min_size = 1 << 13;
    let (_reg, stack) = alloc_stack(min_size);
    assert!(stack.size.get() >= min_size);
    #[cfg(feature = "std")]
    {
      let page_size = crate::os::page_size().get();
      assert!(stack.base.is_aligned_to(page_size));
    }
  }

  #[test]
  fn byte_offset_in_bounds() {
    let (_reg, stack) = alloc_stack(16);
    let addr = stack.byte_offset(4);
    let offset = unsafe { addr.byte_offset_from(stack.bottom()) };
    let expected = match GROWTH_DIR {
      GrowthDir::Upwards => 4,
      GrowthDir::Downwards => -4,
    };
    assert_eq!(offset, expected);
  }

  #[test]
  #[should_panic]
  fn byte_offset_panics_on_overflow() {
    let (_reg, stack) = alloc_stack(4);
    let _ = stack.byte_offset(stack.size.get() + 1);
  }

  /*#[test]
  fn contains() {
    let (_reg, stack) = alloc_stack(1 << 10);
    assert!(stack.contains(stack.base));
    let in_bounds = unsafe { stack.base.byte_add(1 << 8) };
    assert!(stack.contains(in_bounds));
    assert!(!stack.contains(NonNull::dangling()));
    assert_eq!(
      stack.contains(stack.bottom()),
      GROWTH_DIR == GrowthDir::Upwards
    );
  }*/
}

#[cfg(all(test, feature = "std"))]
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
