use super::{GROWTH_DIR, GrowthDir};
use core::num::NonZeroUsize;
use core::ptr;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Returns the default page size on the current architecture.
pub fn page_size() -> NonZeroUsize {
  // Try to remember the page size value across function invocations,
  // to save us the bother of making a system call in the majority of
  // calls. The size of a page is a positive quantity, so we can use
  // zero as a placeholder value.
  static VAL: AtomicUsize = AtomicUsize::new(0);
  let val = match VAL.load(Ordering::Relaxed) {
    0 => {
      #[expect(
        clippy::cast_possible_truncation,
        reason = "`sysconf` returns -1 or a size smaller than $2^{32}$"
      )]
      let val = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) as usize };
      assert!(val.is_power_of_two(), "page size must be a power of 2");
      VAL.store(val, Ordering::Relaxed);
      val
    }
    v => v,
  };
  // SAFETY: Every integer power of 2 is positive.
  unsafe { NonZeroUsize::new_unchecked(val) }
}

pub fn alloc_stack(size: NonZeroUsize) -> NonNull<u8> {
  let page_size = page_size().get();
  assert!(
    size.get().is_multiple_of(page_size),
    "stack size must be a multiple of the system's page size"
  );
  // Reserve a `size`-byte area with read and write permissions only.
  // If the current platform understands the `MAP_STACK` flag, we use
  // it to specify that the region will be used to store a program
  // stack. On success, the kernel promises that address `base` is
  // page-aligned.
  let base = unsafe {
    libc::mmap(
      ptr::null_mut(),
      size.get(),
      libc::PROT_READ | libc::PROT_WRITE,
      libc::MAP_PRIVATE
        | libc::MAP_ANONYMOUS
        | cfg_select! {
          any(target_os = "linux", target_os = "freebsd", target_os = "openbsd") => libc::MAP_STACK,
          _ => 0
        },
      -1,
      0,
    )
  };
  assert_ne!(
    base,
    libc::MAP_FAILED,
    "stack allocation failed: {}",
    last_os_error()
  );
  // Make the topmost page of the stack inaccessible to this process.
  // The goal is to detect the situation when the stack grows too big,
  // and the program attempts to access a memory location within this
  // "overflow" area (called the _guard page_). The operating system
  // will arrange things so that a memory-protection interrupt occurs
  // on any such attempt.
  let guard_base = match GROWTH_DIR {
    // SAFETY: Since `size` is a positive multiple of `page_size`,
    //  offset `size - page_size` is in bounds of the `size`-byte
    //  stack allocation.
    GrowthDir::Upwards => unsafe { base.byte_add(size.get() - page_size) },
    GrowthDir::Downwards => base,
  };
  assert_ne!(
    // Note that `guard_base` is aligned to a page boundary.
    unsafe { libc::mprotect(guard_base, page_size, libc::PROT_NONE) },
    -1,
    "could not apply memory protection to guard page: {}",
    last_os_error()
  );
  // SAFETY: `mmap` returns a nonnull pointer on success.
  unsafe { NonNull::new_unchecked(base.cast()) }
}

/// Frees a memory area previously allocated by [`alloc_stack`].
///
/// # Safety
///
/// The pointer `base` must be the return value of a previous
/// call to [`alloc_stack`] with argument `size`.
pub unsafe fn free_stack(base: NonNull<u8>, size: NonZeroUsize) {
  // Delete the mappings for the stack's address range.
  assert_eq!(
    unsafe { libc::munmap(base.cast().as_ptr(), size.get()) },
    0, // success.
    "failed to remove stack mapping: {}",
    last_os_error()
  );
}

/// Returns the value identifying the last error reported by a system
/// call or library function (if any), as found in variable `errno`.
fn last_os_error() -> i32 {
  unsafe extern "C" {
    #[cfg_attr(target_os = "linux", link_name = "__errno_location")]
    #[cfg_attr(
      any(target_vendor = "apple", target_os = "freebsd"),
      link_name = "__error"
    )]
    #[cfg_attr(target_os = "openbsd", link_name = "__errno")]
    safe fn errno_location() -> *mut libc::c_int;
  }
  // SAFETY: This is equivalent to accessing `errno` in C.
  unsafe { *errno_location() }
}
