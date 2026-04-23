use super::{GROWTH_DIR, GrowthDir};
use core::ffi::c_int;
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

const MAP_STACK: c_int = cfg_select! {
  any(target_os = "linux", target_os = "freebsd", target_os = "openbsd") => libc::MAP_STACK,
  _ => 0
};

pub fn alloc_stack(size: NonZeroUsize) -> NonNull<()> {
  // Reserve a `size`-byte area with read and write permissions only.
  // We pass the `MAP_STACK` flag to indicate that the region will be
  // used to store a program stack. On success, the kernel promises
  // that address `base` is page-aligned.
  let base = unsafe {
    libc::mmap(
      ptr::null_mut(),
      size.get(),
      libc::PROT_READ | libc::PROT_WRITE,
      libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | MAP_STACK,
      -1,
      0,
    )
  };
  assert_ne!(
    base,
    libc::MAP_FAILED,
    "stack alloc failed: {}",
    last_os_error()
  );
  // Make the topmost page of the stack inaccessible to this process.
  // The goal is to detect the situation when the stack grows too big,
  // and the program attempts to access a memory location within this
  // "overflow" area (called the _guard page_). The operating system
  // will arrange things so that a memory-protection interrupt occurs
  // on any such attempt.
  let page_size = page_size().get();
  let guard_base = match GROWTH_DIR {
    GrowthDir::Upwards => unsafe { base.byte_add(size.get() - page_size) },
    GrowthDir::Downwards => base,
  };
  assert_ne!(
    unsafe { libc::mprotect(guard_base, page_size, libc::PROT_NONE) },
    -1,
    "could not apply memory protection to guard page: {}",
    last_os_error()
  );
  // SAFETY: `mmap` returns a nonnull pointer on success.
  unsafe { NonNull::new_unchecked(base.cast()) }
}

pub fn free_stack(base: NonNull<()>, size: NonZeroUsize) {
  // Delete the mappings for the stack's address range.
  assert_eq!(
    unsafe { libc::munmap(base.cast().as_ptr(), size.get()) },
    0, // success.
    "failed to remove stack mapping: {}",
    last_os_error()
  );
}

// todo: rethink `std` strategy.
/*#[cfg(feature = "std")]
fn last_os_error() -> std::io::Error {
  std::io::Error::last_os_error()
}

#[cfg(not(feature = "std"))]*/
fn last_os_error() -> i32 {
  unsafe { *libc::__errno_location() }
}
