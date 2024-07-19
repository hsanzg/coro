use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Returns an object that represents the last error reported by a system call
/// or a library function (if any).
#[cfg(feature = "std")]
pub fn last_os_error() -> std::io::Error {
    std::io::Error::last_os_error()
}

/// Returns the number of the last error reported by a system call or a library
/// function (if any), as found in variable `errno`.
#[cfg(not(feature = "std"))]
pub fn last_os_error() -> i32 {
    unsafe { *libc::__errno_location() }
}

/// Returns the default page size set by current architecture, measured in bytes.
pub fn page_size() -> NonZeroUsize {
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
