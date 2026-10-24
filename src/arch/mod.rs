//! Platform-specific functionality for the current architecture.
#[cfg_attr(target_arch = "x86_64", path = "x86_64.rs")]
#[cfg_attr(target_arch = "aarch64", path = "aarch64.rs")]
mod imp;

use core::ptr::NonNull;
pub use self::imp::*;

/// A function to which [`resume_coro`] may transfer control when
/// starting a coroutine on a fresh stack (namely during its first
/// resumption or any resumption after it becomes [poisoned]).
///
/// [poisoned]: poison
// Follow the System V ABI on all x86-64 systems, including those
// running Windows; this simplifies the implementation of `init_coro`.
#[cfg(target_arch = "x86_64")]
pub type Trampoline = unsafe extern "sysv64-unwind" fn(bot: NonNull<()>) -> !;
#[cfg(not(target_arch = "x86_64"))]
pub type Trampoline = unsafe extern "C-unwind" fn(bot: NonNull<()>) -> !;

/// Makes an `unsafe fn` definition a valid [`Trampoline`].
#[cfg(target_arch = "x86_64")]
macro_rules! trampoline {
  attr() ($(#[$m:meta])* unsafe fn $($rest:tt)*) => {
    $(#[$m])* unsafe extern "sysv64-unwind" fn $($rest)*
  };
}
#[cfg(not(target_arch = "x86_64"))]
macro_rules! trampoline {
  attr() ($(#[$m:meta])* unsafe fn $($rest:tt)*) => {
    $(#[$m])* unsafe extern "C-unwind" fn $($rest)*
  };
}

pub(crate) use trampoline;
