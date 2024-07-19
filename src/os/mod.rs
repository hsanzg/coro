//! Platform-specific functionality for the current operating system.
#[cfg_attr(target_family = "unix", path = "unix.rs")]
#[cfg_attr(target_family = "windows", path = "windows.rs")]
mod imp;

pub use self::imp::*;
