//! Platform-specific functionality for the current operating system.
#[cfg_attr(unix, path = "unix.rs")]
mod imp;

pub use self::imp::*;
