//! Platform-specific functionality for the current architecture.
#[cfg_attr(target_arch = "x86_64", path = "x86_64.rs")]
#[cfg_attr(target_arch = "aarch64", path = "aarch64.rs")]
mod imp;

pub use self::imp::*;
