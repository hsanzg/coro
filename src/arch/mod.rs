//! Platform-specific functionality for the current architecture.
#[cfg_attr(target_arch = "x86_64", path = "x86_64.rs")]
mod imp;

pub use self::imp::*;
