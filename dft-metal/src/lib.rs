//! GPU-accelerated DFT and fused DFT+Merkle for WHIR on Apple Silicon (Metal).
//!
//! Enable the `gpu-metal` feature on macOS/iOS. See the ethresear.ch write-up on
//! GPU-accelerated WHIR proving for architecture and benchmark context.

#![cfg_attr(not(feature = "gpu-metal"), no_std)]

#[cfg(all(feature = "gpu-metal", any(target_os = "macos", target_os = "ios")))]
pub mod gpu_dft;

#[cfg(all(feature = "gpu-metal", any(target_os = "macos", target_os = "ios")))]
pub use gpu_dft::{
    DftCommitFusion, GpuChallenger, GpuKeccakMmcs, GpuMmcs, MetalBabyBearDft,
};
