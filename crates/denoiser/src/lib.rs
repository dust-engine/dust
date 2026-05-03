//! Pluggable denoiser/upscaler backends for the dust renderer.
//!
//! Currently exposes the DLSS-RR backend under [`dlss`]; future backends
//! (FSR, XeSS, MetalFX) will live as sibling modules behind their own Cargo
//! features.

pub mod dlss;
