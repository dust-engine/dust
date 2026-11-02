//! Profiling for Dust: the frame timeline ([`timeline`]), which records where
//! each frame's time went from `tracing` spans and events, and the GPU profiler
//! ([`GpuProfiler`]), which times the GPU's work for it.

mod gpu;
pub mod timeline;

pub use gpu::{GpuProfiler, GpuProfilerPlugin, GpuTimerCommands};
