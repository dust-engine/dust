//! `devtools` — graphics debugging and profiling overlays for Dust.
//!
//! The crate hosts [`DevTools`], a Chrome-DevTools-style dock for egui debug
//! panels; its "Graphics Debug" tab with the renderer's debug toggles; and the
//! [`PerformancePanel`] overlay tracking frame rate and GPU timings. Other
//! crates can add DevTools tabs with [`DevToolsAppExt::add_devtools_tab`].
//!
//! The panels render into the primary egui context, which is set up by the
//! renderer plugins (e.g. `dust_pbr::PbrRenderPlugin`). This crate therefore
//! does not register `EguiPlugin` itself.

mod devtools;
mod graphics_debug;
mod performance;

pub use devtools::{DevTools, DevToolsAppExt, DevToolsPlugin, DockSide};
pub use graphics_debug::GraphicsDebugPlugin;
pub use performance::{PerformancePanel, PerformancePanelPlugin};

use bevy::prelude::*;
use dust_profiler::GpuProfilerPlugin;

/// Adds everything in this crate: the DevTools dock with its Graphics Debug
/// tab, the GPU profiler and the performance overlay. It is the single entry
/// point so callers don't have to track individual panel plugins as the crate
/// grows.
#[derive(Default)]
pub struct GfxDebugPlugin;

impl Plugin for GfxDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DevToolsPlugin);
        app.add_plugins(GraphicsDebugPlugin);
        app.add_plugins(GpuProfilerPlugin);
        app.add_plugins(PerformancePanelPlugin);
    }
}
