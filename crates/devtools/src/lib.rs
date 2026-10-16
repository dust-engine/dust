//! `devtools` — graphics debugging and profiling overlays for Dust.
//!
//! The crate hosts [`DevTools`], a Chrome-DevTools-style dock for egui debug
//! panels; its "Graphics Debug" tab with the renderer's debug toggles; its
//! "Entities" tab with the world's entities and their components; its
//! "Console" tab with the log and a command prompt; and the
//! [`PerformancePanel`] overlay tracking frame rate and GPU timings. Other
//! crates can add DevTools tabs with [`DevToolsAppExt::add_devtools_tab`].
//!
//! The panels render into the primary egui context, which is set up by the
//! renderer plugins (e.g. `dust_pbr::PbrRenderPlugin`). This crate therefore
//! does not register `EguiPlugin` itself.

mod console;
mod devtools;
mod entities;
mod graphics_debug;
mod performance;

pub use console::{ConsolePlugin, console_layer};
pub use devtools::{DevTools, DevToolsAppExt, DevToolsPlugin, DockSide};
pub use entities::{EntitiesPlugin, EntityInspector};
pub use graphics_debug::GraphicsDebugPlugin;
pub use performance::{PerformancePanel, PerformancePanelPlugin};

use bevy::prelude::*;
use dust_profiler::GpuProfilerPlugin;

/// Adds everything in this crate: the DevTools dock with its Graphics Debug,
/// Entities and Console tabs, the GPU profiler and the performance overlay. It
/// is the single entry point so callers don't have to track individual panel
/// plugins as the crate grows.
#[derive(Default)]
pub struct GfxDebugPlugin;

impl Plugin for GfxDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DevToolsPlugin);
        app.add_plugins(GraphicsDebugPlugin);
        app.add_plugins(EntitiesPlugin);
        app.add_plugins(ConsolePlugin);
        app.add_plugins(GpuProfilerPlugin);
        app.add_plugins(PerformancePanelPlugin);
    }
}
