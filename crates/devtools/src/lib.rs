//! `devtools` — graphics debugging and profiling overlays for Dust.
//!
//! The crate hosts [`DevTools`], a Chrome-DevTools-style dock for debug
//! panels; its "Graphics Debug" tab with the renderer's debug toggles; its
//! "Console" tab with the log and a command prompt; the performance overlay
//! tracking frame rate and GPU timings; and the player overlay showing the
//! player's position and orientation. Other crates can add DevTools tabs with
//! [`DevToolsAppExt::add_devtools_tab`].
//!
//! The panels are GPUI views on the UI thread, drawn over the game by the
//! [`overlay`] given to `dust_ui::run`. They show the Bevy world with [`watch`]
//! and [`read`], and edit it through `dust_ui`'s `BevyHandle`.

mod console;
mod devtools;
// The "Entities" tab is still drawn with egui; it returns once ported to the
// UI thread.
// mod entities;
mod graphics_debug;
mod performance;
mod player;
mod widgets;

pub use console::{ConsolePlugin, console_layer};
pub use devtools::{DevTools, DevToolsAppExt, DockSide, ToggleDevTools, overlay, read, watch};
pub use graphics_debug::GraphicsDebugPlugin;
pub use performance::{PerformancePanel, PerformancePanelPlugin};

use bevy::prelude::*;
use dust_profiler::GpuProfilerPlugin;

/// Adds the Bevy side of everything in this crate: the DevTools dock's
/// Graphics Debug and Console tabs, the GPU profiler, and the frame timings the
/// performance overlay shows. It is the single entry point so callers don't
/// have to track individual plugins as the crate grows.
///
/// The app also needs `dust_ui`'s `UiHostPlugin`, added before `LogPlugin` so
/// that [`console_layer`] finds the `UiHandle` it inserts.
#[derive(Default)]
pub struct GfxDebugPlugin;

impl Plugin for GfxDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(devtools::TabsPlugin);
        app.add_plugins(GraphicsDebugPlugin);
        app.add_plugins(ConsolePlugin);
        app.add_plugins(GpuProfilerPlugin);
        app.add_plugins(PerformancePanelPlugin);
    }
}
