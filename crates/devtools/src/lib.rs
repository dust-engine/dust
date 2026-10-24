//! `devtools` — graphics debugging and profiling overlays for Dust.
//!
//! The crate hosts [`DevTools`], a Chrome-DevTools-style dock for debug
//! panels; its "Graphics Debug" tab with the renderer's debug toggles; its
//! "Entities" tab with the world's entities and their components; its
//! "Console" tab with the log and a command prompt; the performance overlay
//! tracking frame rate and GPU timings; and the player overlay showing the
//! player's position and orientation. Other crates can add DevTools tabs with
//! [`DevTools::add_tab`].
//!
//! The panels are GPUI views on the UI thread, drawn over the game's
//! `BevyView` by [`overlay`]. They show the Bevy world with [`watch`]
//! and [`read`], and edit it through `dust_ui`'s `BevyChannel`.

mod console;
mod devtools;
mod entities;
mod graphics_debug;
mod performance;
mod player;
mod widgets;

pub use console::{ConsoleLayer, ConsoleLog, console_layer};
pub use devtools::{DevTools, DockSide, ToggleDevTools, overlay, read, watch};
pub use entities::{EntitiesPlugin, EntityInspector};
pub use performance::{PerformancePanel, PerformancePanelPlugin};

use bevy::prelude::*;
use dust_profiler::GpuProfilerPlugin;

/// Adds the Bevy side of everything in this crate: the GPU profiler, the
/// frame timings the performance overlay shows, and the state of the
/// Entities tab. It is the single entry point so callers don't have to track
/// individual plugins as the crate grows. The panels themselves are created
/// on the UI thread, by [`overlay`].
///
/// The app has to be run by a `dust_ui::BevyView`.
#[derive(Default)]
pub struct GfxDebugPlugin;

impl Plugin for GfxDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GpuProfilerPlugin);
        app.add_plugins(PerformancePanelPlugin);
        app.add_plugins(EntitiesPlugin);
    }
}
