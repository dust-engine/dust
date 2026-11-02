//! `devtools` — graphics debugging and profiling overlays for Dust.
//!
//! The crate hosts [`DevTools`], a Chrome-DevTools-style dock for debug
//! panels; its "Graphics Debug" tab with the renderer's debug toggles and the
//! frame pacing page, a timeline of where each frame's time went on the CPU and
//! the GPU (see `dust_profiler::timeline`); its "Entities" tab with the
//! world's entities and their components; its "Schedules" tab with a graph of
//! the systems of a schedule and what orders them; its "Console" tab with the
//! log and a command prompt; and the player overlay showing the player's
//! position and orientation. Other crates can add DevTools tabs with
//! [`DevTools::add_tab`].
//!
//! The panels are GPUI views on the UI thread, drawn over the game's
//! `BevyView` by [`overlay`], or in a window of their own when DevTools is
//! detached. They show the Bevy world with [`watch`],
//! [`read`] and [`fetch`], and edit it through `dust_ui`'s `BevyChannel`.

mod console;
mod devtools;
mod entities;
mod graphics_debug;
mod player;
mod schedules;
mod widgets;

pub use console::{ConsoleLayer, ConsoleLog, console_layer};
pub use devtools::{DevTools, DockSide, ToggleDevTools, fetch, overlay, read, watch};
pub use entities::{EntitiesPlugin, EntityInspector};
pub use schedules::SchedulesPlugin;

use bevy::prelude::*;
use dust_profiler::GpuProfilerPlugin;

/// Adds the Bevy side of everything in this crate: the GPU profiler, the
/// state of the Entities tab, and what the Schedules tab records of the
/// schedules' builds. It is the single entry point so callers don't have to
/// track individual plugins as the crate grows. The panels themselves are
/// created on the UI thread, by [`overlay`].
///
/// The app has to be run by a `dust_ui::BevyView`.
#[derive(Default)]
pub struct GfxDebugPlugin;

impl Plugin for GfxDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GpuProfilerPlugin);
        app.add_plugins(EntitiesPlugin);
        app.add_plugins(SchedulesPlugin);
    }
}
