//! `gfxdebug` — graphics debugging and profiling overlays for Dust.
//!
//! The crate hosts egui-based debug panels. Today it provides a single
//! [`PerformancePanel`] tracking frame rate; more panels (GPU timings, memory,
//! ray-tracing stats) are expected to live here over time.
//!
//! The panels render into the primary egui context, which is set up by the
//! renderer plugins (e.g. `dust_pbr::PbrRenderPlugin`). This crate therefore
//! does not register `EguiPlugin` itself.

mod performance;

pub use performance::{PerformancePanel, PerformancePanelPlugin};

use bevy::prelude::*;

/// Adds every gfxdebug overlay. Currently this is just the performance panel,
/// but it is the single entry point so callers don't have to track individual
/// panel plugins as the crate grows.
#[derive(Default)]
pub struct GfxDebugPlugin;

impl Plugin for GfxDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PerformancePanelPlugin);
    }
}
