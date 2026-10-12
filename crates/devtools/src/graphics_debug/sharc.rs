//! Debug toggles of the "SHARC" stage.

use bevy::prelude::*;
use dust_pbr::sharc::{SharcConfig, SharcDebugMode, SharcDebugState};
use pumicite_egui::egui;

pub(super) fn sharc_debug_ui(
    InMut(ui): InMut<egui::Ui>,
    mut debug_state: ResMut<SharcDebugState>,
    mut config: ResMut<SharcConfig>,
) {
    // Master controls — the two knobs reached for most often.
    ui.horizontal(|ui| {
        ui.checkbox(&mut config.enabled, "Cache enabled");
        if ui.button("Reset cache").clicked() {
            config.reset_pending = true;
        }
    });

    ui.separator();

    // Grid — shared by the live cache and the debug overlay. There is
    // exactly one scene scale: the Update/Resolve passes and the
    // colored-hash overlay both key off it. logarithm base / level
    // bias are compile-time constants (SHARC_GRID_*), shown read-only
    // so the overlay can't drift from the grid the cache uses.
    ui.label(egui::RichText::new("Grid").strong());
    ui.add(egui::Slider::new(&mut config.scene_scale, 1.0..=200.0).text("Scene scale"));

    ui.separator();

    // Cache population (Update / Resolve) knobs.
    ui.collapsing("Cache population", |ui| {
        // Pool capacity ≡ Update ray budget. The pool buffers are
        // sized to SharcConfig::default().pool_capacity at startup, so
        // keep the slider at/under that — growing past it is a no-op
        // without a reallocation.
        ui.add(
            egui::Slider::new(&mut config.pool_capacity, 1024..=(1 << 16))
                .logarithmic(true)
                .text("Pool capacity (Update rays / frame)"),
        );
        ui.add(
            egui::Slider::new(&mut config.accumulation_frame_num, 0..=128)
                .text("Accumulation frames"),
        );
        ui.add(egui::Slider::new(&mut config.stale_frame_num, 0..=256).text("Stale frames"));
        ui.add(egui::Slider::new(&mut config.roughness_min, 0.0..=1.0).text("Min roughness"));
        ui.add(
            egui::Slider::new(&mut config.radiance_scale, 1.0..=1.0e4)
                .logarithmic(true)
                .text("Radiance scale"),
        );
    });

    // Debug view — painted by the SHARC debug compute pass into
    // occlusionTexture, composited over the frame by tonemap. While a
    // mode is active the shadow pass is skipped.
    ui.collapsing("Debug view", |ui| {
        ui.radio_value(&mut debug_state.mode, SharcDebugMode::Off, "Off");
        ui.radio_value(
            &mut debug_state.mode,
            SharcDebugMode::ColoredHash,
            "Colored hash (grid cells)",
        );
        ui.radio_value(
            &mut debug_state.mode,
            SharcDebugMode::CachedRadiance,
            "Cached radiance",
        );
        ui.radio_value(
            &mut debug_state.mode,
            SharcDebugMode::HashCollisions,
            "Hash collisions",
        );
        ui.radio_value(
            &mut debug_state.mode,
            SharcDebugMode::HashOccupancy,
            "Hash occupancy",
        );
        ui.radio_value(
            &mut debug_state.mode,
            SharcDebugMode::CandidatePool,
            "Candidate pool (splat)",
        );
        ui.add(egui::Slider::new(&mut debug_state.brightness, 0.0..=10.0).text("Brightness"));
    });
}
