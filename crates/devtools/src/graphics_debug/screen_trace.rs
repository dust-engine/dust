//! Debug toggles of the "Screen trace" stage.

use bevy::prelude::*;
use dust_pbr::HdrRenderTarget;
use dust_pbr::screen_trace::{ScreenTraceConfig, ScreenTraceDebugMode, ScreenTraceState};
use pumicite_egui::egui;

pub(super) fn screen_trace_ui(
    InMut(ui): InMut<egui::Ui>,
    mut config: ResMut<ScreenTraceConfig>,
    state: Option<Res<ScreenTraceState>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
) {
    ui.checkbox(&mut config.enabled, "Screen-space gather trace");
    ui.add_enabled(
        config.enabled,
        egui::Checkbox::new(
            &mut config.screen_only,
            "Screen only (no hardware rays, no SHARC)",
        ),
    );
    if let Some(state) = state.as_deref() {
        match (state.hardware_rays, hdr_target.as_deref()) {
            _ if config.enabled && config.screen_only => {
                ui.label("Hardware rays: none (misses take the sky)");
            }
            (Some(rays), Some(hdr)) if config.enabled => {
                let pixels = hdr.render_extent.x * hdr.render_extent.y;
                ui.label(format!(
                    "Hardware rays: {rays} ({:.1}% of pixels)",
                    100.0 * rays as f32 / pixels.max(1) as f32
                ));
            }
            _ => {
                ui.label("Hardware rays: every surface pixel");
            }
        }
    }
    ui.separator();
    ui.add(egui::Slider::new(&mut config.max_iterations, 1..=128).text("Max iterations"));
    ui.add(
        egui::Slider::new(&mut config.relative_depth_thickness, 0.0..=0.2)
            .text("Relative thickness"),
    );
    ui.add(
        egui::Slider::new(&mut config.min_world_thickness, 0.0..=1.0).text("Min thickness (world)"),
    );
    ui.add(
        egui::Slider::new(&mut config.history_depth_thickness, 0.0..=0.05)
            .text("History depth thickness"),
    );
    ui.add(egui::Slider::new(&mut config.num_thickness_steps, 0..=16).text("Certainty steps"));
    ui.add(egui::Slider::new(&mut config.pullback_bias, 0.0..=2.0).text("Pullback bias"));
    ui.add(
        egui::Slider::new(&mut config.minimum_occupancy, 0..=32)
            .text("Min wave occupancy (0 = off)"),
    );
    ui.collapsing("Debug view", |ui| {
        ui.radio_value(&mut config.debug_mode, ScreenTraceDebugMode::Off, "Off");
        ui.radio_value(
            &mut config.debug_mode,
            ScreenTraceDebugMode::TraceTypes,
            "Trace types (green hit, yellow rejected, orange uncertain, red miss, blue skipped)",
        );
        ui.radio_value(
            &mut config.debug_mode,
            ScreenTraceDebugMode::HzbMip,
            "HZB mip",
        );
        ui.radio_value(
            &mut config.debug_mode,
            ScreenTraceDebugMode::PrevColor,
            "Previous frame color",
        );
        ui.add(egui::Slider::new(&mut config.debug_mip, 0..=11).text("Mip"));
        ui.add(egui::Slider::new(&mut config.debug_scale, 0.0..=10.0).text("Color scale"));
    });
}
