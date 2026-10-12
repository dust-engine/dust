//! Debug toggles of the "Sky atmosphere" stage.

use bevy::prelude::*;
use dust_pbr::sky::AtmosphereState;
use pumicite_egui::egui;

pub(super) fn egui_ui(InMut(ui): InMut<egui::Ui>, mut atmosphere: ResMut<AtmosphereState>) {
    let mut next = atmosphere.as_ref().clone();

    ui.heading("Sun");
    let elevation_changed = ui
        .add(egui::Slider::new(&mut next.sun_elevation, -0.5..=1.57).text("Elevation"))
        .changed();
    let azimuth_changed = ui
        .add(egui::Slider::new(&mut next.sun_azimuth, -3.14..=3.14).text("Azimuth"))
        .changed();
    if elevation_changed || azimuth_changed {
        next.sync_sun_direction();
    }

    ui.separator();
    ui.heading("Rayleigh Scattering");

    let base_rayleigh = [0.005802f32, 0.013558, 0.033100];
    let mut rayleigh_factor = next.params.rayleigh_scattering[0] / base_rayleigh[0];
    if ui
        .add(egui::Slider::new(&mut rayleigh_factor, 0.0..=2.0).text("Factor"))
        .changed()
    {
        next.params.rayleigh_scattering = [
            base_rayleigh[0] * rayleigh_factor,
            base_rayleigh[1] * rayleigh_factor,
            base_rayleigh[2] * rayleigh_factor,
        ];
    }

    ui.separator();
    ui.heading("Mie Scattering");

    let base_mie_scattering = 0.003996f32;
    let base_mie_extinction = 0.004440f32;
    let mut mie_factor = next.params.mie_scattering[0] / base_mie_scattering;
    if ui
        .add(egui::Slider::new(&mut mie_factor, 0.0..=2.0).text("Factor"))
        .changed()
    {
        next.params.mie_scattering = [base_mie_scattering * mie_factor; 3];
        next.params.mie_extinction = [base_mie_extinction * mie_factor; 3];
    }

    ui.add(egui::Slider::new(&mut next.params.mie_phase_g, 0.0..=0.999).text("Phase G"));

    ui.separator();
    ui.heading("Ground");

    let mut albedo = next.params.ground_albedo[0];
    if ui
        .add(egui::Slider::new(&mut albedo, 0.0..=1.0).text("Albedo"))
        .changed()
    {
        next.params.ground_albedo = [albedo; 3];
    }

    ui.separator();
    if ui.button("Reset to Earth Defaults").clicked() {
        next = AtmosphereState::default();
    }

    ui.separator();
    ui.label("Controls:");
    ui.label("WASD - Move camera");
    ui.label("Right mouse - Look around");
    ui.label("Arrow keys - Sun position");
    ui.label("Shift - Move faster");

    let _ = atmosphere.set_if_neq(next);
}
