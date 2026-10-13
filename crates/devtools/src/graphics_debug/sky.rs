//! Debug toggles of the "Sky atmosphere" stage, and previews of its LUTs.

use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use dust_pbr::OccludingRenderPass;
use dust_pbr::sky::{AtmosphereLUTs, AtmosphereState, SkyAtmosphereLUTRenderSet};
use pumicite_egui::{EguiPostUpdateSet, egui, user_texture_id};

/// The largest a LUT texel is drawn, in egui points, so that the small LUTs
/// don't fill the whole panel.
const MAX_POINTS_PER_TEXEL: f32 = 4.0;

pub(super) fn build(app: &mut App) {
    app.init_resource::<LutPreviews>();
    app.add_systems(
        PostUpdate,
        prepare_lut_textures
            .in_set(DefaultRenderSet)
            .after(EguiPostUpdateSet::EndPass)
            .after(SkyAtmosphereLUTRenderSet)
            .before(OccludingRenderPass)
            .run_if(resource_changed::<LutPreviews>),
    );
}

/// The values drawn as black and white in each LUT's preview. [`egui_ui`]
/// marks this changed on every frame it draws them, which runs
/// [`prepare_lut_textures`].
#[derive(Resource)]
pub(super) struct LutPreviews {
    transmittance: (f32, f32),
    multi_scattering: (f32, f32),
    sky_view: (f32, f32),
}

impl Default for LutPreviews {
    fn default() -> Self {
        Self {
            transmittance: (0.0, 1.0),
            multi_scattering: (0.0, 1.0),
            sky_view: (0.0, 1.0),
        }
    }
}

/// Moves the LUTs into the layout egui samples them in, and makes the LUT
/// compute writes visible to the fragment shader, before egui's render pass
/// begins. Recording the read in each LUT's `ResourceState` also makes the
/// next recompute wait for egui's reads.
fn prepare_lut_textures(mut ctx: SubmissionState, mut luts: ResMut<AtmosphereLUTs>) {
    let luts = &mut *luts;
    ctx.record(|encoder| {
        for lut in [
            &mut luts.transmittance,
            &mut luts.multi_scattering,
            &mut luts.sky_view,
        ] {
            let view = encoder.lock(&lut.view, vk::PipelineStageFlags2::FRAGMENT_SHADER);
            encoder.use_image_resource(
                view,
                &mut lut.state,
                Access::FRAGMENT_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
        }
        encoder.emit_barriers();
    });
}

pub(super) fn egui_ui(
    InMut(ui): InMut<egui::Ui>,
    mut atmosphere: ResMut<AtmosphereState>,
    luts: Res<AtmosphereLUTs>,
    mut previews: ResMut<LutPreviews>,
) {
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

    let _ = atmosphere.set_if_neq(next);

    ui.separator();
    ui.heading("LUTs");
    if !luts.luts_initialized {
        ui.weak("Not computed yet.");
        return;
    }
    previews.set_changed();
    let previews = &mut *previews;
    for (name, lut, (min, max)) in [
        (
            "Transmittance",
            &luts.transmittance,
            &mut previews.transmittance,
        ),
        (
            "Multi-scattering",
            &luts.multi_scattering,
            &mut previews.multi_scattering,
        ),
        ("Sky view", &luts.sky_view, &mut previews.sky_view),
    ] {
        let extent = lut.view.extent();
        ui.label(format!("{name} ({}×{})", extent.x, extent.y));
        // A drag steps by 1% of the range, whatever the LUT's magnitude.
        let speed = ((*max - *min) * 0.01).max(1e-4);
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(min)
                    .speed(speed)
                    .range(f32::MIN..=*max)
                    .prefix("Min "),
            );
            ui.add(
                egui::DragValue::new(max)
                    .speed(speed)
                    .range(*min..=f32::MAX)
                    .prefix("Max "),
            );
        });
        let width = ui
            .available_width()
            .min(extent.x as f32 * MAX_POINTS_PER_TEXEL);
        ui.image((
            user_texture_id(lut.view.handle(), Some(*min..=*max)),
            egui::vec2(width, width * extent.y as f32 / extent.x as f32),
        ));
    }
}
