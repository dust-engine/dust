//! Debug toggles of the "Sky atmosphere" stage.

use bevy::prelude::*;
use dust_pbr::sky::AtmosphereState;
use dust_ui::BevyChannel;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window};
use gpui_component::button::Button;
use gpui_component::{Sizable as _, v_flex};

use crate::read;
use crate::widgets::{NumberSlider, heading};

pub(super) fn sky_snapshot(atmosphere: Res<AtmosphereState>) -> AtmosphereState {
    atmosphere.clone()
}

const BASE_RAYLEIGH: [f32; 3] = [0.005802, 0.013558, 0.033100];
const BASE_MIE_SCATTERING: f32 = 0.003996;
const BASE_MIE_EXTINCTION: f32 = 0.004440;

pub(super) struct SkyPage {
    elevation: NumberSlider,
    azimuth: NumberSlider,
    rayleigh_factor: NumberSlider,
    mie_factor: NumberSlider,
    mie_phase_g: NumberSlider,
    ground_albedo: NumberSlider,
}

/// Sends an edit of the [`AtmosphereState`], marking it changed only if the
/// edit changed it.
fn atmosphere(cx: &mut gpui::App, edit: impl FnOnce(&mut AtmosphereState) + Send + 'static) {
    cx.global::<BevyChannel>().update(move |world| {
        let mut atmosphere = world.resource_mut::<AtmosphereState>();
        let mut next = atmosphere.clone();
        edit(&mut next);
        atmosphere.set_if_neq(next);
    });
}

impl SkyPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        SkyPage {
            elevation: NumberSlider::new(
                "Elevation",
                -0.5..=1.57,
                0.01,
                false,
                window,
                cx,
                |_, v, cx| {
                    atmosphere(cx, move |a| {
                        a.sun_elevation = v;
                        a.sync_sun_direction();
                    })
                },
            ),
            azimuth: NumberSlider::new(
                "Azimuth",
                -3.14..=3.14,
                0.01,
                false,
                window,
                cx,
                |_, v, cx| {
                    atmosphere(cx, move |a| {
                        a.sun_azimuth = v;
                        a.sync_sun_direction();
                    })
                },
            ),
            rayleigh_factor: NumberSlider::new(
                "Factor",
                0.0..=2.0,
                0.01,
                false,
                window,
                cx,
                |_, v, cx| {
                    atmosphere(cx, move |a| {
                        a.params.rayleigh_scattering = BASE_RAYLEIGH.map(|base| base * v)
                    })
                },
            ),
            mie_factor: NumberSlider::new(
                "Factor",
                0.0..=2.0,
                0.01,
                false,
                window,
                cx,
                |_, v, cx| {
                    atmosphere(cx, move |a| {
                        a.params.mie_scattering = [BASE_MIE_SCATTERING * v; 3];
                        a.params.mie_extinction = [BASE_MIE_EXTINCTION * v; 3];
                    })
                },
            ),
            mie_phase_g: NumberSlider::new(
                "Phase G",
                0.0..=0.999,
                0.001,
                false,
                window,
                cx,
                |_, v, cx| atmosphere(cx, move |a| a.params.mie_phase_g = v),
            ),
            ground_albedo: NumberSlider::new(
                "Albedo",
                0.0..=1.0,
                0.01,
                false,
                window,
                cx,
                |_, v, cx| atmosphere(cx, move |a| a.params.ground_albedo = [v; 3]),
            ),
        }
    }
}

impl SkyPage {
    /// Moves the sliders to `state`.
    fn show(&self, state: &AtmosphereState, window: &mut Window, cx: &mut gpui::App) {
        self.elevation.set(state.sun_elevation, window, cx);
        self.azimuth.set(state.sun_azimuth, window, cx);
        self.rayleigh_factor.set(
            state.params.rayleigh_scattering[0] / BASE_RAYLEIGH[0],
            window,
            cx,
        );
        self.mie_factor.set(
            state.params.mie_scattering[0] / BASE_MIE_SCATTERING,
            window,
            cx,
        );
        self.mie_phase_g.set(state.params.mie_phase_g, window, cx);
        self.ground_albedo
            .set(state.params.ground_albedo[0], window, cx);
    }
}

impl Render for SkyPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        read(window, cx, sky_snapshot, |this, state, window, cx| {
            this.show(&state, window, cx)
        });
        v_flex()
            .gap_1()
            .child(heading("Sun"))
            .child(self.elevation.render(cx))
            .child(self.azimuth.render(cx))
            .child(heading("Rayleigh Scattering"))
            .child(self.rayleigh_factor.render(cx))
            .child(heading("Mie Scattering"))
            .child(self.mie_factor.render(cx))
            .child(self.mie_phase_g.render(cx))
            .child(heading("Ground"))
            .child(self.ground_albedo.render(cx))
            .child(
                Button::new("sky-reset")
                    .label("Reset to Earth Defaults")
                    .small()
                    .mt_2()
                    .self_start()
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.show(&AtmosphereState::default(), window, cx);
                        atmosphere(cx, |a| *a = AtmosphereState::default())
                    })),
            )
    }
}
