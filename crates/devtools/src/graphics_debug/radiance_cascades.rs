//! Debug toggles of the "Radiance cascades" stage.

use bevy::prelude::*;
use dust_pbr::radiance_cascades::{
    CASCADES, CacheStats, RadianceCascadesConfig, RadianceCascadesDebugMode, RadianceCascadesState,
};
use dust_ui::BevyChannel;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window, px};
use gpui_component::accordion::Accordion;
use gpui_component::checkbox::Checkbox;
use gpui_component::radio::RadioGroup;
use gpui_component::{Disableable as _, v_flex};

use crate::widgets::{NumberSlider, RADIO_LABEL_OVERHANG};
use crate::{read, watch};

fn radiance_cascades_config(config: Res<RadianceCascadesConfig>) -> RadianceCascadesConfig {
    config.clone()
}

type Stats = (Option<CacheStats>, Option<CacheStats>);

/// The caches' statistics, when they changed.
fn stats(state: Option<Res<RadianceCascadesState>>, mut last: Local<Stats>) -> Option<Stats> {
    let stats = state.map_or((None, None), |state| state.stats());
    (*last != stats).then(|| {
        *last = stats;
        stats
    })
}

/// A cache's statistics as a line of text.
fn stats_text(name: &str, stats: &CacheStats) -> String {
    let mut text = format!("{name}: {} rays; probes", stats.rays);
    for cascade in 0..CASCADES {
        text += &format!(
            " c{cascade} {}/{}",
            stats.probes[cascade], stats.capacity[cascade]
        );
    }
    text
}

const DEBUG_MODES: [(RadianceCascadesDebugMode, &str); 6] = [
    (RadianceCascadesDebugMode::Off, "Off"),
    (
        RadianceCascadesDebugMode::Irradiance,
        "Incoming light (before albedo)",
    ),
    (
        RadianceCascadesDebugMode::Probes,
        "Cascade 0 cells (white: no probe)",
    ),
    (RadianceCascadesDebugMode::Lod, "Levels of detail"),
    (
        RadianceCascadesDebugMode::Coverage,
        "Hemisphere known to the probes",
    ),
    (
        RadianceCascadesDebugMode::Bounce,
        "Secondary cache's incoming light",
    ),
];

pub(super) struct RadianceCascadesPage {
    /// The settings, as read when the page was shown and edited since.
    config: Option<RadianceCascadesConfig>,
    stats: Stats,
    sliders: Vec<(NumberSlider, fn(&RadianceCascadesConfig) -> f32)>,
    debug_scale: NumberSlider,
    debug_open: bool,
}

/// Applies an edit of the [`RadianceCascadesConfig`] to the page's copy, and
/// sends it.
fn config(
    this: &mut RadianceCascadesPage,
    cx: &mut Context<RadianceCascadesPage>,
    edit: impl Fn(&mut RadianceCascadesConfig) + Send + 'static,
) {
    if let Some(config) = &mut this.config {
        edit(config);
        cx.notify();
    }
    cx.global::<BevyChannel>()
        .update(move |world| edit(&mut world.resource_mut::<RadianceCascadesConfig>()));
}

impl RadianceCascadesPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut slider = |label: &'static str,
                          range: std::ops::RangeInclusive<f32>,
                          step: f32,
                          logarithmic: bool,
                          get: fn(&RadianceCascadesConfig) -> f32,
                          set: fn(&mut RadianceCascadesConfig, f32)| {
            let slider = NumberSlider::new(
                label,
                range,
                step,
                logarithmic,
                window,
                cx,
                move |this, v, cx| config(this, cx, move |c| set(c, v)),
            );
            (slider, get)
        };
        let sliders = vec![
            slider(
                "Probe spacing (world)",
                0.03..=2.0,
                0.001,
                true,
                |c| c.probe_spacing,
                |c, v| c.probe_spacing = v,
            ),
            slider(
                "LOD 0 distance (world)",
                1.0..=256.0,
                0.1,
                true,
                |c| c.lod_distance,
                |c, v| c.lod_distance = v,
            ),
            slider(
                "Ray length (probe spacings)",
                0.25..=16.0,
                0.01,
                true,
                |c| c.ray_length_scale,
                |c, v| c.ray_length_scale = v,
            ),
            slider(
                "Rays per probe",
                4.0..=2048.0,
                1.0,
                true,
                |c| c.rays_per_probe,
                |c, v| c.rays_per_probe = v,
            ),
            slider(
                "Temporal alpha (1 = off)",
                0.005..=1.0,
                0.001,
                true,
                |c| c.temporal_alpha,
                |c, v| c.temporal_alpha = v,
            ),
            slider(
                "Max probe age (frames)",
                0.0..=64.0,
                1.0,
                false,
                |c| c.max_probe_age as f32,
                |c, v| c.max_probe_age = v as u32,
            ),
            slider(
                "Bounce ray fraction",
                0.01..=1.0,
                0.01,
                true,
                |c| c.bounce_ray_fraction,
                |c, v| c.bounce_ray_fraction = v,
            ),
            slider(
                "Secondary LOD offset",
                0.0..=4.0,
                1.0,
                false,
                |c| c.secondary_lod_offset as f32,
                |c, v| c.secondary_lod_offset = v as u32,
            ),
        ];
        RadianceCascadesPage {
            config: None,
            stats: (None, None),
            sliders,
            debug_scale: NumberSlider::new(
                "Light scale",
                0.0..=10.0,
                0.01,
                false,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.debug_scale = v),
            ),
            debug_open: true,
        }
    }

    /// Shows the settings read from the world.
    fn show(&mut self, config: RadianceCascadesConfig, window: &mut Window, cx: &mut gpui::App) {
        for (slider, get) in &self.sliders {
            slider.set(get(&config), window, cx);
        }
        self.debug_scale.set(config.debug_scale, window, cx);
        self.config = Some(config);
    }
}

impl Render for RadianceCascadesPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        read(
            window,
            cx,
            radiance_cascades_config,
            |this, config, window, cx| this.show(config, window, cx),
        );
        watch(window, cx, stats, |this, stats, _, _| this.stats = stats);
        let Some(cfg) = &self.config else {
            return v_flex();
        };
        let mode = DEBUG_MODES
            .iter()
            .position(|&(mode, _)| mode == cfg.debug_mode);
        let debug_view = v_flex()
            .gap_1()
            .child(
                RadioGroup::vertical("radiance-cascades-debug-mode")
                    .pr(px(RADIO_LABEL_OVERHANG))
                    .children(DEBUG_MODES.map(|(_, label)| label))
                    .selected_index(mode)
                    .on_click(cx.listener(|this, index: &usize, _, cx| {
                        let mode = DEBUG_MODES[*index].0;
                        config(this, cx, move |c| c.debug_mode = mode);
                    })),
            )
            .child(self.debug_scale.render(cx));
        let debug_open = self.debug_open;
        v_flex()
            .gap_1()
            .child(
                Checkbox::new("radiance-cascades-enabled")
                    .label("Radiance cascades")
                    .checked(cfg.enabled)
                    .on_click(cx.listener(|this, enabled: &bool, _, cx| {
                        let enabled = *enabled;
                        config(this, cx, move |c| c.enabled = enabled);
                    })),
            )
            .child(
                Checkbox::new("radiance-cascades-multibounce")
                    .label("Multiple bounces (secondary cache)")
                    .checked(cfg.multibounce)
                    .disabled(!cfg.enabled)
                    .on_click(cx.listener(|this, multibounce: &bool, _, cx| {
                        let multibounce = *multibounce;
                        config(this, cx, move |c| c.multibounce = multibounce);
                    })),
            )
            .children(self.stats.0.map(|stats| stats_text("Primary", &stats)))
            .children(self.stats.1.map(|stats| stats_text("Secondary", &stats)))
            .children(self.sliders.iter().map(|(slider, _)| slider.render(cx)))
            .child(
                Accordion::new("radiance-cascades-sections")
                    .item(|item| item.title("Debug view").open(debug_open).child(debug_view))
                    .on_toggle_click(cx.listener(|this, open: &[usize], _, cx| {
                        this.debug_open = !open.is_empty();
                        cx.notify();
                    })),
            )
    }
}
