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
use gpui_component::v_flex;

use crate::widgets::{NumberSlider, RADIO_LABEL_OVERHANG};
use crate::{read, watch};

fn radiance_cascades_config(config: Res<RadianceCascadesConfig>) -> RadianceCascadesConfig {
    config.clone()
}

/// The cache's statistics, when they changed.
fn stats(
    state: Option<Res<RadianceCascadesState>>,
    mut last: Local<Option<CacheStats>>,
) -> Option<Option<CacheStats>> {
    let stats = state.and_then(|state| state.stats());
    (*last != stats).then(|| {
        *last = stats;
        stats
    })
}

/// The cache's probe counts as a line of text, short enough for the panel:
/// unwrapped text sets the page's width.
fn stats_line(stats: &CacheStats) -> String {
    let fullest = (0..CASCADES)
        .map(|cascade| stats.probes[cascade] as f32 / stats.capacity[cascade] as f32)
        .fold(0.0, f32::max);
    let [c0, c1, c2, c3] = stats.probes;
    format!("Probes: {c0} {c1} {c2} {c3} ({:.0}% full)", fullest * 100.0)
}

const DEBUG_MODES: [(RadianceCascadesDebugMode, &str); 3] = [
    (RadianceCascadesDebugMode::Off, "Off"),
    (
        RadianceCascadesDebugMode::Probes,
        "Cascade 0 cells (white: no probe)",
    ),
    (RadianceCascadesDebugMode::Lod, "Levels of detail"),
];

pub(super) struct RadianceCascadesPage {
    /// The settings, as read when the page was shown and edited since.
    config: Option<RadianceCascadesConfig>,
    /// The latest statistics, while the cascades are enabled.
    stats: Option<CacheStats>,
    /// Each slider, with the setting it shows.
    sliders: Vec<(NumberSlider, fn(&RadianceCascadesConfig) -> f32)>,
    /// Whether the "Debug view" section is expanded.
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
        ];
        RadianceCascadesPage {
            config: None,
            stats: None,
            sliders,
            debug_open: true,
        }
    }

    /// Shows the settings read from the world.
    fn show(&mut self, config: RadianceCascadesConfig, window: &mut Window, cx: &mut gpui::App) {
        for (slider, get) in &self.sliders {
            slider.set(get(&config), window, cx);
        }
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
        let debug_view = v_flex().gap_1().child(
            RadioGroup::vertical("radiance-cascades-debug-mode")
                .pr(px(RADIO_LABEL_OVERHANG))
                .children(DEBUG_MODES.map(|(_, label)| label))
                .selected_index(mode)
                .on_click(cx.listener(|this, index: &usize, _, cx| {
                    let mode = DEBUG_MODES[*index].0;
                    config(this, cx, move |c| c.debug_mode = mode);
                })),
        );
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
            .children(self.stats.iter().map(stats_line))
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
