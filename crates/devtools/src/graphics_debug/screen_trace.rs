//! Debug toggles of the "Screen trace" stage.

use bevy::prelude::*;
use dust_pbr::HdrRenderTarget;
use dust_pbr::screen_trace::{ScreenTraceConfig, ScreenTraceDebugMode, ScreenTraceState};
use dust_ui::BevyHandle;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window};
use gpui_component::accordion::Accordion;
use gpui_component::checkbox::Checkbox;
use gpui_component::radio::RadioGroup;
use gpui_component::{Disableable as _, v_flex};

use crate::widgets::NumberSlider;
use crate::{read, watch};

pub(super) fn screen_trace_config(config: Res<ScreenTraceConfig>) -> ScreenTraceConfig {
    config.clone()
}

/// `ScreenTraceState::hardware_rays`, if the state exists, and the render
/// extent; sent when either changed.
pub(super) fn hardware_rays(
    state: Option<Res<ScreenTraceState>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
) -> Option<(Option<Option<u32>>, Option<UVec2>)> {
    let changed = state.as_ref().is_some_and(|state| state.is_changed())
        || hdr_target.as_ref().is_some_and(|hdr| hdr.is_changed());
    changed.then(|| {
        (
            state.map(|state| state.hardware_rays),
            hdr_target.map(|hdr| hdr.render_extent),
        )
    })
}

const DEBUG_MODES: [(ScreenTraceDebugMode, &str); 4] = [
    (ScreenTraceDebugMode::Off, "Off"),
    (
        ScreenTraceDebugMode::TraceTypes,
        "Trace types (green hit, yellow rejected, orange uncertain, red miss, blue skipped)",
    ),
    (ScreenTraceDebugMode::HzbMip, "HZB mip"),
    (ScreenTraceDebugMode::PrevColor, "Previous frame color"),
];

pub(super) struct ScreenTracePage {
    /// The settings, as read when the page was shown and edited since.
    config: Option<ScreenTraceConfig>,
    /// `ScreenTraceState::hardware_rays`, if the state exists.
    hardware_rays: Option<Option<u32>>,
    render_extent: Option<UVec2>,
    sliders: Vec<(NumberSlider, fn(&ScreenTraceConfig) -> f32)>,
    debug_mip: NumberSlider,
    debug_scale: NumberSlider,
    debug_open: bool,
}

/// Applies an edit of the [`ScreenTraceConfig`] to the page's copy, and sends
/// it.
fn config(
    this: &mut ScreenTracePage,
    cx: &mut Context<ScreenTracePage>,
    edit: impl Fn(&mut ScreenTraceConfig) + Send + 'static,
) {
    if let Some(config) = &mut this.config {
        edit(config);
        cx.notify();
    }
    cx.global::<BevyHandle>()
        .update(move |world| edit(&mut world.resource_mut::<ScreenTraceConfig>()));
}

impl ScreenTracePage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut slider = |label: &'static str,
                          range: std::ops::RangeInclusive<f32>,
                          step: f32,
                          get: fn(&ScreenTraceConfig) -> f32,
                          set: fn(&mut ScreenTraceConfig, f32)| {
            let slider =
                NumberSlider::new(label, range, step, false, window, cx, move |this, v, cx| {
                    config(this, cx, move |c| set(c, v))
                });
            (slider, get)
        };
        let sliders = vec![
            slider(
                "Max iterations",
                1.0..=128.0,
                1.0,
                |c| c.max_iterations as f32,
                |c, v| c.max_iterations = v as u32,
            ),
            slider(
                "Relative thickness",
                0.0..=0.2,
                0.001,
                |c| c.relative_depth_thickness,
                |c, v| c.relative_depth_thickness = v,
            ),
            slider(
                "Min thickness (world)",
                0.0..=1.0,
                0.01,
                |c| c.min_world_thickness,
                |c, v| c.min_world_thickness = v,
            ),
            slider(
                "History depth thickness",
                0.0..=0.05,
                0.001,
                |c| c.history_depth_thickness,
                |c, v| c.history_depth_thickness = v,
            ),
            slider(
                "Certainty steps",
                0.0..=16.0,
                1.0,
                |c| c.num_thickness_steps as f32,
                |c, v| c.num_thickness_steps = v as u32,
            ),
            slider(
                "Pullback bias",
                0.0..=2.0,
                0.01,
                |c| c.pullback_bias,
                |c, v| c.pullback_bias = v,
            ),
            slider(
                "Min wave occupancy (0 = off)",
                0.0..=32.0,
                1.0,
                |c| c.minimum_occupancy as f32,
                |c, v| c.minimum_occupancy = v as u32,
            ),
        ];
        ScreenTracePage {
            config: None,
            hardware_rays: None,
            render_extent: None,
            sliders,
            debug_mip: NumberSlider::new(
                "Mip",
                0.0..=11.0,
                1.0,
                false,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.debug_mip = v as u32),
            ),
            debug_scale: NumberSlider::new(
                "Color scale",
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
}

impl ScreenTracePage {
    /// Shows the settings read from the world.
    fn show(&mut self, config: ScreenTraceConfig, window: &mut Window, cx: &mut gpui::App) {
        for (slider, get) in &self.sliders {
            slider.set(get(&config), window, cx);
        }
        self.debug_mip.set(config.debug_mip as f32, window, cx);
        self.debug_scale.set(config.debug_scale, window, cx);
        self.config = Some(config);
    }
}

impl Render for ScreenTracePage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        read(
            window,
            cx,
            screen_trace_config,
            |this, config, window, cx| this.show(config, window, cx),
        );
        watch(window, cx, hardware_rays, |this, (rays, extent), _, _| {
            this.hardware_rays = rays;
            this.render_extent = extent;
        });
        let Some(cfg) = &self.config else {
            return v_flex();
        };
        let hardware_rays = self
            .hardware_rays
            .map(|rays| match (rays, self.render_extent) {
                _ if cfg.enabled && cfg.screen_only => {
                    "Hardware rays: none (misses take the sky)".to_owned()
                }
                (Some(rays), Some(extent)) if cfg.enabled => {
                    let pixels = extent.x * extent.y;
                    format!(
                        "Hardware rays: {rays} ({:.1}% of pixels)",
                        100.0 * rays as f32 / pixels.max(1) as f32
                    )
                }
                _ => "Hardware rays: every surface pixel".to_owned(),
            });
        let mode = DEBUG_MODES
            .iter()
            .position(|&(mode, _)| mode == cfg.debug_mode);
        let debug_view = v_flex()
            .gap_1()
            .child(
                RadioGroup::vertical("screen-trace-debug-mode")
                    .children(DEBUG_MODES.map(|(_, label)| label))
                    .selected_index(mode)
                    .on_click(cx.listener(|this, index: &usize, _, cx| {
                        let mode = DEBUG_MODES[*index].0;
                        config(this, cx, move |c| c.debug_mode = mode);
                    })),
            )
            .child(self.debug_mip.render(cx))
            .child(self.debug_scale.render(cx));
        let debug_open = self.debug_open;
        v_flex()
            .gap_1()
            .child(
                Checkbox::new("screen-trace-enabled")
                    .label("Screen-space gather trace")
                    .checked(cfg.enabled)
                    .on_click(cx.listener(|this, enabled: &bool, _, cx| {
                        let enabled = *enabled;
                        config(this, cx, move |c| c.enabled = enabled);
                    })),
            )
            .child(
                Checkbox::new("screen-trace-screen-only")
                    .label("Screen only (no hardware rays, no SHARC)")
                    .checked(cfg.screen_only)
                    .disabled(!cfg.enabled)
                    .on_click(cx.listener(|this, screen_only: &bool, _, cx| {
                        let screen_only = *screen_only;
                        config(this, cx, move |c| c.screen_only = screen_only);
                    })),
            )
            .children(hardware_rays)
            .children(self.sliders.iter().map(|(slider, _)| slider.render(cx)))
            .child(
                Accordion::new("screen-trace-sections")
                    .item(|item| item.title("Debug view").open(debug_open).child(debug_view))
                    .on_toggle_click(cx.listener(|this, open: &[usize], _, cx| {
                        this.debug_open = !open.is_empty();
                        cx.notify();
                    })),
            )
    }
}
