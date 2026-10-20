//! Debug toggles of the "SHARC" stage.

use bevy::prelude::*;
use dust_pbr::sharc::{SharcConfig, SharcDebugMode, SharcDebugState};
use dust_ui::BevyHandle;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window};
use gpui_component::accordion::Accordion;
use gpui_component::button::Button;
use gpui_component::checkbox::Checkbox;
use gpui_component::radio::RadioGroup;
use gpui_component::{Sizable as _, h_flex, v_flex};

use crate::read;
use crate::widgets::{NumberSlider, heading};

pub(super) fn sharc_snapshot(
    config: Res<SharcConfig>,
    debug_state: Res<SharcDebugState>,
) -> (SharcConfig, SharcDebugState) {
    (*config, *debug_state)
}

const DEBUG_MODES: [(SharcDebugMode, &str); 6] = [
    (SharcDebugMode::Off, "Off"),
    (SharcDebugMode::ColoredHash, "Colored hash (grid cells)"),
    (SharcDebugMode::CachedRadiance, "Cached radiance"),
    (SharcDebugMode::HashCollisions, "Hash collisions"),
    (SharcDebugMode::HashOccupancy, "Hash occupancy"),
    (SharcDebugMode::CandidatePool, "Candidate pool (splat)"),
];

pub(super) struct SharcPage {
    /// The settings, as read when the page was shown and edited since.
    snapshot: Option<(SharcConfig, SharcDebugState)>,
    scene_scale: NumberSlider,
    pool_capacity: NumberSlider,
    accumulation_frames: NumberSlider,
    stale_frames: NumberSlider,
    roughness_min: NumberSlider,
    radiance_scale: NumberSlider,
    brightness: NumberSlider,
    /// Indices of the expanded sections.
    open: Vec<usize>,
}

/// Applies an edit of the [`SharcConfig`] to the page's copy, and sends it.
fn config(
    this: &mut SharcPage,
    cx: &mut Context<SharcPage>,
    edit: impl Fn(&mut SharcConfig) + Send + 'static,
) {
    if let Some((config, _)) = &mut this.snapshot {
        edit(config);
        cx.notify();
    }
    cx.global::<BevyHandle>()
        .update(move |world| edit(&mut world.resource_mut::<SharcConfig>()));
}

/// Applies an edit of the [`SharcDebugState`] to the page's copy, and sends it.
fn edit_debug_state(
    this: &mut SharcPage,
    cx: &mut Context<SharcPage>,
    edit: impl Fn(&mut SharcDebugState) + Send + 'static,
) {
    if let Some((_, debug_state)) = &mut this.snapshot {
        edit(debug_state);
        cx.notify();
    }
    cx.global::<BevyHandle>()
        .update(move |world| edit(&mut world.resource_mut::<SharcDebugState>()));
}

impl SharcPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        SharcPage {
            snapshot: None,
            // There is exactly one scene scale: the Update/Resolve passes and
            // the colored-hash overlay both key off it. The logarithm base and
            // level bias are compile-time constants (SHARC_GRID_*).
            scene_scale: NumberSlider::new(
                "Scene scale",
                1.0..=200.0,
                0.1,
                false,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.scene_scale = v),
            ),
            // Pool capacity ≡ Update ray budget. The pool buffers are sized to
            // SharcConfig::default().pool_capacity at startup, so the slider
            // stays at or under that — growing past it is a no-op without a
            // reallocation.
            pool_capacity: NumberSlider::new(
                "Pool capacity (Update rays / frame)",
                1024.0..=(1 << 16) as f32,
                1.0,
                true,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.pool_capacity = v as u32),
            ),
            accumulation_frames: NumberSlider::new(
                "Accumulation frames",
                0.0..=128.0,
                1.0,
                false,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.accumulation_frame_num = v as u32),
            ),
            stale_frames: NumberSlider::new(
                "Stale frames",
                0.0..=256.0,
                1.0,
                false,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.stale_frame_num = v as u32),
            ),
            roughness_min: NumberSlider::new(
                "Min roughness",
                0.0..=1.0,
                0.01,
                false,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.roughness_min = v),
            ),
            radiance_scale: NumberSlider::new(
                "Radiance scale",
                1.0..=1.0e4,
                1.0,
                true,
                window,
                cx,
                |this, v, cx| config(this, cx, move |c| c.radiance_scale = v),
            ),
            brightness: NumberSlider::new(
                "Brightness",
                0.0..=10.0,
                0.01,
                false,
                window,
                cx,
                |this, v, cx| edit_debug_state(this, cx, move |d| d.brightness = v),
            ),
            open: vec![0, 1],
        }
    }
}

impl SharcPage {
    /// Shows the settings read from the world.
    fn show(
        &mut self,
        snapshot: (SharcConfig, SharcDebugState),
        window: &mut Window,
        cx: &mut gpui::App,
    ) {
        let (config, debug_state) = &snapshot;
        self.scene_scale.set(config.scene_scale, window, cx);
        self.pool_capacity
            .set(config.pool_capacity as f32, window, cx);
        self.accumulation_frames
            .set(config.accumulation_frame_num as f32, window, cx);
        self.stale_frames
            .set(config.stale_frame_num as f32, window, cx);
        self.roughness_min.set(config.roughness_min, window, cx);
        self.radiance_scale.set(config.radiance_scale, window, cx);
        self.brightness.set(debug_state.brightness, window, cx);
        self.snapshot = Some(snapshot);
    }
}

impl Render for SharcPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        read(window, cx, sharc_snapshot, |this, snapshot, window, cx| {
            this.show(snapshot, window, cx)
        });
        let Some((sharc, debug_state)) = self.snapshot else {
            return v_flex();
        };
        let mode = DEBUG_MODES
            .iter()
            .position(|&(mode, _)| mode == debug_state.mode);
        let population = v_flex()
            .gap_1()
            .child(self.pool_capacity.render(cx))
            .child(self.accumulation_frames.render(cx))
            .child(self.stale_frames.render(cx))
            .child(self.roughness_min.render(cx))
            .child(self.radiance_scale.render(cx));
        // Painted by the SHARC debug passes into the debug overlay, composited
        // over the frame by tonemap. While a mode is active the shadow pass is
        // skipped.
        let debug_view = v_flex()
            .gap_1()
            .child(
                RadioGroup::vertical("sharc-debug-mode")
                    .children(DEBUG_MODES.map(|(_, label)| label))
                    .selected_index(mode)
                    .on_click(cx.listener(|this, index: &usize, _, cx| {
                        let mode = DEBUG_MODES[*index].0;
                        edit_debug_state(this, cx, move |d| d.mode = mode);
                    })),
            )
            .child(self.brightness.render(cx));
        let open = self.open.clone();
        v_flex()
            .gap_2()
            // Master controls — the two knobs reached for most often.
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Checkbox::new("sharc-enabled")
                            .label("Cache enabled")
                            .checked(sharc.enabled)
                            .on_click(cx.listener(|this, enabled: &bool, _, cx| {
                                let enabled = *enabled;
                                config(this, cx, move |c| c.enabled = enabled);
                            })),
                    )
                    .child(
                        Button::new("sharc-reset")
                            .label("Reset cache")
                            .small()
                            .on_click(cx.listener(|this, _, _, cx| {
                                config(this, cx, |c| c.reset_pending = true)
                            })),
                    ),
            )
            .child(heading("Grid"))
            .child(self.scene_scale.render(cx))
            .child(
                Accordion::new("sharc-sections")
                    .multiple(true)
                    .item(|item| {
                        item.title("Cache population")
                            .open(open.contains(&0))
                            .child(population)
                    })
                    .item(|item| {
                        item.title("Debug view")
                            .open(open.contains(&1))
                            .child(debug_view)
                    })
                    .on_toggle_click(cx.listener(|this, open: &[usize], _, cx| {
                        this.open = open.to_vec();
                        cx.notify();
                    })),
            )
    }
}
