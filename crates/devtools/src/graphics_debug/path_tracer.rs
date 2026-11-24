//! The "Path tracer" page: the path tracer's settings, and its radiance cache's.

use bevy::prelude::*;
use dust_pbr::RenderQualitySettings;
use dust_pbr::sharc::{SharcSettings, SharcView};
use dust_pbr::upscaler::ResetHistory;
use dust_ui::BevyChannel;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window, div, px};
use gpui_component::checkbox::Checkbox;
use gpui_component::radio::RadioGroup;
use gpui_component::{ActiveTheme as _, v_flex};

use crate::widgets::{NumberSlider, RADIO_LABEL_OVERHANG, heading};
use crate::{read, watch};

const VIEWS: [(SharcView, &str); 5] = [
    (SharcView::Off, "Off"),
    (SharcView::Grid, "Grid cells"),
    (SharcView::Radiance, "Cached radiance"),
    (SharcView::Occupancy, "Hash map occupancy (green: taken)"),
    (
        SharcView::Bounces,
        "Bounces (green: the first ended in the cache or the sky)",
    ),
];

pub(super) struct PathTracerPage {
    path_tracing: Option<bool>,
    max_bounces: NumberSlider,
    sharc: Option<SharcSettings>,
    sharc_sliders: Vec<(NumberSlider, fn(&SharcSettings) -> f32)>,
}

/// Applies an edit of the [`SharcSettings`] to the page's copy, and sends it.
fn edit_sharc(
    this: &mut PathTracerPage,
    cx: &mut Context<PathTracerPage>,
    edit: impl Fn(&mut SharcSettings) + Send + 'static,
) {
    if let Some(sharc) = &mut this.sharc {
        edit(sharc);
        cx.notify();
    }
    cx.global::<BevyChannel>().update(move |world| {
        edit(&mut world.resource_mut::<SharcSettings>());
        world.write_message(ResetHistory);
    });
}

impl PathTracerPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let max_bounces = NumberSlider::new(
            "Max bounces",
            0.0..=32.0,
            1.0,
            false,
            window,
            cx,
            |_, bounces, cx| {
                cx.global::<BevyChannel>().update(move |world| {
                    world.resource_mut::<RenderQualitySettings>().max_bounces = bounces as u32;
                    world.write_message(ResetHistory);
                });
            },
        );
        let mut slider = |label: &'static str,
                          range: std::ops::RangeInclusive<f32>,
                          get: fn(&SharcSettings) -> f32,
                          set: fn(&mut SharcSettings, f32)| {
            let slider =
                NumberSlider::new(label, range, 1.0, false, window, cx, move |this, v, cx| {
                    edit_sharc(this, cx, move |sharc| set(sharc, v))
                });
            (slider, get)
        };
        let sharc_sliders = vec![
            slider(
                "A cell per voxel up to (units)",
                1.0..=64.0,
                |sharc| sharc.cell_distance,
                |sharc, v| sharc.cell_distance = v,
            ),
            slider(
                "Update block",
                1.0..=10.0,
                |sharc| sharc.update_block as f32,
                |sharc, v| sharc.update_block = v as u32,
            ),
            slider(
                "Accumulation frames",
                1.0..=256.0,
                |sharc| sharc.accumulation_frames as f32,
                |sharc, v| sharc.accumulation_frames = v as u32,
            ),
            slider(
                "Stale frames",
                8.0..=256.0,
                |sharc| sharc.stale_frames as f32,
                |sharc, v| sharc.stale_frames = v as u32,
            ),
        ];
        PathTracerPage {
            path_tracing: None,
            max_bounces,
            sharc: None,
            sharc_sliders,
        }
    }
}

impl Render for PathTracerPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // The console can change them too.
        watch(
            window,
            cx,
            |settings: Res<RenderQualitySettings>| {
                settings.is_changed().then_some(settings.path_tracing)
            },
            |this, path_tracing, _, _| this.path_tracing = Some(path_tracing),
        );
        watch(
            window,
            cx,
            |sharc: Res<SharcSettings>| sharc.is_changed().then_some(*sharc),
            |this, sharc, _, _| this.sharc = Some(sharc),
        );
        read(
            window,
            cx,
            |settings: Res<RenderQualitySettings>, sharc: Res<SharcSettings>| {
                (settings.max_bounces, *sharc)
            },
            |this, (bounces, sharc), window, cx| {
                this.max_bounces.set(bounces as f32, window, cx);
                for (slider, get) in &this.sharc_sliders {
                    slider.set(get(&sharc), window, cx);
                }
            },
        );
        let (Some(path_tracing), Some(sharc)) = (self.path_tracing, self.sharc) else {
            return v_flex();
        };
        v_flex()
            .gap_1()
            .child(
                Checkbox::new("path-tracer-enabled")
                    .label("Path trace")
                    .checked(path_tracing)
                    .on_click(cx.listener(|this, path_tracing: &bool, _, cx| {
                        let path_tracing = *path_tracing;
                        this.path_tracing = Some(path_tracing);
                        cx.notify();
                        cx.global::<BevyChannel>().update(move |world| {
                            world.resource_mut::<RenderQualitySettings>().path_tracing =
                                path_tracing;
                            world.write_message(ResetHistory);
                        });
                    })),
            )
            .child(self.max_bounces.render(cx))
            .child(
                div()
                    .text_color(cx.theme().muted_foreground)
                    .child("0 bounces: direct light only."),
            )
            .child(heading("Radiance cache (SHaRC)"))
            .child(
                Checkbox::new("sharc-enabled")
                    .label("End paths in the cache (off: the reference)")
                    .checked(sharc.enabled)
                    .on_click(cx.listener(|this, enabled: &bool, _, cx| {
                        let enabled = *enabled;
                        edit_sharc(this, cx, move |sharc| sharc.enabled = enabled);
                    })),
            )
            .children(
                self.sharc_sliders
                    .iter()
                    .map(|(slider, _)| slider.render(cx)),
            )
            .child(heading("Debug view"))
            .child(
                RadioGroup::vertical("sharc-view")
                    .pr(px(RADIO_LABEL_OVERHANG))
                    .children(VIEWS.map(|(_, label)| label))
                    .selected_index(VIEWS.iter().position(|&(view, _)| view == sharc.view))
                    .on_click(cx.listener(|this, index: &usize, _, cx| {
                        let view = VIEWS[*index].0;
                        if let Some(sharc) = &mut this.sharc {
                            sharc.view = view;
                            cx.notify();
                        }
                        // It's drawn over the frame, so the history stays.
                        cx.global::<BevyChannel>().update(move |world| {
                            world.resource_mut::<SharcSettings>().view = view;
                        });
                    })),
            )
    }
}
