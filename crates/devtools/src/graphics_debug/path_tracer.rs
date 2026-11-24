//! The "Path tracer" page: the reference path tracer's settings.

use bevy::prelude::*;
use dust_pbr::RenderQualitySettings;
use dust_pbr::upscaler::ResetHistory;
use dust_ui::BevyChannel;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window, div};
use gpui_component::checkbox::Checkbox;
use gpui_component::{ActiveTheme as _, v_flex};

use crate::widgets::NumberSlider;
use crate::{read, watch};

pub(super) struct PathTracerPage {
    path_tracing: Option<bool>,
    max_bounces: NumberSlider,
}

impl PathTracerPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        PathTracerPage {
            path_tracing: None,
            max_bounces: NumberSlider::new(
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
            ),
        }
    }
}

impl Render for PathTracerPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // The console can change it too.
        watch(
            window,
            cx,
            |settings: Res<RenderQualitySettings>| {
                settings.is_changed().then_some(settings.path_tracing)
            },
            |this, path_tracing, _, _| this.path_tracing = Some(path_tracing),
        );
        read(
            window,
            cx,
            |settings: Res<RenderQualitySettings>| settings.max_bounces,
            |this, bounces, window, cx| this.max_bounces.set(bounces as f32, window, cx),
        );
        let Some(path_tracing) = self.path_tracing else {
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
    }
}
