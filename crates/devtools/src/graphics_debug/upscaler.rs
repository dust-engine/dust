//! Toggles of the "Super-resolution" stage: the denoising / upscaling backend
//! and its render scale, and the reference accumulator's controls.

use bevy::prelude::*;
use dust_pbr::upscaler::reference::ReferenceAccumulator;
use dust_pbr::upscaler::{AvailableUpscalers, ResetHistory, UpscalerBackend, UpscalerSessionState};
use dust_pbr::{HdrRenderTarget, RenderQualitySettings, UpscalerQualityMode};
use dust_ui::BevyChannel;
use gpui::{Context, IntoElement, ParentElement as _, Render, Styled as _, Window, div, px};
use gpui_component::button::Button;
use gpui_component::checkbox::Checkbox;
use gpui_component::radio::{Radio, RadioGroup};
use gpui_component::{ActiveTheme as _, Sizable as _, h_flex, v_flex};

use crate::watch;
use crate::widgets::{RADIO_LABEL_OVERHANG, heading};

/// What the page shows, read from the world.
#[derive(Clone, PartialEq)]
pub(super) struct UpscalerSnapshot {
    settings: RenderQualitySettings,
    /// Whether the device runs each of [`UpscalerBackend::ALL`]; `None` until
    /// the backends are known.
    available: Option<[bool; UpscalerBackend::ALL.len()]>,
    /// The backend running; `None` before the first session.
    active: Option<UpscalerBackend>,
    jitter_phases: usize,
    /// Render and display extents.
    extents: Option<(UVec2, UVec2)>,
    /// Frames the reference accumulator has averaged; `None` while another
    /// backend runs.
    reference_frames: Option<u32>,
    /// Whether `Time<Virtual>` is paused.
    paused: bool,
}

/// The page's [`UpscalerSnapshot`], sent when it differs from the last one
/// sent.
pub(super) fn upscaler_snapshot(
    settings: Res<RenderQualitySettings>,
    available: Option<Res<AvailableUpscalers>>,
    state: Res<UpscalerSessionState>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    time: Res<Time<Virtual>>,
    mut last: Local<Option<UpscalerSnapshot>>,
) -> Option<UpscalerSnapshot> {
    let snapshot = UpscalerSnapshot {
        settings: *settings,
        available: available.map(|available| UpscalerBackend::ALL.map(|b| available.contains(b))),
        active: state.active_backend(),
        jitter_phases: state.jitter_phase_count(),
        extents: hdr_target.map(|hdr| (hdr.render_extent, hdr.display_extent)),
        reference_frames: ReferenceAccumulator::running(&state).map(ReferenceAccumulator::frames),
        paused: time.is_paused(),
    };
    if last.as_ref() == Some(&snapshot) {
        return None;
    }
    *last = Some(snapshot.clone());
    Some(snapshot)
}

/// A backend's radio label, short enough for a docked panel.
fn label(backend: UpscalerBackend) -> &'static str {
    match backend {
        UpscalerBackend::Blit => "Blit (no denoising)",
        UpscalerBackend::MetalFx => "MetalFX denoised scaler",
        UpscalerBackend::Dlss => "DLSS Ray Reconstruction",
        UpscalerBackend::Reference => "Reference (accumulate)",
    }
}

const QUALITY_MODES: [(UpscalerQualityMode, &str); 6] = [
    (
        UpscalerQualityMode::UltraPerformance,
        "Ultra performance (33%)",
    ),
    (UpscalerQualityMode::Performance, "Performance (50%)"),
    (UpscalerQualityMode::Balanced, "Balanced (58%)"),
    (UpscalerQualityMode::Quality, "Quality (67%)"),
    (UpscalerQualityMode::UltraQuality, "Ultra quality (77%)"),
    (UpscalerQualityMode::Dlaa, "Native (100%)"),
];

pub(super) struct UpscalerPage {
    snapshot: Option<UpscalerSnapshot>,
}

/// Applies an edit of the [`RenderQualitySettings`] to the page's copy, and
/// sends it, marking the settings changed only if the edit changed them.
fn settings(
    this: &mut UpscalerPage,
    cx: &mut Context<UpscalerPage>,
    edit: impl Fn(&mut RenderQualitySettings) + Send + 'static,
) {
    if let Some(snapshot) = &mut this.snapshot {
        edit(&mut snapshot.settings);
        cx.notify();
    }
    cx.global::<BevyChannel>().update(move |world| {
        let mut settings = world.resource_mut::<RenderQualitySettings>();
        let mut next = *settings;
        edit(&mut next);
        settings.set_if_neq(next);
    });
}

/// Pauses or resumes `Time<Virtual>`, which stops or restarts the animation
/// and physics, and shows it on the page's copy.
fn set_paused(this: &mut UpscalerPage, cx: &mut Context<UpscalerPage>, paused: bool) {
    if let Some(snapshot) = &mut this.snapshot {
        snapshot.paused = paused;
        cx.notify();
    }
    cx.global::<BevyChannel>().update(move |world| {
        let mut time = world.resource_mut::<Time<Virtual>>();
        if paused {
            time.pause();
        } else {
            time.unpause();
        }
    });
}

impl UpscalerPage {
    pub fn new(_window: &mut Window, _cx: &mut Context<Self>) -> Self {
        UpscalerPage { snapshot: None }
    }
}

impl Render for UpscalerPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        watch(window, cx, upscaler_snapshot, |this, snapshot, _, _| {
            this.snapshot = Some(snapshot);
        });
        let Some(snapshot) = &self.snapshot else {
            return v_flex();
        };
        let muted = cx.theme().muted_foreground;
        let requested = snapshot.settings.upscaler;
        let available = |index: usize| snapshot.available.is_none_or(|a| a[index]);

        let backends = UpscalerBackend::ALL
            .into_iter()
            .enumerate()
            .map(|(index, backend)| {
                let label = if available(index) {
                    label(backend).to_owned()
                } else {
                    format!("{} (unavailable)", label(backend))
                };
                Radio::new(index).label(label).disabled(!available(index))
            });
        let fallback = snapshot
            .active
            .filter(|&active| active != requested)
            .map(|active| {
                div()
                    .text_color(muted)
                    .child(format!("Running: {}", label(active)))
            });

        let reference = snapshot.reference_frames.map(|frames| {
            v_flex()
                .gap_1()
                .child(heading("Reference accumulator"))
                .child(format!("Frames averaged: {frames}"))
                .child(
                    div()
                        .text_color(muted)
                        .child("Restart after the view or scene changes."),
                )
                .child(
                    h_flex()
                        .gap_3()
                        .child(
                            Button::new("reference-restart")
                                .label("Restart")
                                .small()
                                .on_click(cx.listener(|_, _, _, cx| {
                                    cx.global::<BevyChannel>().update(|world| {
                                        world.write_message(ResetHistory);
                                    });
                                })),
                        )
                        .child(
                            Checkbox::new("reference-pause")
                                .label("Pause simulation")
                                .checked(snapshot.paused)
                                .on_click(cx.listener(|this, paused: &bool, _, cx| {
                                    set_paused(this, cx, *paused);
                                })),
                        ),
                )
                .child(
                    div()
                        .text_color(muted)
                        .child("Pausing lets a moving scene converge."),
                )
        });

        let upscales = snapshot.active.is_some_and(UpscalerBackend::upscales);
        let quality = QUALITY_MODES
            .iter()
            .position(|&(mode, _)| mode == snapshot.settings.quality);
        let status = snapshot.extents.map(|(render, display)| {
            v_flex()
                .text_color(muted)
                .child(format!(
                    "Render {}×{} → display {}×{}",
                    render.x, render.y, display.x, display.y
                ))
                .child(format!("Jitter phases: {}", snapshot.jitter_phases))
        });

        v_flex()
            .gap_1()
            .child(heading("Backend"))
            .child(
                RadioGroup::vertical("upscaler-backend")
                    .pr(px(RADIO_LABEL_OVERHANG))
                    .children(backends)
                    .selected_index(UpscalerBackend::ALL.iter().position(|&b| b == requested))
                    .on_click(cx.listener(|this, index: &usize, _, cx| {
                        let backend = UpscalerBackend::ALL[*index];
                        settings(this, cx, move |s| s.upscaler = backend);
                    })),
            )
            .children(fallback)
            .children(reference)
            .child(heading("Render scale"))
            .child(
                RadioGroup::vertical("upscaler-quality")
                    .pr(px(RADIO_LABEL_OVERHANG))
                    .children(QUALITY_MODES.map(|(_, label)| label))
                    .selected_index(quality)
                    .disabled(!upscales)
                    .on_click(cx.listener(|this, index: &usize, _, cx| {
                        let mode = QUALITY_MODES[*index].0;
                        settings(this, cx, move |s| s.quality = mode);
                    })),
            )
            .children((!upscales).then(|| {
                div()
                    .text_color(muted)
                    .child("Renders at display resolution.")
            }))
            .children(status)
    }
}
