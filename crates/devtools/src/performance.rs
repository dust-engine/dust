//! Performance panel: a small overlay reporting frame-rate statistics and GPU
//! timings.

use std::collections::VecDeque;

use bevy::prelude::*;
use dust_pbr::HdrRenderTarget;
use dust_profiler::GpuProfiler;
use gpui::{
    Bounds, Context, Hsla, IntoElement, ParentElement as _, PathBuilder, Pixels, Render,
    Styled as _, canvas, div, point, px, rgb,
};
use gpui_component::{h_flex, v_flex};

use crate::watch;

/// Number of recent frames retained for statistics and the history graph.
const HISTORY_LEN: usize = 120;

/// Records frame timings for the overlay.
#[derive(Default)]
pub struct PerformancePanelPlugin;

impl Plugin for PerformancePanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PerformancePanel>();
        app.add_systems(Update, record_frame_time);
    }
}

/// Frame-rate history, shown in the performance overlay.
#[derive(Resource, Clone)]
pub struct PerformancePanel {
    /// Recent frame durations in seconds, oldest at the front, newest at the
    /// back. Capped at [`HISTORY_LEN`].
    frame_times: VecDeque<f32>,
}

impl Default for PerformancePanel {
    fn default() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(HISTORY_LEN),
        }
    }
}

impl PerformancePanel {
    fn push(&mut self, dt: f32) {
        if self.frame_times.len() == HISTORY_LEN {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt);
    }

    /// Duration of the most recent frame, in milliseconds.
    pub fn last_frame_ms(&self) -> f32 {
        self.frame_times.back().copied().unwrap_or(0.0) * 1000.0
    }

    /// Instantaneous frame rate derived from the most recent frame.
    pub fn current_fps(&self) -> f32 {
        match self.frame_times.back().copied() {
            Some(dt) if dt > 0.0 => 1.0 / dt,
            _ => 0.0,
        }
    }

    /// Frame rate averaged over the retained history. Averaging the frame
    /// *times* (rather than the per-frame FPS values) keeps the figure stable
    /// and matches wall-clock throughput.
    pub fn average_fps(&self) -> f32 {
        let total: f32 = self.frame_times.iter().sum();
        if total > 0.0 {
            self.frame_times.len() as f32 / total
        } else {
            0.0
        }
    }

    /// Longest frame time in the retained history, in milliseconds — the worst
    /// hitch over the window.
    pub fn max_frame_ms(&self) -> f32 {
        self.frame_times.iter().copied().fold(0.0_f32, f32::max) * 1000.0
    }
}

/// Records each frame's duration into the panel's history.
fn record_frame_time(time: Res<Time>, mut panel: ResMut<PerformancePanel>) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        panel.push(dt);
    }
}

pub(crate) struct PerformanceSnapshot {
    panel: PerformancePanel,
    display_resolution: Option<UVec2>,
    render_resolution: Option<UVec2>,
    gpu: Option<GpuTimings>,
}

struct GpuTimings {
    supported: bool,
    enabled: bool,
    timings: Vec<(&'static str, f32)>,
}

/// The overlay, semi-transparent, in the top-left corner of the scene. Clicks
/// go through it to the game.
pub(crate) struct PerformanceOverlay {
    snapshot: Option<PerformanceSnapshot>,
}

impl PerformanceOverlay {
    pub fn new(_: &mut gpui::Window, _: &mut Context<Self>) -> Self {
        PerformanceOverlay { snapshot: None }
    }
}

impl Render for PerformanceOverlay {
    fn render(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) -> impl IntoElement {
        watch(
            window,
            cx,
            |panel: Res<PerformancePanel>,
             profiler: Option<Res<GpuProfiler>>,
             hdr_target: Option<Res<HdrRenderTarget>>|
             -> Option<PerformanceSnapshot> {
                if !panel.is_changed() {
                    return None;
                }
                Some(PerformanceSnapshot {
                    panel: panel.clone(),
                    display_resolution: hdr_target.as_ref().map(|hdr| hdr.display_extent),
                    render_resolution: hdr_target.as_ref().map(|hdr| hdr.render_extent),
                    gpu: profiler.map(|profiler| GpuTimings {
                        supported: profiler.is_supported(),
                        enabled: profiler.enabled,
                        timings: profiler.timings().collect(),
                    }),
                })
            },
            |this, snapshot, _, _| this.snapshot = Some(snapshot),
        );
        let mut contents = v_flex().gap_0p5();
        if let Some(snapshot) = &self.snapshot {
            let panel = &snapshot.panel;
            contents = contents
                .child(
                    div()
                        .text_xl()
                        .font_weight(gpui::FontWeight::BOLD)
                        .child(format!("{:.0} FPS", panel.current_fps())),
                )
                .child(format!("Frame: {:.2} ms", panel.last_frame_ms()))
                .child(separator())
                .child(format!("Average: {:.0} FPS", panel.average_fps()))
                .child(format!("Worst: {:.2} ms", panel.max_frame_ms()));
            if let Some(res) = snapshot.display_resolution {
                contents = contents.child(format!("Display: {} × {}", res.x, res.y));
            }
            if let Some(res) = snapshot.render_resolution {
                let scale = match snapshot.display_resolution {
                    Some(display) if display.x > 0 => res.x as f32 / display.x as f32,
                    _ => 1.0,
                };
                contents = contents.child(format!(
                    "Render: {} × {} ({:.0}%)",
                    res.x,
                    res.y,
                    scale * 100.0
                ));
            }
            contents = contents.child(frame_time_graph(panel));
            if let Some(gpu) = &snapshot.gpu {
                contents = contents.child(separator()).child(gpu_timings(gpu));
            }
        }
        div()
            .w(px(256.))
            .p_2()
            .rounded_md()
            .bg(gpui::black().opacity(0.85))
            .text_sm()
            .text_color(gpui::white())
            .child(contents)
    }
}

/// A rule between groups of figures. The theme's border color is lost on the
/// overlay's near-black background.
fn separator() -> impl IntoElement {
    div()
        .my_1()
        .h(px(1.))
        .w_full()
        .bg(gpui::white().opacity(0.2))
}

/// The per-operation GPU timing breakdown.
fn gpu_timings(gpu: &GpuTimings) -> impl IntoElement {
    let mut contents = v_flex()
        .gap_0p5()
        .child(div().font_weight(gpui::FontWeight::BOLD).child("GPU"));
    if !gpu.supported {
        return contents.child("Timestamp queries unsupported on this device.");
    }
    if gpu.timings.is_empty() {
        return contents.child(if gpu.enabled {
            "Measuring…"
        } else {
            "Disabled"
        });
    }

    // A single stacked bar of per-stage GPU time. Its full width represents one
    // 60 fps frame budget — unless the frame ran over budget, in which case the
    // bar expands to the total so no stage is clipped.
    const BUDGET_MS: f32 = 1000.0 / 60.0;
    let total: f32 = gpu.timings.iter().map(|&(_, ms)| ms).sum();
    let scale_ms = total.max(BUDGET_MS);
    contents = contents.child(
        h_flex()
            .h(px(22.))
            .w_full()
            .bg(gpui::black().opacity(0.4))
            .children(gpu.timings.iter().enumerate().map(|(i, &(_, ms))| {
                div()
                    .h_full()
                    .w(gpui::relative(ms / scale_ms))
                    .bg(rgb(BAR_COLORS[i % BAR_COLORS.len()]))
            })),
    );

    // Legend mapping each color back to its stage and time.
    for (i, &(label, ms)) in gpu.timings.iter().enumerate() {
        contents = contents.child(
            h_flex()
                .gap_1()
                .child(
                    div()
                        .size(px(10.))
                        .rounded_sm()
                        .bg(rgb(BAR_COLORS[i % BAR_COLORS.len()])),
                )
                .child(format!("{label}: {ms:.3} ms")),
        );
    }
    let total_color: Hsla = if total > BUDGET_MS {
        rgb(0xff6b6b).into()
    } else {
        gpui::white().opacity(0.6)
    };
    contents.child(
        div()
            .font_weight(gpui::FontWeight::BOLD)
            .text_color(total_color)
            .child(format!("Total: {total:.3} / {BUDGET_MS:.2} ms")),
    )
}

/// Distinct colors cycled across stages so each segment is easy to tell apart.
const BAR_COLORS: [u32; 6] = [
    0x4c9aff, // blue
    0x42c98a, // green
    0xffc84c, // amber
    0xff7a59, // orange
    0xb284ff, // purple
    0x4cd6d6, // teal
];

/// A simple auto-scaled line graph of the retained frame times, so spikes are
/// visible at a glance.
fn frame_time_graph(panel: &PerformancePanel) -> impl IntoElement {
    let samples: Vec<f32> = panel.frame_times.iter().copied().collect();
    canvas(
        |_, _, _| {},
        move |bounds: Bounds<Pixels>, _, window, _| {
            window.paint_quad(gpui::fill(bounds, gpui::black().opacity(0.4)));
            if samples.len() < 2 {
                return;
            }
            // Scale to the worst frame time in the window (with a small floor
            // so a run of identical frames doesn't divide by ~zero).
            let max_dt = samples.iter().copied().fold(1.0e-4_f32, f32::max);
            let last = (samples.len() - 1) as f32;
            let mut path = PathBuilder::stroke(px(1.5));
            for (i, &dt) in samples.iter().enumerate() {
                let x = bounds.left() + bounds.size.width * (i as f32 / last);
                let h = (dt / max_dt).clamp(0.0, 1.0);
                let y = bounds.bottom() - bounds.size.height * h;
                if i == 0 {
                    path.move_to(point(x, y));
                } else {
                    path.line_to(point(x, y));
                }
            }
            if let Ok(path) = path.build() {
                window.paint_path(path, rgb(0x90ee90));
            }
        },
    )
    .mt_1()
    .h(px(48.))
    .w_full()
}
