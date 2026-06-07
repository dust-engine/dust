//! Performance panel: a small egui overlay reporting frame-rate statistics.

use std::collections::VecDeque;

use bevy::prelude::*;
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use crate::profiler::GpuProfiler;

/// Number of recent frames retained for statistics and the history graph.
const HISTORY_LEN: usize = 120;

/// Adds the "Performance" egui window and the system that feeds it frame
/// timings.
///
/// Requires a primary egui context to already exist (provided by the renderer
/// plugins). It deliberately does not add `EguiPlugin` so it can be combined
/// with other panels without registering egui twice.
#[derive(Default)]
pub struct PerformancePanelPlugin;

impl Plugin for PerformancePanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PerformancePanel>();
        app.add_systems(Update, record_frame_time);
        app.add_systems(EguiPrimaryContextPass, performance_panel_ui);
    }
}

/// Frame-rate history plus renderer-reported figures, shown in the Performance
/// panel.
///
/// Other crates feed it with `Option<ResMut<PerformancePanel>>` and call e.g.
/// [`PerformancePanel::report_resolution`]; the `Option` makes reporting a
/// no-op when the `gfxdebug` plugin isn't installed (the resource won't exist),
/// so it adds no hard dependency on this crate's plugin being present.
#[derive(Resource)]
pub struct PerformancePanel {
    /// Whether the window is currently shown. Toggled by the window's close
    /// button; left public so an app can hide/show it programmatically.
    pub open: bool,
    /// Recent frame durations in seconds, oldest at the front, newest at the
    /// back. Capped at [`HISTORY_LEN`].
    frame_times: VecDeque<f32>,
    /// Most recently reported render resolution, in pixels.
    render_resolution: Option<UVec2>,
}

impl Default for PerformancePanel {
    fn default() -> Self {
        Self {
            open: true,
            frame_times: VecDeque::with_capacity(HISTORY_LEN),
            render_resolution: None,
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

    /// Report the resolution the scene is rendered at, in pixels.
    pub fn report_resolution(&mut self, resolution: UVec2) {
        self.render_resolution = Some(resolution);
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

/// Draws the performance window.
fn performance_panel_ui(
    mut contexts: EguiContexts,
    mut panel: ResMut<PerformancePanel>,
    mut profiler: Option<ResMut<GpuProfiler>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    // `open` is copied out so the window can drive its close button while the
    // closure still borrows `panel` to read stats; written back afterwards.
    let mut open = panel.open;
    egui::Window::new("Performance")
        .open(&mut open)
        .default_width(240.0)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new(format!("{:.0} FPS", panel.current_fps()))
                    .heading()
                    .strong(),
            );
            ui.label(format!("Frame: {:.2} ms", panel.last_frame_ms()));

            ui.separator();

            ui.label(format!("Average: {:.0} FPS", panel.average_fps()));
            ui.label(format!("Worst: {:.2} ms", panel.max_frame_ms()));

            if let Some(res) = panel.render_resolution {
                ui.label(format!("Resolution: {} × {}", res.x, res.y));
            }

            ui.add_space(6.0);
            frame_time_graph(ui, &panel);

            if let Some(profiler) = profiler.as_deref_mut() {
                ui.add_space(6.0);
                ui.separator();
                gpu_timings(ui, profiler);
            }
        });
    panel.open = open;
}

/// Draws the per-operation GPU timing breakdown.
fn gpu_timings(ui: &mut egui::Ui, profiler: &mut GpuProfiler) {
    ui.horizontal(|ui| {
        ui.heading("GPU");
        ui.add_enabled(
            profiler.is_supported(),
            egui::Checkbox::new(&mut profiler.enabled, "Enabled"),
        );
    });

    if !profiler.is_supported() {
        ui.label("Timestamp queries unsupported on this device.");
        return;
    }

    let rows: Vec<(&'static str, f32)> = profiler.timings().collect();
    if rows.is_empty() {
        ui.label(if profiler.enabled {
            "Measuring…"
        } else {
            "Disabled"
        });
        return;
    }

    // A single stacked bar of per-stage GPU time. Its full width represents one
    // 60 fps frame budget — unless the frame ran over budget, in which case the
    // bar expands to the total so no stage is clipped
    const BUDGET_MS: f32 = 1000.0 / 60.0;
    let total: f32 = rows.iter().map(|&(_, ms)| ms).sum();
    let scale_ms = total.max(BUDGET_MS);

    let (rect, _response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 22.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, egui::CornerRadius::ZERO, egui::Color32::from_black_alpha(96));

    let mut x = rect.left();
    for (i, &(_, ms)) in rows.iter().enumerate() {
        let seg_w = (ms / scale_ms) * rect.width();
        if seg_w > 0.0 {
            let seg = egui::Rect::from_min_max(
                egui::pos2(x, rect.top()),
                egui::pos2(x + seg_w, rect.bottom()),
            );
            painter.rect_filled(seg, egui::CornerRadius::ZERO, BAR_COLORS[i % BAR_COLORS.len()]);
        }
        x += seg_w;
    }

    // Legend mapping each color back to its stage and time.
    ui.add_space(4.0);
    for (i, &(label, ms)) in rows.iter().enumerate() {
        ui.horizontal(|ui| {
            let (swatch, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(swatch, egui::CornerRadius::same(2), BAR_COLORS[i % BAR_COLORS.len()]);
            ui.label(format!("{label}: {ms:.3} ms"));
        });
    }

    let total_color = if total > BUDGET_MS {
        egui::Color32::from_rgb(0xff, 0x6b, 0x6b)
    } else {
        egui::Color32::GRAY
    };
    ui.label(
        egui::RichText::new(format!("Total: {total:.3} / {BUDGET_MS:.2} ms"))
            .color(total_color)
            .strong(),
    );
}

/// Distinct colors cycled across stages so each segment is easy to tell apart.
const BAR_COLORS: [egui::Color32; 6] = [
    egui::Color32::from_rgb(0x4c, 0x9a, 0xff), // blue
    egui::Color32::from_rgb(0x42, 0xc9, 0x8a), // green
    egui::Color32::from_rgb(0xff, 0xc8, 0x4c), // amber
    egui::Color32::from_rgb(0xff, 0x7a, 0x59), // orange
    egui::Color32::from_rgb(0xb2, 0x84, 0xff), // purple
    egui::Color32::from_rgb(0x4c, 0xd6, 0xd6), // teal
];

/// Renders a simple auto-scaled line graph of the retained frame times so
/// spikes are visible at a glance.
fn frame_time_graph(ui: &mut egui::Ui, panel: &PerformancePanel) {
    let samples = &panel.frame_times;
    if samples.len() < 2 {
        return;
    }

    let size = egui::vec2(ui.available_width(), 48.0);
    let (rect, _response) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, egui::CornerRadius::ZERO, egui::Color32::from_black_alpha(96));

    // Scale to the worst frame time in the window (with a small floor so a run
    // of identical frames doesn't divide by ~zero).
    let max_dt = samples.iter().copied().fold(1.0e-4_f32, f32::max);

    let last = samples.len() - 1;
    let points: Vec<egui::Pos2> = samples
        .iter()
        .enumerate()
        .map(|(i, &dt)| {
            let x = rect.left() + rect.width() * (i as f32 / last as f32);
            let h = (dt / max_dt).clamp(0.0, 1.0);
            let y = rect.bottom() - rect.height() * h;
            egui::pos2(x, y)
        })
        .collect();

    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.5, egui::Color32::LIGHT_GREEN),
    ));
}
