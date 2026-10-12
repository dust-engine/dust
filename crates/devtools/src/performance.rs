//! Performance panel: a small egui overlay reporting frame-rate statistics.

use std::collections::VecDeque;

use bevy::prelude::*;
use dust_pbr::HdrRenderTarget;
use dust_profiler::GpuProfiler;
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use crate::devtools::{DevTools, draw_devtools};

/// Number of recent frames retained for statistics and the history graph.
const HISTORY_LEN: usize = 120;

/// Adds the performance overlay and the system that feeds it frame timings.
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
        app.add_systems(
            EguiPrimaryContextPass,
            performance_overlay.after(draw_devtools),
        );
    }
}

/// Frame-rate history, shown in the performance overlay.
#[derive(Resource)]
pub struct PerformancePanel {
    /// Whether the overlay is shown. Toggled from the DevTools ⋮ menu; left
    /// public so an app can hide/show it programmatically.
    pub open: bool,
    /// Recent frame durations in seconds, oldest at the front, newest at the
    /// back. Capped at [`HISTORY_LEN`].
    frame_times: VecDeque<f32>,
}

impl Default for PerformancePanel {
    fn default() -> Self {
        Self {
            open: true,
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

/// Draws the overlay, semi-transparent, in the top-left corner of
/// [`DevTools::scene_rect`].
fn performance_overlay(
    mut contexts: EguiContexts,
    panel: Res<PerformancePanel>,
    profiler: Option<Res<GpuProfiler>>,
    hdr_target: Option<Res<HdrRenderTarget>>,
    devtools: Option<Res<DevTools>>,
) {
    if !panel.open {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let scene = devtools.map_or_else(|| ctx.viewport_rect(), |devtools| devtools.scene_rect);
    egui::Area::new(egui::Id::new("performance_overlay"))
        .fixed_pos(scene.min + egui::vec2(8.0, 8.0))
        // Clicks go through to the scene.
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(240))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    // The graphs fill `available_width()`, which an `Area`
                    // doesn't bound on its own.
                    ui.set_width(240.0);
                    performance_panel_ui(ui, &panel, profiler.as_deref(), hdr_target.as_deref());
                });
        });
}

/// Draws the overlay's contents.
fn performance_panel_ui(
    ui: &mut egui::Ui,
    panel: &PerformancePanel,
    profiler: Option<&GpuProfiler>,
    hdr_target: Option<&HdrRenderTarget>,
) {
    ui.label(
        egui::RichText::new(format!("{:.0} FPS", panel.current_fps()))
            .heading()
            .strong(),
    );
    ui.label(format!("Frame: {:.2} ms", panel.last_frame_ms()));

    ui.separator();

    ui.label(format!("Average: {:.0} FPS", panel.average_fps()));
    ui.label(format!("Worst: {:.2} ms", panel.max_frame_ms()));

    let display_resolution = hdr_target.map(|hdr| hdr.display_extent);
    if let Some(res) = display_resolution {
        ui.label(format!("Display: {} × {}", res.x, res.y));
    }
    if let Some(res) = hdr_target.map(|hdr| hdr.render_extent) {
        let scale = match display_resolution {
            Some(display) if display.x > 0 => res.x as f32 / display.x as f32,
            _ => 1.0,
        };
        ui.label(format!(
            "Render: {} × {} ({:.0}%)",
            res.x,
            res.y,
            scale * 100.0
        ));
    }

    ui.add_space(6.0);
    frame_time_graph(ui, panel);

    if let Some(profiler) = profiler {
        ui.add_space(6.0);
        ui.separator();
        gpu_timings(ui, profiler);
    }
}

/// Draws the per-operation GPU timing breakdown.
fn gpu_timings(ui: &mut egui::Ui, profiler: &GpuProfiler) {
    ui.heading("GPU");

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
    painter.rect_filled(
        rect,
        egui::CornerRadius::ZERO,
        egui::Color32::from_black_alpha(96),
    );

    let mut x = rect.left();
    for (i, &(_, ms)) in rows.iter().enumerate() {
        let seg_w = (ms / scale_ms) * rect.width();
        if seg_w > 0.0 {
            let seg = egui::Rect::from_min_max(
                egui::pos2(x, rect.top()),
                egui::pos2(x + seg_w, rect.bottom()),
            );
            painter.rect_filled(
                seg,
                egui::CornerRadius::ZERO,
                BAR_COLORS[i % BAR_COLORS.len()],
            );
        }
        x += seg_w;
    }

    // Legend mapping each color back to its stage and time.
    ui.add_space(4.0);
    for (i, &(label, ms)) in rows.iter().enumerate() {
        ui.horizontal(|ui| {
            let (swatch, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter().rect_filled(
                swatch,
                egui::CornerRadius::same(2),
                BAR_COLORS[i % BAR_COLORS.len()],
            );
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

    painter.rect_filled(
        rect,
        egui::CornerRadius::ZERO,
        egui::Color32::from_black_alpha(96),
    );

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
