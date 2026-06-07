//! Performance panel: a small egui overlay reporting frame-rate statistics.

use std::collections::VecDeque;

use bevy::prelude::*;
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

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

/// Rolling record of recent frame times, used to derive the displayed stats.
#[derive(Resource)]
pub struct PerformancePanel {
    /// Whether the window is currently shown. Toggled by the window's close
    /// button; left public so an app can hide/show it programmatically.
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

/// Draws the performance window.
fn performance_panel_ui(mut contexts: EguiContexts, mut panel: ResMut<PerformancePanel>) {
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

            ui.add_space(6.0);
            frame_time_graph(ui, &panel);
        });
    panel.open = open;
}

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
