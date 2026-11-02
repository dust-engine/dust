//! The "Frame pacing" page of the Graphics Debug tab: the frames on the frame
//! timeline (see `dust_profiler::timeline`). A history of frame times, a
//! timeline of what the frames shown spent their time on, thread by thread
//! and on the GPU, and tables of the GPU passes and the schedules.
//!
//! The page reads the timeline on the UI thread a few times a second while
//! it's drawn, rather than with a Bevy system, so watching adds no work to the
//! frames it shows. The timeline hands a frame over once it's complete, a few
//! frames after it ran, so the page trails the game slightly.

mod history;
mod stats;
mod timeline;

use std::cell::Cell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use bevy::prelude::{Res, World};
use dust_profiler::GpuProfiler;
use dust_profiler::timeline::{FrameRecord, FrameSummary, FrameTimeline, Name, Nanos};
use dust_ui::BevyChannel;
use gpui::{
    AnyElement, Bounds, Context, InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent,
    MouseMoveEvent, ParentElement as _, Pixels, Render, ScrollWheelEvent, SharedString,
    Styled as _, Task, Window, canvas, div, prelude::FluentBuilder as _, px,
};
use gpui_component::button::Button;
use gpui_component::checkbox::Checkbox;
use gpui_component::{ActiveTheme as _, Disableable as _, Sizable as _, h_flex, v_flex};

use crate::read;
use crate::widgets::heading;

use history::{HistoryColors, frame_at, history};
use stats::{Bound, Overview, PassRow, ScheduleRow};
use timeline::{Lanes, TimeView, TimelineColors, ms, short_name};

/// How often the page reads the timeline while it's drawn.
const REFRESH: Duration = Duration::from_millis(100);
/// The frames the page keeps with all their records, as the timeline does.
const MAX_FRAMES: usize = 300;
/// The frame summaries it keeps, for the history.
const MAX_SUMMARIES: usize = 3600;
/// How many frames, up to the selected one, the figures and tables average.
const WINDOW: usize = 120;
/// The most frames the timeline shows at once.
const MAX_SHOWN: usize = 16;
/// The refresh interval the history marks, until the display's is known.
const SIXTY_HZ: Nanos = 16_666_667;

pub(super) struct FramePacingPage {
    timeline: FrameTimeline,
    frames: VecDeque<Arc<FrameRecord>>,
    summaries: VecDeque<FrameSummary>,
    /// The threads' names, by `SpanRecord::thread`.
    threads: Vec<Name>,
    /// The timeline's generation when last read.
    generation: Option<u64>,
    /// Whether the page stopped reading the timeline, to look at what it has.
    paused: bool,
    /// The frame the timeline shows and the figures end at; `None` follows
    /// the newest.
    selected: Option<u32>,
    /// How many frames the timeline shows, ending at the selected one.
    shown: usize,
    /// What's worked out from the frames for drawing; `None` once they change.
    derived: Option<Derived>,
    /// The timeline's horizontal view; `None` fits the frames shown.
    view: Option<TimeView>,
    /// Where a drag on the timeline started, and the view then.
    drag: Option<(Pixels, TimeView)>,
    /// What the bar under the mouse on the timeline is.
    hovered: Option<SharedString>,
    /// Where the history and the timeline were drawn, for the mouse.
    history_at: Rc<Cell<Bounds<Pixels>>>,
    timeline_at: Rc<Cell<Bounds<Pixels>>>,
    refresh: Option<Task<()>>,
    /// Whether the page was drawn since the timeline last changed. While it
    /// isn't, it's hidden, and stops reading.
    drawn: bool,
    /// Whether the device has GPU timestamps, and whether they're on.
    gpu_timestamps: Option<(bool, bool)>,
}

/// What the page draws, worked out from its frames.
struct Derived {
    overview: Overview,
    /// The selected frame's summary, or the newest's.
    frame: Option<FrameSummary>,
    /// The summaries the history shows, up to the selected frame and beyond.
    history: Rc<[FrameSummary]>,
    /// The display's refresh interval, as the frames shown last reported
    /// it, or 60 Hz's.
    refresh_interval: Nanos,
    lanes: Rc<Lanes>,
    passes: Vec<PassRow>,
    schedules: Vec<ScheduleRow>,
    /// Whether the selected frame's records are gone, only its summary kept.
    details_dropped: bool,
}

fn gpu_timestamps(profiler: Option<Res<GpuProfiler>>) -> Option<(bool, bool)> {
    profiler.map(|profiler| (profiler.is_supported(), profiler.enabled))
}

impl FramePacingPage {
    pub fn new(timeline: FrameTimeline, _: &mut Window, _: &mut Context<Self>) -> Self {
        FramePacingPage {
            timeline,
            frames: VecDeque::new(),
            summaries: VecDeque::new(),
            threads: Vec::new(),
            generation: None,
            paused: false,
            selected: None,
            shown: 3,
            derived: None,
            view: None,
            drag: None,
            hovered: None,
            history_at: Rc::default(),
            timeline_at: Rc::default(),
            refresh: None,
            drawn: false,
            gpu_timestamps: None,
        }
    }

    /// Copies the frames completed since the last read. Whether there were
    /// any.
    fn pull(&mut self) -> bool {
        let generation = self.timeline.generation();
        if self.generation == Some(generation) {
            return false;
        }
        self.generation = Some(generation);
        let newest = self.frames.back().map(|frame| frame.index);
        self.frames.extend(self.timeline.frames_after(newest));
        while self.frames.len() > MAX_FRAMES {
            self.frames.pop_front();
        }
        let newest = self.summaries.back().map(|summary| summary.index);
        self.summaries.extend(self.timeline.summaries_after(newest));
        while self.summaries.len() > MAX_SUMMARIES {
            self.summaries.pop_front();
        }
        self.threads = self.timeline.threads();
        self.derived = None;
        true
    }

    /// Reads the timeline again in [`REFRESH`], unless the page is paused or
    /// already will.
    fn schedule_refresh(&mut self, cx: &mut Context<Self>) {
        if self.paused || self.refresh.is_some() {
            return;
        }
        self.refresh = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(REFRESH).await;
            this.update(cx, |this, cx| {
                this.refresh = None;
                if !this.drawn || this.paused {
                    return;
                }
                if this.pull() {
                    // Drawing again schedules the next read; a hidden page
                    // isn't drawn, and stops.
                    this.drawn = false;
                    cx.notify();
                } else {
                    this.schedule_refresh(cx);
                }
            })
            .ok();
        }));
    }

    /// The frame the timeline and the figures end at.
    fn end(&self) -> Option<u32> {
        self.selected
            .or_else(|| self.frames.back().map(|frame| frame.index))
            .or_else(|| self.summaries.back().map(|summary| summary.index))
    }

    fn derive(&mut self) {
        if self.derived.is_some() {
            return;
        }
        let end = self.end();
        let up_to_end = |index: u32| end.is_none_or(|end| index <= end);
        // Both are in frame order: the last `WINDOW` up to the end.
        let last = self
            .summaries
            .partition_point(|summary| up_to_end(summary.index));
        let window: Vec<FrameSummary> = self
            .summaries
            .range(last.saturating_sub(WINDOW)..last)
            .copied()
            .collect();
        let last = self.frames.partition_point(|frame| up_to_end(frame.index));
        let frames: Vec<&FrameRecord> = self
            .frames
            .range(last.saturating_sub(WINDOW)..last)
            .map(|frame| &**frame)
            .collect();
        let shown = &frames[frames.len().saturating_sub(self.shown)..];
        let refresh_interval = frames
            .iter()
            .rev()
            .flat_map(|frame| frame.ticks.iter().rev())
            .find_map(|tick| tick.period)
            .unwrap_or(SIXTY_HZ);
        self.derived = Some(Derived {
            overview: stats::overview(&window, refresh_interval),
            frame: window.last().copied(),
            history: self.summaries.iter().copied().collect(),
            refresh_interval,
            lanes: Rc::new(timeline::lanes(shown, &self.threads)),
            passes: stats::passes(&frames),
            schedules: stats::schedules(&frames),
            details_dropped: end.is_some_and(|end| !frames.iter().any(|frame| frame.index == end)),
        });
    }

    fn set_paused(&mut self, paused: bool, cx: &mut Context<Self>) {
        self.paused = paused;
        if !paused {
            self.selected = None;
            self.view = None;
            self.derived = None;
            self.generation = None;
            self.pull();
        }
        cx.notify();
    }

    fn set_shown(&mut self, shown: usize, cx: &mut Context<Self>) {
        self.shown = shown.clamp(1, MAX_SHOWN);
        self.view = None;
        self.derived = None;
        cx.notify();
    }

    /// Selects the frame under the mouse in the history, and pauses.
    fn select_in_history(
        &mut self,
        event: &MouseDownEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(derived) = &self.derived else {
            return;
        };
        let history = &derived.history;
        let Some(at) = frame_at(history.len(), self.history_at.get(), event.position.x) else {
            return;
        };
        self.selected = Some(history[at].index);
        self.paused = true;
        self.view = None;
        self.derived = None;
        cx.notify();
    }

    /// The timeline's view as drawn: the one set, or the one fitting its
    /// frames.
    fn current_view(&self) -> Option<TimeView> {
        let lanes = &self.derived.as_ref()?.lanes;
        Some(
            self.view
                .unwrap_or_else(|| TimeView::fit(lanes, self.timeline_at.get())),
        )
    }

    fn timeline_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.click_count == 2 {
            self.view = None;
            cx.notify();
            return;
        }
        if let Some(view) = self.current_view() {
            self.drag = Some((event.position.x, view));
        }
    }

    fn timeline_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let bounds = self.timeline_at.get();
        if let Some((from, view)) = self.drag {
            if event.pressed_button == Some(MouseButton::Left) {
                self.view = Some(view.pan(event.position.x - from));
                cx.notify();
                return;
            }
            self.drag = None;
        }
        let Some(view) = self.current_view() else {
            return;
        };
        let lanes = &self.derived.as_ref().unwrap().lanes;
        let hovered =
            timeline::hit(lanes, view, bounds, event.position).map(|bar| bar.detail.clone());
        if hovered != self.hovered {
            self.hovered = hovered;
            cx.notify();
        }
    }

    /// Zooms with Ctrl or ⌘ held, and pans sideways with a horizontal scroll;
    /// a plain scroll scrolls the page.
    fn timeline_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(view) = self.current_view() else {
            return;
        };
        let delta = event.delta.pixel_delta(px(20.));
        let bounds = self.timeline_at.get();
        if event.modifiers.platform || event.modifiers.control {
            self.view = Some(view.zoom(
                bounds,
                event.position.x,
                (f64::from(f32::from(delta.y)) * 0.005).exp(),
            ));
        } else if delta.x != px(0.) || event.modifiers.shift {
            let dx = if delta.x != px(0.) { delta.x } else { delta.y };
            self.view = Some(view.pan(dx));
        } else {
            return;
        }
        cx.stop_propagation();
        cx.notify();
    }

    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let status = match (self.selected, self.paused) {
            (Some(selected), _) => format!("Frame #{selected}"),
            (None, true) => "Paused".to_owned(),
            (None, false) => "Following the newest frames".to_owned(),
        };
        let mut toolbar = h_flex()
            .gap_1()
            .flex_wrap()
            .child(
                Button::new("frame-pacing-pause")
                    .label(if self.paused { "Resume" } else { "Pause" })
                    .small()
                    .tooltip("Pause to look at the frames shown; click a frame in the history to pick it.")
                    .on_click(cx.listener(|this, _, _, cx| this.set_paused(!this.paused, cx))),
            )
            .child(
                Button::new("frame-pacing-fewer")
                    .label("−")
                    .small()
                    .disabled(self.shown <= 1)
                    .tooltip("Show fewer frames")
                    .on_click(cx.listener(|this, _, _, cx| this.set_shown(this.shown - 1, cx))),
            )
            .child(
                div()
                    .text_color(theme.muted_foreground)
                    .child(format!("{} frames", self.shown)),
            )
            .child(
                Button::new("frame-pacing-more")
                    .label("+")
                    .small()
                    .disabled(self.shown >= MAX_SHOWN)
                    .tooltip("Show more frames")
                    .on_click(cx.listener(|this, _, _, cx| this.set_shown(this.shown + 1, cx))),
            )
            .child(
                Button::new("frame-pacing-fit")
                    .label("Fit")
                    .small()
                    .tooltip("Fit the frames shown (or double-click the timeline)")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.view = None;
                        cx.notify();
                    })),
            );
        if let Some((supported, enabled)) = self.gpu_timestamps {
            toolbar = toolbar.child(
                Checkbox::new("frame-pacing-gpu-timestamps")
                    .label(if supported {
                        "GPU timestamps"
                    } else {
                        "GPU timestamps (unsupported)"
                    })
                    .checked(enabled)
                    .disabled(!supported)
                    .on_click(cx.listener(|this, enabled: &bool, _, cx| {
                        let enabled = *enabled;
                        if let Some((supported, _)) = this.gpu_timestamps {
                            this.gpu_timestamps = Some((supported, enabled));
                        }
                        cx.global::<BevyChannel>().update(move |world: &mut World| {
                            if let Some(mut profiler) = world.get_resource_mut::<GpuProfiler>() {
                                profiler.enabled = enabled;
                            }
                        });
                        cx.notify();
                    })),
            );
        }
        toolbar.child(div().text_color(theme.muted_foreground).child(status))
    }
}

impl Render for FramePacingPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        read(window, cx, gpu_timestamps, |this, status, _, _| {
            this.gpu_timestamps = status
        });
        if self.generation.is_none() {
            self.pull();
        }
        self.drawn = true;
        self.schedule_refresh(cx);
        self.derive();
        let toolbar = self.toolbar(cx);
        let theme = cx.theme();
        let Some(derived) = &self.derived else {
            return v_flex().child(toolbar).into_any_element();
        };
        if derived.history.is_empty() {
            return v_flex()
                .gap_2()
                .child(toolbar)
                .child(
                    div()
                        .text_color(theme.muted_foreground)
                        .child("No frames recorded yet."),
                )
                .into_any_element();
        }

        let figures = figures(&derived.overview, derived.frame.as_ref(), theme);
        let budget = derived.refresh_interval;
        let history = div()
            .id("frame-pacing-history")
            .child(history(
                derived.history.clone(),
                self.selected,
                budget,
                HistoryColors {
                    background: theme.muted.opacity(0.4),
                    interval: theme.muted_foreground.opacity(0.35),
                    cpu: theme.info,
                    gpu: theme.warning,
                    dropped: theme.danger,
                    budget: theme.success.opacity(0.7),
                    selected: theme.foreground,
                },
                self.history_at.clone(),
            ))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::select_in_history));

        let lanes = derived.lanes.clone();
        let view = self.view;
        let timeline_at = self.timeline_at.clone();
        let colors = TimelineColors {
            background: theme.muted.opacity(0.25),
            label: theme.muted_foreground,
            text: theme.foreground,
            grid: theme.border,
        };
        let font = gpui::font(theme.font_family.clone());
        let height = timeline::height(&lanes);
        let timeline = div()
            .id("frame-pacing-timeline")
            .w_full()
            .h(height)
            .child(
                canvas(
                    move |bounds, _, _| timeline_at.set(bounds),
                    move |bounds, _, window, cx| {
                        let view = view.unwrap_or_else(|| TimeView::fit(&lanes, bounds));
                        timeline::paint(&lanes, view, bounds, &colors, &font, window, cx);
                    },
                )
                .size_full(),
            )
            .on_mouse_down(MouseButton::Left, cx.listener(Self::timeline_mouse_down))
            .on_mouse_move(cx.listener(Self::timeline_mouse_move))
            .on_scroll_wheel(cx.listener(Self::timeline_scroll));
        let hint = match &self.hovered {
            Some(hovered) => div().child(hovered.clone()),
            None => div()
                .text_color(theme.muted_foreground)
                .child("Drag to pan, Ctrl+scroll to zoom, double-click to fit."),
        };

        let mut page = v_flex()
            .gap_2()
            .child(toolbar)
            .child(figures)
            .child(heading("Frame times"))
            .child(history)
            .child(legend(budget, theme))
            .child(heading("Timeline"));
        if derived.details_dropped {
            page = page.child(div().text_color(theme.muted_foreground).child(format!(
                "Only the last {MAX_FRAMES} frames are kept in full."
            )));
        } else {
            page = page.child(timeline).child(hint.text_xs());
        }
        page.child(heading("GPU passes"))
            .child(pass_table(&derived.passes, theme))
            .child(heading("Schedules"))
            .child(schedule_table(&derived.schedules, theme))
            .into_any_element()
    }
}

/// The figures over the frames up to the selected one, and the selected
/// frame's own.
fn figures(
    overview: &Overview,
    frame: Option<&FrameSummary>,
    theme: &gpui_component::Theme,
) -> impl IntoElement + use<> {
    let figure = |label: &str, value: String| {
        h_flex()
            .gap_1()
            .child(
                div()
                    .text_color(theme.muted_foreground)
                    .child(label.to_owned()),
            )
            .child(
                div()
                    .font_family(theme.mono_font_family.clone())
                    .child(value),
            )
    };
    let optional = |value: Option<Nanos>| value.map_or_else(|| "—".to_owned(), ms);
    let mut over_window = h_flex()
        .gap_3()
        .flex_wrap()
        .child(
            div()
                .text_lg()
                .font_weight(gpui::FontWeight::BOLD)
                .child(match overview.fps {
                    Some(fps) => format!("{fps:.1} FPS"),
                    None => "— FPS".to_owned(),
                }),
        )
        .child(figure("Interval", optional(overview.interval)))
        .child(figure("p99", optional(overview.interval_p99)))
        .child(figure("Max", optional(overview.interval_max)))
        .child(figure("CPU", optional(overview.cpu)))
        .child(figure("GPU", optional(overview.gpu)))
        .child(figure("Waiting", optional(overview.wait)));
    if overview.display_latency.is_some() {
        over_window = over_window.child(figure(
            "Start → display",
            optional(overview.display_latency),
        ));
    }
    if overview.input_latency.is_some() {
        over_window =
            over_window.child(figure("Input → display", optional(overview.input_latency)));
    }
    if overview.dropped > 0 {
        over_window = over_window.child(figure("Dropped", overview.dropped.to_string()));
    }
    if let Some(bound) = overview.bound {
        over_window = over_window.child(figure(
            "Held back by",
            match bound {
                Bound::Gpu => "the GPU",
                Bound::Cpu => "the CPU",
                Bound::Display => "the display",
            }
            .to_owned(),
        ));
    }
    let mut column = v_flex().gap_1().child(over_window).child(
        div()
            .text_xs()
            .text_color(theme.muted_foreground)
            .child(format!("Averages over {} frames.", overview.frames)),
    );
    if let Some(frame) = frame {
        column = column.child(
            h_flex()
                .gap_3()
                .flex_wrap()
                .child(
                    div()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(format!("Frame #{}", frame.index)),
                )
                .child(figure("Interval", optional(frame.interval)))
                .child(figure("CPU", ms(frame.cpu)))
                .child(figure("GPU", optional(frame.gpu)))
                .child(figure("Waiting", ms(frame.wait))),
        );
    }
    column
}

/// What the history's colors mean.
fn legend(budget: Nanos, theme: &gpui_component::Theme) -> impl IntoElement + use<> {
    let key = |color: gpui::Hsla, label: &str| {
        h_flex()
            .gap_1()
            .child(div().size(px(8.)).bg(color))
            .child(label.to_owned())
    };
    h_flex()
        .gap_3()
        .flex_wrap()
        .text_xs()
        .text_color(theme.muted_foreground)
        .child(key(theme.muted_foreground.opacity(0.35), "Interval"))
        .child(key(theme.info, "CPU"))
        .child(key(theme.warning, "GPU"))
        .child(key(theme.danger, "Dropped"))
        .child(key(
            theme.success.opacity(0.7),
            &format!("Refresh interval ({:.1} ms)", budget as f64 / 1e6),
        ))
}

/// A table row: a name column taking the rest of the width, then figures.
fn row(
    name: AnyElement,
    cells: Vec<String>,
    widths: &[f32],
    header: bool,
    theme: &gpui_component::Theme,
) -> impl IntoElement + use<> {
    let mono = theme.mono_font_family.clone();
    let mut row = h_flex()
        .gap_2()
        .when(header, |row| row.text_color(theme.muted_foreground))
        .child(div().flex_1().min_w_0().overflow_hidden().child(name));
    for (cell, &width) in cells.into_iter().zip(widths) {
        row = row.child(
            div()
                .w(px(width))
                .flex_shrink_0()
                .text_right()
                .whitespace_nowrap()
                .when(!header, |cell| cell.font_family(mono.clone()))
                .child(cell),
        );
    }
    row
}

/// The figures in the tables: milliseconds, without the unit, which the
/// header gives.
fn millis(nanos: Nanos) -> String {
    format!("{:.3}", nanos as f64 / 1e6)
}

fn pass_table(passes: &[PassRow], theme: &gpui_component::Theme) -> impl IntoElement + use<> {
    const WIDTHS: [f32; 4] = [60.0, 60.0, 32.0, 60.0];
    if passes.is_empty() {
        return v_flex().child(
            div()
                .text_color(theme.muted_foreground)
                .child("No GPU timings: they need GPU timestamps."),
        );
    }
    let mut table = v_flex().gap_0p5().child(row(
        div().child("Pass").into_any_element(),
        vec!["GPU ms".into(), "Max".into(), "%".into(), "CPU ms".into()],
        &WIDTHS,
        true,
        theme,
    ));
    for pass in passes {
        let name = v_flex()
            .pl(px(12.0 * f32::from(pass.depth.saturating_sub(1))))
            .child(pass.name.to_string())
            // The system that recorded it.
            .when_some(pass.encoded_by.as_ref(), |name, (system, _)| {
                name.child(
                    div()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child(format!("in {}", short_name(system))),
                )
            });
        table = table.child(row(
            name.into_any_element(),
            vec![
                millis(pass.gpu),
                millis(pass.gpu_max),
                pass.share
                    .map_or_else(|| "—".to_owned(), |share| format!("{:.0}", share * 100.0)),
                pass.encoded_by
                    .as_ref()
                    .map_or_else(|| "—".to_owned(), |(_, cpu)| millis(*cpu)),
            ],
            &WIDTHS,
            false,
            theme,
        ));
    }
    table.child(
        div().text_xs().text_color(theme.muted_foreground).child(
            "% is of the GPU's frame. CPU is the time of the system that recorded the pass.",
        ),
    )
}

fn schedule_table(
    schedules: &[ScheduleRow],
    theme: &gpui_component::Theme,
) -> impl IntoElement + use<> {
    const WIDTHS: [f32; 3] = [60.0, 60.0, 36.0];
    let mut table = v_flex().gap_0p5().child(row(
        div().child("Schedule").into_any_element(),
        vec!["ms".into(), "Max".into(), "Runs".into()],
        &WIDTHS,
        true,
        theme,
    ));
    for schedule in schedules {
        let name = div()
            .pl(px(12.0 * schedule.nesting as f32))
            .child(schedule.name.to_string());
        table = table.child(row(
            name.into_any_element(),
            vec![
                millis(schedule.time),
                millis(schedule.time_max),
                format!("{:.1}", schedule.runs),
            ],
            &WIDTHS,
            false,
            theme,
        ));
    }
    table.child(div().text_xs().text_color(theme.muted_foreground).child(
        "Time per frame, all runs together. A schedule's time includes the schedules it runs.",
    ))
}
