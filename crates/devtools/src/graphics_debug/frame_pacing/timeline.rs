//! The timeline of the frames shown: a lane per thread, and per depth on the
//! Bevy thread and the GPU, with each span as a bar.

use std::fmt::Write as _;

use dust_profiler::timeline::{FrameRecord, Name, Nanos, SpanKind, SpanRecord};
use gpui::{
    App, Bounds, ContentMask, Font, Hsla, Pixels, Point, SharedString, TextAlign, TextRun, Window,
    fill, hsla, point, px, size,
};

/// The width of the lane labels, left of the bars.
pub(super) const GUTTER: f32 = 92.0;
/// A lane's height.
pub(super) const LANE: f32 = 16.0;
/// The height of the time axis, above the lanes.
pub(super) const AXIS: f32 = 18.0;
/// The most lanes the Bevy thread gets; deeper spans share the last.
const BEVY_LANES: usize = 12;
const FONT_SIZE: f32 = 11.0;

/// The part of the timeline shown: the time at the left edge of the bars, and
/// how much time a pixel covers.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct TimeView {
    pub start: f64,
    pub ns_per_px: f64,
}

impl TimeView {
    /// The view that fits `lanes` in a timeline drawn in `bounds`.
    pub fn fit(lanes: &Lanes, bounds: Bounds<Pixels>) -> Self {
        let width = (f32::from(bounds.size.width) - GUTTER - 8.0).max(1.0);
        TimeView {
            start: lanes.start as f64,
            ns_per_px: (lanes.end.saturating_sub(lanes.start)).max(1) as f64 / f64::from(width),
        }
    }

    /// The time at `x` in a timeline drawn in `bounds`.
    pub fn time_at(&self, bounds: Bounds<Pixels>, x: Pixels) -> f64 {
        self.start + f64::from(f32::from(x - bounds.left()) - GUTTER) * self.ns_per_px
    }

    fn x(&self, bounds: Bounds<Pixels>, time: Nanos) -> f32 {
        f32::from(bounds.left()) + GUTTER + ((time as f64 - self.start) / self.ns_per_px) as f32
    }

    /// Zooms by `factor` around the time at `x`.
    pub fn zoom(&self, bounds: Bounds<Pixels>, x: Pixels, factor: f64) -> Self {
        let anchor = self.time_at(bounds, x);
        let ns_per_px = (self.ns_per_px / factor).clamp(1.0, 1e9);
        TimeView {
            start: anchor - f64::from(f32::from(x - bounds.left()) - GUTTER) * ns_per_px,
            ns_per_px,
        }
    }

    /// Moves the bars `dx` to the right.
    pub fn pan(&self, dx: Pixels) -> Self {
        TimeView {
            start: self.start - f64::from(f32::from(dx)) * self.ns_per_px,
            ..*self
        }
    }
}

/// What the timeline draws.
pub(super) struct Lanes {
    /// The frames' time span.
    pub start: Nanos,
    pub end: Nanos,
    /// When each frame started, for the lines across the lanes.
    pub frame_starts: Vec<Nanos>,
    pub lanes: Vec<Lane>,
}

pub(super) struct Lane {
    /// Shown left of the first lane of each thread, or of the GPU.
    pub label: Option<SharedString>,
    pub bars: Vec<Bar>,
}

pub(super) struct Bar {
    pub begin: Nanos,
    pub end: Nanos,
    pub color: Hsla,
    /// Written on the bar when it fits.
    pub text: SharedString,
    /// What the line under the timeline says while the mouse is on the bar.
    pub detail: SharedString,
}

/// The lanes of `frames`, whose spans ran on `threads`.
pub(super) fn lanes(frames: &[&FrameRecord], threads: &[Name]) -> Lanes {
    let thread_name = |thread: u32| threads.get(thread as usize).map_or("?", |name| &**name);
    let bevy = threads
        .iter()
        .position(|name| &**name == "bevy")
        .map(|bevy| bevy as u32);
    let start = frames
        .iter()
        .map(|frame| frame.begin)
        .filter(|&begin| begin > 0)
        .min()
        .unwrap_or(0);
    let mut end = start;
    let mut lanes = Vec::new();

    let mut frame_lane = Lane {
        label: Some("frames".into()),
        bars: Vec::new(),
    };
    for frame in frames {
        if frame.begin == 0 {
            continue;
        }
        end = end.max(frame.end);
        frame_lane.bars.push(Bar {
            begin: frame.begin,
            end: frame.end,
            color: hsla(0.6, 0.1, 0.4, 1.0),
            text: format!("#{}", frame.index).into(),
            detail: format!("Frame #{} · {}", frame.index, ms(frame.end - frame.begin)).into(),
        });
    }
    lanes.push(frame_lane);

    // The display: the refreshes the runner woke for, as marks where the frame
    // started then should be shown, and when each frame was.
    let mut display = Lane {
        label: Some("display".into()),
        bars: Vec::new(),
    };
    for frame in frames {
        for tick in &frame.ticks {
            if let Some(shown) = tick.present.or(tick.target) {
                display.bars.push(Bar {
                    begin: shown,
                    end: shown,
                    color: hsla(0.0, 0.0, 0.55, 1.0),
                    text: SharedString::default(),
                    detail: format!(
                        "Refresh for frame #{}: shown at {}{}",
                        frame.index,
                        ms(shown.saturating_sub(frame.begin)),
                        tick.period
                            .map_or_else(String::new, |period| format!(", every {}", ms(period))),
                    )
                    .into(),
                });
            }
        }
        if let Some(at) = frame.displayed.and_then(|displayed| displayed.at) {
            end = end.max(at);
            display.bars.push(Bar {
                begin: at,
                end: at,
                color: hsla(0.38, 0.6, 0.5, 1.0),
                text: format!("#{}", frame.index).into(),
                detail: format!(
                    "Frame #{} shown, {} after it started",
                    frame.index,
                    ms(at.saturating_sub(frame.begin))
                )
                .into(),
            });
        }
    }
    if !display.bars.is_empty() {
        lanes.push(display);
    }

    // Input: each poll, as a bar from the oldest input it read.
    let mut input = Lane {
        label: Some("input".into()),
        bars: Vec::new(),
    };
    for frame in frames {
        for poll in &frame.inputs {
            let oldest = poll.oldest.unwrap_or(poll.at);
            input.bars.push(Bar {
                begin: oldest,
                end: poll.at,
                color: hsla(0.33, 0.5, 0.38, 1.0),
                text: poll.count.to_string().into(),
                detail: format!(
                    "{} inputs read by frame #{}, the oldest {} before",
                    poll.count,
                    frame.index,
                    ms(poll.at.saturating_sub(oldest))
                )
                .into(),
            });
        }
    }
    if !input.bars.is_empty() {
        lanes.push(input);
    }

    // Latency: from the oldest input each frame read, or its start, to when it
    // was shown; a lane per frame, as they overlap.
    for frame in frames {
        let Some(shown) = frame.displayed.and_then(|displayed| displayed.at) else {
            continue;
        };
        let oldest_input = frame
            .inputs
            .iter()
            .map(|poll| poll.oldest.unwrap_or(poll.at))
            .min();
        let from = oldest_input.unwrap_or(frame.begin);
        let what = if oldest_input.is_some() {
            "input"
        } else {
            "start"
        };
        lanes.push(Lane {
            label: Some(format!("#{} latency", frame.index).into()),
            bars: vec![Bar {
                begin: from,
                end: shown,
                color: hsla(0.15, 0.35, 0.35, 1.0),
                text: format!("{what} → display {}", ms(shown.saturating_sub(from))).into(),
                detail: format!(
                    "Frame #{}: {} from its {what} to the display",
                    frame.index,
                    ms(shown.saturating_sub(from))
                )
                .into(),
            }],
        });
    }

    // The Bevy thread runs the frame and the schedules, nested deep: a lane
    // per depth. The other threads run systems: a lane each.
    let mut bevy_lanes: Vec<Lane> = Vec::new();
    let mut other: Vec<(u32, Lane)> = Vec::new();
    for frame in frames {
        for span in frame
            .spans
            .iter()
            .filter(|span| span.kind != SpanKind::Frame)
        {
            end = end.max(span.end);
            let bar = span_bar(span, frame.index, thread_name(span.thread));
            if Some(span.thread) == bevy {
                let lane = usize::from(span.depth.max(1) - 1).min(BEVY_LANES - 1);
                while bevy_lanes.len() <= lane {
                    bevy_lanes.push(Lane {
                        label: bevy_lanes.is_empty().then(|| "bevy".into()),
                        bars: Vec::new(),
                    });
                }
                bevy_lanes[lane].bars.push(bar);
            } else if span.depth == 0 {
                let at = match other.iter().position(|(thread, _)| *thread == span.thread) {
                    Some(at) => at,
                    None => {
                        other.push((
                            span.thread,
                            Lane {
                                label: Some(short_thread_name(thread_name(span.thread)).into()),
                                bars: Vec::new(),
                            },
                        ));
                        other.len() - 1
                    }
                };
                other[at].1.bars.push(bar);
            }
        }
    }
    lanes.extend(bevy_lanes);
    other.sort_by_key(|(thread, _)| *thread);
    lanes.extend(other.into_iter().map(|(_, lane)| lane));

    // GPU work placed on the CPU's clock: a lane per depth.
    let mut gpu_lanes: Vec<Lane> = Vec::new();
    for frame in frames {
        for gpu in &frame.gpu {
            let (Some(begin), Some(end_at)) = (gpu.begin, gpu.end) else {
                continue;
            };
            end = end.max(end_at);
            let lane = usize::from(gpu.depth);
            while gpu_lanes.len() <= lane {
                gpu_lanes.push(Lane {
                    label: gpu_lanes.is_empty().then(|| "GPU".into()),
                    bars: Vec::new(),
                });
            }
            let mut detail = format!("{} · GPU · {}", gpu.name, ms(gpu.duration));
            if let Some(set) = &gpu.set {
                let _ = write!(detail, " · {set}");
            }
            if !gpu.calibrated {
                detail.push_str(" · estimated placement");
            }
            let _ = write!(detail, " · frame #{}", frame.index);
            gpu_lanes[lane].bars.push(Bar {
                begin,
                end: end_at,
                color: hashed_color(&gpu.name, 0.55, 0.45),
                text: gpu.name.to_string().into(),
                detail: detail.into(),
            });
        }
    }
    lanes.extend(gpu_lanes);

    Lanes {
        start,
        end,
        frame_starts: frames
            .iter()
            .map(|frame| frame.begin)
            .filter(|&begin| begin > 0)
            .collect(),
        lanes,
    }
}

fn span_bar(span: &SpanRecord, frame: u32, thread: &str) -> Bar {
    let (kind, color) = match span.kind {
        SpanKind::Frame => ("frame", hsla(0.6, 0.1, 0.4, 1.0)),
        SpanKind::Schedule => ("schedule", hashed_color(&span.name, 0.45, 0.42)),
        SpanKind::System => ("system", hashed_color(&span.name, 0.3, 0.36)),
        SpanKind::Commands => ("commands", hsla(0.0, 0.0, 0.32, 1.0)),
        SpanKind::Pass => ("GPU pass encoding", hsla(0.5, 0.5, 0.38, 1.0)),
        SpanKind::Wait => ("waiting", hsla(0.0, 0.0, 0.22, 1.0)),
        SpanKind::Submit => ("submission", hsla(0.08, 0.6, 0.45, 1.0)),
        SpanKind::Swapchain => ("swapchain", hsla(0.78, 0.45, 0.45, 1.0)),
        SpanKind::Input => ("input", hsla(0.33, 0.5, 0.38, 1.0)),
        SpanKind::Ui => ("UI", hsla(0.6, 0.3, 0.42, 1.0)),
        SpanKind::Other => ("span", hsla(0.0, 0.0, 0.4, 1.0)),
    };
    let mut detail = format!("{} · {kind} · {}", span.name, ms(span.end - span.begin));
    if let Some(detail_text) = &span.detail {
        let _ = write!(detail, " · {detail_text}");
    }
    let _ = write!(detail, " · {thread} · frame #{frame}");
    Bar {
        begin: span.begin,
        end: span.end,
        color,
        text: short_name(&span.name).to_owned().into(),
        detail: detail.into(),
    }
}

/// The colors the timeline paints with, other than the bars'.
pub(super) struct TimelineColors {
    pub background: Hsla,
    pub label: Hsla,
    pub text: Hsla,
    pub grid: Hsla,
}

/// Paints `lanes` into `bounds`, as `view` shows them.
pub(super) fn paint(
    lanes: &Lanes,
    view: TimeView,
    bounds: Bounds<Pixels>,
    colors: &TimelineColors,
    font: &Font,
    window: &mut Window,
    cx: &mut App,
) {
    window.paint_quad(fill(bounds, colors.background));
    let bars_area = Bounds {
        origin: point(bounds.left() + px(GUTTER), bounds.top()),
        size: size(bounds.size.width - px(GUTTER), bounds.size.height),
    };
    window.with_content_mask(Some(ContentMask { bounds: bars_area }), |window| {
        // The time axis, from the first frame's start.
        let step = tick_step(view.ns_per_px * 80.0);
        let first = ((view.start - lanes.start as f64) / step).floor() as i64;
        let mut tick = first;
        loop {
            let time = lanes.start as f64 + tick as f64 * step;
            let x =
                f32::from(bounds.left()) + GUTTER + ((time - view.start) / view.ns_per_px) as f32;
            if x > f32::from(bounds.right()) {
                break;
            }
            window.paint_quad(fill(
                Bounds::new(
                    point(px(x), bounds.top() + px(AXIS - 4.0)),
                    size(px(1.0), bounds.size.height),
                ),
                colors.grid,
            ));
            let label = tick_label(tick as f64 * step, step);
            paint_text(
                &label,
                point(px(x + 3.0), bounds.top() + px(2.0)),
                colors.label,
                font,
                window,
                cx,
            );
            tick += 1;
        }
        // Each frame's start, across the lanes.
        for &start in &lanes.frame_starts {
            let x = view.x(bounds, start);
            window.paint_quad(fill(
                Bounds::new(
                    point(px(x), bounds.top() + px(AXIS)),
                    size(px(1.0), bounds.size.height),
                ),
                colors.label.opacity(0.5),
            ));
        }
        for (index, lane) in lanes.lanes.iter().enumerate() {
            let top = f32::from(bounds.top()) + AXIS + index as f32 * LANE;
            for bar in &lane.bars {
                let left = view.x(bounds, bar.begin);
                let right = view.x(bounds, bar.end);
                if right < f32::from(bars_area.left()) || left > f32::from(bars_area.right()) {
                    continue;
                }
                let width = (right - left).max(1.0);
                window.paint_quad(fill(
                    Bounds::new(
                        point(px(left), px(top + 1.0)),
                        size(px(width), px(LANE - 2.0)),
                    ),
                    bar.color,
                ));
                if width > 24.0 {
                    let x = left.max(f32::from(bars_area.left())) + 3.0;
                    let room = right.min(f32::from(bars_area.right())) - x - 2.0;
                    let line = shape(&bar.text, colors.text, font, window);
                    if f32::from(line.width) <= room {
                        line.paint(
                            point(px(x), px(top + 1.5)),
                            px(LANE - 2.0),
                            TextAlign::Left,
                            None,
                            window,
                            cx,
                        )
                        .ok();
                    }
                }
            }
        }
    });
    for (index, lane) in lanes.lanes.iter().enumerate() {
        if let Some(label) = &lane.label {
            let top = f32::from(bounds.top()) + AXIS + index as f32 * LANE;
            if index > 0 {
                window.paint_quad(fill(
                    Bounds::new(
                        point(bounds.left(), px(top)),
                        size(bounds.size.width, px(1.0)),
                    ),
                    colors.grid,
                ));
            }
            paint_text(
                label,
                point(bounds.left() + px(4.0), px(top + 1.5)),
                colors.label,
                font,
                window,
                cx,
            );
        }
    }
}

/// The height of a timeline of `lanes`.
pub(super) fn height(lanes: &Lanes) -> Pixels {
    px(AXIS + lanes.lanes.len() as f32 * LANE + 2.0)
}

/// The bar at `position` in a timeline drawn in `bounds`, or the nearest one
/// within a few pixels of it in its lane.
pub(super) fn hit(
    lanes: &Lanes,
    view: TimeView,
    bounds: Bounds<Pixels>,
    position: Point<Pixels>,
) -> Option<&Bar> {
    let y = f32::from(position.y - bounds.top()) - AXIS;
    if y < 0.0 || f32::from(position.x - bounds.left()) < GUTTER {
        return None;
    }
    let lane = lanes.lanes.get((y / LANE) as usize)?;
    let time = view.time_at(bounds, position.x);
    let slack = 3.0 * view.ns_per_px;
    lane.bars
        .iter()
        .filter(|bar| bar.begin as f64 - slack <= time && time <= bar.end as f64 + slack)
        .min_by(|a, b| {
            let distance = |bar: &Bar| {
                if (bar.begin as f64..=bar.end as f64).contains(&time) {
                    0.0
                } else {
                    (bar.begin as f64 - time)
                        .abs()
                        .min((bar.end as f64 - time).abs())
                }
            };
            distance(a).total_cmp(&distance(b))
        })
}

fn shape(text: &SharedString, color: Hsla, font: &Font, window: &Window) -> gpui::ShapedLine {
    let run = TextRun {
        len: text.len(),
        font: font.clone(),
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    window
        .text_system()
        .shape_line(text.clone(), px(FONT_SIZE), &[run], None)
}

fn paint_text(
    text: &str,
    origin: Point<Pixels>,
    color: Hsla,
    font: &Font,
    window: &mut Window,
    cx: &mut App,
) {
    let text = SharedString::from(text.to_owned());
    shape(&text, color, font, window)
        .paint(origin, px(LANE - 2.0), TextAlign::Left, None, window, cx)
        .ok();
}

/// A round step between axis ticks, at least `at_least` nanoseconds.
fn tick_step(at_least: f64) -> f64 {
    let mut step = 1_000.0;
    loop {
        for factor in [1.0, 2.0, 5.0] {
            if step * factor >= at_least {
                return step * factor;
            }
        }
        step *= 10.0;
    }
}

fn tick_label(time: f64, step: f64) -> String {
    if step < 1e6 {
        format!("{:.0} µs", time / 1e3)
    } else if step < 1e9 {
        format!("{:.0} ms", time / 1e6)
    } else {
        format!("{:.0} s", time / 1e9)
    }
}

/// A color for `name`, the same each time, told apart from other names'.
fn hashed_color(name: &str, saturation: f32, lightness: f32) -> Hsla {
    // FNV-1a.
    let hash = name.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    hsla((hash % 360) as f32 / 360.0, saturation, lightness, 1.0)
}

/// A system's or schedule's name without its module path or generic
/// arguments: `bevy_pumicite::rtx::tlas::tlas_build_system<…>` is
/// `tlas_build_system`.
pub(super) fn short_name(name: &str) -> &str {
    let name = name.split('<').next().unwrap_or(name);
    let mut segments = name.rsplit("::");
    let last = segments.next().unwrap_or(name);
    // A closure is named after what it's in.
    if last.starts_with('{') {
        let start = name.len() - last.len();
        let parent = segments.next().map_or(0, |parent| parent.len() + 2);
        return &name[start - parent..];
    }
    last
}

/// Bevy's task pool threads, shorter: `Compute Task Pool (3)` is `compute 3`.
fn short_thread_name(name: &str) -> String {
    for (pool, short) in [
        ("Async Compute Task Pool (", "async "),
        ("Compute Task Pool (", "compute "),
        ("IO Task Pool (", "io "),
    ] {
        if let Some(rest) = name.strip_prefix(pool) {
            return format!("{short}{}", rest.trim_end_matches(')'));
        }
    }
    if name == "main" {
        return "UI".to_owned();
    }
    name.to_owned()
}

/// A duration, in milliseconds.
pub(super) fn ms(nanos: Nanos) -> String {
    format!("{:.3} ms", nanos as f64 / 1e6)
}
