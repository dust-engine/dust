//! The frame history: a bar per frame, newest on the right, as tall as the time
//! from its start to the next frame's, with its CPU time inside it and its GPU
//! time as a mark.

use std::cell::Cell;
use std::rc::Rc;

use dust_profiler::timeline::{FrameSummary, Nanos};
use gpui::{
    Bounds, Hsla, IntoElement, Pixels, Styled as _, canvas, fill, outline, point, px, size,
};

/// The width of a frame's bar, gap included.
const BAR: f32 = 3.0;
pub(super) const HEIGHT: f32 = 56.0;

pub(super) struct HistoryColors {
    pub background: Hsla,
    pub interval: Hsla,
    pub cpu: Hsla,
    pub gpu: Hsla,
    pub dropped: Hsla,
    pub budget: Hsla,
    pub selected: Hsla,
}

/// The history of `summaries`, with `selected` outlined and a line at
/// `budget`, the interval a frame should take. Where it's drawn goes to
/// `drawn_at`, for [`frame_at`].
pub(super) fn history(
    summaries: Rc<[FrameSummary]>,
    selected: Option<u32>,
    budget: Nanos,
    colors: HistoryColors,
    drawn_at: Rc<Cell<Bounds<Pixels>>>,
) -> impl IntoElement {
    canvas(
        move |bounds, _, _| drawn_at.set(bounds),
        move |bounds, _, window, _| {
            window.paint_quad(fill(bounds, colors.background));
            let count = (f32::from(bounds.size.width) / BAR) as usize;
            let shown = &summaries[summaries.len().saturating_sub(count)..];
            let scale = shown
                .iter()
                .map(|summary| summary.interval.unwrap_or(summary.duration))
                .max()
                .unwrap_or(0)
                .max(budget + budget / 4)
                .max(1) as f32;
            let height =
                |time: Nanos| px(f32::from(bounds.size.height) * (time as f32 / scale).min(1.0));
            for (from_right, summary) in shown.iter().rev().enumerate() {
                let x = bounds.right() - px(BAR * (from_right + 1) as f32);
                let bar = |time: Nanos| {
                    Bounds::new(
                        point(x, bounds.bottom() - height(time)),
                        size(px(BAR - 1.0), height(time)),
                    )
                };
                let interval = summary.interval.unwrap_or(summary.duration);
                let interval_color = if summary.dropped {
                    colors.dropped
                } else {
                    colors.interval
                };
                window.paint_quad(fill(bar(interval), interval_color));
                window.paint_quad(fill(bar(summary.cpu), colors.cpu));
                if let Some(gpu) = summary.gpu {
                    let mark = Bounds::new(
                        point(x, bounds.bottom() - height(gpu)),
                        size(px(BAR - 1.0), px(1.5)),
                    );
                    window.paint_quad(fill(mark, colors.gpu));
                }
                if selected == Some(summary.index) {
                    let column = Bounds::new(
                        point(x - px(1.0), bounds.top()),
                        size(px(BAR + 1.0), bounds.size.height),
                    );
                    window.paint_quad(outline(column, colors.selected, gpui::BorderStyle::Solid));
                }
            }
            let line = Bounds::new(
                point(bounds.left(), bounds.bottom() - height(budget)),
                size(bounds.size.width, px(1.0)),
            );
            window.paint_quad(fill(line, colors.budget));
        },
    )
    .h(px(HEIGHT))
    .w_full()
}

/// Which of `count` frames, newest last, the history drawn in `bounds` shows
/// at `x`: an index into them.
pub(super) fn frame_at(count: usize, bounds: Bounds<Pixels>, x: Pixels) -> Option<usize> {
    if x < bounds.left() || x > bounds.right() {
        return None;
    }
    let from_right = (f32::from(bounds.right() - x) / BAR) as usize;
    count.checked_sub(from_right + 1)
}
