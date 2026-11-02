//! The frame timeline: where each frame's time went, collected from `tracing`.
//!
//! Everything that takes time in a frame is a `tracing` span or event: Bevy's
//! schedules and systems, the runner's waits, input polling, swapchain acquire
//! and present, command submission, and, read back a few frames later, the
//! GPU's work and when the frame reached the display. [`FrameTimelineLayer`],
//! added to the global subscriber by way of [`timeline_layer`], records the
//! spans and events of the schema below into a [`FrameRecord`] per frame, kept
//! by the [`FrameTimeline`] DevTools reads.
//!
//! ```ignore
//! let (timeline_layer, timeline) = dust_profiler::timeline::timeline_layer();
//! tracing_subscriber::registry().with(timeline_layer).init();
//! ```
//!
//! ## Schema
//!
//! The layer records only these spans and events. [`is_timeline`] tells the
//! other layers to leave them out, so they don't print as log lines or as the
//! spans of log lines.
//!
//! | target | name | kind | fields |
//! |---|---|---|---|
//! | `dust::frame` | `frame` | span | `index`: Bevy's `FrameCount` during the frame |
//! | `dust::frame` | `compositor tick`, `display link` | event | `target_offset_ns`, `present_offset_ns`, `period_ns` |
//! | `dust::input` | `poll` | span; event | `count`, `oldest_offset_ns`, `newest_offset_ns` |
//! | `dust::swapchain` | `acquire`, `present` | span | |
//! | `dust::swapchain` | `displayed` | event | `frame`, `offset_ns`, `shown` |
//! | `dust::gpu` | `pass` | span | `name` |
//! | `dust::gpu` | `gpu` | event | `frame`, `pass`, `set`, `queue`, `depth`, `duration_ns`, `begin_offset_ns`, `end_offset_ns`, `calibrated` |
//! | `pumicite` | `submit` | span | `set` |
//! | `bevy_ecs::…` | `schedule`, `system`, `system_commands` | span | `name` |
//! | `gpui::…` | `draw`, `present`, `dispatch_event` | span | |
//!
//! Any other span in a `dust::*` or `pumicite` target is recorded too, as
//! [`SpanKind::Other`]. Spans there whose name starts with `wait`, such as
//! `wait for GPU`, are time spent blocked: the frame's CPU time leaves them
//! out. Events are told apart by name, so producers give it with `name:`:
//!
//! ```ignore
//! tracing::info!(name: "displayed", target: "dust::swapchain", frame, offset_ns, shown = true);
//! ```
//!
//! Times on events are signed nanosecond offsets from the moment the event is
//! emitted (`*_offset_ns`). A producer computes them from its own clock — QPC,
//! `CACurrentMediaTime`, calibrated GPU ticks — and the layer anchors them to
//! when it received the event, so producers share no epoch. An event about a
//! frame that has ended, such as its GPU timings, names it with `frame`;
//! other events and all spans belong to the frame running when they start.

mod layer;
mod store;
#[cfg(test)]
mod tests;

use std::sync::Arc;

pub use layer::{FrameTimelineLayer, TimelineFilter};
pub use store::{
    DisplayTick, Displayed, FrameRecord, FrameSummary, FrameTimeline, GpuRecord, InputRecord,
    Nanos, SpanKind, SpanRecord, TickSource,
};

use tracing::Metadata;
use tracing_subscriber::Layer as _;
use tracing_subscriber::filter::Filtered;
use tracing_subscriber::registry::LookupSpan;

/// The schema's targets.
pub mod target {
    /// The frame and the runner's waits and display ticks.
    pub const FRAME: &str = "dust::frame";
    /// Input polling.
    pub const INPUT: &str = "dust::input";
    /// Swapchain acquire and present, and when frames were displayed.
    pub const SWAPCHAIN: &str = "dust::swapchain";
    /// GPU scopes: their encoding on the CPU and, read back, their GPU times.
    pub const GPU: &str = "dust::gpu";
    /// pumicite's submission spans.
    pub const PUMICITE: &str = "pumicite";
}

/// A span or event name; also a thread's. Names are interned, so a record
/// costs a reference count rather than an allocation.
pub type Name = Arc<str>;

/// How many frames the timeline keeps, and when a frame is done changing.
#[derive(Clone, Copy, Debug)]
pub struct TimelineConfig {
    /// How many frames after its own a frame stops taking late data, such as
    /// its GPU timings and when it was displayed, and is handed to readers.
    pub complete_after: u32,
    /// The complete frames kept with all their records.
    pub max_frames: usize,
    /// The complete frames kept as a [`FrameSummary`].
    pub max_summaries: usize,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            // GPU timings come back after the frame's command buffers finish,
            // at most 3 frames later, and display feedback a few frames after
            // that.
            complete_after: 8,
            max_frames: 300,
            max_summaries: 3600,
        }
    }
}

/// Makes the layer that records the timeline, already filtered to the schema,
/// and the [`FrameTimeline`] it records into. Add the layer to the global
/// subscriber before the Bevy app is built: Bevy names its systems' spans when
/// it makes them.
pub fn timeline_layer<S>() -> (
    Filtered<FrameTimelineLayer, TimelineFilter, S>,
    FrameTimeline,
)
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    timeline_layer_with(TimelineConfig::default())
}

/// [`timeline_layer`], keeping frames as `config` says.
pub fn timeline_layer_with<S>(
    config: TimelineConfig,
) -> (
    Filtered<FrameTimelineLayer, TimelineFilter, S>,
    FrameTimeline,
)
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    let timeline = FrameTimeline::new(config);
    let layer = FrameTimelineLayer::new(timeline.clone()).with_filter(TimelineFilter);
    (layer, timeline)
}

/// Whether the timeline records spans or events from `metadata`'s callsite.
/// Other layers leave these out.
pub fn is_timeline(metadata: &Metadata<'_>) -> bool {
    if metadata.is_span() {
        layer::span_kind(metadata).is_some()
    } else {
        layer::event_kind(metadata).is_some()
    }
}
