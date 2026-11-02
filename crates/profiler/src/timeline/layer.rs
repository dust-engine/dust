//! The `tracing` layer that records the timeline, and its filter.

use std::cell::RefCell;
use std::fmt::{self, Write as _};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber, span};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, Filter};
use tracing_subscriber::registry::LookupSpan;

use super::Name;
use super::store::{
    DisplayTick, Displayed, FrameTimeline, GpuRecord, InputRecord, ItemData, Nanos, Shared,
    SpanKind, SpanRecord, ThreadBuffer, TickSource,
};
use super::target::{FRAME, GPU, INPUT, PUMICITE, SWAPCHAIN};

/// Records the [schema](super)'s spans and events into a [`FrameTimeline`].
/// Made by [`timeline_layer`](super::timeline_layer), with its
/// [`TimelineFilter`].
///
/// A span is timed from its enter to its exit, on the thread it ran on, and
/// nested in the spans open on that thread then; parents don't matter, since
/// Bevy makes its systems' spans without one. Each thread keeps its records to
/// itself until the `frame` span ends, when the thread running the frame
/// collects them.
pub struct FrameTimelineLayer {
    timeline: FrameTimeline,
    /// Tells this layer's per-thread state from another's, as in tests, which
    /// make a layer each.
    id: usize,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl FrameTimelineLayer {
    pub(super) fn new(timeline: FrameTimeline) -> Self {
        FrameTimelineLayer {
            timeline,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    fn shared(&self) -> &Shared {
        &self.timeline.shared
    }

    /// Runs `f` on the calling thread's state for this layer, made on first
    /// use. `None` while the thread is exiting, or if `f` would run inside
    /// itself.
    fn with_local<R>(&self, f: impl FnOnce(&mut Local) -> R) -> Option<R> {
        LOCALS
            .try_with(|locals| {
                let mut locals = locals.try_borrow_mut().ok()?;
                let at = match locals.iter().position(|local| local.layer == self.id) {
                    Some(at) => at,
                    None => {
                        locals.push(Local {
                            layer: self.id,
                            buffer: self.shared().register_thread(),
                            stack: Vec::new(),
                        });
                        locals.len() - 1
                    }
                };
                Some(f(&mut locals[at]))
            })
            .ok()
            .flatten()
    }
}

thread_local! {
    static LOCALS: RefCell<Vec<Local>> = const { RefCell::new(Vec::new()) };
}

/// A thread's state for one layer.
struct Local {
    layer: usize,
    buffer: Arc<ThreadBuffer>,
    /// The recorded spans open on the thread, outermost first.
    stack: Vec<Open>,
}

struct Open {
    id: span::Id,
    begin: Nanos,
    /// The frame running when it was entered.
    frame: Option<u32>,
    info: SpanInfo,
}

/// What the layer keeps of a span, in its extensions.
#[derive(Clone)]
struct SpanInfo {
    kind: SpanKind,
    name: Name,
    detail: Option<Name>,
    /// A `frame` span's frame.
    index: Option<u32>,
}

impl<S> Layer<S> for FrameTimelineLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let metadata = attrs.metadata();
        let Some(kind) = span_kind(metadata) else {
            return;
        };
        let mut fields = SpanFields::new(self.shared());
        attrs.record(&mut fields);
        // Bevy's spans and GPU scopes all have one name and give the
        // schedule, system or scope in a field.
        let name = match (kind, fields.name) {
            (
                SpanKind::Schedule | SpanKind::System | SpanKind::Commands | SpanKind::Pass,
                Some(name),
            ) => name,
            _ => self.shared().intern(metadata.name()),
        };
        let info = SpanInfo {
            kind,
            name,
            detail: fields.detail,
            index: fields.index,
        };
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(info);
        }
    }

    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        let Some(info) = ctx
            .span(id)
            .and_then(|span| span.extensions().get::<SpanInfo>().cloned())
        else {
            return;
        };
        let shared = self.shared();
        if info.kind == SpanKind::Frame
            && let Some(index) = info.index
        {
            shared.set_frame(index);
        }
        let begin = shared.now();
        let frame = shared.frame();
        self.with_local(|local| {
            local.stack.push(Open {
                id: id.clone(),
                begin,
                frame,
                info,
            })
        });
    }

    fn on_exit(&self, id: &span::Id, _: Context<'_, S>) {
        let shared = self.shared();
        let end = shared.now();
        let recording = shared.is_recording();
        let ended = self.with_local(|local| {
            // Spans are exited in reverse order of entering, so this is
            // nearly always the last one.
            let depth = local.stack.iter().rposition(|open| open.id == *id)?;
            let open = local.stack.remove(depth);
            let frame = open.frame?;
            let kind = open.info.kind;
            if recording {
                local.buffer.push(
                    frame,
                    ItemData::Span(SpanRecord {
                        kind,
                        name: open.info.name,
                        detail: open.info.detail,
                        thread: local.buffer.index,
                        depth: depth as u16,
                        begin: open.begin,
                        end,
                    }),
                );
            }
            Some(kind)
        });
        if ended.flatten() == Some(SpanKind::Frame) {
            shared.collect();
        }
    }

    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let Some(kind) = event_kind(event.metadata()) else {
            return;
        };
        let shared = self.shared();
        if !shared.is_recording() {
            return;
        }
        let at = shared.now();
        let mut fields = EventFields::new(shared);
        event.record(&mut fields);
        let running = shared.frame();
        let resolve = |offset: Option<i64>| offset.map(|offset| at.saturating_add_signed(offset));
        let (frame, data) = match kind {
            EventKind::Tick(source) => (
                running,
                ItemData::Tick(DisplayTick {
                    source,
                    at,
                    target: resolve(fields.target_offset_ns),
                    present: resolve(fields.present_offset_ns),
                    period: fields.period_ns,
                }),
            ),
            EventKind::Input => (
                running,
                ItemData::Input(InputRecord {
                    at,
                    count: fields.count.unwrap_or(0) as u32,
                    oldest: resolve(fields.oldest_offset_ns),
                    newest: resolve(fields.newest_offset_ns),
                }),
            ),
            EventKind::Displayed => (
                fields.frame.or(running),
                ItemData::Displayed(Displayed {
                    at: resolve(fields.offset_ns),
                    shown: fields.shown.unwrap_or(true),
                }),
            ),
            EventKind::Gpu => {
                let begin = resolve(fields.begin_offset_ns);
                let end = resolve(fields.end_offset_ns);
                let duration = fields
                    .duration_ns
                    .or_else(|| Some(end?.saturating_sub(begin?)))
                    .unwrap_or(0);
                let depth = fields
                    .depth
                    .unwrap_or(if fields.pass.is_some() { 1 } else { 0 });
                let Some(name) = fields.pass.or_else(|| fields.set.clone()) else {
                    return;
                };
                (
                    fields.frame.or(running),
                    ItemData::Gpu(GpuRecord {
                        name,
                        set: fields.set,
                        queue: fields.queue,
                        depth: depth as u16,
                        begin,
                        end,
                        duration,
                        calibrated: fields.calibrated.unwrap_or(false),
                    }),
                )
            }
        };
        if let Some(frame) = frame {
            self.with_local(|local| local.buffer.push(frame, data));
        }
    }
}

/// The filter of [`FrameTimelineLayer`]: the [schema](super)'s spans and
/// events, decided once per callsite.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimelineFilter;

impl<S> Filter<S> for TimelineFilter {
    fn enabled(&self, metadata: &Metadata<'_>, _: &Context<'_, S>) -> bool {
        super::is_timeline(metadata)
    }

    fn callsite_enabled(&self, metadata: &'static Metadata<'static>) -> Interest {
        if super::is_timeline(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    /// The schema's spans and events are all at `INFO`.
    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::INFO)
    }
}

/// The kind of span the schema says a callsite makes, if it's in the schema.
pub(super) fn span_kind(metadata: &Metadata<'_>) -> Option<SpanKind> {
    let (target, name) = (metadata.target(), metadata.name());
    let ours = target == PUMICITE || target.starts_with("dust::");
    if ours && name.starts_with("wait") {
        return Some(SpanKind::Wait);
    }
    match (target, name) {
        (FRAME, "frame") => Some(SpanKind::Frame),
        (INPUT, "poll") => Some(SpanKind::Input),
        (SWAPCHAIN, _) => Some(SpanKind::Swapchain),
        (GPU, "pass") => Some(SpanKind::Pass),
        (PUMICITE, "submit") => Some(SpanKind::Submit),
        _ if ours => Some(SpanKind::Other),
        (_, "schedule") if target.starts_with("bevy_ecs") => Some(SpanKind::Schedule),
        (_, "system") if target.starts_with("bevy_ecs") => Some(SpanKind::System),
        (_, "system_commands") if target.starts_with("bevy_ecs") => Some(SpanKind::Commands),
        (_, "draw" | "present" | "dispatch_event") if target.starts_with("gpui") => {
            Some(SpanKind::Ui)
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum EventKind {
    Tick(TickSource),
    Input,
    Displayed,
    Gpu,
}

/// The kind of event the schema says a callsite makes, if it's in the schema.
pub(super) fn event_kind(metadata: &Metadata<'_>) -> Option<EventKind> {
    match (metadata.target(), metadata.name()) {
        (FRAME, "compositor tick") => Some(EventKind::Tick(TickSource::Compositor)),
        (FRAME, "display link") => Some(EventKind::Tick(TickSource::DisplayLink)),
        (INPUT, "poll") => Some(EventKind::Input),
        (SWAPCHAIN, "displayed") => Some(EventKind::Displayed),
        (GPU, "gpu") => Some(EventKind::Gpu),
        _ => None,
    }
}

/// The fields of a span the layer keeps.
struct SpanFields<'a> {
    shared: &'a Shared,
    name: Option<Name>,
    detail: Option<Name>,
    index: Option<u32>,
}

impl<'a> SpanFields<'a> {
    fn new(shared: &'a Shared) -> Self {
        SpanFields {
            shared,
            name: None,
            detail: None,
            index: None,
        }
    }

    fn text(&mut self, field: &Field, value: &str) {
        match field.name() {
            "name" => self.name = Some(self.shared.intern(value)),
            "set" => self.detail = Some(self.shared.intern(value)),
            _ => {}
        }
    }
}

impl Visit for SpanFields<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.text(field, value);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "index" {
            self.index = u32::try_from(value).ok();
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "index" {
            self.index = u32::try_from(value).ok();
        }
    }

    /// Bevy gives a schedule's name as its label's `Debug`.
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if matches!(field.name(), "name" | "set") {
            let mut text = String::new();
            let _ = write!(text, "{value:?}");
            self.text(field, &text);
        }
    }
}

/// The fields of the schema's events.
struct EventFields<'a> {
    shared: &'a Shared,
    frame: Option<u32>,
    offset_ns: Option<i64>,
    target_offset_ns: Option<i64>,
    present_offset_ns: Option<i64>,
    begin_offset_ns: Option<i64>,
    end_offset_ns: Option<i64>,
    oldest_offset_ns: Option<i64>,
    newest_offset_ns: Option<i64>,
    period_ns: Option<u64>,
    duration_ns: Option<u64>,
    count: Option<u64>,
    depth: Option<u64>,
    shown: Option<bool>,
    calibrated: Option<bool>,
    pass: Option<Name>,
    set: Option<Name>,
    queue: Option<Name>,
}

impl<'a> EventFields<'a> {
    fn new(shared: &'a Shared) -> Self {
        EventFields {
            shared,
            frame: None,
            offset_ns: None,
            target_offset_ns: None,
            present_offset_ns: None,
            begin_offset_ns: None,
            end_offset_ns: None,
            oldest_offset_ns: None,
            newest_offset_ns: None,
            period_ns: None,
            duration_ns: None,
            count: None,
            depth: None,
            shown: None,
            calibrated: None,
            pass: None,
            set: None,
            queue: None,
        }
    }

    fn integer(&mut self, field: &Field, value: i128) {
        let signed = i64::try_from(value).ok();
        let unsigned = u64::try_from(value).ok();
        match field.name() {
            "frame" => self.frame = u32::try_from(value).ok(),
            "offset_ns" => self.offset_ns = signed,
            "target_offset_ns" => self.target_offset_ns = signed,
            "present_offset_ns" => self.present_offset_ns = signed,
            "begin_offset_ns" => self.begin_offset_ns = signed,
            "end_offset_ns" => self.end_offset_ns = signed,
            "oldest_offset_ns" => self.oldest_offset_ns = signed,
            "newest_offset_ns" => self.newest_offset_ns = signed,
            "period_ns" => self.period_ns = unsigned,
            "duration_ns" => self.duration_ns = unsigned,
            "count" => self.count = unsigned,
            "depth" => self.depth = unsigned,
            _ => {}
        }
    }

    fn text(&mut self, field: &Field, value: &str) {
        if !matches!(field.name(), "pass" | "set" | "queue") {
            return;
        }
        let name = Some(self.shared.intern(value));
        match field.name() {
            "pass" => self.pass = name,
            "set" => self.set = name,
            _ => self.queue = name,
        }
    }
}

impl Visit for EventFields<'_> {
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.integer(field, value.into());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.integer(field, value.into());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        match field.name() {
            "shown" => self.shown = Some(value),
            "calibrated" => self.calibrated = Some(value),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.text(field, value);
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if matches!(field.name(), "pass" | "set" | "queue") {
            let mut text = String::new();
            let _ = write!(text, "{value:?}");
            self.text(field, &text);
        }
    }
}
