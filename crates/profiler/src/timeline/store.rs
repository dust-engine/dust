//! What the timeline keeps: each frame's records, and the summaries of older
//! frames.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use super::{Name, TimelineConfig};

/// Nanoseconds since the timeline was made.
pub type Nanos = u64;

/// What a span measures.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SpanKind {
    /// A whole Bevy update: `dust::frame` `frame`.
    Frame,
    /// A Bevy schedule run.
    Schedule,
    /// A Bevy system run; for a render pass, that includes recording its
    /// commands.
    System,
    /// Applying a Bevy system's commands.
    Commands,
    /// The CPU side of a GPU scope: recording its commands.
    Pass,
    /// Time blocked, waiting for the GPU, the display or the compositor.
    Wait,
    /// A command buffer submission.
    Submit,
    /// Swapchain acquire or present.
    Swapchain,
    /// Input polling.
    Input,
    /// GPUI's work on the UI thread.
    Ui,
    /// Any other span in a `dust::*` or `pumicite` target.
    Other,
}

/// A span, from when it was entered to when it was exited.
#[derive(Clone, Debug)]
pub struct SpanRecord {
    pub kind: SpanKind,
    pub name: Name,
    /// What the span is about beyond its name, such as the submission set a
    /// `submit` submits.
    pub detail: Option<Name>,
    /// The thread it ran on: an index into [`FrameTimeline::threads`].
    pub thread: u32,
    /// How many recorded spans were open around it on its thread.
    pub depth: u16,
    pub begin: Nanos,
    pub end: Nanos,
}

/// GPU work, read back after the frame: a GPU scope, or a whole submission.
#[derive(Clone, Debug)]
pub struct GpuRecord {
    /// The scope's name; the submission set's, for a whole submission.
    pub name: Name,
    /// The submission set it was recorded in, if known.
    pub set: Option<Name>,
    /// The queue it ran on, if known.
    pub queue: Option<Name>,
    /// 0 for a whole submission, 1 for a scope in one, 2 for a scope in that,
    /// and so on.
    pub depth: u16,
    /// When it started and ended, on the timeline's clock; `None` when the
    /// GPU's clock couldn't be placed on it.
    pub begin: Option<Nanos>,
    pub end: Option<Nanos>,
    pub duration: Nanos,
    /// Whether `begin` and `end` come from timestamps calibrated against the
    /// CPU's clock, rather than estimated.
    pub calibrated: bool,
}

/// One input poll.
#[derive(Clone, Debug)]
pub struct InputRecord {
    /// When the poll handed the input to the game.
    pub at: Nanos,
    /// How many readings or events it took.
    pub count: u32,
    /// When the oldest and newest of them happened, if the platform says.
    pub oldest: Option<Nanos>,
    pub newest: Option<Nanos>,
}

/// What reported a [`DisplayTick`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TickSource {
    /// The Windows compositor's clock.
    Compositor,
    /// A `CAMetalDisplayLink`, on macOS.
    DisplayLink,
}

/// A display refresh the runner woke for.
#[derive(Clone, Debug)]
pub struct DisplayTick {
    pub source: TickSource,
    /// When the runner got it.
    pub at: Nanos,
    /// The deadline: when a frame started now should be done.
    pub target: Option<Nanos>,
    /// When a frame presented for this refresh will be shown.
    pub present: Option<Nanos>,
    /// The display's refresh interval.
    pub period: Option<Nanos>,
}

/// When a frame reached the display.
#[derive(Clone, Copy, Debug)]
pub struct Displayed {
    /// When it was shown; `None` if it wasn't, or the platform didn't say.
    pub at: Option<Nanos>,
    /// Whether it was shown at all; false for a frame the compositor dropped.
    pub shown: bool,
}

/// Everything recorded about one frame.
#[derive(Clone, Debug)]
pub struct FrameRecord {
    /// Bevy's `FrameCount` during the frame.
    pub index: u32,
    /// The `frame` span; both zero if it wasn't recorded.
    pub begin: Nanos,
    pub end: Nanos,
    /// Its spans, the `frame` span among them.
    pub spans: Vec<SpanRecord>,
    pub gpu: Vec<GpuRecord>,
    pub ticks: Vec<DisplayTick>,
    pub inputs: Vec<InputRecord>,
    pub displayed: Option<Displayed>,
}

impl FrameRecord {
    fn new(index: u32) -> Self {
        FrameRecord {
            index,
            begin: 0,
            end: 0,
            spans: Vec::new(),
            gpu: Vec::new(),
            ticks: Vec::new(),
            inputs: Vec::new(),
            displayed: None,
        }
    }
}

/// A frame's figures, for the frame history.
#[derive(Clone, Copy, Debug)]
pub struct FrameSummary {
    pub index: u32,
    pub begin: Nanos,
    /// From the frame's start to the next's; `None` if the next wasn't
    /// recorded.
    pub interval: Option<Nanos>,
    /// The `frame` span's length.
    pub duration: Nanos,
    /// `duration` without the time inside it spent blocked in `wait` spans.
    pub cpu: Nanos,
    /// Time blocked in `wait` spans from the frame's start to the next's.
    pub wait: Nanos,
    /// GPU busy time: the union of its timed submissions, or else the sum of
    /// its outermost GPU scopes; `None` without GPU timings.
    pub gpu: Option<Nanos>,
    /// From the frame's start to its display.
    pub display_latency: Option<Nanos>,
    /// From the oldest input it read to its display.
    pub input_latency: Option<Nanos>,
    /// Whether the compositor dropped it.
    pub dropped: bool,
}

/// The frames a [`FrameTimelineLayer`](super::FrameTimelineLayer) recorded. A
/// cheap handle: clones share the frames.
///
/// A frame shows here once it's complete, `complete_after` frames after its
/// own (see [`TimelineConfig`]), and doesn't change after that.
#[derive(Clone)]
pub struct FrameTimeline {
    pub(super) shared: Arc<Shared>,
}

impl FrameTimeline {
    pub(super) fn new(config: TimelineConfig) -> Self {
        FrameTimeline {
            shared: Arc::new(Shared {
                epoch: Instant::now(),
                config,
                recording: AtomicBool::new(true),
                current_frame: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                names: Mutex::new(HashSet::new()),
                threads: Mutex::new(Threads::default()),
                store: Mutex::new(Store::default()),
            }),
        }
    }

    /// Changes whenever frames complete, so a reader can skip reading when
    /// nothing did.
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    pub fn is_recording(&self) -> bool {
        self.shared.recording.load(Ordering::Relaxed)
    }

    /// Stops or resumes recording; frames run meanwhile are missing.
    pub fn set_recording(&self, recording: bool) {
        self.shared.recording.store(recording, Ordering::Relaxed);
    }

    /// The names of the threads spans ran on, indexed by
    /// [`SpanRecord::thread`].
    pub fn threads(&self) -> Vec<Name> {
        lock(&self.shared.threads).names.clone()
    }

    /// The complete frames newer than `after`, oldest first, with all their
    /// records: up to [`TimelineConfig::max_frames`] of them.
    pub fn frames_after(&self, after: Option<u32>) -> Vec<Arc<FrameRecord>> {
        let store = lock(&self.shared.store);
        store
            .complete
            .iter()
            .filter(|frame| after.is_none_or(|after| frame.index > after))
            .cloned()
            .collect()
    }

    /// The summaries of the complete frames newer than `after`, oldest first:
    /// up to [`TimelineConfig::max_summaries`] of them.
    pub fn summaries_after(&self, after: Option<u32>) -> Vec<FrameSummary> {
        let store = lock(&self.shared.store);
        store
            .summaries
            .iter()
            .filter(|summary| after.is_none_or(|after| summary.index > after))
            .copied()
            .collect()
    }
}

/// The state shared by the layer, the threads recording into it, and readers.
pub(super) struct Shared {
    epoch: Instant,
    config: TimelineConfig,
    recording: AtomicBool,
    /// The frame running, or between frames the last one, plus one; 0 before
    /// the first.
    current_frame: AtomicU64,
    generation: AtomicU64,
    names: Mutex<HashSet<Name>>,
    threads: Mutex<Threads>,
    store: Mutex<Store>,
}

#[derive(Default)]
struct Threads {
    /// The buffers of the threads that recorded; dropped once their thread
    /// has exited and they're drained.
    buffers: Vec<Arc<ThreadBuffer>>,
    /// Every thread's name, by [`ThreadBuffer::index`].
    names: Vec<Name>,
}

/// Where one thread puts its records until the frame ends. Only that thread
/// pushes, so the lock is contended only while the frame's records are taken.
pub(super) struct ThreadBuffer {
    pub(super) index: u32,
    items: Mutex<Vec<Item>>,
}

impl ThreadBuffer {
    pub(super) fn push(&self, frame: u32, data: ItemData) {
        lock(&self.items).push(Item { frame, data });
    }
}

/// A record, and the frame it belongs to.
pub(super) struct Item {
    frame: u32,
    data: ItemData,
}

pub(super) enum ItemData {
    Span(SpanRecord),
    Gpu(GpuRecord),
    Tick(DisplayTick),
    Input(InputRecord),
    Displayed(Displayed),
}

impl Shared {
    pub(super) fn now(&self) -> Nanos {
        self.epoch.elapsed().as_nanos() as Nanos
    }

    pub(super) fn is_recording(&self) -> bool {
        self.recording.load(Ordering::Relaxed)
    }

    /// The frame running, or between frames the last one.
    pub(super) fn frame(&self) -> Option<u32> {
        match self.current_frame.load(Ordering::Relaxed) {
            0 => None,
            frame => Some((frame - 1) as u32),
        }
    }

    pub(super) fn set_frame(&self, index: u32) {
        self.current_frame
            .store(u64::from(index) + 1, Ordering::Relaxed);
    }

    pub(super) fn intern(&self, name: &str) -> Name {
        let mut names = lock(&self.names);
        if let Some(name) = names.get(name) {
            return name.clone();
        }
        let name: Name = name.into();
        names.insert(name.clone());
        name
    }

    /// Gives the calling thread a buffer.
    pub(super) fn register_thread(&self) -> Arc<ThreadBuffer> {
        let thread = std::thread::current();
        let name = match thread.name() {
            Some(name) => self.intern(name),
            None => self.intern(&format!("{:?}", thread.id())),
        };
        let mut threads = lock(&self.threads);
        let buffer = Arc::new(ThreadBuffer {
            index: threads.names.len() as u32,
            items: Mutex::new(Vec::new()),
        });
        threads.names.push(name);
        threads.buffers.push(buffer.clone());
        buffer
    }

    /// Moves every thread's records into their frames, and completes the
    /// frames old enough. Called as each frame ends, on the thread running it.
    pub(super) fn collect(&self) {
        let buffers = {
            let mut threads = lock(&self.threads);
            // An exited thread's handle to its buffer is gone.
            threads
                .buffers
                .retain(|buffer| Arc::strong_count(buffer) > 1 || !lock(&buffer.items).is_empty());
            threads.buffers.clone()
        };
        let mut items = Vec::new();
        for buffer in &buffers {
            items.append(&mut lock(&buffer.items));
        }
        let mut store = lock(&self.store);
        for item in items {
            store.merge(item);
        }
        if store.complete(&self.config) {
            self.generation.fetch_add(1, Ordering::Release);
        }
    }
}

#[derive(Default)]
struct Store {
    /// The frames still taking records, oldest first.
    pending: VecDeque<FrameRecord>,
    /// The complete frames, oldest first. Shared, so readers copy pointers
    /// rather than records while they hold the lock the frame's end waits on.
    complete: VecDeque<Arc<FrameRecord>>,
    summaries: VecDeque<FrameSummary>,
    /// The newest complete frame.
    completed: Option<u32>,
    /// The newest frame whose `frame` span ended.
    latest: Option<u32>,
}

impl Store {
    fn merge(&mut self, item: Item) {
        let ended = matches!(&item.data, ItemData::Span(span) if span.kind == SpanKind::Frame);
        let Some(frame) = self.frame_mut(item.frame) else {
            return;
        };
        match item.data {
            ItemData::Span(span) => {
                if span.kind == SpanKind::Frame {
                    frame.begin = span.begin;
                    frame.end = span.end;
                }
                frame.spans.push(span);
            }
            ItemData::Gpu(gpu) => frame.gpu.push(gpu),
            ItemData::Tick(tick) => frame.ticks.push(tick),
            ItemData::Input(input) => frame.inputs.push(input),
            ItemData::Displayed(displayed) => frame.displayed = Some(displayed),
        }
        if ended {
            self.latest = Some(
                self.latest
                    .map_or(item.frame, |latest| latest.max(item.frame)),
            );
        }
    }

    /// The pending frame `index`, made if it's new; `None` once it's complete.
    fn frame_mut(&mut self, index: u32) -> Option<&mut FrameRecord> {
        if self.completed.is_some_and(|completed| index <= completed) {
            return None;
        }
        let at = match self.pending.iter().rposition(|frame| frame.index <= index) {
            Some(at) if self.pending[at].index == index => at,
            Some(at) => {
                self.pending.insert(at + 1, FrameRecord::new(index));
                at + 1
            }
            None => {
                self.pending.push_front(FrameRecord::new(index));
                0
            }
        };
        Some(&mut self.pending[at])
    }

    /// Completes the frames `complete_after` frames older than the latest, and
    /// drops what's past the limits. Whether any frame completed.
    fn complete(&mut self, config: &TimelineConfig) -> bool {
        let Some(latest) = self.latest else {
            return false;
        };
        let mut changed = false;
        while self
            .pending
            .front()
            .is_some_and(|frame| frame.index.saturating_add(config.complete_after) <= latest)
        {
            let frame = self.pending.pop_front().unwrap();
            let next = self
                .pending
                .front()
                .map(|next| next.begin)
                .filter(|&begin| begin > 0);
            self.summaries.push_back(summarize(&frame, next));
            self.completed = Some(frame.index);
            self.complete.push_back(Arc::new(frame));
            changed = true;
        }
        while self.summaries.len() > config.max_summaries {
            self.summaries.pop_front();
        }
        while self.complete.len() > config.max_frames {
            self.complete.pop_front();
        }
        changed
    }
}

fn summarize(frame: &FrameRecord, next_begin: Option<Nanos>) -> FrameSummary {
    let duration = frame.end.saturating_sub(frame.begin);
    let waits = || {
        frame
            .spans
            .iter()
            .filter(|span| span.kind == SpanKind::Wait)
            .map(|span| (span.begin, span.end))
    };
    let blocked_inside = union_len(waits(), frame.begin, frame.end);
    let until = next_begin.unwrap_or(frame.end).max(frame.end);
    let wait = union_len(waits(), frame.begin, until);

    // The outermost GPU records: the submissions when they're timed, else the
    // scopes.
    let gpu = frame
        .gpu
        .iter()
        .map(|gpu| gpu.depth)
        .min()
        .map(|outermost| {
            let outer = frame.gpu.iter().filter(|gpu| gpu.depth == outermost);
            let timed: Option<Vec<_>> = outer
                .clone()
                .map(|gpu| Some((gpu.begin?, gpu.end?)))
                .collect();
            match timed {
                Some(intervals) => union_len(intervals.into_iter(), 0, Nanos::MAX),
                None => outer.map(|gpu| gpu.duration).sum(),
            }
        });

    let shown_at = frame
        .displayed
        .filter(|displayed| displayed.shown)
        .and_then(|displayed| displayed.at);
    let oldest_input = frame
        .inputs
        .iter()
        .map(|input| input.oldest.unwrap_or(input.at))
        .min();
    FrameSummary {
        index: frame.index,
        begin: frame.begin,
        interval: next_begin.map(|next| next.saturating_sub(frame.begin)),
        duration,
        cpu: duration.saturating_sub(blocked_inside),
        wait,
        gpu,
        display_latency: shown_at.map(|at| at.saturating_sub(frame.begin)),
        input_latency: shown_at
            .zip(oldest_input)
            .map(|(at, input)| at.saturating_sub(input)),
        dropped: frame.displayed.is_some_and(|displayed| !displayed.shown),
    }
}

/// The length of the union of `intervals`, clipped to `from..to`.
fn union_len(intervals: impl Iterator<Item = (Nanos, Nanos)>, from: Nanos, to: Nanos) -> Nanos {
    let mut clipped: Vec<_> = intervals
        .map(|(begin, end)| (begin.max(from), end.min(to)))
        .filter(|(begin, end)| begin < end)
        .collect();
    clipped.sort_unstable();
    let mut total = 0;
    let mut covered_to = 0;
    for (begin, end) in clipped {
        let begin = begin.max(covered_to);
        if end > begin {
            total += end - begin;
            covered_to = end;
        }
    }
    total
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
