//! GPU timing with timestamp queries, reported to the frame timeline.
//!
//! [`GpuProfiler`] times two things. Every submission of a submission set, from
//! the start of its command buffer to its end, through pumicite's
//! [`SubmissionHook`]: that covers all of a frame's GPU work, including what no
//! pass marks, such as uploads, acceleration structure builds and barriers.
//! And the scopes render passes mark on the command encoder with
//! [`GpuTimerCommands::timing_scope`]:
//!
//! ```ignore
//! ctx.record(move |encoder| {
//!     // … bind pipeline, set up descriptors …
//!     encoder.timing_scope(profiler.as_deref_mut(), "primary ray", |encoder| {
//!         encoder.trace_rays(/* … */);
//!     });
//! });
//! ```
//!
//! A scope also times its recording on the CPU, as a `dust::gpu` `pass` span,
//! and labels its commands for graphics debuggers. `timing_scope` takes
//! `Option<&mut GpuProfiler>`, so without [`GpuProfilerPlugin`] (the resource
//! is absent, `None`), it still does those but times nothing on the GPU.
//!
//! A few frames later the profiler reads each timing back and sends it as a
//! `gpu` event of the frame timeline (see [`timeline`]) for the frame that
//! recorded it: its duration, `(end - start) * timestampPeriod`, and, when the
//! device has `VK_KHR_calibrated_timestamps` (or the EXT), when it started and
//! ended on the CPU's clock.
//!
//! ## Query pool as a dynamic ring
//!
//! The [`QueryPool`] is treated as a ring of timestamp slots. Each timing
//! dynamically grabs the next two free slots (start + stop), wrapping at the
//! end of the pool and reusing slots whose frame has already been read back.
//! There is no fixed per-frame budget — a frame uses as many slots as it has
//! timings. Slots are reset from the host with `vkResetQueryPool` immediately
//! before reuse, which is safe because the ring only reuses slots that belong
//! to already-completed frames.
//!
//! The pool is shared slot-by-slot between the host and every in-flight frame,
//! so it is not wrapped in a [`GPUMutex`](pumicite::sync::GPUMutex): a
//! `GPUMutex` hands the *whole* pool to one owner at a time, and the host could
//! then only read results once no frame at all is in flight. Instead the pool
//! is an `Arc<QueryPool>`, accessed through pumicite's `_unchecked` query
//! methods, and the slot allocation and `free_frame` uphold the per-slot
//! ownership those methods require.
//!
//! When the in-flight working set genuinely doesn't fit (readback fell behind
//! or a frame has an unusually large number of timings), the pool is
//! reallocated at a larger capacity. Every command buffer that writes a
//! timestamp [`CommandEncoder::retain`]s a clone of the `Arc`, so dropping the
//! old pool defers its destruction until the GPU stops referencing it.

use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use bevy::diagnostic::FrameCount;
use bevy::ecs::schedule::InternedSystemSet;
use bevy::prelude::*;
use bevy_pumicite::{CreateDevice, SubmissionHook, SubmissionHooks};
use glam::Vec4;
use pumicite::{
    Device, ash::vk, command::CommandEncoder, device::DeviceBuilder, query::QueryPool,
    utils::AsVkHandle,
};
use tracing::warn;

use crate::timeline;

/// Initial pool size in timestamp slots. Grows on demand.
const INITIAL_CAPACITY: u32 = 256;
/// Upper bound on pool growth, so a runaway never allocates without limit.
const MAX_CAPACITY: u32 = 1 << 16;
/// Cap on frames awaiting readback. Bounds memory if results never drain.
const MAX_PENDING: usize = 16;
/// How often the GPU's clock is sampled against the CPU's, for their drift.
const CALIBRATION_INTERVAL: Duration = Duration::from_secs(1);

/// Pipeline stage sampled by the start/stop timestamps. `ALL_COMMANDS` records
/// the moment every preceding command has fully completed, so the difference
/// between two of them is the device time spent on the bracketed work.
const TS_STAGE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::ALL_COMMANDS;

/// The color of the debug labels around scopes.
const LABEL_COLOR: Vec4 = Vec4::new(0.3, 0.8, 0.8, 1.0);

/// One query result read back with `WITH_AVAILABILITY | TYPE_64`: the 64-bit
/// timestamp value plus a 64-bit availability flag.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TimestampResult {
    value: u64,
    available: u64,
}

/// What a pair of timestamps timed.
enum Timed {
    /// A whole submission.
    Submission,
    /// A scope, `depth` scopes deep in its submission (1 for the outermost).
    Scope { label: &'static str, depth: u16 },
}

/// A completed timing. Its two timestamps occupy the adjacent slots `start`
/// (begin) and `start + 1` (end).
struct Record {
    start: u32,
    timed: Timed,
    /// The submission set it was recorded in, if known.
    set: Option<Arc<str>>,
    /// The queue family of its command buffer, if known.
    queue_family: Option<u32>,
}

/// A frame's timings plus the ring arc they occupy. Used both for the frame
/// currently being recorded and for frames queued awaiting readback.
///
/// The frame occupies the contiguous (wrapping) arc `[start, start + slots)`,
/// which is host-reset and freed back to the ring once the frame is resolved
/// or dropped.
#[derive(Default)]
struct Frame {
    /// Bevy's `FrameCount` during the frame, which names it on the frame
    /// timeline.
    index: u32,
    /// First ring slot this frame occupies (`head` when the frame began).
    start: u32,
    /// Slots this frame took from the ring, two per timing begun.
    slots: u32,
    records: Vec<Record>,
}

/// A submission whose command buffer is being recorded.
struct OpenSubmission {
    set: Arc<str>,
    queue_family: u32,
    /// Its own begin slot, or `None` if its queue can't write timestamps.
    start: Option<u32>,
    /// Scopes open in it now.
    depth: u16,
}

/// A GPU timestamp and the CPU's clock at the same moment.
#[derive(Clone, Copy)]
struct Calibration {
    gpu: u64,
    cpu: Instant,
}

/// A scope begun by [`Shared::scope_begin`].
struct OpenScope {
    start: Option<u32>,
    label: &'static str,
    depth: u16,
    set: Option<Arc<str>>,
    queue_family: Option<u32>,
}

/// Outcome of attempting to read one pending frame's results.
enum ReadState {
    Ready,
    NotReady,
    Error,
}

/// Records GPU timings and reports them to the frame timeline. Insert as a
/// resource via [`GpuProfilerPlugin`].
#[derive(Resource)]
pub struct GpuProfiler {
    shared: Arc<Shared>,
}

/// The state the resource shares with the [`SubmissionHook`], which pumicite
/// calls from the submission sets' systems, on any thread.
struct Shared {
    device: Device,
    /// Nanoseconds per timestamp tick (`timestampPeriod`).
    period_ns: f64,
    /// Whether the device and its queues support timestamp queries at all.
    supported: bool,
    /// Whether the device's clock can be sampled against the CPU's.
    calibrateable: bool,
    /// The runtime toggle.
    enabled: AtomicBool,
    state: Mutex<State>,
}

struct State {
    /// Whether this frame times anything: `enabled` when the frame began.
    recording: bool,
    /// The timestamp pool, shared with in-flight command buffers (see the
    /// module docs). `None` only if allocation failed.
    pool: Option<Arc<QueryPool>>,
    /// Current pool size in slots.
    capacity: u32,
    /// Next free slot in the ring.
    head: u32,
    /// Slots currently outstanding (allocated, not yet freed back to the ring).
    live_slots: u32,
    /// Set when an allocation failed; triggers a grow on the next frame.
    overflowed: bool,
    /// The frame being recorded; moved into `pending` at end-of-frame.
    current: Frame,
    /// Recorded frames awaiting GPU completion and readback.
    pending: VecDeque<Frame>,
    /// The submissions recording now, by their command buffer.
    open: HashMap<vk::CommandBuffer, OpenSubmission>,
    /// The valid bits of each queue family's timestamps, as a mask.
    valid_masks: HashMap<u32, u64>,
    /// The submission sets' names.
    set_names: HashMap<InternedSystemSet, Arc<str>>,
    /// The scopes' labels, for the debug labels.
    labels: HashMap<&'static str, CString>,
    calibration: Option<Calibration>,
}

impl GpuProfiler {
    fn new(device: Device, period_ns: f64, calibrateable: bool) -> Self {
        // Vulkan requires every query to be reset before its first use, so reset
        // the whole pool once up front. After that, slots are recycled on
        // readback (see `free_frame`), keeping resets off the hot path.
        let pool = (period_ns > 0.0)
            .then(|| {
                QueryPool::new(device.clone(), vk::QueryType::TIMESTAMP, INITIAL_CAPACITY).ok()
            })
            .flatten()
            .map(|mut pool| {
                pool.host_reset(0..INITIAL_CAPACITY);
                Arc::new(pool)
            });
        let supported = pool.is_some();
        GpuProfiler {
            shared: Arc::new(Shared {
                device,
                period_ns,
                supported,
                calibrateable,
                enabled: AtomicBool::new(supported),
                state: Mutex::new(State {
                    recording: false,
                    pool,
                    capacity: INITIAL_CAPACITY,
                    head: 0,
                    live_slots: 0,
                    overflowed: false,
                    current: Frame::default(),
                    pending: VecDeque::new(),
                    open: HashMap::new(),
                    valid_masks: HashMap::new(),
                    set_names: HashMap::new(),
                    labels: HashMap::new(),
                    calibration: None,
                }),
            }),
        }
    }

    /// Whether GPU timestamp queries are available on this device.
    pub fn is_supported(&self) -> bool {
        self.shared.supported
    }

    /// Whether the GPU's timings can be placed on the CPU's clock.
    pub fn is_calibrated(&self) -> bool {
        self.shared.calibrateable
    }

    /// Whether the profiler times the GPU's work.
    pub fn is_enabled(&self) -> bool {
        self.shared.enabled.load(Ordering::Relaxed)
    }

    /// Turns timing on or off, from the next frame.
    pub fn set_enabled(&self, enabled: bool) {
        self.shared.enabled.store(enabled, Ordering::Relaxed);
    }
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The mask of the valid bits of `queue_family`'s timestamps; 0 if it can't
    /// write any.
    fn valid_mask(&self, state: &mut State, queue_family: u32) -> u64 {
        *state.valid_masks.entry(queue_family).or_insert_with(|| {
            match self
                .device
                .physical_device()
                .timestamp_valid_bits(queue_family)
            {
                0 => 0,
                64.. => u64::MAX,
                bits => (1 << bits) - 1,
            }
        })
    }

    // ─── Per-frame lifecycle ────────────────────────────────────────────────

    /// Starts frame `index`: reads back what earlier frames' timings are done,
    /// recalibrates, and grows the pool if it overflowed.
    fn begin_frame(&self, index: u32) {
        let mut state = self.state();
        state.recording = self.supported && self.enabled.load(Ordering::Relaxed);
        self.resolve_pending(&mut state);
        if !state.recording {
            state.current = Frame::default();
            return;
        }
        if self.calibrateable
            && state
                .calibration
                .is_none_or(|calibration| calibration.cpu.elapsed() >= CALIBRATION_INTERVAL)
        {
            self.calibrate(&mut state);
        }
        if state.pool.is_none() || state.overflowed {
            self.grow(&mut state);
        }
        // Start a fresh frame at the current ring position.
        state.current = Frame {
            index,
            start: state.head,
            slots: 0,
            records: Vec::new(),
        };
    }

    /// Finalizes this frame's timings into the pending-readback queue.
    fn end_frame(&self) {
        let mut state = self.state();
        if state.current.slots == 0 {
            return;
        }
        let frame = std::mem::take(&mut state.current);
        state.pending.push_back(frame);
        // If readback has fallen this far behind, the frames piling up may still be
        // in flight — resetting/reusing their slots would be unsafe, so ask for a
        // fresh pool next frame (the old one is retired via its `Arc` drop).
        if state.pending.len() > MAX_PENDING {
            state.overflowed = true;
        }
    }

    /// Samples the GPU's clock, and takes the CPU's halfway through the call.
    fn calibrate(&self, state: &mut State) {
        let before = Instant::now();
        let Ok((timestamps, _)) = self
            .device
            .get_calibrated_timestamps(&[vk::TimeDomainKHR::DEVICE])
        else {
            return;
        };
        let after = Instant::now();
        state.calibration = Some(Calibration {
            gpu: timestamps[0],
            cpu: before + (after - before) / 2,
        });
    }

    /// Reads back every pending frame whose timestamps the GPU has finished
    /// writing, reports them to the frame timeline and frees their slots. Stops
    /// at the first frame that is not yet available (frames complete in order).
    fn resolve_pending(&self, state: &mut State) {
        // Take the pool out so `state` can be mutated freely while reading. The
        // `Arc` is moved, not dropped, so the pool stays alive.
        let Some(pool) = state.pool.take() else {
            return;
        };
        while let Some(front) = state.pending.front() {
            let mut results = Vec::with_capacity(front.records.len());
            let mut read = ReadState::Ready;
            for record in &front.records {
                let mut pair = [TimestampResult::default(); 2];
                match pool.get_results(
                    record.start,
                    &mut pair,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                ) {
                    Ok(()) if pair[0].available != 0 && pair[1].available != 0 => {
                        results.push((pair[0].value, pair[1].value));
                    }
                    Ok(()) | Err(vk::Result::NOT_READY) => {
                        read = ReadState::NotReady;
                        break;
                    }
                    Err(_) => {
                        read = ReadState::Error;
                        break;
                    }
                }
            }
            match read {
                ReadState::NotReady => break,
                ReadState::Ready => {
                    let frame = state.pending.pop_front().unwrap();
                    self.free_frame(state, &pool, &frame);
                    self.report(state, &frame, &results);
                }
                ReadState::Error => {
                    // Couldn't read results; force a fresh pool next frame
                    // rather than risk resetting slots in an unknown state.
                    state.pending.pop_front();
                    state.overflowed = true;
                    break;
                }
            }
        }
        state.pool = Some(pool);
    }

    /// Sends `frame`'s timings, whose timestamps are `results`, to the frame
    /// timeline.
    fn report(&self, state: &mut State, frame: &Frame, results: &[(u64, u64)]) {
        let now = Instant::now();
        let calibration = state.calibration;
        for (record, &(begin, end)) in frame.records.iter().zip(results) {
            let mask = match record.queue_family {
                Some(queue_family) => self.valid_mask(state, queue_family),
                None => u64::MAX,
            };
            let ticks = end.wrapping_sub(begin) & mask;
            let duration_ns = (ticks as f64 * self.period_ns) as u64;
            // Where the timestamps fall on the CPU's clock, as offsets from
            // now.
            let offset_ns = |timestamp: u64| {
                let calibration = calibration?;
                let ticks = timestamp.wrapping_sub(calibration.gpu) & mask;
                // Timestamps before the calibration wrap around to the top of
                // the mask.
                let ticks = if ticks > mask / 2 {
                    ticks as i128 - (mask as i128 + 1)
                } else {
                    ticks as i128
                };
                let since_calibration = (ticks as f64 * self.period_ns) as i128;
                let calibration_from_now = signed_nanos(calibration.cpu, now);
                Some((calibration_from_now + since_calibration) as i64)
            };
            let (pass, depth) = match record.timed {
                Timed::Submission => (None, 0),
                Timed::Scope { label, depth } => (Some(label), depth),
            };
            tracing::info!(
                name: "gpu",
                target: timeline::target::GPU,
                frame = frame.index,
                pass,
                set = record.set.as_deref(),
                queue = record.queue_family,
                depth,
                duration_ns,
                begin_offset_ns = offset_ns(begin),
                end_offset_ns = offset_ns(end),
                calibrated = calibration.is_some(),
            );
        }
    }

    /// Frees a frame's slots back to the ring and host-resets them so they are
    /// ready for reuse. Resetting here (once per frame, in 1–2 `vkResetQueryPool`
    /// calls) keeps the API overhead off the per-scope recording path.
    fn free_frame(&self, state: &mut State, pool: &QueryPool, frame: &Frame) {
        state.live_slots -= frame.slots;
        let end = frame.start + frame.slots;
        // SAFETY: every timing the frame recorded was just read back available,
        // so the GPU is done with its slots, and the ring hands them to no
        // other frame until they are freed here.
        unsafe {
            if end <= state.capacity {
                pool.host_reset_unchecked(frame.start..end);
            } else {
                // The frame's arc wrapped the end of the ring; reset both parts.
                pool.host_reset_unchecked(frame.start..state.capacity);
                pool.host_reset_unchecked(0..end - state.capacity);
            }
        }
    }

    /// Reallocates the pool, larger than before, retiring the old one. Our `Arc`
    /// to the old pool is dropped here; command buffers still using it hold
    /// their own clones. Pending readbacks reference the old pool and are lost.
    fn grow(&self, state: &mut State) {
        // Already at the ceiling with a working pool: nothing more to do.
        if state.pool.is_some() && state.capacity >= MAX_CAPACITY {
            state.overflowed = false;
            return;
        }
        let new_capacity = if state.pool.is_none() {
            state.capacity.max(INITIAL_CAPACITY)
        } else {
            state.capacity.saturating_mul(2).min(MAX_CAPACITY)
        };
        if let Ok(mut pool) =
            QueryPool::new(self.device.clone(), vk::QueryType::TIMESTAMP, new_capacity)
        {
            // Reset every query once before first use. The old pool (and its
            // pending readbacks) is retired via the `Arc` drop.
            pool.host_reset(0..new_capacity);
            state.pool = Some(Arc::new(pool));
            state.capacity = new_capacity;
            state.head = 0;
            state.live_slots = 0;
            state.pending.clear();
        }
        state.overflowed = false;
    }

    // ─── Timing (called from the command encoder) ───────────────────────────

    /// Allocates a pair of slots from the ring and records the start timestamp.
    /// Returns the start slot (stop is `start + 1`), or `None` if not recording
    /// / out of space. The slots were already host-reset when freed (or at pool
    /// creation), so no reset happens on this hot path.
    fn timestamp_begin(&self, state: &mut State, encoder: &mut CommandEncoder) -> Option<u32> {
        if !state.recording {
            return None;
        }
        if state.live_slots + 2 > state.capacity {
            state.overflowed = true;
            return None;
        }
        // `capacity` is even and `head` advances by 2, so a pair always fits
        // before the ring end — no straddling, no wasted slots.
        let start = state.head;
        // `retain` keeps the pool alive until this command buffer completes, even
        // if `grow` drops our `Arc` first.
        let pool = encoder.retain(state.pool.as_ref()?.clone()).deref_inner();
        // SAFETY: this command buffer owns the two slots handed out here: no
        // other frame gets them, and the host resets them only after their
        // results come back available.
        unsafe { encoder.write_timestamp_unchecked(pool, TS_STAGE, start) };
        state.head = start + 2;
        if state.head == state.capacity {
            state.head = 0; // wrap exactly at the boundary
        }
        state.live_slots += 2;
        state.current.slots += 2;
        Some(start)
    }

    /// Records the stop timestamp of the pair begun at `start`.
    fn timestamp_end(&self, state: &State, encoder: &mut CommandEncoder, start: u32) {
        let Some(pool) = state.pool.as_ref() else {
            return;
        };
        let pool = encoder.retain(pool.clone()).deref_inner();
        // SAFETY: as in `timestamp_begin`; `start + 1` is the second slot it
        // handed out.
        unsafe { encoder.write_timestamp_unchecked(pool, TS_STAGE, start + 1) };
    }

    fn scope_begin(&self, encoder: &mut CommandEncoder, label: &'static str) -> OpenScope {
        let mut state = self.state();
        let label_name = state
            .labels
            .entry(label)
            .or_insert_with(|| CString::new(label).unwrap_or_default());
        encoder.begin_label(label_name, LABEL_COLOR);
        let (set, queue_family, depth) = match state.open.get_mut(&encoder.buffer().vk_handle()) {
            Some(submission) => {
                submission.depth += 1;
                (
                    Some(submission.set.clone()),
                    Some(submission.queue_family),
                    submission.depth,
                )
            }
            None => (None, None, 1),
        };
        let writes_timestamps =
            queue_family.is_none_or(|queue_family| self.valid_mask(&mut state, queue_family) != 0);
        let start = writes_timestamps
            .then(|| self.timestamp_begin(&mut state, encoder))
            .flatten();
        OpenScope {
            start,
            label,
            depth,
            set,
            queue_family,
        }
    }

    fn scope_end(&self, encoder: &mut CommandEncoder, scope: OpenScope) {
        let mut state = self.state();
        if let Some(submission) = state.open.get_mut(&encoder.buffer().vk_handle()) {
            submission.depth -= 1;
        }
        if let Some(start) = scope.start {
            self.timestamp_end(&state, encoder, start);
            state.current.records.push(Record {
                start,
                timed: Timed::Scope {
                    label: scope.label,
                    depth: scope.depth,
                },
                set: scope.set,
                queue_family: scope.queue_family,
            });
        }
        encoder.end_label();
    }
}

/// Times every submission, as pumicite begins and ends its command buffer.
struct SubmissionTimer(Arc<Shared>);

impl SubmissionHook for SubmissionTimer {
    fn begin(&self, set: InternedSystemSet, queue_family: u32, encoder: &mut CommandEncoder) {
        let shared = &self.0;
        let mut state = shared.state();
        let name = state
            .set_names
            .entry(set)
            .or_insert_with(|| format!("{set:?}").into())
            .clone();
        let start = if shared.valid_mask(&mut state, queue_family) != 0 {
            shared.timestamp_begin(&mut state, encoder)
        } else {
            None
        };
        state.open.insert(
            encoder.buffer().vk_handle(),
            OpenSubmission {
                set: name,
                queue_family,
                start,
                depth: 0,
            },
        );
    }

    fn end(&self, _: InternedSystemSet, _: u32, encoder: &mut CommandEncoder) {
        let shared = &self.0;
        let mut state = shared.state();
        let Some(submission) = state.open.remove(&encoder.buffer().vk_handle()) else {
            return;
        };
        if let Some(start) = submission.start {
            shared.timestamp_end(&state, encoder, start);
            state.current.records.push(Record {
                start,
                timed: Timed::Submission,
                set: Some(submission.set),
                queue_family: Some(submission.queue_family),
            });
        }
    }
}

/// `to - from`, in nanoseconds, negative when `to` is earlier.
fn signed_nanos(to: Instant, from: Instant) -> i128 {
    match to.checked_duration_since(from) {
        Some(after) => after.as_nanos() as i128,
        None => -(from.duration_since(to).as_nanos() as i128),
    }
}

/// Command-encoder extension for timing a scope of GPU work.
pub trait GpuTimerCommands {
    /// Runs `f`, timing the GPU's work on the commands it records under
    /// `label`, and its recording on the CPU.
    ///
    /// `profiler` is `Option<&mut GpuProfiler>` and no-ops on `None`, so call
    /// sites don't have to branch on whether profiling is enabled. The closure
    /// receives the same encoder and its return value is forwarded.
    fn timing_scope<R>(
        &mut self,
        profiler: Option<&mut GpuProfiler>,
        label: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R;
}

impl GpuTimerCommands for CommandEncoder<'_> {
    fn timing_scope<R>(
        &mut self,
        profiler: Option<&mut GpuProfiler>,
        label: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let _recording =
            tracing::info_span!(target: timeline::target::GPU, "pass", name = label).entered();
        let Some(profiler) = profiler else {
            return f(self);
        };
        let scope = profiler.shared.scope_begin(self, label);
        let result = f(self);
        profiler.shared.scope_end(self, scope);
        result
    }
}

/// Installs the [`GpuProfiler`] resource, times every submission, and keeps the
/// profiler's frames in step with Bevy's.
#[derive(Default)]
pub struct GpuProfilerPlugin;

impl Plugin for GpuProfilerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (
                enable_device_features.before(CreateDevice),
                setup_gpu_profiler.after(CreateDevice),
            ),
        );
        // A frame's timings are recorded in `PostUpdate`, between these.
        app.add_systems(First, gpu_profiler_begin_frame);
        app.add_systems(Last, gpu_profiler_end_frame);
    }
}

/// Asks for calibrated timestamps, where the device has them, and for
/// resetting queries from the host, which the ring relies on.
fn enable_device_features(mut device_builder: ResMut<DeviceBuilder>) {
    device_builder
        .enable_extension::<pumicite::ash::khr::calibrated_timestamps::Meta>()
        .or_else(|_| {
            device_builder.enable_extension::<pumicite::ash::ext::calibrated_timestamps::Meta>()
        })
        .ok();
    device_builder
        .enable_feature::<vk::PhysicalDeviceHostQueryResetFeatures>(|features| {
            &mut features.host_query_reset
        })
        .ok();
}

fn setup_gpu_profiler(mut commands: Commands, device: Res<Device>) {
    let period_ns = f64::from(
        device
            .physical_device()
            .properties()
            .limits
            .timestamp_period,
    );
    let calibrateable = device
        .calibrateable_time_domains()
        .contains(&vk::TimeDomainKHR::DEVICE);
    let profiler = GpuProfiler::new(device.clone(), period_ns, calibrateable);
    if !profiler.is_supported() {
        warn!("GPU timestamp queries unsupported (period {period_ns} ns)");
    } else if !calibrateable {
        warn!(
            "GPU timestamps can't be calibrated against the CPU's clock: the frame \
             timeline shows how long GPU work took, but not when"
        );
    }
    if profiler.is_supported() {
        commands.insert_resource(SubmissionHooks(Arc::new(SubmissionTimer(
            profiler.shared.clone(),
        ))));
    }
    commands.insert_resource(profiler);
}

fn gpu_profiler_begin_frame(profiler: Option<Res<GpuProfiler>>, frame_count: Res<FrameCount>) {
    if let Some(profiler) = profiler {
        profiler.shared.begin_frame(frame_count.0);
    }
}

fn gpu_profiler_end_frame(profiler: Option<Res<GpuProfiler>>) {
    if let Some(profiler) = profiler {
        profiler.shared.end_frame();
    }
}
