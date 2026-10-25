//! GPU timing via elapsed timer (timestamp) queries.
//!
//! [`GpuProfiler`] lets render passes measure how long specific GPU operations
//! take. A pass wraps its work in [`GpuTimerCommands::timing_scope`] on the
//! command encoder; the profiler records a timestamp before and after, and a
//! few frames later reads the pair back and reports the elapsed time as
//! `(end - start) * VkPhysicalDeviceLimits::timestampPeriod`.
//!
//! ## Usage
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
//! `timing_scope` takes `Option<&mut GpuProfiler>`, so when [`GpuProfilerPlugin`]
//! isn't installed (the resource is absent → `None`) or the profiler is toggled
//! off, the scope just runs the closure and records nothing.
//!
//! ## Query pool as a dynamic ring
//!
//! The [`QueryPool`] is treated as a ring of timestamp slots. Each scope
//! dynamically grabs the next two free slots (start + stop), wrapping at the
//! end of the pool and reusing slots whose frame has already been read back.
//! There is no fixed per-frame budget — a frame uses as many slots as it has
//! scopes. Slots are reset from the host with `vkResetQueryPool` immediately
//! before reuse, which is safe because the ring only reuses slots that belong
//! to already-completed frames.
//!
//! The pool is shared slot-by-slot between the host and every in-flight frame,
//! so it is not wrapped in a [`GPUMutex`](pumicite::sync::GPUMutex): a
//! `GPUMutex` hands the *whole* pool to one owner at a time, and the host could
//! then only read results once no frame at all is in flight. Instead the pool
//! is an `Arc<QueryPool>`, accessed through pumicite's `_unchecked` query
//! methods, and `scope_begin` / `free_frame` uphold the per-slot ownership
//! those methods require.
//!
//! When the in-flight working set genuinely doesn't fit (readback fell behind
//! or a frame has an unusually large number of scopes), the pool is
//! reallocated at a larger capacity. Every command buffer that writes a
//! timestamp [`CommandEncoder::retain`]s a clone of the `Arc`, so dropping the
//! old pool defers its destruction until the GPU stops referencing it.

use std::{collections::VecDeque, sync::Arc};

use bevy::prelude::*;
use bevy_pumicite::{CreateDevice, DefaultRenderSet};
use pumicite::{
    Device, Instance, ash::vk, command::CommandEncoder, physical_device::PhysicalDevice,
    query::QueryPool, utils::AsVkHandle,
};

/// Initial pool size in timestamp slots. Grows on demand.
const INITIAL_CAPACITY: u32 = 256;
/// Upper bound on pool growth, so a runaway never allocates without limit.
const MAX_CAPACITY: u32 = 1 << 16;
/// Cap on frames awaiting readback. Bounds memory if results never drain.
const MAX_PENDING: usize = 16;

/// Pipeline stage sampled by the start/stop timestamps. `ALL_COMMANDS` records
/// the moment every preceding command has fully completed, so the difference
/// between two of them is the device time spent on the bracketed work.
const TS_STAGE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::ALL_COMMANDS;

/// One query result read back with `WITH_AVAILABILITY | TYPE_64`: the 64-bit
/// timestamp value plus a 64-bit availability flag.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TimestampResult {
    value: u64,
    available: u64,
}

/// A completed scope within a frame. Its two timestamps occupy the adjacent
/// slots `start` (begin) and `start + 1` (end).
struct ScopeRecord {
    label: &'static str,
    start: u32,
}

/// A frame's scopes plus the ring arc they occupy. Used both for the frame
/// currently being recorded and for frames queued awaiting readback.
///
/// Each scope uses two slots, so the frame occupies the contiguous (wrapping)
/// arc `[start, start + 2 * scopes.len())`, which is host-reset and freed back
/// to the ring once the frame is resolved or dropped.
#[derive(Default)]
struct Frame {
    /// First ring slot this frame occupies (`head` when the frame began).
    start: u32,
    scopes: Vec<ScopeRecord>,
}

impl Frame {
    /// Slots this frame occupies in the ring (two per scope).
    fn slots(&self) -> u32 {
        self.scopes.len() as u32 * 2
    }
}

/// Outcome of attempting to read one pending frame's results.
enum ReadState {
    Ready,
    NotReady,
    Error,
}

/// Records and reports GPU timings for named scopes. Insert as a resource via
/// [`GpuProfilerPlugin`].
#[derive(Resource)]
pub struct GpuProfiler {
    /// Runtime toggle. When false, all scope recording is skipped.
    pub enabled: bool,
    /// Whether the device/queue actually supports timestamp queries.
    supported: bool,

    device: Device,
    /// Nanoseconds per timestamp tick (`timestampPeriod`).
    period_ns: f32,
    /// Mask for the queue's valid timestamp bits.
    valid_mask: u64,

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

    /// Latest resolved durations as `(label, milliseconds)`, in first-seen
    /// order for stable display.
    timings: Vec<(&'static str, f32)>,
}

impl GpuProfiler {
    fn new(device: Device, period_ns: f32, valid_bits: u32) -> Self {
        let timestamps_ok = period_ns > 0.0 && valid_bits > 0;
        let valid_mask = if valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << valid_bits) - 1
        };
        // Vulkan requires every query to be reset before its first use, so reset
        // the whole pool once up front. After that, slots are recycled on
        // readback (see `free_frame`), keeping resets off the hot path.
        let pool = timestamps_ok
            .then(|| {
                QueryPool::new(device.clone(), vk::QueryType::TIMESTAMP, INITIAL_CAPACITY).ok()
            })
            .flatten()
            .map(|mut qp| {
                qp.host_reset(0..INITIAL_CAPACITY);
                Arc::new(qp)
            });
        let supported = timestamps_ok && pool.is_some();
        Self {
            enabled: supported,
            supported,
            device,
            period_ns,
            valid_mask,
            pool,
            capacity: INITIAL_CAPACITY,
            head: 0,
            live_slots: 0,
            overflowed: false,
            current: Frame::default(),
            pending: VecDeque::new(),
            timings: Vec::new(),
        }
    }

    /// Whether GPU timestamp queries are available on this device.
    pub fn is_supported(&self) -> bool {
        self.supported
    }

    /// Latest resolved timings in first-seen order: `(label, milliseconds)`.
    pub fn timings(&self) -> impl Iterator<Item = (&'static str, f32)> + '_ {
        self.timings.iter().copied()
    }

    // ─── Per-frame lifecycle ────────────────────────────────────────────────

    /// Reads back every pending frame whose timestamps the GPU has finished
    /// writing, updating the reported timings and freeing their slots. Stops at
    /// the first frame that is not yet available (frames complete in order).
    fn resolve_pending(&mut self) {
        // Take the pool out so `self` can be mutated freely while reading. The
        // `Arc` is moved, not dropped, so the pool stays alive.
        let Some(pool) = self.pool.take() else {
            return;
        };
        while let Some(front) = self.pending.front() {
            let mut results: Vec<(&'static str, f32)> = Vec::with_capacity(front.scopes.len());
            let mut state = ReadState::Ready;
            for sc in &front.scopes {
                let mut pair = [TimestampResult::default(); 2];
                match pool.get_results(
                    sc.start,
                    &mut pair,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                ) {
                    Ok(()) if pair[0].available != 0 && pair[1].available != 0 => {
                        let s = pair[0].value & self.valid_mask;
                        let e = pair[1].value & self.valid_mask;
                        let ticks = e.wrapping_sub(s) & self.valid_mask;
                        let ms = ticks as f64 * self.period_ns as f64 / 1.0e6;
                        results.push((sc.label, ms as f32));
                    }
                    Ok(()) => {
                        state = ReadState::NotReady;
                        break;
                    }
                    Err(vk::Result::NOT_READY) => {
                        state = ReadState::NotReady;
                        break;
                    }
                    Err(_) => {
                        state = ReadState::Error;
                        break;
                    }
                }
            }
            match state {
                ReadState::NotReady => break,
                ReadState::Ready => {
                    let frame = self.pending.pop_front().unwrap();
                    self.free_frame(&pool, &frame);
                    for (label, ms) in results {
                        self.record_timing(label, ms);
                    }
                }
                ReadState::Error => {
                    // Couldn't read results; force a fresh pool next frame
                    // rather than risk resetting slots in an unknown state.
                    self.pending.pop_front();
                    self.overflowed = true;
                    break;
                }
            }
        }
        self.pool = Some(pool);
    }

    /// Frees a frame's slots back to the ring and host-resets them so they are
    /// ready for reuse. Resetting here (once per frame, in 1–2 `vkResetQueryPool`
    /// calls) keeps the API overhead off the per-scope recording path.
    fn free_frame(&mut self, pool: &QueryPool, frame: &Frame) {
        let slots = frame.slots();
        self.live_slots -= slots;
        let end = frame.start + slots;
        // SAFETY: `frame` was just read back with every timestamp available, so
        // the GPU is done with its slots, and the ring hands them to no other
        // frame until they are freed here.
        unsafe {
            if end <= self.capacity {
                pool.host_reset_unchecked(frame.start..end);
            } else {
                // The frame's arc wrapped the end of the ring; reset both parts.
                pool.host_reset_unchecked(frame.start..self.capacity);
                pool.host_reset_unchecked(0..end - self.capacity);
            }
        }
    }

    /// Reallocates the pool, larger than before, retiring the old one. Our `Arc`
    /// to the old pool is dropped here; command buffers still using it hold
    /// their own clones. Pending readbacks reference the old pool and are lost.
    fn grow(&mut self) {
        // Already at the ceiling with a working pool: nothing more to do.
        if self.pool.is_some() && self.capacity >= MAX_CAPACITY {
            self.overflowed = false;
            return;
        }
        let new_capacity = if self.pool.is_none() {
            self.capacity.max(INITIAL_CAPACITY)
        } else {
            self.capacity.saturating_mul(2).min(MAX_CAPACITY)
        };
        if let Ok(mut pool) =
            QueryPool::new(self.device.clone(), vk::QueryType::TIMESTAMP, new_capacity)
        {
            // Reset every query once before first use. The old pool (and its
            // pending readbacks) is retired via the `Arc` drop.
            pool.host_reset(0..new_capacity);
            self.pool = Some(Arc::new(pool));
            self.capacity = new_capacity;
            self.head = 0;
            self.live_slots = 0;
            self.pending.clear();
        }
        self.overflowed = false;
    }

    fn record_timing(&mut self, label: &'static str, ms: f32) {
        if let Some(entry) = self.timings.iter_mut().find(|(l, _)| *l == label) {
            entry.1 = ms;
        } else {
            self.timings.push((label, ms));
        }
    }

    // ─── Scope recording (called from the command encoder) ──────────────────

    /// Allocates a pair of slots from the ring and records the start timestamp.
    /// Returns the start slot (stop is `start + 1`), or `None` if not recording
    /// / out of space. The slots were already host-reset when freed (or at pool
    /// creation), so no reset happens on this hot path.
    fn scope_begin(&mut self, encoder: &mut CommandEncoder, _label: &'static str) -> Option<u32> {
        if !self.enabled {
            return None;
        }
        if self.live_slots + 2 > self.capacity {
            self.overflowed = true;
            return None;
        }

        // `capacity` is even and `head` advances by 2, so a pair always fits
        // before the ring end — no straddling, no wasted slots.
        let start = self.head;
        // `retain` keeps the pool alive until this command buffer completes, even
        // if `grow` drops our `Arc` first.
        let pool = encoder.retain(self.pool.as_ref()?.clone()).deref_inner();
        // SAFETY: this command buffer owns the two slots handed out here: no
        // other frame gets them, and the host resets them only after their
        // results come back available.
        unsafe { encoder.write_timestamp_unchecked(pool, TS_STAGE, start) };

        self.head = start + 2;
        if self.head == self.capacity {
            self.head = 0; // wrap exactly at the boundary
        }
        self.live_slots += 2;
        Some(start)
    }

    /// Records the stop timestamp for a scope opened at `start`.
    fn scope_end(&mut self, encoder: &mut CommandEncoder, label: &'static str, start: u32) {
        let Some(pool) = self.pool.as_ref() else {
            return;
        };
        let pool = encoder.retain(pool.clone()).deref_inner();
        // SAFETY: as in `scope_begin`; `start + 1` is the second slot it handed out.
        unsafe { encoder.write_timestamp_unchecked(pool, TS_STAGE, start + 1) };
        self.current.scopes.push(ScopeRecord { label, start });
    }
}

/// Command-encoder extension for timing a scope of GPU work.
pub trait GpuTimerCommands {
    /// Runs `f`, recording the GPU time it takes under `label`.
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
        mut profiler: Option<&mut GpuProfiler>,
        label: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let start = profiler.as_mut().and_then(|x| x.scope_begin(self, label));
        let result = f(self);
        if let Some(start) = start {
            if let Some(profiler) = profiler {
                profiler.scope_end(self, label, start);
            }
        }
        result
    }
}

/// Installs the [`GpuProfiler`] resource and its per-frame bookkeeping.
#[derive(Default)]
pub struct GpuProfilerPlugin;

impl Plugin for GpuProfilerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_gpu_profiler.after(CreateDevice));
        app.add_systems(PostUpdate, gpu_profiler_begin_frame.before(DefaultRenderSet));
        // In `Last`, after the submissions that follow `DefaultRenderSet`'s in
        // `PostUpdate`, such as the swapchain's.
        app.add_systems(Last, gpu_profiler_end_frame);
    }
}

fn setup_gpu_profiler(
    mut commands: Commands,
    device: Res<Device>,
    instance: Res<Instance>,
    physical_device: Res<PhysicalDevice>,
) {
    // timestampValidBits is a per-queue-family property; take the best across
    // families that can carry graphics or compute work (where our passes run).
    let valid_bits = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device.vk_handle())
    }
    .into_iter()
    .filter(|f| {
        f.queue_flags
            .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    })
    .map(|f| f.timestamp_valid_bits)
    .max()
    .unwrap_or(0);

    let period_ns = physical_device.properties().limits.timestamp_period;

    let profiler = GpuProfiler::new(device.clone(), period_ns, valid_bits);
    if !profiler.supported {
        warn!("GPU timestamp queries unsupported (period {period_ns}, valid_bits {valid_bits})");
    }
    commands.insert_resource(profiler);
}

/// Drains finished readbacks and (re)allocates the pool if it overflowed.
fn gpu_profiler_begin_frame(profiler: Option<ResMut<GpuProfiler>>) {
    let Some(mut profiler) = profiler else {
        return;
    };

    if !profiler.enabled || !profiler.supported {
        profiler.current = Frame::default();
        return;
    }

    profiler.resolve_pending();

    if profiler.pool.is_none() || profiler.overflowed {
        profiler.grow();
    }

    // Start a fresh frame at the current ring position.
    profiler.current = Frame {
        start: profiler.head,
        scopes: Vec::new(),
    };
}

/// Finalizes this frame's recorded scopes into the pending-readback queue.
fn gpu_profiler_end_frame(profiler: Option<ResMut<GpuProfiler>>) {
    let Some(mut profiler) = profiler else {
        return;
    };
    if profiler.current.scopes.is_empty() {
        return;
    }

    let frame = std::mem::take(&mut profiler.current);
    profiler.pending.push_back(frame);

    // If readback has fallen this far behind, the frames piling up may still be
    // in flight — resetting/reusing their slots would be unsafe, so ask for a
    // fresh pool next frame (the old one is retired via its `Arc` drop).
    if profiler.pending.len() > MAX_PENDING {
        profiler.overflowed = true;
    }
}
