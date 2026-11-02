//! Figures over the frames shown: the summary and the tables.

use std::collections::HashMap;

use dust_profiler::timeline::{FrameRecord, FrameSummary, Name, Nanos, SpanKind, SpanRecord};

/// What held the frames back, judged from their averages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Bound {
    /// The frames missed refreshes, and the GPU was the busier.
    Gpu,
    /// The frames missed refreshes, and the Bevy thread was the busier.
    Cpu,
    /// The frames kept up with the display's refreshes.
    Display,
}

/// Averages over the frames shown.
pub(super) struct Overview {
    pub frames: usize,
    pub fps: Option<f32>,
    pub interval: Option<Nanos>,
    pub interval_p99: Option<Nanos>,
    pub interval_max: Option<Nanos>,
    pub cpu: Option<Nanos>,
    pub gpu: Option<Nanos>,
    pub wait: Option<Nanos>,
    pub display_latency: Option<Nanos>,
    pub input_latency: Option<Nanos>,
    pub dropped: usize,
    pub bound: Option<Bound>,
}

/// How much longer than the refresh interval frames can take on average and
/// still count as keeping up, for jitter.
const KEEPING_UP: f64 = 1.15;

/// The figures over `summaries`, on a display refreshing every
/// `refresh_interval`.
pub(super) fn overview(summaries: &[FrameSummary], refresh_interval: Nanos) -> Overview {
    let mut intervals: Vec<Nanos> = summaries
        .iter()
        .filter_map(|summary| summary.interval)
        .collect();
    intervals.sort_unstable();
    let interval = mean(intervals.iter().copied());
    let cpu = mean(summaries.iter().map(|summary| summary.cpu));
    let gpu = mean(summaries.iter().filter_map(|summary| summary.gpu));
    let bound = interval.map(|interval| {
        if interval as f64 <= KEEPING_UP * refresh_interval as f64 {
            Bound::Display
        } else if gpu.unwrap_or(0) >= cpu.unwrap_or(0) {
            Bound::Gpu
        } else {
            Bound::Cpu
        }
    });
    Overview {
        frames: summaries.len(),
        fps: interval
            .filter(|&interval| interval > 0)
            .map(|interval| 1e9 / interval as f32),
        interval,
        interval_p99: percentile(&intervals, 0.99),
        interval_max: intervals.last().copied(),
        cpu,
        gpu,
        wait: mean(summaries.iter().map(|summary| summary.wait)),
        display_latency: mean(
            summaries
                .iter()
                .filter_map(|summary| summary.display_latency),
        ),
        input_latency: mean(summaries.iter().filter_map(|summary| summary.input_latency)),
        dropped: summaries.iter().filter(|summary| summary.dropped).count(),
        bound,
    }
}

/// A GPU pass, or a whole submission, over the frames shown.
pub(super) struct PassRow {
    pub name: Name,
    /// 0 for a submission, 1 for a scope in one, and so on.
    pub depth: u16,
    pub gpu: Nanos,
    pub gpu_max: Nanos,
    /// Its share of the frames' GPU time.
    pub share: Option<f32>,
    /// The system that recorded it, and that system's average time.
    pub encoded_by: Option<(Name, Nanos)>,
}

/// The GPU records of `frames`, averaged per name, the most expensive first.
pub(super) fn passes(frames: &[&FrameRecord]) -> Vec<PassRow> {
    struct Totals {
        depth: u16,
        frames: usize,
        total: Nanos,
        max: Nanos,
        encoders: HashMap<Name, (usize, Nanos)>,
    }
    let mut totals: HashMap<(u16, Name), Totals> = HashMap::new();
    let mut gpu_frames = Vec::new();
    for frame in frames {
        let mut per_frame: HashMap<(u16, Name), Nanos> = HashMap::new();
        for gpu in &frame.gpu {
            *per_frame.entry((gpu.depth, gpu.name.clone())).or_default() += gpu.duration;
        }
        if let Some(outermost) = frame.gpu.iter().map(|gpu| gpu.depth).min() {
            gpu_frames.push(
                frame
                    .gpu
                    .iter()
                    .filter(|gpu| gpu.depth == outermost)
                    .map(|gpu| gpu.duration)
                    .sum::<Nanos>(),
            );
        }
        let encoders = encoders(frame);
        for ((depth, name), duration) in per_frame {
            let entry = totals.entry((depth, name.clone())).or_insert(Totals {
                depth,
                frames: 0,
                total: 0,
                max: 0,
                encoders: HashMap::new(),
            });
            entry.frames += 1;
            entry.total += duration;
            entry.max = entry.max.max(duration);
            if let Some((system, cpu)) = encoders.get(&name) {
                let encoder = entry.encoders.entry(system.clone()).or_default();
                encoder.0 += 1;
                encoder.1 += cpu;
            }
        }
    }
    let gpu_frame = mean(gpu_frames.into_iter());
    let mut rows: Vec<PassRow> = totals
        .into_iter()
        .map(|((_, name), totals)| {
            let gpu = totals.total / totals.frames as Nanos;
            PassRow {
                name,
                depth: totals.depth,
                gpu,
                gpu_max: totals.max,
                share: gpu_frame
                    .filter(|&frame| frame > 0)
                    .map(|frame| gpu as f32 / frame as f32),
                encoded_by: totals
                    .encoders
                    .into_iter()
                    .max_by_key(|(_, (count, _))| *count)
                    .map(|(system, (count, cpu))| (system, cpu / count as Nanos)),
            }
        })
        .collect();
    rows.sort_by(|a, b| a.depth.cmp(&b.depth).then(b.gpu.cmp(&a.gpu)));
    rows
}

/// For each GPU scope recorded in `frame`, the system that recorded it — the
/// system span around the scope's CPU span — and that system's time.
fn encoders(frame: &FrameRecord) -> HashMap<Name, (Name, Nanos)> {
    frame
        .spans
        .iter()
        .filter(|span| span.kind == SpanKind::Pass)
        .filter_map(|pass| {
            let system = innermost_around(&frame.spans, pass, SpanKind::System)?;
            Some((
                pass.name.clone(),
                (system.name.clone(), system.end - system.begin),
            ))
        })
        .collect()
}

/// The innermost span of `kind` around `span`, on its thread.
fn innermost_around<'a>(
    spans: &'a [SpanRecord],
    span: &SpanRecord,
    kind: SpanKind,
) -> Option<&'a SpanRecord> {
    spans
        .iter()
        .filter(|outer| {
            outer.kind == kind
                && outer.thread == span.thread
                && outer.depth < span.depth
                && outer.begin <= span.begin
                && span.end <= outer.end
        })
        .max_by_key(|outer| outer.depth)
}

/// A schedule, over the frames shown.
pub(super) struct ScheduleRow {
    pub name: Name,
    /// How many schedules it runs in.
    pub nesting: usize,
    /// Time per frame, all of its runs together.
    pub time: Nanos,
    pub time_max: Nanos,
    /// Runs per frame; `FixedMain` and what it runs can run any number of
    /// times, or none.
    pub runs: f32,
}

/// The schedules run in `frames`, in the order the newest frame ran them.
pub(super) fn schedules(frames: &[&FrameRecord]) -> Vec<ScheduleRow> {
    struct Totals {
        nesting: usize,
        /// When the newest frame first ran it.
        first: Nanos,
        total: Nanos,
        max: Nanos,
        runs: usize,
    }
    let mut totals: HashMap<Name, Totals> = HashMap::new();
    for frame in frames {
        let schedules: Vec<&SpanRecord> = frame
            .spans
            .iter()
            .filter(|span| span.kind == SpanKind::Schedule)
            .collect();
        let mut per_frame: HashMap<Name, (Nanos, usize, usize, Nanos)> = HashMap::new();
        for schedule in &schedules {
            let nesting = schedules
                .iter()
                .filter(|outer| {
                    outer.thread == schedule.thread
                        && outer.depth < schedule.depth
                        && outer.begin <= schedule.begin
                        && schedule.end <= outer.end
                })
                .count();
            let entry =
                per_frame
                    .entry(schedule.name.clone())
                    .or_insert((0, 0, nesting, schedule.begin));
            entry.0 += schedule.end - schedule.begin;
            entry.1 += 1;
            entry.3 = entry.3.min(schedule.begin);
        }
        for (name, (time, runs, nesting, first)) in per_frame {
            let entry = totals.entry(name).or_insert(Totals {
                nesting,
                first,
                total: 0,
                max: 0,
                runs: 0,
            });
            entry.total += time;
            entry.max = entry.max.max(time);
            entry.runs += runs;
            // Frames come oldest first: the newest frame's order wins.
            entry.first = first;
            entry.nesting = nesting;
        }
    }
    let count = frames.len().max(1) as Nanos;
    let mut rows: Vec<(Nanos, ScheduleRow)> = totals
        .into_iter()
        .map(|(name, totals)| {
            (
                totals.first,
                ScheduleRow {
                    name,
                    nesting: totals.nesting,
                    time: totals.total / count,
                    time_max: totals.max,
                    runs: totals.runs as f32 / count as f32,
                },
            )
        })
        .collect();
    rows.sort_by_key(|(first, _)| *first);
    rows.into_iter().map(|(_, row)| row).collect()
}

fn mean(values: impl Iterator<Item = Nanos>) -> Option<Nanos> {
    let (sum, count) = values.fold((0u128, 0u128), |(sum, count), value| {
        (sum + u128::from(value), count + 1)
    });
    (count > 0).then(|| (sum / count) as Nanos)
}

/// The value `fraction` of the way through `sorted`.
fn percentile(sorted: &[Nanos], fraction: f64) -> Option<Nanos> {
    let last = sorted.len().checked_sub(1)?;
    let at = ((sorted.len() as f64 * fraction).ceil() as usize).saturating_sub(1);
    Some(sorted[at.min(last)])
}
