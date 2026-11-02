use tracing::{Dispatch, dispatcher, info, info_span};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;

use super::target::{FRAME, GPU, INPUT, PUMICITE, SWAPCHAIN};
use super::*;

/// Bevy's targets for its spans.
const SCHEDULE: &str = "bevy_ecs::schedule::schedule";
const SYSTEM: &str = "bevy_ecs::system::function_system";

/// A dispatcher of only a timeline layer that completes frames
/// `complete_after` frames after their own, and its timeline.
fn timeline(complete_after: u32) -> (Dispatch, FrameTimeline) {
    let (layer, timeline) = timeline_layer_with(TimelineConfig {
        complete_after,
        ..TimelineConfig::default()
    });
    (Dispatch::new(Registry::default().with(layer)), timeline)
}

/// Runs `f` in frame `index`, as the runner does.
fn frame(index: u32, f: impl FnOnce()) {
    let _frame = info_span!(target: FRAME, "frame", index).entered();
    f();
}

/// A schedule label, whose `Debug` Bevy gives as the schedule's name.
#[derive(Debug)]
struct Update;

fn span<'a>(frame: &'a FrameRecord, name: &str) -> &'a SpanRecord {
    frame
        .spans
        .iter()
        .find(|span| &*span.name == name)
        .unwrap_or_else(|| panic!("no span {name}"))
}

#[test]
fn nests_schedules_and_systems_in_their_frame() {
    let (dispatch, timeline) = timeline(1);
    dispatcher::with_default(&dispatch, || {
        for index in 0..3 {
            frame(index, || {
                let _schedule = info_span!(target: SCHEDULE, "schedule", name = ?Update).entered();
                let _system =
                    info_span!(target: SYSTEM, "system", name = "game::move_player").entered();
            });
        }
    });
    // Frame 2 is the latest, so frames up to 1 are complete.
    let frames = timeline.frames_after(None);
    assert_eq!(
        frames.iter().map(|frame| frame.index).collect::<Vec<_>>(),
        [0, 1]
    );
    let first = &frames[0];
    let whole = span(first, "frame");
    let schedule = span(first, "Update");
    let system = span(first, "game::move_player");
    assert_eq!(
        (whole.kind, schedule.kind, system.kind),
        (SpanKind::Frame, SpanKind::Schedule, SpanKind::System)
    );
    assert_eq!((whole.depth, schedule.depth, system.depth), (0, 1, 2));
    assert!(whole.begin <= schedule.begin && schedule.begin <= system.begin);
    assert!(system.end <= schedule.end && schedule.end <= whole.end);
    assert_eq!((first.begin, first.end), (whole.begin, whole.end));
    assert_eq!(timeline.frames_after(Some(0)).len(), 1);
}

#[test]
fn records_other_threads_in_the_running_frame() {
    let (dispatch, timeline) = timeline(1);
    dispatcher::with_default(&dispatch, || {
        for index in 0..2 {
            frame(index, || {
                let dispatch = dispatch.clone();
                std::thread::Builder::new()
                    .name("worker".into())
                    .spawn(move || {
                        dispatcher::with_default(&dispatch, || {
                            let _system =
                                info_span!(target: SYSTEM, "system", name = "game::physics")
                                    .entered();
                        })
                    })
                    .unwrap()
                    .join()
                    .unwrap();
            });
        }
    });
    let frames = timeline.frames_after(None);
    let [frame] = &frames[..] else {
        panic!("expected one complete frame");
    };
    let system = span(frame, "game::physics");
    assert_eq!(&*timeline.threads()[system.thread as usize], "worker");
    // Nothing else was open on the worker.
    assert_eq!(system.depth, 0);
    assert!(frame.begin <= system.begin && system.end <= frame.end);
}

#[test]
fn merges_late_events_into_their_frame() {
    let (dispatch, timeline) = timeline(3);
    dispatcher::with_default(&dispatch, || {
        // Far enough from the timeline's start for the offsets back from it.
        std::thread::sleep(std::time::Duration::from_millis(1));
        for index in 0..7u32 {
            frame(index, || match index {
                2 => info!(
                    name: "gpu",
                    target: GPU,
                    frame = 0u32,
                    pass = "primary ray",
                    duration_ns = 2_000_000u64,
                    calibrated = false
                ),
                3 => info!(
                    name: "displayed",
                    target: SWAPCHAIN,
                    frame = 0u32,
                    offset_ns = -500i64,
                    shown = true
                ),
                _ => {}
            });
        }
    });
    let frames = timeline.frames_after(None);
    let first = &frames[0];
    let [gpu] = &first.gpu[..] else {
        panic!("expected one GPU record");
    };
    assert_eq!(&*gpu.name, "primary ray");
    assert_eq!((gpu.depth, gpu.duration, gpu.begin), (1, 2_000_000, None));
    let displayed = first.displayed.expect("frame 0 was displayed");
    let third = &frames.iter().find(|frame| frame.index == 3).unwrap();
    let at = displayed.at.unwrap();
    // 500 ns before the event, sent during frame 3.
    assert!(third.begin - 500 <= at && at <= third.end - 500);

    let summary = timeline.summaries_after(None)[0];
    assert_eq!(summary.gpu, Some(2_000_000));
    assert_eq!(summary.display_latency, Some(at - first.begin));
}

#[test]
fn leaves_waits_out_of_cpu_time() {
    let (dispatch, timeline) = timeline(1);
    dispatcher::with_default(&dispatch, || {
        for index in 0..3 {
            frame(index, || {
                let _wait = info_span!(target: PUMICITE, "wait for GPU", set = "Render").entered();
                std::thread::sleep(std::time::Duration::from_millis(2));
            });
            // Between frames, as the runner waits for the next.
            let _wait = info_span!(target: FRAME, "wait compositor clock").entered();
        }
    });
    let frames = timeline.frames_after(None);
    let first = &frames[0];
    let inside = span(first, "wait for GPU");
    let between = span(first, "wait compositor clock");
    assert_eq!(inside.kind, SpanKind::Wait);
    assert_eq!(inside.detail.as_deref(), Some("Render"));
    assert!(between.begin >= first.end);

    let summary = timeline.summaries_after(None)[0];
    assert_eq!(summary.duration, first.end - first.begin);
    assert_eq!(summary.cpu, summary.duration - (inside.end - inside.begin));
    assert_eq!(
        summary.wait,
        (inside.end - inside.begin) + (between.end - between.begin)
    );
    assert_eq!(summary.interval, Some(frames[1].begin - first.begin));
}

#[test]
fn reads_input_offsets() {
    let (dispatch, timeline) = timeline(1);
    dispatcher::with_default(&dispatch, || {
        // Far enough from the timeline's start for the offsets back from it.
        std::thread::sleep(std::time::Duration::from_millis(1));
        for index in 0..2 {
            frame(index, || {
                info!(
                    name: "poll",
                    target: INPUT,
                    count = 3u64,
                    oldest_offset_ns = -4_000i64,
                    newest_offset_ns = -1_000i64
                );
            });
        }
    });
    let frames = timeline.frames_after(None);
    let [input] = &frames[0].inputs[..] else {
        panic!("expected one input poll");
    };
    assert_eq!(input.count, 3);
    assert_eq!(input.oldest, Some(input.at - 4_000));
    assert_eq!(input.newest, Some(input.at - 1_000));
}

#[test]
fn ignores_other_spans_and_events() {
    let (dispatch, timeline) = timeline(1);
    dispatcher::with_default(&dispatch, || {
        for index in 0..2 {
            frame(index, || {
                let _span = info_span!("load", model = "castle").entered();
                info!("loaded");
                let _executor =
                    info_span!(target: "bevy_ecs::schedule::executor", "multithreaded executor")
                        .entered();
            });
        }
    });
    let frames = timeline.frames_after(None);
    let names: Vec<_> = frames[0].spans.iter().map(|span| &*span.name).collect();
    assert_eq!(names, ["frame"]);
}

#[test]
fn tells_other_layers_what_it_records() {
    // A registry alone enables every span, so each has its metadata.
    dispatcher::with_default(&Dispatch::new(Registry::default()), || {
        let recorded = |span: tracing::Span| is_timeline(span.metadata().expect("an enabled span"));
        assert!(recorded(info_span!(target: FRAME, "frame", index = 0u32)));
        assert!(recorded(
            info_span!(target: SYSTEM, "system", name = "game::move_player")
        ));
        assert!(recorded(info_span!(target: PUMICITE, "wait for GPU")));
        assert!(!recorded(info_span!("load")));
        assert!(!recorded(info_span!(target: SCHEDULE, "check_conditions")));
    });
}

#[test]
fn records_nothing_while_paused() {
    let (dispatch, timeline) = timeline(1);
    dispatcher::with_default(&dispatch, || {
        frame(0, || {});
        timeline.set_recording(false);
        frame(1, || {});
        timeline.set_recording(true);
        for index in 2..4 {
            frame(index, || {});
        }
    });
    let indices: Vec<_> = timeline
        .frames_after(None)
        .iter()
        .filter(|frame| !frame.spans.is_empty())
        .map(|frame| frame.index)
        .collect();
    assert_eq!(indices, [0, 2]);
}
