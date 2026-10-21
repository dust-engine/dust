use std::sync::mpsc;

use bevy::log::tracing::{self, subscriber};
use bevy::log::tracing_subscriber::filter::LevelFilter;
use bevy::log::tracing_subscriber::layer::SubscriberExt;
use bevy::log::tracing_subscriber::registry::Registry;

use super::command::Outcome;
use super::layer::ConsoleLayer;
use super::*;
use crate::DockSide;

/// Runs `f` with a subscriber of only a [`ConsoleLayer`], whose filter lets
/// through messages up to `max_level`, and returns what the layer sent.
fn capture_at(max_level: LevelFilter, f: impl FnOnce()) -> Vec<ConsoleMessage> {
    let (sender, receiver) = mpsc::channel();
    let subscriber = Registry::default()
        .with(ConsoleLayer {
            send: Box::new(move |message| {
                let _ = sender.send(message);
            }),
        })
        .with(max_level);
    subscriber::with_default(subscriber, f);
    receiver.try_iter().collect()
}

fn capture(f: impl FnOnce()) -> Vec<ConsoleMessage> {
    capture_at(LevelFilter::TRACE, f)
}

fn fields(fields: &[Field]) -> Vec<(&str, &str, ValueKind)> {
    fields
        .iter()
        .map(|field| (field.name, field.value.as_str(), field.kind))
        .collect()
}

fn output(text: &str) -> ConsoleMessage {
    ConsoleMessage::evaluation(MessageKind::Output, text.to_owned())
}

#[test]
fn tracing_event() {
    let messages = capture(|| {
        let span = tracing::info_span!("load", model = "castle", lod = 2);
        let _entered = span.enter();
        tracing::warn!(
            voxels = 1024,
            name = "teapot",
            hollow = true,
            "loaded in {} ms",
            12
        );
    });
    let [message] = &messages[..] else {
        panic!("expected one message");
    };
    assert_eq!(message.kind, MessageKind::Log(Level::WARN));
    assert_eq!(message.text, "loaded in 12 ms");
    assert_eq!(
        fields(&message.fields),
        [
            ("voxels", "1024", ValueKind::Number),
            ("name", "teapot", ValueKind::Str),
            ("hollow", "true", ValueKind::Bool),
        ]
    );
    let [span] = &message.spans[..] else {
        panic!("expected one span");
    };
    assert_eq!(span.name, "load");
    assert_eq!(
        fields(&span.fields),
        [
            ("model", "castle", ValueKind::Str),
            ("lod", "2", ValueKind::Number)
        ]
    );
    let source = message.source.as_ref().unwrap();
    assert_eq!(source.target, module_path!());
    assert_eq!(source.file.as_deref(), Some(file!()));
    assert!(source.line.is_some());
}

#[test]
fn log_record() {
    // Turns `log` records into `tracing` events, as `LogPlugin` does. It's
    // global, so another test may have installed it already.
    let _ = tracing_log::LogTracer::init();
    let messages = capture(|| tracing_log::log::error!(target: "assets", "missing {}", "a.vox"));
    let [message] = &messages[..] else {
        panic!("expected one message");
    };
    assert_eq!(message.kind, MessageKind::Log(Level::ERROR));
    assert_eq!(message.text, "missing a.vox");
    // The record's metadata, not the `log.*` fields carrying it.
    assert!(message.fields.is_empty());
    let source = message.source.as_ref().unwrap();
    assert_eq!(source.target, "assets");
    assert_eq!(source.module_path.as_deref(), Some(module_path!()));
    assert_eq!(source.file.as_deref(), Some(file!()));
}

#[test]
fn span_fields_recorded_later() {
    let messages = capture(|| {
        let span = tracing::info_span!("frame", number = tracing::field::Empty);
        span.record("number", 7);
        span.in_scope(|| tracing::info!("drawn"));
    });
    assert_eq!(
        fields(&messages[0].spans[0].fields),
        [("number", "7", ValueKind::Number)]
    );
}

#[test]
fn error_with_sources() {
    #[derive(Debug)]
    struct Error(&'static str, Option<Box<Error>>);
    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }
    impl std::error::Error for Error {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.1.as_deref().map(|error| error as _)
        }
    }

    let error = Error("open failed", Some(Box::new(Error("no such file", None))));
    let messages = capture(|| {
        tracing::error!(
            error = &error as &(dyn std::error::Error + 'static),
            "loading"
        );
    });
    assert_eq!(
        fields(&messages[0].fields),
        [("error", "open failed: no such file", ValueKind::Error)]
    );
}

#[test]
fn filter_default_level() {
    use bevy::log::tracing_subscriber::EnvFilter;

    let allows = |directives: &str, level| {
        let filter = EnvFilter::builder().parse_lossy(directives);
        subscriber::with_default(Registry::default().with(filter), || {
            filter_default_allows(level)
        })
    };
    // Built as `LogPlugin` builds it. It lets debug messages through for one
    // target, `calloop::sources`, but not by default.
    let log_plugin = format!("{},{}", Level::INFO, bevy::log::DEFAULT_FILTER);
    assert!(allows(&log_plugin, Level::INFO));
    assert!(!allows(&log_plugin, Level::DEBUG));
    assert!(allows("debug", Level::DEBUG) && !allows("debug", Level::TRACE));
    assert!(!allows("warn,dust_pbr=debug", Level::INFO));
}

#[test]
fn repeats_count_on_one_row() {
    let mut console = Console::default();
    for _ in 0..3 {
        console.push(output("same"));
    }
    console.push(output("other"));
    console.push(output("same"));
    let counts: Vec<_> = console
        .rows
        .iter()
        .map(|row| (row.message.text.as_str(), row.count))
        .collect();
    assert_eq!(counts, [("same", 3), ("other", 1), ("same", 1)]);
}

#[test]
fn oldest_rows_dropped() {
    let mut console = Console::default();
    for i in 0..MAX_ROWS + 5 {
        console.push(output(&i.to_string()));
    }
    assert_eq!(console.rows.len(), MAX_ROWS);
    assert_eq!(console.rows[0].message.text, "5");
}

#[test]
fn scopes_and_filter() {
    let mut console = Console::default();
    let warning = capture(|| tracing::warn!(path = "castle.vox", "slow load")).remove(0);
    let command = ConsoleMessage::evaluation(MessageKind::Command, "echo hi".to_owned());
    assert!(console.shows(&warning, "") && console.shows(&command, ""));

    console.scopes[Scope::Warnings as usize] = true;
    assert!(console.shows(&warning, "") && !console.shows(&command, ""));

    // Matches the text, field values and target, ignoring case.
    console.scopes = Default::default();
    assert!(console.shows(&warning, "SLOW") && console.shows(&warning, "castle"));
    assert!(console.shows(&warning, "console::tests"));
    assert!(!console.shows(&warning, "teapot") && !console.shows(&command, "slow"));
}

#[test]
fn prompt_history() {
    let mut console = Console::default();
    let mut submitted = Vec::new();
    for line in ["echo a", "  ", "echo b", "echo b"] {
        console.prompt.line = line.to_owned();
        submitted.extend(console.submit());
    }
    // Blank lines are dropped, and a repeated line is kept once.
    assert_eq!(console.prompt.history, ["echo a", "echo b"]);
    assert_eq!(submitted, ["echo a", "echo b", "echo b"]);

    let prompt = &mut console.prompt;
    prompt.line = "ec".to_owned();
    let mut browse = |older| {
        prompt.browse(older);
        prompt.line.clone()
    };
    assert_eq!(browse(true), "echo b");
    assert_eq!(browse(true), "echo a");
    assert_eq!(browse(true), "echo a");
    assert_eq!(browse(false), "echo b");
    assert_eq!(browse(false), "ec");
    assert_eq!(browse(false), "ec");
}

/// The text a command printed.
fn printed(result: Result<Outcome, String>) -> String {
    match result {
        Ok(Outcome::Output(text)) => text,
        _ => panic!("the command printed nothing"),
    }
}

#[test]
fn commands() {
    let mut world = World::new();

    assert_eq!(printed(command::run("echo 'a  b' c", &mut world)), "a  b c");
    let help = printed(command::run("help", &mut world));
    assert!(help.contains("echo") && help.contains("dock"));
    let typo = command::run("ecko hi", &mut world).err().unwrap();
    assert!(typo.contains("a similar subcommand exists: 'echo'"));

    assert!(matches!(
        command::run("dock left", &mut world),
        Ok(Outcome::Console(ConsoleEvent::Dock(DockSide::Left)))
    ));
    assert!(command::run("dock top", &mut world).is_err());

    let mut console = Console::default();
    console.push(output("x"));
    let Ok(Outcome::Console(event)) = command::run("clear", &mut world) else {
        panic!("clear didn't clear the console");
    };
    console.apply(event);
    assert!(console.rows.is_empty());
}

#[test]
fn log_command() {
    let _ = tracing_log::LogTracer::init();
    let mut world = World::new();
    let messages = capture(|| {
        assert!(matches!(
            command::run("log -l warn hello there", &mut world),
            Ok(Outcome::Nothing)
        ));
        assert!(matches!(
            command::run("log --log-crate oh no", &mut world),
            Ok(Outcome::Nothing)
        ));
    });
    let logged: Vec<_> = messages
        .iter()
        .map(|message| {
            let source = message.source.as_ref().unwrap();
            (message.kind, message.text.as_str(), source.target.as_str())
        })
        .collect();
    assert_eq!(
        logged,
        [
            (MessageKind::Log(Level::WARN), "hello there", "console"),
            (MessageKind::Log(Level::INFO), "oh no", "console"),
        ]
    );

    // A level the filter drops is reported instead of vanishing.
    capture_at(LevelFilter::INFO, || {
        let error = command::run("log -l debug hidden", &mut world)
            .err()
            .unwrap();
        assert!(error.contains("RUST_LOG=console=debug"), "{error}");
    });
}

#[test]
fn submitted_lines_run() {
    let mut world = World::new();
    let mut console = Console::default();
    for line in ["echo hi", "nope"] {
        console.prompt.line = line.to_owned();
        let line = console.submit().unwrap();
        if let Some(event) = run_line(&line, &mut world) {
            console.apply(event);
        }
    }
    let rows: Vec<_> = console
        .rows
        .iter()
        .map(|row| (row.message.kind, row.message.text.lines().next().unwrap()))
        .collect();
    assert_eq!(
        rows,
        [
            (MessageKind::Command, "echo hi"),
            (MessageKind::Output, "hi"),
            (MessageKind::Command, "nope"),
            (
                MessageKind::CommandError,
                "error: unrecognized subcommand 'nope'"
            ),
        ]
    );
}

#[test]
fn plugin_builds_with_and_without_layer() {
    let mut app = App::new();
    app.add_plugins(ConsolePlugin);
    app.update();

    let mut app = App::new();
    assert!(console_layer(&mut app).is_none());

    let mut app = App::new();
    let (to_ui, _from_bevy) = futures_channel::mpsc::unbounded();
    app.insert_resource(dust_ui::GPUIChannel(to_ui));
    let layer = console_layer(&mut app);
    assert!(layer.is_some());
    app.add_plugins(ConsolePlugin);
    app.update();
}
