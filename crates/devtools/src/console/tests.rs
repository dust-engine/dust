use std::path::Path;

use tracing::subscriber;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;

use super::*;

/// Runs `f` with a subscriber of only a [`ConsoleLayer`], whose filter lets
/// through messages up to `max_level`, and returns what the layer sent.
fn capture_at(max_level: LevelFilter, f: impl FnOnce()) -> Vec<ConsoleMessage> {
    let (layer, log) = console_layer();
    // Messages reach only the receivers active when they're logged.
    let mut messages = log.messages.activate_cloned();
    subscriber::with_default(Registry::default().with(layer).with(max_level), f);
    std::iter::from_fn(|| messages.try_recv().ok()).collect()
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
    ConsoleMessage::evaluation(MessageKind::Output, text.to_owned(), Instant::now())
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

    // As the console's socket sends it, after the seconds since logging
    // started.
    let text = message.to_string();
    let (_seconds, text) = text.split_once(' ').unwrap();
    assert_eq!(
        text,
        format!(
            " WARN load{{model=\"castle\" lod=2}}: {}: loaded in 12 ms voxels=1024 \
             name=\"teapot\" hollow=true",
            module_path!()
        )
    );
}

#[test]
fn log_record() {
    // Turns `log` records into `tracing` events, as the app's subscriber does. It's
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
    use tracing_subscriber::EnvFilter;

    let allows = |directives: &str, level| {
        let filter = EnvFilter::builder().parse_lossy(directives);
        subscriber::with_default(Registry::default().with(filter), || {
            filter_default_allows(level)
        })
    };
    // Debug messages from one target, but not by default.
    let one_target = "info,dust_pbr=debug";
    assert!(allows(one_target, Level::INFO));
    assert!(!allows(one_target, Level::DEBUG));
    assert!(allows("debug", Level::DEBUG) && !allows("debug", Level::TRACE));
    assert!(!allows("warn,dust_pbr=debug", Level::INFO));
}

#[test]
fn repeats_count_on_one_row() {
    let mut transcript = Transcript::default();
    for _ in 0..3 {
        transcript.push(output("same"));
    }
    transcript.push(output("other"));
    transcript.push(output("same"));
    let counts: Vec<_> = transcript
        .rows
        .iter()
        .map(|row| (row.message.text.as_str(), row.count))
        .collect();
    assert_eq!(counts, [("same", 3), ("other", 1), ("same", 1)]);
    // Rows are numbered by their first message, repeats included.
    let serials: Vec<_> = transcript.rows.iter().map(|row| row.serial).collect();
    assert_eq!(serials, [1, 4, 5]);
    assert_eq!(transcript.row(4).unwrap().message.text, "other");
    assert!(transcript.row(2).is_none());
}

#[test]
fn oldest_rows_dropped() {
    let mut transcript = Transcript::default();
    for i in 0..MAX_ROWS + 5 {
        transcript.push(output(&i.to_string()));
    }
    assert_eq!(transcript.rows.len(), MAX_ROWS);
    assert_eq!(transcript.rows[0].message.text, "5");
    assert!(transcript.row(5).is_none());
    assert_eq!(transcript.row(6).unwrap().message.text, "5");
}

#[test]
fn scopes_and_filter() {
    let mut scopes = [false; Scope::ALL.len()];
    let warning = capture(|| tracing::warn!(path = "castle.vox", "slow load")).remove(0);
    let command =
        ConsoleMessage::evaluation(MessageKind::Command, "echo hi".to_owned(), Instant::now());
    assert!(shows(&scopes, "", &warning) && shows(&scopes, "", &command));

    scopes[Scope::Warnings as usize] = true;
    assert!(shows(&scopes, "", &warning) && !shows(&scopes, "", &command));

    // Matches the text, field values and target, ignoring case.
    scopes = Default::default();
    assert!(shows(&scopes, "SLOW", &warning) && shows(&scopes, "castle", &warning));
    assert!(shows(&scopes, "console::tests", &warning));
    assert!(!shows(&scopes, "teapot", &warning) && !shows(&scopes, "slow", &command));
}

#[test]
fn prompt_history() {
    let mut transcript = Transcript::default();
    for line in ["echo a", "echo b", "echo b"] {
        transcript.remember(line);
    }
    // A repeated line is kept once.
    assert_eq!(transcript.history, ["echo a", "echo b"]);

    let mut recall = Recall::default();
    let mut line = "ec".to_owned();
    let mut browse = |older| {
        line = recall.browse(&transcript.history, std::mem::take(&mut line), older);
        line.clone()
    };
    assert_eq!(browse(true), "echo b");
    assert_eq!(browse(true), "echo a");
    assert_eq!(browse(true), "echo a");
    assert_eq!(browse(false), "echo b");
    assert_eq!(browse(false), "ec");
    assert_eq!(browse(false), "ec");
}

/// Replaces part of `line` as `edit` says.
fn apply(line: &mut String, edit: complete::Edit) {
    line.replace_range(edit.range, &edit.text);
}

/// Tab at the end of `line`, completing paths from `directory`: the line
/// after, and the candidates listed, as the line would have them.
fn tab_in(line: &str, directory: Option<&Path>) -> (String, Vec<String>) {
    let (edit, completion) = complete::complete(line, line.len(), directory);
    let mut line = line.to_owned();
    if let Some(edit) = edit {
        apply(&mut line, edit);
    }
    let listed = completion.map_or_else(Vec::new, |completion| {
        let candidates = completion.candidates().iter();
        candidates.map(|candidate| candidate.text.clone()).collect()
    });
    (line, listed)
}

fn tab(line: &str) -> (String, Vec<String>) {
    tab_in(line, None)
}

#[test]
fn complete_commands() {
    // One candidate replaces the word, and a space follows it.
    assert_eq!(tab("scr"), ("screenshot ".to_owned(), vec![]));
    assert_eq!(tab("  fr").0, "  frame ");
    let (line, listed) = tab("");
    assert_eq!(line, "");
    assert_eq!(
        listed,
        [
            "echo",
            "log",
            "frame",
            "screenshot",
            "upscaler",
            "time",
            "clear",
            "help"
        ]
    );

    // Options, and their values.
    assert_eq!(tab("log --le").0, "log --level ");
    assert_eq!(tab("log --level w").0, "log --level warn ");
    assert_eq!(tab("log --level 'w").0, "log --level warn ");
    assert_eq!(tab("upscaler m").0, "upscaler metalfx ");
    assert_eq!(tab("help up").0, "help upscaler ");

    // Several extend the word by what they all start with.
    let (line, listed) = tab("upscaler -q ultra");
    assert_eq!(line, "upscaler -q ultra-");
    assert_eq!(listed, ["ultra-performance", "ultra-quality"]);

    // Nothing to complete.
    assert_eq!(tab("frob"), ("frob".to_owned(), vec![]));
    assert_eq!(tab("echo #fr"), ("echo #fr".to_owned(), vec![]));
}

#[test]
fn complete_paths() {
    let directory =
        std::env::temp_dir().join(format!("dust-console-complete-{}", std::process::id()));
    std::fs::create_dir_all(directory.join("shots")).unwrap();
    std::fs::write(directory.join("shot one.png"), b"").unwrap();
    let tab = |line| tab_in(line, Some(&directory));

    // Quoted as the line needs it.
    assert_eq!(
        tab("screenshot sh"),
        (
            "screenshot shot".to_owned(),
            vec!["'shot one.png'".to_owned(), "shots/".to_owned()]
        )
    );
    assert_eq!(tab("screenshot 'shot o").0, "screenshot 'shot one.png' ");
    // A directory's files may follow it, so no space does.
    assert_eq!(tab("screenshot shots").0, "screenshot shots/");

    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn completion_list() {
    let open = || {
        let line = "upscaler -q ultra";
        let (edit, completion) = complete::complete(line, line.len(), None);
        let mut line = line.to_owned();
        apply(&mut line, edit.unwrap());
        (line, completion.unwrap())
    };

    // Tab selects the next candidate, putting it in the line; Shift-Tab the
    // one before. Both go around.
    let (mut line, mut completion) = open();
    assert!(completion.is_current(&line, line.len()));
    let mut select = |forward| {
        apply(&mut line, completion.select_next(forward));
        line.clone()
    };
    assert_eq!(select(true), "upscaler -q ultra-performance");
    assert_eq!(select(true), "upscaler -q ultra-quality");
    assert_eq!(select(true), "upscaler -q ultra-performance");
    assert_eq!(select(false), "upscaler -q ultra-quality");
    assert_eq!(completion.selected(), Some(1));
    assert!(completion.is_current(&line, line.len()));
    // An edit of the user's, or moving the cursor, ends the completion.
    assert!(!completion.is_current("upscaler -q ultra-qualit", 24));
    assert!(!completion.is_current(&line, 0));

    // Enter keeps the candidate selected.
    apply(&mut line, completion.accept().unwrap());
    assert_eq!(line, "upscaler -q ultra-quality ");

    // Escape puts back the word as it was.
    let (mut line, mut completion) = open();
    apply(&mut line, completion.select_next(false));
    apply(&mut line, completion.cancel().unwrap());
    assert_eq!(line, "upscaler -q ultra-");

    // With none selected, neither changes the line.
    assert!(open().1.accept().is_none());
    assert!(open().1.cancel().is_none());

    // A click picks a candidate.
    let (mut line, completion) = open();
    apply(&mut line, completion.pick(0));
    assert_eq!(line, "upscaler -q ultra-performance ");
}
