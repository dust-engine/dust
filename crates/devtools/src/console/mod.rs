//! The "Console" DevTools tab, modeled on Safari's: the messages logged through
//! `tracing` and `log`, a scope bar filtering them by severity, and a prompt
//! that runs commands; and the console's socket, which runs commands sent to
//! it.
//!
//! The [`ConsoleLayer`] broadcasts each message from whichever thread logged
//! it, to whoever listens then: the [`Console`], from when DevTools is made,
//! and the connections to the console's socket. The console keeps the last
//! [`MAX_ROWS`] rows; the game's own log on stderr has them all.
//!
//! The rest is in four parts:
//! - [`Console`]: the console's rows and the prompt's history, kept for as
//!   long as the app runs, whether or not the console's tab is shown.
//! - [`command`]: running a command line, the same for the prompt and the
//!   socket. It runs on the UI thread, and awaits the Bevy thread for what it
//!   needs of the world.
//! - [`ui`]: the view of the console, which exists only while its tab is
//!   shown: the list of rows, the filter and scope bar, and the prompt.
//! - [`remote`]: the socket and its connections, from when the app starts if
//!   `DUST_CONSOLE_SOCKET` is set, whether or not the view is shown.

mod command;
mod format;
mod layer;
mod remote;
mod screenshot;
#[cfg(test)]
mod tests;
mod ui;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::task::Poll;
use std::time::Duration;

use async_broadcast::{InactiveReceiver, Receiver, TryRecvError};
use bevy::platform::time::Instant;
use futures_util::StreamExt as _;
use gpui::{Context, Task};
use tracing::Level;

use command::Outcome;
pub use layer::ConsoleLayer;
pub use remote::serve_console_socket;
pub(crate) use ui::{ClearConsole, ConsoleView};

/// Makes the `tracing` layer that copies log messages into the DevTools
/// console, and the [`ConsoleLog`] to listen to it with. Add the layer to the
/// global subscriber before anything logs, and pass the log to
/// [`overlay`](crate::overlay) and [`serve_console_socket`]:
///
/// ```ignore
/// let (console_layer, console_log) = dust_devtools::console_layer();
/// tracing_subscriber::registry()
///     .with(EnvFilter::new("info"))
///     .with(tracing_subscriber::fmt::layer())
///     .with(console_layer)
///     .init();
/// ```
///
/// The console gets the messages that pass the subscriber's filter: `tracing`
/// events, and `log` records, which the subscriber's `init` has turned into
/// `tracing` events.
pub fn console_layer() -> (ConsoleLayer, ConsoleLog) {
    let (mut sender, messages) = async_broadcast::broadcast(MAX_ROWS);
    // A receiver that falls behind, such as a connection to the socket that
    // stopped reading, loses the oldest messages, rather than the layer
    // waiting for it on whichever thread logged. The console keeps no more
    // than this anyway.
    sender.set_overflow(true);
    let startup = Instant::now();
    let layer = ConsoleLayer {
        messages: sender,
        startup,
    };
    let log = ConsoleLog {
        messages: messages.deactivate(),
        startup,
    };
    (layer, log)
}

/// What listens to the [`ConsoleLayer`]'s messages starts from this: the
/// [`Console`], and each connection to the console's socket. A message reaches
/// only the receivers active when it's logged.
#[derive(Clone)]
pub struct ConsoleLog {
    /// Inactive, so that it keeps no messages itself.
    messages: InactiveReceiver<ConsoleMessage>,
    /// When logging started; message times count from it.
    startup: Instant,
}

/// The most rows the console keeps, the oldest dropped first, and the most
/// messages a receiver falls behind by before it loses the oldest.
const MAX_ROWS: usize = 10_000;

/// `path` as the user meant it, made absolute. Under `bazel run`, which runs
/// the game in its runfiles, a relative path is from the directory `bazel run`
/// was started in.
fn user_path(path: PathBuf) -> PathBuf {
    let path = match std::env::var_os("BUILD_WORKING_DIRECTORY") {
        Some(directory) if path.is_relative() => Path::new(&directory).join(path),
        _ => path,
    };
    std::path::absolute(&path).unwrap_or(path)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MessageKind {
    /// A `tracing` event, or a `log` record, which the subscriber turns into
    /// one.
    Log(Level),
    /// A line submitted at the prompt.
    Command,
    /// What a command printed.
    Output,
    /// Why a command failed, such as clap's report of a misspelled argument.
    CommandError,
}

#[derive(Clone)]
pub(crate) struct ConsoleMessage {
    kind: MessageKind,
    /// The formatted message, the submitted line, or the command's output.
    text: String,
    /// The event's fields other than `message`, in the order they were
    /// recorded.
    fields: Vec<Field>,
    /// The spans the event was logged in, outermost first.
    spans: Vec<SpanContext>,
    /// Where a log message was logged; `None` for commands and their output.
    source: Option<Source>,
    /// When the message arrived, since logging started.
    time: Duration,
}

/// A field of an event or span, such as `count = 3` in
/// `info!(count = 3, "…")`.
#[derive(Clone, PartialEq)]
struct Field {
    name: &'static str,
    /// The value as text. Strings are kept without quotes.
    value: String,
    kind: ValueKind,
}

/// The type a field's value was recorded as, which picks its color.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ValueKind {
    Str,
    Number,
    Bool,
    /// An error, followed by its sources.
    Error,
    /// Any other value, formatted with `Debug`.
    Debug,
}

#[derive(Clone, PartialEq)]
struct SpanContext {
    name: &'static str,
    fields: Vec<Field>,
}

/// Where a log message was logged.
#[derive(Clone)]
struct Source {
    target: String,
    module_path: Option<String>,
    file: Option<String>,
    line: Option<u32>,
    /// Name of the thread that logged the message.
    thread: Option<String>,
}

impl ConsoleMessage {
    /// A command line, or what it printed. `startup` is when logging started.
    fn evaluation(kind: MessageKind, text: String, startup: Instant) -> Self {
        Self {
            kind,
            text,
            fields: Vec::new(),
            spans: Vec::new(),
            source: None,
            time: startup.elapsed(),
        }
    }

    /// Whether `self` repeats `previous`: the same content, logged from the
    /// same place. A repeat only increments the previous row's count, as in
    /// Safari.
    fn repeats(&self, previous: &Self) -> bool {
        let same_place = match (&self.source, &previous.source) {
            (Some(a), Some(b)) => a.target == b.target && a.file == b.file && a.line == b.line,
            (None, None) => true,
            _ => false,
        };
        same_place
            && self.kind == previous.kind
            && self.text == previous.text
            && self.fields == previous.fields
            && self.spans == previous.spans
    }

    /// Whether the text, a field value or the target contains `needle`,
    /// ignoring ASCII case.
    fn contains(&self, needle: &str) -> bool {
        let contains = |haystack: &str| {
            haystack
                .as_bytes()
                .windows(needle.len())
                .any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
        };
        contains(&self.text)
            || self.fields.iter().any(|field| contains(&field.value))
            || self
                .source
                .as_ref()
                .is_some_and(|source| contains(&source.target))
    }
}

/// A button of the scope bar. Picking scopes shows only their rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Evaluations,
    Errors,
    Warnings,
    Infos,
    Debugs,
    Traces,
}

impl Scope {
    const ALL: [Scope; 6] = [
        Scope::Evaluations,
        Scope::Errors,
        Scope::Warnings,
        Scope::Infos,
        Scope::Debugs,
        Scope::Traces,
    ];

    fn of(kind: MessageKind) -> Self {
        match kind {
            MessageKind::Log(Level::ERROR) => Scope::Errors,
            MessageKind::Log(Level::WARN) => Scope::Warnings,
            MessageKind::Log(Level::INFO) => Scope::Infos,
            MessageKind::Log(Level::DEBUG) => Scope::Debugs,
            MessageKind::Log(_) => Scope::Traces,
            MessageKind::Command | MessageKind::Output | MessageKind::CommandError => {
                Scope::Evaluations
            }
        }
    }

    /// The level of the log messages in the scope; `None` for evaluations.
    fn level(self) -> Option<Level> {
        match self {
            Scope::Evaluations => None,
            Scope::Errors => Some(Level::ERROR),
            Scope::Warnings => Some(Level::WARN),
            Scope::Infos => Some(Level::INFO),
            Scope::Debugs => Some(Level::DEBUG),
            Scope::Traces => Some(Level::TRACE),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Scope::Evaluations => "Evaluations",
            Scope::Errors => "Errors",
            Scope::Warnings => "Warnings",
            Scope::Infos => "Infos",
            Scope::Debugs => "Debugs",
            Scope::Traces => "Traces",
        }
    }
}

/// What the console has received, for as long as the app runs: the rows of
/// the messages logged and the lines run at the prompt, and the prompt's
/// history. Made with DevTools, it receives messages whether or not its tab
/// is shown. A [`ConsoleView`] shows it while the tab is, and observes it:
/// the console notifies when its rows change.
pub(crate) struct Console {
    transcript: Transcript,
    /// The log messages broadcast to the console. The console keeps its
    /// receiver, rather than the task, so that it can take the ones sent
    /// before a line it shows.
    messages: Receiver<ConsoleMessage>,
    /// When logging started.
    startup: Instant,
    /// Shows the messages logged as they arrive.
    _receive: Task<()>,
}

impl Console {
    /// A console of the messages logged to `log` from now on.
    pub(crate) fn new(log: &ConsoleLog, cx: &mut Context<Self>) -> Self {
        let messages = log.messages.activate_cloned();
        // Polls the console's receiver, which then wakes the task when more
        // messages arrive.
        let receive = cx.spawn(async move |console, cx| {
            std::future::poll_fn(|task| {
                let polled = console.update(cx, |console, cx| {
                    let mut changed = false;
                    let poll = loop {
                        // Skips past messages dropped while it was behind.
                        match console.messages.poll_next_unpin(task) {
                            Poll::Ready(Some(message)) => {
                                console.transcript.push(message);
                                changed = true;
                            }
                            Poll::Ready(None) => break Poll::Ready(()),
                            Poll::Pending => break Poll::Pending,
                        }
                    };
                    if changed {
                        cx.notify();
                    }
                    poll
                });
                // The console is gone.
                polled.unwrap_or(Poll::Ready(()))
            })
            .await
        });
        Console {
            transcript: Transcript::default(),
            messages,
            startup: log.startup,
            _receive: receive,
        }
    }

    /// Runs `line`, submitted at the prompt: adds it to the history, and shows
    /// it and what it printed, or why it failed. What it printed shows even if
    /// the view is gone by the time the command finishes.
    fn run(&mut self, line: String, cx: &mut Context<Self>) {
        self.transcript.remember(&line);
        let submitted =
            ConsoleMessage::evaluation(MessageKind::Command, line.clone(), self.startup);
        self.show(submitted, cx);
        cx.spawn(async move |console, cx| {
            let result = command::run(&line, cx).await;
            console
                .update(cx, |console, cx| {
                    let (kind, text) = match result {
                        Ok(Outcome::Nothing) => return,
                        Ok(Outcome::Clear) => return console.clear(cx),
                        Ok(Outcome::Output(output)) => (MessageKind::Output, output),
                        Err(error) => (MessageKind::CommandError, error),
                    };
                    let message = ConsoleMessage::evaluation(kind, text, console.startup);
                    console.show(message, cx);
                })
                .ok();
        })
        .detach();
    }

    /// Shows `message`, after the log messages sent to the console before,
    /// such as those of the line before, so that rows keep the order things
    /// happened in.
    fn show(&mut self, message: ConsoleMessage, cx: &mut Context<Self>) {
        loop {
            match self.messages.try_recv() {
                Ok(message) => self.transcript.push(message),
                // Some were dropped while the console was behind; the rest
                // follow.
                Err(TryRecvError::Overflowed(_)) => {}
                Err(TryRecvError::Empty | TryRecvError::Closed) => break,
            }
        }
        self.transcript.push(message);
        cx.notify();
    }

    fn clear(&mut self, cx: &mut Context<Self>) {
        self.transcript.rows.clear();
        cx.notify();
    }
}

/// The console's rows, and the prompt's history.
#[derive(Default)]
struct Transcript {
    /// Oldest first, which is also in the order of their `serial`s.
    rows: VecDeque<Row>,
    /// How many messages have been pushed, repeats included, which numbers
    /// the rows.
    pushed: u64,
    /// The scopes a message has arrived in, indexed by `Scope as usize`; kept
    /// when the console is cleared.
    received: [bool; Scope::ALL.len()],
    /// The lines submitted at the prompt, oldest first.
    history: Vec<String>,
}

struct Row {
    message: ConsoleMessage,
    /// How many times in a row the message arrived.
    count: u32,
    /// The number of the row's first message, counting from 1. Unique among
    /// rows, even those cleared: the view tracks the rows its list shows by
    /// it, and keys their elements with it.
    serial: u64,
}

impl Transcript {
    fn push(&mut self, message: ConsoleMessage) {
        self.received[Scope::of(message.kind) as usize] = true;
        self.pushed += 1;
        if let Some(last) = self.rows.back_mut()
            && message.repeats(&last.message)
        {
            last.count += 1;
            return;
        }
        if self.rows.len() == MAX_ROWS {
            self.rows.pop_front();
        }
        self.rows.push_back(Row {
            message,
            count: 1,
            serial: self.pushed,
        });
    }

    /// The row numbered `serial`, if the console still has it.
    fn row(&self, serial: u64) -> Option<&Row> {
        let index = self
            .rows
            .binary_search_by_key(&serial, |row| row.serial)
            .ok()?;
        Some(&self.rows[index])
    }

    /// Adds `line`, submitted at the prompt, to the history, unless it
    /// repeats the last line there.
    fn remember(&mut self, line: &str) {
        if self.history.last().map(String::as_str) != Some(line) {
            self.history.push(line.to_owned());
        }
    }
}

/// Whether the scope bar and the filter field let `message` through. `scopes`
/// are the scopes picked in the bar, indexed by `Scope as usize`, with none
/// picked letting every scope through ("All"); `filter` is the field's text,
/// trimmed.
fn shows(scopes: &[bool; Scope::ALL.len()], filter: &str, message: &ConsoleMessage) -> bool {
    let picked = !scopes.contains(&true) || scopes[Scope::of(message.kind) as usize];
    picked && (filter.is_empty() || message.contains(filter))
}

/// Recalling the prompt's earlier lines with ↑ and ↓, as in a shell.
#[derive(Default)]
struct Recall {
    /// While ↑ and ↓ browse the history: the index of the entry shown, and
    /// the line that was being typed before browsing started.
    browsing: Option<(usize, String)>,
}

impl Recall {
    /// The line to show at the prompt in place of `line`: the entry of
    /// `history` before (`older`) or after the one shown. Going past the
    /// newest entry brings back the line typed before browsing.
    fn browse(&mut self, history: &[String], line: String, older: bool) -> String {
        let shown = self
            .browsing
            .as_ref()
            .map_or(history.len(), |(index, _)| *index);
        let next = if older {
            shown.checked_sub(1)
        } else {
            Some(shown + 1)
        };
        match next {
            Some(next) if next < history.len() => {
                let typed = match self.browsing.take() {
                    Some((_, typed)) => typed,
                    None => line,
                };
                self.browsing = Some((next, typed));
                history[next].clone()
            }
            // Past the newest entry.
            Some(_) => self.browsing.take().map_or(line, |(_, typed)| typed),
            // Before the oldest.
            None => line,
        }
    }
}

/// Whether the subscriber's filter lets through messages at `level` from the
/// targets it has no directive for. The empty target matches only the
/// filter's default directive, such as the `debug` of `RUST_LOG=debug`.
///
/// Not the filter's maximum level (`LevelFilter::current()`): that includes
/// directives for single targets, such as `RUST_LOG=dust_pbr=debug`.
fn filter_default_allows(level: Level) -> bool {
    // `enabled!` needs the level as a constant.
    match level {
        Level::ERROR => tracing::enabled!(target: "", Level::ERROR),
        Level::WARN => tracing::enabled!(target: "", Level::WARN),
        Level::INFO => tracing::enabled!(target: "", Level::INFO),
        Level::DEBUG => tracing::enabled!(target: "", Level::DEBUG),
        _ => tracing::enabled!(target: "", Level::TRACE),
    }
}
