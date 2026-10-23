//! The "Console" DevTools tab, modeled on Safari's: the messages logged through
//! `tracing` and `log`, a scope bar filtering them by severity, and a prompt
//! that runs commands.
//!
//! The console's rows live with its view on the UI thread. Changes reach them
//! through a channel, the [`ConsoleLog`], which the view drains: log messages,
//! sent by the [`ConsoleLayer`] from whichever thread logged them, and what
//! the commands submitted at the prompt did. A command runs on the Bevy
//! thread, with the world. The commands live in [`command`].

mod command;
mod layer;
#[cfg(test)]
mod tests;
mod ui;

use std::collections::VecDeque;

use bevy::platform::time::Instant;
use bevy::prelude::World;
use futures_channel::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::Level;

use dust_ui::BevyChannel;
pub use layer::ConsoleLayer;
pub(crate) use ui::ConsoleView;

/// Makes the `tracing` layer that copies log messages into the DevTools
/// console, and the [`ConsoleLog`] they wait in until the console takes them.
/// Add the layer to the global subscriber before anything logs, and pass the
/// log to [`overlay`](crate::overlay):
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
    let (sender, events) = futures_channel::mpsc::unbounded();
    let layer = ConsoleLayer {
        events: sender.clone(),
    };
    let log = ConsoleLog {
        sender,
        events,
        startup: Instant::now(),
    };
    (layer, log)
}

/// The changes on their way to the console view: the messages its
/// [`ConsoleLayer`] copies, and what the commands run at its prompt did.
pub struct ConsoleLog {
    /// Where the commands send what they did.
    sender: UnboundedSender<ConsoleEvent>,
    events: UnboundedReceiver<ConsoleEvent>,
    /// When logging started; the console shows message times from it.
    startup: Instant,
}

/// A change to the console, sent to its view from another thread.
pub(crate) enum ConsoleEvent {
    Message(ConsoleMessage),
    /// Removes all rows.
    Clear,
}

/// The most rows the console keeps; the oldest are dropped first.
const MAX_ROWS: usize = 10_000;

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
    time: Instant,
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
struct Source {
    target: String,
    module_path: Option<String>,
    file: Option<String>,
    line: Option<u32>,
    /// Name of the thread that logged the message.
    thread: Option<String>,
}

impl ConsoleMessage {
    /// A command line, or what it printed.
    fn evaluation(kind: MessageKind, text: String) -> Self {
        Self {
            kind,
            text,
            fields: Vec::new(),
            spans: Vec::new(),
            source: None,
            time: Instant::now(),
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

#[derive(Default)]
struct Console {
    rows: VecDeque<Row>,
    /// The `serial` of the next row.
    next_serial: u64,
    /// The scopes picked in the scope bar, indexed by `Scope as usize`. With
    /// none picked, every row shows ("All").
    scopes: [bool; Scope::ALL.len()],
    /// Only rows containing this text show.
    filter: String,
    prompt: Prompt,
    /// Makes the list jump to its end; set when a line is submitted.
    scroll_to_bottom: bool,
    /// The scopes a message has arrived in, indexed like `scopes`; kept when
    /// the console is cleared.
    received: [bool; Scope::ALL.len()],
}

struct Row {
    message: ConsoleMessage,
    /// How many times in a row the message arrived.
    count: u32,
    /// Unique among rows: the view tracks the rows its list shows by it, and
    /// keys their elements with it.
    serial: u64,
    /// Whether the row shows the message's fields, spans and source.
    expanded: bool,
}

impl Console {
    fn push(&mut self, message: ConsoleMessage) {
        self.received[Scope::of(message.kind) as usize] = true;
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
            serial: self.next_serial,
            expanded: false,
        });
        self.next_serial += 1;
    }

    /// Whether the scope bar and the filter field let `message` through.
    /// `filter` is [`Self::filter`], trimmed.
    fn shows(&self, message: &ConsoleMessage, filter: &str) -> bool {
        let picked = !self.scopes.contains(&true) || self.scopes[Scope::of(message.kind) as usize];
        picked && (filter.is_empty() || message.contains(filter))
    }

    /// Takes the prompt's line, to be run by [`run_line`].
    fn submit(&mut self) -> Option<String> {
        let line = std::mem::take(&mut self.prompt.line);
        self.prompt.browsing = None;
        if line.trim().is_empty() {
            return None;
        }
        if self.prompt.history.last() != Some(&line) {
            self.prompt.history.push(line.clone());
        }
        self.push(ConsoleMessage::evaluation(
            MessageKind::Command,
            line.clone(),
        ));
        self.scroll_to_bottom = true;
        Some(line)
    }

    fn apply(&mut self, event: ConsoleEvent) {
        match event {
            ConsoleEvent::Message(message) => self.push(message),
            ConsoleEvent::Clear => self.rows.clear(),
        }
    }
}

#[derive(Default)]
struct Prompt {
    /// The line being typed.
    line: String,
    /// The lines submitted, oldest first.
    history: Vec<String>,
    /// While ↑ and ↓ browse `history`: the index of the entry in `line`, and
    /// the line that was being typed before browsing started.
    browsing: Option<(usize, String)>,
}

impl Prompt {
    /// Replaces the line with the history entry before (`older`) or after the
    /// one shown, as ↑ and ↓ do in a shell. Going past the newest entry brings
    /// back the line typed before browsing.
    fn browse(&mut self, older: bool) {
        let shown = self
            .browsing
            .as_ref()
            .map_or(self.history.len(), |(index, _)| *index);
        let Some(next) = (if older {
            shown.checked_sub(1)
        } else {
            Some(shown + 1)
        }) else {
            return;
        };
        if next < self.history.len() {
            let typed = match self.browsing.take() {
                Some((_, typed)) => typed,
                None => std::mem::take(&mut self.line),
            };
            self.line = self.history[next].clone();
            self.browsing = Some((next, typed));
        } else if let Some((_, typed)) = self.browsing.take() {
            self.line = typed;
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

/// Runs a line submitted at the prompt in a `command` span, and returns what
/// it did to the console.
fn run_line(line: &str, world: &mut World) -> Option<ConsoleEvent> {
    let result = tracing::info_span!(target: "console", "command", line = line)
        .in_scope(|| command::run(line, world));
    let (kind, text) = match result {
        Ok(command::Outcome::Nothing) => return None,
        Ok(command::Outcome::Console(event)) => return Some(event),
        Ok(command::Outcome::Output(output)) => (MessageKind::Output, output),
        Err(error) => (MessageKind::CommandError, error),
    };
    let text = text.trim_end().to_owned();
    Some(ConsoleEvent::Message(ConsoleMessage::evaluation(
        kind, text,
    )))
}

/// Runs `line` on the Bevy thread, sending what it did back to the console
/// through `console`.
fn submit_line(cx: &gpui::App, console: UnboundedSender<ConsoleEvent>, line: String) {
    cx.global::<BevyChannel>().update(move |world| {
        if let Some(event) = run_line(&line, world) {
            console.unbounded_send(event).ok();
        }
    });
}
