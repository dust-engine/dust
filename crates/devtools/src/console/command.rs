//! The commands the console's prompt runs, parsed with clap.

use bevy::log::Level;
use bevy::log::tracing::{self, event};
use bevy::prelude::*;
use clap::error::ErrorKind;
use clap::{Parser, Subcommand, ValueEnum};
use tracing_log::log;

use super::ConsoleEvent;
use crate::DockSide;

/// A line typed at the prompt. As in a shell, its first word names the
/// command: `multicall` has clap match it against the subcommands instead of
/// taking it for the program name.
#[derive(Parser)]
#[command(multicall = true)]
struct CommandLine {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the words given
    Echo {
        /// Words to print, separated by a space
        words: Vec<String>,
    },
    /// Log a message, which then shows in the console like any other
    Log {
        /// The message's level
        #[arg(short, long, value_enum, default_value_t = LogLevel::Info)]
        level: LogLevel,
        /// Log through the `log` crate instead of `tracing`
        #[arg(long)]
        log_crate: bool,
        /// The message, its words separated by a space
        #[arg(required = true)]
        message: Vec<String>,
    },
    /// Remove all messages from the console
    Clear,
    /// Move DevTools to a side of the window, or into a window of its own
    Dock {
        /// Where DevTools goes
        side: Side,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Clone, Copy, ValueEnum)]
enum Side {
    Left,
    Bottom,
    Right,
    Undock,
}

/// What a command did, when it didn't fail.
pub(super) enum Outcome {
    Nothing,
    /// What it printed.
    Output(String),
    /// A change to the console or DevTools, which live on the UI thread.
    Console(ConsoleEvent),
}

/// Runs `line`, returning what it did, or why it failed.
pub(super) fn run(line: &str, _world: &mut World) -> Result<Outcome, String> {
    let words = shlex::split(line).ok_or("error: unterminated quote")?;
    let command = match CommandLine::try_parse_from(words) {
        Ok(command_line) => command_line.command,
        // `help` and `--help` come back as errors too.
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            return Ok(Outcome::Output(error.render().to_string()));
        }
        Err(error) => return Err(error.render().to_string()),
    };
    match command {
        Command::Echo { words } => Ok(Outcome::Output(words.join(" "))),
        Command::Log {
            level,
            log_crate,
            message,
        } => log_message(level, log_crate, &message.join(" ")).map(|()| Outcome::Nothing),
        Command::Clear => Ok(Outcome::Console(ConsoleEvent::Clear)),
        Command::Dock { side } => Ok(Outcome::Console(ConsoleEvent::Dock(match side {
            Side::Left => DockSide::Left,
            Side::Bottom => DockSide::Bottom,
            Side::Right => DockSide::Right,
            Side::Undock => DockSide::Floating,
        }))),
    }
}

/// Logs `message` with the target "console". A message `LogPlugin`'s filter
/// drops would never show, so for one the command fails, saying so.
fn log_message(level: LogLevel, log_crate: bool, message: &str) -> Result<(), String> {
    // The `tracing` macros need the level as a constant.
    macro_rules! log_at {
        ($tracing_level:expr, $log_level:expr) => {{
            if !tracing::enabled!(target: "console", $tracing_level) {
                return Err(format!(
                    "error: LogPlugin's filter drops {} messages with the target \"console\"; \
                     RUST_LOG=console={} lets them through",
                    $tracing_level,
                    $tracing_level.as_str().to_lowercase(),
                ));
            }
            if log_crate {
                log::log!(target: "console", $log_level, "{message}");
            } else {
                event!(target: "console", $tracing_level, "{message}");
            }
        }};
    }
    match level {
        LogLevel::Error => log_at!(Level::ERROR, log::Level::Error),
        LogLevel::Warn => log_at!(Level::WARN, log::Level::Warn),
        LogLevel::Info => log_at!(Level::INFO, log::Level::Info),
        LogLevel::Debug => log_at!(Level::DEBUG, log::Level::Debug),
        LogLevel::Trace => log_at!(Level::TRACE, log::Level::Trace),
    }
    Ok(())
}
