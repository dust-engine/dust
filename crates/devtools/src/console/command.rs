//! The commands the console's prompt runs, parsed with clap.

use bevy::log::Level;
use bevy::log::tracing::{self, event};
use bevy::prelude::*;
use clap::error::ErrorKind;
use clap::{Parser, Subcommand, ValueEnum};
use tracing_log::log;

use super::Console;
use crate::{DevTools, DockSide};

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

/// Runs `line`, returning what it printed, if anything, or why it failed.
pub(super) fn run(line: &str, world: &mut World) -> Result<Option<String>, String> {
    let words = shlex::split(line).ok_or("error: unterminated quote")?;
    let command = match CommandLine::try_parse_from(words) {
        Ok(command_line) => command_line.command,
        // `help` and `--help` come back as errors too.
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            return Ok(Some(error.render().to_string()));
        }
        Err(error) => return Err(error.render().to_string()),
    };
    match command {
        Command::Echo { words } => Ok(Some(words.join(" "))),
        Command::Log {
            level,
            log_crate,
            message,
        } => log_message(level, log_crate, &message.join(" ")),
        Command::Clear => {
            world.resource_mut::<Console>().rows.clear();
            Ok(None)
        }
        Command::Dock { side } => {
            let mut devtools = world
                .get_resource_mut::<DevTools>()
                .ok_or("error: DevTools isn't set up")?;
            devtools.dock = match side {
                Side::Left => DockSide::Left,
                Side::Bottom => DockSide::Bottom,
                Side::Right => DockSide::Right,
                Side::Undock => DockSide::Floating,
            };
            Ok(None)
        }
    }
}

/// Logs `message` with the target "console". A message `LogPlugin`'s filter
/// drops would never show, so for one the command fails, saying so.
fn log_message(level: LogLevel, log_crate: bool, message: &str) -> Result<Option<String>, String> {
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
    Ok(None)
}
