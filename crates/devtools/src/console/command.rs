//! The commands the console runs, from its prompt or its socket, parsed with
//! clap, and running them. Running a line needs no console state, so the
//! console's view and its socket run lines the same way. A command that needs
//! the Bevy world asks the Bevy thread for it, and awaits the answer.

use std::ffi::OsStr;
use std::path::PathBuf;

use bevy::diagnostic::FrameCount;
use clap::error::ErrorKind;
use clap::{Parser, Subcommand, ValueEnum};
use dust_pbr::screenshot::{ScreenshotKind, Screenshots};
use dust_pbr::upscaler::{AvailableUpscalers, UpscalerBackend, UpscalerSessionState};
use dust_pbr::{HdrRenderTarget, RenderQualitySettings, UpscalerQualityMode};
use dust_ui::BevyChannel;
use gpui::AsyncApp;
use tracing::{Instrument as _, Level, event};
use tracing_log::log;

use super::screenshot;

/// Why a command that needed the world failed when the Bevy app dropped its
/// call.
const GAME_EXITED: &str = "error: the game has exited";

/// Why the `upscaler` command failed in a game without the PBR renderer.
const NO_UPSCALER: &str = "error: the game has no upscaler";

// A line typed at the prompt. As in a shell, its first word names the
// command: `multicall` has clap match it against the subcommands instead of
// taking it for the program name. Not a doc comment, which `help` would
// print.
#[derive(Parser)]
#[command(
    multicall = true,
    about = "The DevTools console's commands. `help <command>` describes one."
)]
pub(super) struct CommandLine {
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
    /// Print how many frames the game has run
    Frame,
    /// Save a screenshot of the next frame the game draws, without the UI
    ///
    /// The path's extension picks what's saved: `.png`, the frame as an SDR
    /// display shows it, in 8-bit sRGB; `.ktx2`, the frame as presented, in the
    /// swapchain's own format and color space. A relative path is from the
    /// directory the game was started from.
    Screenshot {
        /// Where to save it, ending in .png or .ktx2
        path: PathBuf,
        /// Save the HDR scene the tone mapper reads instead, as .ktx2: linear
        /// radiance before exposure, with BT.709 primaries, in 16-bit floats
        #[arg(long)]
        scene: bool,
    },
    /// Print the denoising / upscaling backend and its render scale, or pick
    /// them
    ///
    /// Given neither, prints the backend picked, the one running, those the
    /// device runs, the render scale, and the render and display resolutions.
    /// A backend the device doesn't run can't be picked; before the game has
    /// found which it runs, any can, and one it doesn't run runs the blit
    /// instead. The blit renders at display resolution, whatever the render
    /// scale.
    Upscaler {
        /// The backend to run
        #[arg(value_enum)]
        backend: Option<Upscaler>,
        /// The render scale of the backends that upscale
        #[arg(short, long, value_enum)]
        quality: Option<Quality>,
    },
    /// Remove all messages from the console tab, when typed there
    Clear,
}

/// The [`UpscalerBackend`]s, as the `upscaler` command names them.
#[derive(Clone, Copy, ValueEnum)]
enum Upscaler {
    /// A linear-filtered blit at display resolution, without denoising
    Blit,
    /// The MetalFX temporal denoised scaler, on Apple GPUs
    #[value(name = "metalfx")]
    MetalFx,
    /// DLSS Ray Reconstruction, on NVIDIA GPUs
    Dlss,
}

impl From<Upscaler> for UpscalerBackend {
    fn from(upscaler: Upscaler) -> Self {
        match upscaler {
            Upscaler::Blit => Self::Blit,
            Upscaler::MetalFx => Self::MetalFx,
            Upscaler::Dlss => Self::Dlss,
        }
    }
}

/// `backend`'s name, as the `upscaler` command takes it.
fn upscaler_name(backend: UpscalerBackend) -> &'static str {
    match backend {
        UpscalerBackend::Blit => "blit",
        UpscalerBackend::MetalFx => "metalfx",
        UpscalerBackend::Dlss => "dlss",
    }
}

/// The [`UpscalerQualityMode`]s, as the `upscaler` command names them.
#[derive(Clone, Copy, ValueEnum)]
enum Quality {
    /// 33% of the display resolution
    UltraPerformance,
    /// 50% of the display resolution
    Performance,
    /// 58% of the display resolution
    Balanced,
    /// 67% of the display resolution
    Quality,
    /// 77% of the display resolution
    UltraQuality,
    /// The display resolution: antialiasing and denoising only
    Native,
}

impl From<Quality> for UpscalerQualityMode {
    fn from(quality: Quality) -> Self {
        match quality {
            Quality::UltraPerformance => Self::UltraPerformance,
            Quality::Performance => Self::Performance,
            Quality::Balanced => Self::Balanced,
            Quality::Quality => Self::Quality,
            Quality::UltraQuality => Self::UltraQuality,
            Quality::Native => Self::Dlaa,
        }
    }
}

/// `quality`'s name, as the `upscaler` command takes it.
fn quality_name(quality: UpscalerQualityMode) -> &'static str {
    match quality {
        UpscalerQualityMode::UltraPerformance => "ultra-performance",
        UpscalerQualityMode::Performance => "performance",
        UpscalerQualityMode::Balanced => "balanced",
        UpscalerQualityMode::Quality => "quality",
        UpscalerQualityMode::UltraQuality => "ultra-quality",
        UpscalerQualityMode::Dlaa => "native",
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// What a command did, when it didn't fail.
pub(super) enum Outcome {
    Nothing,
    /// What it printed, without trailing whitespace.
    Output(String),
    /// Remove all rows from the console view that ran it; nothing anywhere
    /// else.
    Clear,
}

/// Runs `line`, in a `command` span, and finishes with what it did, or why it
/// failed. A command that doesn't
/// need the world, or a mistake, finishes without waiting for a Bevy frame;
/// one that does waits for as long as the game runs none. A client that
/// can't wait gives up on its own, as `dustctl` does after `--timeout`.
pub(super) async fn run(line: &str, cx: &AsyncApp) -> Result<Outcome, String> {
    let span = tracing::info_span!(target: "console", "command", line = line);
    match execute(line, cx).instrument(span).await {
        Ok(Outcome::Output(output)) => Ok(Outcome::Output(output.trim_end().to_owned())),
        Err(error) => Err(error.trim_end().to_owned()),
        outcome => outcome,
    }
}

/// Parses `line` and runs it, for [`run`].
async fn execute(line: &str, cx: &AsyncApp) -> Result<Outcome, String> {
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
        Command::Frame => {
            let bevy = cx.update(|cx| cx.global::<BevyChannel>().clone());
            let frames = bevy
                .run(|world| world.get_resource::<FrameCount>().map(|frames| frames.0))
                .await
                .ok_or(GAME_EXITED)?
                .ok_or("error: the game doesn't count its frames")?;
            Ok(Outcome::Output(frames.to_string()))
        }
        Command::Screenshot { path, scene } => take_screenshot(path, scene, cx).await,
        Command::Upscaler {
            backend: None,
            quality: None,
        } => print_upscaler(cx).await,
        Command::Upscaler { backend, quality } => {
            set_upscaler(backend.map(Into::into), quality.map(Into::into), cx).await
        }
        Command::Clear => Ok(Outcome::Clear),
    }
}

/// Prints the upscaler backend picked, the one running, those the device
/// runs, the render scale, and the render and display resolutions.
async fn print_upscaler(cx: &AsyncApp) -> Result<Outcome, String> {
    let bevy = cx.update(|cx| cx.global::<BevyChannel>().clone());
    let (settings, running, available, extents) = bevy
        .run(|world| {
            let settings = *world.get_resource::<RenderQualitySettings>()?;
            let running = world
                .get_resource::<UpscalerSessionState>()
                .and_then(UpscalerSessionState::active_backend);
            let available = world.get_resource::<AvailableUpscalers>().map(|available| {
                UpscalerBackend::ALL
                    .into_iter()
                    .filter(|&backend| available.contains(backend))
                    .map(upscaler_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            let extents = world
                .get_resource::<HdrRenderTarget>()
                .map(|target| (target.render_extent, target.display_extent));
            Some((settings, running, available, extents))
        })
        .await
        .ok_or(GAME_EXITED)?
        .ok_or(NO_UPSCALER)?;
    let resolution = extents.map_or("not known yet".to_owned(), |(render, display)| {
        format!("{}×{} → {}×{}", render.x, render.y, display.x, display.y)
    });
    Ok(Outcome::Output(format!(
        "backend: {}\nrunning: {}\navailable: {}\nquality: {}\nresolution: {resolution}",
        upscaler_name(settings.upscaler),
        running.map_or("none yet", upscaler_name),
        available.as_deref().unwrap_or("not found yet"),
        quality_name(settings.quality),
    )))
}

/// Picks `backend` to run, unless the device is known not to run it, and the
/// render scale `quality`; keeps the setting of each that's `None`.
async fn set_upscaler(
    backend: Option<UpscalerBackend>,
    quality: Option<UpscalerQualityMode>,
    cx: &AsyncApp,
) -> Result<Outcome, String> {
    let bevy = cx.update(|cx| cx.global::<BevyChannel>().clone());
    bevy.run(move |world| {
        let available = world.get_resource::<AvailableUpscalers>();
        if let Some(backend) =
            backend.filter(|&backend| available.is_some_and(|a| !a.contains(backend)))
        {
            return Err(format!("error: this device doesn't run {}", backend.name()));
        }
        let mut settings = world
            .get_resource_mut::<RenderQualitySettings>()
            .ok_or(NO_UPSCALER)?;
        let next = RenderQualitySettings {
            upscaler: backend.unwrap_or(settings.upscaler),
            quality: quality.unwrap_or(settings.quality),
        };
        // Only a change, which reallocates the render targets, marks the
        // settings changed.
        if *settings != next {
            *settings = next;
        }
        Ok(Outcome::Nothing)
    })
    .await
    .ok_or(GAME_EXITED)?
}

/// Saves a screenshot of the next frame the game draws at `path`, of the HDR
/// scene if `scene`, and prints where.
async fn take_screenshot(path: PathBuf, scene: bool, cx: &AsyncApp) -> Result<Outcome, String> {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase);
    let kind = match (extension.as_deref(), scene) {
        (Some("png"), false) => ScreenshotKind::PresentedSDR,
        (Some("ktx2"), false) => ScreenshotKind::Presented,
        (Some("ktx2"), true) => ScreenshotKind::Scene,
        (_, false) => return Err("error: a screenshot's path ends in .png or .ktx2".to_owned()),
        (_, true) => {
            return Err(
                "error: the scene saves only as KTX2, so its path ends in .ktx2".to_owned(),
            );
        }
    };
    let path = super::user_path(path);
    let bevy = cx.update(|cx| cx.global::<BevyChannel>().clone());
    let screenshot = bevy
        .run(move |world| {
            world
                .get_resource_mut::<Screenshots>()
                .map(|mut screenshots| screenshots.take_screenshot(kind))
        })
        .await
        .ok_or(GAME_EXITED)?
        .ok_or("error: the game takes no screenshots")?
        .await
        .map_err(|error| format!("error: {error}"))?;
    // Encoding an image takes too long for the UI thread.
    cx.background_executor()
        .spawn(async move {
            if let Some(directory) = path.parent() {
                std::fs::create_dir_all(directory).map_err(|error| {
                    format!("error: can't create {}: {error}", directory.display())
                })?;
            }
            match kind {
                ScreenshotKind::PresentedSDR => screenshot::save_png(&path, &screenshot)?,
                ScreenshotKind::Presented | ScreenshotKind::Scene => {
                    screenshot::save_ktx2(&path, &screenshot)?
                }
            }
            Ok(Outcome::Output(format!(
                "{} ({}×{})",
                path.display(),
                screenshot.extent.x,
                screenshot.extent.y
            )))
        })
        .await
}

/// Logs `message` with the target "console". A message the subscriber's
/// filter drops would never show, so for one the command fails, saying so.
fn log_message(level: LogLevel, log_crate: bool, message: &str) -> Result<(), String> {
    // The `tracing` macros need the level as a constant.
    macro_rules! log_at {
        ($tracing_level:expr, $log_level:expr) => {{
            if !tracing::enabled!(target: "console", $tracing_level) {
                return Err(format!(
                    "error: the log filter drops {} messages with the target \"console\"; \
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
