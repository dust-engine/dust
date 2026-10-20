//! The window, the UI and the thread split between them.
//!
//! [`run`] gives the main thread to GPUI and runs the Bevy app on a thread of its
//! own. GPUI opens the window and draws the UI into its own `CAMetalLayer`; the
//! game presents through Vulkan into a second `CAMetalLayer`, in a view placed
//! under GPUI's. The window is transparent, so the game shows wherever the UI
//! draws nothing. Each layer is presented by its own thread, independently.
//!
//! The two threads reach each other by queueing closures, through
//! [`UiHandle`] (a resource of the Bevy app) and [`BevyHandle`] (a global of
//! the GPUI app); see the [`bridge`] module. This crate uses them for the
//! window (its layer and size), which input is the game's (`dust_input`'s
//! `InputGate`), the cursor lock, and quitting once the Bevy app has exited. UI
//! built on top, such as the devtools, uses the same handles.
//!
//! The keyboard and mouse don't pass through the UI thread; `dust_input` reads
//! them on the Bevy thread.

pub mod bridge;

mod plugin;

#[cfg(target_os = "macos")]
mod host;

pub use bridge::{BevyHandle, UiHandle};
pub use plugin::{UiHostPlugin, UiSet};

/// Runs the Bevy app built by `build_app` on its own thread and the UI on this
/// one. Returns when the app exits.
///
/// `build_app` receives the [`UiHostPlugin`] to add. The plugin installs the
/// app's runner, so `build_app` returns the app without running it.
///
/// `build_overlay` builds the UI drawn over the game. It runs once, on the main
/// thread, while the window is created; the [`BevyHandle`] global is set by
/// then. Calls the Bevy app queues through the [`UiHandle`] run only after it
/// returns. Elements of the overlay that should
/// take the mouse from the game call `.occlude()`; everything else lets clicks
/// and scrolls through to the game.
///
/// Must be called on the main thread.
#[cfg(target_os = "macos")]
pub fn run(
    build_app: impl FnOnce(UiHostPlugin) -> bevy::app::App + Send + 'static,
    build_overlay: impl FnOnce(&mut gpui::Window, &mut gpui::App) -> gpui::AnyView + 'static,
) {
    host::run(build_app, build_overlay)
}
