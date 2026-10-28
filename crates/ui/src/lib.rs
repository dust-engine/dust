//! The game in a GPUI window, and the thread split between the game and the UI.
//!
//! [`BevyView`] is a GPUI view that runs a Bevy app on a thread of its own. GPUI
//! draws the UI into its own compositor layer; the game draws into a second
//! layer, placed under GPUI's: the platform module's `GameLayer`. The window is
//! transparent, so the game shows wherever the UI draws nothing. Each layer is
//! presented by its own thread, independently. On macOS, the game's layer is a
//! `CAMetalLayer` in a view under GPUI's: a `CAMetalDisplayLink` on it updates
//! the Bevy app once per display frame, and the game renders straight into the
//! link's drawables (see the `apple` module). On Windows, it's a
//! DirectComposition visual under GPUI's: the Bevy app updates once per
//! compositor frame, and the game renders straight into buffers it presents to
//! the visual with the composition swapchain API (see the `win32` module).
//!
//! The two threads reach each other by queueing closures, through
//! [`GPUIChannel`] (a resource of the Bevy app) and [`BevyChannel`] (a global of
//! the GPUI app); see the [`bridge`] module. This crate uses them to set
//! `dust_input`'s `CursorMode` and for the app's exit. UI built on top, such as
//! the devtools, uses the same channels.
//!
//! The keyboard and mouse don't pass through the UI thread; `dust_input` reads
//! them on the Bevy thread.
//!
//! The game renders into the [`Swapchain`] resource's current image in
//! [`SwapchainSet`], a submission of its own after `bevy_pumicite`'s
//! `DefaultRenderSet`. On Apple platforms they're the `apple` module's, made from
//! the display link's drawables; on Windows the `win32` module's, made from the
//! presentation buffers; elsewhere they're `bevy_pumicite`'s. All have the same
//! API.

#[cfg(not(any(target_os = "macos", windows)))]
compile_error!(
    "dust_ui only has game layers for macOS (the `apple` module) and Windows (the `win32` module); other platforms need a module with a `GameLayer` of their own"
);

pub mod bridge;

mod view;

// Each platform has a module with a `GameLayer`: the layer the game draws into,
// under GPUI's. `BevyView` creates it with `new(&Window)`, fits it to the view
// with `set_bounds(bounds, scale_factor)`, locks the cursor for the game with
// `set_cursor_locked(bool)`, and runs the Bevy app with the runner from
// `create_bevy_runner()`, which adds the `Swapchain` and calls
// `finish_plugins` before the first update.
#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
use apple as platform;
#[cfg(windows)]
mod win32;
#[cfg(windows)]
use win32 as platform;

use std::sync::mpsc::Receiver;

use bevy::{
    app::{App, Last, PluginsState},
    ecs::{
        schedule::IntoScheduleConfigs,
        system::{LocalBuilder, ParamBuilder, SystemParamBuilder},
    },
};

pub use bridge::{BevyChannel, GPUIChannel};
pub use view::{BevyExited, BevyView};

#[cfg(target_vendor = "apple")]
pub use apple::drawable::{
    DrawableImage as SwapchainImage, DrawableImageMetadata as SwapchainImageMetadata,
    MetalDrawable as Swapchain, SwapchainSet,
};

#[cfg(windows)]
pub use win32::swapchain::{
    PresentationImage as SwapchainImage, PresentationImageMetadata as SwapchainImageMetadata,
    Swapchain, SwapchainSet,
};

#[cfg(not(any(target_vendor = "apple", windows)))]
pub use bevy_pumicite::{
    pumicite::swapchain::SwapchainImageInner as SwapchainImage,
    swapchain::{Swapchain, SwapchainSet},
};

/// Adds the Bevy side of the bridge to `app`: the [`GPUIChannel`] resource, and
/// the system that runs the calls received from the UI thread through `bevy`.
fn build_gpui_app(app: &mut App, ui: GPUIChannel, bevy: Receiver<bridge::BevyCall>) {
    // Built eagerly: the receiver isn't `Sync`, so a lazy `build_system` builder
    // can't be a system, but the `Local` it ends up in can.
    let run_every_frame = (ParamBuilder, ParamBuilder, LocalBuilder(Some(bevy)))
        .build_state(app.world_mut())
        .build_system(bridge::run_every_frame);
    app.insert_resource(ui).add_systems(
        Last,
        run_every_frame.after(bevy::ecs::system::despawn_unused_registered_systems),
    );
}

/// Finishes building `app`, as a runner does before its first update: waits for
/// plugins that are still being added, then finishes and cleans them up.
fn finish_plugins(app: &mut App) {
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }
}
