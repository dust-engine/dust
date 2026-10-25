//! The game in a GPUI window, and the thread split between the game and the UI.
//!
//! [`BevyView`] is a GPUI view that runs a Bevy app on a thread of its own. GPUI
//! draws the UI into its own `CAMetalLayer`; the game draws into a second
//! `CAMetalLayer`, in a view placed under GPUI's. The window is transparent, so
//! the game shows wherever the UI draws nothing. Each layer is presented by its
//! own thread, independently: a `CAMetalDisplayLink` on the game's layer updates
//! the Bevy app once per display frame, and the game renders straight into the
//! link's drawables (see the `apple` module).
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
//! the display link's drawables; elsewhere they're `bevy_pumicite`'s. Both have
//! the same API.

#[cfg(not(target_os = "macos"))]
compile_error!(
    "dust_ui only has a game layer for macOS (the `apple` module); other platforms need their own"
);

pub mod bridge;

mod view;

#[cfg(target_vendor = "apple")]
mod apple;

use std::sync::mpsc::Receiver;

use bevy::{
    app::{App, Last},
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

#[cfg(not(target_vendor = "apple"))]
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
