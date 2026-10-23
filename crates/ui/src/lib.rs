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

#[cfg(not(target_os = "macos"))]
compile_error!(
    "dust_ui only has a game layer for macOS (the `apple` module); other platforms need their own"
);

pub mod bridge;

mod view;

#[cfg(target_os = "macos")]
mod apple;

use std::sync::mpsc::Receiver;

use bevy::{
    app::{App, Last, PostUpdate},
    ecs::{
        schedule::IntoScheduleConfigs,
        system::{LocalBuilder, ParamBuilder, SystemParamBuilder},
    },
    prelude::{Res, SystemSet},
};

use bevy_pumicite::DefaultRenderSet;
pub use bridge::{BevyChannel, GPUIChannel};
pub use view::{BevyExited, BevyView};

#[cfg(target_os = "macos")]
pub use apple::drawable::{
    DrawableImage as SwapchainImage, DrawableImageMetadata as SwapchainImageMetadata,
    MetalDrawable as Swapchain,
};

/// Where systems using the [`Swapchain`]'s current image run: after the image is
/// acquired, and before it's presented. `bevy_pumicite`'s `DefaultRenderSet` runs in
/// it, and it's skipped on frames without an image.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SwapchainSet;

/// The run condition for [`SwapchainSet`]
fn swapchain_image_acquired(swapchain: Option<Res<Swapchain>>) -> bool {
    swapchain.is_some_and(|swapchain| swapchain.current_image().is_some())
}

/// Adds the Bevy side of the bridge to `app`: the [`GPUIChannel`] resource, and
/// the system that runs the calls received from the UI thread through `bevy`.
fn build_gpui_app(app: &mut App, ui: GPUIChannel, bevy: Receiver<bridge::BevyCall>) {
    // Built eagerly: the receiver isn't `Sync`, so a lazy `build_system` builder
    // can't be a system, but the `Local` it ends up in can.
    let run_every_frame = (ParamBuilder, ParamBuilder, LocalBuilder(Some(bevy)))
        .build_state(app.world_mut())
        .build_system(bridge::run_every_frame);
    app.insert_resource(ui)
        .add_systems(
            Last,
            run_every_frame.after(bevy::ecs::system::despawn_unused_registered_systems),
        )
        .configure_sets(
            PostUpdate,
            (
                DefaultRenderSet.in_set(SwapchainSet),
                SwapchainSet.run_if(swapchain_image_acquired),
            ),
        );
}
