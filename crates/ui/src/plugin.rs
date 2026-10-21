//! The Bevy side of the UI host.

use bevy::prelude::*;

use bevy_pumicite::DefaultRenderSet;

use crate::bridge::{self, UiHandle};
use crate::{SwapchainSet, swapchain_image_acquired};

/// Connects the Bevy app to the UI thread, and inserts the [`UiHandle`]
/// resource. [`BevyView`](crate::BevyView) adds it before the rest of the app.
pub(crate) struct UiHostPlugin {
    pub(crate) ui: UiHandle,
}

impl Plugin for UiHostPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.ui.clone())
            .add_systems(
                Last,
                bridge::run_every_frame.after(bevy::ecs::system::despawn_unused_registered_systems),
            )
            .configure_sets(
                PostUpdate,
                (
                    DefaultRenderSet.in_set(SwapchainSet),
                    SwapchainSet.run_if(swapchain_image_acquired),
                ),
            );
        #[cfg(target_os = "macos")]
        crate::apple::build(app);
    }
}
