//! Everything macOS-specific: the game's layer under GPUI's ([`layer`]), the runner
//! that updates the Bevy app from a `CAMetalDisplayLink` on that layer
//! ([`display_link`]), and the swapchain made from the link's drawables
//! ([`drawable`]).

mod display_link;
pub(crate) mod drawable;
mod layer;

use bevy::prelude::*;

pub(crate) use display_link::displaylink_runner;
pub(crate) use layer::{GameLayer, set_cursor};

/// Adds the Bevy side of the macOS host to the app.
pub(crate) fn build(app: &mut App) {
    app.add_plugins(drawable::CAMetalDrawablePlugin);
}
