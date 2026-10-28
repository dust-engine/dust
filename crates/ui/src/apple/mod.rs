//! Everything macOS-specific: the [`GameLayer`], the game's layer beneath GPUI's
//! layer, with the runner that updates the Bevy app from a `CAMetalDisplayLink`
//! on that layer ([`layer`]), and the swapchain made from the link's drawables
//! ([`drawable`]).

pub(crate) mod drawable;
mod layer;

pub(crate) use layer::GameLayer;
