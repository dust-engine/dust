//! Everything Windows-specific: the [`GameLayer`], a DirectComposition visual under
//! GPUI's, with the runner that updates the Bevy app once per compositor frame
//! ([`layer`]), and the swapchain that presents into that visual with the
//! composition swapchain API ([`swapchain`]).

mod layer;
pub(crate) mod swapchain;

pub(crate) use layer::GameLayer;
