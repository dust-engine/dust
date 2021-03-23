#![feature(array_methods)]

pub use gfx_hal as hal;

#[cfg(feature = "vulkan")]
pub use gfx_backend_vulkan as back;

#[cfg(feature = "dx12")]
pub use gfx_backend_dx12 as back;

mod camera_projection;
mod descriptor_pool;
mod frame;
mod raytracer;
mod renderer;
mod shared_buffer;
pub use renderer::Config;
pub use renderer::Renderer;

pub use crate::camera_projection::CameraProjection;
use glam::TransformRT;

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
}
