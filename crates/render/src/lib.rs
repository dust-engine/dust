#![feature(array_methods)]
#![feature(array_map)]

pub use gfx_hal as hal;

#[cfg(feature = "vulkan")]
pub use gfx_backend_vulkan as back;

#[cfg(feature = "dx12")]
pub use gfx_backend_dx12 as back;

mod camera_projection;
mod shared_buffer;
mod voxel;

pub use crate::camera_projection::CameraProjection;
use glam::TransformRT;

pub use voxel::Voxel;
pub type Octree = svo::octree::Octree<Voxel>;

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
}

pub mod renderer;
pub mod swapchain;
pub use renderer::Renderer;
