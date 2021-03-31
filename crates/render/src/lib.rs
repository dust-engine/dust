#![feature(array_methods)]
#![feature(array_map)]

mod block_alloc;
mod camera_projection;
mod device_info;
mod raytracer;
pub mod renderer;
mod shared_buffer;
pub mod swapchain;
mod voxel;

pub use crate::camera_projection::CameraProjection;
use glam::TransformRT;

pub use voxel::Voxel;
pub type Octree = svo::octree::Octree<Voxel>;

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
}

pub use renderer::Renderer;
