#![feature(array_methods)]
#![feature(array_map)]
#[macro_use]
extern crate memoffset;

mod block_alloc;
mod device_info;
mod raytracer;
pub mod renderer;
mod shared_buffer;
//mod material_repo;
//mod material;
pub mod swapchain;

use dust_core::CameraProjection;
use dust_core::SunLight;
use glam::TransformRT;

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
    pub sunlight: &'a SunLight,
}

pub use renderer::Renderer;
