mod marching_cube;
mod stack;
mod surface;

use glam::{Vec2, Vec3};

#[derive(Debug)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub uvs: Vec<Vec2>,
    pub normals: Vec<Vec3>,
}

pub use marching_cube::MarchingCubeMeshBuilder;
