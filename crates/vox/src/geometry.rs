use std::sync::Arc;

use crate::Tree;

use bevy_ecs::{system::lifetimeless::SRes, world::World};
use bevy_ecs::system::SystemParamItem;

use dust_vdb::{IsLeaf, Node};
use glam::{UVec3, Vec3A};
use rhyolite::ResidentBuffer;


#[derive(bevy_reflect::TypeUuid)]
#[uuid = "307feebb-14b8-4135-be09-ae828decc6a4"]
pub struct VoxGeometry {
    tree: Tree,
    size: [u8; 3],
    pub unit_size: f32,

    blas_input_buffer: ResidentBuffer,
}

impl VoxGeometry {
    pub fn from_tree(tree: Tree, size: [u8; 3], unit_size: f32) -> Self {
        Self {
            tree,
            unit_size,
            size,
            blas_input_buffer: todo!()
        }
    }
    pub fn new(unit_size: f32) -> Self {
        Self {
            tree: Tree::new(),
            size: [255; 3],
            unit_size,
            blas_input_buffer: todo!()
        }
    }
    pub fn set(&mut self, coords: UVec3, value: Option<bool>) {
        self.tree.set_value(coords, value)
    }
    pub fn get(&mut self, coords: UVec3) -> Option<bool> {
        self.tree.get_value(coords)
    }
}