#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use bevy::ecs::entity::{Entity, MapEntities};
use bevy::ecs::reflect::{ReflectComponent, ReflectMapEntities};
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::{
    asset::{Asset, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use dot_vox::Color;
use dust_vdb::{hierarchy};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

mod loader;
mod material;

pub use material::VoxMaterial;

#[derive(Debug, Clone)]
pub struct VoxLeafNode {
    /// Note: we store this value as a vk::AabbPositionsKHR which is less
    /// efficient than possible. We could get away with storing a u16vec4.
    /// That helps us to get down the overhead to 12 bytes
    /// (8 for aabb, 2 for material_ptr, 2 for reserved) instead of 32 bytes.
    aabb: vk::AabbPositionsKHR,
    material_ptr: u32,
    reserved: u32,
}
impl Default for VoxLeafNode {
    fn default() -> Self {
        Self {
            aabb: vk::AabbPositionsKHR {
                min_x: f32::NAN,
                min_y: f32::NAN,
                min_z: f32::NAN,
                max_x: f32::NAN,
                max_y: f32::NAN,
                max_z: f32::NAN,
            },
            material_ptr: 0,
            reserved: 0,
        }
    }
}
/// Leaf node size: 96 bytes
type TreeRoot = hierarchy!(3, 2, 3, VoxLeafNode);

type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

#[derive(Asset, TypePath)]
pub struct VoxGeometry {
    tree: Tree,
    unit_size: f32,
}
impl Deref for VoxGeometry {
    type Target = Tree;
    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}
impl DerefMut for VoxGeometry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tree
    }
}

#[derive(Asset, TypePath)]
pub struct VoxPalette(Box<[Color; 256]>);


impl Deref for VoxPalette {
    type Target = [Color];
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
impl DerefMut for VoxPalette {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}
impl VoxPalette {
    pub fn colorful() -> Self {
        use bevy::color::{Hsva, Srgba};
        let mut hue = 0.0;
        let saturation = 0.8;
        let value = 0.9;

        let mut arr: Box<[MaybeUninit<Color>; 255]> = Box::new([MaybeUninit::uninit(); 255]);
        for x in 0..255 {
            let color = Hsva::new(hue, saturation, value, 1.0);
            let rgb_color: Srgba = color.into();
            let rgb_color: [u8; 4] = rgb_color.to_u8_array();
            arr[x].write(Color {
                r: rgb_color[0],
                g: rgb_color[1],
                b: rgb_color[2],
                a: rgb_color[3],
            });
            hue += 360.0 / 255.0;
        }
        unsafe { std::mem::transmute(arr) }
    }
}

/// Marker component for Vox instances
#[derive(Component, Reflect)]
#[reflect(Component, MapEntities)]
pub struct VoxInstance {
    pub model: Entity,
}
impl MapEntities for VoxInstance {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        self.model = entity_mapper.get_mapped(self.model);
    }
}

impl Default for VoxInstance {
    fn default() -> Self {
        Self {
            model: Entity::PLACEHOLDER
        }
    }
}

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxModel {
    pub geometry: Handle<VoxGeometry>,
    pub material: Handle<VoxMaterial>,
    pub palette: Handle<VoxPalette>,
}


#[derive(Bundle, Default)]
pub struct VoxInstanceBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub instance: VoxInstance,
}
