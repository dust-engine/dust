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
use dust_vdb::hierarchy;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

mod geometry;
mod loader;
mod material;

pub use material::{VoxLeafNode, VoxMaterial};

/// Leaf node size: 96 bytes
type TreeRoot = hierarchy!(3, 2, 3, VoxLeafNode);

type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

pub use geometry::VoxGeometry;

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
            model: Entity::PLACEHOLDER,
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

pub struct VoxPlugin;
impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();

        app.add_plugins(rhyolite_bevy::rtx::blas::BLASBuilderPlugin::<geometry::BlasBuilder>::default());
    }
    fn finish(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>();
    }
}
