#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use attributes::AttributeAllocator;
use bevy::app::Update;
use bevy::ecs::entity::{Entity, MapEntities};
use bevy::ecs::reflect::{ReflectComponent, ReflectMapEntities};
use bevy::math::UVec3;
use bevy::prelude::EntityMapper;
use bevy::reflect::Reflect;
use bevy::{
    app::{App, Plugin},
    asset::{Asset, AssetApp, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use dot_vox::Color;
use dust_vdb::{hierarchy, TreeLike};
use rhyolite::ash::vk;
use rhyolite::utils::AssetUploadPlugin;
use rhyolite::RhyoliteApp;
use std::ops::{Deref, DerefMut};

mod attributes;
mod builder;
mod loader;
mod physics;
mod resource;

type TreeRoot = hierarchy!(3, 3, 2, u32);
type Tree = dust_vdb::MutableTree<TreeRoot>;
type ImmutableTree = dust_vdb::ImmutableTree<TreeRoot>;

pub use loader::*;
use rhyolite_rtx::{BLASBuilderPlugin, RtxPlugin, SbtPlugin, TLASBuilderPlugin};

#[derive(Asset, TypePath)]
pub struct VoxGeometry {
    tree: ImmutableTree,
    aabb_min: UVec3,
    aabb_max: UVec3,
    unit_size: f32,
}
impl VoxGeometry {
    pub fn from_tree_with_unit_size(tree: ImmutableTree, unit_size: f32) -> Self {
        Self {
            aabb_min: UVec3::ZERO,
            aabb_max: tree.extent(),
            tree,
            unit_size,
        }
    }
    pub fn aabb(&self) -> (UVec3, UVec3) {
        (self.aabb_min, self.aabb_max)
    }
    pub fn size(&self) -> UVec3 {
        self.aabb_max - self.aabb_min + UVec3::ONE
    }
}
impl Deref for VoxGeometry {
    type Target = ImmutableTree;
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
pub struct VoxMaterial(AttributeAllocator);

#[derive(Asset, TypePath)]
pub struct VoxPalette(Vec<Color>);

impl Deref for VoxPalette {
    type Target = [Color];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for VoxPalette {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Marker component for Vox instances
#[derive(Component, Reflect)]
#[reflect(Component, MapEntities)]
pub struct VoxInstance {
    model: Entity,
}
impl MapEntities for VoxInstance {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        self.model = entity_mapper.map_entity(self.model);
    }
}

impl Default for VoxInstance {
    fn default() -> Self {
        Self {
            model: Entity::from_raw(u32::MAX),
        }
    }
}

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxModel;

/// Entities loaded into the scene will have this bundle added.
#[derive(Bundle, Default)]
pub struct VoxModelBundle {
    pub geometry: Handle<VoxGeometry>,
    pub material: Handle<VoxMaterial>,
    pub palette: Handle<VoxPalette>,
    pub marker: VoxModel,
}

#[derive(Bundle, Default)]
pub struct VoxInstanceBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    instance: VoxInstance,
}

pub struct VoxPlugin;

impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();

        app.add_plugins((
            RtxPlugin,
            BLASBuilderPlugin::<builder::VoxBLASBuilder>::default(),
            TLASBuilderPlugin::<builder::VoxTLASBuilder>::default(),
            SbtPlugin::<builder::VoxSbtBuilder>::default(),
            AssetUploadPlugin::<crate::resource::VoxPaletteGPU>::default(),
            AssetUploadPlugin::<crate::resource::VoxGeometryGPU>::default(),
        ));

        app.enable_feature::<vk::PhysicalDeviceFeatures>(|x| &mut x.shader_int16)
            .unwrap();
        app.enable_feature::<vk::PhysicalDevice8BitStorageFeatures>(|x| {
            &mut x.storage_buffer8_bit_access
        })
        .unwrap();
        app.enable_feature::<vk::PhysicalDevice16BitStorageFeatures>(|x| {
            &mut x.storage_buffer16_bit_access
        })
        .unwrap();
        app.enable_feature::<vk::PhysicalDeviceShaderFloat16Int8Features>(|x| &mut x.shader_int8)
            .unwrap();

        app.add_systems(Update, physics::insert_collider_system);
    }
    fn finish(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>();
    }
}
