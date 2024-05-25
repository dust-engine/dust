#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use bevy::ecs::entity::Entity;
use bevy::ecs::reflect::ReflectComponent;
use bevy::ecs::system::lifetimeless::{SRes, SResMut};
use bevy::ecs::system::{Res, SystemParamItem};
use bevy::prelude::IntoSystemConfigs;
use bevy::reflect::Reflect;
use bevy::{
    app::{App, Plugin, PostUpdate, Update},
    asset::{
        processor::{LoadAndSave, LoadTransformAndSave},
        Asset, AssetApp, Handle,
    },
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use blas::{VoxBLASBuilder, VoxSbtBuilder, VoxTLASBuilder};
use dot_vox::Color;
use dust_vdb::hierarchy;
use rhyolite::commands::TransferCommands;
use rhyolite::staging::StagingBelt;
use rhyolite::utils::{AssetUpload, AssetUploadPlugin};
use std::ops::{Deref, DerefMut};

mod blas;
mod loader;

type TreeRoot = hierarchy!(4, 2, 2);
type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;
use rhyolite::ash::vk;
use rhyolite::{Allocator, Buffer, RhyoliteApp};
use rhyolite_rtx::{BLASBuilderPlugin, RtxPlugin, SbtPlugin, TLASBuilderPlugin, TLASBuilderSet};

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
pub struct VoxMaterial(Box<[u8]>);
impl Deref for VoxMaterial {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for VoxMaterial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Asset, TypePath)]
pub struct VoxPalette(Vec<Color>);
#[derive(Asset, TypePath)]
pub struct VoxPaletteGPU(Buffer);
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
impl AssetUpload for VoxPalette {
    type GPUAsset = VoxPaletteGPU;

    type Params = (SRes<Allocator>, SResMut<StagingBelt>);

    fn upload_asset(
        &self,
        commands: &mut impl TransferCommands,
        (allocator, staging_belt): &mut SystemParamItem<Self::Params>,
    ) -> Self::GPUAsset {
        let data =
            unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, self.0.len() * 4) };
        let buffer = Buffer::new_resource_init(
            allocator.clone(),
            staging_belt,
            data,
            1,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            commands,
        );
        VoxPaletteGPU(buffer.unwrap())
    }
}

/// Marker component for Vox instances
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct VoxInstance {
    model: Entity,
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
    geometry: Handle<VoxGeometry>,
    material: Handle<VoxMaterial>,
    palette: Handle<VoxPalette>,
    marker: VoxModel,
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
        app.init_asset_loader::<VoxLoader>()
            .init_asset::<VoxGeometry>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();

        app.add_plugins((
            RtxPlugin,
            BLASBuilderPlugin::<VoxBLASBuilder>::default(),
            TLASBuilderPlugin::<VoxTLASBuilder>::default(),
            SbtPlugin::<VoxSbtBuilder>::default(),
            AssetUploadPlugin::<VoxPalette>::default(),
        ));
    }
}
