#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use bevy::ecs::reflect::ReflectComponent;
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
use blas::{VoxBLASBuilder, VoxTLASBuilder};
use dot_vox::Color;
use dust_vdb::hierarchy;
use std::ops::{Deref, DerefMut};

mod blas;
mod loader;

type TreeRoot = hierarchy!(4, 2, 2);
type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;
use rhyolite::ash::vk;
use rhyolite::RhyoliteApp;
use rhyolite_rtx::{
    BLASBuilderSet, BLASStagingBuilderPlugin, RtxPlugin, TLASBuilder, TLASBuilderPlugin,
};

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
#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxInstance;

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxModel;

/// Entities loaded into the scene will have this bundle added.
#[derive(Bundle, Default)]
pub struct VoxBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    geometry: Handle<VoxGeometry>,
    material: Handle<VoxMaterial>,
    palette: Handle<VoxPalette>,
    marker: VoxInstance,
}

pub struct VoxPlugin;

impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>()
            .init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>();

        app.add_plugins((
            RtxPlugin,
            BLASStagingBuilderPlugin::<VoxBLASBuilder>::default(),
            TLASBuilderPlugin::<VoxTLASBuilder>::default(),
        ));

        app.add_systems(
            PostUpdate,
            blas::sync_asset_events_system.before(BLASBuilderSet),
        );
    }
}
