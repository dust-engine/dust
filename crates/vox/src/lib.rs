#![feature(generic_const_exprs)]

use std::ops::{Deref, DerefMut};

use bevy::{
    app::{App, Plugin}, asset::{processor::{LoadAndSave, LoadTransformAndSave}, Asset, AssetApp, Handle}, ecs::{bundle::Bundle, component::Component}, reflect::TypePath, transform::components::{GlobalTransform, Transform}
};
use dot_vox::Color;
use dust_vdb::{hierarchy};

mod loader;
mod blas;

type TreeRoot = hierarchy!(4, 2, 2);
type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

#[derive(Asset, TypePath)]
pub struct VoxGeometry(Tree);
impl Deref for VoxGeometry {
    type Target = Tree;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for VoxGeometry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
#[derive(Component)]
pub struct VoxInstance;

/// Entities loaded into the scene will have this bundle added.
#[derive(Bundle, Default)]
pub struct VoxBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    geometry: Handle<VoxGeometry>,
    material: Handle<VoxMaterial>,
    palette: Handle<VoxPalette>,
}

pub struct VoxPlugin;

impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>()
            .init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterial>();
    }
}
