#![feature(generic_const_exprs)]

use std::ops::{Deref, DerefMut};

use bevy::{
    app::{App, Plugin}, asset::{processor::{LoadAndSave, LoadTransformAndSave}, Asset, AssetApp, Handle}, ecs::bundle::Bundle, reflect::TypePath, transform::components::{GlobalTransform, Transform}
};
use dot_vox::Color;
use dust_vdb::{hierarchy};

mod loader;
mod transformer;

type TreeRoot = hierarchy!(4, 2, 2);
type Tree = dust_vdb::Tree<TreeRoot>;


#[derive(Asset, TypePath)]
pub struct VoxTree(Tree);
impl Deref for VoxTree {
    type Target = Tree;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for VoxTree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub use loader::*;

pub use transformer::*;


#[derive(Asset, TypePath)]
pub struct VoxModel(dot_vox::Model);
impl Deref for VoxModel {
    type Target = dot_vox::Model;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for VoxModel {
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

/// Entities loaded into the scene will have this bundle added.
#[derive(Bundle, Default)]
pub struct VoxBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    model: Handle<VoxModel>,
    palette: Handle<VoxPalette>,
}

pub struct VoxPlugin;

impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>()
            .init_asset::<VoxModel>()
            .init_asset::<VoxPalette>();
    }
}
