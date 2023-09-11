#![feature(generic_const_exprs)]
#![feature(generators)]

mod collector;
mod loader;
mod palette;

use bevy_asset::{AssetApp, Handle};
mod geometry;
mod material;

use dust_render::{GeometryPlugin, MaterialPlugin, Renderable};
use dust_vdb::hierarchy;
pub use geometry::VoxGeometry;
pub use loader::*;
use material::DiffuseMaterial;
pub use material::PaletteMaterial;
pub use palette::VoxPalette;

pub type TreeRoot = hierarchy!(4, 2, 2);
pub type Tree = dust_vdb::Tree<TreeRoot>;

#[derive(Default)]
pub struct VoxPlugin;
impl bevy_app::Plugin for VoxPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_asset_loader::<loader::VoxLoader>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxGeometry>()
            .init_asset::<PaletteMaterial>()
            .init_asset::<DiffuseMaterial>()
            .add_plugins(GeometryPlugin::<VoxGeometry>::default())
            .add_plugins(MaterialPlugin::<DiffuseMaterial>::default());
    }
}

#[derive(bevy_ecs::bundle::Bundle)]
pub struct VoxBundle {
    transform: bevy_transform::prelude::Transform,
    global_transform: bevy_transform::prelude::GlobalTransform,
    geometry_handle: Handle<VoxGeometry>,
    material_handle: Handle<DiffuseMaterial>,
    renderable: Renderable,
}
impl VoxBundle {
    pub fn from_geometry_material(
        geometry: Handle<VoxGeometry>,
        material: Handle<DiffuseMaterial>,
    ) -> Self {
        VoxBundle {
            transform: bevy_transform::prelude::Transform::default(),
            global_transform: bevy_transform::prelude::GlobalTransform::default(),
            geometry_handle: geometry,
            material_handle: material,
            renderable: Default::default(),
        }
    }
}
