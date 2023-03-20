#![feature(generic_const_exprs)]
#![feature(test)]

mod collector;
mod palette;
mod loader;

use bevy_asset::{AddAsset, Handle};
mod geometry;
mod material;

use dust_vdb::hierarchy;
pub use geometry::VoxGeometry;
pub use material::{PaletteMaterial};
pub use palette::VoxPalette;
pub use loader::*;


pub type TreeRoot = hierarchy!(4, 2, 2);
pub type Tree = dust_vdb::Tree<TreeRoot>;


#[derive(Default)]
pub struct VoxPlugin;
impl bevy_app::Plugin for VoxPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_asset_loader::<loader::VoxLoader>()
            .add_asset::<VoxPalette>();
    }
}

#[derive(bevy_ecs::bundle::Bundle)]
pub struct VoxBundle {
    transform: bevy_transform::prelude::Transform,
    global_transform: bevy_transform::prelude::GlobalTransform,
    geometry_handle: Handle<VoxGeometry>,
    material_handle: Handle<PaletteMaterial>,
}
impl VoxBundle {
    pub fn from_geometry_material(
        geometry: Handle<VoxGeometry>,
        material: Handle<PaletteMaterial>,
    ) -> Self {
        VoxBundle {
            transform: bevy_transform::prelude::Transform::default(),
            global_transform: bevy_transform::prelude::GlobalTransform::default(),
            geometry_handle: geometry,
            material_handle: material,
        }
    }
}
