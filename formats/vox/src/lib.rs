#![feature(generic_const_exprs)]
#![feature(test)]

mod collector;
mod palette;
mod vox_loader;

use bevy_asset::{AddAsset, Handle};
mod geometry;
mod material;
use dust_render::{render_asset::RenderAssetStore, renderable::Renderable, RenderApp};

pub use geometry::VoxGeometry;
pub use material::{GPUPaletteMaterial, PaletteMaterial};
pub use palette::VoxPalette;
pub use vox_loader::*;
#[derive(Default)]
pub struct VoxPlugin;
impl bevy_app::Plugin for VoxPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_asset_loader(vox_loader::VoxLoader::default())
            .add_asset::<VoxPalette>() // TODO: find better way to unify this with others
            .add_plugin(dust_render::geometry::GeometryPlugin::<VoxGeometry>::default())
            .add_plugin(dust_render::render_asset::RenderAssetPlugin::<VoxPalette>::default());

        app.sub_app_mut(RenderApp)
            .init_resource::<RenderAssetStore<VoxPalette>>();
    }
}

#[derive(bevy_ecs::bundle::Bundle)]
pub struct VoxBundle {
    renderable: Renderable,
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
            renderable: Renderable::default(),
            transform: bevy_transform::prelude::Transform::default(),
            global_transform: bevy_transform::prelude::GlobalTransform::default(),
            geometry_handle: geometry,
            material_handle: material,
        }
    }
}
