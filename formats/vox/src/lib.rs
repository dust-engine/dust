#![feature(generic_const_exprs)]
#![feature(test)]

mod vox_loader;
use bevy_asset::AddAsset;
mod geometry;
mod material;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, RenderAsset},
};
pub use geometry::VoxGeometry;
pub use material::{DummyMaterial, GPUDummyMaterial};
pub use vox_loader::*;
#[derive(Default)]
pub struct VoxPlugin;
impl bevy_app::Plugin for VoxPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_asset_loader(vox_loader::VoxLoader::default())
            .add_plugin(dust_render::geometry::GeometryPlugin::<VoxGeometry>::default());
    }
}
