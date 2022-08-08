#![feature(generic_const_exprs)]
#![feature(test)]

mod collector;
mod vox_loader;
mod palette;
use std::sync::Arc;

use bevy_asset::AddAsset;
mod geometry;
mod material;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, RenderAsset, RenderAssetStore}, RenderApp,
};
pub use palette::VoxPalette;
use dustash::resources::alloc::Allocator;
pub use geometry::VoxGeometry;
pub use material::{PaletteMaterial, GPUPaletteMaterial};
pub use vox_loader::*;
#[derive(Default)]
pub struct VoxPlugin;
impl bevy_app::Plugin for VoxPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_asset_loader(vox_loader::VoxLoader::new(app.world.resource::<Arc<Allocator>>().clone()))
            .add_asset::<VoxPalette>() // TODO: find better way to unify this with others
            .add_plugin(dust_render::geometry::GeometryPlugin::<VoxGeometry>::default())
            .add_plugin(dust_render::render_asset::RenderAssetPlugin::<VoxPalette>::default());
        
        app.sub_app_mut(RenderApp)
        .init_resource::<RenderAssetStore<VoxPalette>>();
    }
}
