#![feature(generic_const_exprs)]
#![feature(test)]

mod collector;
mod palette;
mod vox_loader;

use bevy_asset::AddAsset;
mod geometry;
mod material;
use dust_render::{render_asset::RenderAssetStore, RenderApp};

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
