#![feature(generic_const_exprs)]
#![feature(test)]

mod vox_loader;
use bevy_asset::AddAsset;
mod material;
pub use vox_loader::*;
pub use material::{DummyMaterial, GPUDummyMaterial};

pub type VoxGeometry = dust_format_vdb::VdbGeometry<vox_loader::TreeRoot>;
#[derive(Default)]
pub struct VoxPlugin;
impl bevy_app::Plugin for VoxPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_asset_loader(vox_loader::VoxLoader::default())
            .add_plugin(dust_format_vdb::VdbPlugin::<vox_loader::TreeRoot>::default());
    }
}
