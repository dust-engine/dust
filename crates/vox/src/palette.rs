use bevy_asset::Asset;
use rhyolite::{future::RenderData, ResidentBuffer};

#[derive(bevy_reflect::TypePath, Asset)]
pub struct VoxPalette {
    pub colors: Box<[dot_vox::Color; 255]>,
    pub buffer: ResidentBuffer,
}
impl RenderData for VoxPalette {}
