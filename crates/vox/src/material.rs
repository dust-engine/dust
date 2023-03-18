use bevy_asset::Handle;
use dust_render::{StandardPipeline, MaterialType};

use crate::VoxPalette;

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "a830cefc-beee-4ee9-89af-3436c0eefe0a"]
pub struct PaletteMaterial {
    palette: Handle<VoxPalette>,
    data: Vec<u8>,
}

impl dust_render::Material for PaletteMaterial {
    type Pipeline = StandardPipeline;

    const TYPE: MaterialType = MaterialType::Procedural;

    fn rahit_shader(ray_type: u32) -> Option<SpecializedShader> {
        None
    }

    fn rchit_shader(ray_type: u32) -> Option<SpecializedShader> {
        None
    }

    fn intersection_shader(ray_type: u32) -> Option<SpecializedShader> {
        None
    }

    type ShaderParameters;

    fn parameters(&self, ray_type: u32) -> Self::ShaderParameters {
        todo!()
    }

    type Object;
}