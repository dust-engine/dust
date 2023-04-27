use bevy_asset::{AssetServer, Assets, Handle};
use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use dust_render::{MaterialType, StandardPipeline};

use crate::{VoxGeometry, VoxPalette};
use dust_render::SpecializedShader;
use rhyolite::{ash::vk, BufferLike, ResidentBuffer};

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "a830cefc-beee-4ee9-89af-3436c0eefe0a"]
pub struct PaletteMaterial {
    palette: Handle<VoxPalette>,
    pub(crate) geometry: Handle<VoxGeometry>,
    /// Compacted list of indexes into the palette array.
    data: ResidentBuffer,
}

impl PaletteMaterial {
    pub fn new(
        geometry: Handle<VoxGeometry>,
        palette: Handle<VoxPalette>,
        data: ResidentBuffer,
    ) -> Self {
        Self {
            palette,
            data,
            geometry,
        }
    }
}

pub struct PaletteMaterialShaderParams {
    /// Pointer to a list of u64 indexed by block id
    geometry_ptr: u64,

    /// Pointer to a list of u8, indexed by voxel id, each denoting offset into palette_ptr.
    /// Voxel id is defined as block id + offset inside block.
    material_ptr: u64,

    /// Pointer to a list of 256 u8 colors
    palette_ptr: u64,
}

impl dust_render::Material for PaletteMaterial {
    type Pipeline = StandardPipeline;

    const TYPE: MaterialType = MaterialType::Procedural;

    fn rahit_shader(_ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader> {
        None
    }

    fn rchit_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader> {
        match ray_type {
            Self::Pipeline::PRIMARY_RAYTYPE => Some(SpecializedShader::for_shader(
                asset_server.load("hit.rchit.spv"),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )),
            Self::Pipeline::PHOTON_RAYTYPE => Some(SpecializedShader::for_shader(
                asset_server.load("photon.rchit.spv"),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )),
            _ => None,
        }
    }

    fn intersection_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader> {
        Some(SpecializedShader::for_shader(
            asset_server.load("hit.rint.spv"),
            vk::ShaderStageFlags::INTERSECTION_KHR,
        ))
    }

    type ShaderParameters = PaletteMaterialShaderParams;
    type ShaderParameterParams = (SRes<Assets<VoxGeometry>>, SRes<Assets<VoxPalette>>);
    fn parameters(
        &self,
        _ray_type: u32,
        params: &mut SystemParamItem<Self::ShaderParameterParams>,
    ) -> Self::ShaderParameters {
        let (geometry_store, palette_store) = params;
        let geometry = geometry_store.get(&self.geometry).unwrap();
        let palette = palette_store.get(&self.palette).unwrap();
        PaletteMaterialShaderParams {
            geometry_ptr: geometry.geometry_buffer().device_address(),
            material_ptr: self.data.device_address(),
            palette_ptr: palette.buffer.device_address(),
        }
    }
}
