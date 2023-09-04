use bevy_asset::{Asset, AssetServer, Assets, Handle};
use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use dust_render::{MaterialType, StandardPipeline};

use crate::{VoxGeometry, VoxPalette};
use dust_render::SpecializedShader;
use rhyolite::{ash::vk, BufferLike, ResidentBuffer};

#[derive(bevy_reflect::TypePath, Asset)]
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

#[derive(bevy_reflect::TypePath, Asset)]
pub struct DiffuseMaterial {
    material: Handle<PaletteMaterial>,
    pub(crate) irradiance_cache: ResidentBuffer,
}

impl DiffuseMaterial {
    pub fn new(material: Handle<PaletteMaterial>, irradiance_cache: ResidentBuffer) -> Self {
        Self {
            material,
            irradiance_cache,
        }
    }
}

pub struct DiffuseMaterialIrradianceCacheEntryFace {
    irradiance: [u16; 3],
    /// Represents 4x4 faces.
    mask: u16,
}
pub struct DiffuseMaterialIrradianceCacheEntry {
    /// The six faces. 8 bytes per face, 48 bytes in total.
    faces: [DiffuseMaterialIrradianceCacheEntryFace; 6],
    lastAccessedFrames: [u16; 6],
    _reserved: u32,
}

#[repr(C)]
pub struct DiffuseMaterialShaderParams {
    /// Pointer to a list of u64 indexed by block id
    geometry_ptr: u64,

    /// Pointer to a list of u8, indexed by voxel id, each denoting offset into palette_ptr.
    /// Voxel id is defined as block id + offset inside block.
    material_ptr: u64,

    /// Pointer to a list of 256 u8 colors
    palette_ptr: u64,

    /// number of boxes of entries, each entry has 6 faces.
    irradiance_cache: u64,
}

impl dust_render::Material for DiffuseMaterial {
    type Pipeline = StandardPipeline;

    const TYPE: MaterialType = MaterialType::Procedural;

    fn rahit_shader(_ray_type: u32, _asset_server: &AssetServer) -> Option<SpecializedShader> {
        None
    }

    fn rchit_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader> {
        match ray_type {
            Self::Pipeline::PRIMARY_RAYTYPE => Some(SpecializedShader::for_shader(
                asset_server.load("hit.rchit"),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )),
            Self::Pipeline::PHOTON_RAYTYPE => Some(SpecializedShader::for_shader(
                asset_server.load("photon.rchit"),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )),
            Self::Pipeline::FINAL_GATHER_RAYTYPE => Some(SpecializedShader::for_shader(
                asset_server.load("final_gather.rchit"),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )),
            _ => None,
        }
    }

    fn intersection_shader(
        _ray_type: u32,
        asset_server: &AssetServer,
    ) -> Option<SpecializedShader> {
        Some(SpecializedShader::for_shader(
            asset_server.load("hit.rint"),
            vk::ShaderStageFlags::INTERSECTION_KHR,
        ))
    }

    type ShaderParameters = DiffuseMaterialShaderParams;
    type ShaderParameterParams = (
        SRes<Assets<VoxGeometry>>,
        SRes<Assets<VoxPalette>>,
        SRes<Assets<PaletteMaterial>>,
    );
    fn parameters(
        &self,
        _ray_type: u32,
        params: &mut SystemParamItem<Self::ShaderParameterParams>,
    ) -> Self::ShaderParameters {
        let (geometry_store, palette_store, material_store) = params;
        let material = material_store.get(&self.material).unwrap();
        let geometry = geometry_store.get(&material.geometry).unwrap();
        let palette = palette_store.get(&material.palette).unwrap();
        DiffuseMaterialShaderParams {
            geometry_ptr: geometry.geometry_buffer().device_address(),
            material_ptr: material.data.device_address(),
            palette_ptr: palette.buffer.device_address(),
            irradiance_cache: self.irradiance_cache.device_address(),
        }
    }
}
