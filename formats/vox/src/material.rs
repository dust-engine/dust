use ash::vk;
use bevy_asset::AssetLoader;
use bevy_asset::AssetServer;
use bevy_asset::Handle;
use bevy_ecs::system::lifetimeless::SRes;
use bevy_ecs::system::SystemParamItem;
use dust_render::material::{GPUMaterial, Material};
use dust_render::render_asset::BindlessGPUAssetDescriptors;
use dust_render::render_asset::GPURenderAsset;
use dust_render::render_asset::GPURenderAssetBuildResult;
use dust_render::render_asset::RenderAsset;
use dust_render::render_asset::RenderAssetStore;
use dust_render::shader::SpecializedShader;
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer, MemoryAllocScenario};
use std::sync::Arc;

use crate::VoxPalette;

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "75a9a733-04d7-4acb-8600-9a7d24ff0599"] // TODO: better UUID
pub struct PaletteMaterial {
    palette: Handle<VoxPalette>,
    data: Vec<u8>,
}

impl PaletteMaterial {
    pub fn new(palette: Handle<VoxPalette>, data: Vec<u8>) -> PaletteMaterial {
        Self { palette, data }
    }
}

impl RenderAsset for PaletteMaterial {
    type GPUAsset = GPUPaletteMaterial;
    type CreateBuildDataParam = SRes<dust_render::Allocator>;
    type BuildData = (Handle<VoxPalette>, Arc<MemBuffer>);
    fn create_build_data(
        &mut self,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        let mut staging_buffer = allocator
            .allocate_buffer(&BufferRequest {
                size: self.data.len() as u64,
                alignment: 4,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                scenario: MemoryAllocScenario::StagingBuffer,
                ..Default::default()
            })
            .unwrap();
        staging_buffer.map_scoped(|slice| {
            slice.copy_from_slice(self.data.as_slice());
        });
        (self.palette.clone(), Arc::new(staging_buffer)) // TODO: upload data here instead.
    }
}

impl Material for PaletteMaterial {
    type Geometry = crate::VoxGeometry;

    fn anyhit_shader(
        _asset_server: &AssetServer,
    ) -> Option<dust_render::shader::SpecializedShader> {
        None
    }

    fn closest_hit_shader(
        asset_server: &AssetServer,
    ) -> Option<dust_render::shader::SpecializedShader> {
        Some(SpecializedShader {
            shader: asset_server.load("plain.rchit.spv"),
            specialization: None,
        })
    }
}

pub struct GPUPaletteMaterial {
    data: Arc<MemBuffer>,
    palette_data: Arc<MemBuffer>,
}

impl GPURenderAsset<PaletteMaterial> for GPUPaletteMaterial {
    type BuildParam = SRes<RenderAssetStore<VoxPalette>>;

    fn build(
        (palette_handle, material_buffer): <PaletteMaterial as RenderAsset>::BuildData,
        commands_future: &mut dustash::sync::CommandsFuture,
        render_asset_store: &mut bevy_ecs::system::SystemParamItem<Self::BuildParam>,
    ) -> GPURenderAssetBuildResult<PaletteMaterial> {
        // maybe get the future of that different thing, and chain it here?
        if let Some(palette) = render_asset_store.get(&palette_handle) {
            println!("Build GPUPaletteMaterial");
            GPURenderAssetBuildResult::Success(Self {
                data: material_buffer.make_device_local(commands_future),
                palette_data: palette.palette.clone(),
            })
        } else {
            // defer
            println!("Deferred GPUPaletteMaterial");
            GPURenderAssetBuildResult::MissingDependency((palette_handle, material_buffer))
        }
    }
}

impl GPUMaterial<PaletteMaterial> for GPUPaletteMaterial {
    type SbtData = (u64, u64); // material_address, palette_address

    type MaterialInfoParams = SRes<BindlessGPUAssetDescriptors>;
    fn material_info(
        &self,
        _handle: &Handle<PaletteMaterial>,
        _bindless_store: &mut SystemParamItem<Self::MaterialInfoParams>,
    ) -> Self::SbtData {
        let material_address = self.data.device_address();
        let palette_address = self.palette_data.device_address();
        (material_address, palette_address)
    }
}

// questions about the asset system:
// - barrier between render world and main world
// - asset dependency
// - precompiling
// - state transfers
