use ash::vk;
use bevy_asset::AssetLoader;
use bevy_asset::AssetServer;
use bevy_asset::Handle;
use bevy_ecs::system::lifetimeless::SRes;
use bevy_ecs::system::SystemParamItem;
use dust_render::material::{GPUMaterial, Material};
use dust_render::render_asset::BindlessGPUAsset;
use dust_render::render_asset::BindlessGPUAssetDescriptors;
use dust_render::render_asset::GPURenderAsset;
use dust_render::render_asset::RenderAsset;
use dust_render::shader::SpecializedShader;
use dustash::queue::QueueType;
use dustash::queue::Queues;
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer, MemoryAllocScenario};
use dustash::resources::Image;
use dustash::Device;
use std::sync::Arc;

#[derive(bevy_reflect::TypeUuid, Default)]
#[uuid = "75a9a733-04d7-4acb-8600-9a7d24ff0599"] // TODO: better UUID
pub struct DummyMaterial {}

impl RenderAsset for DummyMaterial {
    type GPUAsset = GPUDummyMaterial;
    type CreateBuildDataParam = SRes<Arc<Allocator>>;
    type BuildData = ();
    fn create_build_data(
        &mut self,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        println!("Create nbuild data");
    }
}

impl Material for DummyMaterial {
    type Geometry = crate::VoxGeometry;

    fn anyhit_shader(asset_server: &AssetServer) -> Option<dust_render::shader::SpecializedShader> {
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

pub struct GPUDummyMaterial {}

impl GPURenderAsset<DummyMaterial> for GPUDummyMaterial {
    type BuildParam = (SRes<Arc<Device>>, SRes<Arc<Allocator>>, SRes<Arc<Queues>>);

    fn build(
        build_set: <DummyMaterial as RenderAsset>::BuildData,
        commands_future: &mut dustash::sync::CommandsFuture,
        params: &mut bevy_ecs::system::SystemParamItem<Self::BuildParam>,
    ) -> Self {
        Self {}
    }
}

impl GPUMaterial<DummyMaterial> for GPUDummyMaterial {
    type SbtData = u32;

    type MaterialInfoParams = (SRes<BindlessGPUAssetDescriptors>);
    fn material_info(
        &self,
        handle: &Handle<DummyMaterial>,
        bindless_store: &mut SystemParamItem<Self::MaterialInfoParams>,
    ) -> Self::SbtData {
        0
    }
}
