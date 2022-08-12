use std::sync::Arc;

use ash::vk;
use bevy_ecs::system::lifetimeless::SRes;
use dot_vox::Color;
use dust_render::render_asset::{GPURenderAsset, GPURenderAssetBuildResult, RenderAsset};
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer, MemoryAllocScenario};

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "c7713cf2-527f-45ac-8eed-cbbcdc7302fd"]
pub struct VoxPalette(pub Box<[dot_vox::Color; 255]>);

pub struct VoxPaletteGPU {
    pub(crate) palette: Arc<MemBuffer>,
}

impl RenderAsset for VoxPalette {
    type GPUAsset = VoxPaletteGPU;

    type BuildData = Arc<MemBuffer>;

    type CreateBuildDataParam = SRes<dust_render::Allocator>;

    fn create_build_data(
        &mut self,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        let layout = std::alloc::Layout::new::<[Color; 256]>();
        let mut buf = allocator
            .allocate_buffer(&BufferRequest {
                size: layout.size() as u64,
                alignment: layout.align() as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                scenario: MemoryAllocScenario::StagingBuffer,
                ..Default::default()
            })
            .unwrap();
        buf.map_scoped(|slice| unsafe {
            let palette_data = std::slice::from_raw_parts(
                self.0.as_ptr() as *const u8,
                std::mem::size_of::<[Color; 256]>(),
            );
            slice.copy_from_slice(palette_data);
        });
        Arc::new(buf)
    }
}

impl GPURenderAsset<VoxPalette> for VoxPaletteGPU {
    type BuildParam = ();

    fn build(
        build_set: Arc<MemBuffer>,
        commands_future: &mut dustash::sync::CommandsFuture,
        _params: &mut bevy_ecs::system::SystemParamItem<Self::BuildParam>,
    ) -> GPURenderAssetBuildResult<VoxPalette> {
        println!("Build asset for vox palette GPU");
        let palette = build_set.make_device_local(commands_future);
        GPURenderAssetBuildResult::Success(Self { palette })
    }
}
