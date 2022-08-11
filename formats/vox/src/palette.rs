use std::sync::Arc;

use ash::vk;
use bevy_ecs::system::lifetimeless::SRes;
use dust_render::render_asset::{GPURenderAsset, GPURenderAssetBuildResult, RenderAsset};
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer, MemoryAllocScenario};

#[repr(C)]
#[derive(Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "75a9a733-04d8-4acb-8600-9a7d24ff0599"] // TODO: better UUID
pub struct VoxPalette(pub Box<[Color; 255]>); // TODO: 256

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
