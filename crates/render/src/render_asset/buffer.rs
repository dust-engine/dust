use std::sync::Arc;

use ash::vk;
use bevy_asset::AssetLoader;
use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy_reflect::TypeUuid;
use dustash::resources::alloc::{MemBuffer, BufferRequest};

use super::{RenderAsset, GPURenderAsset};

// A buffer of generic data.


#[derive(TypeUuid)]
#[uuid = "26db7a93-d0cf-441a-ba6e-376f46571627"]
pub struct RawBuffer {
    buf: Box<[u8]>
}

impl RenderAsset for RawBuffer {
    type GPUAsset = GPURawBuffer;

    /// Data needed to send this asset to the GPU.
    /// This is usually a GPU Resource such as a staging MemBuffer,
    /// or in the case of an integrated GPU, a DEVICE_VISIBLE MemBuffer.
    type BuildData = Arc<MemBuffer>;

    type CreateBuildDataParam = SRes<crate::Allocator>;

    /// Create build data by either copying data into the staging buffer
    /// or moving out the staging buffer.
    /// This is executed right after the asset was created.
    /// A mutable self reference is passed in so that the implementation
    /// can choose to delete the original buffer or moving out from self,
    /// if the data is supposed to be consumed by the GPU only.
    fn create_build_data(
        &mut self,
        allocator: &mut SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        println!("Create build data");
        let mut buf = allocator.allocate_buffer(&BufferRequest{
            size: self.buf.len() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            scenario: dustash::resources::alloc::MemoryAllocScenario::StagingBuffer,
            ..Default::default()
        }).unwrap();
        buf.map_scoped(|target| {
            target.copy_from_slice(&self.buf);
        });
        Arc::new(buf)
    }
}


pub struct GPURawBuffer {
    buf: Arc<MemBuffer>,
}
impl std::ops::Deref for GPURawBuffer {
    type Target = Arc<MemBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl GPURenderAsset<RawBuffer> for GPURawBuffer {
    type BuildParam = SRes<crate::Allocator>;
    fn build(
        build_set: Arc<MemBuffer>,
        commands_future: &mut dustash::sync::CommandsFuture,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::BuildParam>,
    ) -> super::GPURenderAssetBuildResult<RawBuffer> {
        println!("Building GPU Raw Bufer");
        if build_set.device_local() {
            super::GPURenderAssetBuildResult::Success(GPURawBuffer {
                buf: build_set
            })
        } else {
            let buf = allocator.allocate_buffer(&BufferRequest{
                size: build_set.size(),
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                scenario: dustash::resources::alloc::MemoryAllocScenario::DeviceAccess,
                ..Default::default()
            }).unwrap();
            let buf = Arc::new(buf);
            commands_future.then_commands(|mut recorder| {
                recorder.copy_buffer(build_set, buf.clone(), &[
                    vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: buf.size(),
                    }
                ]);
            });
            super::GPURenderAssetBuildResult::Success(GPURawBuffer {
                buf
            })
        }
    }
}


#[derive(Default)]
pub struct RawBufferLoader;
impl AssetLoader for RawBufferLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut bevy_asset::LoadContext,
    ) -> bevy_asset::BoxedFuture<'a, anyhow::Result<(), anyhow::Error>> {
        Box::pin(async {
            let bytes: Vec<u8> = bytes.to_vec();
            load_context.set_default_asset(bevy_asset::LoadedAsset::new(RawBuffer {
                buf: bytes.into_boxed_slice()
            }));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["bin"]
    }
}
