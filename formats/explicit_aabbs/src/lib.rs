pub mod loader;
use ash::vk;
use bevy_app::Plugin;
use bevy_asset::{AddAsset, AssetServer, Handle};
use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy_reflect::TypeUuid;
use dust_render::geometry::{GPUGeometry, Geometry};
use dustash::{
    command::recorder::CommandRecorder,
    ray_tracing::sbt::SpecializationInfo,
    resources::alloc::{
        Allocator, BufferRequest, MemBuffer, MemoryPropertyFlags, MemoryUsageFlags,
    },
};
use loader::ExplicitAABBPrimitivesLoader;
use std::sync::Arc;

#[derive(TypeUuid)]
#[uuid = "75a9a733-04d7-4abb-8600-9a7d24ff0597"]
pub struct AABBGeometry {
    primitives: Box<[ash::vk::AabbPositionsKHR]>,
}

pub struct AABBGPUGeometry {
    primitives_buffer: Arc<MemBuffer>,
}

impl Geometry for AABBGeometry {
    type GPUGeometry = AABBGPUGeometry;

    type ChangeSet = ();

    type BuildSet = MemBuffer;

    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        todo!()
    }

    fn intersection_shader(asset_server: &AssetServer) -> Handle<dust_render::shader::Shader> {
        todo!()
    }

    fn specialization() -> SpecializationInfo {
        todo!()
    }

    type GenerateBuildsParam = SRes<Arc<Allocator>>;

    fn generate_builds(
        &mut self,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::GenerateBuildsParam>,
    ) -> Self::BuildSet {
        let size = std::mem::size_of_val(&self.primitives as &[ash::vk::AabbPositionsKHR]);
        let mut buffer = allocator
            .allocate_buffer(BufferRequest {
                size: size as u64,
                alignment: 0,
                // TODO: also make this ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR for integrated GPUs.
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_usage: MemoryUsageFlags::UPLOAD,
                ..Default::default()
            })
            .unwrap();
        buffer.write_bytes(0, unsafe {
            std::slice::from_raw_parts(self.primitives.as_ptr() as *const u8, size)
        });
        buffer
    }

    type EmitChangesParam = ();

    fn emit_changes(
        &mut self,
        param: &mut bevy_ecs::system::SystemParamItem<Self::EmitChangesParam>,
    ) -> Self::ChangeSet {
        todo!()
    }
}

impl GPUGeometry<AABBGeometry> for AABBGPUGeometry {
    type BuildParam = SRes<Arc<Allocator>>;
    fn build(
        build_set: MemBuffer,
        commands_future: &mut dustash::sync::CommandsFuture,
        allocator: &mut SystemParamItem<Self::BuildParam>,
    ) -> Self {
        if build_set
            .memory()
            .props()
            .contains(MemoryPropertyFlags::DEVICE_LOCAL)
        {
            Self {
                primitives_buffer: Arc::new(build_set),
            }
        } else {
            let size = build_set.size();
            let device_local_buffer = allocator
                .allocate_buffer(BufferRequest {
                    size,
                    alignment: build_set.alignment(),
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    memory_usage: MemoryUsageFlags::FAST_DEVICE_ACCESS
                        | MemoryUsageFlags::DEVICE_ADDRESS,
                    ..Default::default()
                })
                .unwrap();
            let device_local_buffer = Arc::new(device_local_buffer);
            commands_future.then_commands(|mut recorder| {
                recorder.copy_buffer(
                    build_set,
                    device_local_buffer.clone(),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    }],
                );
            });
            Self {
                primitives_buffer: device_local_buffer,
            }
        }
    }

    type ApplyChangeParam = ();
    fn apply_change_set(
        &mut self,
        change_set: <AABBGeometry as Geometry>::ChangeSet,
        commands_future: &mut dustash::sync::CommandsFuture,
        params: &mut SystemParamItem<Self::ApplyChangeParam>,
    ) {
        todo!()
    }

    fn blas_input_buffer(&self) -> &Arc<MemBuffer> {
        &self.primitives_buffer
    }
}

#[derive(Default)]
pub struct ExplicitAABBPlugin;
impl Plugin for ExplicitAABBPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(dust_render::geometry::GeometryPlugin::<AABBGeometry>::default())
            .add_asset_loader(ExplicitAABBPrimitivesLoader::default());
    }
}
