pub mod loader;
pub mod material;
use ash::vk;
use bevy_app::Plugin;
use bevy_asset::{AddAsset, AssetServer, Handle};
use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy_reflect::TypeUuid;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, RenderAsset},
    shader::SpecializedShader,
};
use dustash::{
    command::recorder::CommandRecorder,
    resources::alloc::{
        AllocationCreateFlags, Allocator, BufferRequest, MemBuffer, MemoryAllocScenario,
    },
    shader::SpecializationInfo,
    HasDevice,
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
impl RenderAsset for AABBGeometry {
    type GPUAsset = AABBGPUGeometry;

    type BuildData = MemBuffer;

    type CreateBuildDataParam = SRes<Arc<Allocator>>;

    fn create_build_data(
        &mut self,
        allocator: &mut bevy_ecs::system::SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        let size = std::mem::size_of_val(&self.primitives as &[ash::vk::AabbPositionsKHR]);
        let mut buffer = allocator
            .allocate_buffer(&BufferRequest {
                size: size as u64,
                // TODO: also make this ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR for integrated GPUs.
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                scenario: MemoryAllocScenario::StagingBuffer,
                ..Default::default()
            })
            .unwrap();
        let data =
            unsafe { std::slice::from_raw_parts(self.primitives.as_ptr() as *const u8, size) };
        buffer.map_scoped(|slice| {
            slice.copy_from_slice(data);
        });
        buffer
    }
}
impl Geometry for AABBGeometry {
    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        todo!()
    }

    fn intersection_shader(asset_server: &AssetServer) -> SpecializedShader {
        let handle = asset_server.load("dda.rint.spv");
        SpecializedShader {
            shader: handle,
            specialization: Default::default(),
        }
    }
}

impl GPURenderAsset<AABBGeometry> for AABBGPUGeometry {
    type BuildParam = SRes<Arc<Allocator>>;
    fn build(
        build_set: MemBuffer,
        commands_future: &mut dustash::sync::CommandsFuture,
        allocator: &mut SystemParamItem<Self::BuildParam>,
    ) -> Self {
        if build_set
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            Self {
                primitives_buffer: Arc::new(build_set),
            }
        } else {
            let size = build_set.size();
            let device_local_buffer = allocator
                .allocate_buffer(&BufferRequest {
                    size,
                    alignment: build_set.alignment(),
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
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
}

impl GPUGeometry<AABBGeometry> for AABBGPUGeometry {
    fn blas_input_buffer(&self) -> &Arc<MemBuffer> {
        &self.primitives_buffer
    }
    fn geometry_info(&self) -> u64 {
        self.primitives_buffer.device_address()
    }

    type SbtInfo = u64;
}

#[derive(Default)]
pub struct ExplicitAABBPlugin;
impl Plugin for ExplicitAABBPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(dust_render::geometry::GeometryPlugin::<AABBGeometry>::default())
            .add_asset_loader(ExplicitAABBPrimitivesLoader::default())
            .add_asset_loader(material::DensityMaterialLoader::default());
    }
}
