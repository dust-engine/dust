use std::sync::Arc;

use crate::material::{DummyMaterial, GPUDummyMaterial};
use crate::vox_loader::*;
use ash::vk;
use bevy_asset::AddAsset;
use bevy_ecs::system::lifetimeless::SRes;
use bevy_ecs::system::SystemParamItem;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, RenderAsset},
};
use dust_vdb::{IsLeaf, Node};
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer};
use glam::{Vec3, Vec3A};

// size: 8 x u32 = 32 bytes
#[repr(C)]
struct GPUVoxNode {
    aabb: vk::AabbPositionsKHR,
    mask: u64,
}

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "4b169454-eb57-446f-adc3-b7c409f60f5b"]
/// Wrapper for VoxGeometryInner. Without the wrapper, the linking fails due to a rustc bug
pub struct VoxGeometry {
    tree: Tree,
    pub unit_size: Vec3,
}

impl VoxGeometry {
    pub fn new(tree: Tree, unit_size: Vec3) -> Self {
        Self { tree, unit_size }
    }
}

impl RenderAsset for VoxGeometry {
    type GPUAsset = VoxGPUGeometry;

    type BuildData = MemBuffer;

    type CreateBuildDataParam = SRes<Arc<Allocator>>;

    fn create_build_data(
        &mut self,
        allocator: &mut SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        let leaf_extent: Vec3A = <<TreeRoot as Node>::LeafType as Node>::EXTENT.as_vec3a();
        let leaf_extent: Vec3A = Vec3A::from(self.unit_size) * leaf_extent;

        let nodes: Vec<GPUVoxNode> = self
            .tree
            .iter_leaf()
            .map(|(position, d)| {
                let position = position.as_vec3a();
                let max_position = leaf_extent + position;
                let aabb = vk::AabbPositionsKHR {
                    min_x: position.x,
                    min_y: position.y,
                    min_z: position.z,
                    max_x: max_position.x,
                    max_y: max_position.y,
                    max_z: max_position.z,
                };
                let mut mask = [0_u64; 1];
                d.get_occupancy(&mut mask);
                GPUVoxNode {
                    aabb,
                    mask: mask[0],
                }
            })
            .collect();
        let size = std::mem::size_of_val(nodes.as_slice());
        assert_eq!(size, nodes.len() * 32);
        println!("Allocating some buffer for {} elements", nodes.len());
        let mut buffer = allocator
            .allocate_buffer(&BufferRequest {
                size: size as u64,
                alignment: 16,
                // TODO: also make this ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR for integrated GPUs.
                usage: dust_render::vk::BufferUsageFlags::TRANSFER_SRC
                    | dust_render::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                scenario: dustash::resources::alloc::MemoryAllocScenario::StagingBuffer,
                ..Default::default()
            })
            .unwrap();

        let data = unsafe { std::slice::from_raw_parts(nodes.as_ptr() as *const u8, size) };
        buffer.map_scoped(|slice| {
            slice.copy_from_slice(data);
        });
        buffer
    }
}

impl Geometry for VoxGeometry {
    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        unimplemented!()
    }

    fn intersection_shader(
        asset_server: &bevy_asset::AssetServer,
    ) -> dust_render::shader::SpecializedShader {
        let handle = asset_server.load("dda.rint.spv");
        dust_render::shader::SpecializedShader {
            shader: handle,
            specialization: Default::default(),
        }
    }
}

pub struct VoxGPUGeometry {
    /// Buffer with vk::AabbPositionKHR and u64 mask interleaved.
    buffer: Arc<MemBuffer>,
}
impl GPURenderAsset<VoxGeometry> for VoxGPUGeometry {
    type BuildParam = SRes<Arc<Allocator>>;

    fn build(
        build_set: <VoxGeometry as RenderAsset>::BuildData,
        commands_future: &mut dustash::sync::CommandsFuture,
        allocator: &mut SystemParamItem<Self::BuildParam>,
    ) -> Self {
        if build_set
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            println!("Using device local");
            Self {
                buffer: Arc::new(build_set),
            }
        } else {
            let size = build_set.size();
            println!(
                "Alignment is ->>>>>>>>>>>>>>>>>>>>>>>>{}",
                build_set.alignment()
            );
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
                buffer: device_local_buffer,
            }
        }
    }
}

impl GPUGeometry<VoxGeometry> for VoxGPUGeometry {
    fn blas_input_buffer(&self) -> &Arc<MemBuffer> {
        &self.buffer
    }
    fn blas_input_layout() -> std::alloc::Layout {
        std::alloc::Layout::new::<(ash::vk::AabbPositionsKHR, u64)>()
    }

    type SbtInfo = u64;

    type GeometryInfoParams = ();

    fn geometry_info(
        &self,
        handle: &bevy_asset::Handle<VoxGeometry>,
        params: &mut bevy_ecs::system::SystemParamItem<Self::GeometryInfoParams>,
    ) -> Self::SbtInfo {
        self.buffer.device_address()
    }
}
