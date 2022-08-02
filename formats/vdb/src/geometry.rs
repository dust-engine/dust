use std::{mem::size_of, sync::Arc};

use bevy_asset::AssetServer;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, RenderAsset},
    vk,
};

use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy_reflect::TypeUuid;
use dustash::{
    resources::alloc::{Allocator, BufferRequest, MemBuffer},
    sync::CommandsFuture,
};
use glam::{Vec3, Vec3A};

use crate::{size_of_grid, tree::TreeMeta, IsLeaf, LeafNode, Node, NodeConst, Tree};

pub struct VdbGeometry<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: Tree<ROOT>,
    pub unit_size: Vec3,
}

impl<ROOT: Node> VdbGeometry<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    pub fn new(tree: Tree<ROOT>, unit_size: Vec3) -> Self {
        Self { tree, unit_size }
    }
}

impl<ROOT: ~const NodeConst> const TypeUuid for VdbGeometry<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    const TYPE_UUID: bevy_reflect::Uuid =
        bevy_reflect::Uuid::from_u64_pair(0x3c2e398cf93e490d, Tree::<ROOT>::ID);
}
impl<ROOT: NodeConst + Sync + Send> RenderAsset for VdbGeometry<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    type GPUAsset = GPUVdbGeometry;

    type BuildData = (MemBuffer);

    type CreateBuildDataParam = SRes<Arc<Allocator>>;

    fn create_build_data(
        &mut self,
        allocator: &mut SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        let leaf_extent: Vec3A = <ROOT::LeafType as Node>::EXTENT.as_vec3a();
        let leaf_extent: Vec3A = Vec3A::from(self.unit_size) * leaf_extent;

        let nodes: Vec<GPUVdbNode> = self
            .tree
            .iter_leaf()
            .map(|(position, d)| {
                let position = position.as_vec3a() * leaf_extent;
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
                GPUVdbNode {
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

// size: 8 x u32 = 32 bytes
struct GPUVdbNode {
    aabb: vk::AabbPositionsKHR,
    mask: u64,
}
pub struct GPUVdbGeometry {
    /// Buffer with vk::AabbPositionKHR and u64 mask interleaved.
    buffer: Arc<MemBuffer>,
}
impl<ROOT: NodeConst + Sync + Send> GPURenderAsset<VdbGeometry<ROOT>> for GPUVdbGeometry
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    type BuildParam = SRes<Arc<Allocator>>;

    fn build(
        build_set: <VdbGeometry<ROOT> as RenderAsset>::BuildData,
        commands_future: &mut CommandsFuture,
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

impl<ROOT: NodeConst + Sync + Send> Geometry for VdbGeometry<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        todo!()
    }

    fn intersection_shader(asset_server: &AssetServer) -> dust_render::shader::SpecializedShader {
        let handle = asset_server.load("dda.rint.spv");
        dust_render::shader::SpecializedShader {
            shader: handle,
            specialization: Default::default(),
        }
    }
}
impl<ROOT: NodeConst + Sync + Send> GPUGeometry<VdbGeometry<ROOT>> for GPUVdbGeometry
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    fn blas_input_buffer(&self) -> &Arc<MemBuffer> {
        &self.buffer
    }

    type SbtInfo = u64;

    type GeometryInfoParams = ();

    fn geometry_info(
        &self,
        handle: &bevy_asset::Handle<VdbGeometry<ROOT>>,
        params: &mut SystemParamItem<Self::GeometryInfoParams>,
    ) -> Self::SbtInfo {
        self.buffer.device_address()
    }
}
