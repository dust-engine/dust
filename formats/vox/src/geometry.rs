use std::sync::Arc;

use crate::vox_loader::*;
use ash::vk;

use bevy_ecs::{system::lifetimeless::SRes, world::World};
use bevy_ecs::system::SystemParamItem;
use dust_render::{
    geometry::{GPUGeometry, Geometry},
    render_asset::{GPURenderAsset, GPURenderAssetBuildResult, RenderAsset},
};
use dust_vdb::{IsLeaf, Node};
use dustash::resources::alloc::{Allocator, BufferRequest, MemBuffer};
use glam::{UVec3, Vec3A};

// size: 8 x u32 = 32 bytes
#[repr(C)]
struct GPUVoxNode {
    x: u16,
    y: u16,
    z: u16,
    w: u16,
    mask: u64,
    material_ptr: u32,
    reserved: u32,
}

#[derive(bevy_reflect::TypeUuid)]
#[uuid = "4b169454-eb57-446f-adc3-b7c409f60f5b"]
/// Wrapper for VoxGeometryInner. Without the wrapper, the linking fails due to a rustc bug
pub struct VoxGeometry {
    tree: Tree,
    size: [u8; 3],
    pub unit_size: f32,
}

impl VoxGeometry {
    pub fn from_tree(tree: Tree, size: [u8; 3], unit_size: f32) -> Self {
        Self {
            tree,
            unit_size,
            size,
        }
    }
    pub fn new(unit_size: f32) -> Self {
        Self {
            tree: Tree::new(),
            size: [255; 3],
            unit_size,
        }
    }
    pub fn set(&mut self, coords: UVec3, value: Option<bool>) {
        self.tree.set_value(coords, value)
    }
    pub fn get(&mut self, coords: UVec3) -> Option<bool> {
        self.tree.get_value(coords)
    }
}

impl RenderAsset for VoxGeometry {
    type GPUAsset = VoxGPUGeometry;

    /// AABBBuffer, GeometryBuffer
    type BuildData = (MemBuffer, MemBuffer);

    type CreateBuildDataParam = SRes<dust_render::Allocator>;

    fn create_build_data(
        &mut self,
        allocator: &mut SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData {
        let leaf_extent_int = <<TreeRoot as Node>::LeafType as Node>::EXTENT;
        let leaf_extent: Vec3A = leaf_extent_int.as_vec3a();
        let leaf_extent: Vec3A = self.unit_size * leaf_extent;

        let (aabbs, nodes): (Vec<vk::AabbPositionsKHR>, Vec<GPUVoxNode>) = self
            .tree
            .iter_leaf()
            .map(|(position, d)| {
                let aabb = {
                    let position = position.as_vec3a();
                    let max_position = leaf_extent + position;
                    vk::AabbPositionsKHR {
                        min_x: position.x,
                        min_y: position.y,
                        min_z: position.z,
                        max_x: max_position.x,
                        max_y: max_position.y,
                        max_z: max_position.z,
                    }
                };
                let mut mask = [0_u64; 1];
                d.get_occupancy(&mut mask);
                let mask = mask[0];
                let node = {
                    GPUVoxNode {
                        x: position.x as u16,
                        y: position.y as u16,
                        z: position.z as u16,
                        w: 0,
                        mask,
                        material_ptr: d.material_ptr,
                        reserved: 0,
                    }
                };
                (aabb, node)
            })
            .unzip();
        let aabb_buffer = {
            let size = std::mem::size_of_val(aabbs.as_slice());
            assert_eq!(size, aabbs.len() * 24);
            let mut buffer = allocator
                .allocate_buffer(&BufferRequest {
                    size: size as u64,
                    alignment: 0,
                    // TODO: also make this ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR for integrated GPUs.
                    usage: dust_render::vk::BufferUsageFlags::TRANSFER_SRC
                        | dust_render::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    scenario: dustash::resources::alloc::MemoryAllocScenario::StagingBuffer,
                    ..Default::default()
                })
                .unwrap();

            let data = unsafe { std::slice::from_raw_parts(aabbs.as_ptr() as *const u8, size) };
            buffer.map_scoped(|slice| {
                slice.copy_from_slice(data);
            });
            buffer
        };
        let geometry_buffer = {
            let size = std::mem::size_of_val(nodes.as_slice());
            assert_eq!(size, nodes.len() * 24);
            let mut buffer = allocator
                .allocate_buffer(&BufferRequest {
                    size: size as u64,
                    alignment: 0,
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
        };
        (aabb_buffer, geometry_buffer)
    }
}

impl Geometry for VoxGeometry {
    fn aabb(&self) -> dust_render::geometry::GeometryAABB {
        unimplemented!()
    }

    fn intersection_shader(
        world: &World,
        asset_server: &bevy_asset::AssetServer,
    ) -> dust_render::shader::SpecializedShaderHandle {
        let device = world.resource::<dust_render::Device>();
        let vendor_id = device.physical_device().properties().vendor_id;

        // Usually one would expect the `hitAttributeEXT` value in the closest hit shader to be the one set
        // during the intersection shader invocation of the closest hit point. This is the case on NVIDIA GPUs.
        // However, on my AMD GPU (6600XT), the `hitAttributeEXT` value seems to reflect the value set during
        // the most recent intersection shader invocation instead.
        // Meaning that if additional intersection tests were performed after the closest hit, wrong values
        // would be reflected in the closest hit shader.
        // Here we manually make the comparison when setting the closest hit shader on AMD GPUs.
        let auto_closest_hit_attributes = vendor_id != 4098;

        let handle = asset_server.load("dda.rint.spv");
        dust_render::shader::SpecializedShaderHandle::new(handle)
            .with_bool(0, auto_closest_hit_attributes) // AUTO_CLOSEST_HIT_ATTRIBUTES
    }
}

pub struct VoxGPUGeometry {
    /// Buffer with vk::AabbPositionKHR and u64 mask interleaved.
    aabb_buffer: Arc<MemBuffer>,
    geometry_buffer: Arc<MemBuffer>,
}
impl GPURenderAsset<VoxGeometry> for VoxGPUGeometry {
    type BuildParam = SRes<dust_render::Allocator>;

    fn build(
        build_set: <VoxGeometry as RenderAsset>::BuildData,
        commands_future: &mut dustash::sync::CommandsFuture,
        allocator: &mut SystemParamItem<Self::BuildParam>,
    ) -> GPURenderAssetBuildResult<VoxGeometry> {
        let (aabb_buffer, geometry_buffer) = build_set;
        if geometry_buffer
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            println!("Using device local");
            GPURenderAssetBuildResult::Success(Self {
                aabb_buffer: Arc::new(aabb_buffer),
                geometry_buffer: Arc::new(geometry_buffer),
            })
        } else {
            let device_local_aabb_buffer = allocator
                .allocate_buffer(&BufferRequest {
                    size: aabb_buffer.size(),
                    alignment: aabb_buffer.alignment(),
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                })
                .unwrap();
            let device_local_aabb_buffer = Arc::new(device_local_aabb_buffer);
            let device_local_geometry_buffer = allocator
                .allocate_buffer(&BufferRequest {
                    size: geometry_buffer.size(),
                    alignment: geometry_buffer.alignment(),
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                })
                .unwrap();
            let device_local_geometry_buffer = Arc::new(device_local_geometry_buffer);
            commands_future.then_commands(|mut recorder| {
                let size = geometry_buffer.size();
                recorder.copy_buffer(
                    geometry_buffer,
                    device_local_geometry_buffer.clone(),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    }],
                );
                let size = aabb_buffer.size();
                recorder.copy_buffer(
                    aabb_buffer,
                    device_local_aabb_buffer.clone(),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    }],
                );
            });
            GPURenderAssetBuildResult::Success(Self {
                aabb_buffer: device_local_aabb_buffer,
                geometry_buffer: device_local_geometry_buffer,
            })
        }
    }
}

impl GPUGeometry<VoxGeometry> for VoxGPUGeometry {
    fn blas_input_buffer(&self) -> &Arc<MemBuffer> {
        &self.aabb_buffer
    }

    type SbtInfo = u64;

    type GeometryInfoParams = ();

    fn geometry_info(
        &self,
        _handle: &bevy_asset::Handle<VoxGeometry>,
        _params: &mut bevy_ecs::system::SystemParamItem<Self::GeometryInfoParams>,
    ) -> Self::SbtInfo {
        self.geometry_buffer.device_address()
    }
}
