use std::{any::Any, sync::Arc};

use crate::{Tree, VoxLeafNode, VoxModel};
use bevy::{ecs::system::lifetimeless::{SRes, SResMut}, prelude::*};
use dust_vdb::pool::PoolStorage;
use rhyolite::{Allocator, HasDevice, ash::vk, buffer::{Buffer, BufferLike, RingBufferSuballocation}, command::CommandEncoder, debug::DebugObject, utils::AsVkHandle};
use rhyolite_bevy::{shader::ComputePipeline, staging::DeviceLocalRingBuffer};
use smallvec::SmallVec;

#[derive(Asset, TypePath)]
pub struct VoxGeometry {
    pub tree: Tree,

    /// Model space size of each voxel
    pub unit_size: f32,
}

pub struct VoxGeometryLeafStorage {
    allocator: Allocator,
    // A host-cached buffer that is preferably device-visible.
    // TODO: make this a managed buffer.
    buffer: Option<Arc<Buffer>>,
    alignment: usize,
    size: usize,
}
impl VoxGeometryLeafStorage {
    pub fn new(allocator: Allocator, alignment: usize) -> Self {
        Self {
            allocator,
            buffer: None,
            alignment,
            size: 0,
        }
    }
}
impl PoolStorage for VoxGeometryLeafStorage {
    fn device_address(&self) -> u64 {
        if let Some(buffer) = self.buffer.as_ref() {
            buffer.device_address()
        } else {
            0
        }
    }
    fn resize(&mut self, size: usize) -> *mut u8 {
        let mut new_buffer = Buffer::new_dynamic(
            self.allocator.clone(),
            size as u64,
            self.alignment as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )
        .unwrap()
        .with_name(c"VoxGeometryLeafStorage");
        unsafe {
            if let Some(old_buffer) = self.buffer.take() {
                std::ptr::copy_nonoverlapping(
                    old_buffer.as_ptr(),
                    new_buffer.as_mut_ptr(),
                    self.size.min(size),
                );
            }
        }

        let ptr = new_buffer.as_mut_ptr();
        assert!(!ptr.is_null());
        self.buffer = Some(Arc::new(new_buffer));
        self.size = size;
        ptr
    }
}

impl VoxGeometry {
    pub fn new(allocator: Allocator, unit_size: f32) -> Self {
        let tree = crate::Tree::new_with_leaf_storage(Box::new(VoxGeometryLeafStorage::new(
            allocator,
            crate::Tree::metas()[0].layout.align(),
        )));

        Self { tree, unit_size }
    }
}

#[derive(Resource)]
pub struct BlasBuilder {
    copy_coords_pipeline: Handle<ComputePipeline>
}
impl FromWorld for BlasBuilder {
    fn from_world(world: &mut World) -> Self {
        BlasBuilder {
            copy_coords_pipeline: world.load_asset("embedded://dust_vox/shaders/blas_builder_copy_coords.comp.pipeline.ron")
        }
    }
}

impl rhyolite_bevy::rtx::blas::BLASBuilder for BlasBuilder {
    type QueryData = &'static VoxModel;

    type QueryFilter = ();

    type Params = (
        SRes<Assets<VoxGeometry>>,
        SResMut<DeviceLocalRingBuffer>,
        SRes<Assets<ComputePipeline>>
    );

    type BufferType = RingBufferSuballocation;

    fn geometries<'w, 's, 't, 't2, 'b, 'bb>(
        &mut self,
        (geometries, device_local_ring_buffer, compute_pipelines): &mut bevy::ecs::system::SystemParamItem<'w, 's, Self::Params>,
        model: &VoxModel,
        recorder: &'bb mut CommandEncoder<'b>,
    ) -> impl Future<Output = SmallVec<[rhyolite_bevy::rtx::blas::BLASBuildGeometry<'b, RingBufferSuballocation>; 1]>> + use<'w, 's, 't, 't2, 'b, 'bb> {
        let copy_coords_pipeline = compute_pipelines.get(&self.copy_coords_pipeline).unwrap().clone();
        let geometry = geometries.get(&model.geometry).unwrap();

        let primitive_count = geometry.tree.pools()[0].used_capacity();
        let leaf_storage: &dyn Any = geometry.tree.pools()[0].storage();
        let leaf_storage = leaf_storage.downcast_ref::<VoxGeometryLeafStorage>().unwrap();

        // TODO: This buffer may not be device-local. Test perf when everything was pre-copied to device.
        let device_buffer = leaf_storage.buffer.as_ref().map(Arc::clone);
        let coords_buffer = if device_buffer.is_some() {
            
            // TODO: query the alignment using minStorageBufferOffsetAlignment.
            Some(device_local_ring_buffer.allocate_buffer(primitive_count as u64 * size_of::<vk::AabbPositionsKHR>() as u64, 16))
        } else {
            None
        };

        // Calculate the size of each leaf node AABB primitive.
        let unit_size = geometry.unit_size * 4.0;

        async move {
            let Some(device_buffer) = device_buffer else {
                return SmallVec::new();
            };
            let device_buffer = recorder.retain(device_buffer);
            let coords_buffer = recorder.retain(Box::new(coords_buffer.unwrap()));
            let copy_coords_pipeline = recorder.retain(copy_coords_pipeline.into_inner());
            recorder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, copy_coords_pipeline);
            recorder.push_descriptor_set(vk::PipelineBindPoint::COMPUTE, copy_coords_pipeline.layout(), 0, &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    ..Default::default()
                }
                .buffer_info(&[vk::DescriptorBufferInfo {
                    buffer: device_buffer.vk_handle(),
                    offset: device_buffer.offset(),
                    range: device_buffer.size(),
                },vk::DescriptorBufferInfo {
                    buffer: coords_buffer.vk_handle(),
                    offset: coords_buffer.offset(),
                    range: coords_buffer.size(),
                }])
            ]);
            recorder.push_constants(copy_coords_pipeline.layout(), vk::ShaderStageFlags::COMPUTE, 0, unsafe {
                std::slice::from_raw_parts(&unit_size as *const f32 as *const u8, std::mem::size_of_val(&unit_size))
            });
            recorder.dispatch(UVec3 { x: primitive_count.div_ceil(32), y: 1, z: 1 });
            [rhyolite_bevy::rtx::blas::BLASBuildGeometry::Aabbs {
                buffer: coords_buffer,
                stride: size_of::<vk::AabbPositionsKHR>() as u64,
                flags: vk::GeometryFlagsKHR::OPAQUE,
                primitive_count
            }].into()
        }
    }
}
