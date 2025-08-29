use std::{any::Any, sync::Arc};

use crate::{Tree, VoxLeafNode, VoxModel};
use bevy::{ecs::system::lifetimeless::{SRes, SResMut}, prelude::*};
use dust_vdb::pool::PoolStorage;
use rhyolite::{ash::vk, buffer::{Buffer, ManagedBuffer}, command::CommandEncoder, utils::AsVkHandle, Allocator};
use smallvec::SmallVec;

#[derive(Asset, TypePath)]
pub struct VoxGeometry {
    tree: Tree,
    pub unit_size: f32,
}

pub struct VoxGeometryLeafStorage {
    allocator: Allocator,
    buffer: Option<ManagedBuffer>,
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
    fn resize(&mut self, size: usize) -> *mut u8 {
        let mut new_buffer = ManagedBuffer::new(
            self.allocator.clone(),
            size as u64,
            self.alignment as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS |
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )
        .unwrap();
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
        self.buffer = Some(new_buffer);
        self.size = size;
        ptr
    }
}

impl VoxGeometry {
    pub fn new(tree: Tree, unit_size: f32) -> Self {
        Self { tree, unit_size }
    }
}
pub struct BlasBuilder;


impl rhyolite_bevy::rtx::blas::BLASBuilder for BlasBuilder {
    type QueryData = &'static VoxModel;

    type QueryFilter = ();

    type Params = (
        SRes<Assets<VoxGeometry>>,
        SRes<Allocator>
    );

    type BufferType = Buffer;

    type BufferContainerType = Arc<Buffer>;

    fn geometries<'w, 's, 't, 't2, 'b>(
        (geometries, allocator): &mut bevy::ecs::system::SystemParamItem<'w, 's, Self::Params>,
        model: &VoxModel,
        recorder: &mut CommandEncoder<'b>,
    ) -> impl Future<Output = SmallVec<[rhyolite_bevy::rtx::blas::BLASBuildGeometry<Self::BufferContainerType>; 1]>> + use<'w, 's, 't, 't2, 'b> {
        let geometry = geometries.get(&model.geometry).unwrap();

        let primitive_count = geometry.tree.pools()[0].used_capacity();
        let leaf_storage: &dyn Any = geometry.tree.pools()[0].storage();
        let leaf_storage = leaf_storage.downcast_ref::<VoxGeometryLeafStorage>().unwrap();

        let device_buffer = leaf_storage.buffer.as_ref().map(|x| x.device_buffer().clone());

        async move {
            let Some(device_buffer) = device_buffer else {
                return SmallVec::new();
            };
            [rhyolite_bevy::rtx::blas::BLASBuildGeometry::Aabbs {
                buffer: device_buffer,
                stride: size_of::<VoxLeafNode>() as u64,
                flags: vk::GeometryFlagsKHR::OPAQUE,
                primitive_count
            }].into()
        }
    }
}
