use crate::Tree;
use bevy::prelude::*;
use dust_vdb::pool::PoolStorage;
use rhyolite::{Allocator, ash::vk, buffer::ManagedBuffer};

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
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
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

/*

pub struct BlasBuilder {

}


impl rhyolite_bevy::rtx::blas::BLASBuilder for BlasBuilder {
    type QueryData = &'static VoxModel;

    type QueryFilter = ();

    type Params = (
        SRes<Assets<VoxGeometry>>,
        SResMut<Uploader>
    );

    type BufferType;

    type BufferContainerType;

    fn geometries(
        (geometries, uploader): &mut bevy::ecs::system::SystemParamItem<Self::Params>,
        model: &VoxModel,
    ) -> impl Future<Output = SmallVec<[rhyolite_bevy::rtx::blas::BLASBuildGeometry<Self::BufferContainerType>; 1]>> + use<> {
        let geometry = geometries.get(&model.geometry).unwrap();

        let num_leaves = geometry.count_leaves();


        let leaf_extent_int = <<TreeRoot as Node>::LeafType as Node>::EXTENT;
        let leaf_extent: Vec3A = leaf_extent_int.as_vec3a();
        let leaf_extent: Vec3A = geometry.unit_size * leaf_extent;


        uploader.allocate_buffer((num_leaves * size_of::<vk::AabbPositionsKHR>()) as u64, 4, |buf | {
            assert_eq!(buf.len(), num_leaves * size_of::<vk::AabbPositionsKHR>());
            let buf = unsafe { std::slice::from_raw_parts_mut(buf.as_ptr() as *mut vk::AabbPositionsKHR, num_leaves)};
            for ((position, _), target) in geometry.iter_leaf().zip(buf.iter_mut()) {
                let position = position.as_vec3a();
                let max_position = leaf_extent + position;
                *target = vk::AabbPositionsKHR {
                    min_x: position.x,
                    min_y: position.y,
                    min_z: position.z,
                    max_x: max_position.x,
                    max_y: max_position.y,
                    max_z: max_position.z,
                };
            }
        });
        rhyolite_bevy::rtx::blas::BLASBuildGeometry::Aabbs {
            buffer: (), stride: (), flags: (), primitive_count: ()
        }
    }
}
    */
