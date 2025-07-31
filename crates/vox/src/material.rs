use std::marker::PhantomData;

use bevy::{asset::Asset, math::UVec3, reflect::TypePath};
use bitvec::{BitArr, array::BitArray};
use dust_vdb::{AttributeAllocator, Node};
use rhyolite::{
    Allocator,
    ash::vk,
    buffer::{BufferLike, ManagedBuffer},
};

#[derive(Debug, Clone)]
pub struct VoxLeafNode {
    /// Note: we store this value as a vk::AabbPositionsKHR which is less
    /// efficient than possible. We could get away with storing a u16vec4.
    /// That helps us to get down the overhead to 12 bytes
    /// (8 for aabb, 2 for material_ptr, 2 for reserved) instead of 32 bytes.
    aabb: vk::AabbPositionsKHR,
    material_ptr: u32,
    reserved: u32,
}
impl Default for VoxLeafNode {
    fn default() -> Self {
        Self {
            aabb: vk::AabbPositionsKHR {
                min_x: f32::NAN,
                min_y: f32::NAN,
                min_z: f32::NAN,
                max_x: f32::NAN,
                max_y: f32::NAN,
                max_z: f32::NAN,
            },
            material_ptr: 0,
            reserved: 0,
        }
    }
}

#[derive(Asset, TypePath)]
pub struct VoxMaterial {
    pub attribute_allocator: AttributeAllocator,
    buffer: ManagedBuffer,
}
impl VoxMaterial {
    pub fn new(allocator: Allocator) -> Self {
        let buffer = ManagedBuffer::new(
            allocator,
            16 * 1024, // 16 KB to start,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();
        Self {
            attribute_allocator: AttributeAllocator::new_with_capacity(16, 512),
            buffer,
        }
    }

    fn reserve(&mut self, size: u64) {
        if size > self.buffer.size() {
            let mut new_buffer = ManagedBuffer::new(
                self.buffer.allocator().clone(),
                size.next_power_of_two(),
                4,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            .unwrap();
            new_buffer.as_slice_mut()[0..self.buffer.size() as usize]
                .copy_from_slice(self.buffer.as_slice());
            self.buffer = new_buffer;
        }
    }
}

impl dust_vdb::Attributes for VoxMaterial {
    /// 0 for air, and 1 ..= 255 for the offset into the palette.
    type Value = u8;
    type Ptr = VoxLeafNode;
    type Occupancy = BitArr!(for 512);

    const MAX_OCCUPANCY: Self::Occupancy = BitArray {
        data: [usize::MAX; 8],
        _ord: PhantomData,
    };

    fn free_attributes(&mut self, ptr: &Self::Ptr, num_attributes: u32) {
        self.attribute_allocator
            .free(ptr.material_ptr, num_attributes);
    }

    fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value {
        let slice: &[Self::Value] = bytemuck::cast_slice(self.buffer.as_slice());
        slice[ptr.material_ptr as usize + offset as usize]
    }
    fn get_attributes(&self, ptr: &Self::Ptr, len: u32) -> &[Self::Value] {
        let slice: &[Self::Value] = bytemuck::cast_slice(self.buffer.as_slice());
        &slice[ptr.material_ptr as usize..(ptr.material_ptr as usize + len as usize)]
    }

    fn copy_attribute(
        &mut self,
        ptr: &Self::Ptr,
        original_mask: &Self::Occupancy,
        new_mask: &Self::Occupancy,
        coords: &UVec3,
    ) -> Self::Ptr {
        let new_ptr = self
            .attribute_allocator
            .allocate(new_mask.count_ones() as u32);
        self.reserve(new_ptr as u64 + new_mask.count_ones() as u64);
        let mut new_ptr_cur = new_ptr;
        let mut old_ptr_cur = ptr.material_ptr;

        let slice: &mut [Self::Value] = bytemuck::cast_slice_mut(self.buffer.as_slice_mut());
        for bit in (*original_mask | new_mask).iter_ones() {
            if *new_mask.get(bit).unwrap() && *original_mask.get(bit).unwrap() {
                // copy it over
                slice[new_ptr_cur as usize] = slice[old_ptr_cur as usize];
            }
            if *new_mask.get(bit).unwrap() {
                new_ptr_cur += 1;
            }
            if *original_mask.get(bit).unwrap() {
                old_ptr_cur += 1;
            }
        }

        let leaf_extent = <<crate::TreeRoot as Node>::LeafType as Node>::EXTENT;
        let min = *coords & UVec3::splat(!0b111);
        let max = min + leaf_extent;
        let aabb = vk::AabbPositionsKHR {
            min_x: min.x as f32,
            min_y: min.y as f32,
            min_z: min.z as f32,
            max_x: max.x as f32,
            max_y: max.y as f32,
            max_z: max.z as f32,
        };
        VoxLeafNode {
            aabb,
            material_ptr: new_ptr,
            reserved: 0,
        }
    }

    fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value) {
        let slice: &mut [Self::Value] = bytemuck::cast_slice_mut(self.buffer.as_slice_mut());
        slice[ptr.material_ptr as usize + offset as usize] = value;
    }
}
