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
    /// 10 bits for X, Y, Z.
    coords: u32,
    material_ptr: u32,
}
impl Default for VoxLeafNode {
    fn default() -> Self {
        Self {
            coords: u32::MAX,
            material_ptr: u32::MAX,
        }
    }
}

#[derive(Asset, TypePath)]
pub struct VoxMaterial {
    pub attribute_allocator: AttributeAllocator,
    pub buffer: ManagedBuffer, // Wait so this actually do need to be a managedbuffer.
}
impl VoxMaterial {
    pub fn new(allocator: Allocator) -> Self {
        let buffer = ManagedBuffer::new(
            allocator,
            16 * 1024, // 16 KB to start,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
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
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .unwrap();
            new_buffer.as_slice_mut()[0..self.buffer.size() as usize]
                .copy_from_slice(self.buffer.as_slice());
            self.buffer = new_buffer;
        }
    }
}

impl dust_vdb::Attributes for VoxMaterial {
    /// 0 .. 255 for the offset into the palette.
    type Value = u8;
    type Ptr = VoxLeafNode;
    type Occupancy = BitArr!(for 64);

    const MAX_OCCUPANCY: Self::Occupancy = BitArray {
        data: [usize::MAX; 1],
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

        let block_coords: UVec3 = coords >> 2;
        let packed_coords = (block_coords.x << 20) | (block_coords.y << 10) | (block_coords.z);
        VoxLeafNode {
            coords: packed_coords,
            material_ptr: new_ptr,
        }
    }

    fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value) {
        let slice: &mut [Self::Value] = bytemuck::cast_slice_mut(self.buffer.as_slice_mut());
        slice[ptr.material_ptr as usize + offset as usize] = value - 1;
    }
}
