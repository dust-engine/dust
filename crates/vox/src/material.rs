use bevy::{asset::Asset, math::UVec3, reflect::TypePath};
use dust_vdb::{AttributeAllocator, Node};
use pumicite::{
    Allocator,
    ash::vk,
    buffer::{BufferLike, ManagedBuffer},
};

#[derive(Debug, Clone, Copy)]
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

impl dust_vdb::AttributePtr<u32> for VoxLeafNode {
    fn attribute_ptr(&self) -> u32 {
        self.material_ptr
    }

    fn set_attribute_ptr(&mut self, ptr: u32) {
        self.material_ptr = ptr;
    }
}

impl dust_vdb::AttributePtr<VoxLeafNode> for VoxLeafNode {
    fn attribute_ptr(&self) -> VoxLeafNode {
        self.clone()
    }

    fn set_attribute_ptr(&mut self, ptr: VoxLeafNode) {
        *self = ptr;
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

    fn free_attributes(&mut self, _index: u32, ptr: &Self::Ptr, num_attributes: u32) {
        self.attribute_allocator
            .free(ptr.material_ptr, num_attributes);
    }

    fn get_attribute(
        &self,
        _index: u32,
        ptr: &Self::Ptr,
        fitted_offset: u32,
        _inflated_offset: u32,
    ) -> Self::Value {
        let slice: &[Self::Value] = bytemuck::cast_slice(self.buffer.as_slice());
        slice[ptr.material_ptr as usize + fitted_offset as usize]
    }

    fn copy_attribute(
        &mut self,
        original_leaf: u32,
        new_leaf: u32,
        ptr: &Self::Ptr,
        original_mask: &[usize],
        new_mask: &[usize],
        coords: &UVec3,
    ) -> Self::Ptr {
        let new_len = dust_vdb::mask_count_ones(new_mask);
        let new_ptr = self.attribute_allocator.allocate(new_len);
        self.reserve(new_ptr as u64 + new_len as u64);

        let mut new_ptr_cur = new_ptr;
        let mut old_ptr_cur = ptr.material_ptr;

        let slice: &mut [Self::Value] = bytemuck::cast_slice_mut(self.buffer.as_slice_mut());
        for (_, in_original, in_new) in dust_vdb::iter_mask_union(original_mask, new_mask) {
            if in_new && in_original {
                // copy it over
                slice[new_ptr_cur as usize] = slice[old_ptr_cur as usize];
            }
            if in_new {
                new_ptr_cur += 1;
            }
            if in_original {
                old_ptr_cur += 1;
            }
        }
        if original_leaf == new_leaf {
            // In-place re-home: the original range is dead (contract).
            let old_len = dust_vdb::mask_count_ones(original_mask);
            if old_len > 0 {
                self.attribute_allocator.free(ptr.material_ptr, old_len);
            }
        }

        let block_coords: UVec3 = coords >> 2;
        let packed_coords = (block_coords.x << 20) | (block_coords.y << 10) | (block_coords.z);
        VoxLeafNode {
            coords: packed_coords,
            material_ptr: new_ptr,
        }
    }

    fn set_attribute(
        &mut self,
        _index: u32,
        ptr: &Self::Ptr,
        fitted_offset: u32,
        _inflated_offset: u32,
        value: Self::Value,
    ) {
        let slice: &mut [Self::Value] = bytemuck::cast_slice_mut(self.buffer.as_slice_mut());
        slice[ptr.material_ptr as usize + fitted_offset as usize] = value - 1;
    }
}
