use bevy::{asset::Asset, math::UVec3, reflect::TypePath};
use dust_vdb::{AttributeAllocator, Node};
use rhyolite::ash::vk;

use crate::VoxLeafNode;


#[derive(Asset, TypePath)]
pub struct VoxMaterial {
    pub attribute_allocator: AttributeAllocator,
}

impl dust_vdb::Attributes for VoxMaterial {
    /// 0 for air, and 1 ..= 255 for the offset into the palette.
    type Value = u8;
    type Ptr = VoxLeafNode;
    type Occupancy = dust_vdb::BitMask<512>;

    fn free_attributes(&mut self, ptr: &Self::Ptr, num_attributes: u32) {
        self.attribute_allocator.free(ptr.material_ptr, num_attributes);
    }

    fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value {
        self.attribute_allocator.buffer()[ptr.material_ptr as usize + offset as usize]
    }
    fn get_attributes(&self, ptr: &Self::Ptr, len: u32) -> &[Self::Value] {
        &self.attribute_allocator.buffer()[ptr.material_ptr as usize..(ptr.material_ptr as usize + len as usize)]
    }

    fn copy_attribute(
        &mut self,
        ptr: &Self::Ptr,
        original_mask: &Self::Occupancy,
        new_mask: &Self::Occupancy,
        coords: &UVec3,
    ) -> Self::Ptr {
        let new_ptr = self.attribute_allocator.allocate(new_mask.count_ones() as u32);
        let mut new_ptr_cur = new_ptr;
        let mut old_ptr_cur = ptr.material_ptr;
        for bit in (original_mask | new_mask).iter_set_bits() {
            if new_mask.get(bit) && original_mask.get(bit) {
                // copy it over
                self.0.buffer_mut()[new_ptr_cur as usize] = self.0.buffer()[old_ptr_cur as usize];
            }
            if new_mask.get(bit) {
                new_ptr_cur += 1;
            }
            if original_mask.get(bit) {
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
        self.0.buffer_mut()[ptr.material_ptr as usize + offset as usize] = value;
    }
}
