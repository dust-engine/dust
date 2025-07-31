use glam::UVec3;

pub trait IsDefault {
    fn is_default(&self) -> bool;
}
impl<T> IsDefault for T
where
    T: Default + Eq,
{
    fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

pub trait Attributes {
    /// The type of the attribute pointer.
    /// The attribute pointers are stored on the vdb leaf nodes, one per node.
    /// This is typically u32.
    type Ptr;
    /// The occupancy mask of the attribute pointer.
    /// If we have 4x4x4 leaf nodes, this would be BitMask<64>.
    /// If we have 8x8x8 leaf nodes, this would be BitMask<512>.
    type Occupancy;
    const MAX_OCCUPANCY: Self::Occupancy;
    /// The type of the attribute values. For a MagicaVoxel grid, this would be a u8 palette index.
    type Value: Default + IsDefault;
    fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value;
    fn get_attributes(&self, ptr: &Self::Ptr, len: u32) -> &[Self::Value];
    fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value);
    fn free_attributes(&mut self, ptr: &Self::Ptr, num_attributes: u32);

    /// Allocate a new attribute range using the new mask. Then, copy the attributes from the attribute range
    /// pointed to by `ptr` to the newly allocated attribute range. Returns the pointer to the new attribute range.
    ///
    /// Only attribute values that are set in both the original mask and the new mask will be copied.
    ///
    /// The original attribute range will not be freed. It is the responsibility of the caller to free the original attribute range.
    ///
    /// Note that the original mask may be zeroed. In this case, `ptr` is meaningless, and the function will allocate
    /// a new attribute range without performing any copy.
    fn copy_attribute(
        &mut self,
        ptr: &Self::Ptr,
        original_mask: &Self::Occupancy,
        new_mask: &Self::Occupancy,
        coords: &UVec3,
    ) -> Self::Ptr; // need a value to represent: what are the ones to delete, and what are the ones to add?
}

/// Virtual buffer designed specifically for allocating attributes.
pub struct AttributeAllocator {
    freelists: Box<[Vec<u32>]>,
    alignment: u32,
    max_allocation: u32,
    head: u32,
    wasted_bytes: u32,
}

impl AttributeAllocator {
    fn freelist_for_size(&mut self, size: u32) -> &mut Vec<u32> {
        let freelist_index = (size - 1) / self.alignment;
        &mut self.freelists[freelist_index as usize]
    }
    pub fn new_with_capacity(alignment: u32, max_allocation: u32) -> Self {
        let num_freelists = max_allocation.div_ceil(alignment);
        Self {
            alignment,
            max_allocation,
            freelists: vec![Vec::new(); num_freelists as usize].into_boxed_slice(),
            head: 0,
            wasted_bytes: 0,
        }
    }
    pub fn allocate(&mut self, size: u32) -> u32 {
        assert!(size <= self.max_allocation);
        let increment = size.next_multiple_of(self.alignment);
        self.wasted_bytes += increment - size;
        if let Some(indice) = self.freelist_for_size(size).pop() {
            return indice;
        }
        let old_head = self.head;
        self.head += increment;
        return old_head;
    }
    pub fn realloc(&mut self, ptr: u32, old_size: u32, new_size: u32) -> u32 {
        let old_increment = old_size.next_multiple_of(self.alignment);
        let new_increment = new_size.next_multiple_of(self.alignment);
        if old_increment == new_increment {
            return ptr;
        }
        self.free(ptr, old_size);
        self.allocate(new_size)
    }
    pub fn free(&mut self, ptr: u32, size: u32) {
        assert!(size <= self.max_allocation);
        self.freelist_for_size(size).push(ptr);
        self.wasted_bytes -= size.next_multiple_of(self.alignment) - size;
    }
}
