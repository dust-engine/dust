use crate::alloc::ArenaAllocator;
use std::marker::PhantomData;

use crate::alloc::{ArenaAllocated, Handle};
use crate::{Corner, Voxel};

pub mod accessor;
// mod io;

#[repr(C)]
pub union NodeInternal<T: Voxel> {
    node: Node<T>,
    extended_occupancy: [u8; 8],
}

impl<T: Voxel> Default for NodeInternal<T> {
    fn default() -> Self {
        NodeInternal { node: Default::default() }
    }
}

impl<T: Voxel> ArenaAllocated for NodeInternal<T> {}

#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
pub struct Node<T: Voxel> {
    _padding: u8,
    occupancy: u8,
    sizemask: u16, // Two bits per child, either 11 or 01 or 00 depending on whether child has eo or not. 
    children: Handle,
    _marker: PhantomData<T>,
}

impl<T: Voxel> Node<T> {
    pub fn child_handle(&self, corner: Corner) -> Handle {
        // Given a mask and a location, returns n where the given '1' on the location
        // is the nth '1' counting from the least significant bit.
        fn mask_location_nth_one(mask: u16, location: u8) -> u8 {
            (mask & ((1 << location) - 1)).count_ones() as u8
        }
        self.children
            .offset(mask_location_nth_one(self.sizemask, 2 * corner as u8) as u32)
    }
}

pub struct Octree<T: Voxel> {
    arena: ArenaAllocator<NodeInternal<T>>,
    pub(crate) root: Handle,
    root_occupancy: bool,
    _root_marker: PhantomData<T>,
}

impl<T: Voxel> Octree<T> {
    pub fn new(mut arena: ArenaAllocator<NodeInternal<T>>) -> Self {
        let root = arena.alloc(2);
        Octree {
            arena,
            root,
            root_occupancy: false,
            _root_marker: Default::default(),
        }
    }
    pub fn reshape(&mut self, node_ref: &mut Node<T>, new_mask: u16) {
        let old_mask = node_ref.sizemask;
        let old_child_handle = node_ref.children;

        if old_mask == 0 && new_mask == 0 {
            return;
        }
        let new_num_items = new_mask.count_ones();
        if old_mask == 0 {
            let new_child_handle = self.arena.alloc(new_num_items);
            node_ref.sizemask = new_mask;
            node_ref.children = new_child_handle;
            return;
        }
        if new_mask == 0 {
            node_ref.sizemask = 0;
            unsafe {
                self.arena
                    .free(old_child_handle, old_mask.count_ones() as u8);
            }
            return;
        }
        let new_child_handle = self.arena.alloc(new_num_items);
        let mut new_slot_num: u8 = 0;
        let mut old_slot_num: u8 = 0;
        // println!("Reshaping: Old {:#b}, New {:#b}", old_mask, new_mask);
        for i in 0..16 {
            let old_have_children_at_i = old_mask & (1 << i) != 0;
            let new_have_children_at_i = new_mask & (1 << i) != 0;
            if old_have_children_at_i && new_have_children_at_i {
                unsafe {
                    std::ptr::copy(
                        self.arena.get(old_child_handle.offset(old_slot_num as u32)),
                        self.arena
                            .get_mut(new_child_handle.offset(new_slot_num as u32)),
                        1,
                    );
                }
            }
            if old_have_children_at_i {
                old_slot_num += 1;
            }
            if new_have_children_at_i {
                new_slot_num += 1;
            }
        }
        self.arena.changed_block(new_child_handle, new_num_items);
        node_ref.sizemask = new_mask;
        node_ref.children = new_child_handle;
        unsafe {
            self.arena
                .free(old_child_handle, old_mask.count_ones() as u8);
        }
    }
    pub fn flush(&mut self) {
        self.arena.flush();
    }
}

#[cfg(untested)]
mod tests {
    use super::*;

    #[test]
    fn test_set() {
        let block_allocator = crate::alloc::SystemBlockAllocator::new();
        let arena: ArenaAllocator<Node<u16>> = ArenaAllocator::new(Box::new(block_allocator));
        let mut octree: Octree<u16> = Octree::new(arena);
        for (i, corner) in Corner::all().enumerate() {
            let (x, y, z) = corner.position_offset();
            octree.set(x as u32, y as u32, z as u32, 8, 3);
            assert_eq!(octree.get(x as u32, y as u32, z as u32, 8), 3);
            if i < 4 {
                assert_eq!(octree.get(0, 0, 0, 4), 0);
            } else if i > 4 {
                assert_eq!(octree.get(0, 0, 0, 4), 3);
            }
            if i < 7 {
                assert_eq!(octree.arena.size, 3);
                assert_eq!(octree.arena.num_blocks, 3);
            } else {
                assert_eq!(octree.arena.size, 2);
                assert_eq!(octree.arena.num_blocks, 2);
            }
        }
        for (i, corner) in Corner::all().enumerate() {
            let (x, y, z) = corner.position_offset();
            octree.set(2 + x as u32, y as u32, z as u32, 8, 5);
            assert_eq!(octree.get(2 + x as u32, y as u32, z as u32, 8), 5);
            if i < 4 {
                assert_eq!(octree.get(1, 0, 0, 4), 0);
            } else if i > 4 {
                assert_eq!(octree.get(1, 0, 0, 4), 5);
            }
            if i < 7 {
                assert_eq!(octree.arena.size, 3);
                assert_eq!(octree.arena.num_blocks, 3);
            } else {
                assert_eq!(octree.arena.size, 2);
                assert_eq!(octree.arena.num_blocks, 2);
            }
        }
    }
}
