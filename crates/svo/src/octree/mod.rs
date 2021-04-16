use crate::alloc::ArenaAllocator;

use crate::alloc::{ArenaAllocated, Handle};
use crate::{Corner, Voxel};
use std::fmt::{Debug, Formatter, Write};

pub mod accessor;
mod io;
pub mod supertree;

#[derive(Default, Clone)]
#[repr(C)]
pub struct Node<T: Voxel> {
    _reserved: u8,
    freemask: u8,
    _reserved2: u16,
    children: Handle,
    data: [T; 8],
}

impl<T: Voxel> Debug for Node<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for i in 0 .. 8 {
            if self.freemask & (1 << i)  == 0 {
                self.data[i].fmt(f)?;
            } else {
                f.write_fmt(format_args!("\x1b[6;30;42m{:?}\x1b[0m", self.data[i]))?;
            }
            if i != 7 {
                f.write_str(" | ")?;
            }
        }
        Ok(())
    }
}

impl<T: Voxel> ArenaAllocated for Node<T> {}

impl<T: Voxel> Node<T> {
    pub fn child_handle(&self, corner: Corner) -> Handle {
        // Given a mask and a location, returns n where the given '1' on the location
        // is the nth '1' counting from the least significant bit.
        fn mask_location_nth_one(mask: u8, location: u8) -> u8 {
            (mask & ((1 << location) - 1)).count_ones() as u8
        }
        self.children
            .offset(mask_location_nth_one(self.freemask, corner as u8) as u32)
    }
}

pub struct Octree<T: Voxel> {
    arena: ArenaAllocator<Node<T>>,
    pub(crate) root: Handle,
    root_data: T,
}

impl<T: Voxel> Octree<T> {
    pub fn new(mut arena: ArenaAllocator<Node<T>>) -> Self {
        let root = arena.alloc(1);
        Octree {
            arena,
            root,
            root_data: Default::default(),
        }
    }
    pub fn reshape(&mut self, node_handle: Handle, new_mask: u8) {
        let node_ref = self.arena.get(node_handle);
        let old_mask = node_ref.freemask;
        let old_child_handle = node_ref.children;

        if old_mask == 0 && new_mask == 0 {
            return;
        }
        self.arena.changed(node_handle);
        let new_num_items = new_mask.count_ones();
        if old_mask == 0 {
            let new_child_handle = self.arena.alloc(new_num_items);
            let node_ref = self.arena.get_mut(node_handle);
            node_ref.freemask = new_mask;
            node_ref.children = new_child_handle;
            return;
        }
        if new_mask == 0 {
            let node_ref = self.arena.get_mut(node_handle);
            node_ref.freemask = 0;
            unsafe {
                self.arena
                    .free(old_child_handle, old_mask.count_ones() as u8);
            }
            return;
        }
        let new_child_handle = self.arena.alloc(new_num_items);
        let mut new_slot_num: u8 = 0;
        let mut old_slot_num: u8 = 0;
        for i in 0..8 {
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
        let node_ref = self.arena.get_mut(node_handle);
        node_ref.freemask = new_mask;
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

#[cfg(test)]
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
