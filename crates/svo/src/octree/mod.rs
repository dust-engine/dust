use crate::alloc::ArenaAllocator;

use crate::alloc::{ArenaAllocated, Handle};
use crate::{Corner, Voxel};

pub mod accessor;
mod io;
mod sdf;

#[derive(Default)]
#[repr(C)]
pub struct Node<T: Voxel> {
    _reserved: u8,
    freemask: u8,
    _reserved2: u16,
    children: Handle,
    data: [T; 8],
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
