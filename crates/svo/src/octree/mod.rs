use crate::{Corner, Voxel};
use gfx_alloc::ArenaAllocator;
use gfx_alloc::CHUNK_SIZE;
use gfx_alloc::{ArenaAllocated, BlockAllocator, Handle};
use std::mem::size_of;

pub struct Node<T: Voxel> {
    block_size: u8,
    freemask: u8,
    _reserved: u16,
    children: Handle,
    data: [T; 8],
}

unsafe impl<T: Voxel> ArenaAllocated for Node<T> {}

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

pub struct Octree<BA: BlockAllocator<CHUNK_SIZE>, T: Voxel>
where
    [T; CHUNK_SIZE / size_of::<Node<T>>()]: Sized,
{
    arena: ArenaAllocator<BA, Node<T>>,
    root: Handle,
    root_data: T,
}

impl<BA: BlockAllocator<CHUNK_SIZE>, T: Voxel> Octree<BA, T>
where
    [T; CHUNK_SIZE / size_of::<Node<T>>()]: Sized,
{
    pub fn new(mut arena: ArenaAllocator<BA, Node<T>>) -> Self {
        let root = arena.alloc(1);
        Octree {
            arena,
            root,
            root_data: Default::default(),
        }
    }
    pub fn reshape(&mut self, node_handle: Handle, new_mask: u8) {
        let node_ref = &mut self.arena[node_handle];
        let old_mask = node_ref.freemask;
        let old_child_handle = node_ref.children;

        if old_mask == 0 && new_mask == 0 {
            return;
        }
        let new_num_items = new_mask.count_ones();
        if old_mask == 0 {
            let new_child_handle = self.arena.alloc(new_num_items);
            let node_ref = &mut self.arena[node_handle];
            node_ref.freemask = new_mask;
            node_ref.children = new_child_handle;
            return;
        }
        if new_mask == 0 {
            let node_ref = &mut self.arena[node_handle];
            node_ref.freemask = 0;
            self.arena.free(old_child_handle);
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
                        &mut self.arena[old_child_handle.offset(old_slot_num as u32)],
                        &mut self.arena[new_child_handle.offset(new_slot_num as u32)],
                        1,
                    );
                    // TODO: set block size
                }
            }
            if old_have_children_at_i {
                old_slot_num += 1;
            }
            if new_have_children_at_i {
                new_slot_num += 1;
            }
        }
        let node_ref = &mut self.arena[node_handle];
        node_ref.freemask = new_mask;
        node_ref.children = new_child_handle;
        self.arena.free(old_child_handle);
    }
    fn set_internal(
        &mut self,
        handle: Handle,
        mut x: u32,
        mut y: u32,
        mut z: u32,
        mut gridsize: u32,
        item: T,
    ) -> (T, bool) {
        gridsize = gridsize / 2;
        let mut corner: u8 = 0;
        if x >= gridsize {
            corner |= 0b100;
            x -= gridsize;
        }
        if y >= gridsize {
            corner |= 0b010;
            y -= gridsize;
        }
        if z >= gridsize {
            corner |= 0b001;
            z -= gridsize;
        }
        if gridsize <= 1 {
            // is leaf node
            let node_ref = &mut self.arena[handle];
            node_ref.data[corner as usize] = item;
            if node_ref.freemask & (1 << corner) != 0 {
                // has children. Cut them off.
                todo!()
            }
        } else {
            let node_ref = &mut self.arena[handle];
            let freemask = node_ref.freemask;
            if freemask & (1 << corner) == 0 {
                // no children
                self.reshape(handle, freemask | (1 << corner));
                todo!()
            }

            let new_handle = self.arena[handle].child_handle(corner.into());
            let (avg, collapsed) = self.set_internal(new_handle, x, y, z, gridsize, item);

            let node_ref = &mut self.arena[handle];
            let freemask = node_ref.freemask;
            node_ref.data[corner as usize] = avg;
            if collapsed {
                self.reshape(handle, freemask & !(1 << corner));
            }
        }

        let node_ref = &mut self.arena[handle];
        if node_ref.freemask == 0 {
            // node has no children
            if node_ref.data.iter().all(|a| *a == item) {
                // collapse node
                return (item, true);
            }
        }

        return (T::avg(&node_ref.data), false);
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, gridsize: u32, item: T) {
        let (data, _collapsed) = self.set_internal(self.root, x, y, z, gridsize, item);
        self.root_data = data;
    }

    pub fn get(&self, mut x: u32, mut y: u32, mut z: u32, mut gridsize: u32) -> T {
        let mut handle = self.root;
        while gridsize > 2 {
            gridsize = gridsize / 2;
            let mut corner: u8 = 0;
            if x >= gridsize {
                corner |= 0b100;
                x -= gridsize;
            }
            if y >= gridsize {
                corner |= 0b010;
                y -= gridsize;
            }
            if z >= gridsize {
                corner |= 0b001;
                z -= gridsize;
            }
            let node_ref = &self.arena[handle];
            if node_ref.freemask & (1 << corner) == 0 {
                return node_ref.data[corner as usize];
            }
            handle = node_ref.child_handle(corner.into());
        }
        // gridsize is now equal to 2
        debug_assert_eq!(gridsize, 2);
        let mut corner: u8 = 0;
        if x >= 1 {
            corner |= 0b100;
        }
        if y >= 1 {
            corner |= 0b010;
        }
        if z >= 1 {
            corner |= 0b001;
        }
        self.arena[handle].data[corner as usize]
    }
}
