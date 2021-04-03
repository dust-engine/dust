use super::super::Octree;
use crate::alloc::Handle;

use crate::Voxel;

fn set_recursive<T: Voxel>(
    octree: &mut Octree<T>,
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
        let node_ref = octree.arena.get_mut(handle);
        node_ref.data[corner as usize] = item;
        if node_ref.freemask & (1 << corner) != 0 {
            // has children. Cut them off.
            todo!()
        }
    } else {
        let node_ref = octree.arena.get_mut(handle);
        let freemask = node_ref.freemask;
        if freemask & (1 << corner) == 0 {
            // no children
            octree.reshape(handle, freemask | (1 << corner));
        }

        let new_handle = octree.arena.get(handle).child_handle(corner.into());
        let (avg, collapsed) = set_recursive(octree, new_handle, x, y, z, gridsize, item);

        let node_ref = octree.arena.get_mut(handle);
        let freemask = node_ref.freemask;
        node_ref.data[corner as usize] = avg;
        if collapsed {
            octree.reshape(handle, freemask & !(1 << corner));
        }
    }

    let node_ref = octree.arena.get_mut(handle);
    if node_ref.freemask == 0 {
        // node has no children
        if node_ref.data.iter().all(|a| *a == item) {
            // collapse node
            return (item, true);
        }
    }
    let avg = T::avg(&node_ref.data);
    octree.arena.changed(handle);
    return (avg, false);
}

pub fn get<T: Voxel>(
    octree: &Octree<T>,
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut gridsize: u32,
) -> T {
    let mut handle = octree.root;
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
        let node_ref = &octree.arena.get(handle);
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
    octree.arena.get(handle).data[corner as usize]
}

pub struct RandomAccessor<'a, T: Voxel> {
    pub octree: &'a Octree<T>,
}

impl<'a, T: Voxel> RandomAccessor<'a, T> {
    pub fn get(&self, x: u32, y: u32, z: u32, gridsize: u32) -> T {
        get(self.octree, x, y, z, gridsize)
    }
}

pub struct RandomMutator<'a, T: Voxel> {
    pub octree: &'a mut Octree<T>,
}

impl<'a, T: Voxel> RandomMutator<'a, T> {
    pub fn get(&self, x: u32, y: u32, z: u32, gridsize: u32) -> T {
        get(self.octree, x, y, z, gridsize)
    }
    pub fn set(&mut self, x: u32, y: u32, z: u32, gridsize: u32, item: T) {
        let (data, _collapsed) =
            set_recursive(self.octree, self.octree.root, x, y, z, gridsize, item);
        self.octree.root_data = data;
    }
}
