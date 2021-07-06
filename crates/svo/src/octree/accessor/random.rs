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
    occupancy: bool,
) -> (bool, u8, bool) {
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
        if !occupancy {
            node_ref.occupancy &= !(1 << corner);
        } else {
            node_ref.occupancy |= 1 << corner;
        }
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
        let (avg, all, collapsed) = set_recursive(octree, new_handle, x, y, z, gridsize, occupancy);

        let node_ref = octree.arena.get_mut(handle);
        let freemask = node_ref.freemask;
        if !avg {
            node_ref.occupancy &= !(1 << corner);
        } else {
            node_ref.occupancy |= 1 << corner;
        }
        node_ref.extended_occupancy[corner as usize] = all;
        if collapsed {
            octree.reshape(handle, freemask & !(1 << corner));
        }
    }

    let node_ref = octree.arena.get_mut(handle);
    if node_ref.freemask == 0 {
        // node has no children
        if (node_ref.occupancy == 255 || node_ref.occupancy == 1)
            && node_ref.occupancy >> 7 == occupancy as u8
        {
            // collapse node
            return (occupancy, node_ref.occupancy, true);
        }
    }
    let avg = node_ref.occupancy != 0;
    let all = node_ref.occupancy;
    octree.arena.changed(handle);
    return (avg, all, false);
}

pub fn get<T: Voxel>(
    octree: &Octree<T>,
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut gridsize: u32,
) -> bool {
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
            return node_ref.occupancy & (1 << corner) != 0;
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
    octree.arena.get(handle).occupancy & (1 << corner) != 0
}

pub struct RandomAccessor<'a, T: Voxel> {
    pub octree: &'a Octree<T>,
}

impl<'a, T: Voxel> RandomAccessor<'a, T> {
    pub fn get(&self, x: u32, y: u32, z: u32, gridsize: u32) -> bool {
        get(self.octree, x, y, z, gridsize)
    }
}

pub struct RandomMutator<'a, T: Voxel> {
    pub octree: &'a mut Octree<T>,
}

impl<'a, T: Voxel> RandomMutator<'a, T> {
    pub fn get(&self, x: u32, y: u32, z: u32, gridsize: u32) -> bool {
        get(self.octree, x, y, z, gridsize)
    }
    pub fn set(&mut self, x: u32, y: u32, z: u32, gridsize: u32, item: bool) {
        let (data, _ext, _collapsed) =
            set_recursive(self.octree, self.octree.root, x, y, z, gridsize, item);
        self.octree.root_occupancy = data;
    }
}

impl<T: Voxel> Octree<T> {
    pub fn get_random_accessor(&self) -> RandomAccessor<T> {
        RandomAccessor { octree: self }
    }
    pub fn get_random_mutator(&mut self) -> RandomMutator<T> {
        RandomMutator { octree: self }
    }
}
