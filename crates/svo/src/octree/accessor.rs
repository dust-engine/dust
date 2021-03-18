use crate::{Voxel, Corner, Bounds};
use crate::alloc::{Handle, CHUNK_SIZE};
use crate::octree::{Octree, Node};
use std::mem::size_of;

pub struct Accessor<'a, T: Voxel>
    where
        [T; CHUNK_SIZE / size_of::<Node<T>>()]: Sized {
    handle: Handle,
    data: T,
    bounds: Bounds,
    octree: &'a Octree<T>
}

impl<'a, T: Voxel>  Accessor<'a, T>
    where
        [T; CHUNK_SIZE / size_of::<Node<T>>()]: Sized {
    pub fn child(&self, dir: Corner) -> Self {
        let new_bounds = self.bounds.half(dir);
        if self.handle.is_none() {
            // Virtual Node
            Accessor {
                handle: Handle::none(),
                data: self.data,
                bounds: new_bounds,
                octree: self.octree
            }
        } else {
            let node_ref = &self.octree.arena[self.handle];
            let new_handle = if node_ref.freemask & (1 << (dir as u8)) == 0 {
                Handle::none()
            } else {
                node_ref.child(dir)
            };
            Accessor {
                handle: new_handle,
                data: node_ref.data[dir as usize],
                bounds: new_bounds,
                octree: self.octree
            }
        }
    }
}
