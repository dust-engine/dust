pub mod io;
use crate::{Arena, ArenaHandle, Bounds, Corner, IndexPath, Voxel};
use glam::Vec3;

struct NodeInner<T: Voxel> {
    handle: ArenaHandle<T>,
    index_path: IndexPath,
    bounds: Bounds,
    data: T,
}

impl<T: Voxel> NodeInner<T> {
    fn child(&self, dir: Corner, octree: &Octree<T>) -> NodeInner<T> {
        let new_index_path = self.index_path.push(dir);
        let new_bounds = self.bounds.half(dir);
        if self.handle.is_none() {
            // Virtual Node
            NodeInner {
                handle: ArenaHandle::none(),
                index_path: new_index_path,
                data: self.data,
                bounds: new_bounds,
            }
        } else {
            let node_ref = &octree.arena[self.handle];
            let new_handle = if node_ref.freemask & (1 << (dir as u8)) == 0 {
                ArenaHandle::none()
            } else {
                node_ref.child(dir)
            };
            NodeInner {
                handle: new_handle,
                index_path: new_index_path,
                data: node_ref.data[dir as usize],
                bounds: new_bounds,
            }
        }
    }
}

pub struct NodeRef<'a, T: Voxel> {
    inner: NodeInner<T>,
    octree: &'a Octree<T>,
}
impl<'a, T: Voxel> NodeRef<'a, T> {
    #[inline]
    pub fn get(&self) -> T {
        self.inner.data
    }

    #[inline]
    pub fn child(&self, dir: Corner) -> NodeRef<T> {
        NodeRef {
            octree: self.octree,
            inner: self.inner.child(dir, self.octree),
        }
    }

    #[inline]
    pub fn get_bounds(&self) -> &Bounds {
        &self.inner.bounds
    }

    #[inline]
    pub fn is_virtual(&self) -> bool {
        self.inner.handle.is_none()
    }
}

pub struct NodeRefMut<'a, T: Voxel> {
    inner: NodeInner<T>,
    octree: &'a mut Octree<T>,
}
impl<'a, T: Voxel> NodeRefMut<'a, T> {
    pub fn get(&self) -> T {
        self.inner.data
    }
    /// Only applicable on leaf nodes
    pub fn set_leaf_child(&mut self, dir: Corner, value: T) {
        let node_ref = &mut self.octree.arena[self.inner.handle];
        node_ref.data[dir as usize] = value;
    }
    pub fn child(&mut self, dir: Corner) -> NodeRefMut<T> {
        let inner = self.inner.child(dir, self.octree);
        NodeRefMut {
            octree: self.octree,
            inner,
        }
    }
    pub fn get_bounds(&self) -> &Bounds {
        &self.inner.bounds
    }
    /// Only applicable on leaf nodes
    /// Assuming that the current node is a leaf node (self.leafmask == 0),
    /// this function allocates enough space for the given new leaf mask,
    /// and update the current freemask and children pointer for the node.
    pub fn set_leaf_childmask(&mut self, childmask: u8) {
        if childmask == 0 {
            return;
        }
        let new_child_handle = self.octree.arena.alloc(childmask.count_ones());
        let handle = self.inner.handle;
        let node_ref = &mut self.octree.arena[handle];
        debug_assert_eq!(node_ref.freemask, 0);
        node_ref.freemask = childmask;
        node_ref.children = new_child_handle;
    }
    pub fn is_virtual(&self) -> bool {
        self.inner.handle.is_none()
    }
}

pub struct Octree<T: Voxel> {
    arena: Arena<T>,
    root: ArenaHandle<T>,
    root_data: T,
}

impl<T: Voxel> Octree<T> {
    pub fn new() -> Self {
        let mut arena: Arena<T> = Arena::new();
        let root = arena.alloc(1);
        Octree {
            arena,
            root,
            root_data: Default::default(),
        }
    }
    pub fn root_mut(&mut self) -> NodeRefMut<T> {
        let data = self.root_data;
        let handle = self.root;
        NodeRefMut {
            octree: self,
            inner: NodeInner {
                handle,
                index_path: IndexPath::new(),
                bounds: Bounds::new(),
                data,
            },
        }
    }
    pub fn root(&self) -> NodeRef<T> {
        let data = self.root_data;
        let handle = self.root;
        NodeRef {
            octree: self,
            inner: NodeInner {
                handle,
                index_path: IndexPath::new(),
                bounds: Bounds::new(),
                data,
            },
        }
    }

    pub fn get(&self, mut x: u32, mut y: u32, mut z: u32, mut gridsize: u32) -> T {
        let mut handle = self.root().inner.handle;
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
            handle = node_ref.child(corner.into());
        }
        // gridsize is now equal to 2
        assert_eq!(gridsize, 2);
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

    fn set_internal(
        &mut self,
        handle: ArenaHandle<T>,
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
                self.arena.realloc(handle, freemask | (1 << corner));
            }

            let new_handle = self.arena[handle].child(corner.into());
            let (avg, collapsed) = self.set_internal(new_handle, x, y, z, gridsize, item);

            let node_ref = &mut self.arena[handle];
            let freemask = node_ref.freemask;
            node_ref.data[corner as usize] = avg;
            if collapsed {
                self.arena.realloc(handle, freemask & !(1 << corner));
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
        let (data, _collapsed) =
            self.set_internal(self.root().inner.handle, x, y, z, gridsize, item);
        self.root_data = data;
    }
    #[inline]
    pub fn total_data_size(&self) -> usize {
        self.arena.total_data_size()
    }
    #[inline]
    pub fn copy_into_slice(&self, slice: &mut [u8]) {
        self.arena.copy_data_into_slice(slice)
    }
}

impl<T: Voxel> Octree<T> {
    fn signed_distance_field_recursive<F>(
        signed_distance_field: &F,
        fill: T,
        lod: u8,
        mut node: NodeRefMut<T>,
    ) where
        F: Fn(Vec3) -> f32,
    {
        assert!(!node.inner.handle.is_none()); // Can't be a virtual node
        let mut childmask: u8 = 0;
        for (i, dir) in Corner::all().enumerate() {
            let child = node.child(dir);
            let mut corners_flag: u8 = 0;
            for (i, dir) in Corner::all().enumerate() {
                let corner = child.get_bounds().corner(dir);
                let value = signed_distance_field(corner);
                if value > 0.0 {
                    corners_flag |= 1 << i;
                }
            }
            if corners_flag == 0 {
                // All corners smaller than zero. Do not fill.
            } else if corners_flag == std::u8::MAX {
                // All corners larger than zero. Fill.
                node.set_leaf_child(dir, fill);
            } else {
                // This child node needs to be subdivided
                childmask |= 1 << i;
                if corners_flag.count_ones() >= 4 {
                    // relative full
                    node.set_leaf_child(dir, fill);
                }
                // Set the childmask here so it can be handled later on.
            }
        }
        if lod > 0 {
            node.set_leaf_childmask(childmask);

            for (i, dir) in Corner::all().enumerate() {
                if (1 << i) & childmask == 0 {
                    // No child on this corner
                    continue;
                }
                let child = node.child(dir);
                Octree::signed_distance_field_recursive(
                    signed_distance_field,
                    fill,
                    lod - 1,
                    child,
                );
            }
        }
    }
    pub fn from_signed_distance_field<F>(field: F, fill: T, lod: u8) -> Octree<T>
        where
            F: Fn(Vec3) -> f32,
    {
        let mut octree = Octree::new();
        Octree::signed_distance_field_recursive(&field, fill, lod, octree.root_mut());
        octree
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_distance_field() {
        let _octree: Octree<u16> =
            Octree::from_signed_distance_field(|l: Vec3| 3.0 - l.length(), 1, 2);
    }

    #[test]
    fn test_set() {
        let mut octree: Octree<u16> = Octree::new();
        for (i, corner) in Corner::all().enumerate() {
            let (x, y, z) = corner.position_offset();
            octree.set(x as u32, y as u32, z as u32, 8, 3);
            assert_eq!(octree.get(x as u32, y as u32, z as u32, 8), 3);

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
