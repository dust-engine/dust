use std::marker::PhantomData;
use crate::alloc::Handle;
use crate::octree::Octree;
use crate::{Bounds, Corner, Voxel};

struct NodeInner<T: Voxel> {
    handle: Handle,
    bounds: Bounds,
    occupancy: bool,
    _data_marker: PhantomData<T>,
}
impl<T: Voxel> NodeInner<T> {
    fn child(&self, dir: Corner, octree: &Octree<T>) -> NodeInner<T> {
        let new_bounds = self.bounds.half(dir);
        if self.handle.is_none() {
            // Virtual Node
            NodeInner {
                handle: Handle::none(),
                occupancy: self.occupancy,
                bounds: new_bounds,
                _data_marker: PhantomData,
            }
        } else {
            let node_ref = octree.arena.get(self.handle);
            let new_handle = if node_ref.freemask & (1 << (dir as u8)) == 0 {
                Handle::none()
            } else {
                node_ref.child_handle(dir)
            };
            NodeInner {
                handle: new_handle,
                occupancy: node_ref.occupancy & (1 << dir as u32) != 0,
                bounds: new_bounds,
                _data_marker: PhantomData,
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
    pub fn get(&self) -> bool {
        self.inner.occupancy
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
    pub fn get(&self) -> bool {
        self.inner.occupancy
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
    pub fn is_virtual(&self) -> bool {
        self.inner.handle.is_none()
    }
}

impl<T: Voxel> Octree<T> {
    pub fn get_tree_accessor(&self) -> NodeRef<T> {
        let occupancy = self.root_occupancy;
        let handle = self.root;
        NodeRef {
            octree: self,
            inner: NodeInner {
                handle,
                bounds: Bounds::new(),
                occupancy,
                _data_marker: PhantomData,
            },
        }
    }
    pub fn get_tree_mutator(&mut self) -> NodeRefMut<T> {
        let occupancy = self.root_occupancy;
        let handle = self.root;
        NodeRefMut {
            octree: self,
            inner: NodeInner {
                handle,
                bounds: Bounds::new(),
                occupancy,
                _data_marker: PhantomData,
            },
        }
    }
}
