use crate::alloc::Handle;
use crate::octree::Octree;
use crate::{Bounds, Corner, Voxel};

struct NodeInner<T: Voxel> {
    handle: Handle,
    bounds: Bounds,
    data: T,
}
impl<T: Voxel> NodeInner<T> {
    fn child(&self, dir: Corner, octree: &Octree<T>) -> NodeInner<T> {
        let new_bounds = self.bounds.half(dir);
        if self.handle.is_none() {
            // Virtual Node
            NodeInner {
                handle: Handle::none(),
                data: self.data,
                bounds: new_bounds,
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
        let data = self.root_data;
        let handle = self.root;
        NodeRef {
            octree: self,
            inner: NodeInner {
                handle,
                bounds: Bounds::new(),
                data,
            },
        }
    }
    pub fn get_tree_mutator(&mut self) -> NodeRefMut<T> {
        let data = self.root_data;
        let handle = self.root;
        NodeRefMut {
            octree: self,
            inner: NodeInner {
                handle,
                bounds: Bounds::new(),
                data,
            },
        }
    }
}
