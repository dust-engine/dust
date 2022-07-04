use std::{alloc::Layout, mem::MaybeUninit};

use glam::UVec3;

use crate::{Node, Pool};

pub struct Tree<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub(crate) root: ROOT,
    pool: [Pool; ROOT::LEVEL as usize],
}

/// ```
/// #![feature(generic_const_exprs)]
/// use dust_vdb::{hierarchy, Node, Tree};
/// use glam::UVec3;
/// let mut tree = Tree::<hierarchy!(2, 2)>::new();
/// tree.set_value(UVec3{x: 0, y: 4, z: 0}, Some(true));
/// tree.set_value(UVec3{x: 0, y: 2, z: 2}, Some(false));
/// assert_eq!(tree.get_value(UVec3::new(0, 4, 0)), Some(true));
/// assert_eq!(tree.get_value(UVec3::new(0, 3, 0)), None);
/// assert_eq!(tree.get_value(UVec3::new(0, 2, 2)), Some(false));
/// ```
impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub fn new() -> Self {
        let mut layouts: [MaybeUninit<Layout>; ROOT::LEVEL as usize] = MaybeUninit::uninit_array();
        ROOT::write_layout::<ROOT>(&mut layouts);
        let layouts: [Layout; ROOT::LEVEL as usize] = unsafe {
            // https://github.com/rust-lang/rust/issues/61956#issuecomment-1075275504
            (&*(&MaybeUninit::new(layouts) as *const _ as *const MaybeUninit<_>)).assume_init_read()
        };
        let pools = layouts.map(|layout| Pool::new(layout, 10));
        Self {
            root: ROOT::default(),
            pool: pools,
        }
    }
    pub unsafe fn alloc_node<CHILD: Node>(&mut self) -> u32 {
        if ROOT::LEVEL <= CHILD::LEVEL {
            panic!("Can not allocate root node");
        }
        let pool = &mut self.pool[CHILD::LEVEL as usize];
        let ptr = pool.alloc();
        let new_node = self.get_node_mut::<CHILD>(ptr);
        *new_node = Default::default();
        ptr
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node<CHILD: Node>(&self, ptr: u32) -> &CHILD {
        if CHILD::LEVEL == ROOT::LEVEL {
            // specialization for root
            return &*(&self.root as *const ROOT as *const CHILD);
        }
        &*(self.pool[CHILD::LEVEL as usize].get(ptr) as *const CHILD)
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node_mut<CHILD: Node>(&mut self, ptr: u32) -> &mut CHILD {
        if CHILD::LEVEL == ROOT::LEVEL {
            // specialization for root
            return &mut *(&mut self.root as *mut ROOT as *mut CHILD);
        }
        &mut *(self.pool[CHILD::LEVEL as usize].get_mut(ptr) as *mut CHILD)
    }

    #[inline]
    pub fn get_value(&self, coords: UVec3) -> Option<ROOT::Voxel> {
        ROOT::get(self, coords, 123)
    }

    #[inline]
    pub fn set_value(&mut self, coords: UVec3, value: Option<ROOT::Voxel>) {
        ROOT::set(self, coords, 123, value)
    }
}
