use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{
    AabbU32, Node, NodeConst, NodeMeta,
    pool::{Pool, PoolStorage},
};

pub struct Tree<ROOT: Node>
where
    [(); ROOT::LEVEL]: Sized,
{
    pub(crate) root: ROOT,
    pub(crate) pool: [Pool; ROOT::LEVEL],
    pub(crate) aabb: AabbU32,
}

impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    pub fn new() -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL];
        for (i, meta) in ROOT::META.iter().take(ROOT::LEVEL).enumerate() {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }

        let pools: [Pool; ROOT::LEVEL] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
        }
    }
    pub fn new_with_leaf_storage(storage: Box<dyn PoolStorage>) -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL];
        for (i, meta) in ROOT::META.iter().take(ROOT::LEVEL).enumerate().skip(1) {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }
        pools[0].write(Pool::new_with_storage(ROOT::META[0].layout, storage));

        let pools: [Pool; ROOT::LEVEL] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
        }
    }
    pub fn pools(&self) -> &[Pool] {
        &self.pool
    }
    pub unsafe fn alloc_node<CHILD: Node>(&mut self) -> u32 {
        unsafe {
            if ROOT::LEVEL <= CHILD::LEVEL {
                panic!("Can not allocate root node");
            }
            let pool = &mut self.pool[CHILD::LEVEL as usize];
            pool.alloc::<CHILD>()
        }
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node<CHILD: Node>(&self, ptr: u32) -> &CHILD {
        unsafe {
            if CHILD::LEVEL == ROOT::LEVEL {
                // specialization for root
                return &*(&self.root as *const ROOT as *const CHILD);
            }
            &*(self.pool[CHILD::LEVEL as usize].get(ptr) as *const CHILD)
        }
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node_mut<CHILD: Node>(&mut self, ptr: u32) -> &mut CHILD {
        unsafe {
            if CHILD::LEVEL == ROOT::LEVEL {
                // specialization for root
                return &mut *(&mut self.root as *mut ROOT as *mut CHILD);
            }
            &mut *(self.pool[CHILD::LEVEL as usize].get_mut(ptr) as *mut CHILD)
        }
    }

    pub fn iter<'a>(&'a self) -> ROOT::Iterator<'a> {
        self.root.iter(&self.pool, UVec3 { x: 0, y: 0, z: 0 })
    }

    pub fn iter_leaf<'a>(
        &'a self,
    ) -> impl Iterator<Item = (UVec3, &'a <ROOT as NodeConst>::LeafType)> {
        self.root
            .iter_leaf(&self.pool, UVec3 { x: 0, y: 0, z: 0 })
            .map(|(position, leaf)| unsafe {
                let leaf: &'a ROOT::LeafType = &*leaf.get();
                (position, leaf)
            })
    }

    pub fn iter_leaf_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (UVec3, &'a mut ROOT::LeafType)> {
        self.root
            .iter_leaf(&mut self.pool, UVec3 { x: 0, y: 0, z: 0 })
            .map(|(position, leaf)| unsafe {
                let leaf: &'a mut ROOT::LeafType = &mut *leaf.get();
                (position, leaf)
            })
    }

    pub fn count_leaves(&self) -> usize {
        self.root.count_leaves(&self.pool)
    }
}
