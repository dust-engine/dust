use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{AabbU32, Node, NodeMeta, pool::Pool, pool::PoolStorage};

pub struct Tree<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub(crate) root: ROOT,
    pub(crate) pool: [Pool; ROOT::LEVEL as usize],
    pub(crate) aabb: AabbU32,
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
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    pub fn new() -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize];
        let metas = Self::metas();
        for (i, meta) in metas.iter().take(ROOT::LEVEL).enumerate() {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }

        let pools: [Pool; ROOT::LEVEL as usize] = unsafe { MaybeUninit::array_assume_init(pools) };
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
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize];
        let metas = Self::metas();
        for (i, meta) in metas.iter().take(ROOT::LEVEL).enumerate().skip(1) {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }
        pools[0].write(Pool::new_with_storage(metas[0].layout, storage));

        let pools: [Pool; ROOT::LEVEL as usize] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
        }
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

    /// ```
    /// #![feature(generic_const_exprs)]
    /// use dust_vdb::{Tree, hierarchy};
    /// use glam::UVec3;
    /// let mut tree = Tree::<hierarchy!(4, 2)>::new();
    /// tree.set_value(UVec3::new(0, 1, 2), Some(true));
    /// tree.set_value(UVec3::new(63, 1, 3), Some(true));
    /// tree.set_value(UVec3::new(63, 63, 63), Some(true));
    /// let mut iter = tree.iter();
    /// assert_eq!(iter.next().unwrap(), UVec3::new(0, 1, 2));
    /// assert_eq!(iter.next().unwrap(), UVec3::new(63, 1, 3));
    /// assert_eq!(iter.next().unwrap(), UVec3::new(63, 63, 63));
    /// assert!(iter.next().is_none());
    ///
    /// ```
    pub fn iter<'a>(&'a self) -> ROOT::Iterator<'a> {
        self.root.iter(&self.pool, UVec3 { x: 0, y: 0, z: 0 })
    }

    pub fn iter_leaf<'a>(&'a self) -> impl Iterator<Item = (UVec3, &'a <ROOT as Node>::LeafType)> {
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

    pub fn metas() -> [NodeMeta<ROOT::LeafType>; ROOT::LEVEL as usize + 1] {
        let mut arr = [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize + 1];
        ROOT::write_meta(&mut arr);
        unsafe { MaybeUninit::array_assume_init(arr) }
    }
}
