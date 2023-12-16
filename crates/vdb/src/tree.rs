use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{Node, NodeConst, NodeMeta, Pool};

pub struct Tree<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub(crate) root: ROOT,
    pub(crate) pool: [Pool; ROOT::LEVEL as usize],
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
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    pub fn new() -> Self
    where
        ROOT: NodeConst,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] = MaybeUninit::uninit_array();
        for (i, meta) in Self::METAS.iter().take(ROOT::LEVEL).enumerate() {
            let pool = Pool::new(meta.layout, 10);
            pools[i].write(pool);
        }

        let pools: [Pool; ROOT::LEVEL as usize] = unsafe {
            // https://github.com/rust-lang/rust/issues/61956#issuecomment-1075275504
            (&*(&MaybeUninit::new(pools) as *const _ as *const MaybeUninit<_>)).assume_init_read()
        };
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
        pool.alloc::<CHILD>()
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
        self.root.get(&self.pool, coords, &mut [])
    }

    #[inline]
    pub fn set_value(&mut self, coords: UVec3, value: Option<ROOT::Voxel>) {
        self.root.set(&mut self.pool, coords, value, &mut [])
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

    pub fn iter_leaf<'a>(&'a self) -> impl Iterator<Item = (UVec3, &'a ROOT::LeafType)> {
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
}

/// Workaround for https://github.com/rust-lang/rust/issues/88424#issuecomment-911158795
#[const_trait]
pub(crate) trait TreeMeta<ROOT: Node>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    const METAS: [NodeMeta<ROOT::Voxel>; ROOT::LEVEL as usize + 1];
    const META_MASK: UVec3;
    const ID: u64;
}

impl<ROOT: ~const NodeConst> const TreeMeta<ROOT> for Tree<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    const METAS: [NodeMeta<ROOT::Voxel>; ROOT::LEVEL as usize + 1] = {
        let mut metas: [MaybeUninit<NodeMeta<ROOT::Voxel>>; ROOT::LEVEL as usize + 1] =
            MaybeUninit::uninit_array();

        ROOT::write_meta(&mut metas);
        let metas: [crate::node::NodeMeta<ROOT::Voxel>; ROOT::LEVEL as usize + 1] = unsafe {
            // https://github.com/rust-lang/rust/issues/61956#issuecomment-1075275504
            (&*(&MaybeUninit::new(metas) as *const _ as *const MaybeUninit<_>)).assume_init_read()
        };

        metas
    };
    const META_MASK: UVec3 = {
        let mut mask: UVec3 = UVec3::ZERO;
        let mut i = 0;
        while i < Self::METAS.len() {
            let meta = &Self::METAS[i];
            mask = UVec3 {
                x: mask.x | (1 << (meta.extent_log2.x - 1)),
                y: mask.y | (1 << (meta.extent_log2.y - 1)),
                z: mask.z | (1 << (meta.extent_log2.z - 1)),
            };
            i += 1;
        }
        mask
    };
    const ID: u64 = {
        let mut id: u64 = 0;
        let mut i = 0;
        while i < Self::METAS.len() {
            let meta = &Self::METAS[i];
            id = id * 32 + meta.extent_log2.x as u64;
            id = id * 32 + meta.extent_log2.y as u64;
            id = id * 32 + meta.extent_log2.z as u64;
            i += 1;
        }
        id
    };
}
