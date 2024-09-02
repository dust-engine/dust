use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{AabbU32, IsLeaf, Node, NodeMeta, Pool};

pub struct MutableTree<ROOT: Node>
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
impl<ROOT: Node> MutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    pub fn new() -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] = MaybeUninit::uninit_array();
        for (i, meta) in Self::metas().iter().take(ROOT::LEVEL).enumerate() {
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
            aabb: AabbU32::default(),
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
    pub fn set_value(&mut self, coords: UVec3, value: bool) {
        if value {
            self.aabb.min = self.aabb.min.min(coords);
            self.aabb.max = self.aabb.max.max(coords);
        }
        let mut _result = false;
        //self.root.set(&mut self.pool, coords, value, &mut [], &mut _result);
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

    pub fn metas() -> Vec<NodeMeta<ROOT::LeafType>> {
        let mut vec = Vec::with_capacity(ROOT::LEVEL + 1);
        ROOT::write_meta(&mut vec);
        vec
    }
}

pub trait TreeLike: Send + Sync {
    fn get_value(&self, coords: UVec3) -> bool;

    fn aabb(&self) -> AabbU32;

    fn extent(&self) -> UVec3;

    #[cfg(feature = "physics")]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        solid: bool,
        initial_intersection_t: glam::Vec2,
    ) -> Option<parry3d::query::RayIntersection>;
}

impl<ROOT: Node> TreeLike for MutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn get_value(&self, coords: UVec3) -> bool {
        let mut result = false;
        //self.root.get(&self.pool, coords, &mut [], &mut result);
        result
    }
    fn aabb(&self) -> AabbU32 {
        self.aabb
    }
    fn extent(&self) -> UVec3 {
        ROOT::EXTENT
    }
    #[cfg(feature = "physics")]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        solid: bool,
        initial_intersection_t: glam::Vec2,
    ) -> Option<parry3d::query::RayIntersection> {
        self.root
            .cast_local_ray_and_get_normal(ray, solid, initial_intersection_t, &self.pool)
    }
}
