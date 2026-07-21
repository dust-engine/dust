mod internal;
mod leaf;
//mod root;

use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

use glam::UVec3;
pub use internal::*;
pub use leaf::*;
//pub use root::*;

use crate::{ConstUVec3, pool::Pool};

pub struct NodeMeta<V> {
    pub layout: Layout,
    pub(crate) getter: for<'a> fn(
        pools: &'a [Pool],
        coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<&'a V>,
    pub(crate) setter: for<'a> fn(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        value: bool,
        cached_path: &mut [u32],
    ) -> &'a mut V,
    pub(crate) extent_log2: UVec3,
    pub(crate) fanout_log2: UVec3,

    pub(crate) extent_mask: UVec3, // = (1 << extent_log2) - 1
}

pub trait Node: 'static + Send + Sync + Default + Clone {
    /// span of the node.
    type LeafType: IsLeaf;
    const EXTENT_LOG2: UVec3;
    const EXTENT: UVec3;
    const EXTENT_MASK: UVec3; // = (1 << extent_log2) - 1
    const META_MASK: UVec3;
    /// Max number of child nodes.
    const SIZE: usize;

    /// This is 0 for leaf nodes and +1 for each layer of nodes above leaves.
    const LEVEL: usize;

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was owned.
    /// Implementation will write to cached_path for all levels below the current level.
    fn get<'a>(
        &'a self,
        pools: &'a [Pool],
        coords: UVec3,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType>;

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    fn get_in_pools<'a>(
        pools: &'a [Pool],
        coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType>;

    fn set<'a>(
        &'a mut self,
        pools: &'a mut [Pool],
        coords: UVec3,
        value: bool,
        cached_path: &mut [u32],
    ) -> &'a mut Self::LeafType;
    /// Set the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    /// Returns a reference to the old leaf, and a reference to the new one.
    /// The old leaf is None if the node was not changed or if the old leaf node did not exist.
    fn set_in_pools<'a>(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        value: bool,
        cached_path: &mut [u32],
    ) -> &'a mut Self::LeafType;

    type Iterator<'a>: Iterator<Item = UVec3>;
    /// This is called when the node was owned as the root node in the tree.
    fn iter<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::Iterator<'a>;
    /// This is called when the node was located in a node pool.
    fn iter_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::Iterator<'a>;

    type LeafIterator<'a>: Iterator<Item = (UVec3, &'a UnsafeCell<Self::LeafType>)>;
    /// This is called when the node was owned as the root node in the tree.
    fn iter_leaf<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a>;
    /// This is called when the node was located in a node pool.
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a>;

    fn count_leaves(&self, pools: &[Pool]) -> usize;

    fn write_meta(metas: &mut [MaybeUninit<NodeMeta<Self::LeafType>>]);
}

/// Macro that simplifies tree type construction.
///
/// The last argument is the per-leaf value type (e.g. an attribute pointer); the preceding
/// arguments are the log2 extents of each level, from the root down to the leaves.
/// ```
/// #![feature(generic_const_exprs)]
/// use dust_vdb::{hierarchy, Node};
/// // A 4x4x4 LeafNode with a `u32` per-leaf value.
/// type Leaf = hierarchy!(2, u32);
/// // A two-level tree with 2x2x2 leaves and a 8x8x8 root (16x16x16 total).
/// type TwoLevels = hierarchy!(3, 1, u32);
/// // A three-level tree with 2x2x2 leaves, 4x4x4 intermediate nodes and a 4x4x4 root
/// // (32x32x32 total).
/// type ThreeLevels = hierarchy!(2, 2, 1, u32);
///
/// assert_eq!(<Leaf as Node>::EXTENT_LOG2.x, 2);
/// assert_eq!(<TwoLevels as Node>::EXTENT_LOG2.x, 4);
/// assert_eq!(<ThreeLevels as Node>::EXTENT_LOG2.x, 5);
/// ```
///
/// The `hierarchy!(#, ...)` form maps to the hash-map based `RootNode` for unbounded domains,
/// which is currently disabled.
#[macro_export]
macro_rules! hierarchy {
    ($e: tt, $t: ty) => {
        $crate::LeafNode<{dust_vdb::ConstUVec3{x:$e,y:$e,z:$e}}, $t>
    };
    (#, $($n:tt),+) => {
        $crate::RootNode<hierarchy!($($n),*)>
    };
    ($e: tt, $($n:tt),+) => {
        $crate::InternalNode::<hierarchy!($($n),*), {dust_vdb::ConstUVec3{x:$e,y:$e,z:$e}}>
    };
}

/// Returns the size of a grid represented by the log2 of its extent.
/// This is needed because of Rust limitations.
/// Won't need this once we're allowed to use Self::Size in the bounds.
pub const fn size_of_grid(log2: ConstUVec3) -> usize {
    return 1 << (log2.x + log2.y + log2.z);
}
