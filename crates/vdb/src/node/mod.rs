mod internal;
mod leaf;
//mod root;

use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::{alloc::Layout, result};

use glam::UVec3;
pub use internal::*;
pub use leaf::*;
//pub use root::*;

use crate::{ConstUVec3, Pool};

pub struct NodeMeta<V> {
    pub(crate) layout: Layout,
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

    fn write_meta(metas: &mut Vec<NodeMeta<Self::LeafType>>);

    #[cfg(feature = "physics")]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        solid: bool,
        initial_intersection_t: glam::Vec2,
        pools: &[Pool],
    ) -> Option<parry3d::query::RayIntersection> {
        None
    }
}

/// Macro that simplifies tree type construction.
/// ```
/// use dust_vdb::{hierarchy, Node};
/// // Create a 4x4x4 LeafNode
/// let hierarchy = <hierarchy!(2)>::new();
/// // Create a two-level tree with 2x2x2 leafs and 8x8x8 root.
/// let hierarchy = <hierarchy!(3, 1)>::new();
/// // Create a three-level tree with 2x2x2 leafs, 4x4x4 intermediate nodes and 4x4x4 root.
/// let hierarchy = <hierarchy!(2, 2, 1)>::new();
/// // Create a three-level tree with infinite size (implemented with a HashMap), 4x4x4 intermediate nodes and 2x2x2 leafs.
/// let hierarchy = <hierarchy!(#, 2, 1)>::new();
/// ```
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
