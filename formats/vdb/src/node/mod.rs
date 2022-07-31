mod internal;
mod leaf;
mod root;

use std::alloc::Layout;
use std::fmt::Debug;
use std::mem::{size_of, MaybeUninit};

use glam::UVec3;
pub use internal::*;
pub use leaf::*;
pub use root::*;

use crate::Pool;

pub struct NodeMeta<V> {
    pub(crate) layout: Layout,
    pub(crate) getter:
        fn(pools: &[Pool], coords: UVec3, ptr: u32, cached_path: &mut [u32]) -> Option<V>,
    pub(crate) setter:
        fn(pools: &mut [Pool], coords: UVec3, ptr: u32, value: Option<V>, cached_path: &mut [u32]),
    pub(crate) extent_log2: UVec3,
    pub(crate) fanout_log2: UVec3,

    pub(crate) extent_mask: UVec3, // = (1 << extent_log2) - 1
}

pub trait Node: 'static + Default + Debug {
    /// span of the node.
    type LeafType: IsLeaf;
    const EXTENT_LOG2: UVec3;
    const EXTENT: UVec3;
    const EXTENT_MASK: UVec3; // = (1 << extent_log2) - 1
    /// Max number of child nodes.
    const SIZE: usize;

    /// This is 0 for leaf nodes and +1 for each layer of nodes above leaves.
    const LEVEL: usize;
    fn new() -> Self;

    type Voxel;

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was owned.
    /// Implementation will write to cached_path for all levels below the current level.
    fn get(&self, pools: &[Pool], coords: UVec3, cached_path: &mut [u32]) -> Option<Self::Voxel>;
    /// Set the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was owned.
    /// Implementation will write to cached_path for all levels below the current level.
    fn set(
        &mut self,
        pools: &mut [Pool],
        coords: UVec3,
        value: Option<Self::Voxel>,
        cached_path: &mut [u32],
    );

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    fn get_in_pools(
        pools: &[Pool],
        coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<Self::Voxel>;
    /// Set the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    fn set_in_pools(
        pools: &mut [Pool],
        coords: UVec3,
        ptr: u32,
        value: Option<Self::Voxel>,
        cached_path: &mut [u32],
    );

    type Iterator<'a>: Iterator<Item = UVec3>;
    /// This is called when the node was owned as the root node in the tree.
    fn iter<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::Iterator<'a>;
    /// This is called when the node was located in a node pool.
    fn iter_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::Iterator<'a>;

    type LeafIterator<'a>: Iterator<Item = (UVec3, &'a Self::LeafType)>;
    /// This is called when the node was owned as the root node in the tree.
    fn iter_leaf<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a>;
    /// This is called when the node was located in a node pool.
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a>;
}

/// Trait that contains const methods for the node.
pub trait NodeConst: Node {
    /// Method that congregates metadata of each level of the tree into an array.
    /// Implementation shoud write NodeMeta into `metas[Self::LEVEL]`.
    fn write_meta(metas: &mut [MaybeUninit<NodeMeta<Self::Voxel>>]);
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
    ($e: tt) => {
        $crate::LeafNode<{glam::UVec3{x:$e,y:$e,z:$e}}>
    };
    (#, $($n:tt),+) => {
        $crate::RootNode<hierarchy!($($n),*)>
    };
    ($e: tt, $($n:tt),+) => {
        $crate::InternalNode::<hierarchy!($($n),*), {glam::UVec3{x:$e,y:$e,z:$e}}>
    };
}

/// Returns the size of a grid represented by the log2 of its extent.
/// This is needed because of Rust limitations.
/// Won't need this once we're allowed to use Self::Size in the bounds.
pub const fn size_of_grid(log2: UVec3) -> usize {
    return 1 << (log2.x + log2.y + log2.z);
}
