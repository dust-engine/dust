mod internal;
mod leaf;
mod root;

use std::alloc::Layout;
use std::mem::MaybeUninit;

use glam::UVec3;
pub use internal::*;
pub use leaf::*;
pub use root::*;

use crate::Tree;

pub trait Node: 'static + Default {
    /// span of the node.
    const EXTENT_LOG2: UVec3;
    /// Max number of child nodes.
    const SIZE: usize;

    /// This is 0 for leaf nodes and +1 for each layer of nodes above leaves.
    const LEVEL: u8;
    fn new() -> Self;

    type Voxel;
    fn get<ROOT: Node>(tree: &Tree<ROOT>, coords: UVec3, ptr: u32) -> Option<Self::Voxel>
    where
        [(); ROOT::LEVEL as usize]: Sized;
    fn set<ROOT: Node>(tree: &mut Tree<ROOT>, coords: UVec3, ptr: u32, value: Option<Self::Voxel>)
    where
        [(); ROOT::LEVEL as usize]: Sized;

    fn write_layout<ROOT: Node>(sizes: &mut [MaybeUninit<Layout>]);
    /*
    type Iterator: Iterator<Item = Self::Voxel>;
    fn iter_active<ROOT: Node>(tree: &Tree<ROOT>, ptr: u32) -> Self::Iterator
    where
        [(); ROOT::LEVEL as usize]: Sized;
        */
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
