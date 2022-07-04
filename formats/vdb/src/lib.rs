#![feature(int_roundings)]
#![feature(stdsimd)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![feature(maybe_uninit_uninit_array)]
use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{size_of, MaybeUninit},
};

mod bitmask;
mod pool;
pub use bitmask::BitMask;
use glam::u32::UVec3;
pub use pool::Pool;
mod tree;
pub use tree::Tree;

/// Returns the size of a grid represented by the log2 of its extent.
/// This is needed because of Rust limitations.
/// Won't need this once we're allowed to use Self::Size in the bounds.
pub const fn size_of_grid(log2: UVec3) -> usize {
    return 1 << (log2.x + log2.y + log2.z);
}

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
}

/// Nodes are always 4x4x4 so that each leaf node contains exactly 64 voxels,
/// so that the occupancy mask happens to be exactly 64 bits.
/// Size: 3 u32
#[repr(C)]
#[derive(Default)]
pub struct LeafNode<const LOG2: UVec3>
where
    [(); size_of_grid(LOG2) / size_of::<usize>()]: Sized,
{
    /// This is 1 for occupied voxels and 0 for unoccupied voxels
    pub occupancy: BitMask<{ size_of_grid(LOG2) }>,
    /// This is 1 for voxels located on the surface
    pub active: BitMask<{ size_of_grid(LOG2) }>,
    /// A pointer to self.occupancy.count_ones() material values
    pub material_ptr: u32,
}

impl<const LOG2: UVec3> Node for LeafNode<LOG2>
where
    [(); size_of_grid(LOG2) / size_of::<usize>()]: Sized,
{
    /// Total number of voxels contained within the leaf node.
    const SIZE: usize = size_of_grid(LOG2);
    /// Extent of the leaf node in each axis.
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: LOG2.x,
        y: LOG2.y,
        z: LOG2.z,
    };
    const LEVEL: u8 = 0;
    fn new() -> Self {
        Self {
            occupancy: BitMask::new(),
            active: BitMask::new(),
            material_ptr: 0,
        }
    }

    type Voxel = bool;
    #[inline]
    fn get<ROOT: Node>(tree: &Tree<ROOT>, coords: UVec3, ptr: u32) -> Option<Self::Voxel>
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        let leaf_node = unsafe { tree.get_node::<Self>(ptr) };
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        let occupied = leaf_node.occupancy.get(index);
        if !occupied {
            return None;
        }
        let active = leaf_node.active.get(index);
        return Some(active);
    }

    #[inline]
    fn set<ROOT: Node>(tree: &mut Tree<ROOT>, coords: UVec3, ptr: u32, value: Option<Self::Voxel>)
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        let leaf_node = unsafe { tree.get_node_mut::<Self>(ptr) };
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        if let Some(voxel) = value {
            leaf_node.occupancy.set(index, true);
            leaf_node.active.set(index, voxel);
        } else {
            leaf_node.occupancy.set(index, false);
        }
    }

    fn write_layout<ROOT: Node>(sizes: &mut [MaybeUninit<Layout>]) {
        if ROOT::LEVEL != Self::LEVEL {
            let layout = std::alloc::Layout::new::<Self>();
            sizes[Self::LEVEL as usize].write(layout);
        }
    }
}

#[derive(Clone, Copy)]
pub union InternalNodeEntry {
    /// The corresponding bit on child_mask is set. Points to another node.
    pub occupied: u32,
    /// The corresponding bit on child_mask is not set.
    /// Points to a value in the material array that describes all child nodes within the current node.
    /// If this is u32::MAX, this is air.
    pub free: u32,
}

/// Internal nodes are always 4x4x4 so that the child mask contains exactly 64 voxels.
/// Size: 3 - 66 u32
#[repr(C)]
pub struct InternalNode<CHILD: Node, const FANOUT_LOG2: UVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>()]: Sized,
{
    /// This is 0 if that tile is completely air, and 1 otherwise.
    pub child_mask: BitMask<{ size_of_grid(FANOUT_LOG2) }>,
    /// points to self.child_mask.count_ones() LeafNodes or InternalNodes
    pub child_ptrs: [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
    _marker: PhantomData<CHILD>,
}
impl<CHILD: Node, const FANOUT_LOG2: UVec3> Default for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>()]: Sized,
{
    fn default() -> Self {
        Self {
            child_mask: Default::default(),
            child_ptrs: [InternalNodeEntry { free: 0 }; size_of_grid(FANOUT_LOG2)],
            _marker: Default::default(),
        }
    }
}
impl<CHILD: Node, const FANOUT_LOG2: UVec3> Node for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>()]: Sized,
{
    const SIZE: usize = size_of_grid(FANOUT_LOG2);
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: FANOUT_LOG2.x + CHILD::EXTENT_LOG2.x,
        y: FANOUT_LOG2.y + CHILD::EXTENT_LOG2.y,
        z: FANOUT_LOG2.z + CHILD::EXTENT_LOG2.z,
    };
    const LEVEL: u8 = CHILD::LEVEL + 1;
    fn new() -> Self {
        Self {
            child_mask: BitMask::new(),
            child_ptrs: [InternalNodeEntry { free: 0 }; size_of_grid(FANOUT_LOG2)],
            _marker: PhantomData,
        }
    }

    type Voxel = CHILD::Voxel;
    #[inline]
    fn get<ROOT: Node>(tree: &Tree<ROOT>, coords: UVec3, ptr: u32) -> Option<Self::Voxel>
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        let node = unsafe { tree.get_node::<Self>(ptr) };
        let has_child = node.child_mask.get(index);
        if !has_child {
            return None;
        }
        unsafe {
            let child_ptr = node.child_ptrs[index].occupied;
            let new_coords = UVec3 {
                x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
            };
            <CHILD as Node>::get(tree, new_coords, child_ptr)
        }
    }

    #[inline]
    fn set<ROOT: Node>(tree: &mut Tree<ROOT>, coords: UVec3, ptr: u32, value: Option<Self::Voxel>)
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        let node = unsafe { tree.get_node_mut::<Self>(ptr) };

        if value.is_some() {
            // set
            let has_child = node.child_mask.get(index);
            if !has_child {
                // ensure have children
                node.child_mask.set(index, true);
                unsafe {
                    // allocate a child node
                    let allocated_ptr = tree.alloc_node::<CHILD>();
                    let node = unsafe { tree.get_node_mut::<Self>(ptr) };
                    node.child_ptrs[index].occupied = allocated_ptr;
                }
            }
            // TODO: propagate when filled.
        } else {
            // clear
            todo!() // TODO: clear recursively, propagate if completely cleared
        }
        let node = unsafe { tree.get_node_mut::<Self>(ptr) };
        unsafe {
            let new_coords = UVec3 {
                x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
            };
            let child_ptr = node.child_ptrs[index].occupied;
            <CHILD as Node>::set(tree, new_coords, child_ptr, value)
        }
    }

    fn write_layout<ROOT: Node>(sizes: &mut [MaybeUninit<Layout>]) {
        if ROOT::LEVEL != Self::LEVEL {
            let layout = std::alloc::Layout::new::<Self>();
            sizes[Self::LEVEL as usize].write(layout);
        }
        CHILD::write_layout::<ROOT>(sizes);
    }
}

pub enum RootNodeEntry {
    Occupied(u32),
    Free(u32),
}

#[derive(PartialEq, Eq, Clone)]
pub struct RootKey(UVec3);
impl std::hash::Hash for RootKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let root_hash = (self.0.x as u64).wrapping_mul(73856093_u64)
            ^ (self.0.y as u64).wrapping_mul(19349663_u64)
            ^ (self.0.z as u64).wrapping_mul(83492791_u64);
        state.write_u64(root_hash);
    }
}

/// The root node of the tree implemented with a [`std::collections::HashMap`].
/// This enables trees of infinite size.
#[derive(Default)]
pub struct RootNode<CHILD: Node> {
    /// Map from [`RootKey`] to tiles.
    map: std::collections::HashMap<RootKey, RootNodeEntry, nohash::BuildNoHashHasher<u64>>,
    _marker: PhantomData<CHILD>,
}

impl<CHILD: Node> Node for RootNode<CHILD> {
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: 32,
        y: 32,
        z: 32,
    };

    const SIZE: usize = usize::MAX;

    const LEVEL: u8 = CHILD::LEVEL + 1;

    fn new() -> Self {
        todo!()
    }

    type Voxel = CHILD::Voxel;
    #[inline]
    fn get<ROOT: Node>(tree: &Tree<ROOT>, coords: UVec3, ptr: u32) -> Option<Self::Voxel>
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let node = unsafe { tree.get_node::<Self>(ptr) };
        let entry = node.map.get(&RootKey(root_offset));
        if let Some(entry) = entry {
            match entry {
                RootNodeEntry::Free(_material_id) => None,
                RootNodeEntry::Occupied(ptr) => unsafe {
                    let _child_node = tree.get_node::<CHILD>(*ptr);
                    let new_coords = UVec3 {
                        x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                        y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                        z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
                    };
                    <CHILD as Node>::get(tree, new_coords, *ptr)
                },
            }
        } else {
            None
        }
    }

    fn set<ROOT: Node>(tree: &mut Tree<ROOT>, coords: UVec3, ptr: u32, value: Option<Self::Voxel>)
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        // ptr is meaningless and always 0 for root nodes.
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let key = RootKey(root_offset);

        if value.is_some() {
            // Ensure that the node contains stuff on ptr
            let node = unsafe { tree.get_node_mut::<Self>(ptr) };
            if !node.map.contains_key(&key) {
                let new_node_ptr = unsafe { tree.alloc_node::<CHILD>() };
                let node = unsafe { tree.get_node_mut::<Self>(ptr) };
                node.map
                    .insert(key.clone(), RootNodeEntry::Occupied(new_node_ptr));
            }

            let node = unsafe { tree.get_node_mut::<Self>(ptr) };
            let child_ptr = match node.map.get(&key).unwrap() {
                RootNodeEntry::Occupied(ptr) => *ptr,
                RootNodeEntry::Free(_) => todo!(),
            };
            let new_coords = UVec3 {
                x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
            };
            CHILD::set(tree, new_coords, child_ptr, value)
        }
    }

    fn write_layout<ROOT: Node>(sizes: &mut [MaybeUninit<Layout>]) {
        CHILD::write_layout::<ROOT>(sizes);
    }
}

impl<CHILD: Node> RootNode<CHILD> {
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::with_hasher(nohash::BuildNoHashHasher::<u64>::default()),
            _marker: PhantomData,
        }
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
    ($e: tt) => {
        dust_vdb::LeafNode<{glam::UVec3{x:$e,y:$e,z:$e}}>
    };
    (#, $($n:tt),+) => {
        dust_vdb::RootNode<hierarchy!($($n),*)>
    };
    ($e: tt, $($n:tt),+) => {
        dust_vdb::InternalNode::<hierarchy!($($n),*), {glam::UVec3{x:$e,y:$e,z:$e}}>
    };
}
