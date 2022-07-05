use super::size_of_grid;
use crate::{BitMask, Node, Tree};
use glam::UVec3;
use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{size_of, MaybeUninit},
};

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
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// This is 0 if that tile is completely air, and 1 otherwise.
    pub child_mask: BitMask<{ size_of_grid(FANOUT_LOG2) }>,
    /// points to self.child_mask.count_ones() LeafNodes or InternalNodes
    pub child_ptrs: [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
    _marker: PhantomData<CHILD>,
}
impl<CHILD: Node, const FANOUT_LOG2: UVec3> Default for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
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
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
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
                    let node = tree.get_node_mut::<Self>(ptr);
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
