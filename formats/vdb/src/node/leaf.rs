use super::size_of_grid;
use crate::{BitMask, Node, Tree};
use glam::UVec3;
use std::{
    alloc::Layout,
    mem::{size_of, MaybeUninit},
};

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
