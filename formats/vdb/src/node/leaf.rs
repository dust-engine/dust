use super::{size_of_grid, NodeMeta};
use crate::{bitmask::SetBitIterator, BitMask, Node, NodeConst, Pool};
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
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
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
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// Total number of voxels contained within the leaf node.
    const SIZE: usize = size_of_grid(LOG2);
    /// Extent of the leaf node in each axis.
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: LOG2.x,
        y: LOG2.y,
        z: LOG2.z,
    };
    const EXTENT: UVec3 = UVec3 {
        x: 1 << LOG2.x,
        y: 1 << LOG2.y,
        z: 1 << LOG2.z,
    };
    const EXTENT_MASK: UVec3 = UVec3 {
        x: Self::EXTENT.x - 1,
        y: Self::EXTENT.y - 1,
        z: Self::EXTENT.z - 1,
    };
    const LEVEL: usize = 0;
    fn new() -> Self {
        Self {
            occupancy: BitMask::new(),
            active: BitMask::new(),
            material_ptr: 0,
        }
    }

    type Voxel = bool;

    #[inline]
    fn get(&self, _: &[Pool], coords: UVec3, _cached_path: &mut [u32]) -> Option<Self::Voxel> {
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        let occupied = self.occupancy.get(index);
        if !occupied {
            return None;
        }
        let active = self.active.get(index);
        return Some(active);
    }
    #[inline]
    fn set(
        &mut self,
        _: &mut [Pool],
        coords: UVec3,
        value: Option<Self::Voxel>,
        _cached_path: &mut [u32],
    ) {
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        if let Some(voxel) = value {
            self.occupancy.set(index, true);
            self.active.set(index, voxel);
        } else {
            self.occupancy.set(index, false);
        }
    }
    #[inline]
    fn get_in_pools(
        pools: &[Pool],
        coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<Self::Voxel> {
        if cached_path.len() > 0 {
            cached_path[0] = ptr;
        }
        let leaf_node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        leaf_node.get(&[], coords, cached_path)
    }

    #[inline]
    fn set_in_pools(
        pools: &mut [Pool],
        coords: UVec3,
        ptr: u32,
        value: Option<Self::Voxel>,
        cached_path: &mut [u32],
    ) {
        if cached_path.len() > 0 {
            cached_path[0] = ptr;
        }
        let leaf_node = unsafe { pools[Self::LEVEL].get_item_mut::<Self>(ptr) };
        leaf_node.set(&mut [], coords, value, cached_path)
    }

    type Iterator<'a> = LeafNodeIterator<'a, LOG2>;
    fn iter<'a>(&'a self, _pool: &'a [Pool], offset: UVec3) -> Self::Iterator<'a> {
        LeafNodeIterator {
            location_offset: offset,
            bits_iterator: self.occupancy.iter_set_bits(),
        }
    }
    fn iter_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::Iterator<'a> {
        let node = unsafe { pools[0].get_item::<Self>(ptr) };
        LeafNodeIterator {
            location_offset: offset,
            bits_iterator: node.occupancy.iter_set_bits(),
        }
    }
}

impl<const LOG2: UVec3> const NodeConst for LeafNode<LOG2>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn write_meta(metas: &mut [MaybeUninit<NodeMeta<Self::Voxel>>]) {
        metas[Self::LEVEL].write(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            getter: Self::get_in_pools,
            extent_log2: Self::EXTENT_LOG2,
            fanout_log2: LOG2,
            extent_mask: Self::EXTENT_MASK,
            setter: Self::set_in_pools,
        });
    }
}

impl<const LOG2: UVec3> std::fmt::Debug for LeafNode<LOG2>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LeafNode\n")?;
        self.occupancy.fmt(f)?;
        Ok(())
    }
}

pub struct LeafNodeIterator<'a, const LOG2: UVec3>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    location_offset: UVec3,
    bits_iterator: SetBitIterator<'a, { size_of_grid(LOG2) }>,
}
impl<'a, const LOG2: UVec3> Iterator for LeafNodeIterator<'a, LOG2>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.bits_iterator.next()?;

        let z = index & ((1 << LOG2.z) - 1);
        let y = (index >> LOG2.z) & ((1 << LOG2.y) - 1);
        let x = index >> (LOG2.z + LOG2.y);
        let location = UVec3::new(x as u32, y as u32, z as u32);
        Some(location + self.location_offset)
    }
}
