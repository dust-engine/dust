use super::{size_of_grid, NodeMeta};
use crate::{bitmask::SetBitIterator, BitMask, Node, NodeConst, Pool, ConstUVec3};
use glam::UVec3;
use std::{
    cell::UnsafeCell,
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
pub struct InternalNode<CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// This is 0 if that tile is completely air, and 1 otherwise.
    pub child_mask: BitMask<{ size_of_grid(FANOUT_LOG2) }>,
    /// points to self.child_mask.count_ones() LeafNodes or InternalNodes
    pub child_ptrs: [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
    _marker: PhantomData<CHILD>,
}
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Default for InternalNode<CHILD, FANOUT_LOG2>
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
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Node for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    type LeafType = CHILD::LeafType;
    const SIZE: usize = size_of_grid(FANOUT_LOG2);
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: FANOUT_LOG2.x + CHILD::EXTENT_LOG2.x,
        y: FANOUT_LOG2.y + CHILD::EXTENT_LOG2.y,
        z: FANOUT_LOG2.z + CHILD::EXTENT_LOG2.z,
    };
    const EXTENT: UVec3 = UVec3 {
        x: 1 << Self::EXTENT_LOG2.x,
        y: 1 << Self::EXTENT_LOG2.y,
        z: 1 << Self::EXTENT_LOG2.z,
    };
    const EXTENT_MASK: UVec3 = UVec3 {
        x: Self::EXTENT.x - 1,
        y: Self::EXTENT.y - 1,
        z: Self::EXTENT.z - 1,
    };
    const LEVEL: usize = CHILD::LEVEL + 1;
    fn new() -> Self {
        Self {
            child_mask: BitMask::new(),
            child_ptrs: [InternalNodeEntry { free: 0 }; size_of_grid(FANOUT_LOG2)],
            _marker: PhantomData,
        }
    }

    type Voxel = CHILD::Voxel;
    #[inline]
    fn get(&self, pools: &[Pool], coords: UVec3, cached_path: &mut [u32]) -> Option<Self::Voxel> {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        let has_child = self.child_mask.get(index);
        if !has_child {
            return None;
        }
        unsafe {
            let child_ptr = self.child_ptrs[index].occupied;
            let new_coords = UVec3 {
                x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
            };
            <CHILD as Node>::get_in_pools(pools, new_coords, child_ptr, cached_path)
        }
    }
    #[inline]
    fn set(
        &mut self,
        pools: &mut [Pool],
        coords: UVec3,
        value: Option<Self::Voxel>,
        cached_path: &mut [u32],
    ) {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        if value.is_some() {
            // set
            let has_child = self.child_mask.get(index);
            if !has_child {
                // ensure have children
                self.child_mask.set(index, true);
                unsafe {
                    // allocate a child node
                    let allocated_ptr = pools[CHILD::LEVEL].alloc::<CHILD>();
                    self.child_ptrs[index].occupied = allocated_ptr;
                }
            }
            // TODO: propagate when filled.
        } else {
            // clear
            todo!() // TODO: clear recursively, propagate if completely cleared
        }
        unsafe {
            let new_coords = coords & CHILD::EXTENT_MASK;
            let child_ptr = self.child_ptrs[index].occupied;
            <CHILD as Node>::set_in_pools(pools, new_coords, child_ptr, value, cached_path)
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
            cached_path[Self::LEVEL] = ptr;
        }
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        node.get(pools, coords, cached_path)
    }

    #[inline]
    fn set_in_pools(
        pools: &mut [Pool],
        coords: UVec3,
        ptr: u32,
        value: Option<Self::Voxel>,
        cached_path: &mut [u32],
    ) {
        // Safety: r was taken from pools[Self::LEVEL] and we know that self.set only access pools[CHILD::LEVEL].
        if cached_path.len() > 0 {
            cached_path[Self::LEVEL] = ptr;
        }
        unsafe {
            let r = pools[Self::LEVEL].get_item_mut::<Self>(ptr) as *mut Self;
            (*r).set(pools, coords, value, cached_path)
        }
    }

    type Iterator<'a> = InternalNodeIterator<'a, CHILD, FANOUT_LOG2>;
    #[inline]
    fn iter<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::Iterator<'a> {
        InternalNodeIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: self.child_mask.iter_set_bits(),
            child_ptrs: &self.child_ptrs,
            child_iterator: None,
        }
    }
    #[inline]
    fn iter_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::Iterator<'a> {
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        InternalNodeIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: node.child_mask.iter_set_bits(),
            child_ptrs: &node.child_ptrs,
            child_iterator: None,
        }
    }

    type LeafIterator<'a> = InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>;

    #[inline]
    fn iter_leaf<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a> {
        InternalNodeLeafIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: self.child_mask.iter_set_bits(),
            child_ptrs: &self.child_ptrs,
            child_iterator: None,
        }
    }

    #[inline]
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a> {
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        InternalNodeLeafIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: node.child_mask.iter_set_bits(),
            child_ptrs: &node.child_ptrs,
            child_iterator: None,
        }
    }
}

impl<CHILD: ~const NodeConst + Node, const FANOUT_LOG2: ConstUVec3> const NodeConst
    for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn write_meta(metas: &mut [MaybeUninit<NodeMeta<Self::Voxel>>]) {
        metas[Self::LEVEL as usize].write(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            getter: Self::get_in_pools,
            setter: Self::set_in_pools,
            extent_log2: Self::EXTENT_LOG2,
            extent_mask: Self::EXTENT_MASK,
            fanout_log2: FANOUT_LOG2.to_glam(),
        });
        CHILD::write_meta(metas);
    }
}

/// When the alternate flag was specified, also print the child pointers.
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> std::fmt::Debug for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Internal Node\n")?;
        self.child_mask.fmt(f)?;
        Ok(())
    }
}

pub struct InternalNodeIterator<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    pools: &'a [Pool],
    location_offset: UVec3,
    child_mask_iterator: SetBitIterator<'a, { size_of_grid(FANOUT_LOG2) }>,
    child_iterator: Option<CHILD::Iterator<'a>>,
    child_ptrs: &'a [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
}
impl<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3> Iterator
    for InternalNodeIterator<'a, CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            if let Some(next_child_index) = self.child_mask_iterator.next() {
                let child_ptr = unsafe { self.child_ptrs[next_child_index].occupied };
                let offset = UVec3 {
                    x: next_child_index as u32 >> (FANOUT_LOG2.z + FANOUT_LOG2.y),
                    y: (next_child_index as u32 >> FANOUT_LOG2.z) & ((1 << FANOUT_LOG2.y) - 1),
                    z: next_child_index as u32 & ((1 << FANOUT_LOG2.z) - 1),
                };
                let offset = offset * CHILD::EXTENT;
                self.child_iterator = Some(CHILD::iter_in_pool(
                    self.pools,
                    child_ptr,
                    self.location_offset + offset,
                ));
                continue;
            } else {
                // Also ran out. We have nothing left.
                return None;
            }
        }
    }
}

pub struct InternalNodeLeafIterator<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    pools: &'a [Pool],
    location_offset: UVec3,
    child_mask_iterator: SetBitIterator<'a, { size_of_grid(FANOUT_LOG2) }>,
    child_iterator: Option<CHILD::LeafIterator<'a>>,
    child_ptrs: &'a [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
}
impl<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3> Iterator
    for InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Item = (UVec3, &'a UnsafeCell<CHILD::LeafType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            if let Some(next_child_index) = self.child_mask_iterator.next() {
                let child_ptr = unsafe { self.child_ptrs[next_child_index].occupied };
                let offset = UVec3 {
                    x: next_child_index as u32 >> (FANOUT_LOG2.z + FANOUT_LOG2.y),
                    y: (next_child_index as u32 >> FANOUT_LOG2.z) & ((1 << FANOUT_LOG2.y) - 1),
                    z: next_child_index as u32 & ((1 << FANOUT_LOG2.z) - 1),
                };
                let offset = offset * CHILD::EXTENT;
                self.child_iterator = Some(CHILD::iter_leaf_in_pool(
                    self.pools,
                    child_ptr,
                    self.location_offset + offset,
                ));
                continue;
            } else {
                // Also ran out. We have nothing left.
                return None;
            }
        }
    }
}
