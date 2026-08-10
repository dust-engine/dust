use super::{NodeMeta, size_of_grid};
use crate::{ConstUVec3, Node, NodeConst, pool::Pool};
use bitvec::array::BitArray;
use glam::UVec3;
use std::{marker::PhantomData, mem::MaybeUninit};

#[derive(Clone, Copy)]
pub union InternalNodeEntry {
    /// The corresponding bit on child_mask is set. Points to another node.
    pub occupied: u32,
    /// The corresponding bit on child_mask is not set.
    /// Points to a value in the material array that describes all child nodes within the current node.
    /// If this is u32::MAX, this is air.
    pub free: u32,
}

/// Internal nodes can be 2*2*2.
/// Size: 8 byte (mask) + 32 byte + 16 bytes for stats
#[repr(C)]
pub struct InternalNode<CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    /// This is 0 if that tile is completely air, and 1 otherwise.
    pub child_mask: BitArray<[usize; size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]>,

    /// points to self.child_mask.count_ones() LeafNodes or InternalNodes
    pub child_ptrs: [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],

    _marker: PhantomData<CHILD>,
}
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Clone for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    fn clone(&self) -> Self {
        Self {
            child_mask: self.child_mask.clone(),
            child_ptrs: self.child_ptrs.clone(),
            _marker: PhantomData,
        }
    }
}
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Default for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    fn default() -> Self {
        Self {
            child_mask: Default::default(),
            child_ptrs: [InternalNodeEntry { free: u32::MAX }; size_of_grid(FANOUT_LOG2)],
            _marker: Default::default(),
        }
    }
}
const impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> NodeConst
    for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    type LeafType = CHILD::LeafType;
    fn write_meta(metas: &mut [MaybeUninit<NodeMeta<Self::LeafType>>]) {
        let (child, this) = metas.split_at_mut(metas.len() - 1);
        CHILD::write_meta(child);

        this[0].write(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            setter: Self::set_in_pools,
            getter: Self::get_in_pools,
            clearer: Self::clear_in_pools,
            extent_mask: Self::EXTENT_MASK,
            fanout_log2: FANOUT_LOG2,
            child_extent: CHILD::EXTENT,
            mask_offset: std::mem::offset_of!(Self, child_mask.data) as u32,
            mask_words: size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize) as u32,
            child_ptrs_offset: std::mem::offset_of!(Self, child_ptrs) as u32,
        });
    }
}
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Node for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
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
    const META_MASK: UVec3 = UVec3 {
        x: CHILD::META_MASK.x | (1 << (Self::EXTENT_LOG2.x - 1)),
        y: CHILD::META_MASK.y | (1 << (Self::EXTENT_LOG2.y - 1)),
        z: CHILD::META_MASK.z | (1 << (Self::EXTENT_LOG2.z - 1)),
    };
    const LEVEL: usize = CHILD::LEVEL + 1;

    fn set<'a>(
        &'a mut self,
        pools: &'a mut [Pool],
        coords: UVec3,
        cached_path: &mut [u32],
        moved: &mut bool,
    ) -> &'a mut Self::LeafType {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        let has_child = *self.child_mask.get(index).unwrap();
        if !has_child {
            unsafe {
                // ensure have children
                let allocated_child_ptr = pools[CHILD::LEVEL].alloc::<CHILD>();
                self.child_mask.set(index, true);

                // allocate a child node
                self.child_ptrs[index].occupied = allocated_child_ptr;
            }
        }
        // TODO: propagate when filled.
        let new_coords = coords & CHILD::EXTENT_MASK;
        let child_ptr = unsafe { &mut self.child_ptrs[index].occupied };
        <CHILD as Node>::set_in_pools(pools, new_coords, child_ptr, cached_path, moved)
    }
    fn clear<'a>(
        &'a mut self,
        pools: &'a mut [Pool],
        coords: UVec3,
        cached_path: &mut [u32],
        leaf_moved: &mut bool,
    ) -> Option<&'a mut Self::LeafType> {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        let has_child = *self.child_mask.get(index).unwrap();
        if !has_child {
            // Clearing inside a cell that is already air: nothing to do.
            return None;
        }
        let old_child = unsafe { self.child_ptrs[index].occupied };
        let mut child_ptr = old_child;
        let new_coords = coords & CHILD::EXTENT_MASK;
        let leaf = <CHILD as Node>::clear_in_pools(
            pools,
            new_coords,
            &mut child_ptr,
            cached_path,
            leaf_moved,
            false,
        )
        .map(|leaf| leaf as *mut Self::LeafType);
        if child_ptr != old_child {
            // The child forked (it was shared with a snapshot): redirect the
            // root's edge and drop the old version's. The root is owned by
            // the tree, so it is always safe to update in place.
            self.child_ptrs[index].occupied = child_ptr;
            let dead = pools[CHILD::LEVEL].release(old_child);
            debug_assert!(!dead);
        }
        leaf.map(|leaf| unsafe { &mut *leaf })
    }
    #[inline]
    fn set_in_pools<'a>(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        cached_path: &mut [u32],
        moved: &mut bool,
    ) -> &'a mut Self::LeafType {
        unsafe {
            // Copy-on-write: if this node is shared with a snapshot, redirect
            // the parent's edge to a private copy before mutating anything at
            // or below it. The children become referenced by both the original
            // (still visible to snapshots) and the copy.
            if pools[Self::LEVEL].is_shared(*ptr) {
                let copy = pools[Self::LEVEL].copy_item::<Self>(*ptr);
                let dead = pools[Self::LEVEL].release(*ptr);
                debug_assert!(!dead);
                let copied_node = pools[Self::LEVEL].get(copy) as *const Self;
                (*copied_node).retain_children(pools);
                *ptr = copy;
            }
            let node: *mut Self = pools[Self::LEVEL].get_item_mut::<Self>(*ptr);

            let internal_offset = coords >> CHILD::EXTENT_LOG2;
            let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
                | ((internal_offset.y as usize) << FANOUT_LOG2.z)
                | (internal_offset.z as usize);
            // set
            let has_child = *(&mut *node).child_mask.get(index).unwrap();
            if !has_child {
                // ensure have children
                let allocated_child_ptr = pools[CHILD::LEVEL].alloc::<CHILD>();
                (&mut *node).child_mask.set(index, true);
                (&mut *node).child_ptrs[index].occupied = allocated_child_ptr;
            }
            // TODO: propagate when filled

            if cached_path.len() > 0 {
                cached_path[Self::LEVEL] = *ptr;
            }
            let new_coords = coords & CHILD::EXTENT_MASK;
            let child_ptr = &mut (&mut *node).child_ptrs[index].occupied;
            <CHILD as Node>::set_in_pools(pools, new_coords, child_ptr, cached_path, moved)
        }
    }

    fn clear_in_pools<'a>(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        cached_path: &mut [u32],
        leaf_moved: &mut bool,
        ancestor_shared: bool,
    ) -> Option<&'a mut Self::LeafType> {
        unsafe {
            let internal_offset = coords >> CHILD::EXTENT_LOG2;
            let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
                | ((internal_offset.y as usize) << FANOUT_LOG2.z)
                | (internal_offset.z as usize);
            // Clearing inside a cell that is already air: nothing to do —
            let node = pools[Self::LEVEL].get_item::<Self>(*ptr);
            if !*node.child_mask.get(index).unwrap() {
                return None;
            }
            // Sharing recorded at an ancestor freezes this node too, even
            // while its own count still reads unique (refcounts are pushed
            // down lazily, by the forks themselves).
            let shared = ancestor_shared || pools[Self::LEVEL].is_shared(*ptr);
            let old_child = node.child_ptrs[index].occupied;
            // Descend on a local copy of the edge: this node may be frozen,
            // so nothing below may write through its storage. Forks are
            // recorded on the way back up, only once the voxel is found — a
            // clear that misses touches nothing.
            let mut child_ptr = old_child;
            let new_coords = coords & CHILD::EXTENT_MASK;
            let leaf = <CHILD as Node>::clear_in_pools(
                &mut *pools,
                new_coords,
                &mut child_ptr,
                cached_path,
                leaf_moved,
                shared,
            )
            .map(|leaf| leaf as *mut Self::LeafType);
            let Some(leaf) = leaf else {
                // No-op below: nothing was forked, nothing to record.
                debug_assert_eq!(child_ptr, old_child);
                return None;
            };
            if child_ptr != old_child {
                // The child forked; record the new edge. A frozen node forks
                // itself first: the copy aliases all children, so retaining
                // them and then releasing the replaced one nets the old
                // child's count back to unchanged. The pre-fork version's own
                // edge stays with our parent — it releases it when it records
                // `ptr`, exactly as we do for our child here.
                if shared {
                    let copy = pools[Self::LEVEL].copy_item::<Self>(*ptr);
                    let copied_node = pools[Self::LEVEL].get(copy) as *const Self;
                    (*copied_node).retain_children(pools);
                    let copied_node = pools[Self::LEVEL].get_item_mut::<Self>(copy);
                    copied_node.child_ptrs[index].occupied = child_ptr;
                    *ptr = copy;
                } else {
                    let node = pools[Self::LEVEL].get_item_mut::<Self>(*ptr);
                    node.child_ptrs[index].occupied = child_ptr;
                }
                let dead = pools[CHILD::LEVEL].release(old_child);
                debug_assert!(!dead);
            }
            if cached_path.len() > 0 {
                cached_path[Self::LEVEL] = *ptr;
            }
            Some(&mut *leaf)
        }
    }

    fn collapse(&mut self, pools: &mut [Pool], coords: UVec3) {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        debug_assert!(
            *self.child_mask.get(index).unwrap(),
            "collapse must target an existing path"
        );
        let new_coords = coords & CHILD::EXTENT_MASK;
        let child_ptr = unsafe { &mut self.child_ptrs[index].occupied };
        <CHILD as Node>::collapse_in_pools(pools, new_coords, child_ptr);
        if unsafe { self.child_ptrs[index].occupied } == u32::MAX {
            // The child emptied and freed itself: detach the cell. The root
            // node itself is owned by the tree and is never freed.
            self.child_mask.set(index, false);
        }
    }

    fn collapse_in_pools(pools: &mut [Pool], coords: UVec3, ptr: &mut u32) {
        unsafe {
            let internal_offset = coords >> CHILD::EXTENT_LOG2;
            let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
                | ((internal_offset.y as usize) << FANOUT_LOG2.z)
                | (internal_offset.z as usize);
            // The cascade mutates parents in place, which is only sound on
            // uniquely owned nodes reached through real parent edges — both
            // guaranteed by the caller (see the trait docs).
            debug_assert!(!pools[Self::LEVEL].is_shared(*ptr));
            let node: *mut Self = pools[Self::LEVEL].get_item_mut::<Self>(*ptr);
            debug_assert!(
                *(*node).child_mask.get(index).unwrap(),
                "collapse must target an existing path"
            );
            let new_coords = coords & CHILD::EXTENT_MASK;
            let child_ptr = &mut (*node).child_ptrs[index].occupied;
            <CHILD as Node>::collapse_in_pools(&mut *pools, new_coords, child_ptr);
            if (*node).child_ptrs[index].occupied == u32::MAX {
                // The child emptied and freed itself: detach the cell.
                (*node).child_mask.set(index, false);
                if (*node).child_mask.not_any() {
                    // Nothing left below this node either (a clear mask means
                    // every cell is air — constant tiles don't exist yet).
                    // Free it and report through the parent's edge in turn.
                    pools[Self::LEVEL].free(*ptr);
                    *ptr = u32::MAX;
                }
            }
        }
    }

    fn retain_children(&self, pools: &mut [Pool]) {
        for index in self.child_mask.iter_ones() {
            let child_ptr = unsafe { self.child_ptrs[index].occupied };
            pools[CHILD::LEVEL].retain(child_ptr);
        }
    }

    fn release_in_pools(
        pools: &mut [Pool],
        ptr: u32,
        leaf_dropped: &mut dyn FnMut(&Self::LeafType),
    ) {
        if pools[Self::LEVEL].release(ptr) {
            // Last parent edge gone: the node dies, and each of its children
            // loses one edge in turn. Releasing never allocates, so the raw
            // pointer into this pool stays valid across the recursion.
            unsafe {
                let node = pools[Self::LEVEL].get(ptr) as *const Self;
                (*node).release_children(pools, leaf_dropped);
            }
            pools[Self::LEVEL].free(ptr);
        }
    }

    fn release_children(&self, pools: &mut [Pool], leaf_dropped: &mut dyn FnMut(&Self::LeafType)) {
        for index in self.child_mask.iter_ones() {
            let child_ptr = unsafe { self.child_ptrs[index].occupied };
            <CHILD as Node>::release_in_pools(pools, child_ptr, leaf_dropped);
        }
    }
    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was owned.
    /// Implementation will write to cached_path for all levels below the current level.
    fn get<'a>(
        &'a self,
        pools: &'a [Pool],
        coords: UVec3,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType> {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        let has_child = *self.child_mask.get(index).unwrap();
        if !has_child {
            // The descent ends here, leaving the lower cache levels holding
            // whatever an older descent wrote — a later re-entry there would
            // read a stale node. Poison them with u32::MAX, which re-entry
            // interprets as "this whole region is known air".
            let stale_levels = Self::LEVEL.min(cached_path.len());
            cached_path[..stale_levels].fill(u32::MAX);
            return None;
        }
        let new_coords = coords & CHILD::EXTENT_MASK;
        let child_ptr = unsafe { self.child_ptrs[index].occupied };
        <CHILD as Node>::get_in_pools(pools, new_coords, child_ptr, cached_path)
    }

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    fn get_in_pools<'a>(
        pools: &'a [Pool],
        coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType> {
        unsafe {
            let node = pools[Self::LEVEL].get_item::<Self>(ptr);
            if cached_path.len() > 0 {
                cached_path[Self::LEVEL] = ptr;
            }
            node.get(pools, coords, cached_path)
        }
    }

    type LeafIterator<'a> = InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>;

    #[inline]
    fn iter_leaf<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a> {
        InternalNodeLeafIterator::new(pools, self, offset)
    }

    #[inline]
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a> {
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        InternalNodeLeafIterator::new(pools, node, offset)
    }
    fn count_leaves(&self, pools: &[Pool]) -> usize {
        if Self::LEVEL == 1 {
            // We're one level above leaves.
            self.child_mask.count_ones()
        } else {
            self.child_mask
                .iter_ones()
                .map(|i| {
                    let child_ptr = unsafe { self.child_ptrs[i].occupied };
                    let child = unsafe { pools[Self::LEVEL - 1].get_item::<CHILD>(child_ptr) };
                    return child.count_leaves(pools);
                })
                .sum()
        }
    }
}

/// When the alternate flag was specified, also print the child pointers.
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> std::fmt::Debug
    for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Internal Node\n")?;
        self.child_mask.fmt(f)?;
        Ok(())
    }
}

pub struct InternalNodeLeafIterator<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    pools: &'a [Pool],
    location_offset: UVec3,
    /// Raw words of the child mask, scanned directly with `trailing_zeros`
    /// (`word` caches the current word's unvisited bits) instead of through
    /// `bitvec`'s `IterOnes`, whose `BitSlice` region machinery dominated the
    /// per-leaf cost.
    mask_words: &'a [usize; size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)],
    word: usize,
    word_idx: u32,
    child_iterator: Option<CHILD::LeafIterator<'a>>,
    child_ptrs: &'a [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
}
impl<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3>
    InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    fn new(
        pools: &'a [Pool],
        node: &'a InternalNode<CHILD, FANOUT_LOG2>,
        location_offset: UVec3,
    ) -> Self {
        Self {
            pools,
            location_offset,
            mask_words: &node.child_mask.data,
            word: node.child_mask.data[0],
            word_idx: 0,
            child_iterator: None,
            child_ptrs: &node.child_ptrs,
        }
    }
}
impl<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3> Iterator
    for InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2).div_ceil(usize::BITS as usize)]: Sized,
{
    type Item = (UVec3, &'a CHILD::LeafType);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            let next_child_index = loop {
                if self.word != 0 {
                    let bit = self.word.trailing_zeros();
                    self.word &= self.word - 1;
                    break Some(self.word_idx as usize * usize::BITS as usize + bit as usize);
                }
                self.word_idx += 1;
                if self.word_idx as usize >= self.mask_words.len() {
                    break None;
                }
                self.word = self.mask_words[self.word_idx as usize];
            };
            if let Some(next_child_index) = next_child_index {
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
