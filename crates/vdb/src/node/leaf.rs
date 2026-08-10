use super::{NodeMeta, size_of_grid};
use crate::{ConstUVec3, Node, NodeConst, pool::Pool};
use bitvec::{
    array::BitArray,
    order::Lsb0,
    slice::{BitSlice, IterOnes},
};
use glam::UVec3;
use std::{
    iter::Once,
    mem::{MaybeUninit, size_of},
    ops::DerefMut,
};

/// Nodes are always 4x4x4 so that each leaf node contains exactly 64 voxels,
/// so that the occupancy mask happens to be exactly 64 bits.
/// Size: 3 u32
#[repr(C)]
#[derive(Default, Clone)]
pub struct LeafNode<const LOG2: ConstUVec3, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// This is 1 for occupied voxels and 0 for unoccupied voxels
    pub occupancy: BitArray<[usize; size_of_grid(LOG2) / size_of::<usize>() / 8]>,
    /// A pointer to self.occupancy.count_ones() material values
    pub value: T,
}

pub trait IsLeaf: Node {
    /// Total number of voxels in the leaf node.
    type Occupancy: DerefMut<Target = BitSlice<usize, Lsb0>> + Clone;
    type Value: Default + Send + Sync + Clone;
    fn get_occupancy(&self) -> &Self::Occupancy;
    fn get_occupancy_mut(&mut self) -> &mut Self::Occupancy;

    fn get_occupancy_at(&self, coords: UVec3) -> bool {
        *self
            .get_occupancy()
            .get(Self::get_inflated_attribute_offset(coords) as usize)
            .expect("get_occupancy_at: coords out of bounds")
    }
    fn set_occupancy_at(&mut self, coords: UVec3, value: bool) {
        let offset = Self::get_inflated_attribute_offset(coords);
        self.get_occupancy_mut().set(offset as usize, value);
    }

    fn get_value(&self) -> &Self::Value;
    fn set_value(&mut self, value: Self::Value);

    fn get_fitted_attribute_offset(&self, coords: UVec3) -> u32;
    fn get_inflated_attribute_offset(coords: UVec3) -> u32;

    type Iterator<'a>: Iterator<Item = UVec3> where Self: 'a;
    fn iter<'a>(&'a self, offset: UVec3) -> Self::Iterator<'a>;
}

impl<const LOG2: ConstUVec3, T: Clone + Send + Sync + 'static + Default> IsLeaf
    for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Value = T;
    type Occupancy = BitArray<[usize; size_of_grid(LOG2) / size_of::<usize>() / 8]>;
    fn get_fitted_attribute_offset(&self, coords: UVec3) -> u32 {
        let coords = coords & Self::EXTENT_MASK;
        let voxel_id = (coords.x << (LOG2.y + LOG2.z)) | (coords.y << LOG2.z) | coords.z;
        debug_assert!(
            self.occupancy.as_raw_slice().len() == 1,
            "Supports up to 64 voxels per leaf node for now"
        );
        let mask: usize = self.occupancy.as_raw_slice()[0];
        let masked = mask & ((1 << voxel_id) - 1);
        masked.count_ones()
    }

    fn get_inflated_attribute_offset(coords: UVec3) -> u32 {
        let coords = coords & Self::EXTENT_MASK;
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        index as u32
    }

    fn get_value(&self) -> &Self::Value {
        &self.value
    }
    fn set_value(&mut self, value: Self::Value) {
        self.value = value;
    }
    fn get_occupancy(&self) -> &Self::Occupancy {
        &self.occupancy
    }
    fn get_occupancy_mut(&mut self) -> &mut Self::Occupancy {
        &mut self.occupancy
    }

    type Iterator<'a> = LeafNodeIterator<'a, LOG2>;
    fn iter<'a>(&'a self, offset: UVec3) -> Self::Iterator<'a> {
        LeafNodeIterator {
            location_offset: offset,
            bits_iterator: self.occupancy.iter_ones(),
        }
    }
}

const impl<const LOG2: ConstUVec3, T: Clone + Send + Sync + 'static + Default> NodeConst
    for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    type LeafType = Self;
    fn write_meta(metas: &mut [MaybeUninit<NodeMeta<Self>>]) {
        metas[0].write(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            extent_mask: Self::EXTENT_MASK,
            setter: Self::set_in_pools,
            getter: Self::get_in_pools,
            clearer: Self::clear_in_pools,
            fanout_log2: LOG2,
            child_extent: UVec3::ZERO,
            mask_offset: std::mem::offset_of!(Self, occupancy.data) as u32,
            mask_words: (size_of_grid(LOG2) / size_of::<usize>() / 8) as u32,
            child_ptrs_offset: 0,
        });
    }
}

impl<const LOG2: ConstUVec3, T: Clone + Send + Sync + 'static + Default> Node for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// Total number of voxels contained within the leaf node.
    const SIZE: usize = size_of_grid(LOG2);
    /// Extent of the leaf node in each axis.
    const EXTENT_LOG2: UVec3 = LOG2.to_glam();
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
    const META_MASK: UVec3 = UVec3 {
        x: 1 << (LOG2.x - 1),
        y: 1 << (LOG2.y - 1),
        z: 1 << (LOG2.z - 1),
    };
    const LEVEL: usize = 0;

    fn set<'a>(
        &'a mut self,
        _pools: &'a mut [Pool],
        _coords: UVec3,
        _cached_path: &mut [u32],
        _moved: &mut bool,
    ) -> &'a mut Self::LeafType {
        // Only ever called if the leaf is the root. Like every descent, this
        // never flips occupancy bits — the caller sets the bit on the
        // returned leaf.
        self
    }

    fn clear<'a>(
        &'a mut self,
        _pools: &'a mut [Pool],
        coords: UVec3,
        _cached_path: &mut [u32],
        _moved: &mut bool,
    ) -> Option<&'a mut Self::LeafType> {
        // Only ever called if the leaf is the root. Like every descent, this
        // never flips occupancy bits — the caller clears the bit on the
        // returned leaf. A missing voxel is a no-op.
        if !self.get_occupancy_at(coords) {
            return None;
        }
        Some(self)
    }
    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was owned.
    /// Implementation will write to cached_path for all levels below the current level.
    fn get<'a>(
        &'a self,
        _pools: &'a [Pool],
        _coords: UVec3,
        _cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType> {
        Some(self)
    }

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    fn get_in_pools<'a>(
        pools: &'a [Pool],
        _coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType> {
        if cached_path.len() > 0 {
            cached_path[Self::LEVEL] = ptr;
        }
        Some(unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) })
    }

    #[inline]
    fn set_in_pools<'a>(
        pools: &'a mut [Pool],
        _coords: UVec3,
        ptr: &mut u32,
        cached_path: &mut [u32],
        leaf_moved: &mut bool,
    ) -> &'a mut Self {
        // Copy-on-write: a leaf shared with a snapshot is frozen. Redirect the
        // parent's edge to a private copy and leave the original untouched.
        // The copy still references the original's attribute range;
        // `leaf_moved` tells the caller to re-home the attributes.
        if pools[Self::LEVEL].is_shared(*ptr) {
            let copy = unsafe { pools[Self::LEVEL].copy_item::<Self>(*ptr) };
            let dead = pools[Self::LEVEL].release(*ptr);
            debug_assert!(!dead);
            *ptr = copy;
            *leaf_moved = true;
        }
        let old_leaf_node: *mut Self = unsafe { pools[Self::LEVEL].get_item_mut::<Self>(*ptr) };
        if cached_path.len() > 0 {
            cached_path[0] = *ptr;
        }
        // The caller sets the occupancy bit on the returned leaf.
        unsafe { &mut *old_leaf_node }
    }

    fn clear_in_pools<'a>(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        cached_path: &mut [u32],
        leaf_moved: &mut bool,
        ancestor_shared: bool,
    ) -> Option<&'a mut Self::LeafType> {
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(*ptr) };
        if !node.get_occupancy_at(coords) {
            // The voxel was never set (or the leaf is already fully empty
            // and awaiting collapse): a no-op. Decided before the fork
            // below, so a no-op clear never copies anything anywhere — the
            // ancestors only fork on the way back up, after this point.
            return None;
        }
        // Fork: a frozen leaf must not be mutated in place. Sharing recorded
        // at an ancestor freezes this leaf too, even while its own count
        // still reads unique. The copy still references the original's
        // attribute range; `leaf_moved` tells the caller to re-home it. The
        // old version's edge belongs to our parent — it releases it when it
        // records the fork on its way back up.
        if ancestor_shared || pools[Self::LEVEL].is_shared(*ptr) {
            let copy = unsafe { pools[Self::LEVEL].copy_item::<Self>(*ptr) };
            *ptr = copy;
            *leaf_moved = true;
        }
        let old_leaf_node: *mut Self = unsafe { pools[Self::LEVEL].get_item_mut::<Self>(*ptr) };
        if cached_path.len() > 0 {
            cached_path[0] = *ptr;
        }
        // The caller flips the occupancy bit on the returned leaf.
        Some(unsafe { &mut *old_leaf_node })
    }

    fn collapse(&mut self, _pools: &mut [Pool], _coords: UVec3) {
        // Only ever called if the leaf is the root, which is owned by the
        // tree: there is nothing to free.
    }

    fn collapse_in_pools(pools: &mut [Pool], _coords: UVec3, ptr: &mut u32) {
        debug_assert!(
            unsafe { pools[Self::LEVEL].get_item::<Self>(*ptr) }
                .occupancy
                .not_any(),
            "only a fully erased leaf may be collapsed"
        );
        // Drop the working tree's edge; the leaf dies with it unless a
        // snapshot still references it.
        Self::release_in_pools(pools, *ptr, &mut |_| {});
        // Tell the parent the cell is air now; u32::MAX is the air marker
        // of `InternalNodeEntry::free`.
        *ptr = u32::MAX;
    }

    fn retain_children(&self, _pools: &mut [Pool]) {
        // Leaves have no children.
    }

    fn release_in_pools(pools: &mut [Pool], ptr: u32, leaf_dropped: &mut dyn FnMut(&Self)) {
        if pools[Self::LEVEL].release(ptr) {
            unsafe {
                let node = pools[Self::LEVEL].get(ptr) as *const Self;
                leaf_dropped(&*node);
            }
            pools[Self::LEVEL].free(ptr);
        }
    }

    fn release_children(&self, _pools: &mut [Pool], _leaf_dropped: &mut dyn FnMut(&Self)) {
        // Leaves have no children.
    }

    type LeafIterator<'a> = Once<(UVec3, &'a Self)>;

    #[inline]
    fn iter_leaf<'a>(&'a self, _pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a> {
        std::iter::once((offset, unsafe { std::mem::transmute(self) }))
    }

    #[inline]
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a> {
        let node = unsafe { pools[0].get_item::<Self>(ptr) };
        std::iter::once((offset, unsafe { std::mem::transmute(node) }))
    }

    fn count_leaves(&self, _pools: &[Pool]) -> usize {
        debug_assert!(
            false,
            "Iteration should've been terminated one level above this"
        );
        1
    }
}

impl<const LOG2: ConstUVec3, T: Send + Sync + 'static> std::fmt::Debug for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LeafNode\n")?;
        self.occupancy.fmt(f)?;
        Ok(())
    }
}

pub struct LeafNodeIterator<'a, const LOG2: ConstUVec3>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    location_offset: UVec3,
    bits_iterator: IterOnes<'a, usize, Lsb0>,
}
impl<'a, const LOG2: ConstUVec3> Iterator for LeafNodeIterator<'a, LOG2>
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
