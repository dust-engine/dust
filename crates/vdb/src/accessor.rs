use crate::{Attributes, IsDefault, IsLeaf, Node, Tree, TreeSnapshot, pool::Pool};
use glam::UVec3;
use std::ops::Deref;

/// Accessors are designed to help accelerate accesses into the tree structures by storing caches
/// to tree branches. When traversing a grid in a spatially coherent pattern, the same branches
/// and nodes of the underlying tree can be hit. Accessors cache the path down to the leaf node so
/// that subsequent neighboring accesses can skip traversing the upper levels of the tree and go
/// directly to the leaf node.
pub struct AccessorMut<'a, ROOT: Node, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
    ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
{
    tree: &'a mut Tree<ROOT>,
    /// Cached path refreshed by reads. It may reference nodes that are shared
    /// with a snapshot, so it must never be used to re-enter the tree for a
    /// write.
    ptrs: [u32; ROOT::LEVEL],
    last_coords: UVec3,
    /// Cached path established exclusively by writes. Copy-on-write uniquifies
    /// every node on a write's descent, so each entry here is uniquely owned
    /// by the working tree and writes may re-enter the tree at any level of
    /// this path.
    set_ptrs: [u32; ROOT::LEVEL],
    last_set_coords: UVec3,
    attributes: &'a mut ATTRIBS,
    last_leaf: Option<u32>,
    last_leaf_coords: UVec3,
}

#[inline]
fn lowest_common_ancestor_level(a: UVec3, b: UVec3, mask: UVec3, root_level: u32) -> u32 {
    let diff = a ^ b;
    // instead, we should get the highest different bit here.
    let last_set_bit = UVec3 {
        x: 1 << (31 - diff.x.leading_zeros().min(31)),
        y: 1 << (31 - diff.y.leading_zeros().min(31)),
        z: 1 << (31 - diff.z.leading_zeros().min(31)),
    };
    let result = mask & !(last_set_bit - 1);
    let parent_index = result
        .x
        .count_ones()
        .min(result.y.count_ones())
        .min(result.z.count_ones());
    root_level + 1 - parent_index
}

impl<'a, ROOT: Node, ATTRIBS> AccessorMut<'a, ROOT, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
    ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
{
    pub fn get(&mut self, coords: UVec3) -> Option<ATTRIBS::Value> {
        // Fast path: reading inside the hot leaf — no descent. The hot
        // leaf's attribute range is inflated (one slot per voxel), so the
        // lookup goes through the fully mapped offset.
        if let Some(last_leaf) = self.last_leaf
            && ((coords ^ self.last_leaf_coords) & !<ROOT::LeafType as Node>::EXTENT_MASK)
                == UVec3::ZERO
        {
            let leaf_node = unsafe { self.tree.get_node::<ROOT::LeafType>(last_leaf) };
            if !leaf_node.get_occupancy_at(coords) {
                return None;
            }
            let attribute = self.attributes.get_attribute(
                leaf_node.get_value(),
                <ROOT::LeafType as IsLeaf>::get_inflated_attribute_offset(coords),
            );
            return Some(attribute);
        }

        let lca_level = lowest_common_ancestor_level(
            self.last_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_coords = coords;
        let leaf_node = if lca_level >= ROOT::LEVEL as u32 {
            self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
        } else {
            let meta = &ROOT::META[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            let ptr = self.ptrs[lca_level as usize];
            if ptr == u32::MAX {
                // Poisoned by an earlier missed descent: an ancestor of this
                // whole region is known to be air.
                return None;
            }
            (meta.getter)(&self.tree.pool, new_coords, ptr, &mut self.ptrs)
        }?;
        // The fast path above caught every read inside the hot leaf, so this
        // descent landed on a different leaf, whose attribute range is
        // fitted: read it through the rank-based offset.
        debug_assert!(self.last_leaf != Some(self.ptrs[0]));
        let occupied = leaf_node.get_occupancy_at(coords);
        if !occupied {
            return None;
        }
        let value = self.attributes.get_attribute(
            leaf_node.get_value(),
            leaf_node.get_fitted_attribute_offset(coords),
        );
        Some(value)
    }

    #[inline]
    pub fn set(&mut self, coords: UVec3, value: ATTRIBS::Value)
    where
        ROOT: Node,
    {
        if value.is_default() {
            // Writing the default value erases the voxel.
            self.erase(coords);
            return;
        }
        // Fast path: writing inside the hot leaf. Its attribute range is
        // inflated (one slot per voxel), so setting the occupancy bit and
        // writing the slot is the whole job — no descent at all. Writes in
        // the hot leaf change no tree structure, so both caches stay valid
        // as they are.
        if let Some(last_leaf) = self.last_leaf
            && ((coords ^ self.last_leaf_coords) & !<ROOT::LeafType as Node>::EXTENT_MASK)
                == UVec3::ZERO
        {
            // The hot leaf was established by a write, so it is uniquely
            // owned: mutating it in place cannot be observed by a snapshot.
            debug_assert!(!self.tree.pool[0].is_shared(last_leaf));
            let leaf_node = unsafe { self.tree.get_node_mut::<ROOT::LeafType>(last_leaf) };
            leaf_node.set_occupancy_at(coords, true);
            self.attributes.set_attribute(
                leaf_node.get_value(),
                <ROOT::LeafType as IsLeaf>::get_inflated_attribute_offset(coords),
                value,
            );
            return;
        }
        let lca_level = lowest_common_ancestor_level(
            self.last_set_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_set_coords = coords;
        let mut moved = false;
        let leaf_node = if lca_level >= ROOT::LEVEL as u32 {
            self.tree
                .root
                .set(&mut self.tree.pool, coords, &mut self.set_ptrs, &mut moved)
        } else {
            let meta = &ROOT::META[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            let mut ptr = self.set_ptrs[lca_level as usize];
            // Writes may only re-enter the tree at a uniquely owned node: a
            // shared node would be copied against a dangling local edge. The
            // set path guarantees this (see the set_ptrs field docs).
            debug_assert!(!self.tree.pool[lca_level as usize].is_shared(ptr));
            (meta.setter)(
                &mut self.tree.pool,
                new_coords,
                &mut ptr,
                &mut self.set_ptrs,
                &mut moved,
            )
        };
        // The write may have copied nodes on its way down, so the read cache
        // could now reference stale pre-copy nodes. set_ptrs holds the fresh
        // path for `coords`; adopt it for reads too.
        self.ptrs = self.set_ptrs;
        self.last_coords = coords;

        // The fast path above caught every write into the hot leaf, so this
        // descent landed on a different leaf: transition the hot leaf away.
        debug_assert!(self.last_leaf != Some(self.set_ptrs[0]));
        // Release reference to leaf_node so that we can borrow prev_access_leaf_node.
        // Safety: It has already been established that prev_access_leaf_node is not leaf_node, so it should be fine to have both mutable references.
        let leaf_node: *mut _ = leaf_node;
        self.purge_prev_access_leaf_node();
        let leaf_node = unsafe { &mut *leaf_node };
        let previously_occupied = leaf_node.get_occupancy_at(coords);

        if moved {
            // This leaf was copied on write and still references the attribute
            // range owned by the snapshot's version of the leaf. Re-home it to
            // a private, fully inflated range. The old range must NOT be freed
            // here: the snapshot leaf keeps it until it is released.
            let new_attrib_ptr = self.attributes.copy_attribute(
                &leaf_node.get_value(),
                leaf_node.get_occupancy(),
                &ATTRIBS::MAX_OCCUPANCY,
                &coords,
            );
            self.last_leaf = Some(self.set_ptrs[0]);
            self.last_leaf_coords = coords;
            leaf_node.set_occupancy_at(coords, true);
            self.attributes.set_attribute(
                &new_attrib_ptr,
                <ROOT::LeafType as IsLeaf>::get_inflated_attribute_offset(coords),
                value,
            );
            leaf_node.set_value(new_attrib_ptr);
            return;
        }

        // Copy to a new leaf node with maxed occupancy.
        if previously_occupied {
            self.attributes.set_attribute(
                leaf_node.get_value(),
                leaf_node.get_fitted_attribute_offset(coords),
                value,
            );
        } else {
            // trick for now: set the bit to false, then after copy attribute, set it back.

            let new_attrib_ptr = self.attributes.copy_attribute(
                &leaf_node.get_value(),
                leaf_node.get_occupancy(), // this original mask is wrong. should be old_attrib_occupancy
                &ATTRIBS::MAX_OCCUPANCY,
                &coords,
            );
            self.last_leaf = Some(self.set_ptrs[0]);
            self.last_leaf_coords = coords;
            // if old_attrib_occupancy.count_ones() > 0, free.
            let old_attrib_occupancy_count = leaf_node.get_occupancy().count_ones() as u32; // can optimize here
            if old_attrib_occupancy_count > 0 {
                self.attributes
                    .free_attributes(leaf_node.get_value(), old_attrib_occupancy_count);
            }
            leaf_node.set_occupancy_at(coords, true);

            // Hint: just need to get the old attrib_occupancy now.
            self.attributes.set_attribute(
                &new_attrib_ptr,
                <ROOT::LeafType as IsLeaf>::get_inflated_attribute_offset(coords),
                value,
            );
            leaf_node.set_value(new_attrib_ptr);
        };
    }

    /// Remove the voxel at `coords`: clear its occupancy bit and drop its
    /// attribute. The clear descent itself never frees nodes; an emptied
    /// leaf is collapsed — freed together with any ancestors that empty
    /// with it — by a separate descent from the root, where the free-cascade
    /// walks real parent edges. The slow path collapses immediately; the
    /// fast path defers to `purge_prev_access_leaf_node` so that set/erase
    /// cycles in the hot leaf can resurrect it for free.
    fn erase(&mut self, coords: UVec3) {
        // Fast path: erasing inside the hot leaf. Its attribute range is
        // inflated (one slot per voxel), so clearing the occupancy bit is the
        // whole job — the dead slot is dropped whenever the leaf is next
        // fitted. Set/erase cycles on one voxel never thrash the tree or the
        // attribute allocator.
        if let Some(last_leaf) = self.last_leaf
            && ((coords ^ self.last_leaf_coords) & !<ROOT::LeafType as Node>::EXTENT_MASK)
                == UVec3::ZERO
        {
            // The hot leaf was established by a write, so it is uniquely
            // owned: mutating it in place cannot be observed by a snapshot.
            debug_assert!(!self.tree.pool[0].is_shared(last_leaf));
            let leaf_node = unsafe { self.tree.get_node_mut::<ROOT::LeafType>(last_leaf) };
            leaf_node.set_occupancy_at(coords, false);
            return;
        }
        // Clear through the write cache. The descent verifies existence
        // itself — a clear for a missing voxel is a no-op returning None —
        // and cannot free nodes (freeing lives in `collapse`, which always
        // descends from the root), so re-entering the tree mid-path is
        // always safe: there is no cascade of frees to escape above the
        // re-entry point.
        let lca_level = lowest_common_ancestor_level(
            self.last_set_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        let mut moved = false;
        let survivor = if lca_level >= ROOT::LEVEL as u32 {
            self.tree
                .root
                .clear(&mut self.tree.pool, coords, &mut self.set_ptrs, &mut moved)
        } else {
            let meta = &ROOT::META[lca_level as usize];
            let mut ptr = self.set_ptrs[lca_level as usize];
            // Writes may only re-enter the tree at a uniquely owned node: a
            // shared node would be copied against a dangling local edge. The
            // set path guarantees this (see the set_ptrs field docs).
            debug_assert!(!self.tree.pool[lca_level as usize].is_shared(ptr));
            (meta.clearer)(
                &mut self.tree.pool,
                coords & meta.extent_mask,
                &mut ptr,
                &mut self.set_ptrs,
                &mut moved,
                // Every ancestor of a write-cache entry was uniquified by the
                // write that cached it (see the set_ptrs field docs).
                false,
            )
        };
        let Some(leaf_node) = survivor else {
            // The voxel never existed. The descent forks shared nodes only
            // on its way back up from a found voxel, so a miss has touched
            // nothing — both caches remain exactly as valid as they were.
            return;
        };
        // The survivor still carries its pre-clear state (the descent flips
        // no bits): capture it for the attribute compaction below.
        let old_value = leaf_node.get_value().clone();
        let old_mask = leaf_node.get_occupancy().clone();
        // Transition the old hot leaf away before adopting this one.
        // Release the reference so the purge can borrow the tree.
        // Safety: the slow path only runs when the target is not the hot
        // leaf, and purging never allocates pool space (its collapse only
        // frees, along a path that was already uniquified), so the pointer
        // is neither aliased nor invalidated.
        let leaf_node: *mut ROOT::LeafType = leaf_node;
        self.purge_prev_access_leaf_node();
        let leaf_node = unsafe { &mut *leaf_node };
        // The purge may have collapsed an emptied hot leaf and invalidated
        // the caches. The survivor's path leads to a live leaf, so the
        // collapse cannot have freed any node on it: adopt it for both
        // caches (the clear uniquified every node on the way down).
        self.last_set_coords = coords;
        self.ptrs = self.set_ptrs;
        self.last_coords = coords;
        // Clear the bit (occupancy is the caller's job, symmetric with
        // sets).
        leaf_node.set_occupancy_at(coords, false);
        if leaf_node.get_occupancy().deref().not_any() {
            // That was the leaf's last voxel. Making a dead leaf hot buys
            // nothing, so skip the inflation and collapse it right away:
            // reclaim its attribute range (unless the snapshot's leaf owns
            // it) and re-descend from the root, where the free-cascade walks
            // real parent edges all the way up. The clear above uniquified
            // this path, as the collapse requires.
            if !moved {
                self.attributes
                    .free_attributes(&old_value, old_mask.count_ones() as u32);
            }
            self.tree.root.collapse(&mut self.tree.pool, coords);
            // The collapse freed nodes that both caches now reference.
            self.last_coords = UVec3::MAX;
            self.last_set_coords = UVec3::MAX;
            return;
        }
        // Re-home the attributes to a fully inflated range and make this the
        // hot leaf: repeated edits in the same leaf — an eraser brush — stay
        // on the fast paths. If the old range is shared with a snapshot's
        // leaf (`moved`), it stays with the snapshot instead of being freed.
        let new_value =
            self.attributes
                .copy_attribute(&old_value, &old_mask, &ATTRIBS::MAX_OCCUPANCY, &coords);
        if !moved {
            self.attributes
                .free_attributes(&old_value, old_mask.count_ones() as u32);
        }
        leaf_node.set_value(new_value);
        self.last_leaf = Some(self.set_ptrs[0]);
        self.last_leaf_coords = coords;
    }

    fn purge_prev_access_leaf_node(&mut self) {
        if let Some(last_leaf) = self.last_leaf {
            // purge prev access leaf node by fitting its attributes
            let prev_access_leaf_node =
                unsafe { self.tree.get_node_mut::<ROOT::LeafType>(last_leaf) };
            let old_attrib_ptr = prev_access_leaf_node.get_value();
            if !prev_access_leaf_node.get_occupancy().deref().any() {
                // The hot leaf was fully erased; its collapse was deferred to
                // here (see the erase fast path). Free the inflated attribute
                // range, then remove the leaf — and any ancestors that empty
                // with it — from the tree. The write that made the leaf hot
                // uniquified its path, as the collapse requires.
                self.attributes
                    .free_attributes(old_attrib_ptr, ROOT::LeafType::SIZE as u32);
                self.tree
                    .root
                    .collapse(&mut self.tree.pool, self.last_leaf_coords);
                // The collapse freed nodes that either cache may reference.
                self.last_coords = UVec3::MAX;
                self.last_set_coords = UVec3::MAX;
                self.last_leaf = None;
                return;
            }
            if !prev_access_leaf_node.get_occupancy().deref().all() {
                // fitting attributes by realloc and copy
                let new_attrib_ptr = self.attributes.copy_attribute(
                    &old_attrib_ptr,
                    &ATTRIBS::MAX_OCCUPANCY,
                    prev_access_leaf_node.get_occupancy(),
                    &self.last_leaf_coords,
                );
                self.attributes
                    .free_attributes(old_attrib_ptr, ROOT::LeafType::SIZE as u32);
                prev_access_leaf_node.set_value(new_attrib_ptr);
            }
            self.last_leaf = None;
        }
    }
}
impl<'a, ROOT: Node, ATTRIBS> Drop for AccessorMut<'a, ROOT, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
    ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
{
    fn drop(&mut self) {
        // Skip the purge while unwinding: it mutates the tree and calls the
        // attribute hooks, which may be mid-operation from the original
        // panic — a second panic here would abort and mask it.
        if !std::thread::panicking() {
            self.purge_prev_access_leaf_node();
        }
    }
}

impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    /// A cached read accessor over the tree's current state.
    ///
    /// Borrows the tree shared, so any number of them may coexist — but the
    /// tree cannot be mutated while they live. To keep reading a fixed
    /// version while the tree gets mutated asynchronously, capture a
    /// [`Tree::snapshot`] and use [`TreeSnapshot::accessor`] instead.
    pub fn accessor<
        'a,
        A: Attributes<
                Ptr = <ROOT::LeafType as IsLeaf>::Value,
                Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
            >,
    >(
        &'a self,
        attributes: &'a A,
    ) -> Accessor<'a, ROOT, A> {
        Accessor {
            root: &self.root,
            pool: &self.pool,
            ptrs: [u32::MAX; ROOT::LEVEL],
            last_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            attributes,
            last_leaf: None,
            last_leaf_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
        }
    }
    /// A cached read/write accessor over the tree, borrowing the tree and
    /// the attribute storage exclusively.
    pub fn accessor_mut<
        'a,
        A: Attributes<
                Ptr = <ROOT::LeafType as IsLeaf>::Value,
                Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
            >,
    >(
        &'a mut self,
        attributes: &'a mut A,
    ) -> AccessorMut<'a, ROOT, A> {
        AccessorMut {
            tree: self,
            ptrs: [u32::MAX; ROOT::LEVEL],
            last_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            set_ptrs: [u32::MAX; ROOT::LEVEL],
            last_set_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            attributes,
            last_leaf: None,
            last_leaf_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
        }
    }
}

impl<ROOT: Node> TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    /// A cached read accessor over the snapshot's captured state. It reads
    /// through the snapshot's pinned pools, so it stays valid — including on
    /// another thread — while the originating tree keeps being mutated.
    pub fn accessor<
        'a,
        A: Attributes<
                Ptr = <ROOT::LeafType as IsLeaf>::Value,
                Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
            >,
    >(
        &'a self,
        attributes: &'a A,
    ) -> Accessor<'a, ROOT, A> {
        Accessor {
            root: &self.root,
            pool: &self.pool,
            ptrs: [u32::MAX; ROOT::LEVEL],
            last_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            attributes,
            last_leaf: None,
            last_leaf_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
        }
    }
}

/// Accessors are designed to help accelerate accesses into the tree structures by storing caches
/// to tree branches. When traversing a grid in a spatially coherent pattern, the same branches
/// and nodes of the underlying tree can be hit. Accessors cache the path down to the leaf node so
/// that subsequent neighboring accesses can skip traversing the upper levels of the tree and go
/// directly to the leaf node.
///
/// Obtained from a live [`Tree::accessor`] or a captured
/// [`TreeSnapshot::accessor`]; a snapshot's accessor reads the snapshot's
/// pinned pools, so it stays valid — including on another thread — while the
/// originating tree keeps being mutated.
///
/// To mutate the tree structure, use [`AccessorMut`] instead.
pub struct Accessor<'a, ROOT: Node, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
    ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
{
    root: &'a ROOT,
    pool: &'a [Pool; ROOT::LEVEL],
    /// Cached descent path for lca re-entry.
    ptrs: [u32; ROOT::LEVEL],
    last_coords: UVec3,
    attributes: &'a ATTRIBS,
    last_leaf: Option<u32>,
    last_leaf_coords: UVec3,
}

impl<'a, ROOT: Node, ATTRIBS> Accessor<'a, ROOT, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
    ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
{
    pub fn get(&mut self, coords: UVec3) -> Option<ATTRIBS::Value> {
        // Fast path: reading inside the last visited leaf — no descent. The
        // tree is read-only, so its attribute ranges are all fitted and the
        // lookup goes through the rank-based offset (never the fully mapped
        // one — that layout only exists for [`AccessorMut`]'s hot leaf).
        if let Some(last_leaf) = self.last_leaf
            && ((coords ^ self.last_leaf_coords) & !<ROOT::LeafType as Node>::EXTENT_MASK)
                == UVec3::ZERO
        {
            let leaf_node = unsafe { self.pool[0].get_item::<ROOT::LeafType>(last_leaf) };
            if !leaf_node.get_occupancy_at(coords) {
                return None;
            }
            let attribute = self.attributes.get_attribute(
                leaf_node.get_value(),
                leaf_node.get_fitted_attribute_offset(coords),
            );
            return Some(attribute);
        }
        let lca_level = lowest_common_ancestor_level(
            self.last_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_coords = coords;
        let leaf_node = if lca_level >= ROOT::LEVEL as u32 {
            self.root.get(self.pool, coords, &mut self.ptrs)
        } else {
            let meta = &ROOT::META[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            let ptr = self.ptrs[lca_level as usize];
            if ptr == u32::MAX {
                // Poisoned by an earlier missed descent: an ancestor of this
                // whole region is known to be air.
                return None;
            }
            (meta.getter)(self.pool, new_coords, ptr, &mut self.ptrs)
        }?;
        // Remember the landing leaf: subsequent reads within its region skip
        // the descent, whether this one hits or misses its occupancy.
        self.last_leaf = Some(self.ptrs[0]);
        self.last_leaf_coords = coords;
        let occupied = leaf_node.get_occupancy_at(coords);
        if !occupied {
            return None;
        }
        let value = self.attributes.get_attribute(
            leaf_node.get_value(),
            leaf_node.get_fitted_attribute_offset(coords),
        );
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use bitvec::array::BitArray;
    use glam::UVec3;

    use super::{Attributes, lowest_common_ancestor_level};
    use crate::{IsLeaf, Node, Tree, hierarchy};

    #[derive(Default)]
    struct TestAttributes {
        attribute_maps: Vec<Vec<u8>>,
    }

    impl Attributes for TestAttributes {
        type Ptr = u32;
        type Occupancy = BitArray<[usize; 64 / size_of::<usize>() / 8]>;
        const MAX_OCCUPANCY: Self::Occupancy = Self::Occupancy {
            _ord: PhantomData,
            data: [usize::MAX; 64 / size_of::<usize>() / 8],
        };
        type Value = u8;

        fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value {
            self.attribute_maps[*ptr as usize][offset as usize]
        }

        fn get_attributes(&self, ptr: &Self::Ptr, len: u32) -> &[Self::Value] {
            let slice = &self.attribute_maps[*ptr as usize];
            assert_eq!(slice.len(), len as usize);
            slice
        }

        fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value) {
            self.attribute_maps[*ptr as usize][offset as usize] = value;
        }

        fn free_attributes(&mut self, ptr: &Self::Ptr, num_attributes: u32) {
            println!("free {} attributes: {}", num_attributes, ptr);
            let slice = &self.attribute_maps[*ptr as usize];
            assert_eq!(slice.len(), num_attributes as usize);
            self.attribute_maps[*ptr as usize] = Vec::new();
        }

        fn copy_attribute(
            &mut self,
            ptr: &Self::Ptr,
            original_mask: &Self::Occupancy,
            new_mask: &Self::Occupancy,
            coords: &UVec3,
        ) -> Self::Ptr {
            if !original_mask.any() {
                let new = vec![0; new_mask.count_ones() as usize];
                println!(
                    "copy_attribute at {:?} from null to {}: {} -> {}",
                    coords,
                    self.attribute_maps.len(),
                    original_mask.count_ones(),
                    new_mask.count_ones()
                );
                self.attribute_maps.push(new);
                return self.attribute_maps.len() as u32 - 1;
            }
            let mut new = vec![0; new_mask.count_ones() as usize];
            let old = &self.attribute_maps[*ptr as usize];
            let mut new_ptr = 0;
            let mut old_ptr = 0;
            for bit in (*original_mask | new_mask).iter_ones() {
                if *new_mask.get(bit).unwrap() && *original_mask.get(bit).unwrap() {
                    // copy it over
                    new[new_ptr] = old[old_ptr as usize];
                }
                if *new_mask.get(bit).unwrap() {
                    new_ptr += 1;
                }
                if *original_mask.get(bit).unwrap() {
                    old_ptr += 1;
                }
            }
            println!(
                "copy_attribute at {:?} from {} to {}: {} -> {}",
                coords,
                ptr,
                self.attribute_maps.len(),
                original_mask.count_ones(),
                new_mask.count_ones()
            );
            self.attribute_maps.push(new);
            self.attribute_maps.len() as u32 - 1
        }
    }

    #[test]
    fn test() {
        type MyTreeRoot = hierarchy!(2, 4, 2, u32);
        let mask: UVec3 = MyTreeRoot::META_MASK;
        assert_eq!(
            mask,
            UVec3 {
                x: 0b10100010,
                y: 0b10100010,
                z: 0b10100010
            }
        );
        assert_eq!(
            lowest_common_ancestor_level(
                UVec3::new(0, 0, 0),
                UVec3::new(255, 255, 255),
                mask,
                MyTreeRoot::LEVEL as u32
            ),
            2
        );
    }

    #[test]
    fn test_accessor() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();

        let mut attributes = TestAttributes::default();
        let mut accessor = tree.accessor_mut(&mut attributes);

        accessor.set(UVec3::new(0, 0, 3), 12);
        // Allocates full map for additional attributes
        assert_eq!(accessor.attributes.attribute_maps[0].len(), 64);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 1, 5)), None);

        accessor.set(UVec3::new(0, 1, 0), 13);
        // Subsequent ops in the same leaf node should not allocate
        assert_eq!(accessor.attributes.attribute_maps[0].len(), 64);
        assert_eq!(accessor.attributes.attribute_maps.len(), 1);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(13));
        assert_eq!(accessor.get(UVec3::new(0, 1, 2)), None);

        accessor.set(UVec3::new(144, 1, 0), 14);
        // Transitioned to new block. The old maxed out block should be freed, with
        // its content copied to a new tightly fitting block.
        assert_eq!(accessor.attributes.attribute_maps[1].len(), 2);
        assert_eq!(accessor.attributes.attribute_maps[2].len(), 64);
        assert_eq!(accessor.attributes.attribute_maps.len(), 3);
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));

        accessor.set(UVec3::new(0, 1, 2), 16);
        // Transitioned back to old block.
        assert_eq!(accessor.attributes.attribute_maps[2].len(), 0);
        assert_eq!(accessor.attributes.attribute_maps[3].len(), 1);
        assert_eq!(accessor.attributes.attribute_maps[4].len(), 64);
        assert_eq!(accessor.attributes.attribute_maps.len(), 5);
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        assert_eq!(accessor.get(UVec3::new(0, 1, 2)), Some(16));

        accessor.set(UVec3::new(144, 1, 0), 18);
        // Updating an existing attribute should not allocate.
        assert_eq!(accessor.attributes.attribute_maps[4].len(), 0);
        assert_eq!(accessor.attributes.attribute_maps[5].len(), 3);
        assert_eq!(accessor.attributes.attribute_maps.len(), 6);
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(18));
        assert_eq!(accessor.get(UVec3::new(0, 1, 2)), Some(16));
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(13));

        drop(accessor);
    }

    /// Frees the attribute range owned by a leaf that the tree dropped during
    /// a snapshot release or restore.
    fn drop_leaf_attributes(
        attributes: &mut TestAttributes,
        leaf: &<hierarchy!(2, 4, 2, u32) as crate::NodeConst>::LeafType,
    ) {
        let occupied = leaf.get_occupancy().count_ones() as u32;
        if occupied > 0 {
            attributes.free_attributes(leaf.get_value(), occupied);
        }
    }

    #[test]
    fn test_snapshot_isolation() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        // Initial content: two voxels in one leaf, one voxel in another subtree.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        accessor.set(UVec3::new(144, 1, 0), 14);
        drop(accessor);

        let baseline_leaves = tree.pools()[0].count();
        let baseline_internal = tree.pools()[1].count();
        assert_eq!(baseline_leaves, 2);
        assert_eq!(baseline_internal, 2);

        let snapshot = tree.snapshot();
        assert_eq!(tree.snapshot_count(), 1);

        // Overwrite one voxel and add a new one, both in the first leaf.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 99);
        accessor.set(UVec3::new(1, 1, 1), 55);
        drop(accessor);

        // The working tree sees the new state...
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(99));
        assert_eq!(accessor.get(UVec3::new(1, 1, 1)), Some(55));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(13));
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        drop(accessor);

        // ...while the snapshot still observes the captured state.
        let leaf = snapshot.get(UVec3::new(0, 0, 3)).unwrap();
        assert!(leaf.get_occupancy_at(UVec3::new(0, 0, 3)));
        assert!(!leaf.get_occupancy_at(UVec3::new(1, 1, 1)));
        assert_eq!(
            attributes.get_attribute(
                leaf.get_value(),
                leaf.get_attribute_offset(UVec3::new(0, 0, 3))
            ),
            12
        );
        assert_eq!(
            attributes.get_attribute(
                leaf.get_value(),
                leaf.get_attribute_offset(UVec3::new(0, 1, 0))
            ),
            13
        );
        assert_eq!(attributes.attribute_maps[1], vec![12, 13]);

        // The edit copied exactly one leaf and one internal node.
        assert_eq!(tree.pools()[0].count(), baseline_leaves + 1);
        assert_eq!(tree.pools()[1].count(), baseline_internal + 1);

        // Releasing the snapshot reclaims the pinned nodes and their attributes.
        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.snapshot_count(), 0);
        assert_eq!(tree.pools()[0].count(), baseline_leaves);
        assert_eq!(tree.pools()[1].count(), baseline_internal);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
        assert_eq!(attributes.attribute_maps[1].len(), 0);

        // The working state is untouched by the release.
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(99));
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        drop(accessor);
    }

    #[test]
    fn test_snapshot_restore() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        drop(accessor);

        let baseline_leaves = tree.pools()[0].count();
        let baseline_internal = tree.pools()[1].count();

        let undo_point = tree.snapshot();

        // Edit: overwrite the voxel and grow a brand-new subtree.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 99);
        accessor.set(UVec3::new(144, 1, 0), 7);
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), baseline_leaves + 2);
        assert_eq!(tree.pools()[1].count(), baseline_internal + 2);

        // Undo: drop the working state, adopt the snapshot's.
        tree.restore(&undo_point, |leaf| {
            drop_leaf_attributes(&mut attributes, leaf)
        });

        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), None);
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), baseline_leaves);
        assert_eq!(tree.pools()[1].count(), baseline_internal);

        // The undo point is still live and can be released independently.
        // Everything it references is also referenced by the restored working
        // state, so nothing gets freed.
        tree.release_snapshot(undo_point, |_| {
            unreachable!("no node is exclusively pinned by the snapshot")
        });
        assert_eq!(tree.pools()[0].count(), baseline_leaves);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);

        // With no snapshots left, edits mutate in place again: no copies.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 50);
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), baseline_leaves);
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(50));
        drop(accessor);
    }

    #[test]
    fn test_multiple_snapshots() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let coords = UVec3::new(0, 0, 3);
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(coords, 1);
        drop(accessor);

        let snap_v1 = tree.snapshot();
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(coords, 2);
        drop(accessor);

        let snap_v2 = tree.snapshot();
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(coords, 3);
        drop(accessor);

        // Three versions of the same leaf coexist.
        let v1 = snap_v1.get(coords).unwrap();
        assert_eq!(
            attributes.get_attribute(v1.get_value(), v1.get_attribute_offset(coords)),
            1
        );
        let v2 = snap_v2.get(coords).unwrap();
        assert_eq!(
            attributes.get_attribute(v2.get_value(), v2.get_attribute_offset(coords)),
            2
        );
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(coords), Some(3));
        drop(accessor);

        // Releasing v1 must not disturb v2 or the working state.
        tree.release_snapshot(snap_v1, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        let v2 = snap_v2.get(coords).unwrap();
        assert_eq!(
            attributes.get_attribute(v2.get_value(), v2.get_attribute_offset(coords)),
            2
        );
        tree.release_snapshot(snap_v2, |leaf| drop_leaf_attributes(&mut attributes, leaf));

        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(coords), Some(3));
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 1);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
    }

    #[test]
    fn test_snapshot_read_then_write() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        drop(accessor);

        let snapshot = tree.snapshot();

        // Reads fill the accessor's read cache with nodes shared with the
        // snapshot. The write that follows must not re-enter the tree through
        // that cache — it has to descend from the root and copy-on-write.
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        accessor.set(UVec3::new(0, 1, 0), 42);
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(42));
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        drop(accessor);

        let leaf = snapshot.get(UVec3::new(0, 1, 0)).unwrap();
        assert_eq!(
            attributes.get_attribute(
                leaf.get_value(),
                leaf.get_attribute_offset(UVec3::new(0, 1, 0))
            ),
            13
        );

        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
    }

    #[test]
    fn test_concurrent_snapshot_read() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        for i in 0..16u32 {
            let coords = UVec3::new((i * 7) % 64, (i * 13) % 64, (i * 29) % 64);
            accessor.set(coords, (i + 1) as u8);
        }
        drop(accessor);

        let snapshot = tree.snapshot();
        let expected: Vec<UVec3> = snapshot.iter().collect();
        assert!(!expected.is_empty());

        // A snapshot is self-contained: another thread traverses it while this
        // thread keeps mutating the tree it was taken from, including enough
        // writes to force the leaf pool to grow into a new allocation (the
        // snapshot keeps the captured allocation alive).
        std::thread::scope(|scope| {
            let snapshot = &snapshot;
            let expected = &expected;
            let reader = scope.spawn(move || {
                for _ in 0..1000 {
                    let seen: Vec<UVec3> = snapshot.iter().collect();
                    assert_eq!(&seen, expected);
                }
            });
            let mut accessor = tree.accessor_mut(&mut attributes);
            for i in 0..64u32 {
                let coords = UVec3::new((i * 3) % 64, (i * 11) % 64, (i * 17) % 64);
                accessor.set(coords, 200);
            }
            drop(accessor);
            reader.join().unwrap();
        });

        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
    }

    #[test]
    fn test_clear_voxel() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        accessor.set(UVec3::new(144, 1, 0), 14);
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 2);
        assert_eq!(tree.pools()[1].count(), 2);

        // Clearing one of two voxels: the leaf survives and becomes the hot
        // leaf (inflated attribute range), and no nodes are freed. Dropping
        // the accessor fits the range back down to the one remaining voxel.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 1, 0), 0);
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), None);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 2);
        assert_eq!(tree.pools()[1].count(), 2);
        assert_eq!(attributes.attribute_maps[1].len(), 0);
        assert_eq!(attributes.attribute_maps[4].len(), 0);
        assert_eq!(attributes.attribute_maps[5], vec![12]);

        // Clearing the last voxel of the leaf: the leaf dies, and the
        // internal node above it became empty and dies with it.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 0);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), None);
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 1);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(attributes.attribute_maps[5].len(), 0);
        // The death path allocates no attribute ranges: the range is freed
        // and the leaf collapses without ever being inflated.
        assert_eq!(attributes.attribute_maps.len(), 6);
        assert_eq!(tree.count_leaves(), 1);
        let occupied: Vec<UVec3> = tree.iter().collect();
        assert_eq!(occupied, vec![UVec3::new(144, 1, 0)]);

        // The collapsed region accepts new voxels again.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 21);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(21));
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 2);
        assert_eq!(tree.pools()[1].count(), 2);
    }

    #[test]
    fn test_clear_with_snapshot() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(144, 1, 0), 14);
        drop(accessor);

        let snapshot = tree.snapshot();

        // Clear the only voxel of the first leaf. The leaf and its emptied
        // ancestors leave the working tree, but the snapshot still owns them:
        // nothing is freed, and the attribute range stays with the snapshot.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 0);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), None);
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 2);
        assert_eq!(tree.pools()[1].count(), 2);
        assert_eq!(attributes.attribute_maps[1], vec![12]);

        let leaf = snapshot.get(UVec3::new(0, 0, 3)).unwrap();
        assert!(leaf.get_occupancy_at(UVec3::new(0, 0, 3)));
        assert_eq!(
            attributes.get_attribute(
                leaf.get_value(),
                leaf.get_attribute_offset(UVec3::new(0, 0, 3))
            ),
            12
        );

        // Releasing the snapshot frees the nodes the clear left behind.
        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].count(), 1);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
        assert_eq!(attributes.attribute_maps[1].len(), 0);

        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        drop(accessor);
    }

    #[test]
    fn test_clear_survivor_with_snapshot() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        // Two voxels in the same leaf.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        drop(accessor);

        let snapshot = tree.snapshot();

        // Clear one of the two: the leaf survives in the working tree, so it
        // must be copied on write — the snapshot's leaf keeps both voxels and
        // its attribute range.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 1, 0), 0);
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), None);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        drop(accessor);

        let leaf = snapshot.get(UVec3::new(0, 1, 0)).unwrap();
        assert!(leaf.get_occupancy_at(UVec3::new(0, 1, 0)));
        assert!(leaf.get_occupancy_at(UVec3::new(0, 0, 3)));
        assert_eq!(
            attributes.get_attribute(
                leaf.get_value(),
                leaf.get_attribute_offset(UVec3::new(0, 1, 0))
            ),
            13
        );
        assert_eq!(attributes.attribute_maps[1], vec![12, 13]);

        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);

        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), None);
        drop(accessor);
    }

    #[test]
    fn test_erase_set_cycle() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let v = UVec3::new(0, 0, 3);
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(v, 1);
        let maps_after_first_set = accessor.attributes.attribute_maps.len();

        // set/unset cycles on one voxel stay inside the hot leaf: no
        // attribute reallocation and no tree surgery, even though the voxel
        // is the leaf's only one — the empty leaf's collapse is deferred.
        for i in 0..100u8 {
            accessor.set(v, 0);
            assert_eq!(accessor.tree.pools()[0].count(), 1);
            assert_eq!(accessor.get(v), None);
            accessor.set(v, i + 2);
            assert_eq!(accessor.get(v), Some(i + 2));
        }
        assert_eq!(
            accessor.attributes.attribute_maps.len(),
            maps_after_first_set
        );
        assert_eq!(accessor.tree.pools()[0].count(), 1);
        assert_eq!(accessor.tree.pools()[1].count(), 1);

        // A no-op erase in unallocated space touches nothing: the hot leaf
        // and both caches survive it, so the next edit is still fast.
        accessor.set(UVec3::new(200, 200, 200), 0);
        accessor.set(v, 77);
        assert_eq!(accessor.get(v), Some(77));
        assert_eq!(
            accessor.attributes.attribute_maps.len(),
            maps_after_first_set
        );

        // Ending on an erase: the empty hot leaf collapses on drop, taking
        // its emptied ancestors with it.
        accessor.set(v, 0);
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 0);
        assert_eq!(tree.pools()[1].count(), 0);
        assert_eq!(tree.count_leaves(), 0);
        assert_eq!(tree.iter().count(), 0);
        for map in &attributes.attribute_maps {
            assert!(map.is_empty());
        }
    }

    #[test]
    fn test_erase_cached_reentry() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        // A second leaf under the same internal node; it becomes the hot one.
        accessor.set(UVec3::new(4, 0, 0), 9);

        // Erasing in the first leaf now: the probe re-enters through the read
        // cache and the clear re-enters through the write cache at the shared
        // internal node — no root descent. The surviving leaf takes over as
        // the hot leaf.
        accessor.set(UVec3::new(0, 1, 0), 0);
        // A follow-up set in the erased leaf rides the write cache and the
        // hot-leaf fast path: no attribute allocation.
        let maps_before = accessor.attributes.attribute_maps.len();
        accessor.set(UVec3::new(0, 2, 0), 7);
        assert_eq!(accessor.attributes.attribute_maps.len(), maps_before);

        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), None);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 2, 0)), Some(7));
        assert_eq!(accessor.get(UVec3::new(4, 0, 0)), Some(9));
        drop(accessor);

        assert_eq!(tree.pools()[0].count(), 2);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(attributes.attribute_maps[3], vec![9]);
        assert_eq!(attributes.attribute_maps[5], vec![12, 7]);
    }

    #[test]
    fn test_erase_hot_leaf_with_snapshot() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        drop(accessor);

        let snapshot = tree.snapshot();

        // The set copies the shared leaf and makes the copy hot; the erase
        // then takes the fast path, flipping a bit on the private copy only.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 99);
        accessor.set(UVec3::new(0, 1, 0), 0);
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), None);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(99));
        drop(accessor);

        // The snapshot's leaf still holds both voxels and its attributes.
        let leaf = snapshot.get(UVec3::new(0, 1, 0)).unwrap();
        assert!(leaf.get_occupancy_at(UVec3::new(0, 1, 0)));
        assert!(leaf.get_occupancy_at(UVec3::new(0, 0, 3)));
        assert_eq!(attributes.attribute_maps[1], vec![12, 13]);

        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].count(), 1);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);

        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(99));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), None);
        drop(accessor);
    }

    #[test]
    fn test_get_miss_then_cached_reentry() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        // A miss in an unallocated root cell. The descent stops early, so the
        // lower levels of the read cache still describe the previous path.
        assert_eq!(accessor.get(UVec3::new(0, 0, 64)), None);
        // The lca with the previous (missed) lookup is at leaf level, so this
        // re-enters the cached path at its lowest level — which the miss never
        // wrote. It must not read the stale leaf from the first set and
        // fabricate a voxel.
        assert_eq!(accessor.get(UVec3::new(0, 0, 67)), None);
        // A real voxel is still found after the misses.
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        drop(accessor);
    }

    #[test]
    fn test_readonly_accessor() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        accessor.set(UVec3::new(144, 1, 0), 14);
        drop(accessor);
        let maps_before = attributes.attribute_maps.len();

        // Read-only accessors borrow the tree shared, so several can coexist.
        let mut reader = tree.accessor(&attributes);
        let mut other = tree.accessor(&attributes);
        // Cold read, then repeated reads in the same leaf ride the fast
        // path — always through the fitted, rank-compacted attribute layout.
        assert_eq!(reader.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(reader.get(UVec3::new(0, 1, 0)), Some(13));
        assert_eq!(reader.get(UVec3::new(0, 1, 2)), None);
        // Leaf transitions and back.
        assert_eq!(reader.get(UVec3::new(144, 1, 0)), Some(14));
        assert_eq!(reader.get(UVec3::new(144, 1, 1)), None);
        assert_eq!(reader.get(UVec3::new(0, 0, 3)), Some(12));
        // A miss in unallocated space poisons the descent cache without
        // disturbing the last-leaf fast path.
        assert_eq!(reader.get(UVec3::new(0, 0, 64)), None);
        assert_eq!(reader.get(UVec3::new(0, 0, 67)), None);
        assert_eq!(reader.get(UVec3::new(0, 1, 0)), Some(13));
        assert_eq!(other.get(UVec3::new(144, 1, 0)), Some(14));
        // Reading allocates nothing: a read-only accessor cannot inflate.
        assert_eq!(attributes.attribute_maps.len(), maps_before);
    }

    #[test]
    fn test_snapshot_accessor() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.set(UVec3::new(0, 1, 0), 13);
        accessor.set(UVec3::new(144, 1, 0), 14);
        drop(accessor);

        let snapshot = tree.snapshot();

        // Diverge the tree from the snapshot: overwrite, insert, and erase.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 99);
        accessor.set(UVec3::new(1, 1, 1), 55);
        accessor.set(UVec3::new(144, 1, 0), 0);
        drop(accessor);

        // The snapshot's accessor keeps observing the captured state —
        // repeated same-leaf reads ride the fast path — through the
        // snapshot's own fitted attribute ranges.
        let mut reader = snapshot.accessor(&attributes);
        assert_eq!(reader.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(reader.get(UVec3::new(0, 1, 0)), Some(13));
        assert_eq!(reader.get(UVec3::new(1, 1, 1)), None);
        assert_eq!(reader.get(UVec3::new(144, 1, 0)), Some(14));
        assert_eq!(reader.get(UVec3::new(0, 0, 3)), Some(12));
        drop(reader);

        // The live tree's accessor sees the diverged state.
        let mut reader = tree.accessor(&attributes);
        assert_eq!(reader.get(UVec3::new(0, 0, 3)), Some(99));
        assert_eq!(reader.get(UVec3::new(1, 1, 1)), Some(55));
        assert_eq!(reader.get(UVec3::new(144, 1, 0)), None);
        drop(reader);

        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
    }

    #[test]
    fn test_clear_missing_noop() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        drop(accessor);
        let maps_before = attributes.attribute_maps.len();

        let snapshot = tree.snapshot();

        // Clearing voxels that do not exist — in an existing leaf and in
        // unallocated space — must change nothing at all, even with a live
        // snapshot sharing the path: the clear descent forks nodes only on
        // its way back up from a found voxel, so a miss makes no
        // copy-on-write copies anywhere. No probe needed for that.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 2), 0);
        accessor.set(UVec3::new(32, 32, 32), 0);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        drop(accessor);
        assert_eq!(tree.pools()[0].count(), 1);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(attributes.attribute_maps.len(), maps_before);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 1);

        tree.release_snapshot(snapshot, |_| {
            unreachable!("nothing was exclusively pinned by the snapshot")
        });
        assert_eq!(tree.pools()[0].count(), 1);
        assert_eq!(tree.pools()[1].count(), 1);
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        drop(accessor);
    }
}
