use crate::{Attributes, IsDefault, IsLeaf, Node, Tree};
use glam::UVec3;
use std::ops::Deref;

/// Accessors are designed to help accelerate accesses into the tree structures by storing caches
/// to tree branches. When traversing a grid in a spatially coherent pattern, the same branches
/// and nodes of the underlying tree can be hit. Accessors cache the path down to the leaf node so
/// that subsequent neighboring accesses can skip traversing the upper levels of the tree and go
/// directly to the leaf node.
pub struct Accessor<'a, ROOT: Node, ATTRIBS>
where
    [(); ROOT::LEVEL]: Sized,
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

impl<
    'a,
    ROOT: Node,
    ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
> Accessor<'a, ROOT, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    pub fn get(&mut self, coords: UVec3) -> Option<ATTRIBS::Value> {
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
            (meta.getter)(&self.tree.pool, new_coords, ptr, &mut self.ptrs)
        }?;
        let occupied = leaf_node.get_occupancy_at(coords);
        if !occupied {
            return None;
        }
        if let Some(last_leaf) = self.last_leaf {
            if last_leaf == self.ptrs[0] {
                let last_leaf = unsafe { self.tree.get_node::<ROOT::LeafType>(last_leaf) };
                if std::ptr::eq(last_leaf, leaf_node) {
                    return Some(self.attributes.get_attribute(
                        leaf_node.get_value(),
                        <ROOT::LeafType as IsLeaf>::get_fully_mapped_offset(coords),
                    ));
                }
            }
        }
        let value = self.attributes.get_attribute(
            leaf_node.get_value(),
            leaf_node.get_attribute_offset(coords),
        );
        Some(value)
    }

    #[inline]
    pub fn set(&mut self, coords: UVec3, value: ATTRIBS::Value)
    where
        ROOT: Node,
    {
        let lca_level = lowest_common_ancestor_level(
            self.last_set_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_set_coords = coords;
        let mut moved = false;
        let leaf_node = if lca_level >= ROOT::LEVEL as u32 {
            self.tree.root.set(
                &mut self.tree.pool,
                coords,
                !value.is_default(),
                &mut self.set_ptrs,
                &mut moved,
            )
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
                !value.is_default(),
                &mut self.set_ptrs,
                &mut moved,
            )
        };
        // The write may have copied nodes on its way down, so the read cache
        // could now reference stale pre-copy nodes. set_ptrs holds the fresh
        // path for `coords`; adopt it for reads too.
        self.ptrs = self.set_ptrs;
        self.last_coords = coords;

        if let Some(last_leaf) = self.last_leaf {
            if last_leaf == self.set_ptrs[0] {
                // Still accessing the same leaf node. A leaf we already wrote
                // to in this session is uniquely owned, so it cannot have been
                // copied on this write.
                debug_assert!(!moved);
                leaf_node.set_occupancy_at(coords, true);
                self.attributes.set_attribute(
                    &leaf_node.get_value(),
                    <ROOT::LeafType as IsLeaf>::get_fully_mapped_offset(coords),
                    value,
                );
                return;
            }
        }
        // Release reference to leaf_node so that we can borrow prev_access_leaf_node.
        // Satefty: It has already been established that prev_access_leaf_node is not leaf_node, so it should be fine to have both mutable references.
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
                <ROOT::LeafType as IsLeaf>::get_fully_mapped_offset(coords),
                value,
            );
            leaf_node.set_value(new_attrib_ptr);
            return;
        }

        // Copy to a new leaf node with maxed occupancy.
        if previously_occupied {
            self.attributes.set_attribute(
                leaf_node.get_value(),
                leaf_node.get_attribute_offset(coords),
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
                <ROOT::LeafType as IsLeaf>::get_fully_mapped_offset(coords),
                value,
            );
            leaf_node.set_value(new_attrib_ptr);
        };
    }

    fn purge_prev_access_leaf_node(&mut self) {
        if let Some(last_leaf) = self.last_leaf {
            // purge prev access leaf node by fitting its attributes
            let prev_access_leaf_node =
                unsafe { self.tree.get_node_mut::<ROOT::LeafType>(last_leaf) };
            let old_attrib_ptr = prev_access_leaf_node.get_value();
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
    pub fn end(mut self) {
        self.purge_prev_access_leaf_node();
    }
}

impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    pub fn accessor_mut<
        'a,
        A: Attributes<
                Ptr = <ROOT::LeafType as IsLeaf>::Value,
                Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
            >,
    >(
        &'a mut self,
        attributes: &'a mut A,
    ) -> Accessor<'a, ROOT, A> {
        Accessor {
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

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use bitvec::{BitArr, array::BitArray};
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

        accessor.end();
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
        accessor.end();

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
        accessor.end();

        // The working tree sees the new state...
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(99));
        assert_eq!(accessor.get(UVec3::new(1, 1, 1)), Some(55));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(13));
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), Some(14));
        accessor.end();

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
        accessor.end();
    }

    #[test]
    fn test_snapshot_restore() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 12);
        accessor.end();

        let baseline_leaves = tree.pools()[0].count();
        let baseline_internal = tree.pools()[1].count();

        let undo_point = tree.snapshot();

        // Edit: overwrite the voxel and grow a brand-new subtree.
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(UVec3::new(0, 0, 3), 99);
        accessor.set(UVec3::new(144, 1, 0), 7);
        accessor.end();
        assert_eq!(tree.pools()[0].count(), baseline_leaves + 2);
        assert_eq!(tree.pools()[1].count(), baseline_internal + 2);

        // Undo: drop the working state, adopt the snapshot's.
        tree.restore(&undo_point, |leaf| {
            drop_leaf_attributes(&mut attributes, leaf)
        });

        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        assert_eq!(accessor.get(UVec3::new(144, 1, 0)), None);
        accessor.end();
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
        accessor.end();
        assert_eq!(tree.pools()[0].count(), baseline_leaves);
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(50));
        accessor.end();
    }

    #[test]
    fn test_multiple_snapshots() {
        type MyTree = Tree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();
        let mut attributes = TestAttributes::default();

        let coords = UVec3::new(0, 0, 3);
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(coords, 1);
        accessor.end();

        let snap_v1 = tree.snapshot();
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(coords, 2);
        accessor.end();

        let snap_v2 = tree.snapshot();
        let mut accessor = tree.accessor_mut(&mut attributes);
        accessor.set(coords, 3);
        accessor.end();

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
        accessor.end();

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
        accessor.end();
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
        accessor.end();

        let snapshot = tree.snapshot();

        // Reads fill the accessor's read cache with nodes shared with the
        // snapshot. The write that follows must not re-enter the tree through
        // that cache — it has to descend from the root and copy-on-write.
        let mut accessor = tree.accessor_mut(&mut attributes);
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        accessor.set(UVec3::new(0, 1, 0), 42);
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(42));
        assert_eq!(accessor.get(UVec3::new(0, 0, 3)), Some(12));
        accessor.end();

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
        accessor.end();

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
            accessor.end();
            reader.join().unwrap();
        });

        tree.release_snapshot(snapshot, |leaf| drop_leaf_attributes(&mut attributes, leaf));
        assert_eq!(tree.pools()[0].refcounts.len(), 0);
        assert_eq!(tree.pools()[1].refcounts.len(), 0);
    }
}
