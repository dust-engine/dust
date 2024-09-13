use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{bitmask::IsBitMask, IsLeaf, MutableTree, Node, NodeMeta};

pub struct Accessor<'a, ROOT: Node, ATTRIBS, TREE>
where
    [(); ROOT::LEVEL + 1]: Sized,
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: TREE,
    ptrs: [u32; ROOT::LEVEL],
    metas: [NodeMeta<ROOT::LeafType>; ROOT::LEVEL + 1],
    last_coords: UVec3,
    attributes: &'a mut ATTRIBS,
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

/*
// For CoW trees
impl<'a, ROOT: Node, A: Attributes<LeafType = ROOT::LeafType>> AccessorMut<'a, ROOT, A>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
    [(); ROOT::LEVEL + 1]: Sized,
{
    /*
    #[inline]
    pub fn get(&mut self, coords: UVec3) -> Option<A::Value>
    where
        ROOT: Node,
    {
        let mut result = false;
        let leaf_node = Accessor::<ROOT, A>::get_inner(&self.tree, &mut self.last_coords, &mut self.ptrs, &self.metas, coords, &mut result)?;
        if !result {
            return None;
        }
        let value = self.attributes.get_attribute(leaf_node.get_value(), leaf_node.get_offset(coords));
        Some(value)
    }
    */
    #[inline]
    fn set(&mut self, coords: UVec3, value: Option<A::Value>)
    where
        ROOT: Node,
    {
        if value.is_some() {
            self.tree.aabb.min = self.tree.aabb.min.min(coords);
            self.tree.aabb.max = self.tree.aabb.max.max(coords);
        }
        let lca_level = lowest_common_ancestor_level(
            self.last_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_coords = coords;
        let mut nodes_to_remove = Vec::new();
        let (old_leaf_node, new_leaf_node) = if lca_level >= ROOT::LEVEL as u32 {
            self.tree
                .root
                .set(&mut self.tree.pool, coords, value.is_some(), &mut self.ptrs, Some(&mut nodes_to_remove))
        } else {
            let meta = &self.metas[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            let mut ptr = self.ptrs[lca_level as usize];
            (meta.setter)(&mut self.tree.pool, new_coords, &mut ptr, value.is_some(), &mut self.ptrs, Some(&mut nodes_to_remove))
        };
        if let Some(old_leaf_node) = old_leaf_node {
            // Needs reallocation
            let new_ptr = self.attributes.move_attribute(
                old_leaf_node.get_value(),
                old_leaf_node.get_occupancy(),
                new_leaf_node.get_occupancy(),
            );
            new_leaf_node.set_value(new_ptr);
        }
    }
}
*/

impl<
        'a,
        ROOT: Node,
        ATTRIBS: Attributes<
            Ptr = <ROOT::LeafType as IsLeaf>::Value,
            Occupancy = <ROOT::LeafType as IsLeaf>::Occupancy,
        >,
    > Accessor<'a, ROOT, ATTRIBS, &'a mut MutableTree<ROOT>>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
    [(); ROOT::LEVEL + 1]: Sized,
{
    #[inline]
    pub fn set(&mut self, coords: UVec3, value: ATTRIBS::Value)
    where
        ROOT: Node,
    {
        let lca_level = lowest_common_ancestor_level(
            self.last_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_coords = coords;
        let prev_access_leaf_node_ptr = self.ptrs[0];
        let (old_leaf_node, leaf_node) = if lca_level >= ROOT::LEVEL as u32 {
            self.tree.root.set(
                &mut self.tree.pool,
                coords,
                !value.is_default(),
                &mut self.ptrs,
                None,
            )
        } else {
            let meta = &self.metas[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            let mut ptr = self.ptrs[lca_level as usize];
            (meta.setter)(
                &mut self.tree.pool,
                new_coords,
                &mut ptr,
                !value.is_default(),
                &mut self.ptrs,
                None,
            )
        };
        assert!(old_leaf_node.is_none()); // Because touched_nodes is None, the tree will not attempt to CoW the tree nodes.
                                          // And the changes occur in-place.  Therefore, old_leafe_node should always be None.

        if lca_level == 0 {
            // Still accessing the same leaf node.
            // The leaf node should have a full occupancy mask.
            assert!(leaf_node.get_occupancy().is_maxed());
            self.attributes.set_attribute(
                &leaf_node.get_value(),
                leaf_node.get_offset(coords),
                value,
            );
            return;
        } else {
            // Release reference to leaf_node so that we can borrow prev_access_leaf_node.
            // Satefty: It has already been established that prev_access_leaf_node is not leaf_node, so it should be fine to have both mutable references.
            let leaf_node: *mut _ = leaf_node;
            if prev_access_leaf_node_ptr != u32::MAX {
                // purge prev access leaf node by fitting its attributes
                let prev_access_leaf_node = unsafe {
                    self.tree
                        .get_node_mut::<ROOT::LeafType>(prev_access_leaf_node_ptr)
                };
                assert_ne!(prev_access_leaf_node as *mut _, leaf_node);
                let old_attrib_ptr = *prev_access_leaf_node.get_value();
                assert!(prev_access_leaf_node.get_occupancy().is_maxed());
                assert_eq!(
                    prev_access_leaf_node.get_occupancy().count_ones(),
                    ROOT::LeafType::SIZE as u32
                );
                let maxed_attributes = self.attributes.get_attributes(
                    prev_access_leaf_node.get_value(),
                    ROOT::LeafType::SIZE as u32,
                );
                let mut new_mask = ATTRIBS::Occupancy::ZEROED;
                for (i, attr_value) in maxed_attributes.iter().enumerate() {
                    new_mask.set(i, attr_value.is_default());
                }
                if !new_mask.is_maxed() {
                    // fitting attributes by realloc and copy
                    let new_attrib_ptr = self.attributes.copy_attribute(
                        prev_access_leaf_node.get_value(),
                        &ATTRIBS::Occupancy::MAXED,
                        &new_mask,
                    );
                    prev_access_leaf_node.set_value(new_attrib_ptr);
                    self.attributes
                        .free_attributes(old_attrib_ptr, ROOT::LeafType::SIZE as u32);
                }
            }
            let leaf_node = unsafe { &mut *leaf_node };

            // Copy to a new leaf node with maxed occupancy.
            let new_attrib_ptr = if leaf_node.get_occupancy().is_maxed() {
                // Occupancy already maxed out.
                *leaf_node.get_value()
            } else {
                // trick for now: set the bit to false, then after copy attribute, set it back.
                leaf_node.set_local_occupancy_bit(coords & ROOT::LeafType::EXTENT_MASK, false);
                let old_attrib_occupancy = leaf_node.get_occupancy();
                let old_attrib_occupancy_count = old_attrib_occupancy.count_ones();

                let new_attrib_ptr = self.attributes.copy_attribute(
                    &leaf_node.get_value(),
                    leaf_node.get_occupancy(), // this original mask is wrong. should be old_attrib_occupancy
                    &ATTRIBS::Occupancy::MAXED,
                );
                // if old_attrib_occupancy.count_ones() > 0, do this.
                if old_attrib_occupancy_count > 0 {
                    self.attributes
                        .free_attributes(*leaf_node.get_value(), old_attrib_occupancy_count);
                }
                leaf_node.set_local_occupancy_bit(coords & ROOT::LeafType::EXTENT_MASK, true);

                // Hint: just need to get the old attrib_occupancy now.
                leaf_node.set_value(new_attrib_ptr);
                *leaf_node.get_occupancy_mut() = ATTRIBS::Occupancy::MAXED;
                new_attrib_ptr
            };

            self.attributes
                .set_attribute(&new_attrib_ptr, leaf_node.get_offset(coords), value);
            // TODO: change get_offset to a more straightforward way of calculation
        }
    }
}

pub trait Attributes {
    /// The type of the attribute pointer.
    /// The attribute pointers are stored on the vdb leaf nodes, one per node.
    /// This is typically u32.
    type Ptr;
    /// The occupancy mask of the attribute pointer.
    /// If we have 4x4x4 leaf nodes, this would be BitMask<64>.
    /// If we have 8x8x8 leaf nodes, this would be BitMask<512>.
    type Occupancy;
    /// The type of the attribute values. For a MagicaVoxel grid, this would be a u8 palette index.
    type Value: Default + IsDefault;
    fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value;
    fn get_attributes(&self, ptr: &Self::Ptr, len: u32) -> &[Self::Value];
    fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value);
    fn free_attributes(&mut self, ptr: Self::Ptr, num_attributes: u32);

    /// Allocate a new attribute range using the new mask. Then, copy the attributes from the attribute range
    /// pointed to by `ptr` to the newly allocated attribute range. Returns the pointer to the new attribute range.
    ///
    /// Only attribute values that are set in both the original mask and the new mask will be copied.
    ///
    /// The original attribute range will not be freed. It is the responsibility of the caller to free the original attribute range.
    ///
    /// Note that the original mask may be zeroed. In this case, `ptr` is meaningless, and the function will allocate
    /// a new attribute range without performing any copy.
    fn copy_attribute(
        &mut self,
        ptr: &Self::Ptr,
        original_mask: &Self::Occupancy,
        new_mask: &Self::Occupancy,
    ) -> Self::Ptr; // need a value to represent: what are the ones to delete, and what are the ones to add?
}

pub trait IsDefault {
    fn is_default(&self) -> bool;
}
impl<T> IsDefault for T
where
    T: Default + Eq,
{
    fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

impl<ROOT: Node> MutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
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
    ) -> Accessor<'a, ROOT, A, &'a mut MutableTree<ROOT>> {
        let mut metas: [MaybeUninit<NodeMeta<_>>; ROOT::LEVEL + 1] = MaybeUninit::uninit_array();
        let metas_src = Self::metas();
        assert_eq!(metas.len(), metas_src.len());
        for (dst, src) in metas.iter_mut().zip(metas_src.into_iter()) {
            dst.write(src);
        }
        Accessor {
            tree: self,
            ptrs: [u32::MAX; ROOT::LEVEL],
            metas: unsafe { MaybeUninit::array_assume_init(metas) },
            last_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            attributes,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;

    use super::{lowest_common_ancestor_level, Attributes};
    use crate::{hierarchy, BitMask, MutableTree, Node};

    struct TestAttributes;

    impl Attributes for TestAttributes {
        type Ptr = u32;
        type Occupancy = BitMask<64>;
        type Value = u8;

        fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value {
            0
        }

        fn get_attributes(&self, ptr: &Self::Ptr, len: u32) -> &[Self::Value] {
            &[]
        }

        fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value) {
            println!("set_attribute {:?} {:?} {:?}", ptr, offset, value);
        }

        fn free_attributes(&mut self, ptr: Self::Ptr, num_attributes: u32) {
            println!("free_attributes {:?} {:?}", ptr, num_attributes);
        }

        fn copy_attribute(
            &mut self,
            ptr: &Self::Ptr,
            original_mask: &Self::Occupancy,
            new_mask: &Self::Occupancy,
        ) -> Self::Ptr {
            println!(
                "copy_attribute {:?} {:?} {:?}",
                ptr, original_mask, new_mask
            );
            *ptr + 1
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
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        type MyTree = MutableTree<hierarchy!(2, 4, 2, u32)>;
        let mut tree = MyTree::new();

        let mut attributes = TestAttributes;
        let mut accessor = tree.accessor_mut(&mut attributes);

        accessor.set(UVec3::new(0, 0, 0), 12);

        accessor.set(UVec3::new(0, 1, 0), 13);

        accessor.set(UVec3::new(144, 1, 0), 14);
    }
}
