use std::{borrow::BorrowMut, mem::MaybeUninit, sync::MutexGuard};

use crate::pool::Pool;
use bevy::reflect::Access;
use glam::UVec3;

use crate::{bitmask::IsBitMask, IsLeaf, Node, NodeMeta, Tree};

pub struct Accessor<'a, ROOT: Node, ATTRIBS>
where
    [(); ROOT::LEVEL + 1]: Sized,
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: &'a mut Tree<ROOT>,
    ptrs: [u32; ROOT::LEVEL],
    metas: [NodeMeta<ROOT::LeafType>; ROOT::LEVEL + 1],
    last_coords: UVec3,
    attributes: &'a mut ATTRIBS,
    last_leaf: Option<u32>,
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
    [(); ROOT::LEVEL as usize + 1]: Sized,
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
            let meta = &self.metas[lca_level as usize];
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
            self.last_coords,
            coords,
            ROOT::META_MASK,
            ROOT::LEVEL as u32,
        );
        self.last_coords = coords;
        let leaf_node = if lca_level >= ROOT::LEVEL as u32 {
            self.tree.root.set(
                &mut self.tree.pool,
                coords,
                !value.is_default(),
                &mut self.ptrs,
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
            )
        };

        if let Some(last_leaf) = self.last_leaf {
            if last_leaf == self.ptrs[0] {
                // Still accessing the same leaf node.
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
                &ATTRIBS::Occupancy::MAXED,
            );
            self.last_leaf = Some(self.ptrs[0]);
            // if old_attrib_occupancy.count_ones() > 0, free.
            let old_attrib_occupancy_count = leaf_node.get_occupancy().count_ones(); // can optimize here
            if old_attrib_occupancy_count > 0 {
                self.attributes
                    .free_attributes(*leaf_node.get_value(), old_attrib_occupancy_count);
            }
            leaf_node.set_occupancy_at(coords, true);

            // Hint: just need to get the old attrib_occupancy now.
            leaf_node.set_value(new_attrib_ptr);
            self.attributes.set_attribute(
                &new_attrib_ptr,
                <ROOT::LeafType as IsLeaf>::get_fully_mapped_offset(coords),
                value,
            );
        };
    }

    fn purge_prev_access_leaf_node(&mut self) {
        if let Some(last_leaf) = self.last_leaf {
            // purge prev access leaf node by fitting its attributes
            let prev_access_leaf_node =
                unsafe { self.tree.get_node_mut::<ROOT::LeafType>(last_leaf) };
            let old_attrib_ptr = *prev_access_leaf_node.get_value();
            if !prev_access_leaf_node.get_occupancy().is_maxed() {
                // fitting attributes by realloc and copy
                let new_attrib_ptr = self.attributes.copy_attribute(
                    &old_attrib_ptr,
                    &ATTRIBS::Occupancy::MAXED,
                    prev_access_leaf_node.get_occupancy(),
                );
                prev_access_leaf_node.set_value(new_attrib_ptr);
                self.attributes
                    .free_attributes(old_attrib_ptr, ROOT::LeafType::SIZE as u32);
            }
            self.last_leaf = None;
        }
    }
    pub fn end(mut self) {
        self.purge_prev_access_leaf_node();
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

impl<ROOT: Node> Tree<ROOT>
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
    ) -> Accessor<'a, ROOT, A> {
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
            last_leaf: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;

    use super::{lowest_common_ancestor_level, Attributes};
    use crate::{hierarchy, BitMask, Node, Tree};

    #[derive(Default)]
    struct TestAttributes {
        attribute_maps: Vec<Vec<u8>>,
    }

    impl Attributes for TestAttributes {
        type Ptr = u32;
        type Occupancy = BitMask<64>;
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

        fn free_attributes(&mut self, ptr: Self::Ptr, num_attributes: u32) {
            println!("free {} attributes: {}", num_attributes, ptr);
            let slice = &self.attribute_maps[ptr as usize];
            assert_eq!(slice.len(), num_attributes as usize);
            self.attribute_maps[ptr as usize] = Vec::new();
        }

        fn copy_attribute(
            &mut self,
            ptr: &Self::Ptr,
            original_mask: &Self::Occupancy,
            new_mask: &Self::Occupancy,
        ) -> Self::Ptr {
            if original_mask.is_zeroed() {
                let new = vec![0; new_mask.count_ones() as usize];
                self.attribute_maps.push(new);
                println!(
                    "copy_attribute from null to {}: {} -> {}",
                    self.attribute_maps.len(),
                    original_mask.count_ones(),
                    new_mask.count_ones()
                );
                return self.attribute_maps.len() as u32 - 1;
            }
            let mut new = vec![0; new_mask.count_ones() as usize];
            let old = &self.attribute_maps[*ptr as usize];
            let mut new_ptr = 0;
            let mut old_ptr = 0;
            for bit in (original_mask | new_mask).iter_set_bits() {
                if new_mask.get(bit) && original_mask.get(bit) {
                    // copy it over
                    new[new_ptr] = old[old_ptr as usize];
                }
                if new_mask.get(bit) {
                    new_ptr += 1;
                }
                if original_mask.get(bit) {
                    old_ptr += 1;
                }
            }
            println!(
                "copy_attribute from {} to {}: {} -> {}",
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

        accessor.set(UVec3::new(0, 0, 0), 12);
        // Allocates full map for additional attributes
        assert_eq!(accessor.attributes.attribute_maps[0].len(), 64);
        assert_eq!(accessor.get(UVec3::new(0, 0, 0)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 1, 5)), None);

        accessor.set(UVec3::new(0, 1, 0), 13);
        // Subsequent ops in the same leaf node should not allocate
        assert_eq!(accessor.attributes.attribute_maps[0].len(), 64);
        assert_eq!(accessor.attributes.attribute_maps.len(), 1);
        assert_eq!(accessor.get(UVec3::new(0, 0, 0)), Some(12));
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
        assert_eq!(accessor.get(UVec3::new(0, 0, 0)), Some(12));
        assert_eq!(accessor.get(UVec3::new(0, 1, 0)), Some(13));

        accessor.end();
    }
}
