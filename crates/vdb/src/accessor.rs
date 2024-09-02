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
        let (old_leaf_node, new_leaf_node) = if lca_level >= ROOT::LEVEL as u32 {
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
        assert!(old_leaf_node.is_none());

        let mut attrib_ptr = *new_leaf_node.get_value();
        if !new_leaf_node.get_occupancy().is_maxed() {
            attrib_ptr = self.attributes.copy_attribute(
                new_leaf_node.get_value(),
                new_leaf_node.get_occupancy(),
                &ATTRIBS::Occupancy::MAXED,
            );
            new_leaf_node.set_value(attrib_ptr);
        }
        self.attributes
            .set_attribute(&attrib_ptr, new_leaf_node.get_offset(coords), value);
        // TODO: change get_offset to a more straightforward way of calculation
    }
}

pub trait Attributes {
    type Ptr;
    type Occupancy;
    type Value: Default + IsDefault;
    fn get_attribute(&self, ptr: &Self::Ptr, offset: u32) -> Self::Value;
    fn set_attribute(&mut self, ptr: &Self::Ptr, offset: u32, value: Self::Value);
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
            ptrs: [0; ROOT::LEVEL],
            metas: unsafe { MaybeUninit::array_assume_init(metas) },
            last_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
            attributes,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;

    use super::lowest_common_ancestor_level;
    use crate::{accessor::EmptyAttributes, hierarchy, MutableTree, Node};

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

        let mut set_locations: Vec<UVec3> = Vec::with_capacity(100);
        for _i in 0..100 {
            let x: u8 = rng.gen();
            let y: u8 = rng.gen();
            let z: u8 = rng.gen();
            let location = UVec3::new(x as u32, y as u32, z as u32);
            set_locations.push(location);
            tree.set_value(location, true);
        }
        let mut empty_attributes = EmptyAttributes::<u32>::default();

        let mut accessor = tree.accessor(&mut empty_attributes);
        for location in set_locations.choose_multiple(&mut rng, 100) {
            let result = accessor.get(*location);
            assert!(result.is_some());
        }
    }
}
