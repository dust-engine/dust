use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{MutableTree, Node, NodeMeta};

pub struct Accessor<'a, ROOT: Node>
where
    [(); ROOT::LEVEL + 1]: Sized,
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: &'a MutableTree<ROOT>,
    ptrs: [u32; ROOT::LEVEL],
    metas: [NodeMeta; ROOT::LEVEL + 1],
    last_coords: UVec3,
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

impl<'a, ROOT: Node> Accessor<'a, ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
    [(); ROOT::LEVEL + 1]: Sized,
{
    #[inline]
    pub fn get(&mut self, coords: UVec3) -> bool
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
        let result = if lca_level >= ROOT::LEVEL as u32 {
            self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
        } else {
            let meta = &self.metas[lca_level as usize];
            let ptr = self.ptrs[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            (meta.getter)(&self.tree.pool, new_coords, ptr, &mut self.ptrs)
        };
        return result;
    }
}

pub struct AccessorMut<'a, ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
    [(); ROOT::LEVEL + 1]: Sized,
{
    tree: &'a mut MutableTree<ROOT>,
    ptrs: [u32; ROOT::LEVEL],
    metas: [NodeMeta; ROOT::LEVEL + 1],
    last_coords: UVec3,
}

impl<'a, ROOT: Node> AccessorMut<'a, ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
    [(); ROOT::LEVEL + 1]: Sized,
{
    #[inline]
    pub fn get(&mut self, coords: UVec3) -> bool
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
        let result = if lca_level >= ROOT::LEVEL as u32 {
            self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
        } else {
            let meta = &self.metas[lca_level as usize];
            let ptr = self.ptrs[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            (meta.getter)(&self.tree.pool, new_coords, ptr, &mut self.ptrs)
        };
        return result;
    }

    #[inline]
    pub fn set(&mut self, coords: UVec3, value: bool)
    where
        ROOT: Node,
    {
        if value {
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
        if lca_level >= ROOT::LEVEL as u32 {
            self.tree
                .root
                .set(&mut self.tree.pool, coords, value, &mut self.ptrs);
        } else {
            let meta = &self.metas[lca_level as usize];
            let new_coords = coords & meta.extent_mask;
            let ptr = self.ptrs[lca_level as usize];
            (meta.setter)(&mut self.tree.pool, new_coords, ptr, value, &mut self.ptrs);
        }
    }
}

impl<ROOT: Node> MutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
    [(); ROOT::LEVEL + 1]: Sized,
{
    pub fn accessor(&self) -> Accessor<ROOT> {
        let mut metas: [MaybeUninit<NodeMeta>; ROOT::LEVEL + 1] = MaybeUninit::uninit_array();
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
        }
    }
    pub fn accessor_mut(&mut self) -> AccessorMut<ROOT> {
        let mut metas: [MaybeUninit<NodeMeta>; ROOT::LEVEL + 1] = MaybeUninit::uninit_array();
        let metas_src = Self::metas();
        assert_eq!(metas.len(), metas_src.len());
        for (dst, src) in metas.iter_mut().zip(metas_src.into_iter()) {
            dst.write(src);
        }
        AccessorMut {
            tree: self,
            ptrs: [0; ROOT::LEVEL],
            metas: unsafe { MaybeUninit::array_assume_init(metas) },
            last_coords: UVec3::new(u32::MAX, u32::MAX, u32::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;

    use super::lowest_common_ancestor_level;
    use crate::{hierarchy, MutableTree, Node};

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

        let mut accessor = tree.accessor();
        for location in set_locations.choose_multiple(&mut rng, 100) {
            let result = accessor.get(*location);
            assert!(result);
        }
    }
}
