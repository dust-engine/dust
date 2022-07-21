use glam::UVec3;

use crate::{tree::TreeMeta, Node, NodeConst, Tree};

pub struct Accessor<'a, ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: &'a Tree<ROOT>,
    ptrs: [u32; ROOT::LEVEL],
    last_coords: Option<UVec3>,
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
{
    pub fn get(&mut self, coords: UVec3) -> Option<ROOT::Voxel>
    where
        ROOT: ~const NodeConst,
    {
        let result = if let Some(last_coords) = self.last_coords {
            let lca_level = lowest_common_ancestor_level(
                last_coords,
                coords,
                <Tree<ROOT> as TreeMeta<ROOT>>::META_MASK,
                ROOT::LEVEL as u32,
            );
            if lca_level == ROOT::LEVEL as u32 {
                self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
            } else {
                let meta = &<Tree<ROOT> as TreeMeta<ROOT>>::METAS[lca_level as usize];
                let ptr = self.ptrs[lca_level as usize];
                let new_coords = UVec3 {
                    x: coords.x & ((1_u32 << meta.extent_log2.x) - 1),
                    y: coords.y & ((1_u32 << meta.extent_log2.y) - 1),
                    z: coords.z & ((1_u32 << meta.extent_log2.z) - 1),
                };
                (meta.getter)(&self.tree.pool, new_coords, ptr, &mut self.ptrs)
            }
        } else {
            self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
        };
        self.last_coords = Some(coords);
        return result;
    }
}


pub struct AccessorMut<'a, ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    tree: &'a mut Tree<ROOT>,
    ptrs: [u32; ROOT::LEVEL],
    last_coords: Option<UVec3>,
}


impl<'a, ROOT: Node> AccessorMut<'a, ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    
    pub fn get(&mut self, coords: UVec3) -> Option<ROOT::Voxel>
    where
        ROOT: ~const NodeConst,
    {
        let result = if let Some(last_coords) = self.last_coords {
            let lca_level = lowest_common_ancestor_level(
                last_coords,
                coords,
                <Tree<ROOT> as TreeMeta<ROOT>>::META_MASK,
                ROOT::LEVEL as u32,
            );
            if lca_level == ROOT::LEVEL as u32 {
                self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
            } else {
                let meta = &<Tree<ROOT> as TreeMeta<ROOT>>::METAS[lca_level as usize];
                let ptr = self.ptrs[lca_level as usize];
                let new_coords = UVec3 {
                    x: coords.x & ((1_u32 << meta.extent_log2.x) - 1),
                    y: coords.y & ((1_u32 << meta.extent_log2.y) - 1),
                    z: coords.z & ((1_u32 << meta.extent_log2.z) - 1),
                };
                (meta.getter)(&self.tree.pool, new_coords, ptr, &mut self.ptrs)
            }
        } else {
            self.tree.root.get(&self.tree.pool, coords, &mut self.ptrs)
        };
        self.last_coords = Some(coords);
        return result;
    }
    pub fn set(&mut self, coords: UVec3, value: Option<ROOT::Voxel>)
    where
        ROOT: ~const NodeConst,
    {
        let result = if let Some(last_coords) = self.last_coords {
            let lca_level = lowest_common_ancestor_level(
                last_coords,
                coords,
                <Tree<ROOT> as TreeMeta<ROOT>>::META_MASK,
                ROOT::LEVEL as u32,
            );
            if lca_level == ROOT::LEVEL as u32 {
                self.tree.root.set(&mut self.tree.pool, coords, value, &mut self.ptrs)
            } else {
                let meta = &<Tree<ROOT> as TreeMeta<ROOT>>::METAS[lca_level as usize];
                let ptr = self.ptrs[lca_level as usize];
                let new_coords = UVec3 {
                    x: coords.x & ((1_u32 << meta.extent_log2.x) - 1),
                    y: coords.y & ((1_u32 << meta.extent_log2.y) - 1),
                    z: coords.z & ((1_u32 << meta.extent_log2.z) - 1),
                };
                (meta.setter)(&mut self.tree.pool, new_coords, ptr, value, &mut self.ptrs)
            }
        } else {
            self.tree.root.set(&mut self.tree.pool, coords, value, &mut self.ptrs)
        };
        self.last_coords = Some(coords);
        return result;
    }
}


impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    pub fn accessor(&self) -> Accessor<ROOT> {
        Accessor {
            tree: self,
            ptrs: [0; ROOT::LEVEL],
            last_coords: None,
        }
    }
    pub fn accessor_mut(&mut self) -> AccessorMut<ROOT> {
        AccessorMut {
            tree: self,
            ptrs: [0; ROOT::LEVEL],
            last_coords: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec3;

    use super::lowest_common_ancestor_level;
    use crate::{hierarchy, tree::TreeMeta, Tree};

    #[test]
    fn test() {
        type MyTree = Tree<hierarchy!(2, 4, 2)>;
        const MASK: UVec3 = MyTree::META_MASK;
        assert_eq!(
            MASK,
            UVec3 {
                x: 0b10100010,
                y: 0b10100010,
                z: 0b10100010
            }
        );
        assert_eq!(
            lowest_common_ancestor_level(
                UVec3::new(0, 0, 0),
                UVec3::new(0, 0, 0), // TODO: solve when same
                MASK,
                2
            ),
            0
        );
    }

    #[test]
    fn test_accessor() {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        type MyTree = Tree<hierarchy!(2, 4, 2)>;
        let mut tree = MyTree::new();

        let mut set_locations: Vec<UVec3> = Vec::with_capacity(100);
        for i in 0..100 {
            let x: u8 = rng.gen();
            let y: u8 = rng.gen();
            let z: u8 = rng.gen();
            let location = UVec3::new(x as u32, y as u32, z as u32);
            set_locations.push(location);
            tree.set_value(location, Some(true));
        }

        let mut accessor = tree.accessor();
        for location in set_locations.choose_multiple(&mut rng, 100) {
            let result = accessor.get(*location);
            assert_eq!(result, Some(true));
        }
    }
}
