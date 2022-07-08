use std::{alloc::Layout, marker::PhantomData, mem::MaybeUninit};

use glam::UVec3;

use crate::{Node, Tree, Pool};

pub enum RootNodeEntry {
    Occupied(u32),
    Free(u32),
}

#[derive(PartialEq, Eq, Clone)]
pub struct RootKey(UVec3);
impl std::hash::Hash for RootKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let root_hash = (self.0.x as u64).wrapping_mul(73856093_u64)
            ^ (self.0.y as u64).wrapping_mul(19349663_u64)
            ^ (self.0.z as u64).wrapping_mul(83492791_u64);
        state.write_u64(root_hash);
    }
}

/// The root node of the tree implemented with a [`std::collections::HashMap`].
/// This enables trees of infinite size.
#[derive(Default)]
pub struct RootNode<CHILD: Node> {
    /// Map from [`RootKey`] to tiles.
    map: std::collections::HashMap<RootKey, RootNodeEntry, nohash::BuildNoHashHasher<u64>>,
    _marker: PhantomData<CHILD>,
}

impl<CHILD: Node> Node for RootNode<CHILD> {
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: 32,
        y: 32,
        z: 32,
    };

    const SIZE: usize = usize::MAX;

    const LEVEL: usize = CHILD::LEVEL + 1;

    fn new() -> Self {
        Self { map: Default::default(), _marker: PhantomData }
    }

    type Voxel = CHILD::Voxel;
    #[inline]
    fn get(&self, pools: &[Pool], coords: UVec3) -> Option<Self::Voxel>
    {
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let entry = self.map.get(&RootKey(root_offset));
        if let Some(entry) = entry {
            match entry {
                RootNodeEntry::Free(_material_id) => None,
                RootNodeEntry::Occupied(ptr) => unsafe {
                    let _child_node = pools[CHILD::LEVEL].get_item::<CHILD>(*ptr);
                    let new_coords = UVec3 {
                        x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                        y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                        z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
                    };
                    CHILD::get_in_pools(pools, new_coords, *ptr)
                },
            }
        } else {
            None
        }
    }

    fn set(&mut self, pools: &mut [Pool], coords: UVec3, value: Option<Self::Voxel>)
    {
        // ptr is meaningless and always 0 for root nodes.
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let key = RootKey(root_offset);

        if value.is_some() {
            // Ensure that the node contains stuff on ptr
            if !self.map.contains_key(&key) {
                let new_node_ptr = unsafe { pools[CHILD::LEVEL].alloc() };
                self.map
                    .insert(key.clone(), RootNodeEntry::Occupied(new_node_ptr));
            }

            let child_ptr = match self.map.get(&key).unwrap() {
                RootNodeEntry::Occupied(ptr) => *ptr,
                RootNodeEntry::Free(_) => todo!(),
            };
            let new_coords = UVec3 {
                x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
            };
            CHILD::set_in_pools(pools, new_coords, child_ptr, value)
        }
    }

    fn write_layout(sizes: &mut [MaybeUninit<Layout>]) {
        CHILD::write_layout(sizes);
    }

    fn get_in_pools(pools: &[Pool], coords: UVec3, ptr: u32) -> Option<Self::Voxel> {
        unreachable!("Root Node is never kept in a pool!")
    }

    fn set_in_pools(pool: &mut [Pool], coords: UVec3, ptr: u32, value: Option<Self::Voxel>) {
        unreachable!("Root Node is never kept in a pool!")
    }
/*
    type Iterator<'a> = RootIterator<'a, ROOT, CHILD>;

    fn iter<'a>(tree: &'a Tree<ROOT>, ptr: u32, offset: UVec3) -> Self::Iterator<'a>
        where [(); ROOT::LEVEL as usize]: Sized {
        todo!()
    }
    */
}

/*
pub struct RootIterator<'a, ROOT: Node<ROOT>, CHILD: Node<ROOT>>  where [(); ROOT::LEVEL as usize]: Sized {
    tree: &'a Tree<ROOT>,
    map_iterator: std::collections::hash_map::Iter<'a, RootKey, RootNodeEntry>,
    child_iterator: Option<CHILD::Iterator<'a>>,
}

impl<'a, ROOT: Node<ROOT>, CHILD: Node<ROOT>> Iterator for RootIterator<'a, ROOT, CHILD>  where [(); ROOT::LEVEL as usize]: Sized {
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<ROOT: Node<ROOT>, CHILD: Node<ROOT>> RootNode<ROOT, CHILD> {
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::with_hasher(nohash::BuildNoHashHasher::<u64>::default()),
            _marker: PhantomData,
        }
    }
}
*/