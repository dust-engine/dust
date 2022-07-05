use std::{alloc::Layout, marker::PhantomData, mem::MaybeUninit};

use glam::UVec3;

use crate::{Node, Tree};

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

    const LEVEL: u8 = CHILD::LEVEL + 1;

    fn new() -> Self {
        todo!()
    }

    type Voxel = CHILD::Voxel;
    #[inline]
    fn get<ROOT: Node>(tree: &Tree<ROOT>, coords: UVec3, ptr: u32) -> Option<Self::Voxel>
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let node = unsafe { tree.get_node::<Self>(ptr) };
        let entry = node.map.get(&RootKey(root_offset));
        if let Some(entry) = entry {
            match entry {
                RootNodeEntry::Free(_material_id) => None,
                RootNodeEntry::Occupied(ptr) => unsafe {
                    let _child_node = tree.get_node::<CHILD>(*ptr);
                    let new_coords = UVec3 {
                        x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                        y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                        z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
                    };
                    <CHILD as Node>::get(tree, new_coords, *ptr)
                },
            }
        } else {
            None
        }
    }

    fn set<ROOT: Node>(tree: &mut Tree<ROOT>, coords: UVec3, ptr: u32, value: Option<Self::Voxel>)
    where
        [(); ROOT::LEVEL as usize]: Sized,
    {
        // ptr is meaningless and always 0 for root nodes.
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let key = RootKey(root_offset);

        if value.is_some() {
            // Ensure that the node contains stuff on ptr
            let node = unsafe { tree.get_node_mut::<Self>(ptr) };
            if !node.map.contains_key(&key) {
                let new_node_ptr = unsafe { tree.alloc_node::<CHILD>() };
                let node = unsafe { tree.get_node_mut::<Self>(ptr) };
                node.map
                    .insert(key.clone(), RootNodeEntry::Occupied(new_node_ptr));
            }

            let node = unsafe { tree.get_node_mut::<Self>(ptr) };
            let child_ptr = match node.map.get(&key).unwrap() {
                RootNodeEntry::Occupied(ptr) => *ptr,
                RootNodeEntry::Free(_) => todo!(),
            };
            let new_coords = UVec3 {
                x: coords.x & ((1_u32 << CHILD::EXTENT_LOG2.x) - 1),
                y: coords.y & ((1_u32 << CHILD::EXTENT_LOG2.y) - 1),
                z: coords.z & ((1_u32 << CHILD::EXTENT_LOG2.z) - 1),
            };
            CHILD::set(tree, new_coords, child_ptr, value)
        }
    }

    fn write_layout<ROOT: Node>(sizes: &mut [MaybeUninit<Layout>]) {
        CHILD::write_layout::<ROOT>(sizes);
    }
}

impl<CHILD: Node> RootNode<CHILD> {
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::with_hasher(nohash::BuildNoHashHasher::<u64>::default()),
            _marker: PhantomData,
        }
    }
}
