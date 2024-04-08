use std::{cell::UnsafeCell, marker::PhantomData, mem::MaybeUninit};

use glam::UVec3;

use crate::{Node, Pool};

use super::NodeMeta;

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
    type LeafType = CHILD::LeafType;
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: 32,
        y: 32,
        z: 32,
    };
    const EXTENT: UVec3 = UVec3 {
        x: u32::MAX,
        y: u32::MAX,
        z: u32::MAX,
    };
    const EXTENT_MASK: UVec3 = Self::EXTENT;
    const META_MASK: UVec3 = CHILD::META_MASK;

    const SIZE: usize = usize::MAX;

    const LEVEL: usize = CHILD::LEVEL + 1;

    fn new() -> Self {
        Self {
            map: Default::default(),
            _marker: PhantomData,
        }
    }

    type Voxel = CHILD::Voxel;
    #[inline]
    fn get(&self, pools: &[Pool], coords: UVec3, cached_path: &mut [u32]) -> Option<Self::Voxel> {
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
                    CHILD::get_in_pools(pools, new_coords, *ptr, cached_path)
                },
            }
        } else {
            None
        }
    }

    fn set(
        &mut self,
        pools: &mut [Pool],
        coords: UVec3,
        value: Option<Self::Voxel>,
        cached_path: &mut [u32],
    ) {
        // ptr is meaningless and always 0 for root nodes.
        let root_offset = coords >> CHILD::EXTENT_LOG2;
        let key = RootKey(root_offset);

        if value.is_some() {
            // Ensure that the node contains stuff on ptr
            if !self.map.contains_key(&key) {
                let new_node_ptr = unsafe { pools[CHILD::LEVEL].alloc::<CHILD>() };
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
            CHILD::set_in_pools(pools, new_coords, child_ptr, value, cached_path)
        }
    }

    fn get_in_pools(
        _pools: &[Pool],
        _coords: UVec3,
        _ptr: u32,
        _cached_path: &mut [u32],
    ) -> Option<Self::Voxel> {
        unreachable!("Root Node is never kept in a pool!")
    }

    fn set_in_pools(
        _pools: &mut [Pool],
        _coords: UVec3,
        _ptr: u32,
        _value: Option<Self::Voxel>,
        _cached_path: &mut [u32],
    ) {
        unreachable!("Root Node is never kept in a pool!")
    }
    type Iterator<'a> = RootIterator<'a, CHILD>;
    fn iter<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::Iterator<'a> {
        RootIterator {
            pools,
            map_iterator: self.map.iter(),
            child_iterator: None,
            location_offset: offset,
        }
    }
    fn iter_in_pool<'a>(_pools: &'a [Pool], _ptr: u32, _offset: UVec3) -> Self::Iterator<'a> {
        unreachable!("Root Node is never kept in a pool!")
    }

    type LeafIterator<'a> = RootLeafIterator<'a, CHILD>;

    fn iter_leaf<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a> {
        RootLeafIterator {
            pools,
            map_iterator: self.map.iter(),
            child_iterator: None,
            location_offset: offset,
        }
    }

    fn iter_leaf_in_pool<'a>(
        _pools: &'a [Pool],
        _ptr: u32,
        _offset: UVec3,
    ) -> Self::LeafIterator<'a> {
        unreachable!("Root Node is never kept in a pool!")
    }
    fn write_meta(metas: &mut Vec<NodeMeta<Self::Voxel>>) {
        CHILD::write_meta(metas);
        metas.push(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            getter: Self::get_in_pools,
            setter: Self::set_in_pools,
            extent_log2: Self::EXTENT_LOG2,
            fanout_log2: Self::EXTENT_LOG2,
            extent_mask: Self::EXTENT_MASK,
        });
    }
}

impl<CHILD: Node> std::fmt::Debug for RootNode<CHILD> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RootNode")
    }
}

pub struct RootIterator<'a, CHILD: Node> {
    pools: &'a [Pool],
    map_iterator: std::collections::hash_map::Iter<'a, RootKey, RootNodeEntry>,
    child_iterator: Option<CHILD::Iterator<'a>>,
    location_offset: UVec3,
}

impl<'a, CHILD: Node> Iterator for RootIterator<'a, CHILD> {
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            if let Some((_, root_node)) = self.map_iterator.next() {
                match root_node {
                    RootNodeEntry::Occupied(ptr) => {
                        self.child_iterator =
                            Some(CHILD::iter_in_pool(self.pools, *ptr, self.location_offset));
                        continue;
                    }
                    _ => {
                        return None;
                    }
                }
            } else {
                // Also ran out. We have nothing left.
                return None;
            }
        }
    }
}
pub struct RootLeafIterator<'a, CHILD: Node> {
    pools: &'a [Pool],
    map_iterator: std::collections::hash_map::Iter<'a, RootKey, RootNodeEntry>,
    child_iterator: Option<CHILD::LeafIterator<'a>>,
    location_offset: UVec3,
}

impl<'a, CHILD: Node> Iterator for RootLeafIterator<'a, CHILD> {
    type Item = (UVec3, &'a UnsafeCell<CHILD::LeafType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            if let Some((_, root_node)) = self.map_iterator.next() {
                match root_node {
                    RootNodeEntry::Occupied(ptr) => {
                        self.child_iterator = Some(CHILD::iter_leaf_in_pool(
                            self.pools,
                            *ptr,
                            self.location_offset,
                        ));
                        continue;
                    }
                    _ => {
                        return None;
                    }
                }
            } else {
                // Also ran out. We have nothing left.
                return None;
            }
        }
    }
}
