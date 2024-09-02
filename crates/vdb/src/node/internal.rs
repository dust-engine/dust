use super::{size_of_grid, IsLeaf, NodeMeta};
use crate::{bitmask::SetBitIterator, BitMask, ConstUVec3, Node, Pool};
use glam::UVec3;
use std::{cell::UnsafeCell, marker::PhantomData, mem::size_of, result};

#[derive(Clone, Copy)]
pub union InternalNodeEntry {
    /// The corresponding bit on child_mask is set. Points to another node.
    pub occupied: u32,
    /// The corresponding bit on child_mask is not set.
    /// Points to a value in the material array that describes all child nodes within the current node.
    /// If this is u32::MAX, this is air.
    pub free: u32,
}

/// Internal nodes can be 2*2*2.
/// Size: 8 byte (mask) + 32 byte + 16 bytes for stats
#[repr(C)]
pub struct InternalNode<CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// This is 0 if that tile is completely air, and 1 otherwise.
    pub child_mask: BitMask<{ size_of_grid(FANOUT_LOG2) }>,

    /// points to self.child_mask.count_ones() LeafNodes or InternalNodes
    pub child_ptrs: [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],

    _marker: PhantomData<CHILD>,
}
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Default for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn default() -> Self {
        Self {
            child_mask: Default::default(),
            child_ptrs: [InternalNodeEntry { free: 0 }; size_of_grid(FANOUT_LOG2)],
            _marker: Default::default(),
        }
    }
}
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> Node for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    type LeafType = CHILD::LeafType;
    const SIZE: usize = size_of_grid(FANOUT_LOG2);
    const EXTENT_LOG2: UVec3 = UVec3 {
        x: FANOUT_LOG2.x + CHILD::EXTENT_LOG2.x,
        y: FANOUT_LOG2.y + CHILD::EXTENT_LOG2.y,
        z: FANOUT_LOG2.z + CHILD::EXTENT_LOG2.z,
    };
    const EXTENT: UVec3 = UVec3 {
        x: 1 << Self::EXTENT_LOG2.x,
        y: 1 << Self::EXTENT_LOG2.y,
        z: 1 << Self::EXTENT_LOG2.z,
    };
    const EXTENT_MASK: UVec3 = UVec3 {
        x: Self::EXTENT.x - 1,
        y: Self::EXTENT.y - 1,
        z: Self::EXTENT.z - 1,
    };
    const META_MASK: UVec3 = UVec3 {
        x: CHILD::META_MASK.x | (1 << (Self::EXTENT_LOG2.x - 1)),
        y: CHILD::META_MASK.y | (1 << (Self::EXTENT_LOG2.y - 1)),
        z: CHILD::META_MASK.z | (1 << (Self::EXTENT_LOG2.z - 1)),
    };
    const LEVEL: usize = CHILD::LEVEL + 1;

    fn set<'a>(
        &'a mut self,
        pools: &'a mut [Pool],
        coords: UVec3,
        value: bool,
        cached_path: &mut [u32],
        touched_nodes: Option<&mut Vec<(u32, u32)>>,
    ) -> (Option<&'a mut Self::LeafType>, &'a mut Self::LeafType) {
        let internal_offset = coords >> CHILD::EXTENT_LOG2;
        let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
            | ((internal_offset.y as usize) << FANOUT_LOG2.z)
            | (internal_offset.z as usize);
        if value {
            // set
            let has_child = self.child_mask.get(index);
            if !has_child {
                unsafe {
                    // ensure have children
                    let allocated_child_ptr = pools[CHILD::LEVEL].alloc::<CHILD>();
                    self.child_mask.set(index, true);

                    // allocate a child node
                    self.child_ptrs[index].occupied = allocated_child_ptr;
                }
            }
            // TODO: propagate when filled.
        } else {
            // clear
            todo!() // TODO: clear recursively, propagate if completely cleared
        }
        let new_coords = coords & CHILD::EXTENT_MASK;
        let child_ptr = unsafe { &mut self.child_ptrs[index].occupied };
        <CHILD as Node>::set_in_pools(
            pools,
            new_coords,
            child_ptr,
            value,
            cached_path,
            touched_nodes,
        )
    }
    #[inline]
    fn set_in_pools<'a>(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        value: bool,
        cached_path: &mut [u32],
        mut touched_nodes: Option<&mut Vec<(u32, u32)>>,
    ) -> (Option<&'a mut Self::LeafType>, &'a mut Self::LeafType) {
        unsafe {
            let mut node: *mut _ = pools[Self::LEVEL].get_item_mut::<Self>(*ptr);

            let internal_offset = coords >> CHILD::EXTENT_LOG2;
            let index = ((internal_offset.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
                | ((internal_offset.y as usize) << FANOUT_LOG2.z)
                | (internal_offset.z as usize);
            if value {
                // set
                let has_child = (&mut *node).child_mask.get(index);
                if !has_child {
                    // ensure have children
                    let allocated_child_ptr = pools[CHILD::LEVEL].alloc::<CHILD>();

                    if let Some(touched_nodes) = touched_nodes.as_mut() {
                        let new_node_ptr = pools[Self::LEVEL].alloc_uninitialized();
                        let new_node = pools[Self::LEVEL].get_item_mut::<Self>(new_node_ptr);
                        std::ptr::copy_nonoverlapping(node, new_node, 1);
                        // allocate a child node
                        touched_nodes.push((Self::LEVEL as u32, *ptr));
                        *ptr = new_node_ptr;
                        node = new_node;
                    }
                    (&mut *node).child_mask.set(index, true);
                    (&mut *node).child_ptrs[index].occupied = allocated_child_ptr;
                }
                // TODO: propagate when filled.
            } else {
                // clear
                todo!() // TODO: clear recursively, propagate if completely cleared
            }
            if cached_path.len() > 0 {
                cached_path[Self::LEVEL] = *ptr;
            }
            let new_coords = coords & CHILD::EXTENT_MASK;
            let child_ptr = &mut (&mut *node).child_ptrs[index].occupied;
            <CHILD as Node>::set_in_pools(
                pools,
                new_coords,
                child_ptr,
                value,
                cached_path,
                touched_nodes,
            )
        }
    }

    type Iterator<'a> = InternalNodeIterator<'a, CHILD, FANOUT_LOG2>;
    #[inline]
    fn iter<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::Iterator<'a> {
        InternalNodeIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: self.child_mask.iter_set_bits(),
            child_ptrs: &self.child_ptrs,
            child_iterator: None,
        }
    }
    #[inline]
    fn iter_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::Iterator<'a> {
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        InternalNodeIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: node.child_mask.iter_set_bits(),
            child_ptrs: &node.child_ptrs,
            child_iterator: None,
        }
    }

    type LeafIterator<'a> = InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>;

    #[inline]
    fn iter_leaf<'a>(&'a self, pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a> {
        InternalNodeLeafIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: self.child_mask.iter_set_bits(),
            child_ptrs: &self.child_ptrs,
            child_iterator: None,
        }
    }

    #[inline]
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a> {
        let node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        InternalNodeLeafIterator {
            pools,
            location_offset: offset,
            child_mask_iterator: node.child_mask.iter_set_bits(),
            child_ptrs: &node.child_ptrs,
            child_iterator: None,
        }
    }
    fn write_meta(metas: &mut Vec<NodeMeta<Self::LeafType>>) {
        CHILD::write_meta(metas);
        metas.push(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            setter: Self::set_in_pools,
            extent_log2: Self::EXTENT_LOG2,
            extent_mask: Self::EXTENT_MASK,
            fanout_log2: FANOUT_LOG2.to_glam(),
        });
    }

    /*
    #[cfg(feature = "physics")]
    #[inline]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        solid: bool,
        initial_intersection_t: glam::Vec2,
        pools: &[Pool],
    ) -> Option<parry3d::query::RayIntersection> {
        // Assume that the node is located at 0.0 - 4.0

        use glam::{IVec3, Vec2, Vec3, Vec3A, Vec3Swizzles};
        let mut hit_distance: f32 = initial_intersection_t.x;
        let initial_intersection_point: Vec3A = (ray.origin + ray.dir * hit_distance).into();
        let fanout: UVec3 = UVec3::new(
            1 << FANOUT_LOG2.x as u32,
            1 << FANOUT_LOG2.y as u32,
            1 << FANOUT_LOG2.z as u32,
        );
        let mut position: IVec3 =
            Vec3::from((initial_intersection_point * fanout.as_vec3a()).floor())
                .as_ivec3()
                .clamp(IVec3::splat(0), fanout.as_ivec3() - IVec3::splat(1));

        let t_coef: Vec3A = 1.0 / Vec3A::from(ray.dir);
        let t_bias: Vec3A = t_coef * Vec3A::from(ray.origin);

        let step = Vec3A::from(ray.dir).signum();

        let mut t_max: Vec3A =
            ((position.as_vec3a() / fanout.as_vec3a()) + step.max(Vec3A::ZERO)) * t_coef - t_bias;

        let t_delta = (1.0 / fanout.as_vec3a()) * t_coef * step; // one to (1 / fanout)?

        loop {
            let comp_result = Vec3A::select(t_max.zxy().cmplt(t_max), Vec3A::ZERO, Vec3A::ONE)
                * Vec3A::select(t_max.yzx().cmplt(t_max), Vec3A::ZERO, Vec3A::ONE);
            let next_t_max = t_max + t_delta * comp_result;
            let index = ((position.x as usize) << (FANOUT_LOG2.y + FANOUT_LOG2.z))
                | ((position.y as usize) << FANOUT_LOG2.z)
                | (position.z as usize);

            let has_child = self.child_mask.get(index);
            if has_child {
                let child = unsafe {
                    let child_ptr = self.child_ptrs[index].occupied;
                    pools[Self::LEVEL - 1].get_item::<CHILD>(child_ptr)
                };
                let mut new_ray = parry3d::query::Ray {
                    dir: ray.dir.component_mul(&fanout.as_vec3().into()),
                    origin: ray.origin,
                };
                new_ray.origin.coords = new_ray
                    .origin
                    .coords
                    .component_mul(&fanout.as_vec3().into())
                    - parry3d::math::Vector::from(position.as_vec3());
                if let Some(hit) = child.cast_local_ray_and_get_normal(
                    &new_ray,
                    solid,
                    Vec2::new(
                        hit_distance,
                        t_max
                            .x
                            .min(t_max.y)
                            .min(t_max.z)
                            .min(initial_intersection_t.y),
                    ),
                    pools,
                ) {
                    // hit_distance problematic here
                    return Some(hit);
                }
            }

            // Move to the next voxel
            let position_delta = (step * comp_result).as_ivec3();
            position += position_delta;
            hit_distance = t_max.x.min(t_max.y).min(t_max.z);
            if hit_distance + 0.001 >= initial_intersection_t.y {
                return None;
            }
            t_max = next_t_max;
        }
    }
    */
}

/// When the alternate flag was specified, also print the child pointers.
impl<CHILD: Node, const FANOUT_LOG2: ConstUVec3> std::fmt::Debug
    for InternalNode<CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Internal Node\n")?;
        self.child_mask.fmt(f)?;
        Ok(())
    }
}

pub struct InternalNodeIterator<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    pools: &'a [Pool],
    location_offset: UVec3,
    child_mask_iterator: SetBitIterator<std::iter::Cloned<std::slice::Iter<'a, usize>>>,
    child_iterator: Option<CHILD::Iterator<'a>>,
    child_ptrs: &'a [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
}
impl<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3> Iterator
    for InternalNodeIterator<'a, CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            if let Some(next_child_index) = self.child_mask_iterator.next() {
                let child_ptr = unsafe { self.child_ptrs[next_child_index].occupied };
                let offset = UVec3 {
                    x: next_child_index as u32 >> (FANOUT_LOG2.z + FANOUT_LOG2.y),
                    y: (next_child_index as u32 >> FANOUT_LOG2.z) & ((1 << FANOUT_LOG2.y) - 1),
                    z: next_child_index as u32 & ((1 << FANOUT_LOG2.z) - 1),
                };
                let offset = offset * CHILD::EXTENT;
                self.child_iterator = Some(CHILD::iter_in_pool(
                    self.pools,
                    child_ptr,
                    self.location_offset + offset,
                ));
                continue;
            } else {
                // Also ran out. We have nothing left.
                return None;
            }
        }
    }
}

pub struct InternalNodeLeafIterator<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    pools: &'a [Pool],
    location_offset: UVec3,
    child_mask_iterator: SetBitIterator<std::iter::Cloned<std::slice::Iter<'a, usize>>>,
    child_iterator: Option<CHILD::LeafIterator<'a>>,
    child_ptrs: &'a [InternalNodeEntry; size_of_grid(FANOUT_LOG2)],
}
impl<'a, CHILD: Node, const FANOUT_LOG2: ConstUVec3> Iterator
    for InternalNodeLeafIterator<'a, CHILD, FANOUT_LOG2>
where
    [(); size_of_grid(FANOUT_LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Item = (UVec3, &'a UnsafeCell<CHILD::LeafType>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try taking it out from the current child
            if let Some(item) = self.child_iterator.as_mut().and_then(|a| a.next()) {
                return Some(item);
            }
            // self.child_iterator is None or ran out. Grab the next child.
            if let Some(next_child_index) = self.child_mask_iterator.next() {
                let child_ptr = unsafe { self.child_ptrs[next_child_index].occupied };
                let offset = UVec3 {
                    x: next_child_index as u32 >> (FANOUT_LOG2.z + FANOUT_LOG2.y),
                    y: (next_child_index as u32 >> FANOUT_LOG2.z) & ((1 << FANOUT_LOG2.y) - 1),
                    z: next_child_index as u32 & ((1 << FANOUT_LOG2.z) - 1),
                };
                let offset = offset * CHILD::EXTENT;
                self.child_iterator = Some(CHILD::iter_leaf_in_pool(
                    self.pools,
                    child_ptr,
                    self.location_offset + offset,
                ));
                continue;
            } else {
                // Also ran out. We have nothing left.
                return None;
            }
        }
    }
}
