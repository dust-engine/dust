use super::{size_of_grid, NodeMeta};
use crate::{
    bitmask::{IsBitMask, SetBitIterator},
    BitMask, ConstUVec3, Node, Pool,
};
use glam::UVec3;
use std::{cell::UnsafeCell, iter::Once, mem::size_of};

/// Nodes are always 4x4x4 so that each leaf node contains exactly 64 voxels,
/// so that the occupancy mask happens to be exactly 64 bits.
/// Size: 3 u32
#[repr(C)]
#[derive(Default, Clone)]
pub struct LeafNode<const LOG2: ConstUVec3, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// This is 1 for occupied voxels and 0 for unoccupied voxels
    pub occupancy: BitMask<{ size_of_grid(LOG2) }>,
    /// A pointer to self.occupancy.count_ones() material values
    pub value: T,
}

pub trait IsLeaf: Node {
    /// Total number of voxels in the leaf node.
    type Occupancy: IsBitMask;
    type Value: Default + Send + Sync + PartialEq + Eq + Clone + Copy;
    fn get_occupancy(&self) -> &Self::Occupancy;
    fn get_occupancy_mut(&mut self) -> &mut Self::Occupancy;

    fn get_occupancy_at(&self, coords: UVec3) -> bool {
        self.get_occupancy()
            .get(Self::get_fully_mapped_offset(coords) as usize)
    }
    fn set_occupancy_at(&mut self, coords: UVec3, value: bool) {
        let offset = Self::get_fully_mapped_offset(coords);
        self.get_occupancy_mut().set(offset as usize, value);
    }

    fn get_value(&self) -> &Self::Value;
    fn set_value(&mut self, value: Self::Value);

    fn get_attribute_offset(&self, coords: UVec3) -> u32;
    fn get_fully_mapped_offset(coords: UVec3) -> u32;
}

impl<const LOG2: ConstUVec3, T: Copy + Eq + Send + Sync + 'static + Default> IsLeaf
    for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Value = T;
    type Occupancy = BitMask<{ size_of_grid(LOG2) }>;
    fn get_attribute_offset(&self, coords: UVec3) -> u32 {
        let coords = coords & Self::EXTENT_MASK;
        let voxel_id = (coords.x << (LOG2.y + LOG2.z)) | (coords.y << LOG2.z) | coords.z;
        let mask: usize = self.occupancy.as_slice()[0];
        let masked = mask & ((1 << voxel_id) - 1);
        masked.count_ones()
    }

    fn get_fully_mapped_offset(coords: UVec3) -> u32 {
        let coords = coords & Self::EXTENT_MASK;
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        index as u32
    }

    fn get_value(&self) -> &Self::Value {
        &self.value
    }
    fn set_value(&mut self, value: Self::Value) {
        self.value = value;
    }
    fn get_occupancy(&self) -> &Self::Occupancy {
        &self.occupancy
    }
    fn get_occupancy_mut(&mut self) -> &mut Self::Occupancy {
        &mut self.occupancy
    }
}

impl<const LOG2: ConstUVec3, T: Copy + Eq + Send + Sync + 'static + Default> Node
    for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    /// Total number of voxels contained within the leaf node.
    const SIZE: usize = size_of_grid(LOG2);
    type LeafType = Self;
    /// Extent of the leaf node in each axis.
    const EXTENT_LOG2: UVec3 = LOG2.to_glam();
    const EXTENT: UVec3 = UVec3 {
        x: 1 << LOG2.x,
        y: 1 << LOG2.y,
        z: 1 << LOG2.z,
    };
    const EXTENT_MASK: UVec3 = UVec3 {
        x: Self::EXTENT.x - 1,
        y: Self::EXTENT.y - 1,
        z: Self::EXTENT.z - 1,
    };
    const META_MASK: UVec3 = UVec3 {
        x: 1 << (LOG2.x - 1),
        y: 1 << (LOG2.y - 1),
        z: 1 << (LOG2.z - 1),
    };
    const LEVEL: usize = 0;

    fn set<'a>(
        &'a mut self,
        _pools: &'a mut [Pool],
        coords: UVec3,
        value: bool,
        _cached_path: &mut [u32],
        _touched_nodes: Option<&mut Vec<(u32, u32)>>,
    ) -> (Option<&'a mut Self::LeafType>, &'a mut Self::LeafType) {
        self.occupancy
            .set(Self::get_fully_mapped_offset(coords) as usize, value);
        (None, self)
    }
    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was owned.
    /// Implementation will write to cached_path for all levels below the current level.
    fn get<'a>(
        &'a self,
        pools: &'a [Pool],
        coords: UVec3,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType> {
        Some(self)
    }

    /// Get the value of a voxel at the specified coordinates within the node space.
    /// This is called when the node was located in a node pool.
    /// Implementation will write to cached_path for all levels including the current level.
    fn get_in_pools<'a>(
        pools: &'a [Pool],
        coords: UVec3,
        ptr: u32,
        cached_path: &mut [u32],
    ) -> Option<&'a Self::LeafType> {
        if cached_path.len() > 0 {
            cached_path[Self::LEVEL] = ptr;
        }
        Some(unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) })
    }

    #[inline]
    fn set_in_pools<'a>(
        pools: &'a mut [Pool],
        coords: UVec3,
        ptr: &mut u32,
        value: bool,
        cached_path: &mut [u32],
        touched_nodes: Option<&mut Vec<(u32, u32)>>,
    ) -> (Option<&'a mut Self>, &'a mut Self) {
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        let (old_leaf_node, old_value): (*mut _, bool) = unsafe {
            let old_leaf_node = pools[Self::LEVEL].get_item_mut::<Self>(*ptr);
            let old_value = old_leaf_node.occupancy.get(index);
            (old_leaf_node, old_value)
        };
        if cached_path.len() > 0 {
            cached_path[0] = *ptr;
        }
        if let Some(touched_nodes) = touched_nodes {
            // Copy on write
            if old_value == value {
                return (None, unsafe { &mut *old_leaf_node });
            }

            let new_node_ptr = unsafe { pools[Self::LEVEL].alloc_uninitialized() };
            if cached_path.len() > 0 {
                cached_path[0] = new_node_ptr;
            }
            touched_nodes.push((Self::LEVEL as u32, *ptr));
            *ptr = new_node_ptr;
            let new_leaf_node = unsafe { pools[Self::LEVEL].get_item_mut::<Self>(new_node_ptr) };
            unsafe { std::ptr::copy_nonoverlapping(old_leaf_node, new_leaf_node, 1) };
            new_leaf_node.occupancy.set(index, value);
            return (Some(unsafe { &mut *old_leaf_node }), new_leaf_node);
        } else {
            let old_leaf_node: &mut _ = unsafe { &mut *old_leaf_node };
            //old_leaf_node.occupancy.set(index, value); the caller should set this on their own
            return (None, old_leaf_node);
        }
    }

    type Iterator<'a> = LeafNodeIterator<'a, LOG2>;
    fn iter<'a>(&'a self, _pool: &'a [Pool], offset: UVec3) -> Self::Iterator<'a> {
        LeafNodeIterator {
            location_offset: offset,
            bits_iterator: self.occupancy.iter_set_bits(),
        }
    }
    fn iter_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::Iterator<'a> {
        let node = unsafe { pools[0].get_item::<Self>(ptr) };
        LeafNodeIterator {
            location_offset: offset,
            bits_iterator: node.occupancy.iter_set_bits(),
        }
    }

    type LeafIterator<'a> = Once<(UVec3, &'a UnsafeCell<Self>)>;

    #[inline]
    fn iter_leaf<'a>(&'a self, _pools: &'a [Pool], offset: UVec3) -> Self::LeafIterator<'a> {
        std::iter::once((offset, unsafe { std::mem::transmute(self) }))
    }

    #[inline]
    fn iter_leaf_in_pool<'a>(pools: &'a [Pool], ptr: u32, offset: UVec3) -> Self::LeafIterator<'a> {
        let node = unsafe { pools[0].get_item::<Self>(ptr) };
        std::iter::once((offset, unsafe { std::mem::transmute(node) }))
    }

    fn write_meta(metas: &mut Vec<NodeMeta<Self>>) {
        metas.push(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            extent_log2: Self::EXTENT_LOG2,
            fanout_log2: LOG2.to_glam(),
            extent_mask: Self::EXTENT_MASK,
            setter: Self::set_in_pools,
            getter: Self::get_in_pools,
        });
    }

    /*
    #[cfg(feature = "physics")]
    #[inline]
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        _solid: bool,
        initial_intersection_t: glam::Vec2,
        _pools: &[Pool],
    ) -> Option<parry3d::query::RayIntersection> {
        // Assume that the node is located at 0.0 - 4.0

        use glam::{IVec3, Vec3, Vec3A, Vec3Swizzles};
        let mut hit_distance: f32 = initial_intersection_t.x;
        let initial_intersection_point: Vec3A = (ray.origin + ray.dir * hit_distance).into();
        let mut position: IVec3 =
            Vec3::from((initial_intersection_point * Self::EXTENT.as_vec3a()).floor())
                .as_ivec3()
                .clamp(IVec3::splat(0), Self::EXTENT.as_ivec3() - IVec3::splat(1));

        let t_coef: Vec3A = 1.0 / Vec3A::from(ray.dir);
        let t_bias: Vec3A = t_coef * Vec3A::from(ray.origin);

        let step = Vec3A::from(ray.dir).signum();

        let mut t_max: Vec3A =
            ((position.as_vec3a() / Self::EXTENT.as_vec3a()) + step.max(Vec3A::ZERO)) * t_coef
                - t_bias;

        let t_delta = Vec3A::ONE * t_coef * step;
        while !{
            let mut result = false;
            self.get(&mut [], position.try_into().unwrap(), &mut [], &mut result);
            result
        } {
            let comp_result = Vec3A::select(t_max.zxy().cmplt(t_max), Vec3A::ZERO, Vec3A::ONE)
                * Vec3A::select(t_max.yzx().cmplt(t_max), Vec3A::ZERO, Vec3A::ONE);
            let position_delta = (step * comp_result).as_ivec3();
            position += position_delta;
            hit_distance = t_max.x.min(t_max.y).min(t_max.z);
            if hit_distance + 0.001 >= initial_intersection_t.y {
                return None;
            }
            t_max += t_delta * comp_result;
        }

        let index = ((position.x as usize) << (LOG2.y + LOG2.z))
            | ((position.y as usize) << LOG2.z)
            | (position.z as usize);
        Some(parry3d::query::RayIntersection {
            feature: parry3d::shape::FeatureId::Vertex(index as u32),
            time_of_impact: hit_distance,
            normal: Default::default(),
        })
    }
    */
}

impl<const LOG2: ConstUVec3, T: Send + Sync + 'static> std::fmt::Debug for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LeafNode\n")?;
        self.occupancy.fmt(f)?;
        Ok(())
    }
}

pub struct LeafNodeIterator<'a, const LOG2: ConstUVec3>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    location_offset: UVec3,
    bits_iterator: SetBitIterator<std::iter::Cloned<std::slice::Iter<'a, usize>>>,
}
impl<'a, const LOG2: ConstUVec3> Iterator for LeafNodeIterator<'a, LOG2>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.bits_iterator.next()?;

        let z = index & ((1 << LOG2.z) - 1);
        let y = (index >> LOG2.z) & ((1 << LOG2.y) - 1);
        let x = index >> (LOG2.z + LOG2.y);
        let location = UVec3::new(x as u32, y as u32, z as u32);
        Some(location + self.location_offset)
    }
}
