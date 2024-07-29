use super::{size_of_grid, NodeMeta};
use crate::{bitmask::SetBitIterator, BitMask, ConstUVec3, Node, Pool};
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
    fn get_occupancy(&self, data: &mut [u64]);
}

impl<const LOG2: ConstUVec3, T: Send + Sync + 'static + Default> IsLeaf for LeafNode<LOG2, T>
where
    [(); size_of_grid(LOG2) / size_of::<usize>() / 8]: Sized,
{
    fn get_occupancy(&self, data: &mut [u64]) {
        debug_assert_eq!(std::mem::size_of::<u64>(), std::mem::size_of::<usize>());
        let len = self.occupancy.data.len();
        debug_assert!(data.len() >= len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.occupancy.data.as_ptr() as *mut u64,
                data.as_mut_ptr(),
                len,
            );
        }
    }
}

impl<const LOG2: ConstUVec3, T: Send + Sync + 'static + Default> Node for LeafNode<LOG2, T>
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

    #[inline]
    fn get(&self, _: &[Pool], coords: UVec3, _cached_path: &mut [u32]) -> bool {
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        self.occupancy.get(index)
    }
    #[inline]
    fn set(
        &mut self,
        _: &mut [Pool],
        coords: UVec3,
        value: bool,
        _cached_path: &mut [u32],
    ) -> bool {
        let index = ((coords.x as usize) << (LOG2.y + LOG2.z))
            | ((coords.y as usize) << LOG2.z)
            | (coords.z as usize);
        let prev_value = self.occupancy.get(index);
        self.occupancy.set(index, value);
        prev_value
    }
    #[inline]
    fn get_in_pools(pools: &[Pool], coords: UVec3, ptr: u32, cached_path: &mut [u32]) -> bool {
        if cached_path.len() > 0 {
            cached_path[0] = ptr;
        }
        let leaf_node = unsafe { pools[Self::LEVEL].get_item::<Self>(ptr) };
        leaf_node.get(&[], coords, cached_path)
    }

    #[inline]
    fn set_in_pools(
        pools: &mut [Pool],
        coords: UVec3,
        ptr: u32,
        value: bool,
        cached_path: &mut [u32],
    ) -> bool {
        if cached_path.len() > 0 {
            cached_path[0] = ptr;
        }
        let leaf_node = unsafe { pools[Self::LEVEL].get_item_mut::<Self>(ptr) };
        leaf_node.set(&mut [], coords, value, cached_path)
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

    fn write_meta(metas: &mut Vec<NodeMeta>) {
        metas.push(NodeMeta {
            layout: std::alloc::Layout::new::<Self>(),
            getter: Self::get_in_pools,
            extent_log2: Self::EXTENT_LOG2,
            fanout_log2: LOG2.to_glam(),
            extent_mask: Self::EXTENT_MASK,
            setter: Self::set_in_pools,
        });
    }

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
        while !self.get(&mut [], position.try_into().unwrap(), &mut []) {
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
    bits_iterator: SetBitIterator<'a, { size_of_grid(LOG2) }>,
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
