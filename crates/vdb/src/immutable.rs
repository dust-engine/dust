use std::{
    cell::UnsafeCell,
    ptr,
    sync::{atomic::AtomicU64, Arc, Mutex},
};

use glam::UVec3;
use parry3d::math::Vector;

use crate::{traversal::TreeTraversal, tree::TreeLike, AabbU32, MutableTree, Node, Pool};

struct ImmutableTreeSharedInfo<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pool: UnsafeCell<[Pool; ROOT::LEVEL as usize]>,
    /// Lock for structural changes of all allocators.
    pool_lock: Mutex<()>,
    latest_generation: AtomicU64,
    recycled_generation: AtomicU64,
}
unsafe impl<ROOT: Node> Send for ImmutableTreeSharedInfo<ROOT> where
    [(); ROOT::LEVEL as usize]: Sized
{
}
unsafe impl<ROOT: Node> Sync for ImmutableTreeSharedInfo<ROOT> where
    [(); ROOT::LEVEL as usize]: Sized
{
}

pub struct ImmutableTree<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    shared: Arc<ImmutableTreeSharedInfo<ROOT>>,
    root: Arc<ROOT>,
    aabb: AabbU32,
}

/// Represents a snapshot of an `ImmutableTree` at a certain point in the past.
/// Difference from `ImmutableTree` is that `ImmutableTreeSnapshot` cannot be further modified.
pub struct ImmutableTreeSnapshot<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    nodes_added: Vec<(u32, u32)>,
    nodes_removed: Vec<(u32, u32)>,
    generation: u64,
    shared: Arc<ImmutableTreeSharedInfo<ROOT>>,
    root: Arc<ROOT>,
    aabb: AabbU32,
}
impl<ROOT: Node> Drop for ImmutableTreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn drop(&mut self) {
        self.shared
            .recycled_generation
            .fetch_max(self.generation, std::sync::atomic::Ordering::Relaxed);
        let lock = self.shared.pool_lock.lock().unwrap();
        let pool = self.shared.pool.get();
        for (level, ptr) in self.nodes_removed.iter() {
            unsafe {
                (*pool)[*level as usize].free(*ptr);
            }
        }
        drop(lock);
    }
}

impl<ROOT: Node> ImmutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub fn snapshot(&self) -> ImmutableTreeSnapshot<ROOT> {
        ImmutableTreeSnapshot {
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            generation: self
                .shared
                .latest_generation
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            shared: self.shared.clone(),
            root: self.root.clone(),
            aabb: self.aabb,
        }
    }

    pub fn iter_leaf<'a>(&'a self) -> impl Iterator<Item = (UVec3, &'a <ROOT as Node>::LeafType)> {
        // No need to lock the pools here. Although the allocators are protected by a mutex,
        // trees have shared ownership to the allocated slots.
        let pools = unsafe { &*self.shared.pool.get() };
        self.root
            .iter_leaf(pools, UVec3 { x: 0, y: 0, z: 0 })
            .map(|(position, leaf)| unsafe {
                let leaf: &'a ROOT::LeafType = &*leaf.get();
                (position, leaf)
            })
    }
}

impl<ROOT: Node> MutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    pub fn freeze(self) -> ImmutableTree<ROOT> {
        let shared = Arc::new(ImmutableTreeSharedInfo {
            pool: UnsafeCell::new(self.pool),
            pool_lock: Mutex::new(()),
            latest_generation: AtomicU64::new(0),
            recycled_generation: AtomicU64::new(0),
        });
        let root = Arc::new(self.root);
        ImmutableTree {
            shared,
            root,
            aabb: self.aabb,
        }
    }
}

impl<ROOT: Node> From<MutableTree<ROOT>> for ImmutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn from(tree: MutableTree<ROOT>) -> Self {
        tree.freeze()
    }
}

impl<ROOT: Node> TreeLike for ImmutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn get_value(&self, coords: UVec3) -> bool {
        todo!()
    }

    fn aabb(&self) -> AabbU32 {
        self.aabb
    }

    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        max_time_of_impact: parry3d::math::Real,
        solid: bool,
    ) -> Option<parry3d::query::RayIntersection> {
        let mut initial_intersection = crate::intersect_aabb(
            ray.origin.into(),
            ray.dir.into(),
            self.aabb.min.as_vec3a(),
            self.aabb.max.as_vec3a(),
        );
        initial_intersection.y = initial_intersection.y.min(max_time_of_impact);
        if initial_intersection.x >= initial_intersection.y {
            // No intersection
            return None;
        }
        if initial_intersection.y <= 0.0 {
            return None;
        }
        let mut ray_prime = parry3d::query::Ray {
            dir: ray.dir.component_div(&ROOT::EXTENT.as_vec3().into()),
            origin: ray.origin,
        };
        ray_prime.origin.coords = ray_prime
            .origin
            .coords
            .component_div(&ROOT::EXTENT.as_vec3().into());

        self.root
            .cast_local_ray_and_get_normal(&ray_prime, solid, initial_intersection, unsafe {
                &*self.shared.pool.get()
            })
    }
}

impl<ROOT: Node> TreeLike for ImmutableTreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn get_value(&self, coords: UVec3) -> bool {
        todo!()
    }

    fn aabb(&self) -> AabbU32 {
        self.aabb
    }

    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        max_time_of_impact: parry3d::math::Real,
        solid: bool,
    ) -> Option<parry3d::query::RayIntersection> {
        let mut initial_intersection = crate::intersect_aabb(
            ray.origin.into(),
            ray.dir.into(),
            self.aabb.min.as_vec3a(),
            self.aabb.max.as_vec3a(),
        );
        initial_intersection.y = initial_intersection.y.min(max_time_of_impact);
        if initial_intersection.x >= initial_intersection.y {
            // No intersection
            return None;
        }
        if initial_intersection.y <= 0.0 {
            return None;
        }
        let mut ray_prime = parry3d::query::Ray {
            dir: ray.dir.component_div(&ROOT::EXTENT.as_vec3().into()),
            origin: ray.origin,
        };
        ray_prime.origin.coords = ray_prime
            .origin
            .coords
            .component_div(&ROOT::EXTENT.as_vec3().into());

        self.root
            .cast_local_ray_and_get_normal(&ray_prime, solid, initial_intersection, unsafe {
                &*self.shared.pool.get()
            })
    }
}

impl<ROOT: Node> TreeTraversal for ImmutableTree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    type ROOT = ROOT;

    fn root(&self) -> &ROOT {
        &self.root
    }
}

impl<ROOT: Node> TreeTraversal for ImmutableTreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    type ROOT = ROOT;

    fn root(&self) -> &ROOT {
        &self.root
    }
}
