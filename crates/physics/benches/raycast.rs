//! Ray casting: the tree-accelerated caster against parry's generic one.
//!
//! Three implementations cast the same rays at the same content:
//!
//! - `_tree`: [`VdbShape`]'s own [`RayCast`] — the tree walk
//!   (`TreeErased::iter_leaf_views_along_ray`) yields only the leaves the ray
//!   pierces, and a voxel walk runs inside each.
//! - `_generic`: parry's `cast_local_ray_on_voxels`, which steps voxel by
//!   voxel through the whole domain and asks the storage about every voxel on
//!   the way. Asking the hierarchy-erased tree about one voxel is a one-voxel
//!   range walk — a descent from the root — which is exactly what this caster
//!   would cost on `VdbShape` (see [`GenericCast`]).
//! - `_parry_voxels`: parry's own [`Voxels`] shape built from the same voxel
//!   list, casting through its chunk BVH — an external reference point for
//!   what parry considers an accelerated voxel caster.
//!
//! The scene is a 256³ domain holding a 4-deep floor slab across the whole
//! footprint plus a 32×32×32 tower in the middle (~295k voxels). Four ray
//! batches of [`RAYS_PER_BATCH`] each:
//!
//! - `hit_near`: straight down from 4 voxels above the surface — almost no
//!   traversal, so this measures fixed per-cast overhead.
//! - `hit_far`: straight down from the top of the domain — a long fall
//!   through empty space before the impact.
//! - `miss`: horizontal rays threading between the floor and the tower top,
//!   crossing all 256 voxels of the domain without touching anything.
//! - `diagonal`: from one corner, descending diagonally into the tower.
//!
//! Results are per iteration; one iteration casts a whole batch, so divide by
//! [`RAYS_PER_BATCH`] for a per-ray figure. Scene construction cross-checks
//! every ray across the three implementations before any timing.
//!
//! Run with `cargo bench -p dust_physics`.

#![feature(generic_const_exprs)]
#![feature(test)]

extern crate test;

use std::sync::{Arc, OnceLock};

use dust_physics::{VdbShape, VdbVoxel, VdbVoxelTypeAttributes};
use dust_vdb::{AttributePtr, Tree, hierarchy};
use glam::UVec3;
use parry3d::math::{IVector, Real, Vector};
use parry3d::query::details::cast_local_ray_on_voxels;
use parry3d::query::{Ray, RayCast};
use parry3d::shape::{QueriedVoxel, VoxelQuery, VoxelType, Voxels};
use test::{Bencher, black_box};

pub const RAYS_PER_BATCH: usize = 256;

/// Longer than any path through the 256³ domain.
const MAX_TOI: Real = 1.0e4;

/// The leaf inline value of the bench hierarchy. Ray casting reads no
/// attributes, so it stores nothing: the `u32` projection is inert.
#[derive(Clone, Debug, Default)]
struct BenchLeaf;

/// For [`dust_vdb::TreeWithValues<u32>`], which [`VdbShape`] requires; the
/// pointer is only read by the mask-store path, which this bench leaves out.
impl AttributePtr<u32> for BenchLeaf {
    fn attribute_ptr(&self) -> u32 {
        0
    }

    fn set_attribute_ptr(&mut self, _ptr: u32) {}
}

/// The production hierarchy's shape (4³ leaves under two 8³ levels, 256³
/// domain), with the bench's leaf value.
type BenchTree = Tree<hierarchy!(3, 3, 2, BenchLeaf)>;

/// [`VdbShape`] behind parry's generic un-accelerated caster.
///
/// `cast_local_ray_on_voxels` drives the [`VoxelQuery`] point lookup
/// [`VoxelQuery::voxel`] once per voxel stepped. `VdbShape` has no point
/// lookup, so this wrapper supplies the one the hierarchy-erased interface
/// affords: a range walk covering exactly one voxel, i.e. one descent from
/// the root per step.
struct GenericCast<'a>(&'a VdbShape);

impl VoxelQuery for GenericCast<'_> {
    type Voxel<'b>
        = VdbVoxel<'b>
    where
        Self: 'b;

    fn voxel_size(&self) -> Vector {
        self.0.voxel_size()
    }

    fn domain(&self) -> [IVector; 2] {
        self.0.domain()
    }

    fn voxel(&self, key: IVector) -> Option<VdbVoxel<'_>> {
        self.0.voxels_in_range(key, key + IVector::splat(1)).next()
    }

    fn linear_id(&self, key: IVector) -> Option<u32> {
        self.voxel(key).map(|voxel| voxel.linear_id())
    }

    fn voxels_in_range(&self, mins: IVector, maxs: IVector) -> impl Iterator<Item = VdbVoxel<'_>> {
        self.0.voxels_in_range(mins, maxs).map(|voxel| voxel)
    }
}

struct Scene {
    shape: VdbShape,
    parry_voxels: Voxels,
    hit_near: Vec<Ray>,
    hit_far: Vec<Ray>,
    miss: Vec<Ray>,
    diagonal: Vec<Ray>,
}

/// Whether the column at `(x, z)` is inside the tower's footprint.
fn in_tower(x: u32, z: u32) -> bool {
    (96..128).contains(&x) && (96..128).contains(&z)
}

fn build_scene() -> Scene {
    // Content: floor slab y in 0..4 over the whole footprint, tower
    // x, z in 96..128, y in 4..36.
    let mut tree = BenchTree::new();
    let mut attributes = VdbVoxelTypeAttributes::new(64);
    let mut keys = Vec::new();
    let mut accessor = tree.accessor_mut(&mut attributes);
    for x in 0..256 {
        for z in 0..256 {
            let height = if in_tower(x, z) { 36 } else { 4 };
            for y in 0..height {
                accessor.set(UVec3::new(x, y, z), VoxelType::Vertex);
                keys.push(IVector::new(x as i32, y as i32, z as i32));
            }
        }
    }
    drop(accessor);
    let shape = VdbShape::new(
        Arc::new(tree.snapshot()),
        Arc::new(attributes),
        None,
        Vector::splat(1.0),
    );
    let parry_voxels = Voxels::new(Vector::splat(1.0), &keys);

    // Ray batches, jittered over the footprint by a small deterministic
    // generator.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let down = Vector::new(0.0, -1.0, 0.0);
    let mut hit_near = Vec::new();
    let mut hit_far = Vec::new();
    let mut miss = Vec::new();
    let mut diagonal = Vec::new();
    for _ in 0..RAYS_PER_BATCH {
        let (x, z) = (next() % 256, next() % 256);
        let surface = if in_tower(x, z) { 36.0 } else { 4.0 };
        let (x, z) = (x as Real + 0.5, z as Real + 0.5);
        hit_near.push(Ray::new(Vector::new(x, surface + 4.0, z), down));
        hit_far.push(Ray::new(Vector::new(x, 255.5, z), down));
        // Between the floor top (4) and the tower top (36), clear of the
        // tower's z range: crosses the whole domain through empty space.
        miss.push(Ray::new(
            Vector::new(
                0.5,
                6.0 + (next() % 28) as Real,
                (next() % 90) as Real + 0.5,
            ),
            Vector::new(1.0, 0.0, 0.0),
        ));
        // From the domain's top corner, sinking toward the tower's side.
        diagonal.push(Ray::new(
            Vector::new(0.5, 35.5, (next() % 32) as Real + 80.5),
            Vector::new(1.0, -0.05, 0.3).normalize(),
        ));
    }

    // Every implementation must agree on every ray before anything is timed.
    // The tree and generic casters share the exact per-voxel impact
    // arithmetic, so their agreement is bitwise; parry's `Voxels` computes
    // the same boxes through its own code, so its impact gets a tolerance.
    for ray in hit_near
        .iter()
        .chain(&hit_far)
        .chain(&miss)
        .chain(&diagonal)
    {
        let tree = shape.cast_local_ray_and_get_normal(ray, MAX_TOI, true);
        let generic = cast_local_ray_on_voxels(&GenericCast(&shape), ray, MAX_TOI, true);
        let parry = parry_voxels.cast_local_ray_and_get_normal(ray, MAX_TOI, true);
        assert_eq!(
            tree.map(|hit| hit.time_of_impact),
            generic.map(|hit| hit.time_of_impact),
            "tree and generic casters disagree on {ray:?}"
        );
        match (tree, parry) {
            (None, None) => {}
            (Some(tree), Some(parry)) => {
                assert!(
                    (tree.time_of_impact - parry.time_of_impact).abs() < 1.0e-4,
                    "tree and parry Voxels disagree on {ray:?}"
                );
            }
            _ => panic!("tree and parry Voxels disagree on {ray:?}"),
        }
    }

    Scene {
        shape,
        parry_voxels,
        hit_near,
        hit_far,
        miss,
        diagonal,
    }
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(build_scene)
}

fn bench_tree(b: &mut Bencher, rays: fn(&Scene) -> &Vec<Ray>) {
    let scene = scene();
    b.iter(|| {
        for ray in rays(scene) {
            black_box(
                scene
                    .shape
                    .cast_local_ray_and_get_normal(ray, MAX_TOI, true),
            );
        }
    });
}

fn bench_generic(b: &mut Bencher, rays: fn(&Scene) -> &Vec<Ray>) {
    let scene = scene();
    let generic = GenericCast(&scene.shape);
    b.iter(|| {
        for ray in rays(scene) {
            black_box(cast_local_ray_on_voxels(&generic, ray, MAX_TOI, true));
        }
    });
}

fn bench_parry_voxels(b: &mut Bencher, rays: fn(&Scene) -> &Vec<Ray>) {
    let scene = scene();
    b.iter(|| {
        for ray in rays(scene) {
            black_box(
                scene
                    .parry_voxels
                    .cast_local_ray_and_get_normal(ray, MAX_TOI, true),
            );
        }
    });
}

#[bench]
fn hit_near_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.hit_near);
}

#[bench]
fn hit_near_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.hit_near);
}

#[bench]
fn hit_near_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.hit_near);
}

#[bench]
fn hit_far_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.hit_far);
}

#[bench]
fn hit_far_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.hit_far);
}

#[bench]
fn hit_far_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.hit_far);
}

#[bench]
fn miss_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.miss);
}

#[bench]
fn miss_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.miss);
}

#[bench]
fn miss_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.miss);
}

#[bench]
fn diagonal_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.diagonal);
}

#[bench]
fn diagonal_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.diagonal);
}

#[bench]
fn diagonal_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.diagonal);
}
