//! Point projection: the tree-accelerated search against parry's generic one.
//!
//! Three implementations project the same points onto the same content:
//!
//! - `_tree`: [`VdbShape`]'s own [`PointQuery`] — the tree walk
//!   (`TreeErased::iter_leaf_views_near_point`) hands out leaf blocks nearest
//!   first, and the search stops at the first block too far to beat the best
//!   candidate.
//! - `_generic`: parry's `project_local_point_on_voxels`, which scans a
//!   growing box around the point — re-scanned from scratch each round —
//!   until it can prove the best candidate optimal. Its box scans run on
//!   [`VdbShape`]'s range walk (`VoxelQuery::voxels_in_range`), which already
//!   skips empty space, so the difference measured here is purely the search
//!   strategy.
//! - `_parry_voxels`: parry's own [`Voxels`] shape built from the same voxel
//!   list, searching through its chunk BVH — an external reference point for
//!   what parry considers an accelerated projection.
//!
//! The scene is the ray benches' 256³ domain: a 4-deep floor slab across the
//! whole footprint plus a 32×32×32 tower in the middle (~295k voxels). Four
//! point batches:
//!
//! - `surface_near`: within one voxel above the surface — the projection is
//!   right there, so this measures fixed per-query overhead.
//! - `beside_tower`: mid-air next to the tower's wall, a few voxels out.
//! - `inside_tower`: buried in the tower's solid interior (the non-solid
//!   projection every search approximates by the nearest voxel's boundary).
//! - `far_above`: high above the content, ~165 voxels from anything. The
//!   generic search's proof box must reach the floor, so it projects onto
//!   every voxel of the scene; the batch holds [`FAR_POINTS`] points instead
//!   of [`POINTS_PER_BATCH`] to keep a `cargo bench` run short.
//!
//! Every query is non-solid: the generic search's `solid` fast path needs
//! the [`VoxelQuery::voxel`] point lookup, which [`VdbShape`] doesn't have
//! yet, and off that fast path `solid` changes nothing about the search.
//!
//! Results are per iteration; one iteration projects a whole batch, so
//! divide by the batch's point count for a per-query figure. Scene
//! construction cross-checks every point across the three implementations
//! before any timing.
//!
//! Run with `cargo bench -p dust_physics`.

#![feature(generic_const_exprs)]
#![feature(test)]

extern crate test;

use std::sync::{Arc, OnceLock};

use dust_physics::{VdbShape, VdbVoxelTypeAttributes};
use dust_vdb::{AttributePtr, Tree, hierarchy};
use glam::UVec3;
use parry3d::math::{IVector, Real, Vector};
use parry3d::query::PointQuery;
use parry3d::query::details::project_local_point_on_voxels;
use parry3d::shape::{VoxelType, Voxels};
use test::{Bencher, black_box};

pub const POINTS_PER_BATCH: usize = 256;

/// The `far_above` batch's size: each generic query there projects onto every
/// voxel of the scene (~295k), so the batch stays small.
pub const FAR_POINTS: usize = 16;

/// The leaf inline value of the bench hierarchy. Point projection reads no
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

struct Scene {
    shape: VdbShape,
    parry_voxels: Voxels,
    surface_near: Vec<Vector>,
    beside_tower: Vec<Vector>,
    inside_tower: Vec<Vector>,
    far_above: Vec<Vector>,
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

    // Point batches, jittered by a small deterministic generator. The
    // fractional offsets keep every point off the integer grid, so none sits
    // exactly on a voxel face.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut surface_near = Vec::new();
    let mut beside_tower = Vec::new();
    let mut inside_tower = Vec::new();
    let mut far_above = Vec::new();
    for i in 0..POINTS_PER_BATCH {
        let (x, z) = (next() % 256, next() % 256);
        let surface = if in_tower(x, z) { 36.0 } else { 4.0 };
        surface_near.push(Vector::new(
            x as Real + 0.31,
            surface + 0.13 + (next() % 100) as Real / 100.0,
            z as Real + 0.73,
        ));
        // Next to the tower's low-z wall (z = 96), one to eight voxels out,
        // high enough that the wall is closer than the floor below.
        beside_tower.push(Vector::new(
            (96 + next() % 32) as Real + 0.31,
            (14 + next() % 16) as Real + 0.57,
            (96 - 1 - next() % 8) as Real + 0.73,
        ));
        // Strictly interior to the tower.
        inside_tower.push(Vector::new(
            (100 + next() % 24) as Real + 0.31,
            (8 + next() % 24) as Real + 0.57,
            (100 + next() % 24) as Real + 0.73,
        ));
        if i < FAR_POINTS {
            // High above everything, clear of the tower's footprint.
            far_above.push(Vector::new(
                (next() % 90) as Real + 0.31,
                (200 + next() % 40) as Real + 0.57,
                (next() % 90) as Real + 0.73,
            ));
        }
    }

    // Every implementation must agree on every point before anything is
    // timed. The tree and generic searches weigh the identical candidate
    // set with the same arithmetic, so their distances agree bitwise (only
    // tie resolution may pick a different voxel); parry's `Voxels` computes
    // its own way, so its distance gets a tolerance.
    for pt in surface_near
        .iter()
        .chain(&beside_tower)
        .chain(&inside_tower)
        .chain(&far_above)
    {
        let tree = shape.project_local_point(*pt, false);
        let generic = project_local_point_on_voxels(&shape, *pt, false)
            .expect("the scene is not empty")
            .0;
        let parry = parry_voxels.project_local_point(*pt, false);
        assert_eq!(
            (tree.point - pt).length(),
            (generic.point - pt).length(),
            "tree and generic searches disagree on {pt:?}"
        );
        assert_eq!(
            tree.is_inside, generic.is_inside,
            "tree and generic searches disagree on {pt:?}"
        );
        assert!(
            ((tree.point - pt).length() - (parry.point - pt).length()).abs() < 1.0e-4,
            "tree and parry Voxels disagree on {pt:?}"
        );
    }

    Scene {
        shape,
        parry_voxels,
        surface_near,
        beside_tower,
        inside_tower,
        far_above,
    }
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(build_scene)
}

fn bench_tree(b: &mut Bencher, points: fn(&Scene) -> &Vec<Vector>) {
    let scene = scene();
    b.iter(|| {
        for pt in points(scene) {
            black_box(scene.shape.project_local_point(*pt, false));
        }
    });
}

fn bench_generic(b: &mut Bencher, points: fn(&Scene) -> &Vec<Vector>) {
    let scene = scene();
    b.iter(|| {
        for pt in points(scene) {
            black_box(project_local_point_on_voxels(&scene.shape, *pt, false));
        }
    });
}

fn bench_parry_voxels(b: &mut Bencher, points: fn(&Scene) -> &Vec<Vector>) {
    let scene = scene();
    b.iter(|| {
        for pt in points(scene) {
            black_box(scene.parry_voxels.project_local_point(*pt, false));
        }
    });
}

#[bench]
fn surface_near_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.surface_near);
}

#[bench]
fn surface_near_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.surface_near);
}

#[bench]
fn surface_near_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.surface_near);
}

#[bench]
fn beside_tower_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.beside_tower);
}

#[bench]
fn beside_tower_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.beside_tower);
}

#[bench]
fn beside_tower_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.beside_tower);
}

#[bench]
fn inside_tower_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.inside_tower);
}

#[bench]
fn inside_tower_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.inside_tower);
}

#[bench]
fn inside_tower_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.inside_tower);
}

#[bench]
fn far_above_tree(b: &mut Bencher) {
    bench_tree(b, |scene| &scene.far_above);
}

#[bench]
fn far_above_generic(b: &mut Bencher) {
    bench_generic(b, |scene| &scene.far_above);
}

#[bench]
fn far_above_parry_voxels(b: &mut Bencher) {
    bench_parry_voxels(b, |scene| &scene.far_above);
}
