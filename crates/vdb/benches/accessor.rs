//! Accessor throughput under coherent and incoherent access patterns.
//!
//! [`dust_vdb::Accessor`] exists to make spatially coherent traversal cheap: it
//! caches the path down to the last leaf it touched, so a neighbouring access
//! can skip the upper levels of the tree — or skip the descent entirely when it
//! stays inside the hot leaf. These benchmarks quantify that by running the
//! *same* set of voxels through the *same* tree in two orders:
//!
//! - `_coherent` walks the box in scan order (z fastest), the order a mesher or
//!   a grid import walks a volume. Runs of four consecutive coordinates share a
//!   leaf, and neighbouring leaves share a parent.
//! - `_random` walks a seeded shuffle of that same list. Consecutive accesses
//!   almost never share a leaf, so each one pays a full descent from the root.
//!
//! The two differ only in order, so the gap between them is what the cache is
//! worth. For writes that gap covers more than traversal: a write that leaves
//! the hot leaf also re-homes attributes — the new leaf's range is inflated to
//! one slot per voxel on entry and the old leaf's is fitted back down on exit —
//! so `set_random` pays an allocate-and-copy per voxel where `set_coherent`
//! pays one per leaf visit. That is the real cost of incoherent writes in this
//! design, not a benchmark artifact.
//!
//! Run with `cargo bench -p dust_vdb`. Results are reported per iteration, and
//! an iteration covers a whole box: divide by [`READ_VOXELS`] or
//! [`WRITE_VOXELS`] for a per-voxel figure.

#![feature(generic_const_exprs)]
#![feature(test)]

extern crate test;


use dust_vdb::{AttributeAllocator, Attributes, Tree, hierarchy};
use glam::UVec3;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use test::{Bencher, black_box};

/// The hierarchy `dust_vox` uses in production: 4³ leaves under two 8³ levels,
/// spanning a 256³ domain. The per-leaf value is the attribute pointer.
type BenchTreeRoot = hierarchy!(3, 3, 2, u32);
type BenchTree = Tree<BenchTreeRoot>;

/// One occupancy bit per voxel in a 4³ leaf.

/// The stored attribute. Must be non-zero: the default value erases the voxel.
const VALUE: u8 = 1;

/// The box the read benchmarks walk: 64³ voxels over 4096 leaves, a few hundred
/// KB of tree. Sized past L1 on purpose, so random reads pay the cache misses
/// they would pay in a real traversal rather than running entirely out of the
/// top-level cache.
const READ_ORIGIN: UVec3 = UVec3::ZERO;
const READ_EXTENT: u32 = 64;
pub const READ_VOXELS: usize = (READ_EXTENT * READ_EXTENT * READ_EXTENT) as usize;

/// Empty space for `get_coherent_miss`, clear of the read box above.
const MISS_ORIGIN: UVec3 = UVec3::splat(128);

/// The box the write benchmarks build. A voxel costs far more to write than to
/// read — every leaf transition allocates, inflates and re-fits an attribute
/// range — so the write box is smaller, purely to keep a full `cargo bench` run
/// short. Offsetting it from the origin keeps it straddling a mid-level node
/// boundary on every axis, so writes still descend through the full hierarchy
/// instead of living inside one subtree.
const WRITE_ORIGIN: UVec3 = UVec3::splat(16);
const WRITE_EXTENT: u32 = 32;
pub const WRITE_VOXELS: usize = (WRITE_EXTENT * WRITE_EXTENT * WRITE_EXTENT) as usize;

/// A flat-arena attribute store, modelled on `dust_vox`'s `VoxMaterial` with
/// the GPU buffer replaced by a `Vec<u8>`. The allocator geometry matches
/// production so the attribute traffic these benchmarks provoke is realistic.
struct BenchAttributes {
    allocator: AttributeAllocator,
    arena: Vec<u8>,
}

impl BenchAttributes {
    fn new() -> Self {
        Self {
            allocator: AttributeAllocator::new_with_capacity(16, 512),
            arena: Vec::new(),
        }
    }

    fn reserve(&mut self, size: usize) {
        if size > self.arena.len() {
            self.arena.resize(size.next_power_of_two(), 0);
        }
    }
}

impl Attributes for BenchAttributes {
    type Ptr = u32;
    type Value = u8;

    fn get_attribute(&self, _leaf: u32, ptr: &Self::Ptr, offset: u32) -> Self::Value {
        self.arena[*ptr as usize + offset as usize]
    }

    fn set_attribute(&mut self, _leaf: u32,  ptr: &Self::Ptr, offset: u32, value: Self::Value) {
        self.arena[*ptr as usize + offset as usize] = value;
    }

    fn free_attributes(&mut self, _leaf: u32, ptr: &Self::Ptr, num_attributes: u32) {
        self.allocator.free(*ptr, num_attributes);
    }

    fn copy_attribute(
        &mut self,
        _original_leaf: u32,
        _new_leaf: u32,
        ptr: &Self::Ptr,
        original_mask: &[usize],
        new_mask: &[usize],
        _coords: &UVec3,
    ) -> Self::Ptr {
        let new_len = dust_vdb::mask_count_ones(new_mask);
        let new_ptr = self.allocator.allocate(new_len);
        self.reserve(new_ptr as usize + new_len as usize);

        let arena = self.arena.as_mut_slice();
        let mut new_cur = new_ptr as usize;
        let mut old_cur = *ptr as usize;
        for (_, in_old, in_new) in dust_vdb::iter_mask_union(original_mask, new_mask) {
            if in_new && in_old {
                arena[new_cur] = arena[old_cur];
            }
            if in_new {
                new_cur += 1;
            }
            if in_old {
                old_cur += 1;
            }
        }
        new_ptr
    }
}

/// Scan order (z fastest) over the `extent`³ box rooted at `origin`.
fn scan_order(origin: UVec3, extent: u32) -> Vec<UVec3> {
    let mut coords = Vec::with_capacity((extent * extent * extent) as usize);
    for x in 0..extent {
        for y in 0..extent {
            for z in 0..extent {
                coords.push(origin + UVec3::new(x, y, z));
            }
        }
    }
    coords
}

/// The same voxels as [`scan_order`], in a seeded random permutation.
fn random_order(origin: UVec3, extent: u32) -> Vec<UVec3> {
    let mut coords = scan_order(origin, extent);
    coords.shuffle(&mut StdRng::seed_from_u64(0x5EED_C0DE));
    coords
}

/// Build a tree holding every voxel in `coords`. The read benchmarks always
/// feed it scan order, so they share one identically laid out tree and differ
/// only in the order they visit it.
fn populate(attributes: &mut BenchAttributes, coords: &[UVec3]) -> BenchTree {
    let mut tree = BenchTree::new();
    let mut accessor = tree.accessor_mut(attributes);
    for &coords in coords {
        accessor.set(coords, VALUE);
    }
    drop(accessor);
    tree
}

/// Writes in scan order. A 4³ leaf holds four consecutive z, so three in four
/// writes take the hot-leaf fast path and the fourth transitions to a
/// neighbouring leaf under the same parent.
///
/// Tree and attribute-store construction and teardown sit inside the timed
/// closure: a build benchmark that reused one tree would measure overwrites
/// instead of inserts. `set_random` pays exactly the same setup, so the
/// comparison between the two stays honest.
#[bench]
fn set_coherent(b: &mut Bencher) {
    let coords = scan_order(WRITE_ORIGIN, WRITE_EXTENT);
    b.iter(|| {
        let mut attributes = BenchAttributes::new();
        let tree = populate(&mut attributes, &coords);
        black_box(&tree);
        black_box(&attributes);
    });
}

/// The same writes in random order: a full descent plus an attribute
/// allocate-and-copy for every single voxel.
#[bench]
fn set_random(b: &mut Bencher) {
    let coords = random_order(WRITE_ORIGIN, WRITE_EXTENT);
    b.iter(|| {
        let mut attributes = BenchAttributes::new();
        let tree = populate(&mut attributes, &coords);
        black_box(&tree);
        black_box(&attributes);
    });
}

/// Reads in scan order over a fully occupied box: the cached path stays valid
/// for nearly every access.
#[bench]
fn get_coherent(b: &mut Bencher) {
    let coords = scan_order(READ_ORIGIN, READ_EXTENT);
    let mut attributes = BenchAttributes::new();
    let mut tree = populate(&mut attributes, &coords);
    let mut accessor = tree.accessor_mut(&mut attributes);
    b.iter(|| {
        for &coords in &coords {
            black_box(accessor.get(coords));
        }
    });
}

/// The same reads in random order: each one invalidates the cached path and
/// descends from the root.
#[bench]
fn get_random(b: &mut Bencher) {
    let scan = scan_order(READ_ORIGIN, READ_EXTENT);
    let shuffled = random_order(READ_ORIGIN, READ_EXTENT);
    let mut attributes = BenchAttributes::new();
    let mut tree = populate(&mut attributes, &scan);
    let mut accessor = tree.accessor_mut(&mut attributes);
    b.iter(|| {
        for &coords in &shuffled {
            black_box(accessor.get(coords));
        }
    });
}

/// Reads in scan order over empty space. A missed descent poisons the cached
/// path with `u32::MAX`, so later reads under the same absent ancestor return
/// `None` without touching the tree at all. This isolates that early out, which
/// the occupied read benchmarks never reach.
#[bench]
fn get_coherent_miss(b: &mut Bencher) {
    let occupied = scan_order(READ_ORIGIN, READ_EXTENT);
    let empty = scan_order(MISS_ORIGIN, READ_EXTENT);
    let mut attributes = BenchAttributes::new();
    let mut tree = populate(&mut attributes, &occupied);
    let mut accessor = tree.accessor_mut(&mut attributes);
    b.iter(|| {
        for &coords in &empty {
            black_box(accessor.get(coords));
        }
    });
}
