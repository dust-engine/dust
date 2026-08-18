//! Leaf and voxel iteration throughput: the hierarchy's own monomorphized
//! iterator ([`dust_vdb::TreeLike::iter_leaf`], full type information, nested
//! per-level iterators) against the hierarchy-erased walker
//! ([`dust_vdb::TreeErasedLeaf::iter_leaf_erased`], runtime tree geometry,
//! inline frame stack).
//!
//! Both variants of a pair run over the *same* tree and use the *same* sink,
//! so any gap between them is the cost (or benefit) of erasure alone.
//!
//! - `_dense` iterates a fully occupied 64³ box: 4096 leaves, every mask bit
//!   set. This is pure yield throughput — no empty words to skip.
//! - `_sparse` iterates 4096 seeded random voxels scattered over the full
//!   256³ domain: leaves hold a single voxel and child masks are nearly
//!   empty, so the walk is dominated by scanning for set bits.
//!
//! Run with `cargo bench -p dust_vdb`. An iteration covers the whole tree;
//! divide by the leaf/voxel counts reported in the source for per-item cost.
//!
//! At these per-item costs the deciding factor is whether `next()` inlines
//! into the consuming loop: without `#[inline(always)]`, LLVM left some
//! instantiations of the template's fused chain out-of-line — a call per
//! leaf, iterator state round-tripping through the stack, and heavy
//! run-to-run variance — which made the erased walker look faster. Forced
//! inline, the monomorphized template beats the walker on every benchmark
//! here; the walker's flat, runtime-geometry loop is the price of erasure
//! and is insensitive to inlining decisions.

#![feature(generic_const_exprs)]
#![feature(test)]

extern crate test;

use std::marker::PhantomData;

use bitvec::array::BitArray;
use dust_vdb::{AttributeAllocator, Attributes, Tree, TreeErased, TreeErasedLeaf, TreeLike, hierarchy};
use glam::UVec3;
use rand::{Rng, SeedableRng, rngs::StdRng};
use test::{Bencher, black_box};

/// The hierarchy `dust_vox` uses in production: 4³ leaves under two 8³ levels,
/// spanning a 256³ domain. The per-leaf value is the attribute pointer.
type BenchTreeRoot = hierarchy!(3, 3, 2, u32);
type BenchTree = Tree<BenchTreeRoot>;

/// One occupancy bit per voxel in a 4³ leaf.
const OCCUPANCY_WORDS: usize = 64 / size_of::<usize>() / 8;

/// The stored attribute. Must be non-zero: the default value erases the voxel.
const VALUE: u8 = 1;

/// The fully occupied box `_dense` iterates: 64³ voxels over 4096 leaves.
const DENSE_EXTENT: u32 = 64;

/// Number of seeded random voxels `_sparse` scatters over the 256³ domain.
const SPARSE_VOXELS: usize = 4096;

/// A minimal attribute store so the accessor can build the benchmark trees;
/// mirrors the one in `benches/accessor.rs`. Iteration itself never touches
/// attributes, so this only matters during setup.
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
    type Occupancy<'a> = &'a BitArray<[usize; OCCUPANCY_WORDS]>;
    const MAX_OCCUPANCY: Self::Occupancy<'static> = &BitArray {
        _ord: PhantomData,
        data: [usize::MAX; OCCUPANCY_WORDS],
    };
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
        ptr: &Self::Ptr,
        original_mask: Self::Occupancy<'_>,
        new_mask: Self::Occupancy<'_>,
        _coords: &UVec3,
    ) -> Self::Ptr {
        let new_len = new_mask.count_ones() as u32;
        let new_ptr = self.allocator.allocate(new_len);
        self.reserve(new_ptr as usize + new_len as usize);

        let arena = self.arena.as_mut_slice();
        let mut new_cur = new_ptr as usize;
        let mut old_cur = *ptr as usize;
        for bit in (*original_mask | new_mask).iter_ones() {
            let in_new = new_mask[bit];
            let in_old = original_mask[bit];
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

fn populate(coords: impl IntoIterator<Item = UVec3>) -> BenchTree {
    let mut attributes = BenchAttributes::new();
    let mut tree = BenchTree::new();
    let mut accessor = tree.accessor_mut(&mut attributes);
    for coords in coords {
        accessor.set(coords, VALUE);
    }
    drop(accessor);
    tree
}

/// Every voxel of the `DENSE_EXTENT`³ box, in scan order.
fn dense_tree() -> BenchTree {
    populate((0..DENSE_EXTENT).flat_map(|x| {
        (0..DENSE_EXTENT)
            .flat_map(move |y| (0..DENSE_EXTENT).map(move |z| UVec3::new(x, y, z)))
    }))
}

/// `SPARSE_VOXELS` seeded random voxels over the full 256³ domain.
fn sparse_tree() -> BenchTree {
    let mut rng = StdRng::seed_from_u64(0x5EED_C0DE);
    populate((0..SPARSE_VOXELS).map(|_| {
        UVec3::new(
            rng.gen_range(0..256),
            rng.gen_range(0..256),
            rng.gen_range(0..256),
        )
    }))
}

/// Consume a leaf iterator with a data dependency on every yielded origin and
/// leaf reference, without loading through the reference — the sink is
/// identical for both variants, so it cancels out of the comparison.
#[inline]
fn drain_leaves<'a, L: 'a>(iter: impl Iterator<Item = (UVec3, &'a L)>) -> u64 {
    let mut acc = 0u64;
    for (origin, leaf) in iter {
        acc = acc
            .wrapping_add((origin.x + origin.y + origin.z) as u64)
            .wrapping_add(leaf as *const L as u64);
    }
    acc
}

/// Consume a voxel iterator with a data dependency on every coordinate.
#[inline]
fn drain_voxels(iter: impl Iterator<Item = UVec3>) -> u64 {
    let mut acc = 0u64;
    for voxel in iter {
        acc = acc.wrapping_add((voxel.x + voxel.y + voxel.z) as u64);
    }
    acc
}

#[bench]
fn leaf_dense_template(b: &mut Bencher) {
    let tree = dense_tree();
    b.iter(|| black_box(drain_leaves(tree.iter_leaf())));
}

#[bench]
fn leaf_dense_erased(b: &mut Bencher) {
    let tree = dense_tree();
    b.iter(|| black_box(drain_leaves(tree.iter_leaf_erased())));
}

#[bench]
fn leaf_sparse_template(b: &mut Bencher) {
    let tree = sparse_tree();
    b.iter(|| black_box(drain_leaves(tree.iter_leaf())));
}

#[bench]
fn leaf_sparse_erased(b: &mut Bencher) {
    let tree = sparse_tree();
    b.iter(|| black_box(drain_leaves(tree.iter_leaf_erased())));
}

#[bench]
fn voxels_dense_template(b: &mut Bencher) {
    let tree = dense_tree();
    b.iter(|| black_box(drain_voxels(tree.iter())));
}

#[bench]
fn voxels_dense_erased(b: &mut Bencher) {
    let tree = dense_tree();
    b.iter(|| black_box(drain_voxels(tree.iter_erased())));
}

#[bench]
fn voxels_sparse_template(b: &mut Bencher) {
    let tree = sparse_tree();
    b.iter(|| black_box(drain_voxels(tree.iter())));
}

#[bench]
fn voxels_sparse_erased(b: &mut Bencher) {
    let tree = sparse_tree();
    b.iter(|| black_box(drain_voxels(tree.iter_erased())));
}

// --- depth scaling ---
//
// The same 4096 fully occupied leaves (a dense 64³ box) under one, two, and
// three internal levels; `leaf_dense_*` above is the two-level point. Each
// `next()` of the template re-enters one nested iterator per level between the
// root and the leaves, while the walker reads only the top of its frame stack.
// If the template's per-leaf cost climbs with depth and the walker's stays
// flat, the nesting protocol is the cost; whatever gap remains at depth one —
// where no nesting exists — is the per-leaf `Once` shuttle through
// `child_iterator`.

/// 16³ leaves directly under the root: one internal level, 64³ domain.
type DepthOneRoot = hierarchy!(4, 2, u32);
/// 4³ fanout at every level: three internal levels, 256³ domain.
type DepthThreeRoot = hierarchy!(2, 2, 2, 2, u32);

macro_rules! depth_bench {
    ($template:ident, $erased:ident, $root:ty) => {
        #[bench]
        fn $template(b: &mut Bencher) {
            let tree = {
                let mut attributes = BenchAttributes::new();
                let mut tree = Tree::<$root>::new();
                let mut accessor = tree.accessor_mut(&mut attributes);
                for x in 0..DENSE_EXTENT {
                    for y in 0..DENSE_EXTENT {
                        for z in 0..DENSE_EXTENT {
                            accessor.set(UVec3::new(x, y, z), VALUE);
                        }
                    }
                }
                drop(accessor);
                tree
            };
            b.iter(|| black_box(drain_leaves(tree.iter_leaf())));
        }
        #[bench]
        fn $erased(b: &mut Bencher) {
            let tree = {
                let mut attributes = BenchAttributes::new();
                let mut tree = Tree::<$root>::new();
                let mut accessor = tree.accessor_mut(&mut attributes);
                for x in 0..DENSE_EXTENT {
                    for y in 0..DENSE_EXTENT {
                        for z in 0..DENSE_EXTENT {
                            accessor.set(UVec3::new(x, y, z), VALUE);
                        }
                    }
                }
                drop(accessor);
                tree
            };
            b.iter(|| black_box(drain_leaves(tree.iter_leaf_erased())));
        }
    };
}

depth_bench!(leaf_depth1_template, leaf_depth1_erased, DepthOneRoot);
depth_bench!(leaf_depth3_template, leaf_depth3_erased, DepthThreeRoot);
