use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{
    AabbU32, InternalNodeEntry, IsLeaf, Node, NodeMeta, pool::{Pool, PoolStorage},
};

pub struct Tree<ROOT: Node>
where
    [(); ROOT::LEVEL]: Sized,
{
    pub(crate) root: ROOT,
    pub(crate) pool: [Pool; ROOT::LEVEL],
    pub(crate) aabb: AabbU32,
    /// Number of snapshots created by [`Tree::snapshot`] and not yet returned
    /// to [`Tree::release_snapshot`].
    pub(crate) snapshot_count: u32,
}

/// A self-contained, read-only version of a [`Tree`], captured by
/// [`Tree::snapshot`].
///
/// The snapshot owns a copy of the tree's root node plus a captured, read-only
/// view of each node pool ([`Pool::capture`]); everything below the root is
/// shared with the live tree through per-slot reference counts
/// ([`Pool::retain`]). Mutating the tree afterwards copies shared nodes on
/// write, so the snapshot keeps observing the exact state it captured, at a
/// cost proportional to the number of nodes actually touched.
///
/// Reads need no access to the originating tree and may run on another thread
/// while the tree is being mutated. This is sound because
/// - every slot reachable from a snapshot is frozen by copy-on-write: the
///   writer redirects edges to private copies instead of mutating shared
///   nodes, and a slot is only freed (and its bytes scribbled) once no version
///   references it;
/// - the captured pools pin their backing allocations, so a live pool growing
///   into a new allocation cannot invalidate a snapshot mid-read.
/// The writer therefore only ever writes bytes a snapshot reader never
/// dereferences.
///
/// A snapshot pins every node reachable from it: it must eventually be given
/// back to [`Tree::release_snapshot`] (or made the current state again via
/// [`Tree::restore`], and then released), otherwise those pool slots are never
/// reclaimed. Reference-count bookkeeping lives with the tree, which is why
/// releasing takes `&mut Tree` while reading does not.
#[must_use = "a snapshot pins pool slots until it is returned to Tree::release_snapshot"]
pub struct TreeSnapshot<ROOT: Node>
where
    [(); ROOT::LEVEL]: Sized,
{
    pub(crate) root: ROOT,
    pub(crate) pool: [Pool; ROOT::LEVEL],
    aabb: AabbU32,
    drop_guard: TreeSnapshotDropGuard,
}
impl<ROOT: Node> TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL]: Sized,
{
    /// Get the leaf node containing `coords` as of the snapshot, if any.
    pub fn get(&self, coords: UVec3) -> Option<&ROOT::LeafType> {
        self.root.get(&self.pool, coords, &mut [])
    }
}

struct TreeSnapshotDropGuard;
impl Drop for TreeSnapshotDropGuard {
    fn drop(&mut self) {
        panic!("TreeSnapshot must be released via Tree::release_snapshot");
    }
}
impl TreeSnapshotDropGuard {
    fn disarm(self) {
        std::mem::forget(self);
    }
}

impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    pub fn new() -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL];
        for (i, meta) in ROOT::META.iter().take(ROOT::LEVEL).enumerate() {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }

        let pools: [Pool; ROOT::LEVEL] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
            snapshot_count: 0,
        }
    }
    pub fn new_with_leaf_storage(storage: Box<dyn PoolStorage>) -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL];
        for (i, meta) in ROOT::META.iter().take(ROOT::LEVEL).enumerate().skip(1) {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }
        pools[0].write(Pool::new_with_storage(ROOT::META[0].layout, storage));

        let pools: [Pool; ROOT::LEVEL] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
            snapshot_count: 0,
        }
    }
    pub fn pools(&self) -> &[Pool] {
        &self.pool
    }
    pub unsafe fn alloc_node<CHILD: Node>(&mut self) -> u32 {
        unsafe {
            if ROOT::LEVEL <= CHILD::LEVEL {
                panic!("Can not allocate root node");
            }
            let pool = &mut self.pool[CHILD::LEVEL as usize];
            pool.alloc::<CHILD>()
        }
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node<CHILD: Node>(&self, ptr: u32) -> &CHILD {
        unsafe {
            if CHILD::LEVEL == ROOT::LEVEL {
                // specialization for root
                return &*(&self.root as *const ROOT as *const CHILD);
            }
            &*(self.pool[CHILD::LEVEL as usize].get(ptr) as *const CHILD)
        }
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node_mut<CHILD: Node>(&mut self, ptr: u32) -> &mut CHILD {
        unsafe {
            if CHILD::LEVEL == ROOT::LEVEL {
                // specialization for root
                return &mut *(&mut self.root as *mut ROOT as *mut CHILD);
            }
            &mut *(self.pool[CHILD::LEVEL as usize].get_mut(ptr) as *mut CHILD)
        }
    }

    /// Number of live snapshots ([`Tree::snapshot`] minus
    /// [`Tree::release_snapshot`]).
    pub fn snapshot_count(&self) -> u32 {
        self.snapshot_count
    }

    /// Capture the current state of the tree as a self-contained read-only
    /// snapshot.
    ///
    /// Cost is independent of tree size: the root node is cloned, each of its
    /// direct children gains one reference count entry, and each pool's
    /// current backing allocation is captured ([`Pool::capture`]). Later
    /// mutations copy shared nodes on write instead of mutating them in place,
    /// so the snapshot keeps observing the captured state — including from
    /// other threads — while the tree moves on.
    pub fn snapshot(&mut self) -> TreeSnapshot<ROOT> {
        self.root.retain_children(&mut self.pool);
        self.snapshot_count += 1;
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL];
        for (capture, pool) in pools.iter_mut().zip(self.pool.iter()) {
            capture.write(pool.capture());
        }
        TreeSnapshot {
            root: self.root.clone(),
            pool: unsafe { MaybeUninit::array_assume_init(pools) },
            aabb: self.aabb,
            drop_guard: TreeSnapshotDropGuard,
        }
    }

    /// Release a snapshot, freeing every node that stayed allocated solely for
    /// it. `leaf_dropped` is called for each leaf node freed this way, so
    /// externally managed per-leaf resources (e.g. the attribute range
    /// referenced by the leaf's value) can be reclaimed by the caller.
    pub fn release_snapshot(
        &mut self,
        snapshot: TreeSnapshot<ROOT>,
        mut leaf_dropped: impl FnMut(&ROOT::LeafType),
    ) {
        snapshot
            .root
            .release_children(&mut self.pool, &mut leaf_dropped);
        self.snapshot_count -= 1;
        snapshot.drop_guard.disarm();
    }

    /// Restore the tree to the state captured by `snapshot` (undo).
    ///
    /// The snapshot stays valid and must still be released eventually; an undo
    /// stack can therefore restore the same snapshot repeatedly. Nodes only
    /// reachable from the abandoned working state are freed, reporting dropped
    /// leaves to `leaf_dropped`.
    pub fn restore(
        &mut self,
        snapshot: &TreeSnapshot<ROOT>,
        mut leaf_dropped: impl FnMut(&ROOT::LeafType),
    ) {
        // Pin the snapshot's children with the edges the restored root is
        // about to hold *before* releasing the current root's edges, so that
        // subtrees referenced by both can never hit refcount zero in between.
        snapshot.root.retain_children(&mut self.pool);
        self.root
            .release_children(&mut self.pool, &mut leaf_dropped);
        self.root = snapshot.root.clone();
        self.aabb = snapshot.aabb;
    }
}

impl<ROOT: Node> TreeLike for Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    type Iterator<'a> = ROOT::LeafIterator<'a>;

    fn iter_leaf(&self) -> Self::Iterator<'_> {
        self.root
            .iter_leaf(&self.pool, UVec3::ZERO)
    }
}

impl<ROOT: Node> TreeLike for TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    type Iterator<'a> = ROOT::LeafIterator<'a>;

    fn iter_leaf(&self) -> Self::Iterator<'_> {
        self.root
            .iter_leaf(&self.pool, UVec3::ZERO)
    }
}

impl<ROOT: Node> TreeErased for Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    fn count_leaves(&self) -> usize {
        self.root.count_leaves(&self.pool)
    }

    fn aabb(&self) -> AabbU32 {
        self.aabb
    }

    type LeafType = ROOT::LeafType;

    fn iter_leaf_erased(&self) -> ErasedLeafIter<'_, ROOT::LeafType> {
        ErasedLeafIter::new(&self.root, &self.pool)
    }
}

impl<ROOT: Node> TreeErased for TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    fn count_leaves(&self) -> usize {
        self.root.count_leaves(&self.pool)
    }

    fn aabb(&self) -> AabbU32 {
        self.aabb
    }

    type LeafType = ROOT::LeafType;

    fn iter_leaf_erased(&self) -> ErasedLeafIter<'_, ROOT::LeafType> {
        ErasedLeafIter::new(&self.root, &self.pool)
    }
}

/// The read-only interface shared by a live [`Tree`] and a captured
/// [`TreeSnapshot`]
///
/// ```
/// # #![feature(generic_const_exprs)]
/// use dust_vdb::{Tree, TreeErased, TreeLike, hierarchy};
///
/// /// Accepts a live tree and a snapshot alike.
/// fn occupied_voxels(version: &impl TreeLike) -> usize {
///     version.iter().count()
/// }
///
/// let tree = Tree::<hierarchy!(3, 3, 2, u32)>::new();
/// assert_eq!(tree.count_leaves(), 0);
/// assert_eq!(occupied_voxels(&tree), 0);
/// ```
pub trait TreeLike: TreeErased {
    /// Iterator returned by [`TreeLike::iter_leaf`]
    type Iterator<'a>: Iterator<Item = (UVec3, &'a Self::LeafType)> where Self: 'a;

    /// Iterate the leaf nodes present in this tree, yielding each leaf with
    /// the tree-global coordinate of its minimum corner (a multiple of
    /// [`Node::EXTENT`] of the leaf). Voxel coordinates inside the leaf are
    /// relative to it.
    ///
    /// Empty regions cost nothing: the walk descends only into occupied cells,
    /// driven by each node's child mask. Every leaf reached by this iterator
    /// holds at least one voxel.
    ///
    /// Order is depth-first, and within a node ascending child index, i.e.
    /// x-major with z varying fastest. It therefore depends only on *which*
    /// cells are occupied, never on insertion order or pool layout: two trees
    /// holding the same voxels iterate identically.
    fn iter_leaf(&self) -> Self::Iterator<'_>;

    /// Iterate every occupied voxel in tree-global coordinates, flattening
    /// [`TreeLike::iter_leaf`] through each leaf's occupancy mask.
    fn iter(&self) -> impl Iterator<Item = UVec3> {
        self.iter_leaf()
            .flat_map(|(position, leaf)| leaf.iter(position))
    }
}

/// The object-safe counterpart of [`TreeLike`], for storing trees or
/// snapshots of different hierarchies uniformly — e.g. behind a
/// `Box<dyn TreeErased<LeafType = L>>`. Only the hierarchy is erased; the
/// leaf type stays concrete in the object type.
///
/// [`TreeLike`] is the zero-cost interface: its iterator is the hierarchy's
/// own nested type, monomorphized with full type information. What keeps this
/// trait object-safe is that [`TreeErased::iter_leaf_erased`] returns
/// [`LeafIter`], a concrete iterator driven by per-level metadata instead of
/// the hierarchy's generics — still no dynamic dispatch and no allocation per
/// step, at the cost of runtime rather than compile-time tree geometry. Both
/// iterators yield the same items in the same documented order.
///
/// ```
/// # #![feature(generic_const_exprs)]
/// use dust_vdb::{Tree, TreeErased, hierarchy};
///
/// let tree = Tree::<hierarchy!(3, 3, 2, u32)>::new();
/// // Hierarchy-erased: only the leaf type remains in the object type.
/// let erased: &dyn TreeErased<LeafType = hierarchy!(2, u32)> = &tree;
/// assert_eq!(erased.count_leaves(), 0);
/// assert_eq!(erased.iter_erased().count(), 0);
/// ```
pub trait TreeErased: Send + Sync + 'static {
    /// The leaf node type of this hierarchy, carrying the occupancy mask and
    /// the per-leaf value.
    type LeafType: IsLeaf;

    /// Number of leaf nodes present in this version, i.e. exactly as many
    /// items as [`TreeLike::iter_leaf`] yields. Useful for sizing a buffer
    /// before filling it from that iterator.
    ///
    /// Not cached: this walks every internal node, summing child masks with
    /// `popcnt` one level above the leaves.
    ///
    /// Meaningful only for hierarchies that have at least one internal level
    /// above the leaves; a tree whose root *is* a leaf trips a debug assertion.
    fn count_leaves(&self) -> usize;

    /// Axis-aligned bounds of this version, in tree-global voxel coordinates.
    fn aabb(&self) -> AabbU32;

    /// Iterate the leaf nodes present in this tree through the
    /// hierarchy-erased walker. Same items, same order as
    /// [`TreeLike::iter_leaf`].
    fn iter_leaf_erased(&self) -> ErasedLeafIter<'_, Self::LeafType>;

    /// Iterate every occupied voxel in tree-global coordinates. Same items,
    /// same order as [`TreeLike::iter`].
    fn iter_erased(&self) -> ErasedVoxelIter<'_, Self::LeafType> {
        ErasedVoxelIter::new(self.iter_leaf_erased())
    }
}

/// Per-level walk constants, copied out of [`Node::META`] when the iterator is
/// created. Everything [`LeafIter::next`] needs is here, which is what makes
/// the iterator independent of the hierarchy's types.
#[derive(Clone, Copy)]
struct LevelInfo {
    /// Decode a child index into a cell coordinate: `x = i >> shift_x`,
    /// `y = (i >> shift_y) & mask_y`, `z = i & mask_z` — the inverse of the
    /// packing used by `InternalNode` (x-major, z fastest).
    shift_x: u32,
    shift_y: u32,
    mask_y: u32,
    mask_z: u32,
    /// Extent of one child cell, in voxels.
    child_extent: UVec3,
    /// Byte offset of the child mask words within the node.
    mask_offset: u32,
    /// Number of `usize` words in the child mask.
    mask_words: u32,
    /// Byte offset of the `[InternalNodeEntry; SIZE]` array within the node.
    child_ptrs_offset: u32,
}

impl LevelInfo {
    fn new<V>(meta: &NodeMeta<V>) -> Self {
        let fanout = meta.fanout_log2;
        Self {
            shift_x: fanout.y + fanout.z,
            shift_y: fanout.z,
            mask_y: (1 << fanout.y) - 1,
            mask_z: (1 << fanout.z) - 1,
            child_extent: meta.child_extent,
            mask_offset: meta.mask_offset,
            mask_words: meta.mask_words,
            child_ptrs_offset: meta.child_ptrs_offset,
        }
    }
}

/// One level of the walk: an internal node whose child mask is being scanned.
#[derive(Clone, Copy)]
struct Frame {
    /// Raw bytes of the node, read through the [`LevelInfo`] offsets.
    node: *const u8,
    /// Bits of the current mask word not yet visited.
    word: usize,
    word_idx: u32,
    /// Voxel coordinate of the node's minimum corner.
    origin: UVec3,
}

impl Frame {
    const EMPTY: Self = Self {
        node: std::ptr::null(),
        word: 0,
        word_idx: 0,
        origin: UVec3::ZERO,
    };
}

/// Read mask word `idx` of the node behind `frame`-style raw bytes.
///
/// Safety: `node` must point to a live node whose layout `info` describes.
#[inline]
unsafe fn mask_word(node: *const u8, info: &LevelInfo, idx: u32) -> usize {
    unsafe { *(node.add(info.mask_offset as usize) as *const usize).add(idx as usize) }
}

/// The iterator returned by [`crate::TreeErased::iter_leaf_erased`]: walks
/// every leaf of a tree, yielding each with the coordinate of its minimum
/// corner.
///
/// The type is erased over the hierarchy — only the leaf type `L` appears —
/// yet `next` involves no dynamic dispatch: instead of nesting per-level
/// generic iterators, the walk runs on a stack of per-level (geometry, mask
/// cursor) records copied from [`Node::META`] at construction. Each step is a
/// `trailing_zeros` on the current mask word, an index decode, and a pool
/// lookup, whichever hierarchy produced the tree.
///
/// The record stack is one boxed slice holding exactly `root_level` entries —
/// the iterator's only allocation, sized by the hierarchy's actual depth with
/// no depth limit (and elided entirely for a hierarchy whose root is a leaf).
pub struct ErasedLeafIter<'a, L> {
    pools: &'a [Pool],
    /// Record for tree level `k` at `levels[k - 1]`, `k = 1..=root_level`;
    /// leaves (level 0) are yielded, never opened, and need no record.
    levels: Box<[(LevelInfo, Frame)]>,
    /// Number of live frames; the deepest open level is
    /// `root_level + 1 - depth`.
    depth: u32,
    root_level: u32,
    /// A hierarchy whose root is itself a leaf yields it here, once.
    root_leaf: Option<&'a L>,
}

/// Everything the iterator dereferences is reachable through the `&'a` borrows
/// it was created from (the root node and the pools), so it inherits their
/// thread-safety: the raw pointers are an implementation detail.
unsafe impl<L: Sync> Send for ErasedLeafIter<'_, L> {}
unsafe impl<L: Sync> Sync for ErasedLeafIter<'_, L> {}

impl<'a, L> ErasedLeafIter<'a, L> {
    /// Walk the leaves of the tree rooted at `root`, whose non-root nodes
    /// live in `pools` (pools[k] holding level-k nodes, exactly as in
    /// [`crate::Tree`]).
    pub(crate) fn new<ROOT: Node<LeafType = L>>(root: &'a ROOT, pools: &'a [Pool]) -> Self
    where
        [(); ROOT::LEVEL + 1]: Sized,
    {
        let mut records: Vec<(LevelInfo, Frame)> = (1..=ROOT::LEVEL)
            .map(|level| (LevelInfo::new(&ROOT::META[level]), Frame::EMPTY))
            .collect();
        let mut root_leaf = None;
        let mut depth = 0;
        if ROOT::LEVEL == 0 {
            // The root is itself a leaf. A level-0 node is its own LeafType
            // (leaf impls define `LeafType = Self`), so the cast is the
            // identity.
            root_leaf = Some(unsafe { &*(root as *const ROOT as *const L) });
        } else {
            let node = root as *const ROOT as *const u8;
            let (info, frame) = records.last_mut().unwrap();
            *frame = Frame {
                node,
                word: unsafe { mask_word(node, info, 0) },
                word_idx: 0,
                origin: UVec3::ZERO,
            };
            depth = 1;
        }
        ErasedLeafIter {
            pools,
            levels: records.into_boxed_slice(),
            depth,
            root_level: ROOT::LEVEL as u32,
            root_leaf,
        }
    }
}

impl<'a, L> Iterator for ErasedLeafIter<'a, L> {
    type Item = (UVec3, &'a L);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(leaf) = self.root_leaf.take() {
            return Some((UVec3::ZERO, leaf));
        }
        while self.depth > 0 {
            let level = (self.root_level + 1 - self.depth) as usize;
            let (info, frame) = &mut self.levels[level - 1];

            // Advance to the next set child bit, popping the frame when its
            // mask is exhausted.
            let index = loop {
                if frame.word != 0 {
                    let bit = frame.word.trailing_zeros();
                    frame.word &= frame.word - 1;
                    break Some(frame.word_idx * usize::BITS + bit);
                }
                frame.word_idx += 1;
                if frame.word_idx >= info.mask_words {
                    break None;
                }
                frame.word = unsafe { mask_word(frame.node, info, frame.word_idx) };
            };
            let Some(index) = index else {
                self.depth -= 1;
                continue;
            };

            // A set mask bit guarantees the entry holds an occupied pointer.
            let child_ptr = unsafe {
                (*(frame.node.add(info.child_ptrs_offset as usize) as *const InternalNodeEntry)
                    .add(index as usize))
                .occupied
            };
            let cell = UVec3 {
                x: index >> info.shift_x,
                y: (index >> info.shift_y) & info.mask_y,
                z: index & info.mask_z,
            };
            let origin = frame.origin + cell * info.child_extent;

            if level == 1 {
                let leaf = unsafe { &*(self.pools[0].get(child_ptr) as *const L) };
                return Some((origin, leaf));
            }
            // Descend into an internal child.
            let child_level = level - 1;
            let node = unsafe { self.pools[child_level].get(child_ptr) };
            let (child_info, child_frame) = &mut self.levels[child_level - 1];
            *child_frame = Frame {
                node,
                word: unsafe { mask_word(node, child_info, 0) },
                word_idx: 0,
                origin,
            };
            self.depth += 1;
        }
        None
    }
}

/// The iterator returned by [`crate::TreeErased::iter_erased`]: every occupied
/// voxel in tree-global coordinates, [`LeafIter`] flattened through each
/// leaf's occupancy mask. Like [`LeafIter`], erased over the hierarchy with no
/// dynamic dispatch anywhere.
pub struct ErasedVoxelIter<'a, L: IsLeaf> {
    leaves: ErasedLeafIter<'a, L>,
    current: Option<L::Iterator<'a>>,
}

impl<'a, L: IsLeaf> ErasedVoxelIter<'a, L> {
    pub fn new(leaves: ErasedLeafIter<'a, L>) -> Self {
        Self {
            leaves,
            current: None,
        }
    }
}

impl<'a, L: IsLeaf> Iterator for ErasedVoxelIter<'a, L> {
    type Item = UVec3;

    fn next(&mut self) -> Option<UVec3> {
        loop {
            if let Some(current) = &mut self.current
                && let Some(voxel) = current.next()
            {
                return Some(voxel);
            }
            let (origin, leaf) = self.leaves.next()?;
            self.current = Some(leaf.iter(origin));
        }
    }
}
