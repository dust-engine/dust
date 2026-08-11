use std::{mem::{ManuallyDrop, MaybeUninit}, ops::Deref};

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
    snapshot_return_receiver: crossbeam::channel::Receiver<ROOT>,
    snapshot_return_sender: crossbeam::channel::Sender<ROOT>,
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
    pub(crate) root: ManuallyDrop<ROOT>,
    pub(crate) pool: [Pool; ROOT::LEVEL],
    aabb: AabbU32,
    return_channel: crossbeam::channel::Sender<ROOT>,
}
impl<ROOT: Node> Drop for TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL]: Sized,
{
    fn drop(&mut self) {
        unsafe {
            let root = ManuallyDrop::take(&mut self.root);
            self.return_channel.send(root).ok();
        }
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
        let (snapshot_return_sender, snapshot_return_receiver) = crossbeam::channel::unbounded();
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
            snapshot_count: 0,
            snapshot_return_sender,
            snapshot_return_receiver,
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
        let (snapshot_return_sender, snapshot_return_receiver) = crossbeam::channel::unbounded();
        Self {
            root: ROOT::default(),
            pool: pools,
            aabb: AabbU32::default(),
            snapshot_count: 0,
            snapshot_return_sender,
            snapshot_return_receiver,
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
            root: ManuallyDrop::new(self.root.clone()),
            pool: unsafe { MaybeUninit::array_assume_init(pools) },
            aabb: self.aabb,
            return_channel: self.snapshot_return_sender.clone(),
        }
    }

    /// Release a snapshot, freeing every node that stayed allocated solely for
    /// it. `leaf_dropped` is called for each leaf node freed this way, so
    /// externally managed per-leaf resources (e.g. the attribute range
    /// referenced by the leaf's value) can be reclaimed by the caller.
    pub fn release_snapshot(
        &mut self,
        snapshot: TreeSnapshot<ROOT>,
        leaf_dropped: impl FnMut(&ROOT::LeafType),
    ) {
        drop(snapshot);
        self.reclaim_dropped_snapshots(leaf_dropped);
    }

    /// Releases every snapshot that was dropped rather than explicitly
    /// released since the last reclamation, freeing the nodes that stayed
    /// allocated solely for them. Returns how many snapshots were reclaimed.
    ///
    /// `leaf_dropped` serves the same purpose as in
    /// [`Tree::release_snapshot`]. Mutation sessions ([`Tree::accessor_mut`])
    /// call this automatically with a callback that frees the leaves'
    /// attribute ranges.
    pub fn reclaim_dropped_snapshots(
        &mut self,
        mut leaf_dropped: impl FnMut(&ROOT::LeafType),
    ) -> u32 {
        let mut num_dropped: u32 = 0;
        for root in self.snapshot_return_receiver.try_iter() {
            root.release_children(&mut self.pool, &mut leaf_dropped);
            num_dropped += 1;
        }
        self.snapshot_count -= num_dropped;
        num_dropped
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
        self.root = snapshot.root.deref().clone();
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

    fn iter_erased(&self) -> ErasedVoxelIter<'_> {
        ErasedVoxelIter::new(&self.root, &self.pool)
    }
}

impl<ROOT: Node> TreeErasedLeaf for Tree<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
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

    fn iter_erased(&self) -> ErasedVoxelIter<'_> {
        ErasedVoxelIter::new(&self.root, &self.pool)
    }
}

impl<ROOT: Node> TreeErasedLeaf for TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
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
pub trait TreeLike: TreeErasedLeaf {
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

/// The fully type-erased read-only interface: nothing about the tree — not
/// the hierarchy, not even its leaf type — appears in the object type.
/// Consumers that only need occupancy (voxel coordinates, counts, bounds) can
/// hold any tree or snapshot as a plain `&dyn TreeErased` or
/// `Box<dyn TreeErased>` without naming a single tree parameter.
///
/// The interface comes in three tiers of erasure:
/// - [`TreeLike`] — full type information, the hierarchy's own monomorphized
///   iterators; the fast path. Not object-safe (its iterator is a generic
///   associated type).
/// - [`TreeErasedLeaf`] — hierarchy erased, leaf type still named:
///   `dyn TreeErasedLeaf<LeafType = L>` iterates leaves as `&L`.
/// - [`TreeErased`] — everything erased: iteration yields bare voxel
///   coordinates, walked through runtime tree geometry.
///
/// Each tier is a supertrait of the one above, so every tree offers all
/// three, and a `&dyn TreeErasedLeaf<LeafType = L>` upcasts to
/// `&dyn TreeErased`.
///
/// ```
/// # #![feature(generic_const_exprs)]
/// use dust_vdb::{Tree, TreeErased, TreeErasedLeaf, hierarchy};
///
/// let tree = Tree::<hierarchy!(3, 3, 2, u32)>::new();
/// // Nothing about the tree's shape or leaves needs to be named:
/// let opaque: &dyn TreeErased = &tree;
/// assert_eq!(opaque.count_leaves(), 0);
/// assert_eq!(opaque.iter_erased().count(), 0);
///
/// // The leaf-typed object upcasts to the fully erased one.
/// let typed: &dyn TreeErasedLeaf<LeafType = hierarchy!(2, u32)> = &tree;
/// let opaque: &dyn TreeErased = typed;
/// assert_eq!(opaque.aabb().min, glam::UVec3::MAX);
/// ```
pub trait TreeErased: Send + Sync + 'static {
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

    /// Iterate every occupied voxel in tree-global coordinates. Same items,
    /// same order as [`TreeLike::iter`].
    fn iter_erased(&self) -> ErasedVoxelIter<'_>;
}

/// The object-safe counterpart of [`TreeLike`], for storing trees or
/// snapshots of different hierarchies uniformly — e.g. behind a
/// `Box<dyn TreeErasedLeaf<LeafType = L>>`. Only the hierarchy is erased; the
/// leaf type stays concrete in the object type, so iteration still hands out
/// typed `&L` leaves. When not even the leaf type is known, use the
/// [`TreeErased`] supertrait instead.
///
/// [`TreeLike`] is the zero-cost interface: its iterator is the hierarchy's
/// own nested type, monomorphized with full type information. What keeps this
/// trait object-safe is that [`TreeErasedLeaf::iter_leaf_erased`] returns
/// [`ErasedLeafIter`], a concrete iterator driven by per-level metadata
/// instead of the hierarchy's generics — still no dynamic dispatch and no
/// allocation per step, at the cost of runtime rather than compile-time tree
/// geometry. Both iterators yield the same items in the same documented order.
///
/// ```
/// # #![feature(generic_const_exprs)]
/// use dust_vdb::{Tree, TreeErased, TreeErasedLeaf, hierarchy};
///
/// let tree = Tree::<hierarchy!(3, 3, 2, u32)>::new();
/// // Hierarchy-erased: only the leaf type remains in the object type.
/// let erased: &dyn TreeErasedLeaf<LeafType = hierarchy!(2, u32)> = &tree;
/// assert_eq!(erased.count_leaves(), 0);
/// assert_eq!(erased.iter_leaf_erased().count(), 0);
/// ```
pub trait TreeErasedLeaf: TreeErased {
    /// The leaf node type of this hierarchy, carrying the occupancy mask and
    /// the per-leaf value.
    type LeafType: IsLeaf;

    /// Iterate the leaf nodes present in this tree through the
    /// hierarchy-erased walker. Same items, same order as
    /// [`TreeLike::iter_leaf`].
    fn iter_leaf_erased(&self) -> ErasedLeafIter<'_, Self::LeafType>;
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

/// The iterator returned by [`crate::TreeErasedLeaf::iter_leaf_erased`]: walks
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
/// voxel in tree-global coordinates.
///
/// Fully type-erased — nothing about the hierarchy, not even the leaf type,
/// appears in the iterator, yet `next` involves no dynamic dispatch. The leaf
/// level needs no special machinery for that: a leaf's occupancy mask is
/// scanned exactly like an internal node's child mask, as one more record on
/// the same stack — the only difference is that a set bit at level 0 decodes
/// to a voxel coordinate instead of a child to descend into.
pub struct ErasedVoxelIter<'a> {
    pools: &'a [Pool],
    /// Record for tree level `k` at `levels[k]`, `k = 0..=root_level`: levels
    /// above 0 scan child masks, level 0 scans leaf occupancy.
    levels: Box<[(LevelInfo, Frame)]>,
    /// Number of live frames; the deepest open level is
    /// `root_level + 1 - depth`.
    depth: u32,
    root_level: u32,
}

/// Everything the iterator dereferences is reachable through the `&'a` borrows
/// it was created from (the root node and the pools), so it inherits their
/// thread-safety: the raw pointers are an implementation detail. Nodes are
/// unconditionally `Sync` ([`Node`] requires it), and only coordinates are
/// yielded.
unsafe impl Send for ErasedVoxelIter<'_> {}
unsafe impl Sync for ErasedVoxelIter<'_> {}

impl<'a> ErasedVoxelIter<'a> {
    /// Walk the voxels of the tree rooted at `root`, whose non-root nodes
    /// live in `pools` (pools[k] holding level-k nodes, exactly as in
    /// [`crate::Tree`]).
    pub(crate) fn new<ROOT: Node>(root: &'a ROOT, pools: &'a [Pool]) -> Self
    where
        [(); ROOT::LEVEL + 1]: Sized,
    {
        let mut records: Vec<(LevelInfo, Frame)> = (0..=ROOT::LEVEL)
            .map(|level| (LevelInfo::new(&ROOT::META[level]), Frame::EMPTY))
            .collect();
        // The root opens like any other node — for a hierarchy whose root is
        // itself a leaf, the walk simply starts (and ends) at level 0,
        // scanning the root's occupancy words.
        let node = root as *const ROOT as *const u8;
        let (info, frame) = records.last_mut().unwrap();
        *frame = Frame {
            node,
            word: unsafe { mask_word(node, info, 0) },
            word_idx: 0,
            origin: UVec3::ZERO,
        };
        ErasedVoxelIter {
            pools,
            levels: records.into_boxed_slice(),
            depth: 1,
            root_level: ROOT::LEVEL as u32,
        }
    }
}

impl Iterator for ErasedVoxelIter<'_> {
    type Item = UVec3;

    #[inline(always)]
    fn next(&mut self) -> Option<UVec3> {
        while self.depth > 0 {
            let level = (self.root_level + 1 - self.depth) as usize;
            let (info, frame) = &mut self.levels[level];

            // Advance to the next set bit, popping the frame when its mask is
            // exhausted.
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

            let cell = UVec3 {
                x: index >> info.shift_x,
                y: (index >> info.shift_y) & info.mask_y,
                z: index & info.mask_z,
            };
            if level == 0 {
                // A set occupancy bit: the decoded cell is the voxel itself.
                return Some(frame.origin + cell);
            }

            // A set child-mask bit guarantees the entry holds an occupied
            // pointer. Descend — into a leaf's occupancy (level 1 → 0) and an
            // internal node's child mask alike.
            let child_ptr = unsafe {
                (*(frame.node.add(info.child_ptrs_offset as usize) as *const InternalNodeEntry)
                    .add(index as usize))
                .occupied
            };
            let origin = frame.origin + cell * info.child_extent;
            let node = unsafe { self.pools[level - 1].get(child_ptr) };
            let (child_info, child_frame) = &mut self.levels[level - 1];
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
