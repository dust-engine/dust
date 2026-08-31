use std::{mem::{ManuallyDrop, MaybeUninit}, ops::Deref};

use glam::UVec3;

use crate::{
    AabbU32, AttributePtr, InternalNodeEntry, IsLeaf, Node, NodeMeta, pool::{Pool, PoolStorage},
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
        leaf_dropped: impl FnMut(u32, &ROOT::LeafType),
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
        mut leaf_dropped: impl FnMut(u32, &ROOT::LeafType),
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
        mut leaf_dropped: impl FnMut(u32, &ROOT::LeafType),
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

    fn extent(&self) -> UVec3 {
        ROOT::EXTENT
    }

    fn leaf_extent(&self) -> UVec3 {
        <ROOT::LeafType as Node>::EXTENT
    }

    fn iter_erased(&self) -> ErasedVoxelIter<'_> {
        ErasedVoxelIter::new(&self.root, &self.pool)
    }

    fn iter_leaf_views_in_range(&self, range: AabbU32) -> ErasedLeafViewIter<'_, ()> {
        ErasedLeafViewIter::new(&self.root, &self.pool, range)
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

    fn extent(&self) -> UVec3 {
        ROOT::EXTENT
    }

    fn leaf_extent(&self) -> UVec3 {
        <ROOT::LeafType as Node>::EXTENT
    }

    fn iter_erased(&self) -> ErasedVoxelIter<'_> {
        ErasedVoxelIter::new(&self.root, &self.pool)
    }

    fn iter_leaf_views_in_range(&self, range: AabbU32) -> ErasedLeafViewIter<'_, ()> {
        ErasedLeafViewIter::new(&self.root, &self.pool, range)
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

    /// The tree's addressable extent along each axis, regardless of
    /// voxel occupancy states. A property of the tree hierarchy.
    fn extent(&self) -> UVec3;

    /// The extent of one leaf node along each axis, regardless of
    /// voxel occupancy states. A property of the tree hierarchy.
    fn leaf_extent(&self) -> UVec3;

    /// Iterate every occupied voxel in tree-global coordinates. Same items,
    /// same order as [`TreeLike::iter`].
    fn iter_erased(&self) -> ErasedVoxelIter<'_>;

    /// Iterate views of the leaves intersecting `range` (inclusive on both
    /// bounds, like every [`AabbU32`]), in [`TreeLike::iter_leaf`] order. The
    /// walk descends only into children whose cells intersect the range, so a
    /// small box over a large tree costs the descent along the box — not a
    /// filtered full iteration.
    ///
    /// Each view exposes the leaf's occupancy words whole, to mask, scan, or
    /// compare against neighbors without re-descending per voxel. A yielded
    /// leaf intersects the range but need not lie inside it: voxels of a
    /// straddling leaf are the caller's to clip. A single-coordinate lookup
    /// is a range of one voxel — every descent starts from the root either
    /// way.
    fn iter_leaf_views_in_range(&self, range: AabbU32) -> ErasedLeafViewIter<'_, ()>;
}

/// A [`TreeErased`] whose leaf values project an attribute pointer of type
/// `V`, through the value's [`AttributePtr<V>`] implementation.
///
/// [`TreeErased`] names nothing about the leaf values, so its leaf views
/// carry no pointer (`ErasedLeafView<()>`). A consumer that reads attribute
/// stores with `Ptr = V` asks for this trait instead — e.g. a collision
/// shape holding `Arc<dyn TreeWithValues<u32>>` accepts a tree of any
/// hierarchy whose leaf value converts to a `u32` pointer — and its views
/// hand the projected pointer out ([`ErasedLeafView::attribute_ptr`]).
///
/// Implemented blanketly: every [`TreeErasedLeaf`] whose
/// [`LeafType`](TreeErasedLeaf::LeafType) has a value implementing
/// [`AttributePtr<V>`] is a `TreeWithValues<V>`. A plain `u32` leaf value is
/// its own pointer (the blanket identity `impl AttributePtr<u32> for u32`),
/// and one value type can project several pointer types — `VoxLeafNode`
/// in `dust_vox` projects its `material_ptr` as the `u32` pointer.
///
/// ```
/// # #![feature(generic_const_exprs)]
/// use dust_vdb::{AabbU32, Tree, TreeWithValues, hierarchy};
/// use glam::UVec3;
///
/// let tree = Tree::<hierarchy!(3, 3, 2, u32)>::new();
/// // Only the pointer type remains in the object type; the hierarchy — and
/// // whether the leaf value *is* a `u32` or merely projects one — is erased.
/// let with_values: &dyn TreeWithValues<u32> = &tree;
/// let everything = AabbU32 { min: UVec3::ZERO, max: UVec3::splat(255) };
/// assert_eq!(
///     with_values
///         .iter_leaf_views_in_range_with_values(everything)
///         .count(),
///     0
/// );
/// ```
pub trait TreeWithValues<V>: TreeErased {
    /// Like [`TreeErased::iter_leaf_views_in_range`], with each yielded view
    /// additionally carrying the leaf value's projected attribute pointer
    /// ([`ErasedLeafView::attribute_ptr`]).
    fn iter_leaf_views_in_range_with_values(
        &self,
        range: AabbU32,
    ) -> ErasedLeafViewIter<'_, V>;
}

impl<T: TreeErasedLeaf, V> TreeWithValues<V> for T
where
    <T::LeafType as IsLeaf>::Value: AttributePtr<V>,
{
    fn iter_leaf_views_in_range_with_values(
        &self,
        range: AabbU32,
    ) -> ErasedLeafViewIter<'_, V> {
        // `LeafType` is exactly the leaf type of the tree the plain walk
        // iterates, which is what makes the byte reinterpretation inside
        // `with_values` correct.
        self.iter_leaf_views_in_range(range)
            .with_values::<T::LeafType, V>()
    }
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

/// A read-only view of one leaf's occupancy bits, yielded by
/// [`crate::TreeErased::iter_leaf_views_in_range`]: once the walk reaches a
/// leaf, any voxel of that leaf can be tested without descending again.
///
/// `T` is what [`ErasedLeafView::attribute_ptr`] hands out: `()` on a plain
/// [`TreeErased`](crate::TreeErased) walk, the leaf value's projected
/// attribute pointer on a [`TreeWithValues`](crate::TreeWithValues) walk.
///
/// Like the erased iterators, the view names nothing about the hierarchy —
/// not even the leaf type — and involves no dynamic dispatch: it reads the
/// occupancy words through offsets from [`Node::META`]. Coordinates are
/// tree-global; the leaf covers `[origin, origin + extent)`. The view also
/// identifies the leaf ([`ErasedLeafView::leaf_index`]) so side tables keyed
/// the way [`Attributes`](crate::Attributes) is driven can be read through it.
#[derive(Clone, Copy)]
pub struct ErasedLeafView<'a, T> {
    /// Tree-global coordinate of the leaf's minimum corner (a multiple of
    /// `extent`).
    origin: UVec3,
    /// Pool index of the leaf node; `u32::MAX` for the root leaf of a
    /// hierarchy whose root is itself a leaf (it lives outside the pools).
    index: u32,
    /// The leaf value's [`AttributePtr::attribute_ptr`] projection, read
    /// when the view was stamped; `()` for views of a plain [`TreeErased`]
    /// walk.
    attribute_ptr: T,
    extent: UVec3,
    /// = extent - 1 (extents are powers of two).
    extent_mask: UVec3,
    /// Decode a leaf-relative coordinate into an occupancy bit index:
    /// `i = (x << shift_x) | (y << shift_y) | z` — the packing used by
    /// `LeafNode` (x-major, z fastest).
    shift_x: u32,
    shift_y: u32,
    /// Byte offset of the occupancy words within the leaf node.
    mask_offset: u32,
    /// Number of `usize` words in the occupancy mask.
    mask_words: u32,
    /// Raw bytes of the leaf node.
    node: *const u8,
    _borrow: std::marker::PhantomData<&'a ()>,
}

/// The raw pointer is a borrow of a live node in disguise (see `_borrow`);
/// nodes are unconditionally `Sync` ([`Node`] requires it).
unsafe impl<T: Send> Send for ErasedLeafView<'_, T> {}
unsafe impl<T: Sync> Sync for ErasedLeafView<'_, T> {}

impl<'a, T> ErasedLeafView<'a, T> {
    /// Tree-global coordinate of the leaf's minimum corner.
    pub fn origin(&self) -> UVec3 {
        self.origin
    }

    /// The pool index of the viewed leaf: the same `leaf` index the tree
    /// passes to [`Attributes`](crate::Attributes) while it is mutated, so
    /// side tables keyed by it can be read back through this view. Stable
    /// within one tree version; a copy-on-write edit re-homes the edited leaf
    /// to a new index (with a matching
    /// [`Attributes::copy_attribute`](crate::Attributes::copy_attribute)
    /// event). `u32::MAX` for the root leaf of a hierarchy whose root is
    /// itself a leaf — it lives outside the pools.
    pub fn leaf_index(&self) -> u32 {
        self.index
    }

    /// The leaf value's attribute pointer, projected through the value's
    /// [`AttributePtr`](crate::AttributePtr) implementation when the view was
    /// stamped. This is the same `ptr` the accessor passes to
    /// [`Attributes`](crate::Attributes) methods for this leaf, so a store
    /// with a matching `Ptr` can be indexed through the view, with no
    /// accessor and no hierarchy type in sight.
    ///
    /// Views of a plain [`TreeErased`](crate::TreeErased) walk carry `()`;
    /// walks started through
    /// [`TreeWithValues`](crate::TreeWithValues::iter_leaf_views_in_range_with_values)
    /// carry the projected pointer.
    pub fn attribute_ptr(&self) -> &T {
        &self.attribute_ptr
    }

    /// Extent of the leaf along each axis: it covers
    /// `[origin, origin + extent)`.
    pub fn extent(&self) -> UVec3 {
        self.extent
    }

    /// Whether the tree-global coordinate `coords` falls within this leaf.
    pub fn contains(&self, coords: UVec3) -> bool {
        ((coords ^ self.origin) & !self.extent_mask) == UVec3::ZERO
    }

    /// Whether the voxel at the tree-global coordinate `coords` is occupied.
    /// `coords` must be within this leaf ([`ErasedLeafView::contains`]).
    pub fn get(&self, coords: UVec3) -> bool {
        let index = self.bit_of_coord(coords);
        let word = unsafe {
            *(self.node.add(self.mask_offset as usize) as *const usize)
                .add(index as usize / usize::BITS as usize)
        };
        word & (1 << (index as usize % usize::BITS as usize)) != 0
    }

    /// The leaf's raw occupancy words. Bit `i` of the concatenated words (LSB
    /// first) is the voxel at the leaf-relative coordinate
    /// [`ErasedLeafView::coord_of_bit`]`(i)` — one bit per voxel, `extent`
    /// voxels total, in x-major order with z varying fastest. This is the
    /// whole leaf in hand at once, for consumers doing their own bit math
    /// (masking a range, diffing against a neighbor); [`ErasedLeafView::get`]
    /// is the one-voxel form.
    pub fn occupancy_words(&self) -> &'a [usize] {
        unsafe {
            std::slice::from_raw_parts(
                self.node.add(self.mask_offset as usize) as *const usize,
                self.mask_words as usize,
            )
        }
    }

    /// The occupancy bit index of the tree-global coordinate `coords`: the
    /// `(x << shift_x) | (y << shift_y) | z` packing over the leaf-relative
    /// coordinate, the inverse of [`ErasedLeafView::coord_of_bit`]. This is
    /// also the voxel's index within its leaf — the `inflated_offset` the
    /// accessor passes to [`Attributes`](crate::Attributes) methods. `coords`
    /// must be within this leaf ([`ErasedLeafView::contains`]).
    pub fn bit_of_coord(&self, coords: UVec3) -> u32 {
        debug_assert!(self.contains(coords), "coords outside of the viewed leaf");
        let cell = coords & self.extent_mask;
        (cell.x << self.shift_x) | (cell.y << self.shift_y) | cell.z
    }

    /// The leaf-relative coordinate of occupancy bit `index`: the inverse of
    /// the `(x << shift_x) | (y << shift_y) | z` packing.
    pub fn coord_of_bit(&self, index: u32) -> UVec3 {
        UVec3 {
            x: index >> self.shift_x,
            y: (index >> self.shift_y) & self.extent_mask.y,
            z: index & self.extent_mask.z,
        }
    }
}

/// The iterator returned by [`crate::TreeErased::iter_leaf_views_in_range`]:
/// an [`ErasedLeafView`] of every leaf intersecting a coordinate box.
///
/// The walk is the fully erased frame stack of [`ErasedVoxelIter`], stopping
/// one level up: leaves are yielded whole (as views over their occupancy
/// words) instead of being scanned bit by bit, and children whose cells lie
/// outside the box are never descended into.
pub struct ErasedLeafViewIter<'a, T> {
    pools: &'a [Pool],
    /// Record for tree level `k` at `levels[k - 1]`, `k = 1..=root_level`;
    /// leaves (level 0) are yielded, never opened, and need no record.
    levels: Box<[(LevelInfo, Frame)]>,
    /// Number of live frames; the deepest open level is
    /// `root_level + 1 - depth`.
    depth: u32,
    root_level: u32,
    /// Inclusive bounds of the walk.
    range: AabbU32,
    /// The leaf-level constants stamped onto every yielded view.
    leaf_extent: UVec3,
    leaf_extent_mask: UVec3,
    leaf_shift_x: u32,
    leaf_shift_y: u32,
    leaf_mask_offset: u32,
    leaf_mask_words: u32,
    /// Stamps [`ErasedLeafView::attribute_ptr`] from a leaf node's raw
    /// bytes: [`no_attribute_ptr`] on a plain walk (`T = ()`),
    /// [`leaf_attribute_ptr`] monomorphized for the tree's leaf type after
    /// [`ErasedLeafViewIter::with_values`].
    read_attribute_ptr: fn(*const u8) -> T,
    /// A hierarchy whose root is itself a leaf yields it here, once.
    root_leaf: Option<ErasedLeafView<'a, T>>,
}

/// The projection of a plain [`TreeErased`] walk: no leaf value is read and
/// every view's `attribute_ptr` is `()`.
fn no_attribute_ptr(_node: *const u8) {}

/// Reads the leaf value's [`AttributePtr<V>`] projection from the raw bytes
/// of a leaf node of type `L`. [`ErasedLeafViewIter`] stamps views from raw
/// node bytes after the leaf type has been erased; this function, stored as
/// a plain fn pointer by [`ErasedLeafViewIter::with_values`] while `L` is
/// still known, carries the typed read across that erasure.
fn leaf_attribute_ptr<L: IsLeaf, V>(node: *const u8) -> V
where
    L::Value: AttributePtr<V>,
{
    unsafe { (*(node as *const L)).get_value().attribute_ptr() }
}

/// See [`ErasedVoxelIter`]'s impls: the raw pointers are borrows of live,
/// unconditionally-`Sync` nodes.
unsafe impl<T: Send> Send for ErasedLeafViewIter<'_ , T> {}
unsafe impl<T: Sync> Sync for ErasedLeafViewIter<'_ , T> {}

impl<'a> ErasedLeafViewIter<'a, ()> {
    /// Walk the leaves of the tree rooted at `root`, whose non-root nodes
    /// live in `pools` (pools[k] holding level-k nodes, exactly as in
    /// [`crate::Tree`]), yielding views of leaves intersecting `range`
    /// (inclusive).
    pub(crate) fn new<ROOT: Node>(root: &'a ROOT, pools: &'a [Pool], range: AabbU32) -> Self
    where
        [(); ROOT::LEVEL + 1]: Sized,
    {
        let leaf_meta = &ROOT::META[0];
        let leaf_fanout = leaf_meta.fanout_log2;
        let leaf_extent = <ROOT::LeafType as Node>::EXTENT;

        let mut records: Vec<(LevelInfo, Frame)> = (1..=ROOT::LEVEL)
            .map(|level| (LevelInfo::new(&ROOT::META[level]), Frame::EMPTY))
            .collect();
        let mut root_leaf = None;
        let mut depth = 0;
        if ROOT::LEVEL == 0 {
            // The root is itself a leaf; yield it if it intersects the range
            // (its block starts at the origin, so only the low bound can
            // exclude it).
            if (leaf_extent - UVec3::ONE).cmpge(range.min).all() {
                root_leaf = Some(ErasedLeafView {
                    origin: UVec3::ZERO,
                    index: u32::MAX,
                    attribute_ptr: (),
                    extent: leaf_extent,
                    extent_mask: <ROOT::LeafType as Node>::EXTENT_MASK,
                    shift_x: leaf_fanout.y + leaf_fanout.z,
                    shift_y: leaf_fanout.z,
                    mask_offset: leaf_meta.mask_offset,
                    mask_words: leaf_meta.mask_words,
                    node: root as *const ROOT as *const u8,
                    _borrow: std::marker::PhantomData,
                });
            }
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
        ErasedLeafViewIter {
            pools,
            levels: records.into_boxed_slice(),
            depth,
            root_level: ROOT::LEVEL as u32,
            range,
            leaf_extent,
            leaf_extent_mask: <ROOT::LeafType as Node>::EXTENT_MASK,
            leaf_shift_x: leaf_fanout.y + leaf_fanout.z,
            leaf_shift_y: leaf_fanout.z,
            leaf_mask_offset: leaf_meta.mask_offset,
            leaf_mask_words: leaf_meta.mask_words,
            read_attribute_ptr: no_attribute_ptr,
            root_leaf,
        }
    }

    /// Upgrades this plain walk into one whose views carry the leaf value's
    /// projected attribute pointer: every view yielded from here on has
    /// [`ErasedLeafView::attribute_ptr`] read from the viewed leaf node
    /// through `L::Value`'s [`AttributePtr<V>`] implementation.
    ///
    /// `L` must be the leaf type of the tree this walk was created from —
    /// its raw node bytes are reinterpreted as `L` to read the value. The
    /// one caller, [`TreeWithValues`](crate::TreeWithValues)'s blanket impl,
    /// passes [`TreeErasedLeaf::LeafType`](crate::TreeErasedLeaf::LeafType),
    /// which is that type by definition.
    pub(crate) fn with_values<L: IsLeaf, V>(self) -> ErasedLeafViewIter<'a, V>
    where
        L::Value: AttributePtr<V>,
    {
        let read_attribute_ptr = leaf_attribute_ptr::<L, V> as fn(*const u8) -> V;
        ErasedLeafViewIter {
            pools: self.pools,
            levels: self.levels,
            depth: self.depth,
            root_level: self.root_level,
            range: self.range,
            leaf_extent: self.leaf_extent,
            leaf_extent_mask: self.leaf_extent_mask,
            leaf_shift_x: self.leaf_shift_x,
            leaf_shift_y: self.leaf_shift_y,
            leaf_mask_offset: self.leaf_mask_offset,
            leaf_mask_words: self.leaf_mask_words,
            read_attribute_ptr,
            // The root leaf (if any) was already stamped with `()`; re-stamp
            // it with the projection.
            root_leaf: self.root_leaf.map(|view| ErasedLeafView {
                origin: view.origin,
                index: view.index,
                attribute_ptr: read_attribute_ptr(view.node),
                extent: view.extent,
                extent_mask: view.extent_mask,
                shift_x: view.shift_x,
                shift_y: view.shift_y,
                mask_offset: view.mask_offset,
                mask_words: view.mask_words,
                node: view.node,
                _borrow: std::marker::PhantomData,
            }),
        }
    }
}

impl<'a, T> Iterator for ErasedLeafViewIter<'a, T> {
    type Item = ErasedLeafView<'a, T>;

    #[inline(always)]
    fn next(&mut self) -> Option<ErasedLeafView<'a, T>> {
        if let Some(leaf) = self.root_leaf.take() {
            return Some(leaf);
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

            let cell = UVec3 {
                x: index >> info.shift_x,
                y: (index >> info.shift_y) & info.mask_y,
                z: index & info.mask_z,
            };
            let origin = frame.origin + cell * info.child_extent;
            if origin.cmpgt(self.range.max).any()
                || (origin + info.child_extent - UVec3::ONE)
                    .cmplt(self.range.min)
                    .any()
            {
                // The child's cell lies entirely outside the box.
                continue;
            }

            // A set mask bit guarantees the entry holds an occupied pointer.
            let child_ptr = unsafe {
                (*(frame.node.add(info.child_ptrs_offset as usize) as *const InternalNodeEntry)
                    .add(index as usize))
                .occupied
            };

            if level == 1 {
                let node = unsafe { self.pools[0].get(child_ptr) };
                return Some(ErasedLeafView {
                    origin,
                    index: child_ptr,
                    attribute_ptr: (self.read_attribute_ptr)(node),
                    extent: self.leaf_extent,
                    extent_mask: self.leaf_extent_mask,
                    shift_x: self.leaf_shift_x,
                    shift_y: self.leaf_shift_y,
                    mask_offset: self.leaf_mask_offset,
                    mask_words: self.leaf_mask_words,
                    node,
                    _borrow: std::marker::PhantomData,
                });
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
