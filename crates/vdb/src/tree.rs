use std::{collections::BinaryHeap, mem::{ManuallyDrop, MaybeUninit}, ops::Deref};

use glam::{BVec3A, IVec3, UVec3, Vec3, Vec3A};

use crate::{
    AabbU32, AttributePtr, Attributes, InternalNodeEntry, IsLeaf, Node, NodeMeta,
    pool::{Pool, PoolStorage},
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
    type Root = ROOT;

    fn iter_leaf(&self) -> Self::Iterator<'_> {
        self.root
            .iter_leaf(&self.pool, UVec3::ZERO)
    }

    fn accessor<'a, A: Attributes>(&'a self, attributes: &'a A) -> crate::Accessor<'a, ROOT, A>
    where
        <ROOT::LeafType as IsLeaf>::Value: AttributePtr<A::Ptr>,
    {
        Tree::accessor(self, attributes)
    }
}

impl<ROOT: Node> TreeLike for TreeSnapshot<ROOT>
where
    [(); ROOT::LEVEL + 1]: Sized,
{
    type Iterator<'a> = ROOT::LeafIterator<'a>;
    type Root = ROOT;

    fn iter_leaf(&self) -> Self::Iterator<'_> {
        self.root
            .iter_leaf(&self.pool, UVec3::ZERO)
    }

    fn accessor<'a, A: Attributes>(&'a self, attributes: &'a A) -> crate::Accessor<'a, ROOT, A>
    where
        <ROOT::LeafType as IsLeaf>::Value: AttributePtr<A::Ptr>,
    {
        TreeSnapshot::accessor(self, attributes)
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

    fn iter_leaf_views_along_ray(
        &self,
        origin: Vec3,
        dir: Vec3,
        t0: f32,
        t1: f32,
    ) -> ErasedLeafViewRayIter<'_> {
        ErasedLeafViewRayIter::new(&self.root, &self.pool, origin, dir, t0, t1)
    }

    fn iter_leaf_views_near_point(
        &self,
        point: Vec3,
        scale: Vec3,
    ) -> ErasedLeafViewNearIter<'_> {
        ErasedLeafViewNearIter::new(&self.root, &self.pool, point, scale)
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

    fn iter_leaf_views_along_ray(
        &self,
        origin: Vec3,
        dir: Vec3,
        t0: f32,
        t1: f32,
    ) -> ErasedLeafViewRayIter<'_> {
        ErasedLeafViewRayIter::new(&self.root, &self.pool, origin, dir, t0, t1)
    }

    fn iter_leaf_views_near_point(
        &self,
        point: Vec3,
        scale: Vec3,
    ) -> ErasedLeafViewNearIter<'_> {
        ErasedLeafViewNearIter::new(&self.root, &self.pool, point, scale)
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

    /// The hierarchy's root node type: the tree covers `Root::EXTENT`, and
    /// its leaves are `Root::LeafType` — the same type as
    /// [`TreeErasedLeaf::LeafType`], which the bound spells out.
    type Root: Node<LeafType = Self::LeafType>;

    /// A cached read accessor over this version — [`Tree::accessor`] and
    /// [`TreeSnapshot::accessor`], reachable through the trait so a consumer
    /// generic over `impl TreeLike` can do cached point lookups
    /// ([`Accessor::leaf_view`](crate::Accessor::leaf_view), or
    /// [`Accessor::get`](crate::Accessor::get) when `attributes` is a real
    /// store). Pass `&()` as `attributes` for occupancy-only access.
    fn accessor<'a, A: Attributes>(&'a self, attributes: &'a A) -> crate::Accessor<'a, Self::Root, A>
    where
        [(); Self::Root::LEVEL + 1]: Sized,
        <Self::LeafType as IsLeaf>::Value: AttributePtr<A::Ptr>;

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

    /// Iterate views of the leaves pierced by a ray, in the order the ray
    /// reaches them.
    ///
    /// `origin` and `dir` are in voxel coordinates — the voxel at coordinate
    /// `c` covers `[c, c + 1)` on each axis — and the walk covers ray
    /// parameters `t` in `[t0, t1]`, where a parameter names the point
    /// `origin + dir * t`. Each yielded item is `(t, view)`: the parameter at
    /// which the ray enters the leaf's block (`t0` when it starts inside),
    /// strictly increasing across items. Which voxels of a yielded leaf the
    /// ray actually crosses is the caller's to walk, on the view's occupancy
    /// bits.
    ///
    /// Empty space costs little: at every level, only children whose mask bit
    /// is set are descended into, so a ray over a large empty region skips it
    /// a whole subtree at a time — the same pruning
    /// [`TreeErased::iter_leaf_views_in_range`] applies to a box, applied to
    /// a ray.
    fn iter_leaf_views_along_ray(
        &self,
        origin: Vec3,
        dir: Vec3,
        t0: f32,
        t1: f32,
    ) -> ErasedLeafViewRayIter<'_>;

    /// Search for the leaves nearest to a point: yielded in nondecreasing
    /// order of the distance between the point and each leaf's block, and
    /// pruned to the leaves that could still hold the point's nearest voxel.
    ///
    /// `point` is in voxel coordinates — the voxel at coordinate `c` covers
    /// `[c, c + 1)` on each axis — and `scale` converts per-axis voxel-space
    /// differences into the caller's metric: pass the per-axis voxel size to
    /// measure in the caller's local space, `Vec3::ONE` to measure in voxel
    /// coordinates. Each yielded item is `(dist_sq, view)`: the squared
    /// distance (in that metric) from `point` to the leaf's block — `0`
    /// while the point lies inside the block — and the leaf's view.
    ///
    /// This is a nearest-neighbor frontier, not a full enumeration. Every
    /// discovered block holds at least one voxel, so the nearest block's
    /// *farthest* corner bounds the answer from above, and any block whose
    /// nearest corner lies beyond that bound — provably unable to hold, or
    /// tie for, the nearest voxel — is skipped, subtree and all. A consumer
    /// tracking the best candidate over the yielded leaves' voxels can
    /// therefore stop at the first block too far to beat it, and what it
    /// never reads was never visited. The frontier's size varies with
    /// occupancy, so — unlike the other erased walks — creating this
    /// iterator allocates.
    fn iter_leaf_views_near_point(&self, point: Vec3, scale: Vec3)
    -> ErasedLeafViewNearIter<'_>;
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
    /// Number of child cells along each axis (`1 << fanout_log2`).
    fanout: UVec3,
    /// Byte offset of the child mask words within the node.
    mask_offset: u32,
    /// Number of `usize` words in the child mask.
    mask_words: u32,
    /// Byte offset of the `[InternalNodeEntry; SIZE]` array within the node.
    child_ptrs_offset: u32,
}

impl LevelInfo {
    /// Filler for the unused tail of a fixed-size level array; never read.
    const EMPTY: Self = Self {
        shift_x: 0,
        shift_y: 0,
        mask_y: 0,
        mask_z: 0,
        child_extent: UVec3::ONE,
        fanout: UVec3::ONE,
        mask_offset: 0,
        mask_words: 0,
        child_ptrs_offset: 0,
    };

    fn new<V>(meta: &NodeMeta<V>) -> Self {
        let fanout = meta.fanout_log2;
        Self {
            shift_x: fanout.y + fanout.z,
            shift_y: fanout.z,
            mask_y: (1 << fanout.y) - 1,
            mask_z: (1 << fanout.z) - 1,
            child_extent: meta.child_extent,
            fanout: UVec3::new(1 << fanout.x, 1 << fanout.y, 1 << fanout.z),
            mask_offset: meta.mask_offset,
            mask_words: meta.mask_words,
            child_ptrs_offset: meta.child_ptrs_offset,
        }
    }
}

/// The deepest hierarchy the hierarchy-erased iterators can walk, counted in
/// internal levels above the leaves (`ROOT::LEVEL`; `hierarchy!(3, 3, 2, _)`
/// has 2). Each iterator keeps its per-level records inline — a small
/// fixed-size array instead of a heap allocation per traversal — so the
/// bound is fixed: creating an erased iterator over a deeper hierarchy
/// panics, and raising this constant is the fix. ([`ErasedVoxelIter`] sizes
/// its array one longer, since it scans leaf occupancy as one more record on
/// the same stack.)
pub const MAX_ERASED_LEVELS: usize = 4;

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
/// The record stack is a fixed-size inline array ([`MAX_ERASED_LEVELS`]), so
/// creating the iterator allocates nothing.
pub struct ErasedLeafIter<'a, L> {
    pools: &'a [Pool],
    /// Record for tree level `k` at `levels[k - 1]`, `k = 1..=root_level`;
    /// leaves (level 0) are yielded, never opened, and need no record.
    /// Entries past `root_level` are unused filler.
    levels: [(LevelInfo, Frame); MAX_ERASED_LEVELS],
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
        assert!(
            ROOT::LEVEL <= MAX_ERASED_LEVELS,
            "hierarchy deeper than the erased walks' inline record arrays; raise MAX_ERASED_LEVELS"
        );
        let mut records = [(LevelInfo::EMPTY, Frame::EMPTY); MAX_ERASED_LEVELS];
        for level in 1..=ROOT::LEVEL {
            records[level - 1].0 = LevelInfo::new(&ROOT::META[level]);
        }
        let mut root_leaf = None;
        let mut depth = 0;
        if ROOT::LEVEL == 0 {
            // The root is itself a leaf. A level-0 node is its own LeafType
            // (leaf impls define `LeafType = Self`), so the cast is the
            // identity.
            root_leaf = Some(unsafe { &*(root as *const ROOT as *const L) });
        } else {
            let node = root as *const ROOT as *const u8;
            // The root's record: the last *used* entry (the array's tail
            // past it is filler).
            let (info, frame) = &mut records[ROOT::LEVEL - 1];
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
            levels: records,
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
    /// above 0 scan child masks, level 0 scans leaf occupancy. Inline and
    /// fixed-size, so creating the iterator allocates nothing; entries past
    /// `root_level` are unused filler.
    levels: [(LevelInfo, Frame); MAX_ERASED_LEVELS + 1],
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
        assert!(
            ROOT::LEVEL <= MAX_ERASED_LEVELS,
            "hierarchy deeper than the erased walks' inline record arrays; raise MAX_ERASED_LEVELS"
        );
        let mut records = [(LevelInfo::EMPTY, Frame::EMPTY); MAX_ERASED_LEVELS + 1];
        for level in 0..=ROOT::LEVEL {
            records[level].0 = LevelInfo::new(&ROOT::META[level]);
        }
        // The root opens like any other node — for a hierarchy whose root is
        // itself a leaf, the walk simply starts (and ends) at level 0,
        // scanning the root's occupancy words. Its record is the last *used*
        // entry (the array's tail past it is filler).
        let node = root as *const ROOT as *const u8;
        let (info, frame) = &mut records[ROOT::LEVEL];
        *frame = Frame {
            node,
            word: unsafe { mask_word(node, info, 0) },
            word_idx: 0,
            origin: UVec3::ZERO,
        };
        ErasedVoxelIter {
            pools,
            levels: records,
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
    /// Inline and fixed-size ([`MAX_ERASED_LEVELS`]), so creating the
    /// iterator allocates nothing; entries past `root_level` are unused
    /// filler.
    levels: [(LevelInfo, Frame); MAX_ERASED_LEVELS],
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
        assert!(
            ROOT::LEVEL <= MAX_ERASED_LEVELS,
            "hierarchy deeper than the erased walks' inline record arrays; raise MAX_ERASED_LEVELS"
        );
        let leaf_meta = &ROOT::META[0];
        let leaf_fanout = leaf_meta.fanout_log2;
        let leaf_extent = <ROOT::LeafType as Node>::EXTENT;

        let mut records = [(LevelInfo::EMPTY, Frame::EMPTY); MAX_ERASED_LEVELS];
        for level in 1..=ROOT::LEVEL {
            records[level - 1].0 = LevelInfo::new(&ROOT::META[level]);
        }
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
            // The root's record: the last *used* entry (the array's tail
            // past it is filler).
            let (info, frame) = &mut records[ROOT::LEVEL - 1];
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
            levels: records,
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

/// One level of a ray walk: an internal node whose child cells the ray is
/// stepping across, with the incremental state of the classic grid walk.
#[derive(Clone, Copy)]
struct RayFrame {
    /// Raw bytes of the node, read through the [`LevelInfo`] offsets.
    node: *const u8,
    /// Voxel coordinate of the node's minimum corner.
    origin: UVec3,
    /// The child cell examined next, in cells relative to the node (one unit
    /// = one child). Steps of the walk move it one cell along one axis, so
    /// it leaves `[0, fanout)` when the ray exits the node.
    cell: IVec3,
    /// Ray parameter at which the ray enters `cell`.
    t_enter: f32,
    /// Per axis, the parameter at which the ray crosses out of `cell`
    /// through that axis's far slab; `+inf` on zero-direction axes. The
    /// smallest lane is the face the ray leaves `cell` through — the next
    /// cell to step to — and a step advances that lane by `t_delta`,
    /// replacing a per-cell slab test with one add.
    t_max: Vec3A,
    /// Per axis, the parameter width of one cell at this level:
    /// `child_extent * |1 / dir|`.
    t_delta: Vec3A,
}

impl RayFrame {
    const EMPTY: Self = Self {
        node: std::ptr::null(),
        origin: UVec3::ZERO,
        cell: IVec3::ZERO,
        t_enter: 0.0,
        t_max: Vec3A::ZERO,
        t_delta: Vec3A::ZERO,
    };
}


/// The ray parameter interval over which `origin + dir * t` lies inside the
/// box `[min, min + extent)`, already intersected with `[t0, t1]` — the
/// interval is empty (and the box missed) unless `t_in <= t_out`. This is
/// the walk's one full slab test, opening the tree's own box; inside the
/// walk, cells advance incrementally through [`RayFrame::t_max`] instead.
///
/// All three axes compute at once, branch-free. `inv_dir` is the per-axis
/// reciprocal of the direction, `±inf` on a zero axis. A product goes NaN
/// exactly when a zero-direction axis' origin sits on one of its slab planes
/// (`0 * inf`); such an axis constrains nothing, which the NaN select
/// encodes. (The select, not the `min`/`max`, must decide this: SSE and NEON
/// disagree on how min/max treat a NaN operand.)
#[inline(always)]
fn ray_box_t(origin: Vec3A, inv_dir: Vec3A, min: Vec3A, extent: Vec3A, t0: f32, t1: f32) -> (f32, f32) {
    let near = (min - origin) * inv_dir;
    let far = (min + extent - origin) * inv_dir;
    let nan = near.is_nan_mask() | far.is_nan_mask();
    let lo = Vec3A::select(nan, Vec3A::NEG_INFINITY, near.min(far));
    let hi = Vec3A::select(nan, Vec3A::INFINITY, near.max(far));
    (lo.max_element().max(t0), hi.min_element().min(t1))
}

/// The iterator returned by [`crate::TreeErased::iter_leaf_views_along_ray`]:
/// an [`ErasedLeafView`] of every leaf the ray pierces, front to back, each
/// with the ray parameter at which the ray enters the leaf's block.
///
/// The walk keeps one [`RayFrame`] per open level and steps it cell by cell
/// in the ray's order (the classic grid walk: leave the current cell through
/// whichever face the ray crosses first). A cell whose child-mask bit is
/// clear is stepped over in one comparison, whatever its size — at the top
/// level one step skips a whole subtree of empty space — and a set bit either
/// descends (internal child) or yields (leaf child).
pub struct ErasedLeafViewRayIter<'a> {
    pools: &'a [Pool],
    /// Record for tree level `k` at `levels[k - 1]`, `k = 1..=root_level`;
    /// leaves (level 0) are yielded, never opened, and need no record.
    /// Inline and fixed-size ([`MAX_ERASED_LEVELS`]) so creating the iterator
    /// allocates nothing; entries past `root_level` are unused filler.
    levels: [(LevelInfo, RayFrame); MAX_ERASED_LEVELS],
    /// Number of live frames; the deepest open level is
    /// `root_level + 1 - depth`.
    depth: u32,
    root_level: u32,
    /// The ray, in voxel coordinates, with the parameter range to cover.
    origin: Vec3A,
    dir: Vec3A,
    /// Per-axis reciprocal of `dir` (`±inf` on zero axes), for slab tests.
    inv_dir: Vec3A,
    /// Lane mask of the axes whose direction is nonzero.
    dir_finite: BVec3A,
    /// Lane mask of the axes whose direction is positive.
    dir_positive: BVec3A,
    /// Per-axis cell step: the sign of `dir`, `0` on zero axes.
    step: IVec3,
    t0: f32,
    t1: f32,
    /// Per level (indexed like `levels`): the reciprocal of the cell extent,
    /// so entering a node multiplies instead of divides. Cell extents are
    /// powers of two, so the reciprocal is exact.
    child_extent_recip: [Vec3A; MAX_ERASED_LEVELS],
    /// Per level (indexed like `levels`): [`RayFrame::t_delta`], which
    /// depends only on the level's cell extent and the ray.
    t_deltas: [Vec3A; MAX_ERASED_LEVELS],
    /// The leaf-level constants stamped onto every yielded view.
    leaf_extent: UVec3,
    leaf_extent_mask: UVec3,
    leaf_shift_x: u32,
    leaf_shift_y: u32,
    leaf_mask_offset: u32,
    leaf_mask_words: u32,
    /// A hierarchy whose root is itself a leaf yields it here, once.
    root_leaf: Option<(f32, ErasedLeafView<'a, ()>)>,
}

/// See [`ErasedVoxelIter`]'s impls: the raw pointers are borrows of live,
/// unconditionally-`Sync` nodes.
unsafe impl Send for ErasedLeafViewRayIter<'_> {}
unsafe impl Sync for ErasedLeafViewRayIter<'_> {}

impl<'a> ErasedLeafViewRayIter<'a> {
    /// Walk the leaves of the tree rooted at `root`, whose non-root nodes
    /// live in `pools` (pools[k] holding level-k nodes, exactly as in
    /// [`crate::Tree`]), yielding views of the leaves the ray pierces within
    /// the parameter range `[t0, t1]`, in ray order.
    pub(crate) fn new<ROOT: Node>(
        root: &'a ROOT,
        pools: &'a [Pool],
        origin: Vec3,
        dir: Vec3,
        t0: f32,
        t1: f32,
    ) -> Self
    where
        [(); ROOT::LEVEL + 1]: Sized,
    {
        assert!(
            ROOT::LEVEL <= MAX_ERASED_LEVELS,
            "hierarchy deeper than the erased walks' inline record arrays; raise MAX_ERASED_LEVELS"
        );
        let leaf_meta = &ROOT::META[0];
        let leaf_fanout = leaf_meta.fanout_log2;
        let leaf_extent = <ROOT::LeafType as Node>::EXTENT;
        let origin = Vec3A::from(origin);
        let dir = Vec3A::from(dir);
        let inv_dir = dir.recip();
        let dir_finite = inv_dir.abs().cmplt(Vec3A::INFINITY);
        let dir_positive = dir.cmpgt(Vec3A::ZERO);
        let step = IVec3::new(
            (dir.x > 0.0) as i32 - (dir.x < 0.0) as i32,
            (dir.y > 0.0) as i32 - (dir.y < 0.0) as i32,
            (dir.z > 0.0) as i32 - (dir.z < 0.0) as i32,
        );

        let mut records = [(LevelInfo::EMPTY, RayFrame::EMPTY); MAX_ERASED_LEVELS];
        let mut child_extent_recip = [Vec3A::ONE; MAX_ERASED_LEVELS];
        let mut t_deltas = [Vec3A::ZERO; MAX_ERASED_LEVELS];
        for level in 1..=ROOT::LEVEL {
            let info = LevelInfo::new(&ROOT::META[level]);
            let cell_extent = info.child_extent.as_vec3a();
            records[level - 1].0 = info;
            child_extent_recip[level - 1] = cell_extent.recip();
            t_deltas[level - 1] = cell_extent * inv_dir.abs();
        }
        let mut iter = ErasedLeafViewRayIter {
            pools,
            levels: records,
            depth: 0,
            root_level: ROOT::LEVEL as u32,
            origin,
            dir,
            inv_dir,
            dir_finite,
            dir_positive,
            step,
            t0,
            t1,
            child_extent_recip,
            t_deltas,
            leaf_extent,
            leaf_extent_mask: <ROOT::LeafType as Node>::EXTENT_MASK,
            leaf_shift_x: leaf_fanout.y + leaf_fanout.z,
            leaf_shift_y: leaf_fanout.z,
            leaf_mask_offset: leaf_meta.mask_offset,
            leaf_mask_words: leaf_meta.mask_words,
            root_leaf: None,
        };
        // Open the root over the interval the ray spends inside the tree's
        // box; a ray that misses it leaves the walk empty.
        let root_node = root as *const ROOT as *const u8;
        let (t_in, t_out) = ray_box_t(
            origin,
            inv_dir,
            Vec3A::ZERO,
            ROOT::EXTENT.as_vec3a(),
            t0,
            t1,
        );
        if t_in <= t_out {
            if ROOT::LEVEL == 0 {
                // The root is itself a leaf.
                iter.root_leaf = Some((
                    t_in,
                    ErasedLeafView {
                        origin: UVec3::ZERO,
                        index: u32::MAX,
                        attribute_ptr: (),
                        extent: leaf_extent,
                        extent_mask: <ROOT::LeafType as Node>::EXTENT_MASK,
                        shift_x: leaf_fanout.y + leaf_fanout.z,
                        shift_y: leaf_fanout.z,
                        mask_offset: leaf_meta.mask_offset,
                        mask_words: leaf_meta.mask_words,
                        node: root_node,
                        _borrow: std::marker::PhantomData,
                    },
                ));
            } else {
                // The root's record: the last *used* entry, at
                // `ROOT::LEVEL - 1` (the array's tail past it is filler).
                let info = iter.levels[ROOT::LEVEL - 1].0;
                iter.levels[ROOT::LEVEL - 1].1 =
                    iter.enter_node(root_node, UVec3::ZERO, &info, ROOT::LEVEL - 1, t_in);
                iter.depth = 1;
            }
        }
        iter
    }

    /// The state of a node's frame at the moment the ray enters it: the
    /// entry cell — the cell containing the ray point at `t_enter`, clamped
    /// into the node's grid, the boundary-precision guard a grid walk needs
    /// since that point typically lies exactly on the node's face — and the
    /// per-axis crossing parameters ([`RayFrame::t_max`], [`RayFrame::t_delta`])
    /// the walk advances incrementally from there. `record` is the node's
    /// index into the per-level arrays (`levels[record].0` is `info`).
    fn enter_node(
        &self,
        node: *const u8,
        node_origin: UVec3,
        info: &LevelInfo,
        record: usize,
        t_enter: f32,
    ) -> RayFrame {
        let cell_extent = info.child_extent.as_vec3a();
        let origin_f = node_origin.as_vec3a();
        let p = self.origin + self.dir * t_enter;
        let cell = ((p - origin_f) * self.child_extent_recip[record])
            .floor()
            .as_ivec3()
            .clamp(IVec3::ZERO, info.fanout.as_ivec3() - IVec3::ONE);
        // The entry cell's far boundary on each axis: the high face on axes
        // pointing positive, the low face otherwise.
        let offset = Vec3A::select(self.dir_positive, Vec3A::ONE, Vec3A::ZERO);
        let boundary = origin_f + (cell.as_vec3a() + offset) * cell_extent;
        let t_max = Vec3A::select(
            self.dir_finite,
            (boundary - self.origin) * self.inv_dir,
            Vec3A::INFINITY,
        );
        RayFrame {
            node,
            origin: node_origin,
            cell,
            t_enter,
            t_max,
            t_delta: self.t_deltas[record],
        }
    }
}

impl<'a> Iterator for ErasedLeafViewRayIter<'a> {
    type Item = (f32, ErasedLeafView<'a, ()>);

    fn next(&mut self) -> Option<(f32, ErasedLeafView<'a, ()>)> {
        if let Some(leaf) = self.root_leaf.take() {
            return Some(leaf);
        }
        while self.depth > 0 {
            let level = (self.root_level + 1 - self.depth) as usize;
            let (info, frame) = &mut self.levels[level - 1];
            let info = *info;

            let cell = frame.cell;
            // Negative coordinates wrap to huge values, so one unsigned
            // compare covers both sides of the grid.
            if !cell.as_uvec3().cmplt(info.fanout).all() {
                // The ray has left this node.
                self.depth -= 1;
                continue;
            }
            if frame.t_enter > self.t1 {
                // This cell — and, entry parameters being increasing, every
                // later cell of this node — starts beyond the covered range.
                self.depth -= 1;
                continue;
            }

            // Capture the cell, then advance the walk past it, so it
            // resumes correctly after a yield. The cell's exit parameter is
            // the nearest of the per-axis crossings, and one add keeps the
            // stepped axis's crossing current — no per-cell slab test.
            let node = frame.node;
            let node_origin = frame.origin;
            let t_in = frame.t_enter.max(self.t0);
            // The face the ray leaves the cell through: a scalar three-way
            // minimum, compiling to two compare/selects — where
            // `Vec3A::min_element` plus a lane-mask argmin compiles to a
            // serial chain several times as long. The indexed updates below
            // stay branch-free: the frame lives in the `levels` array, so a
            // lane is one addressed load/add/store.
            let mut axis = 0;
            let mut t_next = frame.t_max.x;
            if frame.t_max.y < t_next {
                axis = 1;
                t_next = frame.t_max.y;
            }
            if frame.t_max.z < t_next {
                axis = 2;
                t_next = frame.t_max.z;
            }
            if !t_next.is_finite() {
                // No axis makes forward progress (the ray runs parallel to
                // every remaining slab): nothing further in this node.
                self.depth -= 1;
                continue;
            }
            frame.cell[axis] += self.step[axis];
            frame.t_max[axis] += frame.t_delta[axis];
            frame.t_enter = t_next;

            // Enter the captured cell only when the ray actually passes
            // through it — its exit `t_next` not preceding its entry `t_in`;
            // the entry cell of a node is clamped, so the first examined
            // cell can sit one off the ray's true path — and its child-mask
            // bit says it is occupied.
            if t_in > t_next {
                continue;
            }
            let index = ((cell.x as u32) << info.shift_x)
                | ((cell.y as u32) << info.shift_y)
                | cell.z as u32;
            let word = unsafe { mask_word(node, &info, index / usize::BITS) };
            if word & (1 << (index % usize::BITS)) == 0 {
                continue;
            }
            let cell_min = node_origin + cell.as_uvec3() * info.child_extent;

            // A set mask bit guarantees the entry holds an occupied pointer.
            let child_ptr = unsafe {
                (*(node.add(info.child_ptrs_offset as usize) as *const InternalNodeEntry)
                    .add(index as usize))
                .occupied
            };

            if level == 1 {
                let node = unsafe { self.pools[0].get(child_ptr) };
                return Some((
                    t_in,
                    ErasedLeafView {
                        origin: cell_min,
                        index: child_ptr,
                        attribute_ptr: (),
                        extent: self.leaf_extent,
                        extent_mask: self.leaf_extent_mask,
                        shift_x: self.leaf_shift_x,
                        shift_y: self.leaf_shift_y,
                        mask_offset: self.leaf_mask_offset,
                        mask_words: self.leaf_mask_words,
                        node,
                        _borrow: std::marker::PhantomData,
                    },
                ));
            }
            let child_level = level - 1;
            let child_node = unsafe { self.pools[child_level].get(child_ptr) };
            let child_info = self.levels[child_level - 1].0;

            // Before opening the child, reject content the ray merely passes
            // over. Inside the child, the ray's segment runs from its point
            // at `t_in` to its point at the captured cell's exit — so the
            // child cells it can touch span a known range on every axis (the
            // endpoints' cells, widened by one against rounding). For a node
            // of 8×8×8 children the child mask keeps one x-slice per word,
            // with z in each byte and y selecting the byte, so "is any bit
            // of that range set" is a handful of word ORs against one bit
            // pattern. When none is, the ray cannot reach any child — think
            // of a ray crossing high above floor content that sits in the
            // same node — and the whole node is skipped without stepping
            // through it.
            if child_info.shift_x == 6 && child_info.shift_y == 3 {
                let recip = self.child_extent_recip[child_level - 1];
                let cell_min_f = cell_min.as_vec3a();
                let p_in = (self.origin + self.dir * t_in - cell_min_f) * recip;
                let p_out =
                    (self.origin + self.dir * t_next.min(self.t1) - cell_min_f) * recip;
                let widest = child_info.fanout.as_ivec3() - IVec3::ONE;
                let lo = (p_in.min(p_out).floor().as_ivec3() - IVec3::ONE)
                    .clamp(IVec3::ZERO, widest);
                let hi = (p_in.max(p_out).floor().as_ivec3() + IVec3::ONE)
                    .clamp(IVec3::ZERO, widest);
                // Bits z in lo.z..=hi.z, replicated into the bytes y in
                // lo.y..=hi.y (byte-aligned, so the multiply cannot carry).
                let z_bits = (2u64 << hi.z) - (1u64 << lo.z);
                let byte_repl = 0x0101_0101_0101_0101u64 >> (8 * (7 - (hi.y - lo.y)));
                let touched = (z_bits * byte_repl) << (8 * lo.y);
                let mut present = 0u64;
                for x in lo.x..=hi.x {
                    present |= unsafe { mask_word(child_node, &child_info, x as u32) } as u64;
                }
                if present & touched == 0 {
                    continue;
                }
            }

            // Descend into the child, entering at the parameter at which the
            // ray entered its cell.
            self.levels[child_level - 1].1 =
                self.enter_node(child_node, cell_min, &child_info, child_level - 1, t_in);
            self.depth += 1;
        }
        None
    }
}

/// One block on the frontier of a near-point walk: a node or leaf that has
/// been discovered (its parent's mask bit was set) but not yet opened or
/// yielded, keyed by the distance from the query point to its block.
struct NearEntry {
    /// Squared distance from the query point to the entry's block, in the
    /// caller's metric ([`ErasedLeafViewNearIter`]'s `scale`).
    dist_sq: f32,
    /// Tree level of the node: level-0 entries are leaves, yielded rather
    /// than opened.
    level: u32,
    /// Pool index of a level-0 entry (`u32::MAX` for an out-of-pool root
    /// leaf), stamped onto its view as [`ErasedLeafView::leaf_index`].
    index: u32,
    /// Voxel coordinate of the block's minimum corner.
    origin: UVec3,
    /// Raw bytes of the node, resolved from the pools when the entry was
    /// pushed.
    node: *const u8,
}

/// Ordered by distance alone, *reversed*: [`BinaryHeap`] pops its greatest
/// entry, and the walk wants the nearest. Distances are nonnegative floats,
/// so their bit patterns order exactly like their values.
impl Ord for NearEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist_sq.to_bits().cmp(&self.dist_sq.to_bits())
    }
}

impl PartialOrd for NearEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for NearEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.to_bits() == other.dist_sq.to_bits()
    }
}

impl Eq for NearEntry {}

/// Squared distance from `point` to the block `[origin, origin + extent)`,
/// with both endpoints in voxel coordinates and the difference scaled
/// per-axis by `scale` — `0` when the point lies inside the block. Clamping
/// the point into the block finds the block's closest point, so this is
/// exact for the block's closure, and a lower bound for the distance to
/// anything stored inside it.
#[inline(always)]
fn block_dist_sq(point: Vec3A, scale: Vec3A, origin: UVec3, extent: UVec3) -> f32 {
    let lo = origin.as_vec3a();
    let hi = lo + extent.as_vec3a();
    ((point.clamp(lo, hi) - point) * scale).length_squared()
}

/// Squared distance from `point` to the *farthest* corner of the block
/// `[origin, origin + extent)`, measured like [`block_dist_sq`]. A block on
/// the frontier holds at least one voxel somewhere in its extent (a clear
/// mask bit never enters the frontier), so this is an upper bound on the
/// distance from `point` to the nearest voxel — the bound the walk prunes
/// against.
#[inline(always)]
fn block_far_dist_sq(point: Vec3A, scale: Vec3A, origin: UVec3, extent: UVec3) -> f32 {
    let lo = origin.as_vec3a();
    let hi = lo + extent.as_vec3a();
    ((point - lo).abs().max((hi - point).abs()) * scale).length_squared()
}

/// Per-axis squared-distance tables for the child grid of one opened node.
///
/// The metric is axis-aligned, so a cell's squared distance to the query
/// point decomposes into three independent per-axis contributions:
/// `near[0][x] + near[1][y] + near[2][z]` equals [`block_dist_sq`] of the
/// cell at `(x, y, z)` — computed with the same operations in the same
/// order, so bit-identically — and the `far` tables likewise sum to
/// [`block_far_dist_sq`]. One table fill (`fanout` entries per axis)
/// replaces a clamp/multiply/square pipeline per cell with two adds, and
/// the per-axis minima give O(1) lower bounds over every cell sharing an x
/// (a mask word) or an (x, y) pair (a byte of it).
struct AxisTables {
    near: [[f32; 8]; 3],
    far: [[f32; 8]; 3],
    /// Minima of the used `near`/`far` entries of the y and z axes.
    min_near_y: f32,
    min_near_z: f32,
    min_far_y: f32,
    min_far_z: f32,
}

/// Fills [`AxisTables`] for cells `[origin + k * cell_extent,
/// origin + (k + 1) * cell_extent)`, `k < fanout` per axis. `fanout` must
/// not exceed 8 on any axis.
fn axis_tables(
    point: Vec3A,
    scale: Vec3A,
    origin: UVec3,
    cell_extent: UVec3,
    fanout: UVec3,
) -> AxisTables {
    let mut tables = AxisTables {
        near: [[0.0; 8]; 3],
        far: [[0.0; 8]; 3],
        min_near_y: f32::INFINITY,
        min_near_z: f32::INFINITY,
        min_far_y: f32::INFINITY,
        min_far_z: f32::INFINITY,
    };
    let point = point.to_array();
    let scale = scale.to_array();
    let origin = origin.to_array();
    let cell_extent = cell_extent.to_array();
    let fanout = fanout.to_array();
    for axis in 0..3 {
        let p = point[axis];
        let s = scale[axis];
        let e = cell_extent[axis] as f32;
        let mut lo = origin[axis] as f32;
        for k in 0..fanout[axis] as usize {
            let hi = lo + e;
            let near = (p.clamp(lo, hi) - p) * s;
            let far = (p - lo).abs().max((hi - p).abs()) * s;
            tables.near[axis][k] = near * near;
            tables.far[axis][k] = far * far;
            lo = hi;
        }
    }
    for k in 0..fanout[1] as usize {
        tables.min_near_y = tables.min_near_y.min(tables.near[1][k]);
        tables.min_far_y = tables.min_far_y.min(tables.far[1][k]);
    }
    for k in 0..fanout[2] as usize {
        tables.min_near_z = tables.min_near_z.min(tables.near[2][k]);
        tables.min_far_z = tables.min_far_z.min(tables.far[2][k]);
    }
    tables
}

/// The iterator returned by [`crate::TreeErased::iter_leaf_views_near_point`]:
/// an [`ErasedLeafView`] of each leaf that could hold the query point's
/// nearest voxel, in nondecreasing distance from the point to the leaf's
/// block, each with that squared distance.
///
/// The walk is best-first — the tree playing the role a BVH plays for a
/// nearest-neighbor query. The frontier holds the discovered, unopened
/// blocks, nearest first. Popping an internal block pushes its occupied
/// children (a clear mask bit never enters the frontier, so empty subtrees
/// cost nothing); popping a leaf block yields it. A child's block lies
/// inside its parent's, so its distance is no smaller, which is what makes
/// the pops — and therefore the yields — nondecreasing.
///
/// Discovery doubles as pruning: a discovered block holds at least one
/// voxel, so its farthest corner bounds the nearest-voxel distance from
/// above ([`ErasedLeafViewNearIter::upper`]), and blocks whose nearest
/// corner lies beyond the tightest such bound are dropped rather than
/// pushed — without this, one opened node of a wide hierarchy (up to 8³
/// children) would flood the frontier with blocks the consumer's own
/// stopping rule discards unread.
pub struct ErasedLeafViewNearIter<'a> {
    pools: &'a [Pool],
    /// Walk constants for level `k` at `levels[k - 1]`, `k = 1..=root_level`;
    /// leaves (level 0) are yielded, never opened, and need no record.
    /// Inline and fixed-size ([`MAX_ERASED_LEVELS`]); entries past
    /// `root_level` are unused filler.
    levels: [LevelInfo; MAX_ERASED_LEVELS],
    /// The query point, in voxel coordinates.
    point: Vec3A,
    /// Per-axis factor from voxel-coordinate differences to the caller's
    /// metric.
    scale: Vec3A,
    /// Every discovered, unopened block, nearest first.
    frontier: BinaryHeap<NearEntry>,
    /// The tightest proven upper bound (squared) on the distance from the
    /// point to its nearest voxel: the smallest [`block_far_dist_sq`] of any
    /// block discovered so far. A block whose *nearest* corner lies beyond
    /// it cannot hold the nearest voxel — nor tie for it — and is pruned
    /// instead of pushed or opened.
    upper: f32,
    /// The leaf-level constants stamped onto every yielded view.
    leaf_extent: UVec3,
    leaf_extent_mask: UVec3,
    leaf_shift_x: u32,
    leaf_shift_y: u32,
    leaf_mask_offset: u32,
    leaf_mask_words: u32,
}

/// See [`ErasedVoxelIter`]'s impls: the raw pointers are borrows of live,
/// unconditionally-`Sync` nodes.
unsafe impl Send for ErasedLeafViewNearIter<'_> {}
unsafe impl Sync for ErasedLeafViewNearIter<'_> {}

impl<'a> ErasedLeafViewNearIter<'a> {
    /// Walk the leaves of the tree rooted at `root`, whose non-root nodes
    /// live in `pools` (pools[k] holding level-k nodes, exactly as in
    /// [`crate::Tree`]), nearest to `point` first.
    pub(crate) fn new<ROOT: Node>(
        root: &'a ROOT,
        pools: &'a [Pool],
        point: Vec3,
        scale: Vec3,
    ) -> Self
    where
        [(); ROOT::LEVEL + 1]: Sized,
    {
        assert!(
            ROOT::LEVEL <= MAX_ERASED_LEVELS,
            "hierarchy deeper than the erased walks' inline record arrays; raise MAX_ERASED_LEVELS"
        );
        let leaf_meta = &ROOT::META[0];
        let leaf_fanout = leaf_meta.fanout_log2;
        let mut levels = [LevelInfo::EMPTY; MAX_ERASED_LEVELS];
        for level in 1..=ROOT::LEVEL {
            levels[level - 1] = LevelInfo::new(&ROOT::META[level]);
        }
        let point = Vec3A::from(point);
        let scale = Vec3A::from(scale);

        // Sized for one opened node of a wide hierarchy without regrowth;
        // pathological occupancy grows it like any `Vec`.
        let mut frontier = BinaryHeap::with_capacity(128);
        frontier.push(NearEntry {
            dist_sq: block_dist_sq(point, scale, UVec3::ZERO, ROOT::EXTENT),
            level: ROOT::LEVEL as u32,
            index: u32::MAX,
            origin: UVec3::ZERO,
            node: root as *const ROOT as *const u8,
        });

        ErasedLeafViewNearIter {
            pools,
            levels,
            point,
            scale,
            frontier,
            upper: block_far_dist_sq(point, scale, UVec3::ZERO, ROOT::EXTENT),
            leaf_extent: <ROOT::LeafType as Node>::EXTENT,
            leaf_extent_mask: <ROOT::LeafType as Node>::EXTENT_MASK,
            leaf_shift_x: leaf_fanout.y + leaf_fanout.z,
            leaf_shift_y: leaf_fanout.z,
            leaf_mask_offset: leaf_meta.mask_offset,
            leaf_mask_words: leaf_meta.mask_words,
        }
    }
}

impl<'a> ErasedLeafViewNearIter<'a> {
    /// First half of opening an internal block: tighten
    /// [`ErasedLeafViewNearIter::upper`] with the farthest-corner bound of
    /// every occupied child — each holds at least one voxel somewhere in
    /// its cell.
    ///
    /// All distances come from one [`AxisTables`] fill: a cell is two adds,
    /// and the per-axis minima bound a whole mask word (one x) or one byte
    /// of it (one x, y pair) in O(1), so words and rows that cannot improve
    /// the bound are skipped without touching their bits. The word holding
    /// the point's nearest cell goes first, so the bound is already tight
    /// when the rest test against it. The word-level structure exists on
    /// the 8×8×8 node layout (the ray walk's fast path exploits the same
    /// one); other layouts take the per-bit path.
    fn tighten_upper(&mut self, entry: &NearEntry, info: &LevelInfo, tables: &AxisTables) {
        // Local copies of everything the loops read: writes through the
        // frontier's buffer (a separate allocation) stop the compiler from
        // keeping `self`'s fields in registers otherwise, and `self.upper`
        // would round-trip through memory on every child.
        let point = self.point;
        let mut upper = self.upper;
        let cell_extent = info.child_extent;
        let word_is_x_slice = info.shift_x == 6 && info.shift_y == 3;
        let nearest_cell = ((point - entry.origin.as_vec3a()) / cell_extent.as_vec3a())
            .floor()
            .as_ivec3()
            .clamp(IVec3::ZERO, info.fanout.as_ivec3() - IVec3::ONE)
            .as_uvec3();
        for i in 0..info.mask_words {
            let word_idx = if word_is_x_slice {
                // The permutation `nearest_cell.x, 0, 1, ..` (skipping
                // `nearest_cell.x` where it would repeat).
                if i == 0 {
                    nearest_cell.x
                } else if i <= nearest_cell.x {
                    i - 1
                } else {
                    i
                }
            } else {
                i
            };
            let word = unsafe { mask_word(entry.node, info, word_idx) };
            if word == 0 {
                continue;
            }
            if word_is_x_slice {
                let x = word_idx as usize;
                let word_bound = tables.far[0][x] + tables.min_far_y + tables.min_far_z;
                if word_bound >= upper {
                    continue;
                }
                if word == usize::MAX {
                    // Fully occupied: the bound is attained by some cell.
                    upper = word_bound;
                    continue;
                }
                // One byte per (x, y) row of 8 z cells.
                for y in 0..8usize {
                    let row = (word >> (8 * y)) & 0xff;
                    if row == 0
                        || tables.far[0][x] + tables.far[1][y] + tables.min_far_z >= upper
                    {
                        continue;
                    }
                    let mut row = row;
                    while row != 0 {
                        let z = row.trailing_zeros() as usize;
                        row &= row - 1;
                        upper =
                            upper.min(tables.far[0][x] + tables.far[1][y] + tables.far[2][z]);
                    }
                }
            } else {
                let mut word = word;
                while word != 0 {
                    let bit = word.trailing_zeros();
                    word &= word - 1;
                    let index = word_idx * usize::BITS + bit;
                    let x = (index >> info.shift_x) as usize;
                    let y = ((index >> info.shift_y) & info.mask_y) as usize;
                    let z = (index & info.mask_z) as usize;
                    upper = upper.min(tables.far[0][x] + tables.far[1][y] + tables.far[2][z]);
                }
            }
        }
        self.upper = upper;
    }

    /// Second half of opening an internal block: a frontier entry for every
    /// child that can still hold — or tie for — the nearest voxel. Same
    /// [`AxisTables`] scheme as [`Self::tighten_upper`], on the `near`
    /// tables: whole words and rows beyond the bound push nothing, and a
    /// surviving cell's distance is two adds. Against a wide fan-out (an 8³
    /// node has up to 512 occupied children), this is what keeps the
    /// frontier, and the heap traffic, small.
    fn push_children(
        &mut self,
        entry: &NearEntry,
        info: &LevelInfo,
        tables: &AxisTables,
        child_level: u32,
    ) {
        // Split `self` so the frontier alone is borrowed mutably: the pushes
        // below write through the frontier's buffer, and without the split
        // the compiler must reload the bound and the pool from `self` after
        // every push.
        let upper = self.upper;
        let pool = &self.pools[child_level as usize];
        let frontier = &mut self.frontier;
        let cell_extent = info.child_extent;
        let word_is_x_slice = info.shift_x == 6 && info.shift_y == 3;
        let mut push = |dist_sq: f32, cell: UVec3, index: u32| {
            let origin = entry.origin + cell * cell_extent;
            // A set mask bit guarantees the entry holds an occupied pointer.
            let child_ptr = unsafe {
                (*(entry.node.add(info.child_ptrs_offset as usize) as *const InternalNodeEntry)
                    .add(index as usize))
                .occupied
            };
            let node = unsafe { pool.get(child_ptr) };
            frontier.push(NearEntry {
                dist_sq,
                level: child_level,
                index: child_ptr,
                origin,
                node,
            });
        };
        for word_idx in 0..info.mask_words {
            let word = unsafe { mask_word(entry.node, info, word_idx) };
            if word == 0 {
                continue;
            }
            if word_is_x_slice {
                let x = word_idx as usize;
                if tables.near[0][x] + tables.min_near_y + tables.min_near_z > upper {
                    continue;
                }
                for y in 0..8usize {
                    let row = (word >> (8 * y)) & 0xff;
                    if row == 0
                        || tables.near[0][x] + tables.near[1][y] + tables.min_near_z > upper
                    {
                        continue;
                    }
                    let mut row = row;
                    while row != 0 {
                        let z = row.trailing_zeros() as usize;
                        row &= row - 1;
                        let dist_sq =
                            tables.near[0][x] + tables.near[1][y] + tables.near[2][z];
                        if dist_sq > upper {
                            continue;
                        }
                        let cell = UVec3::new(x as u32, y as u32, z as u32);
                        push(dist_sq, cell, (x << 6 | y << 3 | z) as u32);
                    }
                }
            } else {
                let mut word = word;
                while word != 0 {
                    let bit = word.trailing_zeros();
                    word &= word - 1;
                    let index = word_idx * usize::BITS + bit;
                    let cell = UVec3 {
                        x: index >> info.shift_x,
                        y: (index >> info.shift_y) & info.mask_y,
                        z: index & info.mask_z,
                    };
                    let dist_sq = tables.near[0][cell.x as usize]
                        + tables.near[1][cell.y as usize]
                        + tables.near[2][cell.z as usize];
                    if dist_sq > upper {
                        continue;
                    }
                    push(dist_sq, cell, index);
                }
            }
        }
    }
}

impl<'a> Iterator for ErasedLeafViewNearIter<'a> {
    type Item = (f32, ErasedLeafView<'a, ()>);

    fn next(&mut self) -> Option<(f32, ErasedLeafView<'a, ()>)> {
        while let Some(entry) = self.frontier.pop() {
            if entry.dist_sq > self.upper {
                // `upper` tightened after this block was pushed: some other
                // block's entire extent is nearer than this block's nearest
                // corner, so nothing in it can be the nearest voxel.
                continue;
            }
            if entry.level == 0 {
                return Some((
                    entry.dist_sq,
                    ErasedLeafView {
                        origin: entry.origin,
                        index: entry.index,
                        attribute_ptr: (),
                        extent: self.leaf_extent,
                        extent_mask: self.leaf_extent_mask,
                        shift_x: self.leaf_shift_x,
                        shift_y: self.leaf_shift_y,
                        mask_offset: self.leaf_mask_offset,
                        mask_words: self.leaf_mask_words,
                        node: entry.node,
                        _borrow: std::marker::PhantomData,
                    },
                ));
            }
            let info = self.levels[entry.level as usize - 1];
            // One table fill serves both halves of the open.
            let tables =
                axis_tables(self.point, self.scale, entry.origin, info.child_extent, info.fanout);
            self.tighten_upper(&entry, &info, &tables);
            self.push_children(&entry, &info, &tables, entry.level - 1);
        }
        None
    }
}
