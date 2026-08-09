use std::mem::MaybeUninit;

use glam::UVec3;

use crate::{
    AabbU32, IsLeaf, Node, pool::{Pool, PoolStorage},
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
    fn count_leaves(&self) -> usize {
        self.root.count_leaves(&self.pool)
    }

    fn aabb(&self) -> AabbU32 {
        self.aabb
    }

    type LeafType = ROOT::LeafType;

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
    fn count_leaves(&self) -> usize {
        self.root.count_leaves(&self.pool)
    }

    fn aabb(&self) -> AabbU32 {
        self.aabb
    }

    type LeafType = ROOT::LeafType;

    type Iterator<'a> = ROOT::LeafIterator<'a>;

    fn iter_leaf(&self) -> Self::Iterator<'_> {
        self.root
            .iter_leaf(&self.pool, UVec3::ZERO)
    }
}

/// Trait abstracting over all [`Tree`] and [`TreeSnapshot`]
pub trait TreeLike: Send + Sync + 'static {
    fn count_leaves(&self) -> usize;
    fn aabb(&self) -> AabbU32;

    type LeafType: IsLeaf;
    type Iterator<'a>: Iterator<Item = (UVec3, &'a Self::LeafType)> where Self: 'a;
    fn iter_leaf(&self) -> Self::Iterator<'_>;

    fn iter(&self) -> impl Iterator<Item = UVec3> {
        self.iter_leaf()
            .flat_map(|(position, leaf)| leaf.iter(position))
    }
}
