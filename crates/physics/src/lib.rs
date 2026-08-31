//! Collision-detection support for `dust_vdb` voxel trees, built on parry.
//!
//! # Architecture
//!
//! The voxel tree is owned (and freely mutated, via copy-on-write) by its
//! [`VoxGeometry`]-style owner. Physics never touches the live tree: it reads immutable
//! [`TreeSnapshot`]s published by the owner, one per frame, through `dust_vdb`'s
//! type-erased read interface ([`TreeErased`]) — so nothing here is generic over the
//! tree hierarchy.
//!
//! - [`VdbVoxels`] is a borrowed view of any tree version implementing
//!   [`parry3d::shape::VoxelQuery`], the storage abstraction parry's voxel
//!   collision-detection algorithms (contact manifolds, intersection tests, shape-casting,
//!   ray-casting, point projection, mass properties) are generic over. Parry's
//!   neighborhood-aware voxel states are derived on the fly from the tree's occupancy
//!   bits — leaves come whole through [`TreeErased::iter_leaf_views_in_range`],
//!   everything per-voxel is bit math on their words — so the tree remains the single
//!   source of truth.
//! - [`VdbShape`] is a `'static`, [`Shape`]-implementing immutable value over one
//!   `Arc<dyn TreeWithValues<u32>>` (in practice a [`TreeSnapshot`] whose leaf value
//!   projects a `u32` attribute pointer), suitable for
//!   `SharedShape(Arc<dyn Shape>)`. Shapes built from different `hierarchy!`
//!   instantiations are one type — recognized by one [`VdbDispatcher`] and able to
//!   collide with each other. Updating a collider after an edit means building a
//!   `VdbShape` from a fresh snapshot and assigning it through the engine's normal
//!   component-mutation path — engine change detection then reacts like it would to any
//!   other shape change.
//! - Replaced snapshots need no bookkeeping here: when the last `Arc` reference goes
//!   away, the snapshot returns its root to its tree, and the tree reclaims the
//!   copy-on-write nodes on its next mutation session
//!   ([`Tree::reclaim_dropped_snapshots`]).
//!
//! # Per-edit flow
//!
//! ```text
//! Tree (owner, mutable, COW)
//!   ├── edits (Accessor::set / clear)  ◄── reclaims dropped snapshots on session start
//!   └── tree.snapshot() ──► VdbShape::new ──► Collider::set_shape ──► old shape
//!                                    │                                    │
//!                                    ├── physics queries (VoxelQuery)     ▼
//!                                    ├── GPU frame (retained until fence)  last Arc drop
//!                                    └── undo history (Tree::restore)     returns to tree
//! ```
//!
//! [`VoxGeometry`]: https://github.com/dust-engine/dust
//! [`TreeSnapshot`]: dust_vdb::TreeSnapshot
//! [`TreeErased`]: dust_vdb::TreeErased
//! [`TreeErased::iter_leaf_views_in_range`]: dust_vdb::TreeErased::iter_leaf_views_in_range
//! [`TreeWithValues<u32>`]: dust_vdb::TreeWithValues
//! [`Tree::reclaim_dropped_snapshots`]: dust_vdb::Tree::reclaim_dropped_snapshots
//! [`Tree::restore`]: dust_vdb::Tree::restore

// `hierarchy!` in the tests expands to const-generic expressions.
#![cfg_attr(test, feature(generic_const_exprs))]
#![cfg_attr(test, allow(incomplete_features))]

mod dispatcher;

pub use dispatcher::VdbDispatcher;

use std::sync::Arc;

use dust_vdb::{AabbU32, ErasedLeafView, TreeWithValues};
use glam::UVec3;
use parry3d::bounding_volume::{Aabb, BoundingSphere};
use parry3d::mass_properties::MassProperties;
use parry3d::math::{IVector, Real, Vector};
use parry3d::query::details::{cast_local_ray_on_voxels, project_local_point_on_voxels};
use parry3d::query::{PointProjection, PointQuery, Ray, RayCast, RayIntersection};
use parry3d::shape::{
    AxisMask, FeatureId, QueriedVoxel, Shape, ShapeType, TypedShape, VoxelQuery, VoxelState,
    VoxelType,
};

/// A `'static` voxel collision shape over one [`TreeWithValues<u32>`](TreeWithValues)
/// version (in practice an `Arc<TreeSnapshot>` whose leaf value projects a `u32`
/// attribute pointer), implementing parry's [`Shape`] so it can live inside a
/// `SharedShape(Arc<dyn Shape>)`.
///
/// `VdbShape` is a single concrete type for **every** tree hierarchy — that's
/// `dust_vdb`'s own erasure ([`TreeErased`]) at work — so colliders built from different
/// [`hierarchy!`](dust_vdb::hierarchy) instantiations coexist, and any two of them
/// collide against each other through the voxels-voxels algorithms — a [`VdbDispatcher`]
/// recognizes all of them with one downcast.
///
/// The shape is an immutable value pinning one snapshot; clones (including
/// [`Shape::clone_dyn`]) share that same snapshot. To update a collider after editing
/// the tree, build a `VdbShape` from a fresh snapshot and assign it through the engine's
/// normal mutation path (e.g. avian's `Collider::set_shape`) — the ECS exclusivity rules
/// guarantee no query observes the swap mid-flight, and the engine's change detection
/// picks it up like any other shape change. The replaced snapshot returns to its tree
/// once its last `Arc` reference is gone.
#[derive(Clone)]
pub struct VdbShape {
    tree: Arc<dyn TreeWithValues<u32>>,
    voxel_type_attributes: Arc<VdbVoxelTypeAttributes>,
    voxel_mask_attributes: Option<Arc<VdbVoxelMaskAttributes>>,
    voxel_size: Vector,
}

#[derive(Clone)]
/// Stores one [`VoxelType`](parry3d::shape::VoxelType) per voxel, in 2 bits.
///
/// # How a value is stored
///
/// A `VoxelType` (other than `Empty`, which is never stored) is encoded as
/// a 2-bit number: 0 = `Interior`, 1 = `Face`, 2 = `Edge`, 3 = `Vertex`.
/// The low bit of that number is kept in `bitmask1` and the high bit in
/// `bitmask2`, at the same bit position in both vectors. So one bit
/// position, taken across the two vectors together, holds one voxel's
/// value.
///
/// # Which bit position belongs to which voxel
///
/// Each leaf node owns the `leaf_size` consecutive bit positions starting
/// at `leaf * leaf_size`, where `leaf` is the leaf's pool index — the
/// `leaf` argument the accessor passes to every method. Storage is keyed by
/// that index alone, so nothing needs to be stored in the tree: `Ptr = ()`.
///
/// Within a leaf's positions, a voxel's value sits at
/// `leaf * leaf_size + inflated_offset`, where `inflated_offset` is the argument of that name the
/// accessor passes to every `get_attribute`/`set_attribute` call: the
/// voxel's index within its leaf, computed from its coordinates. That index
/// never changes, so changes to the leaf's attribute layout move nothing
/// here; only a copy-on-write fork copies a leaf's block of positions (see
/// `copy_attribute`).
pub struct VdbVoxelTypeAttributes {
    bitmask1: Vec<usize>,
    bitmask2: Vec<usize>,

    // number of voxels in a single leaf node
    leaf_size: u32,
}

impl VdbVoxelTypeAttributes {
    /// `leaf_size` is the number of voxels in a leaf node (64 for 4³ leaves).
    pub fn new(leaf_size: u32) -> Self {
        Self {
            bitmask1: Vec::new(),
            bitmask2: Vec::new(),
            leaf_size,
        }
    }

    /// Reads the 2-bit number stored for one voxel. `leaf` and `voxel` are
    /// the `leaf` and `voxel` arguments the accessor passes to
    /// `get_attribute`; the bit position read in `bitmask1` (low bit) and
    /// `bitmask2` (high bit) is `leaf * leaf_size + inflated_offset`.
    fn get_code(&self, leaf: u32, inflated_offset: u32) -> u8 {
        let bit_index = leaf as usize * self.leaf_size as usize + inflated_offset as usize;
        let word_index = bit_index / usize::BITS as usize;
        let shift = bit_index % usize::BITS as usize;
        let bit1 = (self.bitmask1[word_index] >> shift) & 1;
        let bit2 = (self.bitmask2[word_index] >> shift) & 1;
        (bit1 | (bit2 << 1)) as u8
    }

    /// Writes the 2-bit number for one voxel, growing the two bit vectors
    /// when the position lies past their current end. Same addressing as
    /// [`Self::get_code`].
    fn set_code(&mut self, leaf: u32, inflated_offset: u32, code: u8) {
        let bit_index = leaf as usize * self.leaf_size as usize + inflated_offset as usize;
        let word_index = bit_index / usize::BITS as usize;
        if word_index >= self.bitmask1.len() {
            self.bitmask1.resize(word_index + 1, 0);
        }
        if word_index >= self.bitmask2.len() {
            self.bitmask2.resize(word_index + 1, 0);
        }
        let shift = bit_index % usize::BITS as usize;
        self.bitmask1[word_index] =
            (self.bitmask1[word_index] & !(1 << shift)) | (((code & 1) as usize) << shift);
        self.bitmask2[word_index] =
            (self.bitmask2[word_index] & !(1 << shift)) | ((((code >> 1) & 1) as usize) << shift);
    }

    /// The [`VoxelType`](parry3d::shape::VoxelType) stored for one voxel;
    /// `leaf` and `inflated_offset` address it exactly as in
    /// [`Self::get_code`]. Only meaningful for an occupied voxel: an
    /// unwritten position decodes as `Interior`.
    fn voxel_type(&self, leaf: u32, inflated_offset: u32) -> parry3d::shape::VoxelType {
        match self.get_code(leaf, inflated_offset) {
            0 => parry3d::shape::VoxelType::Interior,
            1 => parry3d::shape::VoxelType::Face,
            2 => parry3d::shape::VoxelType::Edge,
            3 => parry3d::shape::VoxelType::Vertex,
            _ => unreachable!(),
        }
    }
}

impl dust_vdb::Attributes for VdbVoxelTypeAttributes {
    type Ptr = ();
    /// The type of the attribute values. For a MagicaVoxel grid, this would be a u8 palette index.
    type Value = parry3d::shape::VoxelType;
    fn get_attribute(&self, index: u32, _ptr: &Self::Ptr, _fitted_offset: u32, inflated_offset: u32) -> Self::Value {
        self.voxel_type(index, inflated_offset)
    }
    fn set_attribute(&mut self, index: u32, _ptr: &Self::Ptr, _fitted_offset: u32, inflated_offset: u32, value: Self::Value) {
        let code = match value {
            parry3d::shape::VoxelType::Interior => 0,
            parry3d::shape::VoxelType::Face => 1,
            parry3d::shape::VoxelType::Edge => 2,
            parry3d::shape::VoxelType::Vertex => 3,
            parry3d::shape::VoxelType::Empty => panic!("Cannot set attribute to Empty voxel type"),
        };
        self.set_code(index, inflated_offset, code);
    }
    fn free_attributes(&mut self, _index: u32, _ptr: &Self::Ptr, _num_attributes: u32) {
    }

    fn copy_attribute(
        &mut self,
        original_leaf: u32,
        new_leaf: u32,
        _ptr: &Self::Ptr,
        _original_mask: &[usize],
        _new_mask: &[usize],
        _coords: &UVec3,
    ) -> Self::Ptr {
        if original_leaf == new_leaf {
            return;
        }
        debug_assert!(
            self.leaf_size as usize % usize::BITS as usize == 0,
            "blocks are copied word-at-a-time"
        );
        let words_per_leaf = self.leaf_size as usize / usize::BITS as usize;
        let src = original_leaf as usize * words_per_leaf;
        let dst = new_leaf as usize * words_per_leaf;
        let end = dst + words_per_leaf;
        if self.bitmask1.len() < end {
            self.bitmask1.resize(end, 0);
        }
        if self.bitmask2.len() < end {
            self.bitmask2.resize(end, 0);
        }
        for word in 0..words_per_leaf {
            // The source block lies past the vectors' ends when nothing was
            // ever written to `original_leaf`; those positions read as 0.
            let low = self.bitmask1.get(src + word).copied().unwrap_or(0);
            let high = self.bitmask2.get(src + word).copied().unwrap_or(0);
            self.bitmask1[dst + word] = low;
            self.bitmask2[dst + word] = high;
        }
    }
}

/// Full parry [`VoxelState`]s, one byte per voxel, in ranges that *mirror
/// the material channel's*: this channel runs an
/// [`AttributeAllocator`](dust_vdb::AttributeAllocator) with the same
/// parameters as the material's, and the tuple fan-out feeds both channels
/// the identical event stream, so this allocator returns the identical
/// pointers. Its `Ptr` therefore *is* the material pointer — re-derived from
/// the leaf value by [`AttributePtr`](dust_vdb::AttributePtr) rather than
/// stored a second time.
pub struct VdbVoxelMaskAttributes {
    attribute_allocator: dust_vdb::AttributeAllocator,
    values: Vec<VoxelState>
}

impl VdbVoxelMaskAttributes {
    /// `alignment` and `max_allocation` must match the material channel's
    /// allocator parameters: the mirrored pointers depend on both allocators
    /// making identical decisions.
    pub fn new(alignment: u32, max_allocation: u32) -> Self {
        Self {
            attribute_allocator: dust_vdb::AttributeAllocator::new_with_capacity(
                alignment,
                max_allocation,
            ),
            values: Vec::new(),
        }
    }

    fn reserve(&mut self, size: u64) {
        if size as usize > self.values.len() {
            self.values
                .resize((size as usize).next_power_of_two(), VoxelState::EMPTY);
        }
    }
}

impl dust_vdb::Attributes for VdbVoxelMaskAttributes {
    type Ptr = u32;

    /// [`VoxelState`] directly. Its `Default` is [`VoxelState::EMPTY`] — never
    /// the state of an occupied voxel — so the accessor's write-default-erases
    /// rule reads as "writing the empty state erases the voxel", and the
    /// all-free state of an isolated voxel (raw bits 0) round-trips instead of
    /// being mistaken for an erase.
    type Value = VoxelState;

    fn get_attribute(
        &self,
        _leaf: u32,
        ptr: &Self::Ptr,
        fitted_offset: u32,
        _inflated_offset: u32,
    ) -> Self::Value {
        self.values[*ptr as usize + fitted_offset as usize]
    }

    fn set_attribute(
        &mut self,
        _leaf: u32,
        ptr: &Self::Ptr,
        fitted_offset: u32,
        _inflated_offset: u32,
        value: Self::Value,
    ) {
        self.values[*ptr as usize + fitted_offset as usize] = value;
    }

    fn free_attributes(&mut self, leaf: u32, ptr: &Self::Ptr, num_attributes: u32) {
        self.attribute_allocator
            .free(*ptr, num_attributes);
    }

    fn copy_attribute(
        &mut self,
        original_leaf: u32,
        new_leaf: u32,
        ptr: &Self::Ptr,
        original_mask: &[usize],
        new_mask: &[usize],
        _coords: &UVec3,
    ) -> Self::Ptr {
        let new_len = dust_vdb::mask_count_ones(new_mask);
        let new_ptr = self.attribute_allocator.allocate(new_len);
        self.reserve(new_ptr as u64 + new_len as u64);

        let mut new_ptr_cur = new_ptr;
        let mut old_ptr_cur = *ptr;

        let slice: &mut [Self::Value] = self.values.as_mut_slice();
        for (_, in_original, in_new) in dust_vdb::iter_mask_union(original_mask, new_mask) {
            if in_new && in_original {
                // copy it over
                slice[new_ptr_cur as usize] = slice[old_ptr_cur as usize];
            }
            if in_new {
                new_ptr_cur += 1;
            }
            if in_original {
                old_ptr_cur += 1;
            }
        }
        if original_leaf == new_leaf {
            // In-place re-home: the original range is dead (contract).
            let old_len = dust_vdb::mask_count_ones(original_mask);
            if old_len > 0 {
                self.attribute_allocator.free(*ptr, old_len);
            }
        }

        new_ptr
    }
}

impl RayCast for VdbShape {
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_time_of_impact: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        cast_local_ray_on_voxels(self, ray, max_time_of_impact, solid)
    }
}

impl PointQuery for VdbShape {
    fn project_local_point(&self, pt: Vector, solid: bool) -> PointProjection {
        project_local_point_on_voxels(self, pt, solid)
            .map(|(proj, _)| proj)
            .unwrap_or(PointProjection::new(false, Vector::splat(Real::MAX)))
    }

    fn project_local_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        project_local_point_on_voxels(self, pt, false)
            .map(|(proj, id)| (proj, FeatureId::Face(id)))
            .unwrap_or((
                PointProjection::new(false, Vector::splat(Real::MAX)),
                FeatureId::Unknown,
            ))
    }
}

impl Shape for VdbShape {
    fn compute_local_aabb(&self) -> Aabb {
        // The tree's bounds are voxel-tight, so the domain is exact (and `[ZERO, ZERO]`
        // when empty).
        self.local_aabb()
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        self.compute_local_aabb().bounding_sphere()
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }

    fn scale_dyn(&self, scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        Some(Box::new(Self {
            tree: self.tree.clone(),
            voxel_size: self.voxel_size * scale,
            voxel_mask_attributes: self.voxel_mask_attributes.clone(),
            voxel_type_attributes: self.voxel_type_attributes.clone(),
        }))
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        MassProperties::from_voxels(density, self)
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        self.voxel_size.min_element()
    }

    fn ccd_angular_thickness(&self) -> Real {
        core::f32::consts::FRAC_PI_2
    }
}

fn uvec_to_ivec(v: UVec3) -> IVector {
    IVector::new(v.x as i32, v.y as i32, v.z as i32)
}

/// Callers must guarantee `v` is component-wise non-negative.
fn ivec_to_uvec(v: IVector) -> UVec3 {
    UVec3::new(v.x as u32, v.y as u32, v.z as u32)
}

impl VdbShape {
    /// Creates a shape reading `tree` — any [`TreeWithValues<u32>`] version, hierarchy
    /// already erased; an `Arc<TreeSnapshot<_>>` coerces whenever the hierarchy's leaf
    /// value implements `AttributePtr<u32>` — with the given voxel size.
    ///
    /// `voxel_type_attributes` must hold the [`VoxelType`] of every voxel occupied
    /// in this tree version, keyed by the version's leaf pool indices
    /// ([`ErasedLeafView::leaf_index`]); every query reads it during iteration.
    ///
    /// `voxel_mask_attributes` is the optional faster tier for
    /// [`QueriedVoxel::voxel_state`], which contact-manifold computation reads on
    /// contact-candidate voxels. When present, a state is one indexed load. When
    /// absent, it is derived by testing the occupancy of the voxel's six axis
    /// neighbors — in-leaf neighbors on the leaf's own occupancy words, others by
    /// one tree descent each — which is fine for most objects; highly dynamic
    /// objects with many contacts benefit from the stored form.
    pub fn new(
        tree: Arc<dyn TreeWithValues<u32>>,
        voxel_type_attributes: Arc<VdbVoxelTypeAttributes>,
        voxel_mask_attributes: Option<Arc<VdbVoxelMaskAttributes>>,
        voxel_size: Vector,
    ) -> Self {
        Self {
            tree,
            voxel_type_attributes,
            voxel_mask_attributes,
            voxel_size,
        }
    }

    /// The size of each voxel along each local coordinate axis.
    pub fn voxel_size(&self) -> Vector {
        self.voxel_size
    }
    /// The stable identifier of the voxel at `key`, as used for parry [`FeatureId`]s and
    /// contact-manifold sub-shape ids.
    ///
    /// This is a flat index in the tree's addressable extent
    /// ([`TreeErased::extent`](dust_vdb::TreeErased::extent) — a property of the
    /// hierarchy, not of the content), so it is stable across edits and across
    /// snapshots.
    pub fn linear_id_of(&self, key: IVector) -> u32 {
        let e = self.tree.extent();
        (key.x as u32 * e.y + key.y as u32) * e.z + key.z as u32
    }

    /// The inverse of [`Self::linear_id_of`]: the grid coordinates of the voxel with the
    /// given identifier.
    pub fn key_of_linear_id(&self, id: u32) -> IVector {
        let e = self.tree.extent();
        let z = id % e.z;
        let y = (id / e.z) % e.y;
        let x = id / (e.y * e.z);
        IVector::new(x as i32, y as i32, z as i32)
    }
}

/// The six axis neighbors, in parry's `AxisMask` convention.
const NEIGHBOR_DIRS: [(IVector, AxisMask); 6] = [
    (IVector::new(1, 0, 0), AxisMask::X_POS),
    (IVector::new(-1, 0, 0), AxisMask::X_NEG),
    (IVector::new(0, 1, 0), AxisMask::Y_POS),
    (IVector::new(0, -1, 0), AxisMask::Y_NEG),
    (IVector::new(0, 0, 1), AxisMask::Z_POS),
    (IVector::new(0, 0, -1), AxisMask::Z_NEG),
];

impl VoxelQuery for VdbShape {
    fn voxel_size(&self) -> Vector {
        self.voxel_size
    }

    fn domain(&self) -> [IVector; 2] {
        let bounds = self.tree.aabb();
        if bounds.is_empty() {
            [IVector::ZERO, IVector::ZERO]
        } else {
            // `AabbU32` bounds are inclusive; parry domains are semi-open.
            [
                uvec_to_ivec(bounds.min),
                uvec_to_ivec(bounds.max) + IVector::splat(1),
            ]
        }
    }


    fn linear_id(&self, key: IVector) -> Option<u32> {
        todo!()
    }

    fn voxels_in_range(&self, mins: IVector, maxs: IVector) -> impl Iterator<Item = Self::Voxel<'_>> {
        // Voxels only exist within the tree's addressable extent; clip the
        // requested semi-open box `[mins, maxs)` to it.
        let lo = mins.max(IVector::ZERO);
        let hi = maxs.min(uvec_to_ivec(self.tree.extent()));
        // `iter_leaf_views_in_range` takes inclusive bounds, which cannot
        // express an empty box. When the clipped box is empty, hand the walk
        // `[0, 0]` — it visits at most the one leaf at the origin — and let
        // the per-voxel clip against the (empty) box reject everything.
        let range = if lo.cmplt(hi).all() {
            AabbU32 {
                min: ivec_to_uvec(lo),
                max: ivec_to_uvec(hi - IVector::ONE),
            }
        } else {
            AabbU32 {
                min: UVec3::ZERO,
                max: UVec3::ZERO,
            }
        };
        self.tree
            .iter_leaf_views_in_range_with_values(range)
            .flat_map(move |leaf| LeafVoxels::new(self, leaf, lo, hi))
    }

    type Voxel<'a>
        = VdbVoxel<'a>
    where
        Self: 'a;

    fn voxel(&self, key: IVector) -> Option<Self::Voxel<'_>> {
        // Point lookup — same missing piece as `VdbShape::occupied`.
        let _ = key;
        todo!()
    }
}

/// The number of occupied voxels before `coords` in `leaf`'s occupancy words
/// — the `fitted_offset` (in [`dust_vdb::Attributes`] terms) of `coords`
/// within an attribute range holding one slot per occupied voxel of the leaf,
/// in occupancy-bit order.
fn fitted_offset(leaf: &ErasedLeafView<u32>, coords: UVec3) -> u32 {
    let words = leaf.occupancy_words();
    let bit = leaf.bit_of_coord(coords);
    let word = (bit / usize::BITS) as usize;
    words[..word].iter().map(|w| w.count_ones()).sum::<u32>()
        + (words[word] & ((1usize << (bit % usize::BITS)) - 1)).count_ones()
}

/// One occupied voxel of a [`VdbShape`]: what its [`VoxelQuery`] lookups and
/// iterators hand out.
///
/// The view carries the [`ErasedLeafView`] of the leaf containing the voxel,
/// so [`QueriedVoxel::voxel_state`] resolves in-leaf neighbors on that leaf's
/// own occupancy words and only descends the tree for neighbors lying in an
/// adjacent leaf.
#[derive(Clone, Copy)]
pub struct VdbVoxel<'a> {
    shape: &'a VdbShape,
    leaf: ErasedLeafView<'a, u32>,
    /// Tree-global coordinates; always an occupied voxel of `leaf`.
    coords: UVec3,
}

impl<'a> QueriedVoxel<'a> for VdbVoxel<'a> {
    fn voxel_type(&self) -> VoxelType {
        self.shape
            .voxel_type_attributes
            .voxel_type(self.leaf.leaf_index(), self.leaf.bit_of_coord(self.coords))
    }

    fn voxel_state(&self) -> VoxelState {
        let state = match &self.shape.voxel_mask_attributes {
            // Stored states: [`VdbVoxelMaskAttributes`] ranges hold one
            // `VoxelState` per occupied voxel and start at the leaf value's
            // attribute pointer (the `material_ptr` mirror), which the erased
            // view projects.
            Some(mask_attributes) => {
                let ptr = *self.leaf.attribute_ptr();
                mask_attributes.values
                    [ptr as usize + fitted_offset(&self.leaf, self.coords) as usize]
            }
            // No stored states: derive one from the occupancy of the six
            // axis neighbors.
            None => {
                panic!("Unsupported operation; please derive voxel_mask_attributes")
            }
        };
        debug_assert_eq!(
            state.voxel_type(),
            self.voxel_type(),
            "VoxelState at {:?} disagrees with the stored VoxelType",
            self.coords,
        );
        state
    }

    fn linear_id(&self) -> u32 {
        self.shape.linear_id_of(uvec_to_ivec(self.coords))
    }

    fn grid_coords(&self) -> IVector {
        uvec_to_ivec(self.coords)
    }

    fn center(&self) -> Vector {
        self.shape.voxel_center(self.grid_coords())
    }
}

/// The per-leaf stage of [`VdbShape::voxels_in_range`]: walks the set bits of
/// one leaf's occupancy words in order, yielding a [`VdbVoxel`] for each
/// occupied voxel lying within the queried box.
struct LeafVoxels<'a> {
    shape: &'a VdbShape,
    leaf: ErasedLeafView<'a, u32>,
    /// The not-yet-yielded set bits of the occupancy word at `word_index`.
    word: usize,
    word_index: u32,
    /// The semi-open box `[lo, hi)` to clip against.
    lo: IVector,
    hi: IVector,
    /// When the whole leaf lies inside `[lo, hi)`, no per-voxel clip test is
    /// needed.
    fully_inside: bool,
}

impl<'a> LeafVoxels<'a> {
    fn new(shape: &'a VdbShape, leaf: ErasedLeafView<'a, u32>, lo: IVector, hi: IVector) -> Self {
        let origin = uvec_to_ivec(leaf.origin());
        let end = origin + uvec_to_ivec(leaf.extent());
        Self {
            shape,
            word: leaf.occupancy_words().first().copied().unwrap_or(0),
            word_index: 0,
            lo,
            hi,
            fully_inside: origin.cmpge(lo).all() && end.cmple(hi).all(),
            leaf,
        }
    }
}

impl<'a> Iterator for LeafVoxels<'a> {
    type Item = VdbVoxel<'a>;

    fn next(&mut self) -> Option<VdbVoxel<'a>> {
        loop {
            if self.word == 0 {
                self.word_index += 1;
                let words = self.leaf.occupancy_words();
                if self.word_index as usize >= words.len() {
                    return None;
                }
                self.word = words[self.word_index as usize];
                continue;
            }
            let bit = self.word_index * usize::BITS + self.word.trailing_zeros();
            self.word &= self.word - 1;
            let coords = self.leaf.origin() + self.leaf.coord_of_bit(bit);
            if !self.fully_inside {
                let key = uvec_to_ivec(coords);
                if !(key.cmpge(self.lo).all() && key.cmplt(self.hi).all()) {
                    continue;
                }
            }
            return Some(VdbVoxel {
                shape: self.shape,
                leaf: self.leaf,
                coords,
            });
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use dust_vdb::{AttributePtr, Tree, hierarchy};

    /// The leaf inline value of the test hierarchy. Like `VoxLeafNode` in
    /// `dust_vox`, it stores one allocator pointer (`VoxLeafNode` calls its
    /// field `material_ptr`), which doubles as the pointer of every fitted
    /// store whose allocator mirrors it — here, [`VdbVoxelMaskAttributes`].
    #[derive(Clone, Debug)]
    struct TestLeaf {
        ptr: u32,
    }

    impl Default for TestLeaf {
        fn default() -> Self {
            Self { ptr: u32::MAX }
        }
    }

    /// The plain-`u32` projection [`ErasedLeafView::attribute_ptr`] reads.
    impl AttributePtr<u32> for TestLeaf {
        fn attribute_ptr(&self) -> u32 {
            self.ptr
        }

        fn set_attribute_ptr(&mut self, ptr: u32) {
            self.ptr = ptr;
        }
    }

    /// The composite pointer of an accessor session driving
    /// `(VdbVoxelTypeAttributes, VdbVoxelMaskAttributes)`: `()` for the type
    /// store (keyed by leaf index alone) and `u32` for the mask store.
    impl AttributePtr<((), u32)> for TestLeaf {
        fn attribute_ptr(&self) -> ((), u32) {
            ((), self.ptr)
        }

        fn set_attribute_ptr(&mut self, ptr: ((), u32)) {
            self.ptr = ptr.1;
        }
    }

    /// 4³ leaves under one 4³ internal level: a 16³ domain with 64-voxel
    /// leaves, small enough to reason about by hand.
    type TestTree = Tree<hierarchy!(2, 2, TestLeaf)>;

    /// The test content: the solid 3×3×3 cube covering coordinates 2..5 on
    /// every axis. It straddles all eight leaves around the corner at (4, 4,
    /// 4), and contains every [`VoxelType`]: its center voxel is `Interior`,
    /// face centers are `Face`, edge centers are `Edge`, corners are
    /// `Vertex`.
    fn in_cube(key: IVector) -> bool {
        key.cmpge(IVector::splat(2)).all() && key.cmplt(IVector::splat(5)).all()
    }

    /// The state the cube voxel at `key` must have, straight from the cube's
    /// definition: which of the six axis neighbors are also cube voxels.
    fn expected_state(key: IVector) -> VoxelState {
        let mut filled = AxisMask::empty();
        for (dir, axis) in NEIGHBOR_DIRS {
            if in_cube(key + dir) {
                filled |= axis;
            }
        }
        VoxelState::with_filled_neighbors(filled)
    }

    /// Builds the cube through one accessor session driving both stores, and
    /// wraps a snapshot of it into a shape holding the mask store.
    ///
    /// Coverage is range-walk only for now: point lookups (`voxel`,
    /// `linear_id`) and the occupancy-derived `voxel_state` path (used by a
    /// shape without the mask store) funnel through [`VdbShape::occupied`],
    /// which is `todo!()` until point lookup gets a replacement for the
    /// removed `TreeErased::leaf_view_at`.
    #[test]
    fn voxel_query() {
        let mut tree = TestTree::new();
        let mut attributes = (
            VdbVoxelTypeAttributes::new(64),
            VdbVoxelMaskAttributes::new(16, 512),
        );

        let mut accessor = tree.accessor_mut(&mut attributes);
        for x in 2..5 {
            for y in 2..5 {
                for z in 2..5 {
                    let key = IVector::new(x, y, z);
                    let state = expected_state(key);
                    accessor.set(ivec_to_uvec(key), (state.voxel_type(), state));
                }
            }
        }
        drop(accessor);

        let (type_attributes, mask_attributes) = attributes;
        let tree: Arc<dyn TreeWithValues<u32>> = Arc::new(tree.snapshot());
        let with_masks = VdbShape::new(
            tree,
            Arc::new(type_attributes),
            Some(Arc::new(mask_attributes)),
            Vector::splat(1.0),
        );

        assert_eq!(
            with_masks.domain(),
            [IVector::splat(2), IVector::splat(5)]
        );

        // Every voxel comes out of the full iteration exactly once, with the
        // type and state that were stored for it.
        let mut seen = std::collections::HashSet::new();
        for voxel in with_masks.voxels() {
            let key = voxel.grid_coords();
            assert!(in_cube(key));
            assert!(seen.insert((key.x, key.y, key.z)));
            let state = expected_state(key);
            assert_eq!(voxel.voxel_type(), state.voxel_type());
            assert_eq!(voxel.voxel_state(), state);
            assert_eq!(voxel.linear_id(), with_masks.linear_id_of(key));
            assert_eq!(voxel.center(), Vector::new(key.x as Real, key.y as Real, key.z as Real) + Vector::splat(0.5));
        }
        assert_eq!(seen.len(), 27);

        // A partial range: x in [3, 5), y and z in [0, 4) — the per-voxel
        // clip must cut the cube down to x in {3, 4}, y and z in {2, 3}.
        let partial: Vec<IVector> = with_masks
            .voxels_in_range(IVector::new(3, 0, 0), IVector::new(5, 4, 4))
            .map(|voxel| voxel.grid_coords())
            .collect();
        assert_eq!(partial.len(), 8);
        for key in partial {
            assert!(key.x >= 3 && key.y < 4 && key.z < 4 && in_cube(key));
        }

        // Ranges reaching outside the addressable extent clip; empty ranges
        // yield nothing.
        assert_eq!(
            with_masks
                .voxels_in_range(IVector::splat(-100), IVector::splat(100))
                .count(),
            27
        );
        assert_eq!(
            with_masks
                .voxels_in_range(IVector::splat(3), IVector::splat(3))
                .count(),
            0
        );
    }
}
