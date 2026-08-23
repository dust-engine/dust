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
//!   bits — leaves come whole through [`TreeErased::iter_leaf_views_in_range`] and
//!   [`TreeErased::leaf_view_at`], everything per-voxel is bit math on their words — so
//!   the tree remains the single source of truth.
//! - [`VdbShape`] is a `'static`, [`Shape`]-implementing immutable value over one
//!   `Arc<dyn TreeErased>` (in practice a [`TreeSnapshot`]), suitable for
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
//! [`TreeErased::leaf_view_at`]: dust_vdb::TreeErased::leaf_view_at
//! [`Tree::reclaim_dropped_snapshots`]: dust_vdb::Tree::reclaim_dropped_snapshots
//! [`Tree::restore`]: dust_vdb::Tree::restore

mod dispatcher;

pub use dispatcher::VdbDispatcher;

use std::sync::Arc;

use dust_vdb::{ErasedLeafView, ErasedLeafViewIter, TreeErased};
use glam::UVec3;
use parry3d::bounding_volume::{Aabb, BoundingSphere};
use parry3d::mass_properties::MassProperties;
use parry3d::math::{IVector, Real, Vector};
use parry3d::query::details::{cast_local_ray_on_voxels, project_local_point_on_voxels};
use parry3d::query::{PointProjection, PointQuery, Ray, RayCast, RayIntersection};
use parry3d::shape::{
    AxisMask, FeatureId, Shape, ShapeType, TypedShape, VoxelData, VoxelQuery, VoxelState,
};

/// A `'static` voxel collision shape over one [`TreeErased`] version (in practice an
/// `Arc<TreeSnapshot>`), implementing parry's [`Shape`] so it can live inside a
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
    tree: Arc<dyn TreeErased>,
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
}

impl dust_vdb::Attributes for VdbVoxelTypeAttributes {
    type Ptr = ();
    /// The type of the attribute values. For a MagicaVoxel grid, this would be a u8 palette index.
    type Value = parry3d::shape::VoxelType;
    fn get_attribute(&self, index: u32, _ptr: &Self::Ptr, _fitted_offset: u32, inflated_offset: u32) -> Self::Value {
        match self.get_code(index, inflated_offset) {
            0 => parry3d::shape::VoxelType::Interior,
            1 => parry3d::shape::VoxelType::Face,
            2 => parry3d::shape::VoxelType::Edge,
            3 => parry3d::shape::VoxelType::Vertex,
            _ => unreachable!(),
        }
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
        ptr: &Self::Ptr,
        original_mask: &[usize],
        new_mask: &[usize],
        _coords: &UVec3,
    ) -> Self::Ptr {
        // Values are stored at each voxel's index within its leaf, and that
        // index never changes when the leaf's attribute layout changes — so
        // a layout change requires nothing here. Only a copy-on-write fork
        // (`original_leaf != new_leaf`) does: the whole fixed block of bit
        // positions is copied over. The masks are not needed: positions of
        // voxels that no longer exist come along in the copy, but they are
        // never read, because every read is preceded by an occupancy check.
        let _ = (ptr, original_mask, new_mask);
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
    /// Creates a shape reading `tree` — any [`TreeErased`] version, hierarchy already
    /// erased; an `Arc<TreeSnapshot<_>>` coerces — with the given voxel size.
    pub fn new(tree: Arc<dyn TreeErased>, voxel_size: Vector) -> Self {
        todo!()
    }

    /// The size of each voxel along each local coordinate axis.
    pub fn voxel_size(&self) -> Vector {
        self.voxel_size
    }
    /// The stable identifier of the voxel at `key`, as used for parry [`FeatureId`]s and
    /// contact-manifold sub-shape ids.
    ///
    /// This is a flat index in the tree's addressable extent ([`TreeErased::extent`] —
    /// a property of the hierarchy, not of the content), so it is stable across edits
    /// and across snapshots.
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

    /// The occupancy of the leaf containing `key`, if `key` is within the tree's
    /// addressable extent and the leaf exists in this version. One tree descent.
    fn leaf_view(&self, key: IVector) -> Option<ErasedLeafView<'_>> {
        if key.cmplt(IVector::ZERO).any() {
            return None;
        }
        self.tree.leaf_view_at(ivec_to_uvec(key))
    }

    /// Whether the voxel at `key` is filled in this version. Coordinates outside of the
    /// tree's addressable extent (including negative ones) are empty.
    fn occupied(&self, key: IVector) -> bool {
        self.leaf_view(key)
            .is_some_and(|leaf| leaf.get(ivec_to_uvec(key)))
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
        self.occupied(key).then(|| self.linear_id_of(key))
    }

    fn voxels_in_range(&self, mins: IVector, maxs: IVector) -> impl Iterator<Item = Self::Voxel<'_>> {
        let _ = (mins, maxs);
        todo!();
        #[allow(unreachable_code)]
        std::iter::empty()
    }

    /// Placeholder until the leaf-view-cursor voxel view lands: parry's
    /// owned [`VoxelData`] is a valid [`QueriedVoxel`](parry3d::shape) for
    /// the type to compile against.
    type Voxel<'a>
        = VoxelData
    where
        Self: 'a;

    fn voxel(&self, key: IVector) -> Option<Self::Voxel<'_>> {
        let _ = key;
        todo!()
    }
}


