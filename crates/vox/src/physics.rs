//! Collision for Vox content, through [avian](avian3d).
//!
//! A [`VoxGeometry`] turns into an avian [`Collider`] with
//! [`VoxGeometry::collider`]: a [`VdbShape`] over a snapshot of the geometry's tree,
//! with one voxel per `unit_size` cube — the same object space the geometry renders
//! in, so the collider needs no offset from a [`VoxInstance`](crate::VoxInstance)'s
//! `Transform`. The application must install avian's [`QueryDispatcher`] resource
//! with a [`VdbDispatcher`](dust_physics::VdbDispatcher) chained in front of parry's
//! default dispatcher (see that type's docs), or every `VdbShape` pair is
//! unsupported.
//!
//! Nothing runs per frame. A model entity carries its geometry's collider as a
//! [`VoxModelCollider`], and an instance clones it (sharing the shape, an `Arc`)
//! when its [`VoxInstance`](crate::VoxInstance) template is built. The `.vox`
//! loader builds the model's collider while converting the model (off the main
//! thread, in parallel across models) and puts it in the model's scene. Code that
//! builds or edits a geometry itself does the same: [`VoxGeometry::collider`] after
//! the edit, on the model for future instances and on the existing instances'
//! [`Collider`] through the engine's normal component-mutation path, so avian's
//! change detection recomputes bounds and mass properties like it would for any
//! other shape change. Dropped snapshots return their nodes to the tree on its next
//! mutation session; nothing here tracks them.
//!
//! Nothing here decides what is a rigid body either: put a [`RigidBody`] on a Vox
//! scene's root (or any ancestor of its instances) and avian attaches the
//! instances' colliders to it; an instance under no rigid body is a static collider.

use std::sync::Arc;

use avian3d::parry::math::Vector;
use avian3d::parry::shape::SharedShape;
use avian3d::prelude::*;
use bevy::prelude::*;
use dust_physics::{VdbShape, VdbVoxelMaskAttributes, VdbVoxelTypeAttributes};
use dust_vdb::TreeWithValues;

use crate::VoxGeometry;

/// The collider of a [`VoxModel`](crate::VoxModel)'s geometry
/// ([`VoxGeometry::collider`]), on the model entity. Every instance of the model
/// inserts a clone of it as its own [`Collider`] when spawned; see the
/// [module docs](self).
#[derive(Component, Default, Clone)]
pub struct VoxModelCollider(pub Collider);

impl VoxGeometry {
    /// A collider of this geometry as it is now: a [`VdbShape`] over a snapshot of
    /// the tree, with both attribute stores the shape reads derived from the
    /// snapshot's occupancy. Later edits do not affect it; build a new one after
    /// editing and assign it to the geometry's instances.
    pub fn collider(&mut self) -> Collider {
        let snapshot = self.tree.snapshot();
        let masks = VdbVoxelMaskAttributes::from_tree(&snapshot);
        let snapshot: Arc<dyn TreeWithValues<u32>> = Arc::new(snapshot);
        let types = VdbVoxelTypeAttributes::from_masks(&*snapshot, &masks);
        Collider::from(SharedShape::new(VdbShape::new(
            snapshot,
            Arc::new(types),
            Some(Arc::new(masks)),
            Vector::splat(self.unit_size),
        )))
    }
}
