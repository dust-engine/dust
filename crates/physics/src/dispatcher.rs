//! A parry [`QueryDispatcher`] routing pairs that involve a [`VdbShape`] to parry's
//! generic voxel collision-detection algorithms.
//!
//! Parry's [`DefaultQueryDispatcher`] only recognizes its own shape types; a [`VdbShape`]
//! ([`ShapeType::Custom`](parry3d::shape::ShapeType::Custom)) makes every pairwise query
//! return [`Unsupported`]. [`VdbDispatcher`] fills exactly that gap: it downcasts either
//! operand to [`VdbShape`] and runs the same generic algorithms parry uses for its own
//! `Voxels` shape — over the shape's [`VoxelQuery`](parry3d::shape::VoxelQuery) view
//! ([`VdbShape::voxels`]) instead. Pairs without a [`VdbShape`] are reported as
//! [`Unsupported`] so they can fall through a chain.
//!
//! Use it chained with the default dispatcher, mirroring the routing table of
//! [`DefaultQueryDispatcher`] for a physics engine:
//!
//! ```ignore
//! let dispatcher = VdbDispatcher::new().chain(DefaultQueryDispatcher);
//! // rapier: pass to `NarrowPhase::with_query_dispatcher` / `QueryPipeline`.
//! // avian: `app.insert_resource(QueryDispatcher::new(Box::new(dispatcher)))`.
//! ```
//!
//! Query coverage matches what [`DefaultQueryDispatcher`] provides for parry's `Voxels`:
//! contact manifolds (against balls, composite shapes, other voxel shapes — including
//! parry's own `Voxels` — and generic shapes), intersection tests, and linear/nonlinear
//! shape-casts. `distance`, `contact`, and `closest_points` are unsupported for voxel
//! shapes upstream, and remain so here.

use parry3d::math::{Pose, Real, Vector};
use parry3d::query::details::{
    NormalConstraints, cast_shapes_nonlinear_shape_voxels, cast_shapes_nonlinear_voxels_shape,
    cast_shapes_shape_voxels, cast_shapes_voxels_shape, contact_manifolds_voxels_ball,
    contact_manifolds_voxels_composite_shape, contact_manifolds_voxels_shape,
    contact_manifolds_voxels_voxels, intersection_test_shape_voxels,
    intersection_test_voxels_shape,
};
use parry3d::query::{
    ClosestPoints, Contact, ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher,
    NonlinearRigidMotion, PersistentQueryDispatcher, QueryDispatcher, QueryDispatcherChain,
    ShapeCastHit, ShapeCastOptions, Unsupported,
};
use parry3d::shape::{Ball, Shape};

use crate::VdbShape;

/// A [`QueryDispatcher`] handling shape pairs where either side is a [`VdbShape`].
///
/// [`VdbShape`] is already hierarchy-erased (it holds an
/// `Arc<dyn TreeErased>`), so one dispatcher (and one downcast) covers colliders built
/// from every `hierarchy!` instantiation — including pairs mixing two different
/// hierarchies, which route to the voxels-voxels algorithms like any other voxel pair.
/// Stateless and zero-sized; see the [module docs](self) for usage.
#[derive(Clone, Copy, Default)]
pub struct VdbDispatcher;

impl VdbDispatcher {
    pub fn new() -> Self {
        Self
    }

    /// The dispatcher handed down to parry's recursive algorithms for sub-queries
    /// (e.g. the per-voxel convex pairs inside a contact-manifold computation).
    ///
    /// It must handle both voxel and standard pairs, so it is `self` chained with the
    /// default dispatcher — regardless of what the *outer* chain looks like.
    fn full(&self) -> QueryDispatcherChain<Self, DefaultQueryDispatcher> {
        (*self).chain(DefaultQueryDispatcher)
    }

    fn downcast<'a>(&self, shape: &'a dyn Shape) -> Option<&'a VdbShape> {
        shape.as_shape::<VdbShape>()
    }
}

impl QueryDispatcher for VdbDispatcher {
    fn intersection_test(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<bool, Unsupported> {
        if let Some(vdb1) = self.downcast(g1) {
            Ok(intersection_test_voxels_shape(
                &self.full(),
                pos12,
                vdb1,
                g2,
            ))
        } else if let Some(vdb2) = self.downcast(g2) {
            Ok(intersection_test_shape_voxels(
                &self.full(),
                pos12,
                g1,
                vdb2,
            ))
        } else {
            Err(Unsupported)
        }
    }

    // Parry's own dispatcher does not implement `distance`, `contact`, or
    // `closest_points` for voxel shapes either; keep parity and let the chain decide.
    fn distance(&self, _: &Pose, _: &dyn Shape, _: &dyn Shape) -> Result<Real, Unsupported> {
        Err(Unsupported)
    }

    fn contact(
        &self,
        _: &Pose,
        _: &dyn Shape,
        _: &dyn Shape,
        _: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        Err(Unsupported)
    }

    fn closest_points(
        &self,
        _: &Pose,
        _: &dyn Shape,
        _: &dyn Shape,
        _: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        Err(Unsupported)
    }

    fn cast_shapes(
        &self,
        pos12: &Pose,
        local_vel12: Vector,
        g1: &dyn Shape,
        g2: &dyn Shape,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        if let Some(vdb1) = self.downcast(g1) {
            Ok(cast_shapes_voxels_shape(
                &self.full(),
                pos12,
                local_vel12,
                vdb1,
                g2,
                options,
            ))
        } else if let Some(vdb2) = self.downcast(g2) {
            Ok(cast_shapes_shape_voxels(
                &self.full(),
                pos12,
                local_vel12,
                g1,
                vdb2,
                options,
            ))
        } else {
            Err(Unsupported)
        }
    }

    fn cast_shapes_nonlinear(
        &self,
        motion1: &NonlinearRigidMotion,
        g1: &dyn Shape,
        motion2: &NonlinearRigidMotion,
        g2: &dyn Shape,
        start_time: Real,
        end_time: Real,
        stop_at_penetration: bool,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        if let Some(vdb1) = self.downcast(g1) {
            Ok(cast_shapes_nonlinear_voxels_shape(
                &self.full(),
                motion1,
                vdb1,
                motion2,
                g2,
                start_time,
                end_time,
                stop_at_penetration,
            ))
        } else if let Some(vdb2) = self.downcast(g2) {
            Ok(cast_shapes_nonlinear_shape_voxels(
                &self.full(),
                motion1,
                g1,
                motion2,
                vdb2,
                start_time,
                end_time,
                stop_at_penetration,
            ))
        } else {
            Err(Unsupported)
        }
    }
}

impl<ManifoldData, ContactData> PersistentQueryDispatcher<ManifoldData, ContactData>
    for VdbDispatcher
where
    ManifoldData: Default + Clone,
    ContactData: Default + Copy,
{
    fn contact_manifolds(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
        workspace: &mut Option<ContactManifoldsWorkspace>,
    ) -> Result<(), Unsupported> {
        // Mirrors the voxel arms of `DefaultQueryDispatcher::contact_manifolds`, with the
        // downcast targeting `VdbShape` instead of parry's `Voxels`. The flipped variants
        // follow parry's `*_shapes` wrappers: swap the operands, invert the pose, and set
        // `flipped` so the manifolds come out in the caller's order.
        match (self.downcast(g1), self.downcast(g2)) {
            (Some(vdb1), Some(vdb2)) => {
                contact_manifolds_voxels_voxels(
                    &self.full(),
                    pos12,
                    vdb1,
                    vdb2,
                    prediction,
                    manifolds,
                    workspace,
                );
                Ok(())
            }
            (Some(vdb1), None) => {
                if let Some(ball2) = g2.as_shape::<Ball>() {
                    contact_manifolds_voxels_ball(pos12, vdb1, ball2, prediction, manifolds, false);
                } else if let Some(vdb2) = g2.as_voxels() {
                    contact_manifolds_voxels_voxels(
                        &self.full(),
                        pos12,
                        vdb1,
                        vdb2,
                        prediction,
                        manifolds,
                        workspace,
                    );
                } else if let Some(composite2) = g2.as_composite_shape() {
                    contact_manifolds_voxels_composite_shape(
                        &self.full(),
                        pos12,
                    vdb1,
                        composite2,
                        prediction,
                        manifolds,
                        workspace,
                        false,
                    );
                } else {
                    contact_manifolds_voxels_shape(
                        &self.full(),
                        pos12,
                        vdb1,
                        g2,
                        prediction,
                        manifolds,
                        workspace,
                        false,
                    );
                }
                Ok(())
            }
            (None, Some(vdb2)) => {
                let pos21 = pos12.inverse();
                if let Some(ball1) = g1.as_shape::<Ball>() {
                    contact_manifolds_voxels_ball(&pos21, vdb2, ball1, prediction, manifolds, true);
                } else if let Some(vdb1) = g1.as_voxels() {
                    // Voxels-voxels is order-symmetric; keep the caller's order.
                    contact_manifolds_voxels_voxels(
                        &self.full(),
                        pos12,
                        vdb1,
                        vdb2,
                        prediction,
                        manifolds,
                        workspace,
                    );
                } else if let Some(composite1) = g1.as_composite_shape() {
                    contact_manifolds_voxels_composite_shape(
                        &self.full(),
                        &pos21,
                        vdb2,
                        composite1,
                        prediction,
                        manifolds,
                        workspace,
                        true,
                    );
                } else {
                    contact_manifolds_voxels_shape(
                        &self.full(),
                        &pos21,
                        vdb2,
                        g1,
                        prediction,
                        manifolds,
                        workspace,
                        true,
                    );
                }
                Ok(())
            }
            (None, None) => Err(Unsupported),
        }
    }

    /// Voxel shapes are never convex; sub-pairs produced by the manifold algorithms above
    /// are plain convex shapes and are meant for the chained default dispatcher.
    fn contact_manifold_convex_convex(
        &self,
        _: &Pose,
        _: &dyn Shape,
        _: &dyn Shape,
        _: Option<&dyn NormalConstraints>,
        _: Option<&dyn NormalConstraints>,
        _: Real,
        _: &mut ContactManifold<ManifoldData, ContactData>,
    ) -> Result<(), Unsupported> {
        Err(Unsupported)
    }
}
