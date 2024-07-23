use std::sync::Arc;

use glam::{Vec2, Vec3, Vec3A};
use parry3d::query::{PointQuery, RayCast, Unsupported};

use crate::{ImmutableTree, ImmutableTreeSnapshot, MutableTree, Node, TreeLike};

pub struct VdbShape {
    inner: Arc<dyn TreeLike>,
    scale: Vec3,
}

impl parry3d::query::RayCast for VdbShape {
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        max_time_of_impact: parry3d::math::Real,
        solid: bool,
    ) -> Option<parry3d::query::RayIntersection> {
        self.inner
            .cast_local_ray_and_get_normal(ray, max_time_of_impact, solid)
    }
}

impl VdbShape {
    pub fn new(inner: Arc<dyn TreeLike>) -> Self {
        Self {
            inner,
            scale: Vec3::splat(1.0),
        }
    }
}

impl parry3d::query::PointQuery for VdbShape {
    fn project_local_point(
        &self,
        pt: &parry3d::math::Point<parry3d::math::Real>,
        solid: bool,
    ) -> parry3d::query::PointProjection {
        todo!()
    }

    fn project_local_point_and_get_feature(
        &self,
        pt: &parry3d::math::Point<parry3d::math::Real>,
    ) -> (parry3d::query::PointProjection, parry3d::shape::FeatureId) {
        todo!()
    }
}
impl parry3d::shape::Shape for VdbShape {
    fn compute_local_aabb(&self) -> parry3d::bounding_volume::Aabb {
        let aabb = self.inner.aabb();
        let min = aabb.min.as_vec3();
        let max = aabb.max.as_vec3();
        parry3d::bounding_volume::Aabb {
            maxs: max.into(),
            mins: min.into(),
        }
    }

    fn compute_local_bounding_sphere(&self) -> parry3d::bounding_volume::BoundingSphere {
        todo!()
    }

    fn clone_dyn(&self) -> Box<dyn parry3d::shape::Shape> {
        todo!()
    }

    fn scale_dyn(
        &self,
        scale: &parry3d::math::Vector<parry3d::math::Real>,
        _num_subdivisions: u32,
    ) -> Option<Box<dyn parry3d::shape::Shape>> {
        let scale: Vec3 = (*scale).into();
        Some(Box::new(Self {
            inner: self.inner.clone(),
            scale: scale * self.scale,
        }))
    }

    fn mass_properties(
        &self,
        density: parry3d::math::Real,
    ) -> parry3d::mass_properties::MassProperties {
        todo!()
    }

    fn shape_type(&self) -> parry3d::shape::ShapeType {
        parry3d::shape::ShapeType::Custom
    }

    fn as_typed_shape(&self) -> parry3d::shape::TypedShape {
        parry3d::shape::TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> parry3d::math::Real {
        todo!()
    }

    fn ccd_angular_thickness(&self) -> parry3d::math::Real {
        todo!()
    }
}

#[derive(Clone, Copy)]
pub struct VdbQueryDispatcher;
impl parry3d::query::QueryDispatcherComposite for VdbQueryDispatcher {
    fn intersection_test(
        &self,
        root_dispatcher: &dyn parry3d::query::QueryDispatcher,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
    ) -> Result<bool, parry3d::query::Unsupported> {
        Err(Unsupported)
    }

    fn distance(
        &self,
        root_dispatcher: &dyn parry3d::query::QueryDispatcher,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
    ) -> Result<parry3d::math::Real, parry3d::query::Unsupported> {
        Err(Unsupported)
    }

    fn contact(
        &self,
        root_dispatcher: &dyn parry3d::query::QueryDispatcher,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
        prediction: parry3d::math::Real,
    ) -> Result<Option<parry3d::query::Contact>, parry3d::query::Unsupported> {
        Err(Unsupported)
    }

    fn closest_points(
        &self,
        root_dispatcher: &dyn parry3d::query::QueryDispatcher,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
        max_dist: parry3d::math::Real,
    ) -> Result<parry3d::query::ClosestPoints, parry3d::query::Unsupported> {
        // best first search
        Err(Unsupported)
    }

    fn cast_shapes(
        &self,
        root_dispatcher: &dyn parry3d::query::QueryDispatcher,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        local_vel12: &parry3d::math::Vector<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
        options: parry3d::query::ShapeCastOptions,
    ) -> Result<Option<parry3d::query::ShapeCastHit>, parry3d::query::Unsupported> {
        if let Some(g1) = g1.downcast_ref::<VdbShape>()
            && let Some(g2) = g2.as_support_map()
        {
            let support_pt = g2.local_support_point(local_vel12);
            let ray = parry3d::query::Ray::new(pos12.transform_point(&support_pt), *local_vel12);
            println!("Ray: {:?}", ray);
            let ray_cast = g1.cast_local_ray(&ray, 10000.0, false);
            println!("Cast shape: {:?}", ray_cast);
            return Ok(None);
        }
        Err(Unsupported)
    }

    fn cast_shapes_nonlinear(
        &self,
        root_dispatcher: &dyn parry3d::query::QueryDispatcher,
        motion1: &parry3d::query::NonlinearRigidMotion,
        g1: &dyn parry3d::shape::Shape,
        motion2: &parry3d::query::NonlinearRigidMotion,
        g2: &dyn parry3d::shape::Shape,
        start_time: parry3d::math::Real,
        end_time: parry3d::math::Real,
        stop_at_penetration: bool,
    ) -> Result<Option<parry3d::query::ShapeCastHit>, parry3d::query::Unsupported> {
        Err(Unsupported)
    }
}

impl<A, B> parry3d::query::PersistentQueryDispatcherComposite<A, B> for VdbQueryDispatcher {
    fn contact_manifolds(
        &self,
        root_dispatcher: &dyn parry3d::query::PersistentQueryDispatcher<A, B>,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
        prediction: parry3d::math::Real,
        manifolds: &mut Vec<parry3d::query::ContactManifold<A, B>>,
        workspace: &mut Option<parry3d::query::ContactManifoldsWorkspace>,
    ) -> Result<(), parry3d::query::Unsupported> {
        Err(Unsupported)
    }

    fn contact_manifold_convex_convex(
        &self,
        pos12: &parry3d::math::Isometry<parry3d::math::Real>,
        g1: &dyn parry3d::shape::Shape,
        g2: &dyn parry3d::shape::Shape,
        normal_constraints1: Option<&dyn parry3d::query::details::NormalConstraints>,
        normal_constraints2: Option<&dyn parry3d::query::details::NormalConstraints>,
        prediction: parry3d::math::Real,
        manifold: &mut parry3d::query::ContactManifold<A, B>,
    ) -> Result<(), parry3d::query::Unsupported> {
        Err(Unsupported)
    }
}

pub(crate) fn intersect_aabb(origin: Vec3A, dir: Vec3A, box_min: Vec3A, box_max: Vec3A) -> Vec2 {
    let t_min = (box_min - origin) / dir;
    let t_max = (box_max - origin) / dir;
    let t1 = t_min.min(t_max);
    let t2 = t_min.max(t_max);
    let t_min = t1.max_element();
    let t_max = t2.min_element();
    return Vec2::new(t_min, t_max);
}
