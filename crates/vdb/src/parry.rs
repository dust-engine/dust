use std::sync::Arc;

use glam::Vec3;
use parry3d::query::{PointQuery, RayCast};

use crate::{ImmutableTree, ImmutableTreeSnapshot, Node, Tree, TreeLike};

pub struct VdbShape<T> {
    inner: Arc<T>,
    scale: Vec3,
}

impl<T: TreeLike> parry3d::query::RayCast for VdbShape<T>{
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        max_time_of_impact: parry3d::math::Real,
        solid: bool,
    ) -> Option<parry3d::query::RayIntersection> {
        todo!()
    }
}

impl<T> VdbShape<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
            scale: Vec3::splat(1.0),
        }
    }
}

impl<T: TreeLike> parry3d::query::PointQuery for VdbShape<T>{
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
impl<T: TreeLike + Send + Sync + 'static> parry3d::shape::Shape for VdbShape<T> {
    fn compute_local_aabb(&self) -> parry3d::bounding_volume::Aabb {
        let aabb = self.inner.aabb();
        parry3d::bounding_volume::Aabb {
            maxs: aabb.max.as_vec3().into(),
            mins: aabb.min.as_vec3().into(),
        }
    }

    fn compute_local_bounding_sphere(&self) -> parry3d::bounding_volume::BoundingSphere {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn parry3d::shape::Shape> {
        todo!()
    }

    fn clone_scaled(&self, scale: &parry3d::math::Vector<parry3d::math::Real>, _num_subdivisions: u32) -> Option<Box<dyn parry3d::shape::Shape>> {
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
