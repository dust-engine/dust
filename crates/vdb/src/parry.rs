use crate::{Node, Tree};

impl<ROOT: Node> parry3d::query::RayCast for Tree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        max_time_of_impact: parry3d::math::Real,
        solid: bool,
    ) -> Option<parry3d::query::RayIntersection> {
        todo!()
    }
}

impl<ROOT: Node> parry3d::query::PointQuery for Tree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
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
impl<ROOT: Node> parry3d::shape::Shape for Tree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    fn compute_local_aabb(&self) -> parry3d::bounding_volume::Aabb {
        todo!()
    }

    fn compute_local_bounding_sphere(&self) -> parry3d::bounding_volume::BoundingSphere {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn parry3d::shape::Shape> {
        todo!()
    }

    fn mass_properties(
        &self,
        density: parry3d::math::Real,
    ) -> parry3d::mass_properties::MassProperties {
        todo!()
    }

    fn shape_type(&self) -> parry3d::shape::ShapeType {
        todo!()
    }

    fn as_typed_shape(&self) -> parry3d::shape::TypedShape {
        todo!()
    }

    fn ccd_thickness(&self) -> parry3d::math::Real {
        todo!()
    }

    fn ccd_angular_thickness(&self) -> parry3d::math::Real {
        todo!()
    }
}
