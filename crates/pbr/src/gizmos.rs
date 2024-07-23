use bevy::prelude::*;
use rhyolite::{ImageLike, SwapchainImage};

use crate::camera::PinholeProjection;

pub struct PbrRendererDelegate;
impl rhyolite_gizmos::GizmosDrawDelegate for PbrRendererDelegate {
    type Params = Query<'static, 'static, (&'static GlobalTransform, &'static PinholeProjection)>;
    fn get_view_transform(
        params: &mut bevy::ecs::system::SystemParamItem<Self::Params>,
        aspect_ratio: f32,
    ) -> Mat4 {
        let (transform, projection) = params.single();

        let proj = {
            Mat4::perspective_infinite_reverse_rh(projection.fov, aspect_ratio, projection.near)
        };
        let view_proj = proj * transform.compute_matrix().inverse();

        view_proj
    }
}
