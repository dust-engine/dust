use std::{cmp::Ordering, collections::BinaryHeap};

use parry3d::{
    bounding_volume::SimdAabb,
    math::{SimdBool, SimdReal},
};

use crate::{ImmutableTree, Node, TreeLike};

const SIMD_WIDTH: usize = 4;

pub trait TreeTraversal: TreeLike {
    type ROOT: Node;

    fn root(&self) -> &Self::ROOT;

    fn cast_local_ray_and_get_normal(
        &self,
        ray: &parry3d::query::Ray,
        max_time_of_impact: parry3d::math::Real,
        solid: bool,
    ) -> Option<parry3d::query::RayIntersection> {
        None
    }
}
