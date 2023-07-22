use std::{alloc::Layout, marker::PhantomData, sync::Arc};

use bevy_app::{Plugin, PostUpdate};
use bevy_asset::Asset;
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_reflect::{TypePath, TypeUuid};
use rhyolite::{ash::vk, future::GPUCommandFuture, ResidentBuffer};
use rhyolite_bevy::RenderSystems;

use crate::accel_struct::blas::build_blas_system;

pub enum GeometryType {
    AABBs,
    Triangles,
}

pub trait Geometry: Send + Sync + 'static + Asset + TypePath {
    const TYPE: GeometryType;

    type BLASInputBufferFuture: GPUCommandFuture<Output = Arc<ResidentBuffer>>;
    fn blas_input_buffer(&self) -> Self::BLASInputBufferFuture;

    fn geometry_flags(&self) -> vk::GeometryFlagsKHR {
        vk::GeometryFlagsKHR::OPAQUE
    }

    /// Layout for one single AABB entry
    fn layout(&self) -> Layout {
        Layout::new::<vk::AabbPositionsKHR>()
    }
}

pub struct GeometryPlugin<G: Geometry> {
    _marker: PhantomData<G>,
}

impl<G: Geometry> Default for GeometryPlugin<G> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<G: Geometry> Plugin for GeometryPlugin<G> {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            PostUpdate,
            (crate::accel_struct::blas::geometry_normalize_system::<G>
                .in_set(RenderSystems::SetUp)
                .before(build_blas_system),),
        );
    }
}
