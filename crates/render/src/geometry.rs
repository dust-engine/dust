use std::{sync::Arc, ops::{DerefMut, Deref}, alloc::Layout, marker::PhantomData};

use bevy_app::{Plugin, Update};
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_reflect::TypeUuid;
use rhyolite::{BufferLike, future::{GPUCommandFuture, RenderRes}, ResidentBuffer, ash::vk};
use rhyolite_bevy::RenderSystems;

use crate::blas::build_blas_system;

pub enum GeometryType {
    AABBs,
    Triangles
}

pub trait Geometry: Send + Sync + 'static + TypeUuid {
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
    _marker: PhantomData<G>
}

impl<G: Geometry> Default for GeometryPlugin<G> {
    fn default() -> Self {
        Self {
            _marker: PhantomData
        }
    }
}


impl<G: Geometry> Plugin for GeometryPlugin<G> {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(Update, (
            crate::blas::geometry_normalize_system::<G>.in_set(RenderSystems::Render).after(build_blas_system),
        ));
    }
}
