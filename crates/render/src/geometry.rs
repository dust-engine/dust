use std::{borrow::Borrow, future::Future, marker::PhantomData, pin::Pin, sync::Arc};

use crate::{
    render_asset::{GPURenderAsset, RenderAsset, RenderAssetPlugin, RenderAssetStore},
    shader::{Shader, SpecializedShader},
    RenderStage, RenderWorld,
};
use ash::{prelude::VkResult, vk};
use bevy_app::{CoreStage, Plugin};
use bevy_asset::{
    AddAsset, Asset, AssetEvent, AssetServer, Assets, Handle, HandleId, HandleUntyped,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    event::EventReader,
    system::{
        Commands, IntoExclusiveSystem, Query, Res, ResMut, StaticSystemParam, SystemParam,
        SystemParamItem,
    },
    world::World,
};
use bevy_utils::{HashMap, HashSet};
use dustash::{
    accel_struct::AccelerationStructure,
    command::{pool::CommandPool, recorder::CommandRecorder},
    queue::{semaphore::TimelineSemaphoreOp, Queue, QueueType, Queues},
    resources::alloc::MemBuffer,
    shader::SpecializationInfo,
    sync::{CommandsFuture, GPUFuture, HostFuture},
    Device,
};
use std::future::IntoFuture;

pub type GeometryAABB = ash::vk::AabbPositionsKHR;

/// The geometry defines the shape of the voxel object.
/// It serves as the "Mesh" for voxels.
///
/// SVDAG, OpenVDB, 3D Grids and ESVO could be implementors of the GeometryStructure trait.
/// Handle<Geometry> is in the world space.
pub trait Geometry: RenderAsset {
    /// The geometry represented as an array of primitives
    /// This gets persisted in the render world.

    fn aabb(&self) -> GeometryAABB;
    fn intersection_shader(asset_server: &AssetServer) -> SpecializedShader;
}

/// RenderWorld Assets.
pub trait GPUGeometry<T: Geometry>: GPURenderAsset<T> {
    fn blas_input_buffer(&self) -> &Arc<MemBuffer>;
    fn geometry_info(&self) -> u64;
}

/// Insert Handle<T> in the render world for all entities with Handle<T>
#[derive(Component)]
pub struct GPUGeometryPrimitives {
    pub handle: HandleId,
    pub blas_input_primitives: Option<Arc<MemBuffer>>, // None if the geometry hasn't been loaded yet.
    pub geometry_info: u64,
}
fn extract_primitives<T: Geometry>(mut commands: Commands, query: Query<(Entity, &Handle<T>)>) {
    for (entity, geometry_handle) in query.iter() {
        commands
            .get_or_spawn(entity)
            .insert(geometry_handle.clone());
    }
}
/// Insert GPUGeometryPrimitives for all entities with Handle<T>
fn prepare_primitives<T: Geometry>(
    mut commands: Commands,
    geometry_store: Res<RenderAssetStore<T>>,
    query: Query<(Entity, &Handle<T>)>,
) where
    T::GPUAsset: GPUGeometry<T>,
{
    for (entity, geometry_handle) in query.iter() {
        if let Some(geometry) = geometry_store.get(geometry_handle) {
            let buf = geometry.blas_input_buffer().clone();
            commands.entity(entity).insert(GPUGeometryPrimitives {
                handle: geometry_handle.id,
                blas_input_primitives: Some(buf),
                geometry_info: geometry.geometry_info(),
            });
        } else {
            commands.entity(entity).insert(GPUGeometryPrimitives {
                handle: geometry_handle.id,
                blas_input_primitives: None,
                geometry_info: 0,
            });
        }
    }
}

pub struct GeometryPlugin<T: Geometry> {
    _marker: PhantomData<T>,
}
impl<T: Geometry> Default for GeometryPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: Geometry> Plugin for GeometryPlugin<T>
where
    T::GPUAsset: GPUGeometry<T>,
{
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugin(RenderAssetPlugin::<T>::default());

        let render_app = app.sub_app_mut(crate::RenderApp);
        render_app
            .add_system_to_stage(RenderStage::Extract, extract_primitives::<T>)
            .add_system_to_stage(RenderStage::Prepare, prepare_primitives::<T>);
        // TODO: maybe run prepare_primitives after prepare_geometries to decrease frame delay?
    }
}
