use std::{borrow::Borrow, future::Future, marker::PhantomData, pin::Pin, sync::Arc};

use crate::{shader::Shader, RenderStage, RenderWorld};
use ash::{prelude::VkResult, vk};
use bevy_app::{CoreStage, Plugin};
use bevy_asset::{AddAsset, Asset, AssetEvent, AssetServer, Assets, Handle, HandleUntyped};
use bevy_ecs::{
    event::EventReader,
    system::{
        Commands, IntoExclusiveSystem, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem,
    },
    world::World,
};
use bevy_utils::{HashMap, HashSet};
use dustash::{
    accel_struct::AccelerationStructure,
    command::{pool::CommandPool, recorder::CommandRecorder},
    queue::{semaphore::TimelineSemaphoreOp, Queue, QueueType, Queues},
    ray_tracing::sbt::SpecializationInfo,
    sync::{CommandsFuture, GPUFuture, HostFuture},
    Device,
};
use futures_lite::future;
use std::future::IntoFuture;

pub type GeometryAABB = ash::vk::AabbPositionsKHR;

/// The geometry defines the shape of the voxel object.
/// It serves as the "Mesh" for voxels.
///
/// SVDAG, OpenVDB, 3D Grids and ESVO could be implementors of the GeometryStructure trait.
/// Handle<Geometry> is in the world space.
pub trait Geometry: Asset + Sized {
    /// The geometry represented as an array of primitives
    /// This gets persisted in the render world.
    /// This is a GPU state.
    type GPUGeometry: GPUGeometry<Self>;

    /// The change in geometry.
    /// This gets extracted from the main world into the render world each frame.
    type ChangeSet: Send + Sync;
    type BuildSet: Send + Sync;

    fn aabb(&self) -> GeometryAABB;
    fn intersection_shader(asset_server: &AssetServer) -> Handle<Shader>;
    fn specialization() -> SpecializationInfo;

    type GenerateBuildsParam: SystemParam;
    fn generate_builds(
        &mut self,
        param: &mut SystemParamItem<Self::GenerateBuildsParam>,
    ) -> Self::BuildSet;

    type EmitChangesParam: SystemParam;
    fn emit_changes(
        &mut self,
        param: &mut SystemParamItem<Self::EmitChangesParam>,
    ) -> Self::ChangeSet;
}

/// RenderWorld Assets.
pub trait GPUGeometry<T: Geometry>: Send + Sync {
    fn build(build_set: T::BuildSet, commands_future: &mut CommandsFuture) -> Self;
    fn apply_change_set(&mut self, change_set: T::ChangeSet, commands_future: &mut CommandsFuture);
}

enum GPUGeometryUpdate<T: Geometry> {
    Rebuild(T::BuildSet),
    Update(T::ChangeSet),
}
/// Structure that moves geometry data from AppWorld to RenderWorld.
struct GeometryCarrier<T: Geometry> {
    updates: HashMap<Handle<T>, GPUGeometryUpdate<T>>,
    removed: Vec<Handle<T>>,
}
impl<T: Geometry> Default for GeometryCarrier<T> {
    fn default() -> Self {
        Self {
            updates: HashMap::default(),
            removed: Vec::new(),
        }
    }
}

/// This runs in the main world in the PostUpdate stage.
fn generate_changes<A: Geometry>(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<A>>,
    mut geometries: ResMut<Assets<A>>,
    mut generate_builds_param: StaticSystemParam<A::GenerateBuildsParam>,
    mut emit_changes_param: StaticSystemParam<A::EmitChangesParam>,
) {
    // handle, rebuild
    let mut changed_assets: HashMap<Handle<A>, bool> = HashMap::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } => {
                println!("Created");
                // Always rebuild this asset.
                if let Some(entry) = changed_assets.get_mut(handle) {
                    *entry = true;
                } else {
                    changed_assets.insert(handle.clone(), true);
                }
            }
            AssetEvent::Modified { handle } => {
                // If the asset is gonna be rebuilt, just do that. Otherwise, update only.
                if !changed_assets.contains_key(handle) {
                    changed_assets.insert(handle.clone(), false);
                }
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }

    let mut updates: HashMap<Handle<A>, GPUGeometryUpdate<A>> = HashMap::default();
    for (handle, rebuild) in changed_assets.drain() {
        let geometry = geometries.get_mut_untracked(&handle).unwrap();
        if rebuild {
            updates.insert(
                handle,
                GPUGeometryUpdate::Rebuild(geometry.generate_builds(&mut generate_builds_param)),
            );
        } else {
            updates.insert(
                handle,
                GPUGeometryUpdate::Update(geometry.emit_changes(&mut emit_changes_param)),
            );
        }
    }

    // Insert an Option<GeometryCarrier> so that it can be taken.
    commands.insert_resource(Some(GeometryCarrier { updates, removed }));
}

/// This runs in the Extract stage of the Render World.
fn move_change_set_store_to_render_world<T: Geometry>(
    mut commands: Commands,
    mut geometery_carrier: ResMut<Option<GeometryCarrier<T>>>,
) {
    let carrier = geometery_carrier.take().unwrap();
    commands.insert_resource(Some(carrier));
}

struct GPUGeometryStore<T: Geometry> {
    gpu_geometries: HashMap<Handle<T>, T::GPUGeometry>,
    pending_builds: Option<(
        Vec<(Handle<T>, Option<T::GPUGeometry>)>,
        TimelineSemaphoreOp,
    )>,
    buffered_builds: Option<GeometryCarrier<T>>,
}

impl<T: Geometry> Default for GPUGeometryStore<T> {
    fn default() -> Self {
        Self {
            gpu_geometries: HashMap::new(),
            pending_builds: None,
            buffered_builds: None,
        }
    }
}

/// This runs in the Prepare stage of the Render world.
fn prepare_geometries<T: Geometry>(
    mut geometery_carrier: ResMut<Option<GeometryCarrier<T>>>,
    mut geometry_store: ResMut<GPUGeometryStore<T>>,
    queues: Res<Queues>,
) {
    // Merge the new changes into the buffer. Incoming -> Buffered
    if let Some(buffered_builds) = geometry_store.buffered_builds.as_mut() {
        if let Some(new_builds) = geometery_carrier.take() {
            buffered_builds.updates.extend(new_builds.updates);
            buffered_builds.removed.extend(new_builds.removed);
        }
    } else {
        geometry_store.buffered_builds = geometery_carrier.take();
    }

    // Pending -> Existing
    if let Some((mut carrier, signal)) = geometry_store.pending_builds.take() {
        if signal.finished().unwrap() {
            // Finished, put it into the store.
            for (handle, geometry) in carrier.drain(..) {
                if let Some(geometry) = geometry {
                    geometry_store.gpu_geometries.insert(handle, geometry);
                }
                // TODO: send rebuild BLAS signal.
            }
        } else {
            // put it back
            geometry_store.pending_builds = Some((carrier, signal));
            // Has pending work. return early
            return;
        }
    }
    assert!(geometry_store.pending_builds.is_none());

    let mut future =
        dustash::sync::CommandsFuture::new(&queues, queues.of_type(QueueType::Transfer).index());
    // Buffered -> Pending
    let mut pending_builds: Vec<(Handle<T>, Option<T::GPUGeometry>)> = Vec::new();
    if let Some(mut buffered_builds) = geometry_store.buffered_builds.take() {
        for handle in buffered_builds.removed.drain(..) {
            geometry_store.gpu_geometries.remove(&handle);
        }
        for (handle, update) in buffered_builds.updates.drain() {
            match update {
                GPUGeometryUpdate::Rebuild(build_set) => {
                    let geometry =
                        <T::GPUGeometry as GPUGeometry<T>>::build(build_set, &mut future);
                    pending_builds.push((handle, Some(geometry)));
                }
                GPUGeometryUpdate::Update(change_set) => {
                    geometry_store
                        .gpu_geometries
                        .get_mut(&handle)
                        .unwrap()
                        .apply_change_set(change_set, &mut future);
                    pending_builds.push((handle, None));
                }
            }
        }
        let signal = future
            .stage(vk::PipelineStageFlags2::ALL_COMMANDS)
            .then_signal();
        geometry_store.pending_builds = Some((pending_builds, signal));
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

impl<T: Geometry> Plugin for GeometryPlugin<T> {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_asset::<T>()
            .add_system_to_stage(CoreStage::PostUpdate, generate_changes::<T>);

        let render_app = app.sub_app_mut(crate::RenderApp);
        render_app
            .init_resource::<GPUGeometryStore<T>>()
            .add_system_to_stage(
                RenderStage::Extract,
                move_change_set_store_to_render_world::<T>,
            )
            .add_system_to_stage(RenderStage::Prepare, prepare_geometries::<T>);
    }
}
