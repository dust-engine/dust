use std::{borrow::Borrow, future::Future, marker::PhantomData, pin::Pin, sync::Arc};

use crate::{shader::Shader, RenderStage, RenderWorld};
use ash::{prelude::VkResult, vk};
use bevy_app::{CoreStage, Plugin};
use bevy_asset::{AddAsset, Asset, AssetEvent, AssetServer, Assets, Handle, HandleUntyped};
use bevy_ecs::{
    event::EventReader,
    system::{Commands, IntoExclusiveSystem, Res, ResMut},
    world::World,
};
use bevy_utils::{HashMap, HashSet};
use dustash::{
    accel_struct::AccelerationStructure,
    command::{pool::CommandPool, recorder::CommandRecorder},
    queue::{Queue, QueueType, Queues},
    ray_tracing::sbt::SpecializationInfo,
};
use futures_lite::future;

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

    fn generate_builds(&mut self) -> Self::BuildSet;
    fn emit_changes(&mut self) -> Self::ChangeSet;
}

/// RenderWorld Assets.
pub trait GPUGeometry<T: Geometry>: Send + Sync {
    fn build(build_set: T::BuildSet) -> Self;
    fn apply_change_set(&mut self, change_set: T::ChangeSet);
}

/// Structure that moves geometry data from AppWorld to RenderWorld.
struct GeometryCarrier<T: Geometry> {
    builds: Vec<(Handle<T>, T::BuildSet)>,
    updates: Vec<(Handle<T>, T::ChangeSet)>,
    removed: Vec<Handle<T>>,
}
impl<T: Geometry> Default for GeometryCarrier<T> {
    fn default() -> Self {
        Self {
            builds: Vec::new(),
            updates: Vec::new(),
            removed: Vec::new(),
        }
    }
}

/// This runs in the main world in the PostUpdate stage.
fn generate_changes<A: Geometry>(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<A>>,
    mut geometries: ResMut<Assets<A>>,
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

    let mut builds = Vec::new();
    let mut updates = Vec::new();
    for (handle, rebuild) in changed_assets.drain() {
        let geometry = geometries.get_mut_untracked(&handle).unwrap();
        if rebuild {
            builds.push((handle, geometry.generate_builds()));
        } else {
            updates.push((handle, geometry.emit_changes()));
        }
    }

    // Insert an Option<GeometryCarrier> so that it can be taken.
    commands.insert_resource(Some(GeometryCarrier {
        updates,
        builds,
        removed,
    }));
}

/// This runs in the Extract stage of the Render World.
fn move_change_set_store_to_render_world<T: Geometry>(
    mut commands: Commands,
    mut geometery_carrier: ResMut<Option<GeometryCarrier<T>>>,
) {
    let carrier = geometery_carrier.take().unwrap();
    commands.insert_resource(carrier);
}

struct GPUGeometryStore<T: Geometry> {
    gpu_geometries: HashMap<Handle<T>, T::GPUGeometry>,
}

impl<T: Geometry> Default for GPUGeometryStore<T> {
    fn default() -> Self {
        Self {
            gpu_geometries: HashMap::new(),
        }
    }
}

/// This runs in the Prepare stage of the Render world.
fn prepare_geometries<T: Geometry>(
    mut geometery_carrier: ResMut<GeometryCarrier<T>>,
    mut geometry_store: ResMut<GPUGeometryStore<T>>,
) {
    use std::mem::take;
    for handle in take(&mut geometery_carrier.removed) {
        geometry_store.gpu_geometries.remove(&handle);
    }
    for (handle, build_set) in take(&mut geometery_carrier.builds).into_iter() {
        let geometry = <T::GPUGeometry as GPUGeometry<T>>::build(build_set);
        geometry_store.gpu_geometries.insert(handle, geometry);
    }
    for (handle, change_set) in take(&mut geometery_carrier.updates).into_iter() {
        geometry_store
            .gpu_geometries
            .get_mut(&handle)
            .unwrap()
            .apply_change_set(change_set);
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
