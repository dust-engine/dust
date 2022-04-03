use std::{future::Future, marker::PhantomData, pin::Pin, sync::Arc};

use crate::{accel_struct::AccelerationStructureStore, shader::Shader, RenderStage, RenderWorld};
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
pub trait Geometry: Asset {
    /// The geometry represented as an array of primitives
    /// This gets persisted in the render world.
    /// This is a GPU state.
    type Primitives: GeometryPrimitiveArray;

    /// The change in geometry.
    /// This gets extracted from the main world into the render world each frame.
    type ChangeSet: GeometryChangeSet<Self::Primitives>;

    fn aabb(&self) -> GeometryAABB;
    fn intersection_shader(asset_server: &AssetServer) -> Handle<Shader>;
    fn specialization() -> SpecializationInfo;

    /// This gets called in the render world. the produced ChangeSet will be saved as a component in the main world.
    fn generate_changes(&self) -> Self::ChangeSet;
}

// The representation of the geometry in the render world
pub trait GeometryPrimitiveArray: Send + Sync + 'static {
    /// CommandRecorder is a command recorder on the compute queue.
    fn rebuild_blas(
        &self,
        command_recorder: &mut CommandRecorder,
    ) -> dustash::accel_struct::AccelerationStructure;
}

pub trait GeometryChangeSet<T: GeometryPrimitiveArray>: Send + Sync + 'static {
    type Param: bevy_ecs::system::SystemParam;
    fn into_primitives(
        self,
        command_recorder: &mut CommandRecorder,
        params: &mut bevy_ecs::system::SystemParamItem<Self::Param>,
    ) -> (T, Vec<GeometryBLASBuildDependency>);

    /// Returns should_rebuild_blas
    /// Return true if:
    /// - The number of active primitives in the list changed
    /// - The location changed for at least one of the primitives.
    /// Copy data in the ChangeSet into the Primitives.
    fn apply_on(
        self,
        primitives: &mut T,
        command_recorder: &mut CommandRecorder,
        params: &mut bevy_ecs::system::SystemParamItem<Self::Param>,
    ) -> Option<Vec<GeometryBLASBuildDependency>>;
}
/// Buffer regions that will be used later during rebuild_blas
pub struct GeometryBLASBuildDependency {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub src_access_mask: vk::AccessFlags,
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
        app.init_resource::<ChangeSetStore<T>>()
            .add_asset::<T>()
            .add_system_to_stage(CoreStage::PostUpdate, generate_changes::<T>);

        let render_app = app.sub_app_mut(crate::RenderApp);
        render_app
            .init_resource::<ChangeSetStore<T>>()
            .init_resource::<PrimitiveStore<T>>()
            .add_system_to_stage(
                RenderStage::Extract,
                move_change_set_store_to_render_world::<T>,
            )
            .add_system_to_stage(RenderStage::Prepare, apply_changes::<T>);
    }
}

struct ChangeSetStore<T: Geometry> {
    changes: Vec<(Handle<T>, T::ChangeSet)>,
    removed: Vec<Handle<T>>,
}
impl<T: Geometry> Default for ChangeSetStore<T> {
    fn default() -> Self {
        Self {
            changes: Vec::new(),
            removed: Vec::new(),
        }
    }
}

// This is run in the main world
fn generate_changes<A: Geometry>(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<A>>,
    geometries: Res<Assets<A>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } => {
                changed_assets.insert(handle);
            }
            AssetEvent::Modified { handle } => {
                changed_assets.insert(handle);
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }

    let mut changsets = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(geometry) = geometries.get(handle) {
            changsets.push((handle.clone_weak(), geometry.generate_changes()));
        }
    }

    commands.insert_resource(Some(ChangeSetStore {
        changes: changsets,
        removed,
    }));
}

fn move_change_set_store_to_render_world<T: Geometry>(
    mut commands: Commands,
    mut change_set_store: ResMut<Option<ChangeSetStore<T>>>,
) {
    if let Some(change_set_store) = change_set_store.take() {
        commands.insert_resource(change_set_store)
    }
}

// One for each geometry
struct PrimitiveStore<T: Geometry> {
    primitives: HashMap<Handle<T>, T::Primitives>,
}
impl<T: Geometry> Default for PrimitiveStore<T> {
    fn default() -> Self {
        Self {
            primitives: HashMap::new(),
        }
    }
}

/// Runs in the prepare stage of the render world.
fn apply_changes<T: Geometry>(
    mut change_set_store: ResMut<ChangeSetStore<T>>,
    mut primitive_store: ResMut<PrimitiveStore<T>>,
    mut accel_struct_store: ResMut<AccelerationStructureStore>,
    queues: Res<Queues>,
    device: Res<Arc<dustash::Device>>,
    param: bevy_ecs::system::StaticSystemParam<
        <<T as Geometry>::ChangeSet as GeometryChangeSet<T::Primitives>>::Param,
    >,
) {
    let mut param = param.into_inner();

    for removed in std::mem::take(&mut change_set_store.removed) {
        // Each frame, for geometries that are removed, remove the corresponding primitive list as well
        primitive_store.primitives.remove(&removed);
    }

    if change_set_store.changes.len() == 0 {
        return;
    }

    if let Some(fut) = accel_struct_store.accel_structs_build_completion.as_mut() {
        if let Some(result) = future::block_on(future::poll_once(fut)) {
            // finished
            result.unwrap();
            let accel_struct_store = &mut *accel_struct_store;
            accel_struct_store.accel_structs_build_completion = None;
            accel_struct_store
                .accel_structs
                .extend(accel_struct_store.pending_accel_structs.drain());
        }
    } else {
        assert_eq!(accel_struct_store.pending_accel_structs.len(), 0);
        assert_eq!(accel_struct_store.queued_accel_structs.len(), 0);
    }

    let transfer_buf = accel_struct_store.transfer_pool.allocate_one().unwrap();
    let needs_ownership_transfer = queues.of_type(QueueType::Transfer).family_index()
        != queues.of_type(QueueType::Compute).family_index();
    let mut blas_to_build: Vec<Handle<T>> = Vec::new();
    let mut buffer_dependencies: Vec<vk::BufferMemoryBarrier> = Vec::new();

    let transfer_exec = transfer_buf
        .record(
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            |transfer_cr| {
                for (handle, change_set) in std::mem::take(&mut change_set_store.changes) {
                    // Each frame, for changed geometries
                    let prev_prepared_asset = primitive_store.primitives.remove(&handle);
                    let new_prepared_asset = if let Some(mut prev_prepared_asset) =
                        prev_prepared_asset
                    {
                        // If the primitive list was previously generated, apply the change set on the primitive list
                        let rebuild_blas =
                            change_set.apply_on(&mut prev_prepared_asset, transfer_cr, &mut param);
                        if let Some(deps) = rebuild_blas {
                            blas_to_build.push(handle.clone());
                            buffer_dependencies.extend(deps.iter().map(|deps| {
                                vk::BufferMemoryBarrier {
                                    src_access_mask: deps.src_access_mask,
                                    dst_access_mask:
                                        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR, // TODO: do things other than AS builds?
                                    src_queue_family_index: queues
                                        .of_type(QueueType::Transfer)
                                        .family_index(),
                                    dst_queue_family_index: queues
                                        .of_type(QueueType::Compute)
                                        .family_index(),
                                    buffer: deps.buffer,
                                    offset: deps.offset,
                                    size: deps.size,
                                    ..Default::default()
                                }
                            }));
                        }
                        prev_prepared_asset
                    } else {
                        // Otherwise, generate the primitive list from scratch
                        let (primitives, deps) =
                            change_set.into_primitives(transfer_cr, &mut param);

                        // Here, always rebuild BLAS
                        blas_to_build.push(handle.clone());
                        buffer_dependencies.extend(deps.iter().map(|deps| {
                            vk::BufferMemoryBarrier {
                                src_access_mask: deps.src_access_mask,
                                dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR, // TODO: do things other than AS builds?
                                src_queue_family_index: queues
                                    .of_type(QueueType::Transfer)
                                    .family_index(),
                                dst_queue_family_index: queues
                                    .of_type(QueueType::Compute)
                                    .family_index(),
                                buffer: deps.buffer,
                                offset: deps.offset,
                                size: deps.size,
                                ..Default::default()
                            }
                        }));

                        primitives
                    };
                    primitive_store
                        .primitives
                        .insert(handle, new_prepared_asset);
                }

                if needs_ownership_transfer {
                    transfer_cr.pipeline_barrier(
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                        vk::DependencyFlags::BY_REGION,
                        &[],
                        &buffer_dependencies,
                        &[],
                    );
                } else {
                    let src_access_mask = buffer_dependencies
                        .iter()
                        .fold(vk::AccessFlags::empty(), |mask, deps| {
                            deps.src_access_mask | mask
                        });
                    transfer_cr.pipeline_barrier(
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                        vk::DependencyFlags::BY_REGION,
                        &[vk::MemoryBarrier {
                            src_access_mask,
                            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                            // TODO: Maybe some geometry wants to do more than cmd_build_acceleration_structure in build_blas?
                            ..Default::default()
                        }],
                        &[],
                        &[],
                    );
                }

                if needs_ownership_transfer {
                    return;
                }

                for build in blas_to_build.iter() {
                    let untyped_handle = build.clone_weak_untyped();
                    if accel_struct_store
                        .queued_accel_structs
                        .contains(&untyped_handle)
                    {
                        return;
                    }
                    if accel_struct_store
                        .pending_accel_structs
                        .contains_key(&untyped_handle)
                    {
                        accel_struct_store
                            .queued_accel_structs
                            .insert(untyped_handle);
                        return;
                    }

                    let s = primitive_store.primitives.get(build).unwrap();
                    let blas = s.rebuild_blas(transfer_cr);
                    accel_struct_store
                        .pending_accel_structs
                        .insert(untyped_handle, Arc::new(blas));
                }
            },
        )
        .unwrap();

    if needs_ownership_transfer {
        let compute_buf = accel_struct_store.compute_pool.allocate_one().unwrap();
        let exec = compute_buf
            .record(
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                |compute_buf| {
                    compute_buf.pipeline_barrier(
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                        vk::DependencyFlags::BY_REGION,
                        &[],
                        &buffer_dependencies,
                        &[],
                    );
                    for build in blas_to_build.iter() {
                        let untyped_handle = build.clone_weak_untyped();
                        if accel_struct_store
                            .queued_accel_structs
                            .contains(&untyped_handle)
                        {
                            return;
                        }
                        if accel_struct_store
                            .pending_accel_structs
                            .contains_key(&untyped_handle)
                        {
                            accel_struct_store
                                .queued_accel_structs
                                .insert(untyped_handle);
                            return;
                        }

                        let s = primitive_store.primitives.get(build).unwrap();
                        let blas = s.rebuild_blas(compute_buf);
                        accel_struct_store
                            .pending_accel_structs
                            .insert(untyped_handle, Arc::new(blas));
                    }
                },
            )
            .unwrap();
        let mut timeline = dustash::queue::timeline::Timeline::new(device.clone()).unwrap();
        timeline.next(
            queues.of_type(QueueType::Transfer),
            Box::new([Arc::new(transfer_exec)]),
            vk::PipelineStageFlags2::empty(),
            vk::PipelineStageFlags2::TRANSFER,
        );
        timeline.next(
            queues.of_type(QueueType::Compute),
            Box::new([Arc::new(exec)]),
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags2::empty(),
        );
        accel_struct_store.accel_structs_build_completion = Some(Box::pin(timeline.finish()));
        // TODO:
        // 1. semaphores
        // 2. check completion state and push pending AS builds into completed AS builds
        // 3. Build SBT
        // 4. Build Pipeline
    } else {
        let mut timeline = dustash::queue::timeline::Timeline::new(device.clone()).unwrap();
        timeline.next(
            queues.of_type(QueueType::Transfer),
            Box::new([Arc::new(transfer_exec)]),
            vk::PipelineStageFlags2::empty(),
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        );
        accel_struct_store.accel_structs_build_completion = Some(Box::pin(timeline.finish()));
    }
}
