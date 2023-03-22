use std::{collections::{HashMap, HashSet}, sync::Arc, ops::{Deref, DerefMut}, alloc::Layout};

use bevy_asset::{AssetEvent, HandleUntyped, Handle, Assets};
use bevy_ecs::{system::{Query, Res, ResMut, Resource, Local, Commands}, prelude::{Events, EventReader, Entity, Component}, query::{Added, Without}, component::TableStorage};
use bevy_hierarchy::Children;
use bevy_tasks::{Task, IoTaskPool};
use rhyolite::{BufferLike, future::{DynamicCommandFuture, GPUCommandFutureExt, join_vec, GPUCommandJoinVec, GPUCommandFuture, RenderRes}, QueuesRouter, QueueType, ResidentBuffer, ash::vk, accel_struct::{blas::AabbBlasBuilder, build::{AccelerationStructureBuild, AccelerationStructureBatchBuilder}, AccelerationStructure}};
use rhyolite_bevy::AsyncQueues;

use crate::{Renderable, geometry::Geometry};


#[derive(Resource, Default)]
pub struct BlasStore {
    /// Maintains relationship between Geometry handles and Entity.
    /// entities[asset_handle] are entities using 
    entities: HashMap<HandleUntyped, HashSet<Entity>>
}

pub struct NormalizedGeometryInner {
    buffer: Arc<ResidentBuffer>,
    flags: vk::GeometryFlagsKHR,
    layout: Layout
}

#[derive(Component)]
pub struct BLAS {
    blas: Arc<AccelerationStructure>
}

#[derive(Component)]
pub struct NormalizedGeometry(Option<NormalizedGeometryInner>);

pub(crate) fn geometry_normalize_system<G: Geometry>(
    mut commands: Commands,
    mut store: ResMut<BlasStore>,
    assets: Res<Assets<G>>,
    mut events: EventReader<AssetEvent<G>>,
    queues: Res<AsyncQueues>,
    new_geometry_handle_query: Query<(Entity, &Handle<G>), Added<Handle<G>>>,
    mut upload_job: Local<Option<Task<Vec<(Entity, Arc<ResidentBuffer>, vk::GeometryFlagsKHR, Layout)>>>>,
    mut modification_query: Query<(Entity, &mut NormalizedGeometry)>,
    queue_router: Res<rhyolite_bevy::QueuesRouter>,
) {
    if let Some(upload_job_task) = upload_job.as_mut() {
        if upload_job_task.is_finished() {
            let upload_job = upload_job.take().unwrap();
            let upload_job = futures_lite::future::block_on(upload_job);
            for (entity, buffer, flags, layout) in upload_job.into_iter() {
                if let Some(mut normalized_geometry) = modification_query.get_component_mut::<NormalizedGeometry>(entity).ok() {
                    assert!(normalized_geometry.0.is_none());
                    normalized_geometry.0 = Some(NormalizedGeometryInner {
                        buffer,
                        flags,
                        layout
                    });
                }
            }
        }
    }
    for (entity, handle) in new_geometry_handle_query.iter() {
        commands.entity(entity).insert(NormalizedGeometry(None));
        let entities = store.entities.entry(handle.clone_weak_untyped()).or_default();
        entities.insert(entity);
    }
    //TODO: remove detection

    let mut upload_futures = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                let Some(entities) = store.entities.get(&handle.clone_weak_untyped()) else {
                    // Asset was loaded but never added to any entity
                    continue;
                };
                for entity in entities.iter() {
                    let entity = *entity;
                    let asset = assets.get(handle).unwrap();
                    let flags = asset.geometry_flags();
                    let layout = asset.layout();
                    upload_futures.push(asset.blas_input_buffer().map(move |a| (entity, a, flags, layout)));
                }
            }
            AssetEvent::Removed { handle } => {
                store.entities.remove(&handle.clone_weak_untyped());
            },
        }
    }
    if upload_futures.len() == 0 {
        return;
    }
    let future = queues.submit(join_vec(upload_futures).schedule_on_queue(queue_router.of_type(QueueType::Transfer)), &mut Default::default());
    upload_job.replace(IoTaskPool::get().spawn(future));
}

pub(crate) fn build_blas_system(
    mut commands: Commands,
    mut root_query: Query<(Entity, &Renderable, Option<&Children>, Option<&mut NormalizedGeometry>)>,
    mut children_query: Query<(Entity, &mut NormalizedGeometry), Without<Renderable>>,
    allocator: Res<rhyolite_bevy::Allocator>,
    
    queues: Res<AsyncQueues>,
    queue_router: Res<rhyolite_bevy::QueuesRouter>,
    mut upload_job: Local<Option<Task<Vec<(Entity, AccelerationStructure)>>>>,
) {
    if let Some(upload_job_task) = upload_job.as_ref() {
        if upload_job_task.is_finished() {
            let upload_job = futures_lite::future::block_on(upload_job.take().unwrap());
            for (entity, accel_struct) in upload_job.into_iter() {
                commands.entity(entity).insert(BLAS {
                    blas: Arc::new(accel_struct)
                });
            }
        } else {
            return;
        }
    }
    let mut builds: Vec<(Entity, AccelerationStructureBuild)> = Vec::new();
    for (entity, renderable, children, mut normalized_geometry_on_root) in root_query.iter_mut() {
        if let Some(geometry) = normalized_geometry_on_root.as_ref() {
            if geometry.0.is_none() {
                continue;
            }
        }
        if let Some(children) = children {
            for child_entity in children.iter() {
                if children_query.get_component::<NormalizedGeometry>(*child_entity).ok().map(|a| a.0.is_none()).unwrap_or(false) {
                    continue;
                }
            }
        }
        let mut geometries: Vec<NormalizedGeometryInner> = Vec::new();
        if let Some(geometry) = normalized_geometry_on_root.as_mut() {
            // Get the GPUGeometryPrimitives on the root
            geometries.push(geometry.0.take().unwrap());
        }
        if let Some(children) = children {
            for child_entity in children.iter() {
                if let Some(mut geometry) = children_query.get_component_mut::<NormalizedGeometry>(*child_entity).ok() {
                    geometries.push(geometry.0.take().unwrap())
                }
            }
        }
        if geometries.len() == 0 {
            continue;
        }
        let mut blas_builder = AabbBlasBuilder::new(renderable.blas_build_flags);
        for geometry in geometries.into_iter() {
            blas_builder.add_geometry(geometry.buffer, geometry.flags, geometry.layout);
            // TODO: Deduplicate BLAS based on involved geometries
        }
        let build = blas_builder.build(allocator.clone().into_inner()).unwrap();
        builds.push((entity, build));
    }
    if builds.len() == 0 {
        return;
    }
    println!("Scheduled {} BLAS builds", builds.len());
    let batch_builder = AccelerationStructureBatchBuilder::new(allocator.clone().into_inner(), builds);
    
    let future = queues.submit(batch_builder.build().schedule_on_queue(queue_router.of_type(QueueType::Compute)), &mut Default::default());
    upload_job.replace(IoTaskPool::get().spawn(future));
}