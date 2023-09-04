use std::{
    alloc::Layout,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy_asset::{AssetEvent, Assets, Handle, UntypedAssetId, UntypedHandle};
use bevy_ecs::{
    prelude::{Component, Entity, EventReader},
    query::{Added, Without},
    system::{Commands, Local, ParamSet, Query, Res, ResMut, Resource},
};
use bevy_hierarchy::Children;
use bevy_tasks::{IoTaskPool, Task};
use rhyolite::{
    accel_struct::{
        blas::AabbBlasBuilder,
        build::{AccelerationStructureBatchBuilder, AccelerationStructureBuild},
        AccelerationStructure,
    },
    ash::vk,
    future::{join_vec, GPUCommandFutureExt},
    QueueType, ResidentBuffer,
};
use rhyolite_bevy::AsyncQueues;

use crate::{geometry::Geometry, Renderable};

#[derive(Resource, Default)]
pub struct BlasStore {
    /// Maintains relationship between Geometry handles and Entity.
    /// entities[asset_handle] are entities using
    entities: HashMap<UntypedAssetId, HashSet<Entity>>,
}

pub struct NormalizedGeometryInner {
    buffer: Arc<ResidentBuffer>,
    flags: vk::GeometryFlagsKHR,
    layout: Layout,
}

#[derive(Component)]
pub struct BLAS {
    pub blas: Option<Arc<AccelerationStructure>>,
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
    mut upload_job: Local<
        Option<Task<Vec<(Entity, Arc<ResidentBuffer>, vk::GeometryFlagsKHR, Layout)>>>,
    >,
    mut modification_query: Query<(Entity, &mut NormalizedGeometry)>,
    queue_router: Res<rhyolite_bevy::QueuesRouter>,
) {
    if let Some(upload_job_task) = upload_job.as_mut() {
        if upload_job_task.is_finished() {
            let upload_job = upload_job.take().unwrap();
            let upload_job = futures_lite::future::block_on(upload_job);
            for (entity, buffer, flags, layout) in upload_job.into_iter() {
                if let Some(mut normalized_geometry) = modification_query
                    .get_component_mut::<NormalizedGeometry>(entity)
                    .ok()
                {
                    assert!(normalized_geometry.0.is_none());
                    normalized_geometry.0 = Some(NormalizedGeometryInner {
                        buffer,
                        flags,
                        layout,
                    });
                }
            }
        }
    }
    for (entity, handle) in new_geometry_handle_query.iter() {
        commands.entity(entity).insert(NormalizedGeometry(None));
        let entities = store.entities.entry(handle.id().untyped()).or_default();
        entities.insert(entity);
    }
    //TODO: remove detection

    let mut upload_futures = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let Some(entities) = store.entities.get(&id.untyped()) else {
                    // Asset was loaded but never added to any entity
                    continue;
                };
                for entity in entities.iter() {
                    let entity = *entity;
                    let asset = assets.get(id.untyped()).unwrap();
                    let flags = asset.geometry_flags();
                    let layout = asset.layout();
                    upload_futures.push(
                        asset
                            .blas_input_buffer()
                            .map(move |a| (entity, a, flags, layout)),
                    );
                }
            }
            AssetEvent::Removed { id } => {
                store.entities.remove(&id.untyped());
            }
            _ => (),
        }
    }
    if upload_futures.len() == 0 {
        return;
    }
    let future = queues.submit(
        join_vec(upload_futures).schedule_on_queue(queue_router.of_type(QueueType::Transfer)),
        &mut Default::default(),
    );
    upload_job.replace(IoTaskPool::get().spawn(future));
}

pub(crate) fn build_blas_system(
    mut commands: Commands,
    mut root_query: ParamSet<(
        Query<(
            Entity,
            &Renderable,
            Option<&Children>,
            Option<&BLAS>,
            Option<&mut NormalizedGeometry>,
        )>,
        Query<(Entity, &mut BLAS)>,
    )>,
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
                if let Some(mut blas) = root_query.p1().get_component_mut::<BLAS>(entity).ok() {
                    blas.blas = Some(Arc::new(accel_struct))
                }
            }
        } else {
            return;
        }
    }
    let mut builds: Vec<(Entity, AccelerationStructureBuild)> = Vec::new();
    for (entity, renderable, children, blas, mut normalized_geometry_on_root) in
        root_query.p0().iter_mut()
    {
        if blas.is_none() {
            // If the entity doesn't already have a BLAS component, give it.
            commands.entity(entity).insert(BLAS { blas: None });
        }

        // If some normalized geometry isn't ready yet on the root, skip.
        if let Some(geometry) = normalized_geometry_on_root.as_ref() {
            if geometry.0.is_none() {
                continue;
            }
        }
        // If some normalized geometry isn't ready yet on one of the children, skip.
        if let Some(children) = children {
            for child_entity in children.iter() {
                if children_query
                    .get_component::<NormalizedGeometry>(*child_entity)
                    .ok()
                    .map(|a| a.0.is_none())
                    .unwrap_or(false)
                {
                    continue;
                }
            }
        }

        // Collect all the normalized geometry
        let mut geometries: Vec<NormalizedGeometryInner> = Vec::new();
        if let Some(geometry) = normalized_geometry_on_root.as_mut() {
            // Get the GPUGeometryPrimitives on the root
            geometries.push(geometry.0.take().unwrap());
        }
        if let Some(children) = children {
            for child_entity in children.iter() {
                if let Some(mut geometry) = children_query
                    .get_component_mut::<NormalizedGeometry>(*child_entity)
                    .ok()
                {
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
    let batch_builder =
        AccelerationStructureBatchBuilder::new(allocator.clone().into_inner(), builds);

    let future = queues.submit(
        batch_builder
            .build()
            .schedule_on_queue(queue_router.of_type(QueueType::Compute)),
        &mut Default::default(),
    );
    upload_job.replace(IoTaskPool::get().spawn(future));
}
