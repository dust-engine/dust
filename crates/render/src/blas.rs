use std::{process::Child, sync::Arc};

use bevy_app::Plugin;
use bevy_asset::HandleId;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_utils::{HashMap, HashSet};
use dustash::{
    accel_struct::{build::AabbBlasBuilder, AccelerationStructure, AccelerationStructureLoader},
    queue::{QueueType, Queues},
    resources::alloc::{Allocator, MemBuffer},
    sync::CommandsFuture,
};

use crate::{geometry::GPUGeometryPrimitives, renderable::Renderable, RenderStage};
use ash::vk;
use bevy_hierarchy::{Children, Parent};

/// This plugin generates a BLAS for each unique combination of geometries.
#[derive(Default)]
pub struct BlasPlugin;

impl Plugin for BlasPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_system_to_stage(RenderStage::Extract, extract_renderable)
            .add_system_to_stage(RenderStage::Build, build_blas)
            .init_resource::<BlasStore>();
    }
}

/// Copy the root element and their children to the render world.
fn extract_renderable(
    mut commands: Commands,
    query: Query<(Entity, &Renderable, Option<&Children>)>,
) {
    for (entity, renderable, children) in query.iter() {
        let mut item = commands.get_or_spawn(entity);
        item.insert(renderable.clone());
        if let Some(children) = children {
            item.insert(children.clone());
        }
    }
}

#[derive(Default)]
struct BlasStore {
    /// Mapping from a list of GeometryIds to the acceleration structure
    blas: HashMap<Vec<HandleId>, Arc<AccelerationStructure>>,
}

#[derive(Component)]
struct BlasComponent {
    blas: Arc<AccelerationStructure>,
}

/// Collect the list of active geometries for each root.
fn build_blas(
    mut commands: Commands,
    mut blas_store: ResMut<BlasStore>,
    loader: Res<Arc<AccelerationStructureLoader>>,
    allocator: Res<Arc<Allocator>>,
    queues: Res<Queues>,
    query: Query<(
        Entity,
        &Renderable,
        Option<&Children>,
        Option<&GPUGeometryPrimitives>,
    )>,
    children_query: Query<(Entity, Option<&Children>, Option<&GPUGeometryPrimitives>)>,
) {
    fn collect_primitive_ids(
        children: &Children,
        primitive_ids: &mut Vec<HandleId>,
        buffers: &mut Vec<Arc<MemBuffer>>,
        children_query: &Query<(Entity, Option<&Children>, Option<&GPUGeometryPrimitives>)>,
    ) {
        for child in children.iter() {
            let (child_entity, children, primitives) = children_query.get(*child).unwrap();
            if let Some(primitives) = primitives {
                if let Some(blas_input_primitives) = primitives.blas_input_primitives.as_ref() {
                    primitive_ids.push(primitives.handle);
                    buffers.push(blas_input_primitives.clone());
                } else {
                    // Skip if the geometry hasn't been fully loaded
                    primitive_ids.clear();
                    buffers.clear();
                    return;
                }
            }
            if let Some(children) = children {
                collect_primitive_ids(children, primitive_ids, buffers, children_query);
            }
        }
    }
    let mut blas_builder = dustash::accel_struct::build::AccelerationStructureBuilder::new(
        loader.clone(),
        allocator.clone(),
    );
    let mut pending_builds: Vec<Vec<HandleId>> = Vec::new();
    let mut pending_builds_set: HashSet<Vec<HandleId>> = HashSet::new();

    // BLASs that are still needed next frame
    let mut retained_blas: HashMap<Vec<HandleId>, Arc<AccelerationStructure>> = HashMap::new();
    // For all root elements
    for (entity, renderable, children, primitives) in query.iter() {
        let mut primitive_ids: Vec<HandleId> = Vec::new();
        let mut buffers: Vec<Arc<MemBuffer>> = Vec::new();
        if let Some(primitives) = primitives {
            // Get the GPUGeometryPrimitives on the root
            if let Some(blas_input_primitives) = primitives.blas_input_primitives.as_ref() {
                primitive_ids.push(primitives.handle);
                buffers.push(blas_input_primitives.clone());
            } else {
                // Skip if the geometry hasn't been fully loaded
                continue;
            }
        }
        // Collect the GPUGeometryPrimitives on the childrens recursively
        if let Some(children) = children {
            collect_primitive_ids(children, &mut primitive_ids, &mut buffers, &children_query);
        }
        if primitive_ids.len() == 0 {
            continue;
        }
        primitive_ids.sort();
        if let Some(blas) = blas_store.blas.get(&primitive_ids) {
            retained_blas.insert(primitive_ids.clone(), blas.clone());
            commands
                .get_or_spawn(entity)
                .insert(BlasComponent { blas: blas.clone() });
            // TODO: if the BLAS was invalidated, still rebuild the BLAS.
            // The BLAS is invalidated when the geometry was updated.
        } else {
            // Build the BLAS
            if pending_builds_set.contains(&primitive_ids) {
                // Skip: another version of this BLAS is already being built
                continue;
            }
            println!(
                "give me a blas with the following geometries: {:?}",
                primitive_ids
            );
            let mut builder = AabbBlasBuilder::new(
                vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
                    | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            );
            for geometry_primitive in buffers {
                builder.add_geometry::<()>(geometry_primitive, vk::GeometryFlagsKHR::empty());
                // TODO: add geometry flags
            }
            // Queue up the rebuild command.
            blas_builder.add_aabb_blas(builder);
            // Add to HashSet PendingBLASs and PendingBLASs array
            pending_builds.push(primitive_ids.clone());
            pending_builds_set.insert(primitive_ids);
        }
    }
    if pending_builds.len() > 0 {
        let mut commands_future =
            CommandsFuture::new(&queues, queues.of_type(QueueType::Compute).index());

        let acceleration_structures = blas_builder.build(&mut commands_future);
        assert_eq!(acceleration_structures.len(), pending_builds.len());
        for (id, blas) in pending_builds
            .into_iter()
            .zip(acceleration_structures.into_iter())
        {
            let original = retained_blas.insert(id, blas);
            assert!(original.is_none());
        }
    }

    // Drop BLAS HashMap from previous frame and drop any unused BLASs
    blas_store.blas = retained_blas;
}
// 1. Renderable -> BLAS
// 2. GeometryID -> BLAS
