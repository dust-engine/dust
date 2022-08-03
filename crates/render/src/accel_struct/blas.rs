use std::sync::Arc;

use bevy_app::Plugin;
use bevy_asset::HandleId;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_transform::prelude::GlobalTransform;
use bevy_utils::{HashMap, HashSet};
use dustash::{
    accel_struct::{build::AabbBlasBuilder, AccelerationStructure, AccelerationStructureLoader},
    queue::{QueueType, Queues},
    resources::alloc::{Allocator, MemBuffer},
    sync::CommandsFuture,
};

use crate::{material::GPUGeometryMaterial, renderable::Renderable, RenderStage};
use ash::vk;
use bevy_hierarchy::{BuildChildren, Children};

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
    query: Query<(Entity, &Renderable, &GlobalTransform, Option<&Children>)>,
) {
    for (entity, renderable, transform, children) in query.iter() {
        let mut item = commands.get_or_spawn(entity);
        item.insert(renderable.clone()).insert(transform.clone());
        if let Some(children) = children {
            item.insert_children(0, children);
        }
    }
}

#[derive(Default)]
struct BlasStore {
    /// Mapping from a list of GeometryIds to the acceleration structure
    blas: HashMap<Vec<HandleId>, Arc<AccelerationStructure>>,
}

#[derive(Component)]
pub struct BlasComponent {
    // Things in here are guaranteed to be non-null.
    pub geometry_materials: Vec<GPUGeometryMaterial>,
    pub blas: Arc<AccelerationStructure>,
}

/// Collect the list of active geometries for each root.
fn build_blas(
    mut commands: Commands,
    mut blas_store: ResMut<BlasStore>,
    loader: Res<Arc<AccelerationStructureLoader>>,
    allocator: Res<Arc<Allocator>>,
    queues: Res<Arc<Queues>>,
    query: Query<(
        Entity,
        &Renderable,
        Option<&Children>,
        Option<&GPUGeometryMaterial>,
    )>,
    children_query: Query<(Entity, &GPUGeometryMaterial)>,
) {
    fn collect_geometry_material(
        children: &Children,
        geometry_materials: &mut Vec<GPUGeometryMaterial>,
        children_query: &Query<(Entity, &GPUGeometryMaterial)>,
    ) {
        for child in children.iter() {
            let (_child_entity, geometry_material) = children_query.get(*child).unwrap();
            geometry_materials.push(geometry_material.clone())
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
    'outer: for (entity, _renderable, children, geometry_material_on_root) in query.iter() {
        let mut geometry_materials: Vec<GPUGeometryMaterial> = Vec::new();
        if let Some(geometry_material) = geometry_material_on_root {
            // Get the GPUGeometryPrimitives on the root
            geometry_materials.push(geometry_material.clone());
        }
        // Collect the GPUGeometryPrimitives on the childrens recursively
        if let Some(children) = children {
            collect_geometry_material(children, &mut geometry_materials, &children_query);
        }
        if geometry_materials.len() == 0 {
            continue;
        }

        geometry_materials.sort_by_key(|geometry_material| geometry_material.geometry_handle);
        let primitive_ids: Vec<_> = geometry_materials
            .iter()
            .map(|geometry_material| geometry_material.geometry_handle)
            .collect();
        if let Some(blas) = blas_store.blas.get(&primitive_ids) {
            retained_blas.insert(primitive_ids.clone(), blas.clone());
            commands.get_or_spawn(entity).insert(BlasComponent {
                geometry_materials,
                blas: blas.clone(),
            });
            // TODO: if the BLAS was invalidated, still rebuild the BLAS.
            // The BLAS is invalidated when the geometry was updated.
            // TODO: ensure that the BLAS build was completed. Otherwise, use the old one.
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
            
            let mut buffers: Vec<(Arc<MemBuffer>, std::alloc::Layout)> = Vec::with_capacity(geometry_materials.len());
            for geometry_material in geometry_materials.iter() {
                if let Some(buffer) = geometry_material.blas_input_primitives.as_ref() {
                    buffers.push((buffer.clone(), geometry_material.blas_input_layout))
                } else {
                    // Instance is not ready yet because of missing geometry data.
                    continue 'outer;
                }
    
                if geometry_material.sbt_data.is_none() {
                    continue 'outer;
                }
            }
            let mut builder = AabbBlasBuilder::new(
                vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
                    | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            );
            for (primitive_buffer, layout) in buffers {
                builder.add_geometry(primitive_buffer, vk::GeometryFlagsKHR::empty(), layout);
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
            CommandsFuture::new(queues.clone(), queues.of_type(QueueType::Compute).index());

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
    // BLASs in blas_store.blas MIGHT be still pending.
    blas_store.blas = retained_blas;
}
