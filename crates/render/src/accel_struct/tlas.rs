use crate::pipeline::RenderSystem;
use crate::RenderStage;
use crate::{accel_struct::blas::BlasComponent, renderable::Renderable};
use crate::{AccelerationStructureLoader, Allocator, Queues};
use ash::vk;
use bevy_asset::HandleId;
use bevy_ecs::prelude::Query;
use bevy_ecs::prelude::*;
use bevy_transform::prelude::GlobalTransform;
use bevy_utils::HashMap;
use dustash::accel_struct::AccelerationStructure;
use dustash::{queue::QueueType, sync::CommandsFuture};
use std::sync::Arc;

#[derive(Default)]
pub struct TlasPlugin;
impl bevy_app::Plugin for TlasPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<TLASStore>()
            .add_system_to_stage(RenderStage::Render, build_tlas.before(RenderSystem));
    }
}

#[derive(Default, Resource)]
pub struct TLASStore {
    pub tlas: Option<Arc<AccelerationStructure>>,
    pub tlas_build_future: Option<CommandsFuture>,
}

#[derive(Eq, Hash, PartialEq, Clone)]
struct InstanceEntry {
    geometries: Vec<(HandleId, HandleId)>,
}

/// Build a TLAS.
/// This runs in the Render stage of the render world.
fn build_tlas(
    mut store: ResMut<TLASStore>,
    allocator: Res<Allocator>,
    accel_struct_loader: Res<AccelerationStructureLoader>,
    queues: Res<Queues>,
    query: Query<(&GlobalTransform, &BlasComponent, &Renderable)>,
) {
    let mut map: HashMap<InstanceEntry, u32> = Default::default();
    let mut current_sbt_offset: u32 = 0;
    // TODO: skip recreating TLAS when there's no change.
    // TODO: View frustrum culling
    let instances: Vec<_> = query
        .iter()
        .map(|(global_transform, blas, renderable)| {
            let mut transform = vk::TransformMatrixKHR { matrix: [0.0; 12] };
            transform.matrix.clone_from_slice(
                &global_transform
                    .compute_matrix()
                    .transpose()
                    .to_cols_array()[0..12],
            );

            // Deduplicate by geometry and material
            let entry = InstanceEntry {
                geometries: blas
                    .geometry_materials
                    .iter()
                    .map(|geometry_material| {
                        (
                            geometry_material.geometry_handle,
                            geometry_material.material_handle,
                        )
                    })
                    .collect(),
            };
            let sbt_offset = if let Some(sbt_offset) = map.get(&entry) {
                *sbt_offset
            } else {
                let sbt_offset = current_sbt_offset;
                current_sbt_offset += blas.geometry_materials.len() as u32;

                map.insert(entry, sbt_offset);
                sbt_offset
            };
            // We only have 24 bits for the SBT offset.
            assert!(sbt_offset < 1 << 24);

            vk::AccelerationStructureInstanceKHR {
                // a 3x4 row-major affine transformation matrix.
                transform,
                // instance custom index. mask is always 0xFF for now.
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
                // instance sbt record offset.
                // SBT record contains:
                // - Shader handles, and therefore geometry + material combination
                // - Geometry index
                // - Material index
                // Two different BLAS should never alias SBT entry, because they're going to have different geometry index.
                // In fact we can have more than 1 entry for the same BLAS, becauce different instance can have same geometry but different textures.
                // What geometry this instance is, what material and geometry id this is.
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    sbt_offset, // actually this is wrong. sbt offset also needs to consider geometry indices from previous instances
                    renderable.flags.bits(),
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.blas.device_address(),
                },
            }
        })
        .collect();
    if instances.is_empty() {
        return;
    }

    let mut commands_future =
        CommandsFuture::new(queues.clone(), queues.of_type(QueueType::Compute).index());

    let tlas = AccelerationStructure::make_tlas(
        accel_struct_loader.clone(),
        &allocator,
        &instances,
        &mut commands_future,
    );
    store.tlas = Some(tlas);
    store.tlas_build_future = Some(commands_future);
}
