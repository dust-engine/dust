//! Each model's emitter table: its emissive voxel faces, grouped for light
//! sampling (`dust_pbr::emitters`), kept on the `VoxModel` entity.
//!
//! A table depends on all three parts of the model: the geometry (which faces
//! are exposed), the material (each voxel's palette index) and the palette
//! (each index's emission). It is built on the CPU when the model first has all
//! three, and again whenever its BLAS rebuilds or its material or palette
//! changes. Each frame, every instance of a model with emitters is listed in
//! `dust_pbr`'s [`EmitterInstances`] for the SHARC Update to sample.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use bevy::asset::{AssetEvent, AssetId, Assets};
use bevy::ecs::change_detection::Ref;
use bevy::ecs::entity::Entity;
use bevy::ecs::message::MessageReader;
use bevy::ecs::system::{Commands, Query, Res, ResMut};
use bevy::log::debug;
use bevy::math::{IVec3, UVec3, Vec3};
use bevy::prelude::{Component, DetectChanges, GlobalTransform};
use bevy_pumicite::SubmissionState;
use bevy_pumicite::rtx::tlas::TLASInstance;
use dust_pbr::PbrInstanceData;
use dust_pbr::emitters::{
    EmissiveFace, EmitterInstance, EmitterInstances, EmitterTable, EmitterTableHeader,
};
use dust_vdb::TreeLike;
use pumicite::Allocator;
use pumicite::ash::vk;
use pumicite::buffer::ManagedBuffer;
use pumicite::command::GPURefMut;

use crate::{VoxGeometry, VoxGpuMaterial, VoxMaterial, VoxModel, VoxModelBLASRebuild, VoxPalette};

/// The way each face direction points, numbered as SHARC's face ids.
const DIRECTIONS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// A model's emitter table, uploaded for the GPU.
#[derive(Component)]
pub struct VoxModelEmitters {
    /// The table's buffer ([`EmitterTable`] layout), read by device address.
    /// `None` when the model has no emissive faces.
    pub buffer: Option<Arc<ManagedBuffer>>,
    /// The table's header, as written at the start of `buffer`.
    pub header: EmitterTableHeader,
}

/// The emissive faces among `voxels`: every face of an emissive voxel whose
/// neighbor across it is empty, or outside the model. `palette_index` gives a
/// voxel's palette index, or `None` when it's empty; `extent` bounds the model.
/// The radiance of a face is its palette color, still sRGB-encoded, times the
/// entry's emission, as the shaders take it.
pub fn emissive_faces(
    voxels: impl IntoIterator<Item = UVec3>,
    mut palette_index: impl FnMut(UVec3) -> Option<u8>,
    palette: &[VoxGpuMaterial; 256],
    extent: UVec3,
) -> Vec<EmissiveFace> {
    let radiance: Vec<Option<Vec3>> = palette
        .iter()
        .map(|entry| {
            let emission = entry.emission as f32;
            let color = entry.color.truncate().as_vec3() / 255.0;
            (emission > 0.0 && color != Vec3::ZERO).then_some(color * emission)
        })
        .collect();
    if radiance.iter().all(Option::is_none) {
        return Vec::new();
    }

    let mut faces = Vec::new();
    for coords in voxels {
        let Some(index) = palette_index(coords) else {
            continue;
        };
        let Some(radiance) = radiance[index as usize] else {
            continue;
        };
        for (direction, step) in DIRECTIONS.iter().enumerate() {
            let neighbor = coords.as_ivec3() + step;
            let inside = neighbor.cmpge(IVec3::ZERO).all() && neighbor.as_uvec3().cmplt(extent).all();
            if inside && palette_index(neighbor.as_uvec3()).is_some() {
                continue;
            }
            faces.push(EmissiveFace {
                coords,
                direction: direction as u8,
                radiance,
            });
        }
    }
    faces
}

/// The emitter table of a model from its three parts.
pub fn model_emitter_table(
    geometry: &VoxGeometry,
    material: &VoxMaterial,
    palette: &VoxPalette,
) -> EmitterTable {
    let mut accessor = geometry.tree.accessor(material);
    let extent = UVec3::splat(256);
    let faces = emissive_faces(
        geometry.tree.iter(),
        |coords| accessor.get(coords),
        palette.entries(),
        extent,
    );
    EmitterTable::build(geometry.unit_size, &faces)
}

/// Builds the emitter table of every model that needs one, uploads it and
/// stores it on the model.
pub(crate) fn build_emitter_tables(
    mut commands: Commands,
    mut ctx: SubmissionState,
    allocator: Res<Allocator>,
    models: Query<(Entity, &VoxModel, Ref<VoxModelBLASRebuild>, Option<&VoxModelEmitters>)>,
    geometries: Res<Assets<VoxGeometry>>,
    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>,
    mut material_events: MessageReader<AssetEvent<VoxMaterial>>,
    mut palette_events: MessageReader<AssetEvent<VoxPalette>>,
) {
    let modified_materials: HashSet<AssetId<VoxMaterial>> =
        material_events.read().filter_map(modified_id).collect();
    let modified_palettes: HashSet<AssetId<VoxPalette>> =
        palette_events.read().filter_map(modified_id).collect();

    let start = Instant::now();
    let mut built = Vec::new();
    for (entity, model, rebuild, emitters) in &models {
        let stale = emitters.is_none()
            || rebuild.is_changed()
            || modified_materials.contains(&model.material.id())
            || modified_palettes.contains(&model.palette.id());
        if !stale {
            continue;
        }
        let (Some(geometry), Some(material), Some(palette)) = (
            geometries.get(&model.geometry),
            materials.get(&model.material),
            palettes.get(&model.palette),
        ) else {
            // Not loaded yet. Drop the old table so the model is retried until
            // its parts are in.
            if emitters.is_some() {
                commands.entity(entity).remove::<VoxModelEmitters>();
            }
            continue;
        };
        let table = model_emitter_table(geometry, material, palette);
        let buffer = (!table.is_empty()).then(|| {
            let mut buffer = ManagedBuffer::new(
                allocator.clone(),
                table.byte_size() as u64,
                16,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .expect("emitter table buffer");
            table.write_to(buffer.as_slice_mut());
            Arc::new(buffer)
        });
        built.push((entity, table, buffer));
    }
    if built.is_empty() {
        return;
    }

    ctx.record(|encoder, _| {
        for (_, _, buffer) in &built {
            if let Some(buffer) = buffer {
                // Safety: the buffer was created this frame and nothing else
                // holds it, as for the palettes and materials in
                // `sync_buffers_system`.
                unsafe {
                    GPURefMut::new_unchecked(&**buffer).flush(encoder);
                }
            }
        }
    });

    let (mut faces, mut clusters) = (0, 0);
    let count = built.len();
    for (entity, table, buffer) in built {
        faces += table.header.face_count;
        clusters += table.header.cluster_count;
        commands.entity(entity).insert(VoxModelEmitters {
            buffer,
            header: table.header,
        });
    }
    debug!(
        "Built {count} emitter tables: {faces} faces in {clusters} clusters, in {:.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

/// Lists this frame's instances of models with emitters, where they are now.
pub(crate) fn collect_emitter_instances(
    mut emitter_instances: ResMut<EmitterInstances>,
    instances: Query<(Entity, &TLASInstance<PbrInstanceData>, &GlobalTransform)>,
    models: Query<&VoxModelEmitters>,
) {
    let EmitterInstances { instances: list, tables } = &mut *emitter_instances;
    list.clear();
    tables.clear();
    // In entity order, so an instance keeps its index from frame to frame while
    // the set doesn't change: the SHARC cells' reservoirs name their samples by
    // it.
    let mut ordered: Vec<_> = instances.iter().collect();
    ordered.sort_by_key(|&(entity, ..)| entity);
    for (_, instance, transform) in ordered {
        // A disabled instance isn't in the TLAS, so it isn't drawn and mustn't
        // light anything either.
        if instance.disabled {
            continue;
        }
        let Ok(VoxModelEmitters {
            buffer: Some(buffer),
            header,
        }) = models.get(instance.blas)
        else {
            continue;
        };
        list.push(EmitterInstance::new(
            transform.affine(),
            buffer.device_address(),
            header,
        ));
        tables.push(buffer.clone());
    }
}

fn modified_id<A: bevy::asset::Asset>(event: &AssetEvent<A>) -> Option<AssetId<A>> {
    match event {
        AssetEvent::Modified { id } => Some(*id),
        _ => None,
    }
}
