use std::{alloc::Layout, collections::BTreeMap, ops::Deref};

use bevy::{
    asset::{AssetEvent, AssetId, Assets, Handle},
    ecs::{
        change_detection::DetectChangesMut,
        component::Component,
        entity::Entity,
        event::EventReader,
        query::{Added, Changed, Or, QueryItem, With},
        removal_detection::RemovedComponents,
        system::{lifetimeless::SRes, Commands, Local, Query, SystemParamItem},
    },
    math::Vec3A,
    transform::components::GlobalTransform,
    utils::tracing,
};
use dust_pbr::PbrPipeline;
use dust_vdb::Node;
use rhyolite::{ash::vk, Allocator};
use rhyolite_rtx::{BLASBuildGeometry, BLASBuilder, HitgroupHandle, TLASBuilder, BLAS};

use crate::{TreeRoot, VoxGeometry, VoxInstance, VoxMaterial, VoxModel, VoxPalette, VoxPaletteGPU};

/// BLAS builder that builds a BLAS for all entities with `VoxBLASBuilder` and `AssetId<VoxGeometry>` components.
/// Expects asset with `AssetId<VoxGeometry>` to be loaded at the time when the builder is run.
pub struct VoxBLASBuilder;

impl BLASBuilder for VoxBLASBuilder {
    type QueryData = &'static Handle<VoxGeometry>;

    type QueryFilter = ();

    type Params = SRes<Assets<VoxGeometry>>;

    
    type GeometryIterator<'a> = std::iter::Once<BLASBuildGeometry<vk::DeviceSize>>;
    fn geometries<'a>(
        assets: &'a mut SystemParamItem<Self::Params>,
        data: &'a QueryItem<Self::QueryData>,
        dst: &mut [u8],
    ) -> Self::GeometryIterator<'a> {
        let geometry = assets.get(**data).unwrap();

        let leaf_extent_int = <<TreeRoot as Node>::LeafType as Node>::EXTENT;
        let leaf_extent: Vec3A = leaf_extent_int.as_vec3a();
        let leaf_extent: Vec3A = geometry.unit_size * leaf_extent;
        let mut current_location = 0;
        let mut leaf_count = 0;
        for (position, _) in geometry.tree.iter_leaf() {
            leaf_count += 1;
            let aabb = {
                let position = position.as_vec3a();
                let max_position = leaf_extent + position;
                vk::AabbPositionsKHR {
                    min_x: position.x,
                    min_y: position.y,
                    min_z: position.z,
                    max_x: max_position.x,
                    max_y: max_position.y,
                    max_z: max_position.z,
                }
            };
            let dst_slice = &mut dst[current_location
                ..(current_location + std::mem::size_of::<vk::AabbPositionsKHR>())];
            dst_slice.copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &aabb as *const vk::AabbPositionsKHR as *const u8,
                    std::mem::size_of::<vk::AabbPositionsKHR>(),
                )
            });
            current_location += std::mem::size_of::<vk::AabbPositionsKHR>();
        }
        std::iter::once(BLASBuildGeometry::Aabbs {
            buffer: 0,
            stride: std::mem::size_of::<vk::AabbPositionsKHR>() as u64,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            primitive_count: leaf_count,
        })
    }
}

#[derive(Component)]
pub struct BLASRef(Entity);

/// Listen to asset events and spawn BLAS entities for each added asset.
pub(crate) fn sync_asset_events_system(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<VoxGeometry>>,
    mut query: Query<&mut VoxModel>,
    mut entity_map: Local<BTreeMap<AssetId<VoxGeometry>, Entity>>,
    changes: Query<
        (Entity, &AssetId<VoxGeometry>),
        (Or<(Added<BLAS>, Changed<BLAS>)>, With<VoxModel>),
    >,
    mut instance_blas_relations: Local<BTreeMap<Entity, AssetId<VoxGeometry>>>,
    mut instance_blas_relations_reverse: Local<BTreeMap<AssetId<VoxGeometry>, Vec<Entity>>>,
    instances: Query<
        (Entity, &Handle<VoxGeometry>),
        Or<(Added<Handle<VoxGeometry>>, Changed<Handle<VoxGeometry>>)>,
    >,
    mut instances_removal: RemovedComponents<Handle<VoxGeometry>>,
) {
    // Maintain the instance-model tables. (long term, rewrite this with entity relations)
    for removal in instances_removal.read() {
        let handle = instance_blas_relations.remove(&removal).unwrap();
        instance_blas_relations_reverse
            .get_mut(&handle)
            .unwrap()
            .retain(|&entity| entity != removal);
    }
    for (entity, handle) in instances.iter() {
        instance_blas_relations.insert(entity, handle.id());
        instance_blas_relations_reverse
            .entry(handle.id())
            .or_default()
            .push(entity);
    }

    // For each VoxModel, whenever the BLAS gets built, notify the instances and update BLASRef.
    for (entity, change) in changes.iter() {
        for instance in instance_blas_relations_reverse[change].iter() {
            commands.entity(*instance).insert(BLASRef(entity));
        }
    }
    
    // Create VoxModel entities using asset events.
    for event in events.read() {
        match event {
            AssetEvent::Added { id } => {
                tracing::info!("Adding new VoxGeometry Asset {:?}", id);
                let entity = commands.spawn((VoxModel, id.clone())).id();
                entity_map.insert(id.clone(), entity);
            }
            AssetEvent::Modified { id } => {
                tracing::info!("VoxGeometry Asset {:?} modified", id);
                let entity = entity_map.get(id).unwrap();
                if let Ok(mut marker) = query.get_mut(*entity) {
                    marker.set_changed();
                }
                // If we can't get the entity here, it's likely that the entity was just added
                // this frame.
            }
            AssetEvent::Unused { id } => {
                tracing::info!("VoxGeometry Asset {:?} unused", id);
                let entity = entity_map.remove(id).unwrap();
                commands.entity(entity).despawn();
            }
            _ => {}
        }
    }
}

pub struct VoxTLASBuilder;
impl TLASBuilder for VoxTLASBuilder {
    type Marker = VoxInstance;

    type QueryData = (&'static GlobalTransform, &'static BLASRef);

    type QueryFilter = ();

    type ChangeFilter = Changed<GlobalTransform>;

    type AddFilter = Added<BLASRef>;

    type Params = Query<'static, 'static, &'static BLAS>;

    fn instance(
        params: &mut SystemParamItem<Self::Params>,
        (transform, blas): &QueryItem<Self::QueryData>,
        mut dst: rhyolite_rtx::TLASInstanceData,
    ) {
        let blas = params.get(blas.0).unwrap();
        dst.set_transform(transform.compute_matrix());
        dst.set_blas(blas);
        dst.set_custom_index_and_mask(0, 0);
        dst.set_sbt_offset_and_flags(0, vk::GeometryInstanceFlagsKHR::empty());
    }
}


#[repr(C)]
pub struct ShaderParams {
    /// Pointer to a list of u64 indexed by block id
    geometry_ptr: u64,

    /// Pointer to a list of u8, indexed by voxel id, each denoting offset into palette_ptr.
    /// Voxel id is defined as block id + offset inside block.
    material_ptr: u64,

    /// Pointer to a list of 256 u8 colors
    palette_ptr: u64,
}

pub struct VoxSbtBuilder;
impl rhyolite_rtx::SBTBuilder for VoxSbtBuilder {

    type Marker = VoxModel;

    type QueryData = &'static AssetId<VoxGeometry>;

    type QueryFilter = ();

    type Params = (
        SRes<Assets<VoxMaterial>>,
        SRes<Assets<VoxGeometry>>,
        SRes<Assets<VoxPaletteGPU>>,
        SRes<PbrPipeline>,
        Local<'static, Option<HitgroupHandle>>,
    );

    fn hitgroup_param(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
        raytype: u32,
        ret: &mut Self::InlineParam,
    ) {
        ret.geometry_ptr = 1;
        ret.material_ptr = 2;
        ret.palette_ptr = 3;
    }

    fn hitgroup_handle(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> rhyolite_rtx::HitgroupHandle {
        todo!()
    }
    
    type AddFilter = (AssetLoaded<VoxModelGPU>, AssetLoaded<VoxPaletteGPU>, AssetLoaded<VoxGeometryGPU>, Without<SbtHandle>);
    
    type ChangeFilter = Or<(AssetChanged<VoxModelGPU, VoxPaletteGPU, VoxGeometryGPU>)>;
    
    type InlineParam = ShaderParams;
}
