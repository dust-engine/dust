use std::{alloc::Layout, collections::BTreeMap};

use bevy::{
    asset::{AssetEvent, AssetId, Assets, Handle},
    ecs::{
        change_detection::DetectChangesMut,
        component::Component,
        entity::Entity,
        event::EventReader,
        query::QueryItem,
        system::{lifetimeless::SRes, Commands, Local, Query, SystemParamItem},
    },
    math::Vec3A,
    utils::tracing,
};
use dust_vdb::Node;
use rhyolite::{ash::vk, Allocator};
use rhyolite_rtx::{BLASBuildGeometry, BLASBuildMarker, BLASStagingBuilder};

use crate::{TreeRoot, VoxGeometry, VoxInstance};

#[derive(Component)]
pub struct VoxBLASBuilder;

impl BLASBuildMarker for VoxBLASBuilder {
    type Marker = VoxBLASBuilder;

    type QueryData = &'static AssetId<VoxGeometry>;

    type QueryFilter = ();

    type Params = SRes<Assets<VoxGeometry>>;
}

impl BLASStagingBuilder for VoxBLASBuilder {
    fn staging_layout(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> Layout {
        let num_primitives = params.get(**data).unwrap().tree.iter_leaf().count();
        let (layout, stride) = Layout::new::<vk::AabbPositionsKHR>()
            .repeat(num_primitives)
            .unwrap();
        debug_assert_eq!(stride, std::mem::size_of::<vk::AabbPositionsKHR>());
        layout
    }
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

pub(crate) fn sync_asset_events_system(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<VoxGeometry>>,
    mut query: Query<&mut VoxBLASBuilder>,
    mut entity_map: Local<BTreeMap<AssetId<VoxGeometry>, Entity>>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id } => {
                tracing::info!("Adding new VoxGeometry Asset {:?}", id);
                let entity = commands.spawn((VoxBLASBuilder, id.clone())).id();
                entity_map.insert(id.clone(), entity);
            }
            AssetEvent::Modified { id } => {
                tracing::info!("VoxGeometry Asset {:?} modified", id);
                let entity = entity_map.get(id).unwrap();
                let mut marker = query.get_mut(*entity).unwrap();
                marker.set_changed();
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
