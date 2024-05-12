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
        system::{
            lifetimeless::{SRes, SResMut},
            Commands, Local, Query, SystemParamItem,
        },
    },
    math::Vec3A,
    transform::components::GlobalTransform,
    utils::tracing,
};
use dust_pbr::PbrPipeline;
use dust_vdb::Node;
use rhyolite::{ash::vk, commands::TransferCommands, staging::StagingBelt, Allocator, Buffer};
use rhyolite_rtx::{BLASBuildGeometry, BLASBuilder, HitgroupHandle, TLASBuilder, BLAS};

use crate::{TreeRoot, VoxGeometry, VoxInstance, VoxMaterial, VoxModel, VoxPalette, VoxPaletteGPU};

/// BLAS builder that builds a BLAS for all entities with `VoxBLASBuilder` and `AssetId<VoxGeometry>` components.
/// Expects asset with `AssetId<VoxGeometry>` to be loaded at the time when the builder is run.
pub struct VoxBLASBuilder;

impl BLASBuilder for VoxBLASBuilder {
    type QueryData = &'static Handle<VoxGeometry>;

    type QueryFilter = ();

    type Params = (
        SRes<Assets<VoxGeometry>>,
        SRes<Allocator>,
        SResMut<StagingBelt>,
    );

    type BufferType = Buffer;
    type GeometryIterator<'a> = std::iter::Once<BLASBuildGeometry<Buffer>>;
    fn geometries<'a>(
        (assets, allocator, staging_belt): &'a mut SystemParamItem<Self::Params>,
        data: &'a QueryItem<Self::QueryData>,
        commands: &mut impl TransferCommands,
    ) -> Self::GeometryIterator<'a> {
        let geometry = assets.get(*data).unwrap();
        let leaf_count = geometry.tree.iter_leaf().count();

        let leaf_extent_int = <<TreeRoot as Node>::LeafType as Node>::EXTENT;
        let leaf_extent: Vec3A = leaf_extent_int.as_vec3a();
        let leaf_extent: Vec3A = geometry.unit_size * leaf_extent;
        let mut current_location = 0;

        let buffer = Buffer::new_resource_init_with(
            allocator.clone(),
            staging_belt,
            leaf_count as u64 * std::mem::size_of::<vk::AabbPositionsKHR>() as u64,
            1,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            commands,
            |dst| {
                for (position, _) in geometry.tree.iter_leaf() {
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
            },
        )
        .unwrap();

        std::iter::once(BLASBuildGeometry::Aabbs {
            buffer,
            stride: std::mem::size_of::<vk::AabbPositionsKHR>() as u64,
            flags: vk::GeometryFlagsKHR::OPAQUE,
            primitive_count: leaf_count as u32,
        })
    }
}

pub struct VoxTLASBuilder;
impl TLASBuilder for VoxTLASBuilder {
    type QueryData = (&'static GlobalTransform, &'static VoxInstance);

    type QueryFilter = ();

    type ChangeFilter = Changed<GlobalTransform>;

    type Params = Query<'static, 'static, &'static BLAS>;

    fn instance(
        params: &mut SystemParamItem<Self::Params>,
        (transform, instance): &QueryItem<Self::QueryData>,
        mut dst: rhyolite_rtx::TLASInstanceData,
    ) {
        if let Ok(blas) = params.get(instance.model) {
            println!("Did set blas");
            dst.set_blas(blas);
        } else {
            dst.disable();
            return;
        }
        dst.set_transform(transform.compute_matrix());
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
    type ChangeFilter = ();

    type InlineParam = ShaderParams;
}
