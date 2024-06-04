use bevy::{
    asset::{AssetServer, Assets, Handle},
    ecs::{
        change_detection::DetectChanges,
        query::{Changed, QueryItem},
        system::{
            lifetimeless::{SRes, SResMut},
            Local, Query, SystemParamItem,
        },
        world::Ref,
    },
    math::Vec3A,
    transform::components::GlobalTransform,
};
use dust_pbr::PbrPipeline;
use dust_vdb::Node;
use rhyolite::{
    ash::vk, commands::TransferCommands, pipeline::PipelineCache, shader::SpecializedShader,
    staging::StagingBelt, Allocator, Buffer,
};
use rhyolite_rtx::{
    BLASBuildGeometry, BLASBuilder, HitGroup, HitgroupHandle, SbtIndex, SbtManager, TLASBuilder,
    BLAS,
};

use crate::{TreeRoot, VoxGeometry, VoxInstance, VoxMaterial, VoxPaletteGPU};

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
    type QueryData = (Ref<'static, GlobalTransform>, Ref<'static, VoxInstance>);

    type QueryFilter = ();

    type Params = (
        Query<'static, 'static, (Ref<'static, BLAS>, Ref<'static, SbtIndex<PbrPipeline>>)>,
        SRes<SbtManager<PbrPipeline>>,
    );

    fn should_update(
        (query, _): &mut SystemParamItem<Self::Params>,
        (transform, instance): &QueryItem<Self::QueryData>,
    ) -> bool {
        let blas_changed = query
            .get(instance.model)
            .map(|(blas, sbt_index)| blas.is_changed() || sbt_index.is_changed())
            .unwrap_or(false);
        transform.is_changed() || blas_changed || instance.is_changed()
    }

    fn instance(
        (query, sbt_manager): &mut SystemParamItem<Self::Params>,
        (transform, instance): &QueryItem<Self::QueryData>,
        mut dst: rhyolite_rtx::TLASInstanceData,
    ) {
        if let Ok((blas, sbt_index)) = query.get(instance.model) {
            if sbt_manager.index_available(&sbt_index) {
                dst.set_blas(&blas);
                dst.set_sbt_offset_and_flags(**sbt_index, vk::GeometryInstanceFlagsKHR::empty());
                dst.set_transform(transform.compute_matrix());
                dst.set_custom_index_and_mask(23, 0);
                return;
            } else {
                // TODO: Test when item requiring new hitgroup pops into view
                panic!();
            }
        }
        dst.disable();
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
    type QueryData = &'static Handle<VoxGeometry>;

    type QueryFilter = ();

    type Params = (
        SRes<AssetServer>,
        SRes<Assets<VoxMaterial>>,
        SRes<Assets<VoxGeometry>>,
        SRes<Assets<VoxPaletteGPU>>,
        SResMut<PbrPipeline>,
        SRes<PipelineCache>,
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
        let (assets, materials, geometry, palette, pipeline, pipeline_cache, hitgroup_handle) =
            params;
        *hitgroup_handle.get_or_insert_with(|| {
            let mut hitgroup =
                HitGroup::new(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP);
            let rint = hitgroup.add_intersection_shader(SpecializedShader {
                stage: vk::ShaderStageFlags::INTERSECTION_KHR,
                shader: assets.load("shaders/primary/primary.rint"),
                ..Default::default()
            });
            let rchit = hitgroup.add_closest_hit_shader(SpecializedShader {
                stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                shader: assets.load("shaders/primary/primary.rchit"),
                ..Default::default()
            });
            hitgroup.add_group(Some(rchit), None, Some(rint));
            pipeline.primary.add_hitgroup(hitgroup, pipeline_cache)
        })
    }
    type ChangeFilter = Changed<Handle<VoxGeometry>>;

    type InlineParam = ShaderParams;

    const NUM_RAYTYPES: u32 = 1;

    type SbtIndexType = PbrPipeline;

    fn pipeline<'a>(
        params: &'a mut SystemParamItem<Self::Params>,
        raytype: u32,
    ) -> Option<&'a rhyolite_rtx::RayTracingPipeline> {
        let (assets, materials, geometry, palette, pipeline, pipeline_cache, hitgroup_handle) =
            params;
        pipeline
            .primary
            .get_pipeline()
            .and_then(|x| x.get())
            .map(|x| x.get())
    }
}
