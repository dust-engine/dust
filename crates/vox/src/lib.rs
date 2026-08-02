#![feature(generic_const_exprs)]
#![feature(f16)]

use bevy::ecs::entity::Entity;
use bevy::ecs::reflect::ReflectComponent;
use bevy::ecs::schedule::{IntoScheduleConfigs, Schedules};
use bevy::ecs::template::{
    EntityTemplate, FnTemplate, FromTemplate, SceneEntityReference, TemplateContext,
};
use bevy::math::U8Vec4;
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::scene::{RelatedScenes, ResolveContext, ResolveSceneError, ResolvedScene, Scene};
use bevy::{
    asset::{Asset, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use bevy_pumicite::rtx::RtxPipelineManager;
use bevy_pumicite::rtx::tlas::TLASInstance;
use bevy_pumicite::shader::RayTracingPipelineLibrary;
use bevy_pumicite::{CreateDevice, DefaultTransferSet, SubmissionState};
use bytemuck::{Pod, Zeroable};
use dust_pbr::PbrRenderState;
use dust_pbr::sharc::SharcPipelines;
use dust_vdb::hierarchy;
use pumicite::ash::{VkResult, vk};
use pumicite::buffer::{Buffer, BufferLike, ManagedBuffer};
use pumicite::device::DeviceBuilder;
use pumicite::{Allocator, Device};
use std::mem::MaybeUninit;

mod geometry;
mod loader;
mod material;

pub use material::{VoxLeafNode, VoxMaterial};

/// Leaf node size: 96 bytes
type TreeRoot = hierarchy!(3, 3, 2, VoxLeafNode);

type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

pub use geometry::VoxGeometry;

/// A single MagicaVoxel palette entry: the RGBA color plus the per-index PBR
/// material attributes decoded from a `MATL` chunk. 256 of these form the
/// palette, addressed by the 0-based palette index (the same index the
/// per-voxel attribute stores). Uploaded as one GPU storage buffer; the shader
/// reads color and material together from one fetch. Layout must match the
/// Slang `VoxMaterial` struct (16 bytes) — `emission`/`ior`/`transparency` are
/// stored as IEEE-754 half-precision bit patterns (read as `half` on the GPU).
#[derive(Pod, Clone, Copy, Zeroable)]
#[repr(C)]
pub struct VoxGpuMaterial {
    /// Packed bytes, low → high: material type, roughness, metalness, specular.
    /// Type: 0 = diffuse, 1 = metal, 2 = glass, 3 = emit, 4 = media/cloud.
    /// Roughness / metalness / specular are unorm (byte / 255).
    pub packed: u32,
    /// sRGB color packed RGBA8, little-endian (`r | g<<8 | b<<16 | a<<24`).
    pub color: U8Vec4,
    /// Emissive radiance multiplier applied to albedo (f16 bits). 0 if not emissive.
    pub emission: f16,
    /// Index of refraction (f16 bits). 1.0 = vacuum. Relevant to glass.
    pub ior: f16,
    /// Transparency in [0, 1] (f16 bits). Relevant to glass.
    pub transparency: f16,
    /// Padding to 16 bytes (mirrors the Slang struct's trailing `half`).
    pub _padding: u16,
}
impl Default for VoxGpuMaterial {
    fn default() -> Self {
        Self {
            packed: 0,
            color: U8Vec4::ZERO,
            emission: 0.0,
            ior: 1.0,
            transparency: 0.0,
            _padding: 0,
        }
    }
}

/// A MagicaVoxel palette: 256 [`VoxGpuMaterial`] entries (color + PBR material)
/// in a single GPU storage buffer, addressed by the 0-based palette index.
/// Built once per `.vox` file and shared by every model in it.
#[derive(Asset, TypePath)]
pub struct VoxPalette(ManagedBuffer);

impl VoxPalette {
    /// Build a palette from 256 entries, uploading them to one GPU buffer.
    pub fn new(allocator: pumicite::Allocator, entries: &[VoxGpuMaterial; 256]) -> VkResult<Self> {
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let mut buffer = ManagedBuffer::new(
            allocator,
            256 * size_of::<VoxGpuMaterial>() as u64,
            4,
            usage,
        )?;
        buffer
            .as_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(entries));
        Ok(Self(buffer))
    }

    pub fn colorful(allocator: pumicite::Allocator) -> VkResult<Self> {
        use bevy::color::{Hsva, Srgba};
        let mut hue = 0.0;
        let saturation = 0.8;
        let value = 0.9;

        let mut entries = Box::new([VoxGpuMaterial::default(); 256]);
        for entry in entries.iter_mut() {
            let color = Hsva::new(hue, saturation, value, 1.0);
            let rgb_color: Srgba = color.into();
            let rgb_color: [u8; 4] = rgb_color.to_u8_array();
            entry.color = U8Vec4::from_array(rgb_color);
            hue += 360.0 / 256.0;
        }
        Self::new(allocator, &entries)
    }
}

/// Marker component for Vox instances
#[derive(Component, Reflect, Clone)]
#[require(Transform)]
#[reflect(Component)]
pub struct VoxInstance;

#[derive(Default)]
pub struct VoxInstanceTemplate {
    pub model: EntityTemplate,
}
impl Template for VoxInstanceTemplate {
    type Output = VoxInstance;

    fn build_template(&self, context: &mut TemplateContext) -> Result<Self::Output> {
        let blas = self.model.build_template(context)?;
        context
            .entity
            .insert(TLASInstance::<dust_pbr::PbrInstanceData>::new(blas));
        Ok(VoxInstance)
    }

    fn clone_template(&self) -> Self {
        Self {
            model: self.model.clone_template(),
        }
    }
}
impl FromTemplate for VoxInstance {
    type Template = VoxInstanceTemplate;
}

#[derive(Component, Reflect, Clone, FromTemplate)]
#[require(VoxModelBLASRebuild)]
#[reflect(Component)]
pub struct VoxModel {
    pub geometry: Handle<VoxGeometry>,
    pub material: Handle<VoxMaterial>,
    pub palette: Handle<VoxPalette>,
    pub sbt_index: u32,
    pub enable_compaction: bool,
    pub prefer_fast_build: bool,
}
impl Default for VoxModel {
    fn default() -> Self {
        Self {
            geometry: Handle::default(),
            material: Handle::default(),
            palette: Handle::default(),
            sbt_index: u32::MAX,
            enable_compaction: true,
            prefer_fast_build: false,
        }
    }
}

/// A marker trait for requesting BLAS rebuilds.
#[derive(Component, Default, Reflect, Clone)]
#[reflect(Component)]
pub struct VoxModelBLASRebuild;
impl VoxModelBLASRebuild {
    pub fn request_rebuild(&mut self) {
        // No-op. BlasBuilder uses change tracker to schedule BLAS rebuilds.
    }
}

pub struct VoxPlugin;
impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>()
            .register_type::<VoxModelBLASRebuild>();

        // Build a BLAS for all entities with VoxModel and without the BLAS component.
        app.add_plugins(bevy_pumicite::rtx::blas::BLASBuilderPlugin::<
            geometry::BlasBuilder,
        >::default());

        if app
            .world()
            .resource::<pumicite::physical_device::PhysicalDevice>()
            .properties()
            .device_type
            != vk::PhysicalDeviceType::INTEGRATED_GPU
        {
            app.add_systems(PostUpdate, sync_buffers_system.in_set(DefaultTransferSet));
        }

        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<DeviceBuilder>| {
                device_builder
                    .enable_extension::<pumicite::ash::khr::push_descriptor::Meta>()
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFeatures| {
                        &mut features.shader_int64
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFeatures| {
                        &mut features.shader_int16
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFloat16Int8FeaturesKHR| {
                        &mut features.shader_int8
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDevice8BitStorageFeatures| {
                        &mut features.storage_buffer8_bit_access
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFloat16Int8FeaturesKHR| {
                        &mut features.shader_float16
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDevice16BitStorageFeatures| {
                        &mut features.uniform_and_storage_buffer16_bit_access
                    })
                    .unwrap();
            })
            .before(CreateDevice),
        );

        app.add_systems(
            Startup,
            (|allocator: Res<Allocator>, asset_server: Res<AssetServer>| {
                asset_server.register_loader(VoxLoader::new(allocator.clone()));
            })
            .after(CreateDevice),
        );

        app.add_systems(
            Startup,
            setup
                .after(dust_pbr::setup)
                .after(dust_pbr::sharc::setup_sharc),
        );

        app.add_systems(PostUpdate, write_sbt_entries.in_set(dust_pbr::PbrRenderSet));
    }
}

#[derive(Resource)]
pub struct VoxRenderState {
    primary_pipeline: Handle<RayTracingPipelineLibrary>,
    shadow_pipeline: Handle<RayTracingPipelineLibrary>,
    final_gather_pipeline: Handle<RayTracingPipelineLibrary>,
    // Shared closest-hit + intersection attached to the SHARC Update pipeline.
    sharc_update_pipeline: Handle<RayTracingPipelineLibrary>,
    primary_library_index: u16,
    shadow_library_index: u16,
    final_gather_library_index: u16,
    sharc_update_library_index: u16,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut pipeline_manager: ResMut<RtxPipelineManager>,
    pbr_render_state: Res<PbrRenderState>,
    sharc_pipelines: Res<SharcPipelines>,
) {
    let primary_pipeline: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/vox/shaders/vox_pbr.rtx.pipeline.bin");
    let primary_library_index = pipeline_manager
        .add_library_for_pipeline(&pbr_render_state.pipeline, primary_pipeline.clone());

    let shadow_pipeline: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/vox/shaders/vox_shadow.rtx.pipeline.bin");
    let shadow_library_index = pipeline_manager
        .add_library_for_pipeline(&pbr_render_state.shadow_pipeline, shadow_pipeline.clone());

    let final_gather_pipeline: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/vox/shaders/vox_final_gather.rtx.pipeline.bin");
    let final_gather_library_index = pipeline_manager.add_library_for_pipeline(
        &pbr_render_state.final_gather_pipeline,
        final_gather_pipeline.clone(),
    );

    let sharc_update_pipeline: Handle<RayTracingPipelineLibrary> =
        asset_server.load("bazel://dust/crates/vox/shaders/vox_sharc_update.rtx.pipeline.bin");
    let sharc_update_library_index = pipeline_manager.add_library_for_pipeline(
        &sharc_pipelines.update_pipeline,
        sharc_update_pipeline.clone(),
    );

    commands.insert_resource(VoxRenderState {
        primary_pipeline,
        primary_library_index,
        shadow_pipeline,
        shadow_library_index,
        final_gather_pipeline,
        final_gather_library_index,
        sharc_update_pipeline,
        sharc_update_library_index,
    });
}

#[derive(Pod, Clone, Copy, Zeroable)]
#[repr(C)]
struct VoxModelParams {
    geometry_info: u64,
    material_info: u64,
    palette: u64,
    unit_size: f32,
    _pad: u32,
}

fn write_sbt_entries(
    mut models: Query<&mut VoxModel>,
    mut instances: Query<(Entity, &mut TLASInstance<dust_pbr::PbrInstanceData>), With<VoxInstance>>,
    mut pbr_state: ResMut<PbrRenderState>,
    mut sharc_pipelines: ResMut<SharcPipelines>,
    vox_render_state: Res<VoxRenderState>,

    geometry_assets: Res<Assets<VoxGeometry>>,
    material_assets: Res<Assets<VoxMaterial>>,
    palette_assets: Res<Assets<VoxPalette>>,
) {
    let pbr_state = &mut *pbr_state;
    let sharc_state = &mut *sharc_pipelines;
    let (Some(sbt), Some(shadow_sbt), Some(final_gather_sbt)) = (
        pbr_state.sbt.as_mut(),
        pbr_state.shadow_sbt.as_mut(),
        pbr_state.final_gather_sbt.as_mut(),
    ) else {
        for mut model in models.iter_mut() {
            model.sbt_index = u32::MAX;
        }
        for (_, mut instance) in instances.iter_mut() {
            instance.disabled = true;
        }
        tracing::warn!("Missing SBT");
        return;
    };
    let Some(sharc_update_sbt) = sharc_state.update_sbt.as_mut() else {
        for mut model in models.iter_mut() {
            model.sbt_index = u32::MAX;
        }
        for (_, mut instance) in instances.iter_mut() {
            instance.disabled = true;
        }
        tracing::warn!("Missing SHARC SBT");
        return;
    };
    let miss_index = shadow_sbt.push_miss(vox_render_state.shadow_library_index, 1, |_| {});
    assert_eq!(miss_index, 0); // Miss SBT index assumed to be 0 in the shader. If this assumption no longer holds true, update the shader.
    for mut model in models.iter_mut() {
        model.sbt_index = u32::MAX;
        let Some(geometry) = geometry_assets.get(&model.geometry) else {
            println!("no geometry");
            continue;
        };
        let Some(material) = material_assets.get(&model.material) else {
            println!("no material");
            continue;
        };
        let Some(palette) = palette_assets.get(&model.palette) else {
            println!("no palette");
            continue;
        };
        let params = VoxModelParams {
            geometry_info: geometry.tree.pools()[0].storage().device_address(),
            material_info: material.buffer.device_address(),
            palette: palette.0.device_address(),
            unit_size: geometry.unit_size,
            _pad: 0,
        };
        model.sbt_index =
            sbt.push_hitgroup(vox_render_state.primary_library_index, 0, |param_dst| {
                param_dst.copy_from_slice(bytemuck::bytes_of(&params));
            });
        // Push same geometry to shadow SBT (same order ensures matching sbt_offsets)
        let shadow_sbt_index =
            shadow_sbt.push_hitgroup(vox_render_state.shadow_library_index, 0, |param_dst| {
                param_dst.copy_from_slice(bytemuck::bytes_of(&params));
            });
        assert_eq!(shadow_sbt_index, model.sbt_index);
        // Push same geometry to final gather SBT
        let final_gather_sbt_index = final_gather_sbt.push_hitgroup(
            vox_render_state.final_gather_library_index,
            0,
            |param_dst| {
                param_dst.copy_from_slice(bytemuck::bytes_of(&params));
            },
        );
        assert_eq!(final_gather_sbt_index, model.sbt_index);

        let sharc_update_idx = sharc_update_sbt.push_hitgroup(
            vox_render_state.sharc_update_library_index,
            0,
            |param_dst| {
                param_dst.copy_from_slice(bytemuck::bytes_of(&params));
            },
        );
        assert_eq!(sharc_update_idx, model.sbt_index);
    }
    for (entity, mut instance) in instances.iter_mut() {
        instance.disabled = true;
        let Ok(model) = models.get(instance.blas) else {
            tracing::warn!(
                "Missing model {:?} for instance {:?}",
                instance.blas,
                entity
            );
            continue;
        };
        if model.sbt_index == u32::MAX {
            continue;
        }
        instance.set_sbt_offset(model.sbt_index);
        instance.disabled = false;
    }
}

fn sync_buffers_system(
    mut ctx: SubmissionState,
    mut material_events: MessageReader<AssetEvent<VoxMaterial>>,
    mut palette_events: MessageReader<AssetEvent<VoxPalette>>,

    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>,
) {
    ctx.record(|encoder| {
        for event in material_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let material = materials.get(*id).unwrap();
                    material.buffer.flush(encoder);
                }
                _ => (),
            }
        }
        for event in palette_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let palette = palettes.get(*id).unwrap();
                    palette.0.flush(encoder);
                }
                _ => (),
            }
        }
    });
}
