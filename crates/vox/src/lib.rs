#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use bevy::ecs::entity::{Entity, MapEntities};
use bevy::ecs::reflect::{ReflectComponent, ReflectMapEntities};
use bevy::math::U8Vec4;
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::{
    asset::{Asset, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use bytemuck::{Pod, Zeroable};
use dot_vox::Color;
use dust_pbr::PbrRenderState;
use dust_vdb::hierarchy;
use pumicite::Device;
use pumicite::ash::{VkResult, vk};
use pumicite::buffer::{Buffer, BufferLike, ManagedBuffer};
use bevy_pumicite::{DefaultTransferSet, SubmissionState, PumiciteApp};
use bevy_pumicite::rtx::RtxPipelineManager;
use bevy_pumicite::rtx::tlas::TLASInstance;
use bevy_pumicite::shader::{RayTracingPipelineLibrary};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

mod geometry;
mod loader;
mod material;

pub use material::{VoxLeafNode, VoxMaterial};

/// Leaf node size: 96 bytes
type TreeRoot = hierarchy!(3, 3, 2, VoxLeafNode);

type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

pub use geometry::VoxGeometry;

#[derive(Asset, TypePath)]
pub struct VoxPalette(ManagedBuffer);

impl Deref for VoxPalette {
    type Target = [U8Vec4];
    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(self.0.as_slice())
    }
}
impl DerefMut for VoxPalette {
    fn deref_mut(&mut self) -> &mut Self::Target {
        bytemuck::cast_slice_mut(self.0.as_slice_mut())
    }
}
impl VoxPalette {
    pub fn colorful(allocator: pumicite::Allocator) -> VkResult<Self> {
        use bevy::color::{Hsva, Srgba};
        let mut hue = 0.0;
        let saturation = 0.8;
        let value = 0.9;

        let mut arr: Box<[U8Vec4; 255]> = Box::new([U8Vec4::ZERO; 255]);
        for x in 0..255 {
            let color = Hsva::new(hue, saturation, value, 1.0);
            let rgb_color: Srgba = color.into();
            let rgb_color: [u8; 4] = rgb_color.to_u8_array();
            arr[x] = U8Vec4::from_array(rgb_color);
            hue += 360.0 / 255.0;
        }

        let mut buffer = ManagedBuffer::new(allocator, 256 * 4, 4, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)?;
        buffer.as_slice_mut().copy_from_slice(bytemuck::cast_slice(&*arr));
        Ok(Self(buffer))
    }
}

/// Marker component for Vox instances
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct VoxInstance;

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxModel {
    pub geometry: Handle<VoxGeometry>,
    pub material: Handle<VoxMaterial>,
    pub palette: Handle<VoxPalette>,
    pub sbt_index: u32,
}

#[derive(Bundle, Default)]
pub struct VoxInstanceBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub instance: VoxInstance,
    pub tlas_instance: TLASInstance<()>
}

pub struct VoxPlugin;
impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();

        // Build a BLAS for all entities with VoxModel and without the BLAS component.
        app.add_plugins(bevy_pumicite::rtx::blas::BLASBuilderPlugin::<geometry::BlasBuilder>::default());

        use bevy::asset::embedded_asset;
        embedded_asset!(app, "shaders/blas_builder_copy_coords.spv");
        embedded_asset!(app, "shaders/blas_builder_copy_coords.comp.pipeline.ron");
        
        embedded_asset!(app, "shaders/pbr.spv");
        embedded_asset!(app, "shaders/pbr.rtx.pipeline.ron");
        
        app.add_device_extension::<pumicite::ash::khr::push_descriptor::Meta>()
            .unwrap();
        app.enable_feature(|features: &mut vk::PhysicalDeviceFeatures| &mut features.shader_int64).unwrap();
        app.enable_feature(|features: &mut vk::PhysicalDeviceFeatures| &mut features.shader_int16).unwrap();
        app.enable_feature(|features: &mut vk::PhysicalDeviceFloat16Int8FeaturesKHR| &mut features.shader_int8).unwrap();
        app.enable_feature(|features: &mut vk::PhysicalDevice8BitStorageFeatures| &mut features.storage_buffer8_bit_access).unwrap();

        app.add_systems(Startup, setup.after(dust_pbr::setup));

        app.add_systems(PostUpdate, write_sbt_entries.in_set(dust_pbr::PbrRenderSet));
    }
    fn finish(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>();

        
        if app.world().resource::<Device>()
            .physical_device()
            .properties()
            .device_type
            != vk::PhysicalDeviceType::INTEGRATED_GPU {
                app.add_systems(PostUpdate, sync_buffers_system.in_set(DefaultTransferSet));
            }
    }
}

#[derive(Resource)]
pub struct VoxRenderState {
    pipeline: Handle<RayTracingPipelineLibrary>,
    hitgroup_index: u32,
}


fn setup(mut commands: Commands, asset_server: Res<AssetServer>, mut pipeline_manager: ResMut<RtxPipelineManager>, pbr_render_state: Res<PbrRenderState>) {
    let hitgroup_library: Handle<RayTracingPipelineLibrary> = asset_server.load("embedded://dust_vox/shaders/pbr.rtx.pipeline.ron");
    let hitgroup_index = pipeline_manager.add_hitgroup_for_pipeline(&pbr_render_state.pipeline, hitgroup_library.clone());
    commands.insert_resource(VoxRenderState {
        pipeline: hitgroup_library,
        hitgroup_index,
    });
}

#[derive(Pod, Clone, Copy, Zeroable)]
#[repr(C)]
struct VoxModelParams {
    geometry_info: u64,
    material_info: u64,
    palette: u64,
}

fn write_sbt_entries(
    mut models: Query<&mut VoxModel>,
    mut instances: Query<(Entity, &mut TLASInstance<()>), With<VoxInstance>>,
    mut pbr_render_state: ResMut<PbrRenderState>,
    vox_render_state: Res<VoxRenderState>,

    geometry_assets: Res<Assets<VoxGeometry>>,
    material_assets: Res<Assets<VoxMaterial>>,
    palette_assets: Res<Assets<VoxPalette>>,
) {
    let Some(sbt) = 
        pbr_render_state.sbt.as_mut() else {
            for mut model in models.iter_mut() {
                model.sbt_index = u32::MAX;
            }
            for (_, mut instance) in instances.iter_mut() {
                instance.disabled = true;
            }
            tracing::warn!("Missing SBT");
            return;
        };
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
            palette: palette.0.device_address()
        };
        model.sbt_index = sbt.push_hitgroup(vox_render_state.hitgroup_index, |param_dst| {
            param_dst.copy_from_slice(bytemuck::bytes_of(&params));
        });
    }
    for (entity, mut instance) in instances.iter_mut() {
        instance.disabled = true;
        let Ok(model) = models.get(instance.blas) else {
            tracing::warn!("Missing model {:?} for instance {:?}", instance.blas, entity);
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
    mut material_events: EventReader<AssetEvent<VoxMaterial>>,
    mut palette_events: EventReader<AssetEvent<VoxPalette>>,

    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>
) {
    ctx.record(|encoder| {
        for event in material_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let material = materials.get(*id).unwrap();
                    material.buffer.flush(encoder);
                }
                _ => ()
            }
        }
        for event in palette_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let palette = palettes.get(*id).unwrap();
                    palette.0.flush(encoder);
                }
                _ => ()
            }
        }
    });
}
