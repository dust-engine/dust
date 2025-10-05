#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use bevy::ecs::entity::{Entity, MapEntities};
use bevy::ecs::reflect::{ReflectComponent, ReflectMapEntities};
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::{
    asset::{Asset, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use dot_vox::Color;
use dust_pbr::PbrRenderState;
use dust_vdb::hierarchy;
use rhyolite::ash::vk;
use rhyolite_bevy::RhyoliteApp;
use rhyolite_bevy::rtx::RtxPipelineManager;
use rhyolite_bevy::rtx::tlas::TLASInstance;
use rhyolite_bevy::shader::{RayTracingPipelineLibrary};
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
pub struct VoxPalette(Box<[Color; 256]>);

impl Deref for VoxPalette {
    type Target = [Color];
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}
impl DerefMut for VoxPalette {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}
impl VoxPalette {
    pub fn colorful() -> Self {
        use bevy::color::{Hsva, Srgba};
        let mut hue = 0.0;
        let saturation = 0.8;
        let value = 0.9;

        let mut arr: Box<[MaybeUninit<Color>; 255]> = Box::new([MaybeUninit::uninit(); 255]);
        for x in 0..255 {
            let color = Hsva::new(hue, saturation, value, 1.0);
            let rgb_color: Srgba = color.into();
            let rgb_color: [u8; 4] = rgb_color.to_u8_array();
            arr[x].write(Color {
                r: rgb_color[0],
                g: rgb_color[1],
                b: rgb_color[2],
                a: rgb_color[3],
            });
            hue += 360.0 / 255.0;
        }
        unsafe { std::mem::transmute(arr) }
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
        app.add_plugins(rhyolite_bevy::rtx::blas::BLASBuilderPlugin::<geometry::BlasBuilder>::default());

        use bevy::asset::embedded_asset;
        embedded_asset!(app, "shaders/blas_builder_copy_coords.spv");
        embedded_asset!(app, "shaders/blas_builder_copy_coords.comp.pipeline.ron");
        
        embedded_asset!(app, "shaders/pbr.spv");
        embedded_asset!(app, "shaders/pbr.rtx.pipeline.ron");
        
        app.add_device_extension::<rhyolite::ash::khr::push_descriptor::Meta>()
            .unwrap();
        app.add_device_extension::<rhyolite::ash::khr::acceleration_structure::Meta>()
            .unwrap();
        app.add_device_extension::<rhyolite::ash::khr::ray_tracing_pipeline::Meta>()
            .unwrap();
        app.add_device_extension::<rhyolite::ash::khr::pipeline_library::Meta>()
            .unwrap();
         app.enable_feature(
            |rtx_features: &mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR| {
                &mut rtx_features.acceleration_structure
            },
        )
        .unwrap();
        app.enable_feature(|rtx_features: &mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR| &mut rtx_features.ray_tracing_pipeline).unwrap();

        app.add_systems(Startup, setup.after(dust_pbr::setup));

        app.add_systems(PostUpdate, write_sbt_entries.in_set(dust_pbr::PbrRenderSet));
    }
    fn finish(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>();
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

fn write_sbt_entries(mut models: Query<&mut VoxModel>, mut instances: Query<(Entity, &mut TLASInstance<()>), With<VoxInstance>>, mut pbr_render_state: ResMut<PbrRenderState>, vox_render_state: Res<VoxRenderState>) {
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
        model.sbt_index = sbt.push_hitgroup(vox_render_state.hitgroup_index, |param| {

        });
    }
    for (entity, mut instance) in instances.iter_mut() {
        let Ok(model) = models.get(instance.blas) else {
            tracing::warn!("Missing model {:?} for instance {:?}", instance.blas, entity);
            continue;
        };
        instance.set_sbt_offset(0);
        instance.disabled = false;
    }
}
