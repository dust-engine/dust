#![feature(generic_const_exprs)]
#![feature(alloc_layout_extra)]

use bevy::asset::Assets;
use bevy::ecs::entity::{Entity, MapEntities};
use bevy::ecs::reflect::{ReflectComponent, ReflectMapEntities};
use bevy::math::UVec3;
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::{
    asset::{Asset, AssetApp, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use dot_vox::Color;
use dust_vdb::hierarchy;
use rhyolite::ash::vk;
use rhyolite::ecs::{RenderCommands, RenderSystemPass, SubmissionInfo};
use rhyolite::utils::AssetUploadPlugin;
use rhyolite::{Device, Queues, RhyoliteApp};
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

mod attributes;
mod builder;
mod loader;
//mod physics;
mod resource;

type TreeRoot = hierarchy!(3, 3, 2, u32);
type Tree = dust_vdb::Tree<TreeRoot>;

pub use attributes::{AttributeAllocator, VoxMaterial};
pub use loader::*;
use rhyolite_rtx::{BLASBuilderPlugin, RtxPlugin, SbtPlugin, TLASBuilderPlugin};

#[derive(Asset, TypePath)]
pub struct VoxGeometry {
    tree: Tree,
    unit_size: f32,
}
impl VoxGeometry {
    pub fn from_tree_with_unit_size(tree: Tree, unit_size: f32) -> Self {
        Self { tree, unit_size }
    }
}
impl Deref for VoxGeometry {
    type Target = Tree;
    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}
impl DerefMut for VoxGeometry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tree
    }
}

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
#[derive(Component, Reflect)]
#[reflect(Component, MapEntities)]
pub struct VoxInstance {
    pub model: Entity,
}
impl MapEntities for VoxInstance {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        self.model = entity_mapper.map_entity(self.model);
    }
}

impl Default for VoxInstance {
    fn default() -> Self {
        Self {
            model: Entity::from_raw(u32::MAX),
        }
    }
}

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxModel;

/// Entities loaded into the scene will have this bundle added.
#[derive(Bundle, Default)]
pub struct VoxModelBundle {
    pub geometry: Handle<VoxGeometry>,
    pub material: Handle<VoxMaterial>,
    pub palette: Handle<VoxPalette>,
    pub marker: VoxModel,
}

#[derive(Bundle, Default)]
pub struct VoxInstanceBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub instance: VoxInstance,
}

pub struct VoxPlugin;

impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();

        app.add_plugins((
            RtxPlugin,
            BLASBuilderPlugin::<builder::VoxBLASBuilder>::default(),
            TLASBuilderPlugin::<builder::VoxTLASBuilder>::default(),
            SbtPlugin::<builder::VoxSbtBuilder>::default(),
            AssetUploadPlugin::<crate::resource::VoxPaletteGPU>::default(),
            AssetUploadPlugin::<crate::resource::VoxGeometryGPU>::default(),
        ));

        app.enable_feature::<vk::PhysicalDeviceFeatures>(|x| &mut x.shader_int16)
            .unwrap();
        app.enable_feature::<vk::PhysicalDevice8BitStorageFeatures>(|x| {
            &mut x.storage_buffer8_bit_access
        })
        .unwrap();
        app.enable_feature::<vk::PhysicalDevice16BitStorageFeatures>(|x| {
            &mut x.storage_buffer16_bit_access
        })
        .unwrap();
        app.enable_feature::<vk::PhysicalDeviceShaderFloat16Int8Features>(|x| &mut x.shader_int8)
            .unwrap();

        app.add_systems(
            PostUpdate,
            tree_bind_sparse_system.with_option::<RenderSystemPass>(|entry| {
                let item = entry.or_default();
                item.is_queue_op = true;
                item.required_queue_flags = vk::QueueFlags::SPARSE_BINDING;
            }),
        );
    }
    fn finish(&self, app: &mut App) {
        app.init_asset_loader::<VoxLoader>();

        if app
            .world()
            .resource::<Device>()
            .physical_device()
            .properties()
            .memory_model
            .storage_buffer_should_use_staging()
        {
            println!("Using staging buffer for VoxMaterial");
            app.add_systems(PostUpdate, update_materials_system);
        }
    }
}

fn tree_bind_sparse_system(
    mut asset_events: EventReader<AssetEvent<VoxGeometry>>,
    mut geometries: ResMut<Assets<VoxGeometry>>,
    submission_info: SubmissionInfo,
    queues: Res<Queues>,
    device: Res<Device>,
) {
    let mut sparse_memory_binds: Vec<vk::SparseMemoryBind> = Vec::new();
    let mut buffer_binds: Vec<vk::SparseBufferMemoryBindInfo> = Vec::new();

    // For all changed geometries, try to bind the sparse buffer if needed
    for event in asset_events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let geometry = geometries.get_mut_untracked(*id).unwrap();
                let (buffer, iter) = geometry.bind_sparse();
                buffer_binds.push(vk::SparseBufferMemoryBindInfo {
                    buffer,
                    bind_count: iter.len() as u32,
                    ..Default::default()
                });
                sparse_memory_binds.extend(iter);
            }
            _input => {}
        }
    }
    if buffer_binds.is_empty() {
        return;
    }
    // Assign to p_binds later to avoid reallocations when extending sparse_memory_binds
    buffer_binds.iter_mut().fold(0_usize, |count, item| unsafe {
        item.p_binds = sparse_memory_binds.as_ptr().add(count);
        count + item.bind_count as usize
    });

    let queue = queues.get(submission_info.queue);
    let info = submission_info.info.lock().unwrap();
    assert!(!info.last_buf_open);
    let (semaphore_signals, semaphore_signal_values): (Vec<_>, Vec<_>) = info
        .signal_semaphore
        .iter()
        .map(|semaphore| (semaphore.raw(), info.signal_semaphore_value))
        .unzip();
    let (semaphore_waits, semaphore_wait_values): (Vec<_>, Vec<_>) = info
        .wait_semaphores
        .iter()
        .map(|(_stage, semaphore, value)| (semaphore.raw(), *value))
        .unzip();
    unsafe {
        device
            .queue_bind_sparse(
                *queue,
                &[vk::BindSparseInfo {
                    ..Default::default()
                }
                .buffer_binds(&buffer_binds)
                .signal_semaphores(&semaphore_signals)
                .wait_semaphores(&semaphore_waits)
                .push_next(
                    &mut vk::TimelineSemaphoreSubmitInfo {
                        ..Default::default()
                    }
                    .signal_semaphore_values(&semaphore_signal_values)
                    .wait_semaphore_values(&semaphore_wait_values),
                )],
                vk::Fence::null(),
            )
            .unwrap();
    }
}

fn update_materials_system(
    mut render_commands: RenderCommands<'t'>,
    mut asset_events: EventReader<AssetEvent<VoxMaterial>>,
    mut materials: ResMut<Assets<VoxMaterial>>,
) {
    for event in asset_events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let material = materials.get_mut_untracked(*id).unwrap();
                material.0.buffer_mut().sync(&mut render_commands);
            }
            _input => {}
        }
    }
}
