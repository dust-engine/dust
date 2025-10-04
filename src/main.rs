#![feature(impl_trait_in_fn_trait_return)]

use bevy::prelude::*;
use dust_vox::{VoxInstance, VoxInstanceBundle, VoxModel};
use rhyolite::{Allocator, ash::vk, tracking::Access};
use rhyolite_bevy::{DefaultRenderSet, RenderSetSharedStateWrapper, rtx::tlas::{TLASBuilderSet, TLASInstance}, swapchain::SwapchainImage};
fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::DefaultPlugins)
        .add_plugins(rhyolite_bevy::SurfacePlugin::default())
        .add_plugins(rhyolite_bevy::DebugUtilsPlugin::default())
        .add_plugins(rhyolite_bevy::RhyolitePlugin::default())
        .add_plugins(rhyolite_bevy::swapchain::SwapchainPlugin);

    // Dust plugins
    app.add_plugins(dust_pbr::PbrRenderPlugin)
        .add_plugins(dust_vox::VoxPlugin);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(rhyolite_bevy::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.add_systems(Startup, startup_system);
    app.add_systems(PostUpdate, registering_instances);
    app.add_systems(PostUpdate, clear.in_set(DefaultRenderSet));

    // Build a TLAS over everything.
    app.add_plugins(rhyolite_bevy::rtx::tlas::TLASBuilderPlugin::<()>::default());
    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>, allocator: Res<Allocator>,
mut geometries: ResMut<Assets<dust_vox::VoxGeometry>>,
mut materials: ResMut<Assets<dust_vox::VoxMaterial>>,
mut palettes: ResMut<Assets<dust_vox::VoxPalette>>,
) {
    let scene: Handle<Scene> = asset_server.load("castle.vox");
    commands.spawn(SceneRoot(scene));
    return;

    
    let mut geometry = dust_vox::VoxGeometry::new(allocator.clone(), 1.0);
    let mut material = dust_vox::VoxMaterial::new(allocator.clone());
    let mut accessor = geometry.tree.accessor_mut(&mut material);
    accessor.set(UVec3::new(8, 9, 10), 123);
    accessor.end();

    let model = commands.spawn(VoxModel {
        geometry: geometries.add(geometry),
        material: materials.add(material),
        palette: palettes.add(dust_vox::VoxPalette::colorful()),
        sbt_index: u32::MAX,
    }).id();
    commands.spawn(VoxInstanceBundle {
        transform: Default::default(),
        global_transform: Default::default(),
        instance: VoxInstance { model },
    });
}

fn clear(
    mut swapchain_images: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderSetSharedStateWrapper,
) {
    let Ok(mut swapchain_image) = swapchain_images.single_mut() else {
        return;
    };
    state.record(|encoder| {
        let image = encoder.lock(swapchain_image.inner.as_ref().unwrap(), vk::PipelineStageFlags2::BLIT);

        encoder.use_image_resource(
            image,
            &mut swapchain_image.state,
            Access::CLEAR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, 0..1, 0..1, true);

        encoder.emit_barriers();
        encoder.clear_color_image(
            image,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 1.0, 0.0],
            },
        );
    });
}

fn registering_instances(
    query: Query<(Entity, &VoxInstance), Without<TLASInstance<()>>>,
    mut commands: Commands
) {
    for (entity, instance) in query.iter() {
        commands.entity(entity).insert(TLASInstance::<()>::new(instance.model));
    }
}