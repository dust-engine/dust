use std::sync::Arc;

use bevy::asset::{AssetServer, Handle};
use bevy::input::mouse::MouseMotion;
use bevy::input::ButtonInput;
use bevy::math::{Quat, Vec2, Vec3};
use bevy::prelude::*;
use bevy::scene::{Scene, SceneBundle};
use bevy::time::Time;
use bevy::transform::bundles::TransformBundle;
use bevy::transform::components::Transform;
use bevy_rapier3d::control::{
    CharacterAutostep, CharacterLength, KinematicCharacterController,
    KinematicCharacterControllerOutput,
};
use bevy_rapier3d::geometry::Collider;
use bevy_rapier3d::parry::query::{DefaultQueryDispatcher, QueryDispatcher};
use dust_pbr::camera::CameraBundle;
use dust_vox::VoxPlugin;
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, Startup, Update};
use bevy::ecs::system::{Commands, Query, Res};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::{RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainPlugin};

fn main() {
    let mut app = bevy::app::App::new();

    let query_dispatcher = DefaultQueryDispatcher.chain(dust_vdb::VdbQueryDispatcher);
    app.add_plugins(dust_log::LogPlugin)
        .add_plugins((
            bevy::DefaultPlugins
                .set::<bevy::asset::AssetPlugin>(bevy::asset::AssetPlugin {
                    mode: bevy::asset::AssetMode::Processed,
                    ..Default::default()
                })
                .disable::<bevy::log::LogPlugin>(),
            bevy_rapier3d::plugin::RapierPhysicsPlugin::<()>::default()
            .with_query_dispatcher(Arc::new(query_dispatcher))
            .with_narrow_phase_dispatcher(Arc::new(query_dispatcher)),
        ))
        .add_plugins(SurfacePlugin::default())
        .add_plugins(DebugUtilsPlugin::default())
        .add_plugins(RhyolitePlugin::default())
        .add_plugins(SwapchainPlugin::default());

    app.add_plugins(dust_pbr::PbrRendererPlugin);

    app.add_plugins(VoxPlugin);

    app.add_systems(Startup, startup_system)
        .add_systems(Update, teapot_move_system)
        .add_systems(Update, player_movement);

    let world = app.world_mut();

    let primary_window = world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(world)
        .next()
        .unwrap();
    world.entity_mut(primary_window).insert(SwapchainConfig {
        image_usage: vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::STORAGE,
        srgb_format: false,
        ..Default::default()
    });

    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene: Handle<Scene> = asset_server.load("castle.vox");
    commands.spawn(SceneBundle {
        scene,
        ..Default::default()
    });

    commands
        .spawn(SceneBundle {
            scene: asset_server.load("teapot.vox"),
            ..Default::default()
        })
        .insert(TeaPot);

    commands
        .spawn(CameraBundle {
            transform: TransformBundle {
                local: Transform::from_translation(Vec3::new(0.0, 400.0, 0.0)),
                ..Default::default()
            },
            ..Default::default()
        })
        .insert(Collider::capsule_y(10.0, 3.0))
        .insert(KinematicCharacterController {
            custom_mass: Some(5.0),
            up: Vec3::Y,
            offset: CharacterLength::Absolute(0.01),
            slide: true,
            autostep: Some(CharacterAutostep {
                max_height: CharacterLength::Relative(0.3),
                min_width: CharacterLength::Relative(0.5),
                include_dynamic_bodies: false,
            }),
            // Don’t allow climbing slopes larger than 45 degrees.
            max_slope_climb_angle: 45.0_f32.to_radians(),
            // Automatically slide down on slopes smaller than 30 degrees.
            min_slope_slide_angle: 30.0_f32.to_radians(),
            apply_impulse_to_dynamic_bodies: true,
            snap_to_ground: None,
            ..Default::default()
        });
}

#[derive(Component)]
pub struct TeaPot;
fn teapot_move_system(time: Res<Time>, mut query: Query<&mut Transform, With<TeaPot>>) {
    for mut teapot in query.iter_mut() {
        *teapot =
            Transform::from_translation(Vec3::new(time.elapsed_seconds().sin() * 50.0, 200.0, 0.0));
    }
}

fn player_movement(
    time: Res<Time>,
    mut player: Query<(
        &mut Transform,
        &mut KinematicCharacterController,
        Option<&KinematicCharacterControllerOutput>,
    )>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_events: EventReader<MouseMotion>,
    mut orientation: Local<Vec2>,
    mut gravity_movement: Local<f32>,
) {
    let Ok((mut transform, mut controller, output)) = player.get_single_mut() else {
        return;
    };
    let mut movement = Vec3::ZERO;

    // Movement on the xz plane
    if keyboard.pressed(KeyCode::KeyW) {
        movement.z -= 100.0;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        movement.z += 100.0
    }
    if keyboard.pressed(KeyCode::KeyA) {
        movement.x -= 100.0;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        movement.x += 100.0
    }
    if keyboard.pressed(KeyCode::ShiftLeft) {
        movement *= 2.0;
    }

    *gravity_movement -= 1.0;
    movement.y = *gravity_movement; // Gravity

    // Movement on the Y axis
    if keyboard.pressed(KeyCode::Space) {
        // If grounded
        if output.map(|o| o.grounded).unwrap_or(false) {
            movement.y = 5000.0;
            *gravity_movement = 0.0;
        }
    }
    controller.translation =
        Some(Quat::from_axis_angle(Vec3::Y, orientation.x) * (movement * time.delta_seconds()));

    // Look transform

    for event in mouse_events.read() {
        let sensitivity = 0.001;
        orientation.x -= event.delta.x * sensitivity;
        orientation.y -= event.delta.y * sensitivity;
        orientation.y = orientation.y.clamp(-89.9, 89.9); // Limit pitch
    }

    transform.rotation =
        Quat::from_euler(bevy::math::EulerRot::YXZ, orientation.x, orientation.y, 0.0);
}
