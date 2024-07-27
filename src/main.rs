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
use bevy_rapier3d::pipeline::QueryFilter;
use bevy_rapier3d::plugin::RapierContext;
use dust_pbr::camera::CameraBundle;

use bevy::app::{PluginGroup, Startup, Update};
use bevy::ecs::system::{Commands, Query, Res};
use bevy::ecs::{entity::Entity, query::With};

fn main() {
    let mut app = bevy::app::App::new();

    app.add_plugins(
        dust::DustPlugin.set::<bevy::asset::AssetPlugin>(bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        }),
    );

    app.add_systems(Startup, startup_system)
        .add_systems(Update, teapot_move_system)
        .add_systems(Update, ray_cast)
        .add_systems(Update, player_movement);

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
        })
        .insert(Player);
}

#[derive(Component)]
pub struct TeaPot;

#[derive(Component)]
pub struct Player;

fn teapot_move_system(time: Res<Time>, mut query: Query<&mut Transform, With<TeaPot>>) {
    return;
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

    //*gravity_movement -= 1.0;
    //movement.y = *gravity_movement; // Gravity

    // Movement on the Y axis
    if keyboard.pressed(KeyCode::Space) {
        // If grounded
        movement.y = 100.0;
    }

    if keyboard.pressed(KeyCode::ControlLeft) {
        // If grounded
        movement.y = -100.0;
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

fn ray_cast(
    mut player: Query<(Entity, &GlobalTransform), With<Player>>,
    rapier_context: Res<RapierContext>,
) {
    let (player, transform) = player.single();
    let ray = transform.affine().transform_vector3(-Vec3::Z);

    if let Some((entity, toi)) = rapier_context.cast_ray(
        transform.translation(),
        ray,
        100000.0,
        false,
        QueryFilter::default().predicate(&|entity| entity != player),
    ) {
        // The first collider hit has the entity `entity` and it hit after
        // the ray travelled a distance equal to `ray_dir * toi`.
        let hit_point = transform.translation() + ray * toi;
        println!("Entity {:?} hit at point {}", entity, hit_point);
    }
}
