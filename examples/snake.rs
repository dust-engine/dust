#![feature(generic_const_exprs)]

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
use dust_vox::{
    VoxGeometry, VoxInstance, VoxInstanceBundle, VoxMaterial, VoxModelBundle, VoxPalette,
};
use rhyolite::Allocator;

fn main() {
    let mut app = bevy::app::App::new();

    app.add_plugins(
        dust::DustPlugin.set::<bevy::asset::AssetPlugin>(bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        }),
    );

    app.add_systems(Startup, (startup_system, setup_camera));

    app.add_systems(Update, player_movement);

    app.run();
}

fn startup_system(
    mut commands: Commands,
    mut geometries: ResMut<Assets<VoxGeometry>>,
    mut materials: ResMut<Assets<VoxMaterial>>,
    mut palettes: ResMut<Assets<VoxPalette>>,
    allocator: Res<Allocator>,
) {
    use dust_vdb::{hierarchy, MutableTree};
    use dust_vox::{AttributeAllocator, VoxMaterial, VoxModelBundle};
    let mut snake_tree = MutableTree::<hierarchy!(3, 3, 2, u32)>::new();

    let mut material =
        VoxMaterial(AttributeAllocator::new_with_capacity(allocator.clone(), 1024, 4, 64).unwrap());

    let mut accessor = snake_tree.accessor_mut(&mut material);
    accessor.set(UVec3::new(0, 0, 0), 12);
    accessor.set(UVec3::new(1, 0, 0), 13);
    accessor.set(UVec3::new(2, 0, 0), 14);
    accessor.set(UVec3::new(3, 0, 0), 15);
    accessor.set(UVec3::new(4, 0, 0), 131);
    accessor.set(UVec3::new(5, 0, 0), 210);
    accessor.set(UVec3::new(123, 123, 132), 210);
    accessor.set(UVec3::new(6, 0, 0), 210);

    assert_eq!(accessor.get(UVec3::new(0, 0, 0)), Some(12));
    assert_eq!(accessor.get(UVec3::new(1, 0, 0)), Some(13));
    assert_eq!(accessor.get(UVec3::new(2, 0, 0)), Some(14));
    assert_eq!(accessor.get(UVec3::new(3, 0, 0)), Some(15));
    assert_eq!(accessor.get(UVec3::new(4, 0, 0)), Some(131));
    assert_eq!(accessor.get(UVec3::new(5, 0, 0)), Some(210));
    assert_eq!(accessor.get(UVec3::new(123, 123, 132)), Some(210));
    assert_eq!(accessor.get(UVec3::new(6, 0, 0)), Some(210));
    accessor.end();

    let model = commands
        .spawn(VoxModelBundle {
            geometry: geometries.add(VoxGeometry::from_tree_with_unit_size(
                snake_tree.freeze(),
                1.0,
            )),
            material: materials.add(material),
            palette: palettes.add(VoxPalette::colorful()),
            ..Default::default()
        })
        .id();
    commands.spawn(VoxInstanceBundle {
        instance: VoxInstance { model },
        ..Default::default()
    });
}

fn setup_camera(mut commands: Commands) {
    commands
        .spawn(CameraBundle {
            transform: TransformBundle {
                local: Transform::from_translation(Vec3::new(0.0, 0.0, 10.0)),
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
