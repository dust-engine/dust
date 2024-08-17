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
use dust_vox::{VoxGeometry, VoxMaterial, VoxModelBundle, VoxPalette};

fn main() {
    let mut app = bevy::app::App::new();

    app.add_plugins(
        dust::DustPlugin.set::<bevy::asset::AssetPlugin>(bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        }),
    );

    app.add_systems(Startup, startup_system);

    app.run();
}

fn startup_system(
    mut commands: Commands,
    mut geometries: ResMut<Assets<VoxGeometry>>,
    mut materials: ResMut<Assets<VoxMaterial>>,
    mut palettes: ResMut<Assets<VoxPalette>>,
) {
    use dust_vdb::{hierarchy, MutableTree};
    use dust_vox::VoxModelBundle;
    let mut snake_tree = MutableTree::<hierarchy!(3, 3, 2, u32)>::new();

    // Randomly set voxel to be 1
    for _ in 0..100 {
        let mut rand_val = 0;
        while unsafe { std::arch::x86_64::_rdrand64_step(&mut rand_val) == 0 } {}
        let coords = UVec3 {
            x: (rand_val & 0x7FF) as u32,
            y: ((rand_val >> 11) & 0x7FF) as u32,
            z: ((rand_val >> 22) & 0x7FF) as u32,
        };
        println!("{:?} set to 1", coords);
        snake_tree.set_value(coords, true);
    }

    let model = commands
        .spawn(VoxModelBundle {
            geometry: geometries.add(VoxGeometry::from_tree_with_unit_size(
                snake_tree.freeze(),
                1.0,
            )),
            material: materials.add(VoxMaterial::new()),
            palette: todo!(),
            ..Default::default()
        })
        .id();
}
