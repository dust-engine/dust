#![feature(impl_trait_in_fn_trait_return)]

mod flycam;

use std::time::Duration;

use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use bevy_pumicite::rtx::tlas::TLASInstance;
use dust_vox::{
    VoxGeometry, VoxInstance, VoxInstanceBundle, VoxMaterial, VoxModel, VoxModelBLASRebuild,
    VoxModelBundle, VoxPalette,
};
use pumicite::{Allocator, ash::vk, swapchain::SwapchainColorMode};

use crate::flycam::{FlyCamera, FlyCameraPlugin};

#[derive(Component)]
struct MovingTeapot {
    origin: Vec3,
    radius: f32,
    height: f32,
    angular_speed: f32,
    spin_speed: f32,
}

const RAINBOW_INNER_RADIUS: f32 = 60.0;
const RAINBOW_STRIPE_THICKNESS: f32 = 4.0;
const RAINBOW_NUM_STRIPES: u32 = 7;
const RAINBOW_DEPTH: u32 = 6;
const RAINBOW_WEDGES: u32 = 64;
const RAINBOW_HOLD_TICKS: u32 = 24;
const RAINBOW_TICK_SECONDS: f32 = 0.01;
const RAINBOW_WORLD_TRANSLATION: Vec3 = Vec3::new(26.0, 24.0, 24.0);

// Palette indices into VoxPalette::colorful() approximating ROYGBIV.
// VoxMaterial stores `value - 1` as the palette index, and value 0 is the
// "empty" sentinel — so each entry here is (palette_index + 1).
const RAINBOW_STRIPE_VALUES: [u8; RAINBOW_NUM_STRIPES as usize] = [1, 22, 43, 86, 142, 185, 206];

#[derive(Resource)]
struct RainbowDemo {
    model_entity: Entity,
    geometry: Handle<VoxGeometry>,
    material: Handle<VoxMaterial>,
    palette: Handle<VoxPalette>,
    progress: u32,
    hold_remaining: u32,
    timer: Timer,
}

pub fn run() {
    let mut app = bevy::app::App::new();

    app.add_plugins(dust_app::DustApp)
        .add_plugins(bevy::DefaultPlugins)
        .add_plugins(bevy_pumicite::SurfacePlugin::default())
        .add_plugins(bevy_pumicite::DebugUtilsPlugin::default())
        .add_plugins(dust_pbr::super_resolution::SuperResolutionSetupPlugin)
        .add_plugins(bevy_pumicite::PumicitePlugin::default())
        .add_plugins(bevy_pumicite::swapchain::SwapchainPlugin);

    app.add_plugins(FlyCameraPlugin);

    // Dust plugins
    app.add_plugins(dust_pbr::PbrRenderPlugin)
        .add_plugins(dust_vox::VoxPlugin)
        .add_plugins(dust_gfxdebug::GfxDebugPlugin);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert((
            dust_pbr::camera::Camera::default(),
            GlobalTransform::default(),
            Transform::from_translation(Vec3::new(12.2, 30.61, 14.45)),
            FlyCamera::default(),
        ))
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            color_mode: SwapchainColorMode::HDR,
            ..Default::default()
        });

    app.add_systems(
        Startup,
        (startup_system, setup_rainbow_demo).after(CreateDevice),
    )
    .add_systems(Update, (animate_teapot_system));

    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>) {
    let scene: Handle<Scene> = asset_server.load("bazel://dust/assets/castle.vox");
    commands.spawn(SceneRoot(scene));

    let teapot_origin = Vec3::new(12.2, 26.0, 18.0);
    let mut teapot_transform = Transform::from_translation(teapot_origin);
    teapot_transform.scale = Vec3::splat(1.5);

    let teapot: Handle<Scene> = asset_server.load("bazel://dust/assets/teapot.vox");
    commands.spawn((
        SceneRoot(teapot),
        teapot_transform,
        GlobalTransform::default(),
        MovingTeapot {
            origin: teapot_origin,
            radius: 56.0,
            height: 18.0,
            angular_speed: 0.6,
            spin_speed: 1.4,
        },
    ));
    return;
}

fn animate_teapot_system(time: Res<Time>, mut teapots: Query<(&MovingTeapot, &mut Transform)>) {
    let elapsed = time.elapsed_secs();

    for (teapot, mut transform) in teapots.iter_mut() {
        let angle = elapsed * teapot.angular_speed;
        transform.translation = teapot.origin
            + Vec3::new(
                angle.cos() * teapot.radius,
                (angle * 1.7).sin() * teapot.height,
                angle.sin() * teapot.radius,
            );
        transform.rotation = Quat::from_rotation_y(elapsed * teapot.spin_speed);
    }
}

fn rainbow_max_radius() -> f32 {
    RAINBOW_INNER_RADIUS + RAINBOW_STRIPE_THICKNESS * RAINBOW_NUM_STRIPES as f32
}

fn rainbow_origin_offset() -> f32 {
    rainbow_max_radius() + 1.0
}

/// Paint one wedge of the rainbow (all stripes) into `geometry`/`material` at
/// the given progress index. The caller is responsible for triggering the BLAS
/// rebuild afterwards.
fn paint_rainbow_wedge(geometry: &mut VoxGeometry, material: &mut VoxMaterial, progress: u32) {
    let theta_lo = (progress as f32) * std::f32::consts::PI / RAINBOW_WEDGES as f32;
    let theta_hi = (progress as f32 + 1.0) * std::f32::consts::PI / RAINBOW_WEDGES as f32;
    let origin_x = rainbow_origin_offset();

    let mut accessor = geometry.tree.accessor_mut(material);
    for stripe in 0..RAINBOW_NUM_STRIPES {
        let r_inner = RAINBOW_INNER_RADIUS + stripe as f32 * RAINBOW_STRIPE_THICKNESS;
        let r_outer = r_inner + RAINBOW_STRIPE_THICKNESS;
        let value = RAINBOW_STRIPE_VALUES[stripe as usize];

        // Step at <= 0.5 voxel along the outermost arc so the stripe fills solid.
        let arc_len = r_outer * (theta_hi - theta_lo);
        let n_angle = (arc_len * 2.0).ceil().max(1.0) as u32;

        for ai in 0..=n_angle {
            let t = ai as f32 / n_angle as f32;
            let theta = theta_lo + (theta_hi - theta_lo) * t;
            let (sin_t, cos_t) = theta.sin_cos();

            let r_steps = (r_outer - r_inner).ceil() as u32 + 1;
            for ri in 0..r_steps {
                let r = r_inner + ri as f32 * (r_outer - r_inner) / r_steps.max(1) as f32;
                let x = (origin_x + r * cos_t).round() as i32;
                let y = (r * sin_t).round() as i32;
                if x < 0 || y < 0 {
                    continue;
                }
                for z in 0..RAINBOW_DEPTH {
                    accessor.set(UVec3::new(x as u32, y as u32, z), value);
                }
            }
        }
    }
    drop(accessor);
}

fn setup_rainbow_demo(
    mut commands: Commands,
    allocator: Res<Allocator>,
    mut geometries: ResMut<Assets<VoxGeometry>>,
    mut materials: ResMut<Assets<VoxMaterial>>,
    mut palettes: ResMut<Assets<VoxPalette>>,
) {
    // Paint the first wedge before the model is spawned so the BLAS builder
    // never sees empty geometry.
    let mut geometry = VoxGeometry::new(allocator.clone(), 1.0);
    let mut material = VoxMaterial::new(allocator.clone());
    paint_rainbow_wedge(&mut geometry, &mut material, 0);
    let geometry = geometries.add(geometry);
    let material = materials.add(material);
    let palette =
        palettes.add(VoxPalette::colorful(allocator.clone()).expect("rainbow palette allocation"));

    let model_entity = commands
        .spawn(VoxModelBundle {
            model: VoxModel {
                geometry: geometry.clone(),
                material: material.clone(),
                palette: palette.clone(),
                sbt_index: u32::MAX,
                prefer_fast_build: true,
                enable_compaction: false,
            },
            ..Default::default()
        })
        .id();

    commands.spawn(VoxInstanceBundle {
        transform: Transform::from_translation(RAINBOW_WORLD_TRANSLATION),
        global_transform: GlobalTransform::default(),
        instance: VoxInstance,
        tlas_instance: TLASInstance::new(model_entity),
    });

    commands.insert_resource(RainbowDemo {
        model_entity,
        geometry,
        material,
        palette,
        // Wedge 0 was painted above before spawning.
        progress: 1,
        hold_remaining: 0,
        timer: Timer::new(
            Duration::from_secs_f32(RAINBOW_TICK_SECONDS),
            TimerMode::Repeating,
        ),
    });
}

fn update_rainbow_demo_system(
    mut commands: Commands,
    time: Res<Time>,
    allocator: Res<Allocator>,
    mut demo: ResMut<RainbowDemo>,
    mut geometries: ResMut<Assets<VoxGeometry>>,
    mut materials: ResMut<Assets<VoxMaterial>>,

    mut requesting_blas_rebuilds: Query<&mut VoxModelBLASRebuild>,
) {
    if !demo.timer.tick(time.delta()).just_finished() {
        return;
    }

    if demo.progress >= RAINBOW_WEDGES {
        if demo.hold_remaining > 0 {
            demo.hold_remaining -= 1;
            return;
        }
        // Start a new pass. Paint the first wedge before swapping the geometry
        // in so the rebuilt BLAS is never empty, then drive the rebuild through
        // `request_rebuild()` (below) like every other wedge does.
        let mut new_geometry = VoxGeometry::new(allocator.clone(), 1.0);
        let mut new_material = VoxMaterial::new(allocator.clone());
        paint_rainbow_wedge(&mut new_geometry, &mut new_material, 0);
        let new_geometry = geometries.add(new_geometry);
        let new_material = materials.add(new_material);
        demo.geometry = new_geometry.clone();
        demo.material = new_material.clone();
        demo.progress = 1;
        commands.entity(demo.model_entity).insert(VoxModel {
            geometry: new_geometry,
            material: new_material,
            palette: demo.palette.clone(),
            sbt_index: u32::MAX,
            prefer_fast_build: true,
            enable_compaction: false,
        });
        requesting_blas_rebuilds
            .get_mut(demo.model_entity)
            .unwrap()
            .request_rebuild();
        return;
    }

    let Some(geometry) = geometries.get_mut(&demo.geometry) else {
        return;
    };
    let Some(material) = materials.get_mut(&demo.material) else {
        return;
    };

    paint_rainbow_wedge(geometry, material, demo.progress);

    demo.progress += 1;
    if demo.progress >= RAINBOW_WEDGES {
        demo.hold_remaining = RAINBOW_HOLD_TICKS;
    }

    // Request a BLAS rebuild now that this wedge has been painted into the
    // geometry.
    requesting_blas_rebuilds
        .get_mut(demo.model_entity)
        .unwrap()
        .request_rebuild();
}
