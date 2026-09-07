//! Physics-only checks of the controller: no window, no renderer. Time is
//! stepped manually so every `app.update()` is exactly one 64 Hz physics tick.

use std::time::Duration;

use avian3d::prelude::*;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy::window::{CursorOptions, PrimaryWindow};
use dust_character::{Character, CharacterCamera, CharacterPlugin, Mode};

const TICK: f64 = 1.0 / 64.0;
/// Where the capsule centre rests on the floor (top face at y = 0).
const STANDING_Y: f32 = 0.9;

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        PhysicsPlugins::default(),
        CharacterPlugin,
    ));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
        TICK,
    )));
    app.init_resource::<ButtonInput<KeyCode>>();
    app.init_resource::<ButtonInput<MouseButton>>();
    app.init_resource::<AccumulatedMouseMotion>();
    app.world_mut()
        .spawn((PrimaryWindow, Window::default(), CursorOptions::default()));
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(40.0, 1.0, 40.0),
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    // `App::update` alone never finishes plugins; avian registers its
    // diagnostics resources in `Plugin::finish`.
    while app.plugins_state() == bevy::app::PluginsState::Adding {}
    app.finish();
    app.cleanup();
    app
}

fn block(app: &mut App, center: Vec3, size: Vec3) {
    app.world_mut().spawn((
        RigidBody::Static,
        Collider::cuboid(size.x, size.y, size.z),
        Transform::from_translation(center),
    ));
}

fn spawn_character(app: &mut App, translation: Vec3) -> Entity {
    let character = Character::default();
    let collider = character.collider();
    let entity = app
        .world_mut()
        .spawn((
            character,
            collider,
            Transform::from_translation(translation),
        ))
        .id();
    app.world_mut()
        .spawn((CharacterCamera(entity), Transform::default()));
    entity
}

/// Run `ticks` updates with `held` keys down and `tapped` keys pressed on the first tick only.
fn run(app: &mut App, ticks: usize, held: &[KeyCode], tapped: &[KeyCode]) {
    for i in 0..ticks {
        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.clear();
            for key in held {
                keys.press(*key);
            }
            for key in tapped {
                if i == 0 {
                    keys.press(*key);
                } else {
                    keys.release(*key);
                }
            }
        }
        app.update();
    }
    let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
    keys.reset_all();
}

fn position(app: &mut App, entity: Entity) -> Vec3 {
    app.world().get::<Transform>(entity).unwrap().translation
}

fn character(app: &mut App, entity: Entity) -> &Character {
    app.world().get::<Character>(entity).unwrap()
}

#[test]
fn settles_on_floor() {
    let mut app = app();
    let player = spawn_character(&mut app, Vec3::new(0.0, 3.0, 0.0));
    run(&mut app, 200, &[], &[]);
    let p = position(&mut app, player);
    assert!((p.y - STANDING_Y).abs() < 0.05, "resting height {}", p.y);
    assert!(character(&mut app, player).grounded);
}

#[test]
fn walks_and_climbs_a_step() {
    let mut app = app();
    // A 0.25-high ledge covering z in [-20, -1], below the 0.3 step height.
    block(
        &mut app,
        Vec3::new(0.0, 0.125, -10.5),
        Vec3::new(10.0, 0.25, 19.0),
    );
    let player = spawn_character(&mut app, Vec3::new(0.0, STANDING_Y, 0.0));
    run(&mut app, 20, &[], &[]);
    run(&mut app, 128, &[KeyCode::KeyW], &[]);
    let p = position(&mut app, player);
    assert!(p.z < -5.0, "should have walked onto the ledge, z = {}", p.z);
    assert!(
        (p.y - (STANDING_Y + 0.25)).abs() < 0.05,
        "should stand on the ledge, y = {}",
        p.y
    );
    assert!(character(&mut app, player).grounded);
}

#[test]
fn walls_block_walking() {
    let mut app = app();
    // A 3-high wall whose near face is at z = -2.5.
    block(
        &mut app,
        Vec3::new(0.0, 1.5, -3.0),
        Vec3::new(10.0, 3.0, 1.0),
    );
    let player = spawn_character(&mut app, Vec3::new(0.0, STANDING_Y, 0.0));
    run(&mut app, 20, &[], &[]);
    run(&mut app, 128, &[KeyCode::KeyW], &[]);
    let p = position(&mut app, player);
    assert!(p.z > -2.5 + 0.3 - 0.02, "walked into the wall, z = {}", p.z);
    assert!(p.z < -1.5, "should have reached the wall, z = {}", p.z);
    assert!(
        (p.y - STANDING_Y).abs() < 0.05,
        "climbed the wall, y = {}",
        p.y
    );
}

#[test]
fn jumps_and_lands() {
    let mut app = app();
    let player = spawn_character(&mut app, Vec3::new(0.0, STANDING_Y, 0.0));
    run(&mut app, 20, &[], &[]);
    let mut peak = 0.0f32;
    run(&mut app, 1, &[], &[KeyCode::Space]);
    for _ in 0..100 {
        run(&mut app, 1, &[], &[]);
        peak = peak.max(position(&mut app, player).y);
    }
    assert!(peak > STANDING_Y + 0.7, "jump peaked at {}", peak);
    let p = position(&mut app, player);
    assert!((p.y - STANDING_Y).abs() < 0.05, "did not land, y = {}", p.y);
    assert!(character(&mut app, player).grounded);
}

#[test]
fn fly_mode_passes_through_walls() {
    let mut app = app();
    block(
        &mut app,
        Vec3::new(0.0, 1.5, -3.0),
        Vec3::new(10.0, 3.0, 1.0),
    );
    let player = spawn_character(&mut app, Vec3::new(0.0, STANDING_Y, 0.0));
    run(&mut app, 20, &[], &[]);
    run(&mut app, 1, &[], &[KeyCode::KeyF]);
    assert_eq!(character(&mut app, player).mode, Mode::Fly);
    run(&mut app, 64, &[KeyCode::KeyW], &[]);
    let p = position(&mut app, player);
    assert!(
        p.z < -4.0,
        "should have flown through the wall, z = {}",
        p.z
    );
    assert!(
        (p.y - STANDING_Y).abs() < 0.05,
        "fell while flying, y = {}",
        p.y
    );
}

#[test]
fn ledges_above_step_height_block() {
    let mut app = app();
    // A 0.5-high ledge, above the 0.3 step height.
    block(
        &mut app,
        Vec3::new(0.0, 0.25, -10.5),
        Vec3::new(10.0, 0.5, 19.0),
    );
    let player = spawn_character(&mut app, Vec3::new(0.0, STANDING_Y, 0.0));
    run(&mut app, 20, &[], &[]);
    run(&mut app, 128, &[KeyCode::KeyW], &[]);
    let p = position(&mut app, player);
    assert!(p.z > -1.0, "climbed a ledge above step height, z = {}", p.z);
    assert!(
        (p.y - STANDING_Y).abs() < 0.05,
        "left the floor, y = {}",
        p.y
    );
}
