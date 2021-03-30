use bevy::prelude::*;

use super::util::movement_axis;

/// A set of options for initializing a FlyCamera.
/// Attach this component to a [`Camera2dBundle`](https://docs.rs/bevy/0.4.0/bevy/prelude/struct.Camera2dBundle.html) bundle to control it with your keyboard.
/// # Example
/// ```no_compile
/// fn setup(mut commands: Commands) {
///	  commands
///     .spawn(Camera2dBundle::default())
///     .with(FlyCamera2d::default());
/// }
pub struct FlyCamera2d {
    /// The speed the FlyCamera2d accelerates at.
    pub accel: f32,
    /// The maximum speed the FlyCamera can move at.
    pub max_speed: f32,
    /// The amount of deceleration to apply to the camera's motion.
    pub friction: f32,
    /// The current velocity of the FlyCamera2d. This value is always up-to-date, enforced by [FlyCameraPlugin](struct.FlyCameraPlugin.html)
    pub velocity: Vec2,
    /// Key used to move left. Defaults to <kbd>A</kbd>
    pub key_left: KeyCode,
    /// Key used to move right. Defaults to <kbd>D</kbd>
    pub key_right: KeyCode,
    /// Key used to move up. Defaults to <kbd>W</kbd>
    pub key_up: KeyCode,
    /// Key used to move forward. Defaults to <kbd>S</kbd>
    pub key_down: KeyCode,
    /// If `false`, disable keyboard control of the camera. Defaults to `true`
    pub enabled: bool,
}

impl Default for FlyCamera2d {
    fn default() -> Self {
        const MUL_2D: f32 = 10.0;
        Self {
            accel: 3.0 * MUL_2D,
            max_speed: 1.0 * MUL_2D,
            friction: 1.75 * MUL_2D,
            velocity: Vec2::ZERO,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_up: KeyCode::W,
            key_down: KeyCode::S,
            enabled: true,
        }
    }
}

pub fn camera_2d_movement_system(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mut query: Query<(&mut FlyCamera2d, &mut Transform)>,
) {
    for (mut options, mut transform) in query.iter_mut() {
        let (axis_h, axis_v) = if options.enabled {
            (
                movement_axis(&keyboard_input, options.key_right, options.key_left),
                movement_axis(&keyboard_input, options.key_up, options.key_down),
            )
        } else {
            (0.0, 0.0)
        };

        let accel: Vec2 = (Vec2::X * axis_h) + (Vec2::Y * axis_v);
        let accel: Vec2 = if accel.length() != 0.0 {
            accel.normalize() * options.accel
        } else {
            Vec2::ZERO
        };

        let friction: Vec2 = if options.velocity.length() != 0.0 {
            options.velocity.normalize() * -1.0 * options.friction
        } else {
            Vec2::ZERO
        };

        options.velocity += accel * time.delta_seconds();

        // clamp within max speed
        if options.velocity.length() > options.max_speed {
            options.velocity = options.velocity.normalize() * options.max_speed;
        }

        let delta_friction = friction * time.delta_seconds();

        options.velocity =
            if (options.velocity + delta_friction).signum() != options.velocity.signum() {
                Vec2::ZERO
            } else {
                options.velocity + delta_friction
            };

        transform.translation += Vec3::new(options.velocity.x, options.velocity.y, 0.0);
    }
}
