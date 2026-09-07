//! A kinematic capsule character controller built on avian's move-and-slide.
//!
//! [`Character`] holds the tuning parameters and the state that carries between
//! ticks. It expects a capsule [`Collider`] on the same entity (see
//! [`Character::collider`]) and moves the entity's [`Transform`] from
//! `FixedUpdate` through [`MoveAndSlide`]; avian's [`TransformInterpolation`]
//! smooths the result between ticks for rendering.
//!
//! [`CharacterCamera`] marks the entity whose [`Transform`] is written every
//! frame from the character's eye position and look angles. In dust that is the
//! primary window entity, which carries the render camera.
//!
//! Controls: WASD to move, Space to jump (or fly up), Left Control to fly down,
//! Left Shift to sprint, F to toggle flying, V to toggle first/third person,
//! Left Click to grab the cursor, Escape to release it. Mouse look only runs
//! while the cursor is grabbed.

use avian3d::character_controller::prelude::*;
use avian3d::math::*;
use avian3d::prelude::*;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

pub struct CharacterPlugin;

impl Plugin for CharacterPlugin {
    fn build(&self, app: &mut App) {
        // Movement runs on the physics tick so move-and-slide sees consistent
        // collider positions; looking runs every frame so the mouse stays smooth.
        app.add_systems(FixedUpdate, move_character)
            .add_systems(Update, look_and_camera);
    }
}

/// How the character moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Gravity, ground detection, jumping and step-up through move-and-slide.
    Walk,
    /// Free flight along the look direction, through everything. A debug aid.
    Fly,
}

/// Where the camera sits relative to the character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    /// The camera sits at the character's eyes.
    FirstPerson,
    /// The camera orbits behind the character, pulled in when geometry is in the way.
    ThirdPerson,
}

/// A kinematic capsule character. Position the entity with a [`Transform`]; the
/// capsule is centred on the translation, so the feet sit at `height / 2` below it.
#[derive(Component)]
#[require(
    RigidBody = RigidBody::Kinematic,
    CustomPositionIntegration,
    TransformInterpolation,
    LinearVelocity
)]
pub struct Character {
    /// Capsule radius.
    pub radius: f32,
    /// Total capsule height, cap to cap.
    pub height: f32,
    /// Eye height above the feet.
    pub eye_height: f32,
    /// Ground speed in units per second.
    pub walk_speed: f32,
    /// Speed multiplier while Left Shift is held.
    pub sprint_multiplier: f32,
    /// How quickly horizontal velocity turns toward the input while airborne,
    /// as a fraction per second. `0.0` gives no air control.
    pub air_control: f32,
    /// Upward velocity given by a jump.
    pub jump_speed: f32,
    /// Downward acceleration while airborne.
    pub gravity: f32,
    /// Tallest ledge the character climbs automatically when walking into it.
    pub step_height: f32,
    /// Speed in units per second in [`Mode::Fly`].
    pub fly_speed: f32,
    /// Radians of look rotation per pixel of mouse movement.
    pub mouse_sensitivity: f32,
    /// Distance from the eyes to the camera in [`View::ThirdPerson`].
    pub third_person_distance: f32,
    /// Smallest `normal.y` a surface can have and still count as ground. `0.7`
    /// is a little under 45 degrees.
    pub min_ground_normal_y: f32,
    /// How far below the capsule the ground probe looks each tick.
    pub ground_probe: f32,

    /// Look yaw in radians; also the body's facing.
    pub yaw: f32,
    /// Look pitch in radians, clamped just short of straight up and down.
    pub pitch: f32,
    pub mode: Mode,
    pub view: View,
    /// Velocity carried into the next tick. On the ground it is replaced by the
    /// input each tick; in the air it accumulates gravity.
    pub velocity: Vec3,
    /// Whether the ground probe found walkable ground on the last tick.
    pub grounded: bool,
    /// Set when Space is pressed, consumed by the next tick. Kept as a flag
    /// because `FixedUpdate` can run zero times in a frame and would miss the press.
    pub jump_requested: bool,
}

impl Default for Character {
    fn default() -> Self {
        Self {
            radius: 0.3,
            height: 1.8,
            eye_height: 1.6,
            walk_speed: 4.0,
            sprint_multiplier: 2.0,
            air_control: 4.0,
            // Reaches about one unit high under the default gravity.
            jump_speed: 6.3,
            gravity: 20.0,
            step_height: 0.3,
            fly_speed: 12.0,
            mouse_sensitivity: 0.002,
            third_person_distance: 4.0,
            min_ground_normal_y: 0.7,
            ground_probe: 0.05,
            yaw: 0.0,
            pitch: 0.0,
            mode: Mode::Walk,
            view: View::FirstPerson,
            velocity: Vec3::ZERO,
            grounded: false,
            jump_requested: false,
        }
    }
}

impl Character {
    /// The capsule matching `radius` and `height`, to insert next to this component.
    pub fn collider(&self) -> Collider {
        Collider::capsule(self.radius, (self.height - 2.0 * self.radius).max(0.0))
    }

    /// Eye position for a character whose capsule centre is at `translation`.
    pub fn eye(&self, translation: Vec3) -> Vec3 {
        translation + Vec3::Y * (self.eye_height - self.height * 0.5)
    }

    /// Look rotation from `yaw` and `pitch`, looking down -Z when both are zero.
    pub fn look_rotation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }
}

/// Marks the entity whose [`Transform`] follows the character `0`'s eyes and look angles.
#[derive(Component, Clone, Copy, Debug)]
pub struct CharacterCamera(pub Entity);

/// Triggered on the character entity when [`Character::view`] changes, so the app
/// can show or hide the character's visible body.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct ViewChanged {
    pub entity: Entity,
    pub view: View,
}

/// Exclude the character and everything spawned under it (its visible body's
/// colliders attach to the character's rigid body) from its own casts.
fn self_filter(entity: Entity, children: &Query<&Children>) -> SpatialQueryFilter {
    SpatialQueryFilter::from_excluded_entities(
        core::iter::once(entity).chain(children.iter_descendants(entity)),
    )
}

fn move_character(
    mut characters: Query<(
        Entity,
        &mut Character,
        &Collider,
        &mut Transform,
        &mut LinearVelocity,
    )>,
    children: Query<&Children>,
    keys: Res<ButtonInput<KeyCode>>,
    move_and_slide: MoveAndSlide,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let mut input = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        input += Vec3::NEG_Z;
    }
    if keys.pressed(KeyCode::KeyS) {
        input += Vec3::Z;
    }
    if keys.pressed(KeyCode::KeyA) {
        input += Vec3::NEG_X;
    }
    if keys.pressed(KeyCode::KeyD) {
        input += Vec3::X;
    }
    let input = input.normalize_or_zero();
    let sprint = keys.pressed(KeyCode::ShiftLeft);

    for (entity, mut character, collider, mut transform, mut linear_velocity) in &mut characters {
        let character = &mut *character;
        let sprint_multiplier = if sprint {
            character.sprint_multiplier
        } else {
            1.0
        };
        let yaw = Quat::from_rotation_y(character.yaw);
        // The body faces the look yaw. Written here rather than every frame because
        // avian's interpolation resets when the transform changes outside the tick.
        transform.rotation = yaw;
        let start = transform.translation;

        if character.mode == Mode::Fly {
            let mut direction = character.look_rotation() * input;
            if keys.pressed(KeyCode::Space) {
                direction += Vec3::Y;
            }
            if keys.pressed(KeyCode::ControlLeft) {
                direction += Vec3::NEG_Y;
            }
            let velocity = direction.normalize_or_zero() * character.fly_speed * sprint_multiplier;
            transform.translation = start + velocity * dt;
            character.velocity = velocity;
            character.grounded = false;
            character.jump_requested = false;
            linear_velocity.0 = velocity;
            continue;
        }

        let wish = yaw * input * character.walk_speed * sprint_multiplier;
        let rotation = transform.rotation;
        let filter = self_filter(entity, &children);
        let config = MoveAndSlideConfig::default();
        let skin_width = move_and_slide.length_unit.0 * config.skin_width;

        // Ground probe: a short downward sweep (`ground_probe` plus the skin
        // width), so any hit is within reach. `MoveHitData` distances are not
        // used: `cast_move` reports the cast's maximum length as
        // `collision_distance`, not the hit's.
        let touching = move_and_slide.cast_move(
            collider,
            start,
            rotation,
            Vec3::NEG_Y * character.ground_probe,
            skin_width,
            &filter,
        );
        let ground = touching.filter(|hit| {
            // A walkable normal is ground. So is a ledge edge under the capsule's
            // rounded bottom, whose contact normal tilts: a ray under the centre
            // finding walkable ground within a step's height keeps the character
            // grounded there, and walking along the tilted plane climbs the edge.
            hit.normal1.y >= character.min_ground_normal_y
                || ground_under(&move_and_slide, start, character, &filter)
        });
        let grounded = ground.is_some();

        let mut velocity = character.velocity;
        if let Some(ground) = ground {
            // Walk along the contact plane so slopes neither launch the capsule
            // nor let it float, keeping the input speed along the surface.
            let normal = ground.normal1;
            let along = wish - normal * normal.dot(wish);
            velocity = along.normalize_or_zero() * wish.length();
            // A slight push into the ground; move-and-slide clips it against the
            // ground plane, which keeps the capsule seated over small dips.
            velocity.y -= 1.0;
            if character.jump_requested {
                velocity.y = character.jump_speed;
            }
        } else {
            velocity.y -= character.gravity * dt;
            let horizontal = Vec3::new(velocity.x, 0.0, velocity.z);
            velocity += (wish - horizontal) * (character.air_control * dt).min(1.0);
        }
        character.jump_requested = false;

        // Main move. Note any surface too steep to stand on: that is a wall or a
        // ledge face, and the candidate for a step-up.
        let mut blocked = false;
        let out = move_and_slide.move_and_slide(
            collider,
            start,
            rotation,
            velocity,
            time.delta(),
            &config,
            &filter,
            |hit| {
                if hit.normal.y < character.min_ground_normal_y {
                    blocked = true;
                }
                MoveAndSlideHitResponse::Accept
            },
        );
        let mut position = out.position;
        let mut new_velocity = out.projected_velocity;

        if blocked && grounded && wish != Vec3::ZERO {
            if let Some((stepped_position, stepped_velocity)) = step_up(
                &move_and_slide,
                collider,
                start,
                rotation,
                Vec3::new(velocity.x, 0.0, velocity.z),
                time.delta(),
                &config,
                skin_width,
                &filter,
                character,
                position,
            ) {
                position = stepped_position;
                new_velocity = stepped_velocity;
            }
        }

        transform.translation = position;
        character.velocity = new_velocity;
        character.grounded = grounded;
        // What the contact solver sees when a dynamic body touches the capsule.
        linear_velocity.0 = (position - start) / dt;
    }
}

/// Whether a ray straight down from the capsule centre at `position` hits
/// walkable ground no further below the feet than `step_height`.
fn ground_under(
    move_and_slide: &MoveAndSlide,
    position: Vec3,
    character: &Character,
    filter: &SpatialQueryFilter,
) -> bool {
    let reach = character.height * 0.5 + character.step_height + character.ground_probe;
    move_and_slide
        .spatial_query
        .cast_ray(position, Dir3::NEG_Y, reach, true, filter)
        .is_some_and(|hit| hit.normal.y >= character.min_ground_normal_y)
}

/// Try the move again from `step_height` higher: sweep up, slide horizontally,
/// sweep back down. Accept it when the raised capsule has a full radius of room
/// ahead (a step, not a wall), lands on something, and ends up higher than it
/// started. The landing's contact normal is not consulted: the rounded bottom
/// first meets a ledge's front edge, whose normal tilts; `ground_under` then
/// keeps the character grounded while it walks up over the edge.
#[allow(clippy::too_many_arguments)]
fn step_up(
    move_and_slide: &MoveAndSlide,
    collider: &Collider,
    start: Vec3,
    rotation: Quat,
    velocity: Vec3,
    delta: core::time::Duration,
    config: &MoveAndSlideConfig,
    skin_width: f32,
    filter: &SpatialQueryFilter,
    character: &Character,
    slide_end: Vec3,
) -> Option<(Vec3, Vec3)> {
    let up = move_and_slide
        .cast_move(
            collider,
            start,
            rotation,
            Vec3::Y * character.step_height,
            skin_width,
            filter,
        )
        .map_or(character.step_height, |hit| hit.distance);
    if up <= 0.0 {
        return None;
    }
    let raised = start + Vec3::Y * up;
    let Ok(forward) = Dir3::new(velocity) else {
        return None;
    };
    if move_and_slide
        .cast_move(
            collider,
            raised,
            rotation,
            forward * (character.radius + skin_width),
            skin_width,
            filter,
        )
        .is_some()
    {
        return None;
    }
    let out = move_and_slide.move_and_slide(
        collider,
        raised,
        rotation,
        velocity,
        delta,
        config,
        filter,
        |_| MoveAndSlideHitResponse::Accept,
    );
    let down = move_and_slide.cast_move(
        collider,
        out.position,
        rotation,
        Vec3::NEG_Y * (up + character.ground_probe),
        skin_width,
        filter,
    )?;
    let landed = out.position - Vec3::Y * down.distance;
    let progress = |end: Vec3| (end - start).xz().length_squared();
    let accepted = landed.y > start.y + 1e-3 && progress(landed) > progress(slide_end) + 1e-6;
    accepted.then(|| {
        (
            landed,
            Vec3::new(out.projected_velocity.x, 0.0, out.projected_velocity.z),
        )
    })
}

fn look_and_camera(
    mut characters: Query<(&mut Character, &Transform)>,
    mut cameras: Query<(&CharacterCamera, &mut Transform), Without<Character>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    children: Query<&Children>,
    spatial_query: SpatialQuery,
    mut commands: Commands,
) {
    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    } else if mouse_buttons.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
    let grabbed = cursor.grab_mode != CursorGrabMode::None;

    for (camera, mut camera_transform) in &mut cameras {
        let Ok((mut character, transform)) = characters.get_mut(camera.0) else {
            continue;
        };
        let character = &mut *character;

        if grabbed {
            const PITCH_LIMIT: f32 = core::f32::consts::FRAC_PI_2 - 0.01;
            character.yaw -= mouse_motion.delta.x * character.mouse_sensitivity;
            character.pitch = (character.pitch
                - mouse_motion.delta.y * character.mouse_sensitivity)
                .clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }
        if keys.just_pressed(KeyCode::Space) {
            character.jump_requested = true;
        }
        if keys.just_pressed(KeyCode::KeyF) {
            character.mode = match character.mode {
                Mode::Walk => Mode::Fly,
                Mode::Fly => Mode::Walk,
            };
        }
        if keys.just_pressed(KeyCode::KeyV) {
            character.view = match character.view {
                View::FirstPerson => View::ThirdPerson,
                View::ThirdPerson => View::FirstPerson,
            };
            commands.trigger(ViewChanged {
                entity: camera.0,
                view: character.view,
            });
        }

        let look = character.look_rotation();
        let eye = character.eye(transform.translation);
        camera_transform.rotation = look;
        camera_transform.translation = match character.view {
            View::FirstPerson => eye,
            View::ThirdPerson => {
                // Pull the camera in if something sits between it and the eyes,
                // with a margin so the near plane stays clear of the surface.
                const MARGIN: f32 = 0.2;
                let back = look * Vec3::Z;
                let max_distance = character.third_person_distance;
                let distance = spatial_query
                    .cast_ray(
                        eye,
                        Dir3::new_unchecked(back),
                        max_distance,
                        true,
                        &self_filter(camera.0, &children),
                    )
                    .map_or(max_distance, |hit| (hit.distance - MARGIN).max(0.0));
                eye + back * distance
            }
        };
    }
}
