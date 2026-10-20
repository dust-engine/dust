//! The game's keyboard and mouse, read from the platform and written as the
//! messages `bevy_input` reads (`KeyboardInput`, `MouseButtonInput`,
//! `MouseMotion`, `MouseWheel`).
//!
//! The platform reports every key and click while the app is active, including
//! typing into a UI text field and clicks on a UI panel. Which of them are the
//! game's is up to the UI, which sets [`InputGate`]. A backend plugin reads one
//! platform's devices and hands each event to [`GatedInput`], which keeps what
//! the gate lets through:
//! - macOS: [`GameControllerPlugin`](game_controller::GameControllerPlugin).

#[cfg(target_os = "macos")]
pub mod game_controller;

use bevy::ecs::system::SystemParam;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyCode, KeyboardFocusLost, KeyboardInput, NativeKey};
use bevy::input::mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

/// The gate, and the backend for this platform.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InputGate>()
            .configure_sets(PreUpdate, InputSet.before(bevy::input::InputSystems))
            .add_systems(PreUpdate, release_inactive.in_set(InputSet));
        #[cfg(target_os = "macos")]
        app.add_plugins(game_controller::GameControllerPlugin);
    }
}

/// Where input is read: in [`PreUpdate`], before `bevy_input` turns the
/// messages into `ButtonInput` and the accumulators in [`InputSystems`].
/// Backends run in it, after [`release_inactive`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputSet;

/// Which presses and scrolls are the game's. While the primary window's cursor
/// is locked, all of them are, as long as the window is active.
///
/// Releases always reach the game: `ButtonInput` ignores one with no press.
/// Mouse motion always does too: `MouseMotion` is the device's raw motion,
/// whatever the pointer is over.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InputGate {
    /// The window is the active one. When it stops being, every held key and
    /// button is released: the platform reports nothing while it is inactive,
    /// so their releases would never come.
    pub active: bool,
    /// Keys are the game's.
    pub keys: bool,
    /// The pointer is over the game, so clicks and scrolling are the game's.
    pub pointer: bool,
}

/// Releases every held key and button while the window is inactive. Backends
/// run after it.
pub(crate) fn release_inactive(
    gate: Res<InputGate>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    mut focus_lost: MessageWriter<KeyboardFocusLost>,
    mut mouse_buttons: MessageWriter<MouseButtonInput>,
) {
    if gate.active {
        return;
    }
    let Ok(window) = primary_window.single() else {
        return;
    };
    if keys.get_pressed().next().is_some() {
        focus_lost.write(KeyboardFocusLost);
    }
    // `bevy_input` has no `KeyboardFocusLost` for the mouse.
    for &button in buttons.get_pressed() {
        mouse_buttons.write(MouseButtonInput {
            button,
            state: ButtonState::Released,
            window,
        });
    }
}

impl InputGate {
    fn keys_to_game(&self, cursor: &CursorOptions) -> bool {
        self.active && (locked(cursor) || self.keys)
    }

    fn pointer_to_game(&self, cursor: &CursorOptions) -> bool {
        self.active && (locked(cursor) || self.pointer)
    }
}

fn locked(cursor: &CursorOptions) -> bool {
    cursor.grab_mode != CursorGrabMode::None
}

/// Writes a backend's events as `bevy_input` messages, keeping what the
/// [`InputGate`] lets through. Events other than motion are dropped while there
/// is no primary window.
#[derive(SystemParam)]
pub struct GatedInput<'w, 's> {
    gate: Res<'w, InputGate>,
    primary_window: Query<'w, 's, (Entity, &'static CursorOptions), With<PrimaryWindow>>,
    keyboard: MessageWriter<'w, KeyboardInput>,
    mouse_buttons: MessageWriter<'w, MouseButtonInput>,
    mouse_motion: MessageWriter<'w, MouseMotion>,
    mouse_wheel: MessageWriter<'w, MouseWheel>,
}

impl GatedInput<'_, '_> {
    /// A physical key. The platform reports no layout, so the logical key is
    /// `Unidentified`.
    pub fn key(&mut self, key_code: KeyCode, pressed: bool) {
        let Ok((window, cursor)) = self.primary_window.single() else {
            return;
        };
        if !pressed || self.gate.keys_to_game(cursor) {
            self.keyboard.write(KeyboardInput {
                key_code,
                logical_key: Key::Unidentified(NativeKey::Unidentified),
                state: if pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                },
                text: None,
                repeat: false,
                window,
            });
        }
    }

    pub fn button(&mut self, button: MouseButton, pressed: bool) {
        let Ok((window, cursor)) = self.primary_window.single() else {
            return;
        };
        if !pressed || self.gate.pointer_to_game(cursor) {
            self.mouse_buttons.write(MouseButtonInput {
                button,
                state: if pressed {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                },
                window,
            });
        }
    }

    /// Raw mouse motion, y down.
    pub fn motion(&mut self, delta: Vec2) {
        self.mouse_motion.write(MouseMotion { delta });
    }

    /// Scrolling, y up.
    pub fn scroll(&mut self, delta: Vec2, unit: MouseScrollUnit) {
        let Ok((window, cursor)) = self.primary_window.single() else {
            return;
        };
        if self.gate.pointer_to_game(cursor) {
            self.mouse_wheel.write(MouseWheel {
                unit,
                x: delta.x,
                y: delta.y,
                window,
                phase: TouchPhase::Moved,
            });
        }
    }
}
