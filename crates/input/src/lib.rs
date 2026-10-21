//! The game's keyboard and mouse, read from the platform and written as the
//! messages `bevy_input` reads (`KeyboardInput`, `MouseButtonInput`,
//! `MouseMotion`, `MouseWheel`).
//!
//! The game gets them only while the cursor is attached ([`CursorMode`]). A
//! backend plugin reads one platform's devices while it is attached and hands
//! each event to [`InputWriter`]:
//! - macOS: [`GameControllerPlugin`](game_controller::GameControllerPlugin).

#[cfg(target_os = "macos")]
pub mod game_controller;

use bevy::ecs::system::SystemParam;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyCode, KeyboardInput, NativeKey};
use bevy::input::mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::prelude::*;

/// The [`CursorMode`] state, and the backend for this platform. Add it after
/// `StatesPlugin`.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<CursorMode>()
            .configure_sets(PreUpdate, InputSet.before(bevy::input::InputSystems))
            .add_systems(OnExit(CursorMode::Attached), release_held);
        #[cfg(target_os = "macos")]
        app.add_plugins(game_controller::GameControllerPlugin);
    }
}

/// Where input is read: in [`PreUpdate`], before `bevy_input` turns the
/// messages into `ButtonInput` and the accumulators in [`InputSystems`].
/// Backends run in it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputSet;

/// Whether the cursor is the game's. While it is attached, the game gets the
/// keyboard and mouse; while it is detached, they are the UI's, and the game
/// gets none of them. The UI sets it (`dust_ui`'s `BevyView`), and locks and
/// hides the cursor while it is attached: it attaches the cursor when the game
/// is clicked, and detaches it on Escape and when the window stops being the
/// active one.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CursorMode {
    Attached,
    #[default]
    Detached,
}

/// Releases the keys and buttons held when the cursor is detached. The backend
/// stops reading then, so their releases would never come, and they would stay
/// held until pressed again.
fn release_held(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut input: InputWriter,
) {
    for &key in keys.get_pressed() {
        input.key(key, false);
    }
    for &button in buttons.get_pressed() {
        input.button(button, false);
    }
}

/// Writes a backend's events as `bevy_input` messages. There are no Bevy
/// windows, so the messages name [`Entity::PLACEHOLDER`] as theirs.
#[derive(SystemParam)]
pub struct InputWriter<'w> {
    keyboard: MessageWriter<'w, KeyboardInput>,
    mouse_buttons: MessageWriter<'w, MouseButtonInput>,
    mouse_motion: MessageWriter<'w, MouseMotion>,
    mouse_wheel: MessageWriter<'w, MouseWheel>,
}

impl InputWriter<'_> {
    /// A physical key. The platform reports no layout, so the logical key is
    /// `Unidentified`.
    pub fn key(&mut self, key_code: KeyCode, pressed: bool) {
        self.keyboard.write(KeyboardInput {
            key_code,
            logical_key: Key::Unidentified(NativeKey::Unidentified),
            state: button_state(pressed),
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        });
    }

    pub fn button(&mut self, button: MouseButton, pressed: bool) {
        self.mouse_buttons.write(MouseButtonInput {
            button,
            state: button_state(pressed),
            window: Entity::PLACEHOLDER,
        });
    }

    /// Raw mouse motion, y down.
    pub fn motion(&mut self, delta: Vec2) {
        self.mouse_motion.write(MouseMotion { delta });
    }

    /// Scrolling, y up.
    pub fn scroll(&mut self, delta: Vec2, unit: MouseScrollUnit) {
        self.mouse_wheel.write(MouseWheel {
            unit,
            x: delta.x,
            y: delta.y,
            window: Entity::PLACEHOLDER,
            phase: TouchPhase::Moved,
        });
    }
}

fn button_state(pressed: bool) -> ButtonState {
    if pressed {
        ButtonState::Pressed
    } else {
        ButtonState::Released
    }
}
