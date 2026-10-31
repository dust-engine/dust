//! The keyboard and mouse on Windows, read with GameInput.
//!
//! GameInput keeps the recent readings of each keyboard and mouse. A reading is
//! the device's state (the keys held, the buttons held, how far the mouse has
//! moved since it connected). Once per frame, [`write_device_changes`] goes
//! through the readings since the last frame and compares each with its
//! device's [`DeviceState`], writing the changes to [`InputWriter`]. It compares
//! whether or not the cursor is attached, so the states are current when it is
//! attached, but writes only while it is.
//!
//! GameInput is the redistributable runtime (`winget install
//! Microsoft.GameInput`), loaded at startup; without it, the game gets no
//! keyboard or mouse.

mod bindings;

use std::collections::{BTreeSet, HashMap};

use bevy::input::keyboard::NativeKeyCode;
use bevy::input::mouse::MouseScrollUnit;
use bevy::prelude::*;
use windows_core::Interface;

use self::bindings::*;
use crate::{CursorMode, InputSet, InputWriter};

/// How far the wheel turns in one notch (Win32's `WHEEL_DELTA`).
const WHEEL_DELTA: f32 = 120.0;

/// Reads GameInput's keyboards and mice.
pub struct GameInputPlugin;

impl Plugin for GameInputPlugin {
    fn build(&self, app: &mut App) {
        match DeviceTracker::new() {
            Ok(device_tracker) => {
                app.insert_resource(device_tracker)
                    .add_systems(PreUpdate, write_device_changes.in_set(InputSet));
            }
            Err(error) => tracing::error!(
                "GameInput is unavailable, so the game gets no keyboard or mouse \
                 (install it with `winget install Microsoft.GameInput`): {error}"
            ),
        }
    }
}

/// Compares the readings since the last frame with their devices' states, and
/// writes the changes while the cursor is attached.
fn write_device_changes(
    mut device_tracker: ResMut<DeviceTracker>,
    cursor_mode: Res<State<CursorMode>>,
    mut input_writer: InputWriter,
) {
    let cursor_attached = *cursor_mode.get() == CursorMode::Attached;
    for reading in device_tracker.new_readings() {
        let mut device = None;
        unsafe { reading.GetDevice(&mut device) };
        let Some(device) = device else {
            continue;
        };
        device_tracker
            .device_states
            .entry(device.as_raw() as usize)
            .or_insert_with(|| DeviceState::new(device))
            .apply_reading(&reading, cursor_attached.then_some(&mut input_writer));
    }
}

/// Tracks GameInput's keyboards and mice: goes through their readings, and
/// keeps each device's state as of its last reading.
#[derive(Resource)]
struct DeviceTracker {
    game_input: IGameInput,
    /// The timestamp of the newest reading compared, or of when the tracker was
    /// created.
    last_timestamp: u64,
    /// By the device's address.
    device_states: HashMap<usize, DeviceState>,
}

impl DeviceTracker {
    fn new() -> windows_core::Result<Self> {
        let game_input = create_game_input()?;
        Ok(DeviceTracker {
            last_timestamp: unsafe { game_input.GetCurrentTimestamp() },
            game_input,
            device_states: HashMap::new(),
        })
    }

    /// The readings of keyboards and mice newer than the last call's, oldest
    /// first.
    ///
    /// They are found by going back from the current reading rather than on
    /// from the last call's newest: GameInput can't go on from every reading it
    /// returns (at startup, the current reading can be one it answers
    /// `GAMEINPUT_E_REFERENCE_READING_TOO_OLD` for). GameInput keeps only the
    /// recent readings, so if more came since the last call than it keeps, the
    /// oldest are left out. The readings are states, so of those, only a press
    /// that began and ended among them is lost.
    fn new_readings(&mut self) -> impl Iterator<Item = IGameInputReading> + use<> {
        let input_kind = GameInputKindKeyboard | GameInputKindMouse;
        let mut readings = Vec::new();
        // Each call fails, leaving `reading` `None`, when there is no reading
        // to return.
        let mut reading = None;
        let _ = unsafe {
            self.game_input
                .GetCurrentReading(input_kind, std::ptr::null_mut(), &mut reading)
        };
        while let Some(newer) = reading.take() {
            if unsafe { newer.GetTimestamp() } <= self.last_timestamp {
                break;
            }
            let _ = unsafe {
                self.game_input.GetPreviousReading(
                    newer.as_raw(),
                    input_kind,
                    std::ptr::null_mut(),
                    &mut reading,
                )
            };
            readings.push(newer);
        }
        if let Some(newest) = readings.first() {
            self.last_timestamp = unsafe { newest.GetTimestamp() };
        }
        readings.into_iter().rev()
    }
}

/// A keyboard or mouse's state as of its last reading.
struct DeviceState {
    /// Held so that no other device takes its address.
    _device: IGameInputDevice,
    /// The scan codes of the keys held.
    held_scan_codes: BTreeSet<u32>,
    /// `None` until the first reading of the mouse.
    mouse_state: Option<GameInputMouseState>,
}

impl DeviceState {
    fn new(device: IGameInputDevice) -> Self {
        DeviceState {
            _device: device,
            held_scan_codes: BTreeSet::new(),
            mouse_state: None,
        }
    }

    /// Takes the state in `reading` as the device's, writing the changes from
    /// its previous state to `input_writer`, if there is one (while the cursor
    /// is attached).
    fn apply_reading(
        &mut self,
        reading: &IGameInputReading,
        mut input_writer: Option<&mut InputWriter>,
    ) {
        let input_kind = unsafe { reading.GetInputKind() };

        if input_kind & GameInputKindKeyboard != 0 {
            let mut key_states =
                vec![GameInputKeyState::default(); unsafe { reading.GetKeyCount() } as usize];
            let key_count =
                unsafe { reading.GetKeyState(key_states.len() as u32, key_states.as_mut_ptr()) };
            let held_now: BTreeSet<u32> = key_states[..key_count as usize]
                .iter()
                .map(|key_state| key_state.scanCode)
                .collect();
            if let Some(input_writer) = input_writer.as_deref_mut() {
                for &scan_code in self.held_scan_codes.difference(&held_now) {
                    input_writer.key(key_code_from_scan_code(scan_code), false);
                }
                for &scan_code in held_now.difference(&self.held_scan_codes) {
                    input_writer.key(key_code_from_scan_code(scan_code), true);
                }
            }
            self.held_scan_codes = held_now;
        }

        let mut mouse_state = GameInputMouseState::default();
        if input_kind & GameInputKindMouse != 0
            && unsafe { reading.GetMouseState(&mut mouse_state) }
        {
            let previous_mouse_state = self.mouse_state.replace(mouse_state);
            let Some(input_writer) = input_writer else {
                return;
            };
            let previous_buttons =
                previous_mouse_state.map_or(GameInputMouseNone, |previous| previous.buttons);
            for (button_flag, button) in [
                (GameInputMouseLeftButton, MouseButton::Left),
                (GameInputMouseRightButton, MouseButton::Right),
                (GameInputMouseMiddleButton, MouseButton::Middle),
                (GameInputMouseButton4, MouseButton::Back),
                (GameInputMouseButton5, MouseButton::Forward),
            ] {
                if (previous_buttons ^ mouse_state.buttons) & button_flag != 0 {
                    input_writer.button(button, mouse_state.buttons & button_flag != 0);
                }
            }
            // The positions and wheels are totals since the mouse connected, so
            // the first reading has nothing to compare with.
            if let Some(previous) = previous_mouse_state {
                if mouse_state.positions & previous.positions & GameInputMouseRelativePosition != 0
                {
                    let motion = (
                        mouse_state.positionX - previous.positionX,
                        mouse_state.positionY - previous.positionY,
                    );
                    if motion != (0, 0) {
                        input_writer.motion(Vec2::new(motion.0 as f32, motion.1 as f32));
                    }
                }
                let wheel_turn = (
                    mouse_state.wheelX - previous.wheelX,
                    mouse_state.wheelY - previous.wheelY,
                );
                if wheel_turn != (0, 0) {
                    input_writer.scroll(
                        Vec2::new(wheel_turn.0 as f32, wheel_turn.1 as f32) / WHEEL_DELTA,
                        MouseScrollUnit::Line,
                    );
                }
            }
        }
    }
}

/// The key with this scan code: set 1, with `0xE0` in the second byte for the
/// extended keys.
fn key_code_from_scan_code(scan_code: u32) -> KeyCode {
    use KeyCode::*;
    const DIGITS: [KeyCode; 10] = [
        Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0,
    ];
    const TOP_ROW: [KeyCode; 10] = [KeyQ, KeyW, KeyE, KeyR, KeyT, KeyY, KeyU, KeyI, KeyO, KeyP];
    const HOME_ROW: [KeyCode; 9] = [KeyA, KeyS, KeyD, KeyF, KeyG, KeyH, KeyJ, KeyK, KeyL];
    const BOTTOM_ROW: [KeyCode; 7] = [KeyZ, KeyX, KeyC, KeyV, KeyB, KeyN, KeyM];
    const F1_F10: [KeyCode; 10] = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10];
    const F13_F23: [KeyCode; 11] = [F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23];
    let offset_from = |first_scan_code: u32| (scan_code - first_scan_code) as usize;
    match scan_code {
        0x01 => Escape,
        0x02..=0x0B => DIGITS[offset_from(0x02)],
        0x0C => Minus,
        0x0D => Equal,
        0x0E => Backspace,
        0x0F => Tab,
        0x10..=0x19 => TOP_ROW[offset_from(0x10)],
        0x1A => BracketLeft,
        0x1B => BracketRight,
        0x1C => Enter,
        0x1D => ControlLeft,
        0x1E..=0x26 => HOME_ROW[offset_from(0x1E)],
        0x27 => Semicolon,
        0x28 => Quote,
        0x29 => Backquote,
        0x2A => ShiftLeft,
        0x2B => Backslash,
        0x2C..=0x32 => BOTTOM_ROW[offset_from(0x2C)],
        0x33 => Comma,
        0x34 => Period,
        0x35 => Slash,
        0x36 => ShiftRight,
        0x37 => NumpadMultiply,
        0x38 => AltLeft,
        0x39 => Space,
        0x3A => CapsLock,
        0x3B..=0x44 => F1_F10[offset_from(0x3B)],
        0x45 => NumLock,
        0x46 => ScrollLock,
        0x47 => Numpad7,
        0x48 => Numpad8,
        0x49 => Numpad9,
        0x4A => NumpadSubtract,
        0x4B => Numpad4,
        0x4C => Numpad5,
        0x4D => Numpad6,
        0x4E => NumpadAdd,
        0x4F => Numpad1,
        0x50 => Numpad2,
        0x51 => Numpad3,
        0x52 => Numpad0,
        0x53 => NumpadDecimal,
        // Alt + Print Screen.
        0x54 => PrintScreen,
        0x56 => IntlBackslash,
        0x57 => F11,
        0x58 => F12,
        0x59 => NumpadEqual,
        0x64..=0x6E => F13_F23[offset_from(0x64)],
        0x70 => KanaMode,
        0x71 => Lang2,
        0x72 => Lang1,
        0x73 => IntlRo,
        0x76 => F24,
        0x79 => Convert,
        0x7B => NonConvert,
        0x7D => IntlYen,
        0x7E => NumpadComma,
        0xE01C => NumpadEnter,
        0xE01D => ControlRight,
        0xE035 => NumpadDivide,
        0xE037 => PrintScreen,
        0xE038 => AltRight,
        // Ctrl + Pause.
        0xE046 => Pause,
        0xE047 => Home,
        0xE048 => ArrowUp,
        0xE049 => PageUp,
        0xE04B => ArrowLeft,
        0xE04D => ArrowRight,
        0xE04F => End,
        0xE050 => ArrowDown,
        0xE051 => PageDown,
        0xE052 => Insert,
        0xE053 => Delete,
        0xE05B => SuperLeft,
        0xE05C => SuperRight,
        0xE05D => ContextMenu,
        0xE05E => Power,
        _ => Unidentified(NativeKeyCode::Windows(scan_code as u16)),
    }
}
