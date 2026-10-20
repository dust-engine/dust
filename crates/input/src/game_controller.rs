//! The keyboard and mouse on macOS, read with GameController (`GCKeyboard`,
//! `GCMouse`).
//!
//! GameController calls our handlers on a dispatch queue of our own, so input
//! reaches the Bevy thread without waiting on the UI thread, which may be busy
//! drawing. The handlers only push [`GcEvent`]s into a channel, which
//! [`GameControllerPlugin`] drains once per frame into [`GatedInput`].

use std::ptr::NonNull;
use std::sync::mpsc::{Receiver, Sender};

use bevy::input::keyboard::NativeKeyCode;
use bevy::input::mouse::MouseScrollUnit;
use bevy::prelude::*;
use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::{Bool, NSObjectProtocol, ProtocolObject};
use objc2_foundation::{NSNotification, NSNotificationCenter};
use objc2_game_controller::{
    GCControllerButtonInput, GCControllerDirectionPad, GCDevice, GCKeyCode, GCKeyboard,
    GCKeyboardDidConnectNotification, GCKeyboardInput, GCMouse, GCMouseDidConnectNotification,
    GCMouseInput,
};

use crate::{GatedInput, InputSet, release_inactive};

/// Reads GameController's keyboards and mice.
pub struct GameControllerPlugin;

impl Plugin for GameControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            read_events.in_set(InputSet).after(release_inactive),
        );
    }
}

/// `watch` starts reporting when the system is initialized.
fn read_events(watch: Local<InputSubscription>, mut input: GatedInput) {
    for event in watch.events.try_iter() {
        match event {
            GcEvent::Key { usage, pressed } => {
                input.key(key_code(usage), pressed);
            }
            GcEvent::Button { button, pressed } => input.button(button, pressed),
            GcEvent::Motion(delta) => input.motion(delta),
            GcEvent::Scroll(delta) => input.scroll(delta, MouseScrollUnit::Line),
        }
    }
}

/// One change of the keyboard or mouse, as GameController reports it.
enum GcEvent {
    /// `usage` is the `GCKeyCode`: the key's USB HID usage.
    Key {
        usage: GCKeyCode,
        pressed: bool,
    },
    Button {
        button: MouseButton,
        pressed: bool,
    },
    /// Raw mouse motion, y down.
    Motion(Vec2),
    /// Scrolling, in lines, y up.
    Scroll(Vec2),
}

struct InputSubscription {
    events: Receiver<GcEvent>,

    /// Observes GCKeyboardDidConnectNotification
    keyboard_observer: Retained<ProtocolObject<dyn NSObjectProtocol>>,

    /// Observes GCMouseDidConnectNotification
    mouse_observer: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

// SAFETY: The observers are only used to remove them, in `drop`, and
// `NSNotificationCenter` may be called from any thread.
unsafe impl Send for InputSubscription {}

impl Default for InputSubscription {
    /// Starts reporting from the keyboards and mice connected now and later.
    fn default() -> Self {
        let (send, events) = std::sync::mpsc::channel();
        let queue = DispatchQueue::new("dust.game-input", DispatchQueueAttr::SERIAL);

        // The keyboards appear as one: `coalescedKeyboard`. Each mouse is its
        // own device. Devices connected at launch are announced too, but the
        // current ones are set up now in case they were announced already.
        let keyboard = {
            let (send, queue) = (send.clone(), queue.clone());
            move || {
                if let Some(keyboard) = unsafe { GCKeyboard::coalescedKeyboard() } {
                    watch_keyboard(&keyboard, &queue, send.clone());
                }
            }
        };
        let mouse = {
            let (send, queue) = (send.clone(), queue.clone());
            move |mouse: &GCMouse| watch_mouse(mouse, &queue, send.clone())
        };
        keyboard();
        for mouse_device in unsafe { GCMouse::mice() }.iter() {
            mouse(&mouse_device);
        }

        let center = NSNotificationCenter::defaultCenter();
        let on_keyboard = RcBlock::new(move |_: NonNull<NSNotification>| keyboard());
        let on_mouse = RcBlock::new(move |notification: NonNull<NSNotification>| {
            // SAFETY: The notification is valid for the call, and the object of
            // `GCMouseDidConnectNotification` is the `GCMouse`.
            let object = unsafe { notification.as_ref() }.object();
            if let Some(device) = object.and_then(|object| object.downcast::<GCMouse>().ok()) {
                mouse(&device);
            }
        });
        // With no queue, the blocks run on the thread that posts, so setting up
        // a device never waits on another queue.
        unsafe {
            InputSubscription {
                events,
                keyboard_observer: center.addObserverForName_object_queue_usingBlock(
                    Some(GCKeyboardDidConnectNotification),
                    None,
                    None,
                    &on_keyboard,
                ),
                mouse_observer: center.addObserverForName_object_queue_usingBlock(
                    Some(GCMouseDidConnectNotification),
                    None,
                    None,
                    &on_mouse,
                ),
            }
        }
    }
}

impl Drop for InputSubscription {
    fn drop(&mut self) {
        let center = NSNotificationCenter::defaultCenter();
        unsafe {
            center.removeObserver(self.keyboard_observer.as_ref());
            center.removeObserver(self.mouse_observer.as_ref());
        }
    }
}

fn watch_keyboard(
    keyboard: &GCKeyboard,
    queue: &DispatchRetained<DispatchQueue>,
    send: Sender<GcEvent>,
) {
    let Some(input) = (unsafe { keyboard.keyboardInput() }) else {
        return;
    };
    let handler = RcBlock::new(
        move |_: NonNull<GCKeyboardInput>,
              _: NonNull<GCControllerButtonInput>,
              usage: GCKeyCode,
              pressed: Bool| {
            send.send(GcEvent::Key {
                usage,
                pressed: pressed.as_bool(),
            })
            .ok();
        },
    );
    unsafe {
        keyboard.setHandlerQueue(queue);
        // The setter copies the block.
        input.setKeyChangedHandler(RcBlock::as_ptr(&handler));
    }
}

fn watch_mouse(mouse: &GCMouse, queue: &DispatchRetained<DispatchQueue>, send: Sender<GcEvent>) {
    let Some(input) = (unsafe { mouse.mouseInput() }) else {
        return;
    };
    unsafe { mouse.setHandlerQueue(queue) };

    let moved = RcBlock::new({
        let send = send.clone();
        // GameController's y points up.
        move |_: NonNull<GCMouseInput>, x: f32, y: f32| {
            send.send(GcEvent::Motion(Vec2::new(x, -y))).ok();
        }
    });
    unsafe { input.setMouseMovedHandler(RcBlock::as_ptr(&moved)) };

    let scrolled = RcBlock::new({
        let send = send.clone();
        move |_: NonNull<GCControllerDirectionPad>, x: f32, y: f32| {
            send.send(GcEvent::Scroll(Vec2::new(x, y))).ok();
        }
    });
    unsafe {
        input
            .scroll()
            .setValueChangedHandler(RcBlock::as_ptr(&scrolled))
    };

    let mut buttons = vec![(unsafe { input.leftButton() }, MouseButton::Left)];
    buttons.extend(unsafe { input.rightButton() }.map(|b| (b, MouseButton::Right)));
    buttons.extend(unsafe { input.middleButton() }.map(|b| (b, MouseButton::Middle)));
    if let Some(auxiliary) = unsafe { input.auxiliaryButtons() } {
        // The first two auxiliary buttons are taken to be back and forward.
        for (i, b) in auxiliary.iter().enumerate() {
            let button = match i {
                0 => MouseButton::Back,
                1 => MouseButton::Forward,
                _ => MouseButton::Other(i as u16 + 3),
            };
            buttons.push((b, button));
        }
    }
    for (input, button) in buttons {
        let send = send.clone();
        let handler = RcBlock::new(
            move |_: NonNull<GCControllerButtonInput>, _: f32, pressed: Bool| {
                send.send(GcEvent::Button {
                    button,
                    pressed: pressed.as_bool(),
                })
                .ok();
            },
        );
        unsafe { input.setPressedChangedHandler(RcBlock::as_ptr(&handler)) };
    }
}

fn key_code(usage: isize) -> KeyCode {
    use KeyCode::*;
    const LETTERS: [KeyCode; 26] = [
        KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO,
        KeyP, KeyQ, KeyR, KeyS, KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ,
    ];
    const DIGITS: [KeyCode; 10] = [
        Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9, Digit0,
    ];
    const F1_F12: [KeyCode; 12] = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12];
    const NUMPAD_DIGITS: [KeyCode; 10] = [
        Numpad1, Numpad2, Numpad3, Numpad4, Numpad5, Numpad6, Numpad7, Numpad8, Numpad9, Numpad0,
    ];
    const F13_F24: [KeyCode; 12] = [F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24];
    let index = |first: isize| (usage - first) as usize;
    match usage {
        0x04..=0x1D => LETTERS[index(0x04)],
        0x1E..=0x27 => DIGITS[index(0x1E)],
        0x28 => Enter,
        0x29 => Escape,
        0x2A => Backspace,
        0x2B => Tab,
        0x2C => Space,
        0x2D => Minus,
        0x2E => Equal,
        0x2F => BracketLeft,
        0x30 => BracketRight,
        // 0x32 is the ISO key left of Enter, which the W3C codes also call
        // `Backslash`.
        0x31 | 0x32 => Backslash,
        0x33 => Semicolon,
        0x34 => Quote,
        0x35 => Backquote,
        0x36 => Comma,
        0x37 => Period,
        0x38 => Slash,
        0x39 => CapsLock,
        0x3A..=0x45 => F1_F12[index(0x3A)],
        0x46 => PrintScreen,
        0x47 => ScrollLock,
        0x48 => Pause,
        0x49 => Insert,
        0x4A => Home,
        0x4B => PageUp,
        0x4C => Delete,
        0x4D => End,
        0x4E => PageDown,
        0x4F => ArrowRight,
        0x50 => ArrowLeft,
        0x51 => ArrowDown,
        0x52 => ArrowUp,
        0x53 => NumLock,
        0x54 => NumpadDivide,
        0x55 => NumpadMultiply,
        0x56 => NumpadSubtract,
        0x57 => NumpadAdd,
        0x58 => NumpadEnter,
        0x59..=0x62 => NUMPAD_DIGITS[index(0x59)],
        0x63 => NumpadDecimal,
        0x64 => IntlBackslash,
        0x65 => ContextMenu,
        0x66 => Power,
        0x67 => NumpadEqual,
        0x68..=0x73 => F13_F24[index(0x68)],
        0x87 => IntlRo,
        0x88 => KanaMode,
        0x89 => IntlYen,
        0x8A => Convert,
        0x8B => NonConvert,
        0x90 => Lang1,
        0x91 => Lang2,
        0xE0 => ControlLeft,
        0xE1 => ShiftLeft,
        0xE2 => AltLeft,
        0xE3 => SuperLeft,
        0xE4 => ControlRight,
        0xE5 => ShiftRight,
        0xE6 => AltRight,
        0xE7 => SuperRight,
        _ => Unidentified(NativeKeyCode::Unidentified),
    }
}
