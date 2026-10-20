//! The Bevy side of the UI host: the runner, the calls from the UI thread, and
//! what those calls do to the window and the [`InputGate`].

use std::sync::Mutex;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use bevy::app::{AppExit, PluginsState};
use bevy::platform::cell::SyncCell;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow, WindowCreated, WindowResized};
use bevy_pumicite::MetalLayer;
use dust_input::InputGate;

use crate::bridge::{self, BevyCall, UiHandle};

/// Connects the Bevy app to the UI thread, and sets the app's runner.
/// [`crate::run`] creates it.
///
/// It inserts the [`UiHandle`] resource. Add it before `LogPlugin` when a log
/// layer sends messages to the UI through the handle.
pub struct UiHostPlugin {
    // `Plugin` must be `Sync`; the receiver is moved out once, in `build`.
    pub(crate) from_ui: Mutex<Option<Receiver<BevyCall>>>,
    pub(crate) ui: UiHandle,
}

/// Where the calls from the UI thread run: in [`First`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct UiSet;

impl Plugin for UiHostPlugin {
    fn build(&self, app: &mut App) {
        let from_ui = self
            .from_ui
            .lock()
            .unwrap()
            .take()
            .expect("UiHostPlugin added twice");
        let mut from_ui = SyncCell::new(from_ui);
        let run_calls = move |world: &mut World| loop {
            match from_ui.get().try_recv() {
                Ok(call) => call(world),
                Err(TryRecvError::Empty) => break,
                // The UI has quit.
                Err(TryRecvError::Disconnected) => {
                    world.write_message(AppExit::Success);
                    break;
                }
            }
        };
        app.insert_resource(self.ui.clone())
            .add_systems(First, run_calls.in_set(UiSet))
            .add_systems(
                Last,
                bridge::run_every_frame.after(bevy::ecs::system::despawn_unused_registered_systems),
            )
            .set_runner(runner);
        #[cfg(target_os = "macos")]
        app.add_systems(Last, send_cursor_options);
    }
}

fn runner(mut app: App) -> AppExit {
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }
    loop {
        app.update();
        if let Some(exit) = app.should_exit() {
            return exit;
        }
        // With a window, presenting paces the loop. Without one nothing does.
        let window_open = app
            .world_mut()
            .query_filtered::<(), With<MetalLayer>>()
            .iter(app.world())
            .next()
            .is_some();
        if !window_open {
            std::thread::sleep(Duration::from_millis(4));
        }
    }
}

/// The drawable area of the window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WindowSize {
    pub physical_width: u32,
    pub physical_height: u32,
    pub scale_factor: f32,
}

fn primary_window(world: &mut World) -> Option<Entity> {
    world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .single(world)
        .ok()
}

/// The window is open; `layer` is the `CAMetalLayer` the game presents to.
pub(crate) fn window_created(world: &mut World, layer: MetalLayer, size: WindowSize) {
    let Some(window) = primary_window(world) else {
        return;
    };
    set_size(&mut world.get_mut::<Window>(window).unwrap(), size);
    world.entity_mut(window).insert(layer);
    world.write_message(WindowCreated { window });
}

/// The window's drawable area changed size or scale factor.
pub(crate) fn window_resized(world: &mut World, size: WindowSize) {
    let Some(entity) = primary_window(world) else {
        return;
    };
    let mut window = world.get_mut::<Window>(entity).unwrap();
    set_size(&mut window, size);
    let (width, height) = (window.width(), window.height());
    world.write_message(WindowResized {
        window: entity,
        width,
        height,
    });
}

/// Sets which input is the game's. When the window stops being active, the
/// game also unlocks the cursor.
pub(crate) fn set_input_gate(world: &mut World, gate: InputGate) {
    let mut current = world.resource_mut::<InputGate>();
    let deactivated = current.active && !gate.active;
    *current = gate;
    if deactivated
        && let Some(window) = primary_window(world)
        && let Some(mut cursor) = world.get_mut::<CursorOptions>(window)
    {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

fn set_size(window: &mut Window, size: WindowSize) {
    window.resolution.set_scale_factor(size.scale_factor);
    window
        .resolution
        .set_physical_resolution(size.physical_width, size.physical_height);
}

/// Has the UI thread mirror the primary window's `CursorOptions`.
#[cfg(target_os = "macos")]
fn send_cursor_options(
    cursor: Query<&CursorOptions, (With<PrimaryWindow>, Changed<CursorOptions>)>,
    ui: Res<UiHandle>,
) {
    if let Ok(cursor) = cursor.single() {
        let locked = cursor.grab_mode != CursorGrabMode::None;
        let visible = cursor.visible;
        ui.update(move |cx| crate::host::set_cursor(locked, visible, cx));
    }
}
