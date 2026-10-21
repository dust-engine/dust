//! [`BevyView`]: the GPUI view a Bevy app runs and draws in.

use bevy::app::AppExit;
use bevy::prelude::NextState;
use futures_util::StreamExt;
use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, InteractiveElement, IntoElement,
    KeyBinding, MouseButton, MouseDownEvent, Render, Styled, Task, Window, actions, div,
};

use dust_input::CursorMode;

use crate::apple::{self, GameLayer};
use crate::bridge::{BevyHandle, UiHandle};
use crate::plugin::UiHostPlugin;

actions!(dust_ui, [DetachCursor]);

/// Shows a Bevy app, which runs on a thread of its own and draws into a layer
/// under GPUI's (see the crate docs).
///
/// [`BevyView::new`] sets the [`BevyHandle`] global, so a GPUI app has one
/// `BevyView`. The calls the Bevy app queues through the [`UiHandle`] run in a
/// GPUI task, so not before the code that created the view returns: views
/// created next to it, such as an overlay, exist by then.
///
/// The game shows only where nothing is drawn over it, so the views the
/// `BevyView` is in paint no background.
///
/// The game gets the keyboard and mouse (`dust_input`) while the cursor is
/// attached; this view sets the [`CursorMode`] that says whether it is. A left
/// click on the view focuses it and attaches the cursor; elements drawn over the
/// view that should take the click instead call `.occlude()`. Escape, while the
/// view holds focus, detaches the cursor, as does the window stopping being the
/// active one.
pub struct BevyView {
    focus: FocusHandle,
    bevy: BevyHandle,
    game_layer: GameLayer,
    size: WindowSize,
    /// Runs the calls the Bevy app queues for the UI thread.
    _ui_calls: Task<()>,
}

/// Emitted by a [`BevyView`] once its Bevy app has exited and been dropped.
pub struct BevyExited(pub AppExit);

impl EventEmitter<BevyExited> for BevyView {}

impl BevyView {
    /// Starts a Bevy app on a thread of its own, drawing into this view.
    ///
    /// The app is created on that thread, with `dust_ui`'s plugin, which inserts
    /// the [`UiHandle`] and the [`Swapchain`](crate::Swapchain). `build` then adds
    /// the rest of the app, without running it: the view sets the app's runner
    /// and runs it.
    pub fn new(
        build: impl FnOnce(&mut bevy::app::App) + Send + 'static,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (to_bevy, from_ui) = std::sync::mpsc::channel();
        let (to_ui, mut from_bevy) = futures_channel::mpsc::unbounded();
        let ui = UiHandle(to_ui);
        let bevy = BevyHandle(to_bevy);
        cx.set_global(bevy.clone());

        cx.bind_keys([KeyBinding::new("escape", DetachCursor, Some("BevyView"))]);
        cx.observe_window_activation(window, |this, window, cx| {
            if !window.is_window_active() {
                this.set_cursor_mode(CursorMode::Detached, cx);
            }
        })
        .detach();

        let game_layer = GameLayer::new(window);
        let size = window_size(window);
        game_layer.set_size(size);

        // The Bevy app starts once the game's layer exists, and draws into it
        // from its first frame.
        let layer = game_layer.layer();
        let view = cx.weak_entity();
        std::thread::Builder::new()
            .name("bevy".into())
            .spawn(move || {
                let mut app = bevy::app::App::new();
                app.add_plugins(UiHostPlugin { ui: ui.clone() });
                build(&mut app);
                app.set_runner(move |app| apple::displaylink_runner(app, from_ui, layer));
                // `run` moves the app into the runner, which drops it (and with
                // it its images of the layer's drawables) before returning.
                let exit = app.run();
                ui.update(move |cx| {
                    view.update(cx, |_, cx| cx.emit(BevyExited(exit))).ok();
                });
            })
            .expect("failed to spawn the Bevy thread");

        let ui_calls = cx.spawn(async move |_, cx| {
            while let Some(call) = from_bevy.next().await {
                cx.update(call);
            }
        });

        BevyView {
            focus: cx.focus_handle(),
            bevy,
            game_layer,
            size,
            _ui_calls: ui_calls,
        }
    }

    /// Asks the Bevy app to exit. The view emits [`BevyExited`] once it has.
    pub fn exit(&self) {
        self.bevy.update(|world| {
            world.write_message(AppExit::Success);
        });
    }

    /// Locks and hides the cursor for the game, or frees it for the UI, and sets
    /// the game's [`CursorMode`] to match.
    fn set_cursor_mode(&self, mode: CursorMode, cx: &mut App) {
        apple::set_cursor(mode == CursorMode::Attached, cx);
        self.bevy.update(move |world| {
            world
                .resource_mut::<NextState<CursorMode>>()
                .set_if_different(mode);
        });
    }

    /// Resizes the layer when the window's drawable area changes.
    fn sync_size(&mut self, window: &Window) {
        let size = window_size(window);
        if std::mem::replace(&mut self.size, size) != size {
            self.game_layer.set_size(size);
        }
    }
}

impl Focusable for BevyView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for BevyView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_size(window);
        div()
            .key_context("BevyView")
            .size_full()
            .track_focus(&self.focus)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus, cx);
                    this.set_cursor_mode(CursorMode::Attached, cx);
                }),
            )
            .on_action(cx.listener(|this, _: &DetachCursor, _, cx| {
                this.set_cursor_mode(CursorMode::Detached, cx);
            }))
    }
}

/// The drawable area of the window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WindowSize {
    pub physical_width: u32,
    pub physical_height: u32,
    pub scale_factor: f32,
}

/// The window's drawable area.
fn window_size(window: &Window) -> WindowSize {
    let scale_factor = window.scale_factor();
    let viewport = window.viewport_size();
    WindowSize {
        physical_width: (viewport.width.as_f32() * scale_factor).round() as u32,
        physical_height: (viewport.height.as_f32() * scale_factor).round() as u32,
        scale_factor,
    }
}
