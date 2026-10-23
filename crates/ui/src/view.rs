//! [`BevyView`]: the GPUI view a Bevy app runs and draws in.

use bevy::app::AppExit;
use bevy::prelude::NextState;
use futures_util::StreamExt;
use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, InteractiveElement, IntoElement,
    KeyDownEvent, MouseButton, MouseDownEvent, ParentElement, Render, Styled, Task, Window, canvas,
    div,
};

use dust_input::CursorMode;

use crate::apple::NSViewWithMetalLayer;
use crate::bridge::{BevyChannel, GPUIChannel};
use crate::build_gpui_app;

/// Shows a Bevy app, which runs on a thread of its own and draws into a layer
/// under GPUI's (see the crate docs).
///
/// [`BevyView::new`] sets the [`BevyChannel`] global, so a GPUI app has one
/// `BevyView`. The calls the Bevy app queues through the [`GPUIChannel`] run in
/// a GPUI task, so not before the code that created the view returns: views
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
    bevy: BevyChannel,
    game_layer: NSViewWithMetalLayer,
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
    /// the [`GPUIChannel`]. `build` then adds the rest of the app, without running
    /// it: the view sets the app's runner, which adds the
    /// [`Swapchain`](crate::Swapchain), and runs it.
    pub fn new(
        build: impl FnOnce(&mut bevy::app::App) + Send + 'static,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (to_bevy, from_ui) = std::sync::mpsc::channel();
        let (to_ui, mut from_bevy) = futures_channel::mpsc::unbounded();
        let ui = GPUIChannel(to_ui);
        let bevy = BevyChannel(to_bevy);
        cx.set_global(bevy.clone());

        cx.observe_window_activation(window, |this, window, _| {
            if !window.is_window_active() {
                this.set_cursor_mode(CursorMode::Detached);
            }
        })
        .detach();

        let mut game_layer = NSViewWithMetalLayer::new(window);
        let runner = game_layer.create_bevy_runner();

        let view = cx.weak_entity();
        std::thread::Builder::new()
            .name("bevy".into())
            .spawn(move || {
                let mut app = bevy::app::App::new();
                build_gpui_app(&mut app, ui.clone(), from_ui);
                build(&mut app);
                app.set_runner(runner);
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
    pub fn set_cursor_mode(&mut self, mode: CursorMode) {
        self.game_layer
            .set_cursor_locked(mode == CursorMode::Attached);
        self.bevy.update(move |world| {
            world
                .resource_mut::<NextState<CursorMode>>()
                .set_if_different(mode);
        });
    }
}

impl Focusable for BevyView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for BevyView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let this = cx.entity();
        div()
            .size_full()
            // Fits the game's layer to the view's bounds, known once it is laid
            // out, at prepaint.
            .child(
                canvas(
                    move |bounds, window, cx| {
                        this.read(cx)
                            .game_layer
                            .set_bounds(bounds, window.scale_factor());
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .size_full(),
            )
            .track_focus(&self.focus)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus, cx);
                    this.set_cursor_mode(CursorMode::Attached);
                }),
            )
            // A listener rather than a key binding: GPUI hides the cursor until
            // the mouse moves after a key binding's action runs.
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, _| {
                if event.keystroke.key == "escape" {
                    this.set_cursor_mode(CursorMode::Detached);
                }
            }))
    }
}
