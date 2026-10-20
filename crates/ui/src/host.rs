//! The UI thread on macOS: the GPUI app, the game's layer, and which input is the
//! game's.

use std::sync::Mutex;

use bevy::app::AppExit;
use bevy_pumicite::MetalLayer;

use futures_util::StreamExt;
use gpui::{
    AnyView, App, AppContext, Bounds, Context, FocusHandle, InteractiveElement, IntoElement,
    MouseDownEvent, ParentElement, Render, StatefulInteractiveElement, Styled, TitlebarOptions,
    Window, WindowBackgroundAppearance, WindowBounds, WindowOptions, div, px, size,
};
use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSCursor, NSView, NSWindowOrderingMode};
use objc2_core_foundation::CGSize;
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use dust_input::InputGate;

use crate::bridge::{BevyHandle, UiHandle};
use crate::plugin::{self, UiHostPlugin, WindowSize};

unsafe extern "C" {
    // CoreGraphics. Detaches the cursor from the mouse, so it stays put while the
    // mouse still reports motion.
    fn CGAssociateMouseAndMouseCursorPosition(connected: u32) -> i32;
}

pub(crate) fn run(
    build_app: impl FnOnce(UiHostPlugin) -> bevy::app::App + Send + 'static,
    build_overlay: impl FnOnce(&mut Window, &mut App) -> AnyView + 'static,
) {
    let (to_bevy, from_ui) = std::sync::mpsc::channel();
    let (to_ui, mut from_bevy) = futures_channel::mpsc::unbounded();
    let ui = UiHandle(to_ui);
    let plugin = UiHostPlugin {
        from_ui: Mutex::new(Some(from_ui)),
        ui: ui.clone(),
    };
    std::thread::Builder::new()
        .name("bevy".into())
        .spawn(move || {
            let mut app = build_app(plugin);
            // `run` moves the app into the runner, which drops it (and with it the
            // surface on the UI's layer) before returning.
            app.run();
            ui.update(|cx| cx.quit());
        })
        .expect("failed to spawn the Bevy thread");

    gpui_platform::application()
        .with_assets(gpui_kit_assets::Assets)
        .run(move |cx: &mut App| {
            gpui_component::init(cx);
            cx.set_global(BevyHandle(to_bevy));
            cx.set_global(Cursor::default());
            let window_options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(1280.), px(720.)),
                    cx,
                ))),
                titlebar: Some(TitlebarOptions {
                    title: Some("Dust".into()),
                    ..Default::default()
                }),
                // The game shows through wherever the UI draws nothing.
                window_background: WindowBackgroundAppearance::Transparent,
                ..Default::default()
            };
            cx.open_window(window_options, move |window, cx| {
                let overlay = build_overlay(window, cx);
                let game = cx.new(|cx| GameView::new(overlay, window, cx));
                cx.new(|cx| {
                    gpui_component::Root::new(game, window, cx).bg(gpui::transparent_black())
                })
            })
            .expect("failed to open the window");
            // Started once the overlay exists, so the calls Bevy queued for it
            // find what it set up.
            cx.spawn(async move |cx| {
                while let Some(call) = from_bevy.next().await {
                    cx.update(call);
                }
            })
            .detach();
            cx.activate(true);
        });
}

/// Whether the cursor is locked and hidden; a global of the GPUI app.
#[derive(Default)]
struct Cursor {
    locked: bool,
    hidden: bool,
}

impl gpui::Global for Cursor {}

/// Locks or shows the cursor, mirroring the game's `CursorOptions`.
pub(crate) fn set_cursor(locked: bool, visible: bool, cx: &mut App) {
    let cursor = cx.global_mut::<Cursor>();
    if std::mem::replace(&mut cursor.locked, locked) != locked {
        unsafe { CGAssociateMouseAndMouseCursorPosition(!locked as u32) };
    }
    // `hide` and `unhide` nest, so they are kept balanced.
    if std::mem::replace(&mut cursor.hidden, !visible) != !visible {
        if visible {
            NSCursor::unhide();
        } else {
            NSCursor::hide();
        }
    }
}

/// The window's root view: the game's layer under the overlay.
///
/// The game reads the keyboard and mouse itself (`dust_input`); this view tells
/// it which input is its own. Keys are, while this view holds focus
/// (rather than, say, a text field in the overlay); clicks and scrolls are where
/// the pointer is over no occluding overlay element. Clicking the game focuses
/// it.
struct GameView {
    overlay: AnyView,
    focus: FocusHandle,
    bevy: BevyHandle,
    game_layer: GameLayer,
    size: Option<WindowSize>,
    /// The last gate sent to the game.
    gate: InputGate,
    /// Whether the game's element is hovered: the pointer is over no occluding
    /// overlay element.
    hovered: bool,
}

/// The game's view and layer, under GPUI's view in the window's content view.
struct GameLayer {
    _view: Retained<NSView>,
    layer: Retained<CAMetalLayer>,
}

impl GameView {
    fn new(overlay: AnyView, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        cx.on_focus(&focus, window, |_, _, cx| cx.notify()).detach();
        cx.on_blur(&focus, window, |_, _, cx| cx.notify()).detach();
        cx.observe_window_activation(window, |_, window, cx| {
            if !window.is_window_active() {
                set_cursor(false, true, cx);
            }
            cx.notify();
        })
        .detach();
        let bevy = cx.global::<BevyHandle>().clone();
        window.on_window_should_close(cx, {
            let bevy = bevy.clone();
            // The window closes when the app quits, after the Bevy app has exited.
            move |_, _| {
                bevy.update(|world| {
                    world.write_message(AppExit::Success);
                });
                false
            }
        });
        GameView {
            overlay,
            focus,
            bevy,
            game_layer: GameLayer::new(window),
            size: None,
            gate: InputGate::default(),
            hovered: false,
        }
    }

    /// Sends the window's size when it changes; the first time, with the layer.
    fn sync_size(&mut self, window: &Window) {
        let scale_factor = window.scale_factor();
        let viewport = window.viewport_size();
        let size = WindowSize {
            physical_width: (viewport.width.as_f32() * scale_factor).round() as u32,
            physical_height: (viewport.height.as_f32() * scale_factor).round() as u32,
            scale_factor,
        };
        if self.size == Some(size) {
            return;
        }
        let layer = &self.game_layer.layer;
        layer.setContentsScale(scale_factor as f64);
        layer.setDrawableSize(CGSize::new(
            size.physical_width as f64,
            size.physical_height as f64,
        ));
        match self.size {
            None => {
                let layer = MetalLayer(layer.clone());
                self.bevy
                    .update(move |world| plugin::window_created(world, layer, size));
            }
            Some(_) => self
                .bevy
                .update(move |world| plugin::window_resized(world, size)),
        }
        self.size = Some(size);
    }

    /// Sends the [`InputGate`] when it changes. Focus, window activation and
    /// hovering all notify this view, so checking on render catches every change.
    fn sync_gate(&mut self, window: &Window) {
        let gate = InputGate {
            active: window.is_window_active(),
            keys: self.focus.is_focused(window),
            pointer: self.hovered,
        };
        if std::mem::replace(&mut self.gate, gate) != gate {
            self.bevy
                .update(move |world| plugin::set_input_gate(world, gate));
        }
    }
}

impl Render for GameView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_size(window);
        self.sync_gate(window);
        div()
            .id("game")
            .size_full()
            .track_focus(&self.focus)
            .on_any_mouse_down(cx.listener(|this, _: &MouseDownEvent, window, cx| {
                window.focus(&this.focus, cx);
            }))
            // Hovered only where no occluding overlay element is on top.
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                this.hovered = *hovered;
                cx.notify();
            }))
            .child(self.overlay.clone())
    }
}

impl GameLayer {
    /// Adds a layer-hosting view for the game under GPUI's view, filling the
    /// window's content view.
    fn new(window: &Window) -> Self {
        let mtm = MainThreadMarker::new().expect("the UI runs on the main thread");
        let RawWindowHandle::AppKit(handle) =
            HasWindowHandle::window_handle(window).unwrap().as_raw()
        else {
            unreachable!("GPUI windows on macOS have AppKit handles");
        };
        // SAFETY: GPUI's handle is its `NSView`, alive as long as the window.
        let gpui_view: &NSView = unsafe { handle.ns_view.cast().as_ref() };
        let content_view =
            unsafe { gpui_view.superview() }.expect("GPUI's view is in the content view");

        let layer = CAMetalLayer::new();
        layer.setOpaque(true);
        let view = NSView::initWithFrame(NSView::alloc(mtm), content_view.bounds());
        // Setting the layer before `wantsLayer` makes the view layer-hosting: the
        // layer is the view's, and AppKit leaves its contents alone.
        view.setLayer(Some(&layer));
        view.setWantsLayer(true);
        view.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        content_view.addSubview_positioned_relativeTo(
            &view,
            NSWindowOrderingMode::Below,
            Some(gpui_view),
        );
        GameLayer { _view: view, layer }
    }
}
