//! The game's layer, under GPUI's, and the cursor lock.

use gpui::{App, Window, WindowBackgroundAppearance};
use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSCursor, NSView, NSWindowOrderingMode};
use objc2_core_foundation::CGSize;
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::display_link::Layer;
use super::drawable;
use crate::view::WindowSize;

unsafe extern "C" {
    // CoreGraphics. Detaches the cursor from the mouse, so it stays put while the
    // mouse still reports motion.
    fn CGAssociateMouseAndMouseCursorPosition(connected: u32) -> i32;
}

/// Whether the cursor is locked and hidden; a global of the GPUI app.
#[derive(Default)]
struct Cursor {
    locked: bool,
}

impl gpui::Global for Cursor {}

/// Locks and hides the cursor, or frees and shows it.
pub(crate) fn set_cursor(locked: bool, cx: &mut App) {
    let cursor = cx.default_global::<Cursor>();
    if std::mem::replace(&mut cursor.locked, locked) == locked {
        return;
    }
    unsafe { CGAssociateMouseAndMouseCursorPosition(!locked as u32) };
    // `hide` and `unhide` nest, so each is called once per change.
    if locked {
        NSCursor::hide();
    } else {
        NSCursor::unhide();
    }
}

/// The game's view and layer, under GPUI's view in the window's content view.
pub(crate) struct GameLayer {
    _view: Retained<NSView>,
    layer: Retained<CAMetalLayer>,
}

impl GameLayer {
    /// Adds a layer-hosting view for the game under GPUI's view, filling the
    /// window's content view, and makes the window transparent so the game
    /// shows wherever the UI draws nothing.
    pub(crate) fn new(window: &Window) -> Self {
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
        drawable::configure_layer(&layer);
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
        window.set_background_appearance(WindowBackgroundAppearance::Transparent);
        GameLayer { _view: view, layer }
    }

    /// The layer, for the Bevy thread's runner.
    pub(crate) fn layer(&self) -> Layer {
        Layer(self.layer.clone())
    }

    /// Sizes the layer's drawables to the window's drawable area.
    pub(crate) fn set_size(&self, size: WindowSize) {
        self.layer.setContentsScale(size.scale_factor as f64);
        self.layer.setDrawableSize(CGSize::new(
            size.physical_width as f64,
            size.physical_height as f64,
        ));
    }
}
