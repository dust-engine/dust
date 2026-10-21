//! The game's layer, under GPUI's, the cursor lock, and the runner that updates
//! the Bevy app from a `CAMetalDisplayLink` on the layer.

use std::sync::mpsc::Receiver;

use gpui::{Window, WindowBackgroundAppearance};
use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSAutoresizingMaskOptions, NSCursor, NSView, NSWindowOrderingMode};
use objc2_core_foundation::CGSize;
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::cell::{Cell, RefCell};
use std::sync::mpsc::TryRecvError;

use bevy::app::{AppExit, PluginsState};
use objc2::runtime::ProtocolObject;
use objc2::{AllocAnyThread, DefinedClass, define_class, msg_send};
use objc2_core_foundation::CFRunLoop;
use objc2_foundation::{NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSRunLoop};
use objc2_quartz_core::{CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate};

use super::drawable;
use crate::view::WindowSize;

unsafe extern "C" {
    // CoreGraphics. Detaches the cursor from the mouse, so it stays put while the
    // mouse still reports motion.
    fn CGAssociateMouseAndMouseCursorPosition(connected: u32) -> i32;
}
/// The game's view and layer, under GPUI's view in the window's content view.
pub(crate) struct NSViewWithMetalLayer {
    _view: Retained<NSView>,
    layer: Retained<CAMetalLayer>,
    cursor_locked: bool,
}

impl NSViewWithMetalLayer {
    pub fn set_cursor_locked(&mut self, locked: bool) {
        if self.cursor_locked == locked {
            return;
        }
        self.cursor_locked = locked;
        unsafe { CGAssociateMouseAndMouseCursorPosition(!locked as u32) };
        // `hide` and `unhide` nest, so each is called once per change.
        if locked {
            NSCursor::hide();
        } else {
            NSCursor::unhide();
        }
    }
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

        // MTLCreateSystemDefaultDevice is singleton
        layer.setDevice(objc2_metal::MTLCreateSystemDefaultDevice().as_deref());
        layer.setFramebufferOnly(false); // The tonemap pass writes them as storage images.
        layer.setPixelFormat(objc2_metal::MTLPixelFormat::RGB10A2Unorm);
        let color_space = objc2_core_graphics::CGColorSpace::with_name(Some(unsafe {
            objc2_core_graphics::kCGColorSpaceITUR_2100_PQ
        }));
        layer.setColorspace(color_space.as_deref());
        layer.setWantsExtendedDynamicRangeContent(true);
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
        NSViewWithMetalLayer {
            _view: view,
            layer,
            cursor_locked: false,
        }
    }

    /// Sizes the layer's drawables to the window's drawable area.
    pub(crate) fn set_size(&mut self, size: WindowSize) {
        self.layer.setContentsScale(size.scale_factor as f64);
        self.layer.setDrawableSize(CGSize::new(
            size.physical_width as f64,
            size.physical_height as f64,
        ));
    }

    pub fn create_bevy_runner(
        &mut self,
        from_ui: Receiver<crate::bridge::BevyCall>,
    ) -> impl FnOnce(bevy::app::App) -> bevy::app::AppExit + Send + 'static {
        struct LayerWrapper(Retained<CAMetalLayer>);
        unsafe impl Send for LayerWrapper {}
        let layer = LayerWrapper(self.layer.clone());
        move |mut app: bevy::app::App| {
            app.add_plugins(drawable::CAMetalDrawablePlugin);
            let layer_moved = layer;
            displaylink_runner(app, from_ui, &layer_moved.0)
        }
    }
}

use crate::bridge::BevyCall;

/// Runs `app` until it exits or the UI quits, one update per display frame of `layer`.
pub(crate) fn displaylink_runner(
    mut app: bevy::app::App,
    from_ui: Receiver<BevyCall>,
    layer: &CAMetalLayer,
) -> AppExit {
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }

    let delegate = BevyDisplayLinkDelegate::new(app, from_ui);
    let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), layer);
    link.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    // SAFETY: the link stays on this thread's run loop until
    // `BevyDisplayLinkDelegate::stop` invalidates it.
    unsafe { link.addToRunLoop_forMode(&NSRunLoop::currentRunLoop(), NSDefaultRunLoopMode) };
    // Returns once `BevyDisplayLinkDelegate::stop` stops it.
    CFRunLoop::run();

    // The link holds its delegate weakly, so `delegate` is its last reference: the
    // app is dropped with it, before this returns.
    delegate.ivars().exit.take().unwrap_or(AppExit::Success)
}

struct BevyDisplayLinkDelegateIvars {
    app: RefCell<bevy::app::App>,
    from_ui: Receiver<BevyCall>,
    /// Set when the app exits or the UI quits.
    exit: Cell<Option<AppExit>>,
}

define_class!(
    // SAFETY: `NSObject` has no subclassing requirements, and
    // `BevyDisplayLinkDelegate` doesn't implement `Drop`.
    #[unsafe(super(NSObject))]
    #[name = "BevyDisplayLinkDelegate"]
    #[ivars = BevyDisplayLinkDelegateIvars]
    struct BevyDisplayLinkDelegate;

    unsafe impl NSObjectProtocol for BevyDisplayLinkDelegate {}

    unsafe impl CAMetalDisplayLinkDelegate for BevyDisplayLinkDelegate {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn metal_display_link_needs_update(
            &self,
            link: &CAMetalDisplayLink,
            update: &CAMetalDisplayLinkUpdate,
        ) {
            self.update(link, update);
        }
    }
);

impl BevyDisplayLinkDelegate {
    fn new(app: bevy::app::App, from_ui: Receiver<BevyCall>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(BevyDisplayLinkDelegateIvars {
            app: RefCell::new(app),
            from_ui,
            exit: Cell::new(None),
        });
        // SAFETY: `NSObject`'s `init` is its designated initializer.
        unsafe { msg_send![super(this), init] }
    }

    /// One frame: the calls from the UI thread, in the order they were queued, then
    /// `app.update()`, drawing into the update's drawable.
    fn update(&self, link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
        let ivars = self.ivars();
        let mut app = ivars.app.borrow_mut();
        loop {
            match ivars.from_ui.try_recv() {
                Ok(call) => call(app.world_mut()),
                Err(TryRecvError::Empty) => break,
                // The UI has quit.
                Err(TryRecvError::Disconnected) => return self.stop(link, AppExit::Success),
            }
        }

        app.world_mut()
            .resource_mut::<super::drawable::MetalDrawable>()
            .drawable = Some(update.drawable());
        app.update();
        if let Some(exit) = app.should_exit() {
            self.stop(link, exit);
        }
    }

    fn stop(&self, link: &CAMetalDisplayLink, exit: AppExit) {
        link.invalidate();
        self.ivars().exit.set(Some(exit));
        CFRunLoop::current()
            .expect("the Bevy thread has a run loop")
            .stop();
    }
}
