//! The macOS [`GameLayer`]: the game's layer, under GPUI's, the cursor lock, and
//! the runner that updates the Bevy app from a `CAMetalDisplayLink` on the layer.

use gpui::{Bounds, Pixels, Window};
use objc2::rc::Retained;
use objc2::{MainThreadMarker, MainThreadOnly};
use objc2_app_kit::{NSCursor, NSView, NSWindowOrderingMode};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::cell::{Cell, RefCell};

use bevy::app::AppExit;
use objc2::runtime::ProtocolObject;
use objc2::{AllocAnyThread, DefinedClass, define_class, msg_send};
use objc2_core_foundation::CFRunLoop;
use objc2_foundation::{NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSRunLoop};
use objc2_quartz_core::{CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate};

use super::drawable;

unsafe extern "C" {
    // CoreGraphics. Detaches the cursor from the mouse, so it stays put while the
    // mouse still reports motion.
    fn CGAssociateMouseAndMouseCursorPosition(connected: u32) -> i32;
}
/// The game's view and layer, under GPUI's view in the window's content view,
/// where the `BevyView` is laid out.
pub(crate) struct GameLayer {
    view: Retained<NSView>,
    layer: Retained<CAMetalLayer>,
    cursor_locked: bool,
}

impl GameLayer {
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
    /// window's content view until [`Self::set_bounds`] fits it to the
    /// `BevyView`.
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
        content_view.addSubview_positioned_relativeTo(
            &view,
            NSWindowOrderingMode::Below,
            Some(gpui_view),
        );
        GameLayer {
            view,
            layer,
            cursor_locked: false,
        }
    }

    /// Fits the view to `bounds`, the `BevyView`'s, and sizes the layer's
    /// drawables to them, unless they already are. GPUI's coordinates start at
    /// the top left of its view, which fills the content view; the content
    /// view's start at the bottom left.
    pub(crate) fn set_bounds(&self, bounds: Bounds<Pixels>, scale_factor: f32) {
        let content_view =
            unsafe { self.view.superview() }.expect("the game's view is in the content view");
        let content_height = content_view.bounds().size.height;
        let frame = CGRect::new(
            CGPoint::new(
                bounds.origin.x.as_f32() as f64,
                content_height - bounds.bottom().as_f32() as f64,
            ),
            CGSize::new(
                bounds.size.width.as_f32() as f64,
                bounds.size.height.as_f32() as f64,
            ),
        );
        let scale = scale_factor as f64;
        let drawable_size = bounds.size.to_device_pixels(scale_factor);
        let drawable_size =
            CGSize::new(drawable_size.width.0 as f64, drawable_size.height.0 as f64);
        if self.view.frame() == frame
            && self.layer.contentsScale() == scale
            && self.layer.drawableSize() == drawable_size
        {
            return;
        }
        self.view.setFrame(frame);
        self.layer.setContentsScale(scale);
        self.layer.setDrawableSize(drawable_size);
    }

    pub fn create_bevy_runner(
        &mut self,
    ) -> impl FnOnce(bevy::app::App) -> bevy::app::AppExit + Send + 'static {
        struct LayerWrapper(Retained<CAMetalLayer>);
        unsafe impl Send for LayerWrapper {}
        let layer = LayerWrapper(self.layer.clone());
        move |mut app: bevy::app::App| {
            let layer_moved = layer;
            app.add_plugins(drawable::CAMetalLayerPlugin {
                layer: layer_moved.0.clone(),
            });
            displaylink_runner(app, &layer_moved.0)
        }
    }
}

/// Runs `app` until it exits or the UI quits, one update per display frame of `layer`.
pub(crate) fn displaylink_runner(mut app: bevy::app::App, layer: &CAMetalLayer) -> AppExit {
    crate::finish_plugins(&mut app);

    let delegate = BevyDisplayLinkDelegate::new(app);
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
    fn new(app: bevy::app::App) -> Retained<Self> {
        let this = Self::alloc().set_ivars(BevyDisplayLinkDelegateIvars {
            app: RefCell::new(app),
            exit: Cell::new(None),
        });
        // SAFETY: `NSObject`'s `init` is its designated initializer.
        unsafe { msg_send![super(this), init] }
    }

    /// One frame: `app.update()`, drawing into the update's drawable. The calls
    /// from the UI thread run at the end of it, in `Last` (see `bridge`).
    fn update(&self, link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
        let ivars = self.ivars();
        let mut app = ivars.app.borrow_mut();

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
