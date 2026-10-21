//! The Bevy app's runner: a `CAMetalDisplayLink` on the game's layer calls
//! [`Delegate::update`] once per display frame, on the Bevy thread, with the drawable
//! to draw that frame into.

use std::cell::{Cell, RefCell};
use std::sync::mpsc::{Receiver, TryRecvError};

use bevy::app::{App, AppExit, PluginsState};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{AllocAnyThread, DefinedClass, define_class, msg_send};
use objc2_core_foundation::CFRunLoop;
use objc2_foundation::{NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSRunLoop};
use objc2_quartz_core::{
    CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate, CAMetalLayer,
};

use crate::bridge::BevyCall;

/// The game's layer, handed from the UI thread to the Bevy thread.
pub(crate) struct Layer(pub Retained<CAMetalLayer>);

// SAFETY: the Bevy thread only makes a display link on the layer, which vends its
// drawables there; a `CAMetalLayer` allows that from any thread.
unsafe impl Send for Layer {}

/// Runs `app` until it exits or the UI quits, one update per display frame of `layer`.
pub(crate) fn displaylink_runner(
    mut app: App,
    from_ui: Receiver<BevyCall>,
    layer: Layer,
) -> AppExit {
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }

    let delegate = Delegate::new(app, from_ui);
    let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), &layer.0);
    link.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    // SAFETY: the link stays on this thread's run loop until `Delegate::stop`
    // invalidates it.
    unsafe { link.addToRunLoop_forMode(&NSRunLoop::currentRunLoop(), NSDefaultRunLoopMode) };
    // Returns once `Delegate::stop` stops it.
    CFRunLoop::run();

    let ivars = delegate.ivars();
    // Dropped here rather than with the delegate, so its images of the layer's
    // drawables are gone before `run` returns.
    drop(ivars.app.take());
    ivars.exit.take().unwrap_or(AppExit::Success)
}

struct Ivars {
    /// Taken by [`run`] once the display link has stopped.
    app: RefCell<Option<App>>,
    from_ui: Receiver<BevyCall>,
    /// Set when the app exits or the UI quits.
    exit: Cell<Option<AppExit>>,
}

define_class!(
    // SAFETY: `NSObject` has no subclassing requirements, and `Delegate` doesn't
    // implement `Drop`.
    #[unsafe(super(NSObject))]
    #[name = "DustDisplayLinkDelegate"]
    #[ivars = Ivars]
    struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl CAMetalDisplayLinkDelegate for Delegate {
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

impl Delegate {
    fn new(app: App, from_ui: Receiver<BevyCall>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(Ivars {
            app: RefCell::new(Some(app)),
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
        let Some(app) = app.as_mut() else {
            return;
        };
        loop {
            match ivars.from_ui.try_recv() {
                Ok(call) => call(app.world_mut()),
                Err(TryRecvError::Empty) => break,
                // The UI has quit.
                Err(TryRecvError::Disconnected) => return self.stop(link, AppExit::Success),
            }
        }

        app.world_mut()
            .resource_mut::<super::drawable::MetalLayer>()
            .drawable = Some(update.drawable());
        app.update();
        if let Some(exit) = app.should_exit() {
            self.stop(link, exit);
        }
    }

    /// Stops the display link, and the run loop [`run`] waits in.
    fn stop(&self, link: &CAMetalDisplayLink, exit: AppExit) {
        link.invalidate();
        self.ivars().exit.set(Some(exit));
        CFRunLoop::current()
            .expect("the Bevy thread has a run loop")
            .stop();
    }
}
