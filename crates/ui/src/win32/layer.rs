//! The Windows [`GameLayer`]: the game's DirectComposition visual, under GPUI's,
//! the cursor lock, and the runner that updates the Bevy app once per compositor
//! frame.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bevy::app::AppExit;
use gpui::{Bounds, Pixels, Window};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows::Win32::Foundation::{CloseHandle, HWND, POINT, RECT};
use windows::Win32::Graphics::DirectComposition::{
    DCompositionCreateDevice2, DCompositionCreateSurfaceHandle,
    DCompositionWaitForCompositorClock, IDCompositionDesktopDevice, IDCompositionTarget,
    IDCompositionVisual2,
};
use windows::Win32::UI::WindowsAndMessaging::{
    ClipCursor, GWL_EXSTYLE, GetCursorPos, GetWindowLongPtrW, ShowCursor,
    WS_EX_NOREDIRECTIONBITMAP,
};
use windows::core::IUnknown;

use super::swapchain::{CompositionSwapchainPlugin, SurfaceHandle, pack_size};

/// `COMPOSITIONOBJECT_READ | COMPOSITIONOBJECT_WRITE`, which the `windows` crate
/// doesn't define.
const COMPOSITIONOBJECT_ALL_ACCESS: u32 = 0x3;

/// The game's visual, in the window's non-topmost DirectComposition target. GPUI's
/// target is the topmost one, so the game shows under GPUI's UI. The visual's
/// content is a composition surface, which the swapchain presents to.
pub(crate) struct GameLayer {
    device: IDCompositionDesktopDevice,
    _target: IDCompositionTarget,
    visual: IDCompositionVisual2,
    /// The surface's handle, until [`Self::create_bevy_runner`] hands it to the
    /// swapchain.
    surface: Option<SurfaceHandle>,
    /// The game's size in physical pixels (see [`pack_size`]), for the swapchain.
    size: Arc<AtomicU64>,
    /// The visual's offset in the window, in physical pixels.
    offset: Cell<(f32, f32)>,
    cursor_locked: bool,
}

impl GameLayer {
    pub fn set_cursor_locked(&mut self, locked: bool) {
        if self.cursor_locked == locked {
            return;
        }
        self.cursor_locked = locked;
        unsafe {
            // `ShowCursor` keeps a count, so it's called once per change.
            ShowCursor(!locked);
            // Clipped to where it is, the cursor stays put while the mouse still
            // reports motion. Windows drops the clip when another window
            // activates, which detaches the cursor anyway.
            if locked {
                let mut at = POINT::default();
                if GetCursorPos(&mut at).is_ok() {
                    let rect = RECT {
                        left: at.x,
                        top: at.y,
                        right: at.x + 1,
                        bottom: at.y + 1,
                    };
                    ClipCursor(Some(&rect)).ok();
                }
            } else {
                ClipCursor(None).ok();
            }
        }
    }

    /// Adds the game's visual under GPUI's, filling the window's client area
    /// until [`Self::set_bounds`] fits it to the `BevyView`.
    pub(crate) fn new(window: &Window) -> Self {
        let RawWindowHandle::Win32(handle) =
            HasWindowHandle::window_handle(window).unwrap().as_raw()
        else {
            unreachable!("GPUI windows on Windows have Win32 handles");
        };
        let hwnd = HWND(handle.hwnd.get() as *mut _);
        // Without DirectComposition (`GPUI_DISABLE_DIRECT_COMPOSITION`), GPUI
        // draws into an opaque swapchain, and nothing shows under it.
        let style = unsafe { GetWindowLongPtrW(hwnd, GWL_EXSTYLE) } as u32;
        if style & WS_EX_NOREDIRECTIONBITMAP.0 == 0 {
            tracing::error!(
                "GPUI isn't drawing with DirectComposition, so the game can't show under the UI"
            );
        }

        // No rendering device: the visual's only content is the surface.
        let device: IDCompositionDesktopDevice =
            unsafe { DCompositionCreateDevice2(None::<&IUnknown>) }
                .expect("failed to create a DirectComposition device");
        let target = unsafe { device.CreateTargetForHwnd(hwnd, false) }
            .expect("failed to create the game's DirectComposition target");
        let visual = unsafe { device.CreateVisual() }.unwrap();
        let surface = unsafe { DCompositionCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, None) }
            .expect("failed to create a composition surface handle");
        unsafe {
            visual
                .SetContent(&device.CreateSurfaceFromHandle(surface).unwrap())
                .unwrap();
            target.SetRoot(&visual).unwrap();
            device.Commit().unwrap();
        }

        let viewport = window.viewport_size().to_device_pixels(window.scale_factor());
        GameLayer {
            device,
            _target: target,
            visual,
            surface: Some(SurfaceHandle(surface)),
            size: Arc::new(AtomicU64::new(pack_size(
                viewport.width.0,
                viewport.height.0,
            ))),
            offset: Cell::new((0.0, 0.0)),
            cursor_locked: false,
        }
    }

    /// Moves the visual to `bounds`, the `BevyView`'s, and sizes the swapchain's
    /// buffers to them. GPUI's coordinates start at the top left of the client
    /// area, as the visual's do. Empty bounds are ignored: the renderer can't size
    /// its images to them.
    pub(crate) fn set_bounds(&self, bounds: Bounds<Pixels>, scale_factor: f32) {
        let size = bounds.size.to_device_pixels(scale_factor);
        if size.width.0 <= 0 || size.height.0 <= 0 {
            return;
        }
        self.size
            .store(pack_size(size.width.0, size.height.0), Ordering::Relaxed);
        let offset = (
            (bounds.origin.x.as_f32() * scale_factor).round(),
            (bounds.origin.y.as_f32() * scale_factor).round(),
        );
        if self.offset.get() == offset {
            return;
        }
        self.offset.set(offset);
        unsafe {
            self.visual.SetOffsetX2(offset.0).unwrap();
            self.visual.SetOffsetY2(offset.1).unwrap();
            self.device.Commit().unwrap();
        }
    }

    pub fn create_bevy_runner(
        &mut self,
    ) -> impl FnOnce(bevy::app::App) -> bevy::app::AppExit + Send + 'static {
        let surface = self.surface.take().expect("a GameLayer has one runner");
        let size = self.size.clone();
        move |mut app: bevy::app::App| {
            app.add_plugins(CompositionSwapchainPlugin { surface, size });
            let exit = compositor_clock_runner(app);
            // The app, and with it the presentation surface, is gone.
            unsafe { CloseHandle(surface.0) }.ok();
            exit
        }
    }
}

/// Runs `app` until it exits or the UI quits, one update per compositor frame.
fn compositor_clock_runner(mut app: bevy::app::App) -> AppExit {
    crate::finish_plugins(&mut app);
    loop {
        let start = Instant::now();
        unsafe { DCompositionWaitForCompositorClock(None, 100) };
        // It returns at once while the display is off; wait about a frame then
        // instead, as GPUI's `VSyncProvider` does.
        if start.elapsed() < Duration::from_millis(1) {
            std::thread::sleep(Duration::from_millis(16));
        }
        app.update();
        if let Some(exit) = app.should_exit() {
            return exit;
        }
    }
}
