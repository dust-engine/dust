//! DevTools: a Chrome-DevTools-style dock that hosts debug panels as tabs.
//!
//! The dock is a GPUI view on the UI thread, drawn over the game by
//! [`overlay`], or, detached, in a window of its own. Its tabs are GPUI views
//! too, drawn in whichever window the dock is in: the dock creates its own, and
//! [`DevTools::add_tab`] adds more. A view shows the world with [`watch`] or
//! [`read`], called from its `render`, or [`fetch`], called when the user
//! asks for something else, and edits it through `dust_ui`'s `BevyChannel`:
//!
//! ```ignore
//! let devtools = dust_devtools::overlay(console_log, window, cx);
//! devtools.update(cx, |devtools, cx| {
//!     devtools.add_tab("Physics", cx.new(|cx| PhysicsTab::new(window, cx)), cx)
//! });
//!
//! impl Render for PhysicsTab {
//!     fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
//!         read(window, cx, |gravity: Res<Gravity>| gravity.0, |this, gravity, window, cx| {
//!             this.gravity.set(gravity.y, window, cx)
//!         });
//!         // ...
//!     }
//! }
//! // and in a control's handler:
//! cx.global::<BevyChannel>().update(move |world| world.resource_mut::<Gravity>().0.y = y);
//! ```
//!
//! F12, or ⌘⌥I as in Chrome on macOS, toggles DevTools.

use std::panic::Location;

use bevy::ecs::system::{RunSystemOnce as _, SystemHandle};
use bevy::prelude::{DetectChanges as _, In, IntoSystem, Res, World, warn, warn_once};
use dust_profiler::GpuProfiler;
use dust_ui::{BevyChannel, GPUIChannel};
use gpui::{
    AnyView, AnyWindowHandle, App, AppContext as _, Bounds, Context, ElementId, Entity,
    InteractiveElement as _, IntoElement, KeyBinding, ParentElement as _, Render, SharedString,
    Styled as _, TitlebarOptions, WeakEntity, Window, WindowBounds, WindowHandle, WindowOptions,
    actions, div, prelude::FluentBuilder as _, px, size,
};
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_component::tab::TabBar;
use gpui_component::{
    ActiveTheme as _, IconName, Root, Sizable as _, Theme, ThemeMode, h_flex, h_resizable,
    resizable_panel, v_flex, v_resizable,
};

use crate::console::{ConsoleLog, ConsoleView};
use crate::entities::EntitiesView;
use crate::graphics_debug::GraphicsDebugView;
use crate::performance::{PerformanceOverlay, PerformancePanel};
use crate::player::PlayerOverlay;
use crate::schedules::SchedulesView;

actions!(dust_devtools, [ToggleDevTools]);

/// Where the DevTools panel sits: Chrome's "Dock side" options.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DockSide {
    Left,
    Bottom,
    Right,
    /// In a window of its own, as Chrome's "Undock into separate window".
    /// Closing the window closes DevTools.
    Detached,
}

/// Builds the overlay drawn over the game's `BevyView`: the DevTools dock, with
/// its Graphics Debug, Entities, Schedules and Console tabs, and the
/// performance and player overlays. [`DevTools::add_tab`] adds more tabs.
///
/// The console shows the messages logged to `console_log`; see
/// [`console_layer`](crate::console_layer).
pub fn overlay(console_log: ConsoleLog, window: &mut Window, cx: &mut App) -> Entity<DevTools> {
    Theme::change(ThemeMode::Dark, Some(window), cx);

    let devtools = cx.new(|cx| DevTools::new(console_log, window, cx));
    cx.bind_keys([
        KeyBinding::new("f12", ToggleDevTools, None),
        KeyBinding::new("cmd-alt-i", ToggleDevTools, None),
    ]);
    cx.on_action({
        let devtools = devtools.downgrade();
        move |_: &ToggleDevTools, cx| {
            devtools
                .update(cx, |devtools, cx| devtools.set_open(!devtools.open, cx))
                .ok();
        }
    });
    devtools
}

/// The overlay's root view: the dock, and the overlays over the part of the
/// window it leaves to the game. Detached, the dock is in a window of its own,
/// drawn by a [`DetachedPanel`], and the overlays have the whole window.
pub struct DevTools {
    /// Whether DevTools is shown. Toggled by F12 / ⌘⌥I and the ✕ button, and
    /// cleared by closing the detached window.
    open: bool,
    dock: DockSide,
    /// The window the dock is in while DevTools is open and detached; see
    /// [`DevTools::sync_window`].
    window: Option<WindowHandle<Root>>,
    /// Where the detached window was when it last closed, to open it there
    /// again.
    window_bounds: Option<WindowBounds>,
    /// Index of the shown tab in `tabs`.
    selected: usize,
    /// Title and view of each tab, in the order they show.
    tabs: Vec<(SharedString, AnyView)>,
    performance: Entity<PerformanceOverlay>,
    performance_open: bool,
    player: Entity<PlayerOverlay>,
    player_open: bool,
    /// Shown in the tab strip.
    average_fps: f32,
    /// Whether the device supports GPU timings, and whether they are measured;
    /// shown in the ⋮ menu.
    gpu_timings: Option<(bool, bool)>,
}

/// The average frame rate, each frame.
fn average_fps(performance: Option<Res<PerformancePanel>>) -> Option<f32> {
    performance
        .filter(|panel| panel.is_changed())
        .map(|panel| panel.average_fps())
}

fn gpu_timings(profiler: Option<Res<GpuProfiler>>) -> Option<(bool, bool)> {
    profiler.map(|profiler| (profiler.is_supported(), profiler.enabled))
}

impl DevTools {
    fn new(console_log: ConsoleLog, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let graphics_debug = cx.new(|cx| GraphicsDebugView::new(window, cx));
        let entities = cx.new(|cx| EntitiesView::new(window, cx));
        let schedules = cx.new(|cx| SchedulesView::new(window, cx));
        let console = cx.new(|cx| ConsoleView::new(console_log, window, cx));
        DevTools {
            open: true,
            dock: DockSide::Right,
            window: None,
            window_bounds: None,
            selected: 0,
            tabs: vec![
                ("Graphics Debug".into(), graphics_debug.into()),
                ("Entities".into(), entities.into()),
                ("Schedules".into(), schedules.into()),
                ("Console".into(), console.into()),
            ],
            performance: cx.new(|cx| PerformanceOverlay::new(window, cx)),
            performance_open: true,
            player: cx.new(|cx| PlayerOverlay::new(window, cx)),
            player_open: true,
            average_fps: 0.0,
            gpu_timings: None,
        }
    }

    /// Adds a tab titled `title` showing `view`, after the tabs already there.
    pub fn add_tab(
        &mut self,
        title: impl Into<SharedString>,
        view: impl Into<AnyView>,
        cx: &mut Context<Self>,
    ) {
        self.tabs.push((title.into(), view.into()));
        cx.notify();
    }

    /// Shows or hides DevTools.
    fn set_open(&mut self, open: bool, cx: &mut Context<Self>) {
        self.open = open;
        self.sync_window(cx);
    }

    fn set_dock(&mut self, dock: DockSide, cx: &mut Context<Self>) {
        self.dock = dock;
        self.sync_window(cx);
    }

    /// Redraws, and opens or closes the detached window to match `open` and
    /// `dock`: it is open while DevTools is open and detached.
    ///
    /// The window opens and closes once the current update is over: opening it
    /// draws it, which reads DevTools, being updated now; and the change can
    /// come from a click in the window, which can't be closed while it handles
    /// the click.
    fn sync_window(&self, cx: &mut Context<Self>) {
        cx.notify();
        let devtools = cx.entity();
        cx.defer(move |cx| {
            let this = devtools.read(cx);
            let detached = this.open && this.dock == DockSide::Detached;
            match (detached, this.window, this.window_bounds) {
                (true, None, bounds) => {
                    let window = open_detached(devtools.clone(), bounds, cx);
                    devtools.update(cx, |this, _| this.window = window);
                }
                (false, Some(window), _) => {
                    let bounds = window
                        .update(cx, |_, window, _| {
                            window.remove_window();
                            window.window_bounds()
                        })
                        .ok();
                    devtools.update(cx, |this, _| {
                        this.window = None;
                        this.window_bounds = bounds.or(this.window_bounds);
                    });
                }
                _ => {}
            }
        });
    }

    /// The part of the window the dock leaves uncovered, with the overlays.
    fn scene(&self) -> impl IntoElement + use<> {
        div()
            .relative()
            .size_full()
            .when(self.performance_open, |scene| {
                scene.child(
                    div()
                        .absolute()
                        .top_2()
                        .left_2()
                        .child(self.performance.clone()),
                )
            })
            .when(self.player_open, |scene| {
                scene.child(
                    div()
                        .absolute()
                        .bottom_2()
                        .left_2()
                        .child(self.player.clone()),
                )
            })
    }

    /// Tab strip, then the selected tab's contents; drawn in `window`, the
    /// game's or the detached one.
    fn panel(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        watch(window, cx, average_fps, |this, fps, _, _| {
            this.average_fps = fps
        });
        read(window, cx, gpu_timings, |this, timings, _, _| {
            this.gpu_timings = timings
        });
        let menu = self.menu(cx);
        let theme = cx.theme();
        // The tab bar draws the border under the header, so the frame rate and
        // the buttons go in its suffix, above the same border.
        let header = TabBar::new("devtools-tabs")
            .underline()
            .small()
            .px_2()
            .children(self.tabs.iter().map(|(title, _)| title.clone()))
            .selected_index(self.selected)
            .on_click(cx.listener(|this, index: &usize, _, cx| {
                this.selected = *index;
                cx.notify();
            }))
            .suffix(
                h_flex()
                    .gap_1()
                    // Frame rate stays in view whichever tab is selected, where
                    // Chrome keeps its error count.
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .font_family(theme.mono_font_family.clone())
                            .child(format!("{:.0} FPS", self.average_fps)),
                    )
                    .child(menu)
                    // Detached, the window's own close button closes DevTools.
                    .when(self.dock != DockSide::Detached, |suffix| {
                        suffix.child(
                            Button::new("devtools-close")
                                .icon(IconName::Close)
                                .ghost()
                                .xsmall()
                                .tooltip("Close DevTools (F12)")
                                .on_click(cx.listener(|this, _, _, cx| this.set_open(false, cx))),
                        )
                    }),
            );
        let contents = match self.tabs.get(self.selected) {
            Some((_, view)) => div().flex_1().min_h_0().child(view.clone()),
            None => div()
                .p_2()
                .text_color(theme.muted_foreground)
                .child("No debug panels registered."),
        };
        v_flex()
            .id("devtools-panel")
            .size_full()
            .occlude()
            // The size gpui-component's controls draw their text at, so the
            // panels' own text matches them.
            .text_sm()
            .bg(theme.background)
            .border_color(theme.border)
            .map(|panel| match self.dock {
                DockSide::Left => panel.border_r_1(),
                DockSide::Right => panel.border_l_1(),
                DockSide::Bottom => panel.border_t_1(),
                DockSide::Detached => panel,
            })
            .child(header)
            .child(contents)
    }

    /// The ⋮ menu: the dock side, and which overlays show.
    fn menu(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let devtools = cx.weak_entity();
        Button::new("devtools-menu")
            .icon(IconName::EllipsisVertical)
            .ghost()
            .xsmall()
            .tooltip("Customize DevTools")
            .dropdown_menu(move |mut menu, _, cx| {
                let Some(this) = devtools.upgrade() else {
                    return menu;
                };
                let this = this.read(cx);
                menu = menu.label("Dock side");
                for (side, label) in [
                    (DockSide::Detached, "Detached"),
                    (DockSide::Left, "Left"),
                    (DockSide::Bottom, "Bottom"),
                    (DockSide::Right, "Right"),
                ] {
                    let devtools = devtools.clone();
                    menu = menu.item(
                        PopupMenuItem::new(label)
                            .checked(this.dock == side)
                            .on_click(move |_, _, cx| {
                                devtools
                                    .update(cx, |devtools, cx| devtools.set_dock(side, cx))
                                    .ok();
                            }),
                    );
                }
                menu = menu.separator();
                let toggle = |label: &'static str, checked: bool, flip: fn(&mut DevTools)| {
                    let devtools = devtools.clone();
                    PopupMenuItem::new(label)
                        .checked(checked)
                        .on_click(move |_, _, cx| {
                            devtools
                                .update(cx, |devtools, cx| {
                                    flip(devtools);
                                    cx.notify();
                                })
                                .ok();
                        })
                };
                menu = menu
                    .item(toggle("Performance overlay", this.performance_open, |d| {
                        d.performance_open = !d.performance_open
                    }))
                    .item(toggle("Player overlay", this.player_open, |d| {
                        d.player_open = !d.player_open
                    }));
                if let Some((supported, enabled)) = this.gpu_timings {
                    let devtools = devtools.clone();
                    menu = menu.item(
                        PopupMenuItem::new(if supported {
                            "GPU timings"
                        } else {
                            "GPU timings (timestamp queries unsupported)"
                        })
                        .checked(enabled)
                        .disabled(!supported)
                        .on_click(move |_, _, cx| {
                            devtools
                                .update(cx, |devtools, cx| {
                                    devtools.gpu_timings = Some((supported, !enabled));
                                    cx.notify();
                                })
                                .ok();
                            cx.global::<BevyChannel>().update(move |world: &mut World| {
                                if let Some(mut profiler) = world.get_resource_mut::<GpuProfiler>()
                                {
                                    profiler.enabled = !enabled;
                                }
                            });
                        }),
                    );
                }
                menu
            })
    }
}

impl Render for DevTools {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let scene = self.scene();
        if !self.open {
            return div().size_full().child(scene).into_any_element();
        }
        match self.dock {
            DockSide::Right => h_resizable("devtools-dock-right")
                .child(resizable_panel().child(scene))
                .child(
                    resizable_panel()
                        .size(px(480.))
                        .size_range(px(160.)..px(4096.))
                        .child(self.panel(window, cx)),
                )
                .into_any_element(),
            DockSide::Left => h_resizable("devtools-dock-left")
                .child(
                    resizable_panel()
                        .size(px(480.))
                        .size_range(px(160.)..px(4096.))
                        .child(self.panel(window, cx)),
                )
                .child(resizable_panel().child(scene))
                .into_any_element(),
            DockSide::Bottom => v_resizable("devtools-dock-bottom")
                .child(resizable_panel().child(scene))
                .child(
                    resizable_panel()
                        .size(px(320.))
                        .size_range(px(120.)..px(4096.))
                        .child(self.panel(window, cx)),
                )
                .into_any_element(),
            // The panel is in its own window.
            DockSide::Detached => div().size_full().child(scene).into_any_element(),
        }
    }
}

/// The detached window's root view: DevTools' panel, filling the window.
struct DetachedPanel(Entity<DevTools>);

impl Render for DetachedPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Updating DevTools here, as reading it, has this window redrawn when
        // DevTools notifies.
        self.0.update(cx, |devtools, cx| devtools.panel(window, cx))
    }
}

/// Opens the window the panel is in while DevTools is detached, at `bounds`,
/// or centered on the screen. Closing the window closes DevTools.
fn open_detached(
    devtools: Entity<DevTools>,
    bounds: Option<WindowBounds>,
    cx: &mut App,
) -> Option<WindowHandle<Root>> {
    let bounds = bounds.unwrap_or_else(|| {
        WindowBounds::Windowed(Bounds::centered(None, size(px(800.), px(600.)), cx))
    });
    let options = WindowOptions {
        window_bounds: Some(bounds),
        titlebar: Some(TitlebarOptions {
            title: Some("DevTools".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    cx.open_window(options, |window, cx| {
        window.on_window_should_close(cx, {
            let devtools = devtools.downgrade();
            move |_, cx| {
                // Which closes the window, through `sync_window`.
                devtools
                    .update(cx, |devtools, cx| devtools.set_open(false, cx))
                    .ok();
                false
            }
        });
        let panel = cx.new(|_| DetachedPanel(devtools));
        cx.new(|cx| Root::new(panel, window, cx))
    })
    .inspect_err(|error| warn!("Failed to open the DevTools window: {error}"))
    .ok()
}

/// What [`watch`] and [`read`] do with a value from the Bevy thread, on the UI
/// thread.
type Apply<V, T> = fn(&mut V, T, &mut Window, &mut Context<V>);

/// Runs `system` on the Bevy thread at the end of every Bevy frame, for as long
/// as the view calling this is drawn. Each `Some` it returns goes to `apply`,
/// which updates the view; the view then redraws. Call it from the view's
/// `render`, each time it renders.
///
/// The system's `SystemHandle` is kept in GPUI's keyed state for this call
/// site ([`Window::use_keyed_state`]), which GPUI drops in the first frame the
/// view isn't drawn in; dropping the handle stops the system. The system can
/// use change detection to send only what changed: its first run sees
/// everything as changed, so the view gets the current value once it is shown.
#[track_caller]
pub fn watch<V: 'static, T: Send + 'static, M>(
    window: &mut Window,
    cx: &mut Context<V>,
    system: impl IntoSystem<(), Option<T>, M> + Send + 'static,
    apply: Apply<V, T>,
) {
    let view = cx.weak_entity();
    let handle = window.window_handle();
    let key = ElementId::CodeLocation(*Location::caller());
    window.use_keyed_state(key, cx, move |_, cx| {
        let send = move |In(value): In<Option<T>>, ui: Res<GPUIChannel>| {
            if let Some(value) = value {
                send_to_view(&ui, view.clone(), handle, apply, value);
            }
        };
        let slot = cx.weak_entity();
        cx.global::<BevyChannel>()
            .run_every_frame(system.pipe(send), move |system, cx| {
                // Once the view stops being drawn the slot is gone, and
                // `system` is dropped here instead.
                slot.update(cx, |slot, _| *slot = Some(system)).ok();
            });
        None::<SystemHandle>
    });
}

/// Runs `system` on the Bevy thread once, when the view calling this starts
/// being drawn, and passes what it returns to `apply`. Call it from the view's
/// `render`, each time it renders: it runs `system` again only after the view
/// was hidden and shown again.
#[track_caller]
pub fn read<V: 'static, T: Send + 'static, M>(
    window: &mut Window,
    cx: &mut Context<V>,
    system: impl IntoSystem<(), T, M> + Send + 'static,
    apply: Apply<V, T>,
) {
    let view = cx.weak_entity();
    let handle = window.window_handle();
    let key = ElementId::CodeLocation(*Location::caller());
    window.use_keyed_state(key, cx, move |_, cx| {
        run_once(view, handle, system, apply, cx)
    });
}

/// Runs `system` on the Bevy thread once, at the end of the next Bevy frame,
/// and passes what it returns to `apply`, as [`read`] does, but each time
/// it's called: from an event handler, to read the world again when the user
/// asks for something else.
pub fn fetch<V: 'static, T: Send + 'static, M>(
    window: &Window,
    cx: &mut Context<V>,
    system: impl IntoSystem<(), T, M> + Send + 'static,
    apply: Apply<V, T>,
) {
    run_once(cx.weak_entity(), window.window_handle(), system, apply, cx);
}

fn run_once<V: 'static, T: Send + 'static, M>(
    view: WeakEntity<V>,
    window: AnyWindowHandle,
    system: impl IntoSystem<(), T, M> + Send + 'static,
    apply: Apply<V, T>,
    cx: &App,
) {
    cx.global::<BevyChannel>()
        .update(move |world| match world.run_system_once(system) {
            Ok(value) => send_to_view(world.resource::<GPUIChannel>(), view, window, apply, value),
            Err(error) => warn_once!("DevTools read failed: {error}"),
        });
}

/// Has the UI thread pass `value` to `apply` on `view`, then redraw the view.
fn send_to_view<V: 'static, T: Send + 'static>(
    ui: &GPUIChannel,
    view: WeakEntity<V>,
    window: AnyWindowHandle,
    apply: Apply<V, T>,
    value: T,
) {
    ui.update(move |cx| {
        window
            .update(cx, |_, window, cx| {
                view.update(cx, |view, cx| {
                    apply(view, value, window, cx);
                    cx.notify();
                })
            })
            .ok();
    });
}
