//! DevTools: a Chrome-DevTools-style dock that hosts debug panels as tabs.
//!
//! The dock is a GPUI view on the UI thread, drawn over the game by
//! [`overlay`]. A Bevy plugin adds a tab with
//! [`DevToolsAppExt::add_devtools_tab`], passing the constructor of the tab's
//! view. A view shows the world with [`watch`] or [`read`], called from its
//! `render`, and edits it through `dust_ui`'s `BevyHandle`:
//!
//! ```ignore
//! app.add_devtools_tab("Physics", PhysicsTab::new);
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
//! cx.global::<BevyHandle>().update(move |world| world.resource_mut::<Gravity>().0.y = y);
//! ```
//!
//! F12, or ⌘⌥I as in Chrome on macOS, toggles DevTools.

use std::panic::Location;

use bevy::ecs::system::{RunSystemOnce as _, SystemHandle};
use bevy::prelude::{
    DetectChanges as _, In, IntoSystem, Res, ResMut, Resource, Startup, World, warn_once,
};
use dust_profiler::GpuProfiler;
use dust_ui::{BevyHandle, UiHandle};
use gpui::{
    AnyView, AnyWindowHandle, App, AppContext as _, Context, ElementId, Entity,
    InteractiveElement as _, IntoElement, KeyBinding, ParentElement as _, Render, SharedString,
    Styled as _, WeakEntity, Window, actions, div, prelude::FluentBuilder as _, px,
};
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::menu::{DropdownMenu as _, PopupMenuItem};
use gpui_component::tab::TabBar;
use gpui_component::{
    ActiveTheme as _, IconName, Sizable as _, Theme, ThemeMode, h_flex, h_resizable,
    resizable_panel, v_flex, v_resizable,
};

use crate::console::ConsoleSink;
use crate::performance::{PerformanceOverlay, PerformancePanel};
use crate::player::PlayerOverlay;

actions!(dust_devtools, [ToggleDevTools]);

/// Where the DevTools panel sits: Chrome's "Dock side" options.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DockSide {
    Left,
    Bottom,
    Right,
    /// A panel floating over the game, standing in for Chrome's "Undock into
    /// separate window".
    Floating,
}

/// Builds the overlay drawn over the game's `BevyView`: the DevTools dock and the
/// performance and player overlays.
pub fn overlay(window: &mut Window, cx: &mut App) -> AnyView {
    Theme::change(ThemeMode::Dark, Some(window), cx);
    cx.set_global(ConsoleSink::new(window.window_handle()));

    let devtools = cx.new(|cx| DevTools::new(window, cx));
    cx.set_global(DevToolsHandle {
        view: devtools.downgrade(),
        window: window.window_handle(),
    });
    cx.bind_keys([
        KeyBinding::new("f12", ToggleDevTools, None),
        KeyBinding::new("cmd-alt-i", ToggleDevTools, None),
    ]);
    cx.on_action({
        let devtools = devtools.downgrade();
        move |_: &ToggleDevTools, cx| {
            devtools
                .update(cx, |devtools, cx| {
                    devtools.open = !devtools.open;
                    cx.notify();
                })
                .ok();
        }
    });
    devtools.into()
}

/// The [`DevTools`] view and its window, for the tabs and the console's `dock`
/// command.
#[derive(Clone)]
struct DevToolsHandle {
    view: WeakEntity<DevTools>,
    window: AnyWindowHandle,
}

impl gpui::Global for DevToolsHandle {}

pub(crate) fn set_dock(side: DockSide, cx: &mut App) {
    let devtools = cx.global::<DevToolsHandle>().view.clone();
    devtools
        .update(cx, |devtools, cx| {
            devtools.dock = side;
            cx.notify();
        })
        .ok();
}

/// The overlay's root view: the dock, and the overlays over the part of the
/// window it leaves to the game.
pub struct DevTools {
    /// Whether DevTools is shown. Toggled by F12 / ⌘⌥I and the ✕ button.
    open: bool,
    dock: DockSide,
    /// Index of the shown tab, in registration order.
    selected: usize,
    tabs: Vec<(&'static str, AnyView)>,
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
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        DevTools {
            open: true,
            dock: DockSide::Right,
            selected: 0,
            tabs: Vec::new(),
            performance: cx.new(|cx| PerformanceOverlay::new(window, cx)),
            performance_open: true,
            player: cx.new(|cx| PlayerOverlay::new(window, cx)),
            player_open: true,
            average_fps: 0.0,
            gpu_timings: None,
        }
    }

    fn add_tabs(&mut self, tabs: Vec<Tab>, window: &mut Window, cx: &mut Context<Self>) {
        for tab in tabs {
            let view = (tab.build)(window, cx);
            self.tabs.push((tab.title, view));
        }
        cx.notify();
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

    /// Tab strip, then the selected tab's contents.
    fn panel(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let menu = self.menu(cx);
        let theme = cx.theme();
        let tab_bar = TabBar::new("devtools-tabs")
            .underline()
            .small()
            .flex_1()
            .children(
                self.tabs
                    .iter()
                    .map(|(title, _)| SharedString::from(*title)),
            )
            .selected_index(self.selected)
            .on_click(cx.listener(|this, index: &usize, _, cx| {
                this.selected = *index;
                cx.notify();
            }));
        let header = h_flex()
            .gap_1()
            .pr_1()
            .border_b_1()
            .border_color(theme.border)
            .child(tab_bar)
            // Frame rate stays in view whichever tab is selected, where Chrome
            // keeps its error count.
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .font_family(theme.mono_font_family.clone())
                    .child(format!("{:.0} FPS", self.average_fps)),
            )
            .child(menu)
            .child(
                Button::new("devtools-close")
                    .icon(IconName::Close)
                    .ghost()
                    .xsmall()
                    .tooltip("Close DevTools (F12)")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.open = false;
                        cx.notify();
                    })),
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
            .bg(theme.background)
            .border_color(theme.border)
            .map(|panel| match self.dock {
                DockSide::Left => panel.border_r_1(),
                DockSide::Right => panel.border_l_1(),
                DockSide::Bottom => panel.border_t_1(),
                DockSide::Floating => panel.border_1().rounded_md().shadow_lg(),
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
                    (DockSide::Floating, "Undock"),
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
                                    .update(cx, |devtools, cx| {
                                        devtools.dock = side;
                                        cx.notify();
                                    })
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
                            cx.global::<BevyHandle>().update(move |world: &mut World| {
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
        watch(window, cx, average_fps, |this, fps, _, _| {
            this.average_fps = fps
        });
        read(window, cx, gpu_timings, |this, timings, _, _| {
            this.gpu_timings = timings
        });
        let panel = self.panel(cx);
        match self.dock {
            DockSide::Right => h_resizable("devtools-dock-right")
                .child(resizable_panel().child(scene))
                .child(
                    resizable_panel()
                        .size(px(480.))
                        .size_range(px(160.)..px(4096.))
                        .child(panel),
                )
                .into_any_element(),
            DockSide::Left => h_resizable("devtools-dock-left")
                .child(
                    resizable_panel()
                        .size(px(480.))
                        .size_range(px(160.)..px(4096.))
                        .child(panel),
                )
                .child(resizable_panel().child(scene))
                .into_any_element(),
            DockSide::Bottom => v_resizable("devtools-dock-bottom")
                .child(resizable_panel().child(scene))
                .child(
                    resizable_panel()
                        .size(px(320.))
                        .size_range(px(120.)..px(4096.))
                        .child(panel),
                )
                .into_any_element(),
            DockSide::Floating => div()
                .size_full()
                .relative()
                .child(scene)
                .child(
                    div()
                        .absolute()
                        .top(px(40.))
                        .right(px(40.))
                        .w(px(640.))
                        .h(px(480.))
                        .child(panel),
                )
                .into_any_element(),
        }
    }
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
        let send = move |In(value): In<Option<T>>, ui: Res<UiHandle>| {
            if let Some(value) = value {
                send_to_view(&ui, view.clone(), handle, apply, value);
            }
        };
        let slot = cx.weak_entity();
        cx.global::<BevyHandle>()
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
        cx.global::<BevyHandle>()
            .update(move |world| match world.run_system_once(system) {
                Ok(value) => send_to_view(world.resource::<UiHandle>(), view, handle, apply, value),
                Err(error) => warn_once!("DevTools read failed: {error}"),
            });
    });
}

/// Has the UI thread pass `value` to `apply` on `view`, then redraw the view.
fn send_to_view<V: 'static, T: Send + 'static>(
    ui: &UiHandle,
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

pub trait DevToolsAppExt {
    /// Adds a dock tab titled `title`, showing the view `new` creates. Tabs
    /// appear in the order they are added.
    fn add_devtools_tab<V: Render>(
        &mut self,
        title: &'static str,
        new: fn(&mut Window, &mut Context<V>) -> V,
    ) -> &mut Self;
}

impl DevToolsAppExt for bevy::app::App {
    fn add_devtools_tab<V: Render>(
        &mut self,
        title: &'static str,
        new: fn(&mut Window, &mut Context<V>) -> V,
    ) -> &mut Self {
        let build = move |window: &mut Window, cx: &mut App| -> AnyView {
            cx.new(|cx| new(window, cx)).into()
        };
        self.world_mut().get_resource_or_init::<Tabs>().0.push(Tab {
            title,
            build: Box::new(build),
        });
        self
    }
}

struct Tab {
    title: &'static str,
    build: Box<dyn FnOnce(&mut Window, &mut App) -> AnyView + Send>,
}

/// The tabs added while the app is built, handed to the dock at startup.
#[derive(Resource, Default)]
struct Tabs(Vec<Tab>);

// `Tab::build` is only called on the UI thread; the resource only moves it.
unsafe impl Sync for Tabs {}

/// Hands the tabs to the dock once the app starts.
pub(crate) struct TabsPlugin;

impl bevy::app::Plugin for TabsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.init_resource::<Tabs>().add_systems(Startup, send_tabs);
    }
}

fn send_tabs(mut tabs: ResMut<Tabs>, ui: Option<Res<UiHandle>>) {
    let tabs = std::mem::take(&mut tabs.0);
    let Some(ui) = ui else {
        return;
    };
    ui.update(move |cx| {
        let DevToolsHandle { view, window } = cx.global::<DevToolsHandle>().clone();
        window
            .update(cx, |_, window, cx| {
                view.update(cx, |devtools, cx| devtools.add_tabs(tabs, window, cx))
            })
            .ok();
    });
}
