//! DevTools: a Chrome-DevTools-style dock that hosts debug panels as tabs.
//!
//! A panel registers with [`DevToolsAppExt::add_devtools_tab`], passing a
//! system that draws the tab's contents. The system receives the area below
//! the tab strip as its input (`InMut<egui::Ui>`), lays it out and scrolls it
//! as it needs, and otherwise takes ordinary system parameters:
//!
//! ```ignore
//! app.add_devtools_tab("Physics", physics_ui);
//!
//! fn physics_ui(InMut(ui): InMut<egui::Ui>, mut gravity: ResMut<Gravity>) {
//!     ui.add(egui::Slider::new(&mut gravity.0.y, -20.0..=0.0).text("Gravity"));
//! }
//! ```
//!
//! Only the selected tab's system runs, and only while DevTools is open. F12,
//! or ⌘⌥I as in Chrome on macOS, toggles DevTools.

use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use dust_profiler::GpuProfiler;
use pumicite_egui::{
    EguiContext, EguiGlobalSettings, EguiPrimaryContextPass, PrimaryEguiContext, egui,
};

use crate::{PerformancePanel, PlayerOverlay};

/// Adds the DevTools dock and its F12 / ⌘⌥I toggle.
///
/// Like the other plugins in this crate, it draws into the primary egui
/// context set up by the renderer plugins and does not add `EguiPlugin` itself.
#[derive(Default)]
pub struct DevToolsPlugin;

impl Plugin for DevToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DevTools>();
        // Keys typed into DevTools, such as into the console's prompt, and
        // clicks on it shouldn't also reach the game: bevy_egui clears them
        // from `ButtonInput` while egui wants them.
        app.world_mut()
            .get_resource_or_init::<EguiGlobalSettings>()
            .enable_absorb_bevy_input_system = true;
        app.add_systems(Update, toggle_devtools);
        app.add_systems(EguiPrimaryContextPass, draw_devtools);
    }
}

/// Where the DevTools panel sits: Chrome's "Dock side" options.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DockSide {
    Left,
    Bottom,
    Right,
    /// A free-floating egui window, standing in for Chrome's "Undock into
    /// separate window".
    Floating,
}

/// DevTools state: whether it is shown, where, and which tab is selected.
///
/// The public fields let an app drive DevTools programmatically; the tab list
/// only grows through [`DevToolsAppExt::add_devtools_tab`].
#[derive(Resource)]
pub struct DevTools {
    /// Whether DevTools is shown. Toggled by F12 / ⌘⌥I and the ✕ button.
    pub open: bool,
    pub dock: DockSide,
    /// Index of the shown tab, in registration order.
    pub selected: usize,
    tabs: Vec<Tab>,
    /// The part of the window the docked panel leaves uncovered, where the
    /// rendered scene shows. Updated every frame by [`draw_devtools`].
    pub(crate) scene_rect: egui::Rect,
}

impl Default for DevTools {
    fn default() -> Self {
        Self {
            open: true,
            dock: DockSide::Right,
            selected: 0,
            tabs: Vec::new(),
            scene_rect: egui::Rect::ZERO,
        }
    }
}

struct Tab {
    title: &'static str,
    system: SystemId<InMut<'static, egui::Ui>>,
}

pub trait DevToolsAppExt {
    /// Adds a tab titled `title` whose contents are drawn by `system`. Tabs
    /// appear in the order they are added.
    ///
    /// Registering doesn't require [`DevToolsPlugin`]: without it the tab is
    /// recorded but never drawn.
    fn add_devtools_tab<M>(
        &mut self,
        title: &'static str,
        system: impl IntoSystem<InMut<'static, egui::Ui>, (), M> + 'static,
    ) -> &mut Self;
}

impl DevToolsAppExt for App {
    fn add_devtools_tab<M>(
        &mut self,
        title: &'static str,
        system: impl IntoSystem<InMut<'static, egui::Ui>, (), M> + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let system = world.register_system(system);
        world
            .get_resource_or_init::<DevTools>()
            .tabs
            .push(Tab { title, system });
        self
    }
}

fn toggle_devtools(keys: Res<ButtonInput<KeyCode>>, mut devtools: ResMut<DevTools>) {
    let command = keys.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]);
    let option = keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);
    if keys.just_pressed(KeyCode::F12) || (command && option && keys.just_pressed(KeyCode::KeyI)) {
        devtools.open = !devtools.open;
    }
}

/// Draws the dock. Exclusive because the selected tab's system runs in the
/// middle of it, with the `World` that system needs.
pub(crate) fn draw_devtools(world: &mut World) {
    let Ok(mut context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single_mut(world)
    else {
        return;
    };
    // `egui::Context` is a shared handle; cloning it lets the UI closures below
    // borrow `world` mutably.
    let ctx = context.get_mut().clone();

    world.resource_scope(|world, mut devtools: Mut<DevTools>| {
        let viewport = ctx.viewport_rect();
        devtools.scene_rect = viewport;
        if !devtools.open {
            return;
        }
        let side = devtools.dock;
        let contents = |ui: &mut egui::Ui| devtools_contents(ui, world, &mut devtools);
        // `pivot` is the corner of the window the panel is attached to.
        let (panel, pivot) = match side {
            DockSide::Left => (
                egui::Panel::left("devtools_left").default_size(480.0),
                egui::Align2::LEFT_TOP,
            ),
            DockSide::Right => (
                egui::Panel::right("devtools_right").default_size(480.0),
                egui::Align2::RIGHT_TOP,
            ),
            DockSide::Bottom => (
                egui::Panel::bottom("devtools_bottom").default_size(320.0),
                egui::Align2::LEFT_BOTTOM,
            ),
            DockSide::Floating => {
                egui::Window::new("DevTools")
                    .title_bar(false)
                    .default_size([640.0, 480.0])
                    .show(&ctx, contents);
                return;
            }
        };
        // egui counts the pointer as over UI (`is_pointer_over_egui`, which
        // bevy_egui reports in `EguiWantsInput`) only over an `Area`, not over
        // a panel laid out on a `Ui` of our own. So the panel is painted into
        // the layer of an `Area` that is attached at the same corner and kept
        // the panel's size. The panel is laid out on a window-sized `Ui`:
        // since egui 0.36, panels need a parent `Ui`. Its frame is opaque, so
        // the renderer skips tracing the pixels underneath it.
        let scene_rect = egui::Area::new(egui::Id::new("devtools_dock"))
            .pivot(pivot)
            .fixed_pos(pivot.pos_in_rect(&viewport))
            .show(&ctx, |area_ui| {
                let mut root = egui::Ui::new(
                    ctx.clone(),
                    "devtools_root".into(),
                    egui::UiBuilder::new()
                        .layer_id(area_ui.layer_id())
                        .max_rect(viewport),
                );
                let panel_rect = panel
                    .resizable(true)
                    .size_range(120.0..=f32::INFINITY)
                    .show(&mut root, contents)
                    .response
                    .rect;
                area_ui.set_min_size(panel_rect.size());
                root.available_rect_before_wrap()
            })
            .inner;
        devtools.scene_rect = scene_rect;
    });
}

/// Tab strip, then the selected tab's contents.
fn devtools_contents(ui: &mut egui::Ui, world: &mut World, devtools: &mut DevTools) {
    let tab_bar = ui.horizontal(|ui| {
        // Right-to-left first, so the controls claim their space and the tabs
        // get whatever width is left.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, paint_close)
                .on_hover_text("Close DevTools (F12)")
                .clicked()
            {
                devtools.open = false;
            }
            let menu = icon_button(ui, paint_kebab).on_hover_text("Customize DevTools");
            egui::Popup::menu(&menu).show(|ui| {
                ui.label("Dock side");
                ui.horizontal(|ui| {
                    for (side, label) in [
                        (DockSide::Floating, "Undock"),
                        (DockSide::Left, "Left"),
                        (DockSide::Bottom, "Bottom"),
                        (DockSide::Right, "Right"),
                    ] {
                        ui.selectable_value(&mut devtools.dock, side, label);
                    }
                });
                ui.separator();
                if let Some(mut panel) = world.get_resource_mut::<PerformancePanel>() {
                    ui.checkbox(&mut panel.open, "Performance overlay");
                }
                if let Some(mut overlay) = world.get_resource_mut::<PlayerOverlay>() {
                    ui.checkbox(&mut overlay.open, "Player overlay");
                }
                if let Some(mut profiler) = world.get_resource_mut::<GpuProfiler>() {
                    let supported = profiler.is_supported();
                    ui.add_enabled(
                        supported,
                        egui::Checkbox::new(&mut profiler.enabled, "GPU timings"),
                    )
                    .on_disabled_hover_text("Timestamp queries unsupported on this device.");
                }
            });
            // Frame rate stays in view whichever tab is selected, where Chrome
            // keeps its error count.
            if let Some(perf) = world.get_resource::<PerformancePanel>() {
                ui.label(
                    egui::RichText::new(format!("{:.0} FPS", perf.average_fps()))
                        .monospace()
                        .weak(),
                );
            }
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                tab_strip(ui, &devtools.tabs, &mut devtools.selected);
            });
        });
    });
    // A line right along the tab bar's bottom edge, where `ui.separator()`
    // would leave the item spacing and half its own height above its line.
    // Centering the line half its width below the edge keeps it off the tabs'
    // pixels.
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    ui.painter().hline(
        ui.max_rect().x_range(),
        tab_bar.response.rect.bottom() + stroke.width / 2.0,
        stroke,
    );

    let Some(tab) = devtools.tabs.get(devtools.selected) else {
        ui.weak("No debug panels registered.");
        return;
    };
    if let Err(err) = world.run_system_with(tab.system, ui) {
        ui.colored_label(ui.visuals().error_fg_color, err.to_string());
    }
}

const TAB_PADDING: egui::Vec2 = egui::vec2(10.0, 6.0);

/// Draws the tabs that fit into the width left in `ui`, in registration order,
/// and lists the rest under a "»" menu as Chrome does. The selected tab always
/// stays in the strip.
fn tab_strip(ui: &mut egui::Ui, tabs: &[Tab], selected: &mut usize) {
    ui.spacing_mut().item_spacing.x = 0.0;
    let galleys: Vec<_> = tabs
        .iter()
        .map(|tab| {
            egui::WidgetText::from(tab.title).into_galley(
                ui,
                Some(egui::TextWrapMode::Extend),
                f32::INFINITY,
                egui::TextStyle::Button,
            )
        })
        .collect();
    let widths: Vec<f32> = galleys
        .iter()
        .map(|galley| galley.size().x + 2.0 * TAB_PADDING.x)
        .collect();

    let available = ui.available_width();
    let mut fits = vec![true; tabs.len()];
    if widths.iter().sum::<f32>() > available {
        let mut used = ICON_SIZE.x + widths.get(*selected).copied().unwrap_or(0.0);
        for (i, &width) in widths.iter().enumerate() {
            if i != *selected {
                fits[i] = used + width <= available;
                if fits[i] {
                    used += width;
                }
            }
        }
    }

    for (i, galley) in galleys.into_iter().enumerate() {
        if fits[i] && tab_button(ui, galley, i == *selected).clicked() {
            *selected = i;
        }
    }
    if fits.contains(&false) {
        let more = icon_button(ui, paint_chevrons).on_hover_text("More tabs");
        egui::Popup::menu(&more).show(|ui| {
            for (i, tab) in tabs.iter().enumerate() {
                if !fits[i] && ui.button(tab.title).clicked() {
                    *selected = i;
                }
            }
        });
    }
}

/// A Chrome-style tab: plain text, highlighted and underlined in the accent
/// color when selected.
fn tab_button(
    ui: &mut egui::Ui,
    galley: std::sync::Arc<egui::Galley>,
    selected: bool,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(galley.size() + 2.0 * TAB_PADDING, egui::Sense::click());
    let visuals = ui.visuals();
    if response.hovered() {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::ZERO,
            visuals.widgets.hovered.weak_bg_fill,
        );
    }
    let color = if selected {
        visuals.selection.stroke.color
    } else if response.hovered() {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    ui.painter().galley(rect.min + TAB_PADDING, galley, color);
    if selected {
        let underline =
            egui::Rect::from_min_max(egui::pos2(rect.left(), rect.bottom() - 2.0), rect.max);
        ui.painter()
            .rect_filled(underline, egui::CornerRadius::ZERO, color);
    }
    response
}

const ICON_SIZE: egui::Vec2 = egui::vec2(24.0, 24.0);

/// A square button whose icon is drawn by `paint` around the button's center.
/// The icons are painted rather than typed because egui's default fonts don't
/// reliably carry glyphs like ⋮ and ✕.
pub(crate) fn icon_button(
    ui: &mut egui::Ui,
    paint: fn(&egui::Painter, egui::Pos2, egui::Stroke),
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(ICON_SIZE, egui::Sense::click());
    let visuals = ui.style().interact(&response);
    if response.hovered() {
        ui.painter()
            .rect_filled(rect, egui::CornerRadius::same(4), visuals.weak_bg_fill);
    }
    paint(
        ui.painter(),
        rect.center(),
        egui::Stroke::new(1.5, visuals.fg_stroke.color),
    );
    response
}

fn paint_close(painter: &egui::Painter, c: egui::Pos2, stroke: egui::Stroke) {
    let r = 4.5;
    painter.line_segment([c + egui::vec2(-r, -r), c + egui::vec2(r, r)], stroke);
    painter.line_segment([c + egui::vec2(-r, r), c + egui::vec2(r, -r)], stroke);
}

pub(crate) fn paint_kebab(painter: &egui::Painter, c: egui::Pos2, stroke: egui::Stroke) {
    for dy in [-4.5, 0.0, 4.5] {
        painter.circle_filled(c + egui::vec2(0.0, dy), 1.5, stroke.color);
    }
}

fn paint_chevrons(painter: &egui::Painter, c: egui::Pos2, stroke: egui::Stroke) {
    for dx in [-2.5, 2.5] {
        let tip = c + egui::vec2(dx + 2.0, 0.0);
        painter.line_segment([tip + egui::vec2(-4.0, -4.0), tip], stroke);
        painter.line_segment([tip + egui::vec2(-4.0, 4.0), tip], stroke);
    }
}
