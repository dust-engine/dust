//! The tab's view on the UI thread: a toolbar with the schedule picker and a
//! filter, the graph, and beside or below it the inspector, all drawn from
//! the snapshots [`snapshot`] reads.
//!
//! The graph pans when dragged or scrolled, and zooms when pinched or
//! scrolled with ⌘ held. Selecting a system colors what it waits for blue
//! and what waits for it amber, and dims the rest. Selecting a set, by the
//! name at the top of its box, dims the systems not in it.

use std::cell::Cell;
use std::rc::Rc;

use bevy::ecs::schedule::InternedScheduleLabel;
use bevy::prelude::World;
use gpui::{
    AnyElement, App, AppContext as _, Bounds, ClickEvent, ClipboardItem, Context, DragMoveEvent,
    ElementId, Entity, FocusHandle, Font, FontWeight, Hsla, InteractiveElement as _, IntoElement,
    KeyBinding, ParentElement as _, PathBuilder, PinchEvent, Pixels, Point, Render,
    ScrollWheelEvent, SharedString, Size, StatefulInteractiveElement as _, Styled as _,
    Subscription, TextRun, Window, actions, canvas, div, point, prelude::FluentBuilder as _, px,
    rgb, size,
};
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::input::{Input, InputEvent, InputState};
use gpui_component::menu::{ContextMenuExt as _, DropdownMenu as _, PopupMenuItem};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::tooltip::Tooltip;
use gpui_component::{
    ActiveTheme as _, IconName, Sizable as _, Theme, h_flex, h_resizable, resizable_panel, v_flex,
    v_resizable,
};

use super::graph::{
    BuildState, GraphSnapshot, Neighbor, Reason, SetInfo, SystemInfo, SystemKind, Target,
};
use super::layout::{self, Layout, Tree};
use super::{Group, ScheduleEntry, Snapshot, snapshot};
use crate::{fetch, read};

actions!(
    dust_devtools_schedules,
    [FocusFilter, ClearSelection, ZoomIn, ZoomOut, ZoomToFit]
);

/// Height of a system in the graph, at 100%.
const NODE_HEIGHT: f32 = 22.0;

/// Height of a sync point, drawn smaller than a system.
const SYNC_HEIGHT: f32 = 16.0;

/// Room on each side of a system's name.
const NODE_PADDING: f32 = 8.0;

/// Width of the bar in the color of a system's crate, on its left.
const STRIPE: f32 = 3.0;

/// Size of a system's name, at 100%.
const LABEL_SIZE: f32 = 12.0;

/// Size of a set's name, at the top of its box, at 100%.
const SET_LABEL_SIZE: f32 = 11.0;

const MIN_ZOOM: f32 = 0.05;
const MAX_ZOOM: f32 = 2.0;

/// Below this zoom, systems are drawn without their names, too small to
/// read.
const LABEL_ZOOM: f32 = 0.35;

/// The zooms a system's name is measured at, to size the system to fit it
/// at each: the system font spaces its letters wider the smaller it is.
const MEASURED_ZOOMS: [f32; 4] = [LABEL_ZOOM, 0.5, 0.75, 1.0];

/// A schedule is first shown at least this large: whole if it fits, from
/// its start if it doesn't.
const FIRST_ZOOM: f32 = 0.6;

/// Room around the drawing when it's fit to the view.
const MARGIN: f32 = 24.0;

/// A click that moved further than this was a drag, panning the graph.
const CLICK_SLOP: f64 = 4.0;

/// How many edges go in one path: GPUI's paths index their vertices with 16
/// bits.
const EDGES_PER_PATH: usize = 96;

/// The colors of the crates, the crate with the most systems first.
const CRATE_COLORS: [u32; 10] = [
    0x4c9aff, // blue
    0x42c98a, // green
    0xffc84c, // amber
    0xff7a59, // orange
    0xb284ff, // purple
    0x4cd6d6, // teal
    0xff6fb5, // pink
    0xa3d977, // lime
    0xe0a370, // tan
    0x8f9bff, // periwinkle
];

pub(crate) struct SchedulesView {
    schedules: Vec<ScheduleEntry>,
    /// The schedule last picked, shown once its graph arrives; `None` for
    /// `Update`.
    picked: Option<InternedScheduleLabel>,
    graph: Option<Graph>,
    /// Index of the selected system in the graph's snapshot.
    selected: Option<usize>,
    /// Index of the selected set in the graph's snapshot, while no system
    /// is selected.
    selected_set: Option<usize>,
    hovered: Option<usize>,
    filter: Entity<InputState>,
    /// Which systems match the filter; `None` while it's empty.
    matches: Option<Vec<bool>>,
    camera: Camera,
    /// The graph's bounds when it was last laid out, in the window.
    viewport: Rc<Cell<Bounds<Pixels>>>,
    /// Whether the camera is to be set for a schedule just shown, once the
    /// graph's bounds are known.
    frame_pending: bool,
    /// The graph takes the keyboard focus when clicked, for Escape to clear
    /// the selection.
    graph_focus: FocusHandle,
    /// Whether the tab is wider than it's tall, which puts the inspector
    /// beside the graph rather than below it.
    wide: bool,
    _subscriptions: Vec<Subscription>,
}

/// What part of the drawing the graph shows.
#[derive(Clone, Copy)]
struct Camera {
    zoom: f32,
    /// Where the drawing's origin is, from the graph's top left corner.
    offset: Point<Pixels>,
}

/// A schedule's graph and its drawing.
struct Graph {
    snapshot: GraphSnapshot,
    layout: Rc<Layout>,
    /// Each system's size at 100%.
    sizes: Vec<Size<f32>>,
    /// The crates' names, colors, and numbers of systems, the crate with the
    /// most first. Sync points belong to none.
    crates: Vec<(SharedString, Hsla, usize)>,
    /// Each system's crate, as an index into `crates`.
    crate_of: Vec<Option<usize>>,
    /// The edges into and out of each system, as indices into the snapshot's
    /// edges.
    incoming: Vec<Vec<usize>>,
    outgoing: Vec<Vec<usize>>,
    /// The ambiguities of each system, as indices into the snapshot's.
    ambiguous: Vec<Vec<usize>>,
    /// The set each system and set is drawn in.
    tree: Tree,
    /// The sets drawn with a box, outermost first.
    boxes: Vec<usize>,
    /// How each system relates to the selected system or set; empty while
    /// none is selected.
    relations: Vec<Relation>,
}

/// How a system relates to the selected system.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Relation {
    Unrelated,
    Selected,
    /// Runs before it, directly or through other systems.
    Before,
    /// Runs after it, directly or through other systems.
    After,
    /// In the selected set.
    InSet,
}

/// How an edge is drawn, in the order the styles are painted: the last on
/// top. A line drawing several edges is drawn as the last of theirs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum EdgeStyle {
    Dimmed,
    Normal,
    Hovered,
    /// On a path into the selected system.
    Before,
    /// On a path out of the selected system.
    After,
    /// Into the selected system.
    DirectBefore,
    /// Out of the selected system.
    DirectAfter,
}

impl EdgeStyle {
    const ALL: [EdgeStyle; 7] = [
        EdgeStyle::Dimmed,
        EdgeStyle::Normal,
        EdgeStyle::Hovered,
        EdgeStyle::Before,
        EdgeStyle::After,
        EdgeStyle::DirectBefore,
        EdgeStyle::DirectAfter,
    ];

    fn color(self, theme: &Theme) -> Hsla {
        match self {
            EdgeStyle::Dimmed => theme.muted_foreground.opacity(0.12),
            EdgeStyle::Normal => theme.muted_foreground.opacity(0.5),
            EdgeStyle::Hovered => theme.foreground.opacity(0.85),
            EdgeStyle::Before => theme.info.opacity(0.55),
            EdgeStyle::After => theme.warning.opacity(0.55),
            EdgeStyle::DirectBefore => theme.info,
            EdgeStyle::DirectAfter => theme.warning,
        }
    }

    /// The line's width, which stays the same at every zoom.
    fn width(self) -> f32 {
        match self {
            EdgeStyle::Dimmed | EdgeStyle::Normal => 1.0,
            EdgeStyle::Hovered | EdgeStyle::Before | EdgeStyle::After => 1.5,
            EdgeStyle::DirectBefore | EdgeStyle::DirectAfter => 2.0,
        }
    }
}

impl SchedulesView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.bind_keys([
            KeyBinding::new("cmd-f", FocusFilter, Some("Schedules")),
            KeyBinding::new("cmd-=", ZoomIn, Some("Schedules")),
            KeyBinding::new("cmd--", ZoomOut, Some("Schedules")),
            KeyBinding::new("cmd-0", ZoomToFit, Some("Schedules")),
            KeyBinding::new("escape", ClearSelection, Some("ScheduleGraph")),
        ]);
        let filter = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Filter (⌘F)")
                .clean_on_escape()
        });
        let subscriptions = vec![cx.subscribe_in(
            &filter,
            window,
            |this, _, event: &InputEvent, _, cx| match event {
                InputEvent::Change => {
                    this.update_matches(cx);
                    cx.notify();
                }
                InputEvent::PressEnter { .. } => this.select_next_match(cx),
                _ => {}
            },
        )];
        SchedulesView {
            schedules: Vec::new(),
            picked: None,
            graph: None,
            selected: None,
            selected_set: None,
            hovered: None,
            filter,
            matches: None,
            camera: Camera {
                zoom: 1.0,
                offset: point(px(MARGIN), px(MARGIN)),
            },
            viewport: Rc::new(Cell::new(Bounds::default())),
            frame_pending: false,
            graph_focus: cx.focus_handle(),
            wide: false,
            _subscriptions: subscriptions,
        }
    }

    fn receive(&mut self, snapshot: Snapshot, window: &mut Window, cx: &mut Context<Self>) {
        self.schedules = snapshot.schedules;
        let Some(new) = snapshot.graph else {
            self.graph = None;
            self.selected = None;
            self.selected_set = None;
            return;
        };
        let old = self.graph.take();
        let same = old
            .as_ref()
            .is_some_and(|old| old.snapshot.label == new.label);
        // The selection outlives a refresh, by system or set.
        let selected = same
            .then(|| {
                let old = old.as_ref()?;
                let key = old.snapshot.systems.get(self.selected?)?.key;
                new.systems.iter().position(|system| system.key == key)
            })
            .flatten();
        let selected_set = same
            .then(|| {
                let old = old.as_ref()?;
                let key = old.snapshot.sets.get(self.selected_set?)?.key;
                new.sets.iter().position(|set| set.key == key)
            })
            .flatten();
        self.picked = Some(new.label);
        self.graph = Some(Graph::new(new, window, cx));
        self.hovered = None;
        self.update_matches(cx);
        self.select(selected, cx);
        if let Some(set) = selected_set {
            self.select_set(set, cx);
        }
        if !same {
            self.frame_pending = true;
        }
    }

    /// Shows `label`'s graph once it's read.
    fn pick(&mut self, label: InternedScheduleLabel, window: &Window, cx: &mut Context<Self>) {
        self.picked = Some(label);
        cx.notify();
        fetch(
            window,
            cx,
            move |world: &World| snapshot(world, Some(label)),
            |this, snapshot, window, cx| this.receive(snapshot, window, cx),
        );
    }

    /// Reads the schedules and the shown one's graph again.
    fn refresh(&mut self, window: &Window, cx: &mut Context<Self>) {
        let picked = self.picked;
        fetch(
            window,
            cx,
            move |world: &World| snapshot(world, picked),
            |this, snapshot, window, cx| this.receive(snapshot, window, cx),
        );
    }

    fn select(&mut self, system: Option<usize>, cx: &mut Context<Self>) {
        self.selected = system;
        self.selected_set = None;
        if let Some(graph) = &mut self.graph {
            graph.relations = match system {
                Some(system) => graph.relations_to(system),
                None => Vec::new(),
            };
        }
        cx.notify();
    }

    fn select_set(&mut self, set: usize, cx: &mut Context<Self>) {
        self.selected = None;
        self.selected_set = Some(set);
        if let Some(graph) = &mut self.graph {
            graph.relations = graph.members_of(set);
        }
        cx.notify();
    }

    /// Selects `system` and pans the graph to it if it's out of view.
    fn go_to(&mut self, system: usize, cx: &mut Context<Self>) {
        self.select(Some(system), cx);
        self.reveal(system);
    }

    /// Selects `set` and pans the graph to its box, if it has one and it's
    /// out of view.
    fn go_to_set(&mut self, set: usize, cx: &mut Context<Self>) {
        self.select_set(set, cx);
        if let Some(rect) = self
            .graph
            .as_ref()
            .and_then(|graph| self.set_rect(graph, set))
        {
            self.reveal_rect(rect);
        }
    }

    fn update_matches(&mut self, cx: &App) {
        let text = self.filter.read(cx).value().to_string();
        let words: Vec<&str> = text.split_whitespace().collect();
        self.matches = match &self.graph {
            Some(graph) if !words.is_empty() => Some(
                graph
                    .snapshot
                    .systems
                    .iter()
                    .map(|system| {
                        words
                            .iter()
                            .all(|word| system_matches(system, &graph.snapshot.sets, word))
                    })
                    .collect(),
            ),
            _ => None,
        };
    }

    /// Selects the first system after the selected one that matches the
    /// filter, and pans to it.
    fn select_next_match(&mut self, cx: &mut Context<Self>) {
        let Some(matches) = &self.matches else {
            return;
        };
        let start = self.selected.map_or(0, |selected| selected + 1);
        let count = matches.len();
        if let Some(next) = (0..count)
            .map(|i| (start + i) % count)
            .find(|&i| matches[i])
        {
            self.go_to(next, cx);
        }
    }

    /// The camera for a schedule just shown: the whole graph if it fits at
    /// [`FIRST_ZOOM`] or larger, else its top left corner at that zoom.
    fn first_camera(&self) -> Camera {
        let fit = self.fit_camera();
        if fit.zoom >= FIRST_ZOOM {
            return fit;
        }
        Camera {
            zoom: FIRST_ZOOM,
            offset: point(px(MARGIN), px(MARGIN)),
        }
    }

    /// The camera showing the whole graph, centered, no larger than 100%.
    fn fit_camera(&self) -> Camera {
        let view = self.viewport.get().size;
        let (width, height) = (f32::from(view.width), f32::from(view.height));
        let content = self
            .graph
            .as_ref()
            .map_or(size(0.0, 0.0), |graph| graph.layout.size);
        let zoom = ((width - 2.0 * MARGIN) / content.width.max(1.0))
            .min((height - 2.0 * MARGIN) / content.height.max(1.0))
            .clamp(MIN_ZOOM, 1.0);
        Camera {
            zoom,
            offset: point(
                px((width - content.width * zoom) / 2.0),
                px((height - content.height * zoom) / 2.0),
            ),
        }
    }

    /// Zooms by `factor`, keeping the point `at`, in the window, where it is.
    fn zoom_at(&mut self, factor: f32, at: Point<Pixels>) {
        let local = at - self.viewport.get().origin;
        let zoom = (self.camera.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let scale = zoom / self.camera.zoom;
        let offset = self.camera.offset;
        self.camera = Camera {
            zoom,
            offset: point(
                local.x - (local.x - offset.x) * scale,
                local.y - (local.y - offset.y) * scale,
            ),
        };
    }

    /// Zooms by `factor` about the middle of the graph.
    fn zoom_by(&mut self, factor: f32, cx: &mut Context<Self>) {
        self.zoom_at(factor, self.viewport.get().center());
        cx.notify();
    }

    /// Pans the graph to center `system`, unless it's in view already.
    fn reveal(&mut self, system: usize) {
        let Some(graph) = &self.graph else {
            return;
        };
        let rect = self.screen_rect(graph, system);
        self.reveal_rect(rect);
    }

    /// Pans the graph to center `rect`, from the graph's top left corner,
    /// unless it's in view already; or to show its top left corner, if it's
    /// larger than the view.
    fn reveal_rect(&mut self, rect: Bounds<Pixels>) {
        let view = self.viewport.get().size;
        let in_view = rect.origin.x >= px(0.)
            && rect.origin.y >= px(0.)
            && rect.right() <= view.width
            && rect.bottom() <= view.height;
        if in_view {
            return;
        }
        let shift = if rect.size.width <= view.width && rect.size.height <= view.height {
            let center = rect.center();
            point(view.width / 2.0 - center.x, view.height / 2.0 - center.y)
        } else {
            point(px(MARGIN) - rect.origin.x, px(MARGIN) - rect.origin.y)
        };
        self.camera.offset += shift;
    }

    /// Where `system` is drawn, from the graph's top left corner.
    fn screen_rect(&self, graph: &Graph, system: usize) -> Bounds<Pixels> {
        self.to_screen(graph.layout.nodes[system], graph.sizes[system])
    }

    /// Where `set`'s box is drawn, if it has one, from the graph's top left
    /// corner.
    fn set_rect(&self, graph: &Graph, set: usize) -> Option<Bounds<Pixels>> {
        let bounds = graph.layout.sets[set]?;
        Some(self.to_screen(bounds.origin, bounds.size))
    }

    /// Where the drawing's rectangle at `at`, `extent` large, is, from the
    /// graph's top left corner.
    fn to_screen(&self, at: Point<f32>, extent: Size<f32>) -> Bounds<Pixels> {
        let Camera { zoom, offset } = self.camera;
        Bounds {
            origin: point(offset.x + px(at.x * zoom), offset.y + px(at.y * zoom)),
            size: size(px(extent.width * zoom), px(extent.height * zoom)),
        }
    }

    fn scroll(&mut self, event: &ScrollWheelEvent, _: &mut Window, cx: &mut Context<Self>) {
        let delta = event.delta.pixel_delta(px(20.));
        if event.modifiers.platform || event.modifiers.control {
            self.zoom_at((f32::from(delta.y) * 0.005).exp(), event.position);
        } else {
            self.camera.offset += delta;
        }
        cx.stop_propagation();
        cx.notify();
    }

    fn pinch(&mut self, event: &PinchEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.zoom_at(1.0 + event.delta, event.position);
        cx.notify();
    }

    fn pan(&mut self, event: &DragMoveEvent<Pan>, _: &mut Window, cx: &mut Context<Self>) {
        let pan = event.drag(cx);
        let position = event.event.position;
        // From where the drag started moving, rather than where the mouse
        // went down, a few pixels back.
        let origin = match pan.origin.get() {
            Some(origin) => origin,
            None => {
                pan.origin.set(Some(position));
                position
            }
        };
        self.camera.offset = pan.from + (position - origin);
        cx.notify();
    }

    /// The picker, the filter, how many systems match it, and the zoom and
    /// refresh buttons.
    fn toolbar(&self, cx: &Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let count = self.graph.as_ref().map(|graph| {
            let total = graph.snapshot.systems.len();
            match &self.matches {
                Some(matches) => {
                    let matched = matches.iter().filter(|&&matched| matched).count();
                    format!("{matched} of {total}")
                }
                None if total == 1 => "1 system".to_owned(),
                None => format!("{total} systems"),
            }
        });
        let zoom = format!("{:.0}%", self.camera.zoom * 100.0);
        h_flex()
            .flex_shrink_0()
            .gap_1()
            .p_1()
            .border_b_1()
            .border_color(theme.border)
            .child(self.picker(cx))
            .child(
                div()
                    .id("schedules-filter")
                    .flex_1()
                    .min_w(px(80.))
                    .tooltip(|window, cx| {
                        Tooltip::new(
                            "Shows the systems whose name, path, crate or sets contain each \
                             word. Enter selects the next.",
                        )
                        .build(window, cx)
                    })
                    .child(Input::new(&self.filter).xsmall().cleanable(true)),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .children(count),
            )
            .child(
                Button::new("schedules-zoom-out")
                    .icon(IconName::Minus)
                    .ghost()
                    .xsmall()
                    .tooltip("Zoom out (⌘-)")
                    .on_click(cx.listener(|this, _, _, cx| this.zoom_by(0.8, cx))),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .w(px(36.))
                    .text_center()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(zoom),
            )
            .child(
                Button::new("schedules-zoom-in")
                    .icon(IconName::Plus)
                    .ghost()
                    .xsmall()
                    .tooltip("Zoom in (⌘=)")
                    .on_click(cx.listener(|this, _, _, cx| this.zoom_by(1.25, cx))),
            )
            .child(
                Button::new("schedules-fit")
                    .icon(IconName::Maximize)
                    .ghost()
                    .xsmall()
                    .tooltip("Fit to view (⌘0)")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.camera = this.fit_camera();
                        cx.notify();
                    })),
            )
            .child(
                Button::new("schedules-refresh")
                    .icon(IconName::RefreshCw)
                    .ghost()
                    .xsmall()
                    .tooltip("Read the schedules again")
                    .on_click(cx.listener(|this, _, window, cx| this.refresh(window, cx))),
            )
    }

    /// The button showing the schedule shown, and its menu of the schedules,
    /// in the order they run.
    fn picker(&self, cx: &Context<Self>) -> impl IntoElement + use<> {
        let shown = self.graph.as_ref().map(|graph| graph.snapshot.label);
        let label = self
            .schedules
            .iter()
            .find(|entry| Some(entry.label) == self.picked.or(shown))
            .map_or(SharedString::new_static("Schedule"), |entry| {
                entry.name.clone()
            });
        let view = cx.weak_entity();
        Button::new("schedules-picker")
            .label(label)
            .ghost()
            .xsmall()
            .dropdown_caret(true)
            .tooltip("Schedule")
            .dropdown_menu(move |mut menu, _, cx| {
                let Some(this) = view.upgrade() else {
                    return menu;
                };
                let this = this.read(cx);
                let muted = cx.theme().muted_foreground;
                let mut group: Option<Group> = None;
                for entry in &this.schedules {
                    if group != Some(entry.group) {
                        if group.is_some() {
                            menu = menu.separator();
                        }
                        menu = menu.label(entry.group.title());
                        group = Some(entry.group);
                    }
                    let (label, name) = (entry.label, entry.name.clone());
                    let count: SharedString = match entry.systems {
                        Some(count) => count.to_string().into(),
                        None => "running".into(),
                    };
                    let view = view.clone();
                    menu = menu.item(
                        PopupMenuItem::element(move |_, _| {
                            h_flex()
                                .w_full()
                                .gap_4()
                                .justify_between()
                                .child(name.clone())
                                .child(div().text_color(muted).child(count.clone()))
                        })
                        .checked(Some(entry.label) == this.picked)
                        .disabled(entry.systems.is_none())
                        .on_click(move |_, window, cx| {
                            view.update(cx, |this, cx| this.pick(label, window, cx))
                                .ok();
                        }),
                    );
                }
                if this.schedules.iter().any(|entry| entry.systems.is_none()) {
                    menu = menu
                        .separator()
                        .label("DevTools reads the world in Last, while Main runs it.");
                }
                menu.max_h(px(480.)).scrollable(true)
            })
    }

    /// The graph: the edges, the systems in view over them, and the heading
    /// of the unordered systems; with a note above it on what Bevy hasn't
    /// built.
    fn graph_panel(&self, cx: &Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let Some(graph) = &self.graph else {
            let text = if self.schedules.is_empty() {
                "Reading the schedules…"
            } else {
                "No schedule to show."
            };
            return message(text, theme);
        };
        let snapshot = &graph.snapshot;
        let note = match (&snapshot.error, snapshot.state) {
            (Some(error), _) => Some((error.clone(), theme.danger)),
            (None, BuildState::NotBuilt) => Some((
                "Not built: Bevy builds a schedule when it first runs. Until then, the systems' \
                 access and the sync points Bevy adds are unknown."
                    .into(),
                theme.warning,
            )),
            (None, BuildState::Changed) => Some((
                "Changed since Bevy built it: it's rebuilt when it next runs.".into(),
                theme.warning,
            )),
            (None, BuildState::Built) => None,
        };
        let note = note.map(|(text, color)| {
            div()
                .flex_shrink_0()
                .px_2()
                .py_1()
                .text_xs()
                .text_color(color)
                .border_b_1()
                .border_color(theme.border)
                .child(text)
        });

        let view = cx.weak_entity();
        let viewport = self.viewport.clone();
        let edges = EdgePaint {
            layout: graph.layout.clone(),
            styles: self.line_styles(graph),
            colors: EdgeStyle::ALL.map(|style| style.color(theme)),
            camera: self.camera,
            ambiguities: self.ambiguity_lines(graph),
            danger: theme.danger,
        };
        let edges = canvas(
            move |bounds, _, cx| {
                // Known only once laid out, so the first frame of the tab
                // draws again with them.
                let old = viewport.replace(bounds);
                if old.size.width <= px(0.) && bounds.size.width > px(0.) {
                    let view = view.clone();
                    cx.defer(move |cx| {
                        view.update(cx, |_, cx| cx.notify()).ok();
                    });
                }
            },
            move |bounds, _, window, _| edges.paint(bounds, window),
        )
        .absolute()
        .size_full();

        let view_size = self.viewport.get().size;
        // Before the first layout the size is unknown, and all are drawn.
        let culled = |rect: Bounds<Pixels>| {
            view_size.width > px(0.)
                && (rect.right() < px(0.)
                    || rect.bottom() < px(0.)
                    || rect.origin.x > view_size.width
                    || rect.origin.y > view_size.height)
        };
        let boxes: Vec<AnyElement> = graph
            .boxes
            .iter()
            .filter_map(|&set| {
                let rect = self.set_rect(graph, set)?;
                (!culled(rect)).then(|| self.set_box(graph, set, rect, cx))
            })
            .collect();
        let mut nodes = Vec::new();
        for system in 0..snapshot.systems.len() {
            let rect = self.screen_rect(graph, system);
            if !culled(rect) {
                nodes.push(self.node(graph, system, rect, cx));
            }
        }
        let zoom = self.camera.zoom;
        let heading = graph
            .layout
            .grid
            .as_ref()
            .filter(|_| zoom >= LABEL_ZOOM)
            .map(|grid| {
                let offset = self.camera.offset;
                div()
                    .absolute()
                    .left(offset.x + px(grid.at.x * zoom))
                    .top(offset.y + px(grid.at.y * zoom))
                    .h(px(layout::GRID_HEADING * zoom))
                    .text_size(px(LABEL_SIZE * zoom))
                    .whitespace_nowrap()
                    .text_color(theme.muted_foreground)
                    .child(match grid.count {
                        1 => "1 system that nothing orders".to_owned(),
                        count => format!("{count} systems that nothing orders"),
                    })
            });
        let empty = snapshot.systems.is_empty().then(|| {
            div()
                .absolute()
                .p_2()
                .text_color(theme.muted_foreground)
                .child("No systems.")
        });

        let surface = div()
            .id("schedules-graph")
            .relative()
            .flex_1()
            .min_h_0()
            .overflow_hidden()
            .track_focus(&self.graph_focus)
            .key_context("ScheduleGraph")
            .on_action(cx.listener(|this, _: &ClearSelection, _, cx| this.select(None, cx)))
            .on_scroll_wheel(cx.listener(Self::scroll))
            .on_pinch(cx.listener(Self::pinch))
            .on_drag(
                Pan {
                    from: self.camera.offset,
                    origin: Cell::new(None),
                },
                |_, _, _, cx| cx.new(|_| PanPreview),
            )
            .on_drag_move(cx.listener(Self::pan))
            .on_click(cx.listener(|this, event: &ClickEvent, window, cx| {
                this.graph_focus.focus(window, cx);
                if !moved(event) {
                    this.select(None, cx);
                }
            }))
            .children(boxes)
            .child(edges)
            .children(nodes)
            .children(heading)
            .children(empty);
        v_flex()
            .size_full()
            .children(note)
            .child(surface)
            .into_any_element()
    }

    /// A set's box, drawn at `rect`, under the systems in it, with the set's
    /// name at its top, which selects the set when clicked.
    fn set_box(
        &self,
        graph: &Graph,
        set: usize,
        rect: Bounds<Pixels>,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let zoom = self.camera.zoom;
        let selected = self.selected_set == Some(set);
        let foreground = theme.foreground;
        let header = div()
            .id(("schedules-set", set))
            .h(px(layout::SET_HEADER * zoom))
            .flex()
            .items_center()
            .px(px(layout::SET_PADDING * zoom))
            .overflow_hidden()
            .cursor_pointer()
            .text_size(px(SET_LABEL_SIZE * zoom))
            .text_color(if selected {
                foreground
            } else {
                theme.muted_foreground
            })
            .hover(move |header| header.text_color(foreground))
            .when(zoom >= LABEL_ZOOM, |header| {
                header.child(
                    div()
                        .truncate()
                        .child(graph.snapshot.sets[set].name.clone()),
                )
            })
            .on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
                cx.stop_propagation();
                this.graph_focus.focus(window, cx);
                if !moved(event) {
                    this.select_set(set, cx);
                }
            }))
            .tooltip(set_tooltip(graph, set))
            .context_menu({
                let name = graph.snapshot.sets[set].name.clone();
                move |menu, _, _| {
                    let name = name.clone();
                    menu.item(PopupMenuItem::new("Copy name").on_click(move |_, _, cx| {
                        cx.write_to_clipboard(ClipboardItem::new_string(name.to_string()))
                    }))
                }
            });
        div()
            .absolute()
            .left(rect.origin.x)
            .top(rect.origin.y)
            .w(rect.size.width)
            .h(rect.size.height)
            .rounded(px(6.0 * zoom))
            .border_1()
            .border_color(if selected { theme.ring } else { theme.border })
            .when(selected, |set_box| set_box.border_2())
            .bg(theme.secondary.opacity(0.3))
            .child(header)
            .into_any_element()
    }

    /// A system of the graph, drawn at `rect`.
    fn node(
        &self,
        graph: &Graph,
        system: usize,
        rect: Bounds<Pixels>,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let info = &graph.snapshot.systems[system];
        let zoom = self.camera.zoom;
        let relation = graph
            .relations
            .get(system)
            .copied()
            .unwrap_or(Relation::Unrelated);
        let selected = self.selected == Some(system);
        let ambiguous = self.selected.is_some_and(|selected| {
            selected != system
                && graph.ambiguous[selected].iter().any(|&a| {
                    let ambiguity = &graph.snapshot.ambiguities[a];
                    ambiguity.a == system || ambiguity.b == system
                })
        });
        let matched = self.matches.as_ref().is_none_or(|matches| matches[system]);
        let focused = self.selected.is_some() || self.selected_set.is_some();
        let dimmed = !matched || (focused && relation == Relation::Unrelated && !ambiguous);
        let border = match relation {
            Relation::Selected => theme.ring,
            Relation::Before => theme.info,
            Relation::After => theme.warning,
            Relation::Unrelated if ambiguous => theme.danger,
            Relation::Unrelated | Relation::InSet => theme.border,
        };
        let sync = info.kind == SystemKind::SyncPoint;
        let background = if selected {
            theme.list_active
        } else if sync {
            theme.background
        } else if info.kind == SystemKind::Exclusive {
            theme.secondary.blend(theme.warning.opacity(0.15))
        } else {
            theme.secondary
        };
        let height = f32::from(rect.size.height);
        let mut node = div()
            .id(("schedules-node", system))
            .absolute()
            .left(rect.origin.x)
            .top(rect.origin.y)
            .w(rect.size.width)
            .h(rect.size.height)
            .flex()
            .items_center()
            .overflow_hidden()
            .border_1()
            .border_color(border)
            .bg(background)
            .cursor_pointer()
            .when(selected, |node| node.border_2())
            .when(ambiguous, |node| node.border_dashed())
            .when(dimmed, |node| node.opacity(0.3))
            .when(!selected, |node| {
                let hover = theme.secondary_hover;
                node.hover(move |node| node.bg(hover))
            });
        if sync {
            node = node
                .rounded(px(height / 2.0))
                .border_dashed()
                .justify_center()
                .text_color(theme.muted_foreground)
                .when(zoom >= LABEL_ZOOM, |node| {
                    node.text_size(px(LABEL_SIZE * 0.85 * zoom)).child("sync")
                });
        } else {
            let stripe = graph.crate_of[system]
                .map_or(theme.muted_foreground, |krate| graph.crates[krate].1);
            node = node
                .rounded(px(4.0 * zoom))
                .child(
                    div()
                        .w(px(STRIPE * zoom))
                        .h_full()
                        .flex_shrink_0()
                        .bg(stripe),
                )
                .when(zoom >= LABEL_ZOOM, |node| {
                    node.child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .pl(px(NODE_PADDING * zoom))
                            .text_size(px(LABEL_SIZE * zoom))
                            .line_height(px(height))
                            .truncate()
                            .child(info.name.clone()),
                    )
                });
        }
        let (tooltip, build_menu) = (system_tooltip(graph, system), system_menu(info));
        node.on_click(cx.listener(move |this, event: &ClickEvent, window, cx| {
            cx.stop_propagation();
            this.graph_focus.focus(window, cx);
            if moved(event) {
                return;
            }
            this.select(Some(system), cx);
        }))
        .on_hover(cx.listener(move |this, &hovered: &bool, _, cx| {
            if hovered {
                this.hovered = Some(system);
            } else if this.hovered == Some(system) {
                this.hovered = None;
            }
            cx.notify();
        }))
        .tooltip(tooltip)
        .context_menu(move |menu, _, _| build_menu(menu))
        .into_any_element()
    }

    /// How each line is drawn: as the most prominent of the edges it draws.
    fn line_styles(&self, graph: &Graph) -> Vec<EdgeStyle> {
        let relations = &graph.relations;
        let both = |a: usize, b: usize, relation: Relation| {
            relations[a] == relation && relations[b] == relation
        };
        let hovered = |edge: &super::graph::Edge| {
            self.hovered
                .is_some_and(|hovered| edge.before == hovered || edge.after == hovered)
        };
        let mut styles = vec![EdgeStyle::Dimmed; graph.layout.lines.len()];
        for (edge, lines) in graph.snapshot.edges.iter().zip(&graph.layout.edges) {
            let (a, b) = (edge.before, edge.after);
            let style = match self.selected {
                Some(selected) if b == selected => EdgeStyle::DirectBefore,
                Some(selected) if a == selected => EdgeStyle::DirectAfter,
                Some(_) if both(a, b, Relation::Before) => EdgeStyle::Before,
                Some(_) if both(a, b, Relation::After) => EdgeStyle::After,
                _ if hovered(edge) => EdgeStyle::Hovered,
                Some(_) => EdgeStyle::Dimmed,
                None if self.selected_set.is_some() && both(a, b, Relation::InSet) => {
                    EdgeStyle::Normal
                }
                None if self.selected_set.is_some() => EdgeStyle::Dimmed,
                None => match &self.matches {
                    Some(matches) if !matches[a] && !matches[b] => EdgeStyle::Dimmed,
                    _ => EdgeStyle::Normal,
                },
            };
            for &line in lines {
                styles[line] = styles[line].max(style);
            }
        }
        styles
    }

    /// Lines from the selected system to the systems it's ambiguous with.
    fn ambiguity_lines(&self, graph: &Graph) -> Vec<(Point<f32>, Point<f32>)> {
        let Some(selected) = self.selected else {
            return Vec::new();
        };
        let center = |system: usize| {
            let (at, node) = (graph.layout.nodes[system], graph.sizes[system]);
            point(at.x + node.width / 2.0, at.y + node.height / 2.0)
        };
        graph.ambiguous[selected]
            .iter()
            .map(|&a| {
                let ambiguity = &graph.snapshot.ambiguities[a];
                let other = if ambiguity.a == selected {
                    ambiguity.b
                } else {
                    ambiguity.a
                };
                (center(selected), center(other))
            })
            .collect()
    }

    /// The selected system's or set's details, or the schedule's when none
    /// is selected.
    fn inspector_panel(&self, cx: &Context<Self>) -> AnyElement {
        let Some(graph) = &self.graph else {
            return div().into_any_element();
        };
        let contents = match (self.selected, self.selected_set) {
            (Some(system), _) => self.system_details(graph, system, cx).into_any_element(),
            (None, Some(set)) => self.set_details(graph, set, cx).into_any_element(),
            (None, None) => self.summary(graph, cx).into_any_element(),
        };
        div()
            .size_full()
            .overflow_y_scrollbar()
            .child(contents)
            .into_any_element()
    }

    /// The schedule's counts, what Bevy hasn't built, its crates, a key to
    /// the drawing, and its ambiguities.
    fn summary(&self, graph: &Graph, cx: &Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let snapshot = &graph.snapshot;
        let systems = &snapshot.systems;
        let sync_points = systems
            .iter()
            .filter(|system| system.kind == SystemKind::SyncPoint)
            .count();
        let counts = [
            (systems.len() - sync_points, "system", "systems"),
            (snapshot.edges.len(), "ordering", "orderings"),
            (sync_points, "sync point", "sync points"),
            (snapshot.ambiguities.len(), "ambiguity", "ambiguities"),
        ]
        .map(|(count, one, many)| format!("{count} {}", if count == 1 { one } else { many }))
        .join(" · ");

        let crates = graph
            .crates
            .iter()
            .enumerate()
            .map(|(i, (name, color, count))| {
                // The path's start, so that `dust` doesn't match `dust_pbr`.
                let filter: SharedString = format!("{name}::").into();
                h_flex()
                    .id(("schedules-crate", i))
                    .gap_1p5()
                    .px_1()
                    .rounded_sm()
                    .cursor_pointer()
                    .hover(|row| row.bg(theme.list_hover))
                    .child(div().size(px(8.)).rounded_full().bg(*color))
                    .child(div().flex_1().child(name.clone()))
                    .child(
                        div()
                            .text_color(theme.muted_foreground)
                            .child(count.to_string()),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.filter
                            .update(cx, |input, cx| input.set_value(filter.clone(), window, cx));
                        this.update_matches(cx);
                        cx.notify();
                    }))
            });

        let key_row = |swatch: AnyElement, text: &'static str| {
            h_flex()
                .gap_2()
                .child(
                    div()
                        .w(px(28.))
                        .flex_shrink_0()
                        .flex()
                        .justify_center()
                        .child(swatch),
                )
                .child(div().text_color(theme.muted_foreground).child(text))
        };
        let swatch = |background: Hsla, border: Hsla, dashed: bool| {
            div()
                .w(px(24.))
                .h(px(12.))
                .rounded_sm()
                .border_1()
                .border_color(border)
                .bg(background)
                .when(dashed, |swatch| swatch.border_dashed().rounded_full())
                .into_any_element()
        };
        let line = |color: Hsla| div().w(px(24.)).h(px(2.)).bg(color).into_any_element();
        let set_swatch = div()
            .w(px(28.))
            .h(px(16.))
            .rounded(px(4.))
            .border_1()
            .border_color(theme.border)
            .bg(theme.secondary.opacity(0.3))
            .into_any_element();
        let key = v_flex()
            .gap_1()
            .child(key_row(
                swatch(theme.secondary, theme.border, false),
                "A system, its bar in the color of its crate",
            ))
            .child(key_row(set_swatch, "A set, around its systems"))
            .child(key_row(
                swatch(
                    theme.secondary.blend(theme.warning.opacity(0.15)),
                    theme.border,
                    false,
                ),
                "An exclusive system, which runs alone",
            ))
            .child(key_row(
                swatch(theme.background, theme.border, true),
                "A sync point, applying the commands before it",
            ))
            .child(key_row(line(theme.info), "Runs before the selected system"))
            .child(key_row(
                line(theme.warning),
                "Runs after the selected system",
            ));

        let sets = self.set_list(graph, cx);

        let ambiguities = (!snapshot.ambiguities.is_empty()).then(|| {
            let rows = snapshot
                .ambiguities
                .iter()
                .enumerate()
                .map(|(i, ambiguity)| {
                    v_flex()
                        .child(
                            h_flex()
                                .flex_wrap()
                                .gap_x_1()
                                .child(self.link(
                                    graph,
                                    ambiguity.a,
                                    ("schedules-ambiguity-a", i),
                                    cx,
                                ))
                                .child(div().text_color(theme.muted_foreground).child("and"))
                                .child(self.link(
                                    graph,
                                    ambiguity.b,
                                    ("schedules-ambiguity-b", i),
                                    cx,
                                )),
                        )
                        .child(conflicts(&ambiguity.components, theme))
                });
            section(
                format!("Ambiguities ({})", snapshot.ambiguities.len()),
                Some(
                    "Systems that conflict over what they access, with nothing ordering them: \
                     Bevy runs them in either order.",
                ),
                v_flex().gap_1().children(rows),
                theme,
            )
        });

        v_flex()
            .gap_3()
            .p_2()
            .child(
                v_flex()
                    .gap_0p5()
                    .child(
                        div()
                            .text_base()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(format!("{:?}", snapshot.label)),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(counts),
                    ),
            )
            .child(div().text_xs().text_color(theme.muted_foreground).child(
                "Each system is drawn after the systems it waits for, and each set as a box \
                 around its systems. Select one to see what orders it, and why.",
            ))
            .when(!graph.crates.is_empty(), |summary| {
                summary.child(section(
                    "Crates".to_owned(),
                    None,
                    v_flex().children(crates),
                    theme,
                ))
            })
            .child(section("Key".to_owned(), None, key, theme))
            .children(sets)
            .children(ambiguities)
    }

    /// The selected system's name, path and kind; its sets; the systems it
    /// runs after and before, with the orderings that order them; the
    /// systems it's ambiguous with; and what it accesses.
    fn system_details(
        &self,
        graph: &Graph,
        system: usize,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let info = &graph.snapshot.systems[system];
        let sync = info.kind == SystemKind::SyncPoint;
        let title = if sync {
            SharedString::new_static("Sync point")
        } else {
            info.name.clone()
        };
        let krate = graph.crate_of[system].map(|krate| {
            let (name, color, _) = &graph.crates[krate];
            h_flex()
                .flex_shrink_0()
                .gap_1()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(div().size(px(8.)).rounded_full().bg(*color))
                .child(name.clone())
        });
        let mut badges: Vec<&'static str> = Vec::new();
        match info.kind {
            SystemKind::Exclusive => badges.push("Exclusive"),
            SystemKind::SyncPoint => badges.push("ApplyDeferred"),
            SystemKind::System => {}
        }
        if info.deferred {
            badges.push("Commands");
        }
        if info.non_send {
            badges.push("Main thread");
        }
        let badges = h_flex()
            .flex_wrap()
            .gap_1()
            .children(badges.into_iter().map(|badge| chip(badge, theme)));

        let relations = &graph.relations;
        let before_count = relations.iter().filter(|&&r| r == Relation::Before).count();
        let after_count = relations.iter().filter(|&&r| r == Relation::After).count();
        let mut after: Vec<usize> = graph.incoming[system].clone();
        let mut before: Vec<usize> = graph.outgoing[system].clone();
        let edges = &graph.snapshot.edges;
        let name_of = |s: usize| graph.snapshot.systems[s].name.to_lowercase();
        after.sort_by_cached_key(|&e| name_of(edges[e].before));
        before.sort_by_cached_key(|&e| name_of(edges[e].after));
        let neighbors = |list: &[usize], incoming: bool, id: &'static str| {
            v_flex().gap_1().children(list.iter().map(|&e| {
                let edge = &edges[e];
                let other = if incoming { edge.before } else { edge.after };
                v_flex()
                    .child(self.link(graph, other, (id, e), cx))
                    .child(reasons(&edge.reasons, graph, edge, theme))
            }))
        };
        let runs_after = section(
            format!("Runs after ({})", after.len()),
            None,
            if after.is_empty() {
                weak("Nothing: it can run first.", theme).into_any_element()
            } else {
                neighbors(&after, true, "schedules-after").into_any_element()
            },
            theme,
        );
        let runs_before = section(
            format!("Runs before ({})", before.len()),
            None,
            if before.is_empty() {
                weak("Nothing waits for it.", theme).into_any_element()
            } else {
                neighbors(&before, false, "schedules-before").into_any_element()
            },
            theme,
        );
        let reach = weak(
            format!(
                "Waits for {before_count} {} in all, and {after_count} {} for it.",
                if before_count == 1 {
                    "system"
                } else {
                    "systems"
                },
                if after_count == 1 { "waits" } else { "wait" },
            ),
            theme,
        );

        let ambiguities = (!graph.ambiguous[system].is_empty()).then(|| {
            let rows = graph.ambiguous[system].iter().map(|&a| {
                let ambiguity = &graph.snapshot.ambiguities[a];
                let other = if ambiguity.a == system {
                    ambiguity.b
                } else {
                    ambiguity.a
                };
                v_flex()
                    .child(self.link(graph, other, ("schedules-ambiguous", a), cx))
                    .child(conflicts(&ambiguity.components, theme))
            });
            section(
                format!("Ambiguous with ({})", graph.ambiguous[system].len()),
                Some("Nothing orders them, and they conflict: Bevy runs them in either order."),
                v_flex().gap_1().children(rows),
                theme,
            )
        });

        let access = match &info.access {
            None => weak("Unknown until Bevy builds the schedule.", theme).into_any_element(),
            Some(access) => {
                let list = |label: &'static str, names: &[SharedString]| {
                    (!names.is_empty()).then(|| {
                        h_flex()
                            .flex_wrap()
                            .gap_1()
                            .child(div().text_color(theme.muted_foreground).child(label))
                            .children(names.iter().map(|name| code_chip(name.clone(), theme)))
                    })
                };
                let nothing =
                    access.all.is_none() && access.reads.is_empty() && access.writes.is_empty();
                v_flex()
                    .gap_1()
                    .children(access.all.map(|all| div().child(all)))
                    .children(list("Writes", &access.writes))
                    .children(list("Reads", &access.reads))
                    .when(nothing, |access| {
                        access.child(weak("Nothing in the world.", theme))
                    })
                    .into_any_element()
            }
        };

        v_flex()
            .gap_3()
            .p_2()
            .child(
                v_flex()
                    .gap_0p5()
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_base()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child(title),
                            )
                            .children(krate),
                    )
                    .child(
                        div()
                            .text_xs()
                            .font_family(theme.mono_font_family.clone())
                            .text_color(theme.muted_foreground)
                            .child(info.path.clone()),
                    ),
            )
            .child(badges)
            .when(sync, |details| {
                details.child(weak(
                    "Applies the commands of the systems before it, so that the systems after \
                     it see what they did.",
                    theme,
                ))
            })
            .when(!info.sets.is_empty(), |details| {
                details.child(section(
                    "Sets".to_owned(),
                    None,
                    h_flex().flex_wrap().gap_1().children(
                        info.sets
                            .iter()
                            .map(|&set| self.set_chip(graph, set, ("schedules-in", set), cx)),
                    ),
                    theme,
                ))
            })
            .child(runs_after)
            .child(runs_before)
            .child(reach)
            .children(ambiguities)
            .when(!sync, |details| {
                details.child(section("Access".to_owned(), None, access, theme))
            })
    }

    /// A system's name, which selects it and pans to it when clicked.
    fn link(
        &self,
        graph: &Graph,
        system: usize,
        id: impl Into<ElementId>,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let info = &graph.snapshot.systems[system];
        let name = match info.kind {
            SystemKind::SyncPoint => SharedString::new_static("sync point"),
            _ => info.name.clone(),
        };
        div()
            .id(id.into())
            .text_color(theme.link)
            .cursor_pointer()
            .hover(|link| link.underline())
            .child(name)
            .on_click(cx.listener(move |this, _, _, cx| this.go_to(system, cx)))
            .into_any_element()
    }

    /// A set's name, in a chip, which selects it and pans to its box when
    /// clicked.
    fn set_chip(
        &self,
        graph: &Graph,
        set: usize,
        id: impl Into<ElementId>,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let hover = theme.secondary_hover;
        chip(graph.snapshot.sets[set].name.clone(), theme)
            .font_family(theme.mono_font_family.clone())
            .id(id.into())
            .cursor_pointer()
            .hover(move |chip| chip.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| this.go_to_set(set, cx)))
            .into_any_element()
    }

    /// The schedule's sets, each under the set it's drawn in, with how many
    /// systems are in it; a set drawn without a box is muted.
    fn set_list(&self, graph: &Graph, cx: &Context<Self>) -> Option<impl IntoElement + use<>> {
        let theme = cx.theme();
        let sets = &graph.snapshot.sets;
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); sets.len()];
        let mut roots = Vec::new();
        for (set, info) in sets.iter().enumerate() {
            if info.systems.is_empty() {
                continue;
            }
            match graph.tree.parents[set] {
                Some(parent) => children[parent].push(set),
                None => roots.push(set),
            }
        }
        if roots.is_empty() {
            return None;
        }
        let by_name = |list: &mut Vec<usize>| {
            list.sort_by_cached_key(|&set| sets[set].name.to_lowercase());
        };
        by_name(&mut roots);
        let mut rows = Vec::new();
        let mut stack: Vec<(usize, usize)> = roots.iter().rev().map(|&set| (set, 0)).collect();
        while let Some((set, depth)) = stack.pop() {
            let info = &sets[set];
            let boxed = graph.layout.sets[set].is_some();
            rows.push(
                h_flex()
                    .id(("schedules-set-row", set))
                    .gap_1p5()
                    .pl(px(4.0 + 12.0 * depth as f32))
                    .pr_1()
                    .rounded_sm()
                    .cursor_pointer()
                    .hover(|row| row.bg(theme.list_hover))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .font_family(theme.mono_font_family.clone())
                            .text_xs()
                            .when(!boxed, |name| name.text_color(theme.muted_foreground))
                            .child(info.name.clone()),
                    )
                    .child(
                        div()
                            .text_color(theme.muted_foreground)
                            .child(info.systems.len().to_string()),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| this.go_to_set(set, cx))),
            );
            let mut inner = children[set].clone();
            by_name(&mut inner);
            stack.extend(inner.iter().rev().map(|&child| (child, depth + 1)));
        }
        Some(section(
            format!("Sets ({})", rows.len()),
            Some("Muted: drawn without a box. Select one to see why."),
            v_flex().children(rows),
            theme,
        ))
    }

    /// The selected set's name; why it has no box, if it hasn't; the sets
    /// it's in and that are in it; what it's declared to run after and
    /// before; and its systems.
    fn set_details(
        &self,
        graph: &Graph,
        set: usize,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let sets = &graph.snapshot.sets;
        let info = &sets[set];
        let total = info.systems.len();
        let inside = info
            .systems
            .iter()
            .filter(|&&system| graph.tree.holds(set, system))
            .count();
        let note = if total == 0 {
            Some("No system is in it.".to_owned())
        } else if graph.layout.sets[set].is_some() {
            (inside < total).then(|| {
                let others = total - inside;
                format!(
                    "{others} of its systems {} drawn in the boxes of other sets {} in.",
                    if others == 1 { "is" } else { "are" },
                    if others == 1 { "it's" } else { "they're" },
                )
            })
        } else if inside == 0 {
            Some(
                "Drawn without a box: its systems are drawn in the boxes of other sets \
                 they're in."
                    .to_owned(),
            )
        } else {
            Some("Drawn without a box: systems outside it run between its systems.".to_owned())
        };

        let chips = |list: &[usize], id: &'static str| {
            h_flex()
                .flex_wrap()
                .gap_1()
                .children(list.iter().map(|&s| self.set_chip(graph, s, (id, s), cx)))
        };
        let parents = (!info.parents.is_empty()).then(|| {
            section(
                "In".to_owned(),
                None,
                chips(&info.parents, "schedules-set-in"),
                theme,
            )
        });
        let inner: Vec<usize> = (0..sets.len())
            .filter(|&s| sets[s].parents.contains(&set))
            .collect();
        let inner = (!inner.is_empty()).then(|| {
            section(
                "Sets in it".to_owned(),
                None,
                chips(&inner, "schedules-set-inner"),
                theme,
            )
        });

        let orderings = |title: &str, neighbors: &[Neighbor], id: &'static str| {
            let rows: Vec<AnyElement> = neighbors
                .iter()
                .enumerate()
                .map(|(i, neighbor)| match neighbor.target {
                    Some(Target::System(system)) => self.link(graph, system, (id, i), cx),
                    Some(Target::Set(other)) => self.set_chip(graph, other, (id, i), cx),
                    None => div()
                        .font_family(theme.mono_font_family.clone())
                        .text_xs()
                        .child(neighbor.name.clone())
                        .into_any_element(),
                })
                .collect();
            section(
                format!("{title} ({})", neighbors.len()),
                None,
                if rows.is_empty() {
                    weak("Nothing declared on it.", theme).into_any_element()
                } else {
                    v_flex()
                        .gap_1()
                        .items_start()
                        .children(rows)
                        .into_any_element()
                },
                theme,
            )
        };
        let systems = section(
            format!("Systems ({total})"),
            None,
            v_flex().gap_0p5().children(
                info.systems
                    .iter()
                    .map(|&system| self.link(graph, system, ("schedules-member", system), cx)),
            ),
            theme,
        );

        v_flex()
            .gap_3()
            .p_2()
            .child(
                v_flex()
                    .gap_0p5()
                    .child(
                        div()
                            .text_base()
                            .font_weight(FontWeight::SEMIBOLD)
                            .font_family(theme.mono_font_family.clone())
                            .child(info.name.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(match total {
                                1 => "Set of 1 system".to_owned(),
                                count => format!("Set of {count} systems"),
                            }),
                    ),
            )
            .children(note.map(|note| weak(note, theme)))
            .children(parents)
            .children(inner)
            .child(orderings(
                "Ordered after",
                &info.after,
                "schedules-set-after",
            ))
            .child(orderings(
                "Ordered before",
                &info.before,
                "schedules-set-before",
            ))
            .child(systems)
    }
}

impl Render for SchedulesView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let picked = self.picked;
        read(
            window,
            cx,
            move |world: &World| snapshot(world, picked),
            |this, snapshot, window, cx| this.receive(snapshot, window, cx),
        );
        if self.frame_pending && self.viewport.get().size.width > px(0.) {
            self.camera = self.first_camera();
            self.frame_pending = false;
        }
        let toolbar = self.toolbar(cx);
        let graph = self.graph_panel(cx);
        let inspector = self.inspector_panel(cx);
        let split = if self.wide {
            h_resizable("schedules-split-wide")
                .child(resizable_panel().child(graph))
                .child(
                    resizable_panel()
                        .size(px(320.))
                        .size_range(px(200.)..px(4096.))
                        .child(inspector),
                )
                .into_any_element()
        } else {
            v_resizable("schedules-split-tall")
                .child(resizable_panel().child(graph))
                .child(
                    resizable_panel()
                        .size(px(280.))
                        .size_range(px(120.)..px(4096.))
                        .child(inspector),
                )
                .into_any_element()
        };
        let view = cx.weak_entity();
        let measure = canvas(
            move |bounds, _, cx| {
                // Wider than tall by half again: the graph is the wide part.
                let wide = bounds.size.width > bounds.size.height * 1.5;
                if view
                    .upgrade()
                    .is_some_and(|view| view.read(cx).wide != wide)
                {
                    let view = view.clone();
                    cx.defer(move |cx| {
                        view.update(cx, |this, cx| {
                            this.wide = wide;
                            cx.notify();
                        })
                        .ok();
                    });
                }
            },
            |_, _, _, _| {},
        )
        .absolute()
        .size_full();
        v_flex()
            .key_context("Schedules")
            .relative()
            .size_full()
            .on_action(cx.listener(|this, _: &FocusFilter, window, cx| {
                this.filter
                    .update(cx, |filter, cx| filter.focus(window, cx))
            }))
            .on_action(cx.listener(|this, _: &ZoomIn, _, cx| this.zoom_by(1.25, cx)))
            .on_action(cx.listener(|this, _: &ZoomOut, _, cx| this.zoom_by(0.8, cx)))
            .on_action(cx.listener(|this, _: &ZoomToFit, _, cx| {
                this.camera = this.fit_camera();
                cx.notify();
            }))
            .child(toolbar)
            .child(div().flex_1().min_h_0().child(split))
            .child(measure)
    }
}

impl Graph {
    fn new(snapshot: GraphSnapshot, window: &Window, cx: &App) -> Self {
        let font = gpui::font(cx.theme().font_family.clone());
        let sizes: Vec<Size<f32>> = snapshot
            .systems
            .iter()
            .map(|system| match system.kind {
                SystemKind::SyncPoint => {
                    let label = SharedString::new_static("sync");
                    size(
                        label_width(&label, &font, LABEL_SIZE * 0.85, window) + 2.0 * NODE_PADDING,
                        SYNC_HEIGHT,
                    )
                }
                _ => size(
                    STRIPE
                        + 2.0 * NODE_PADDING
                        + label_width(&system.name, &font, LABEL_SIZE, window),
                    NODE_HEIGHT,
                ),
            })
            .collect();
        let edges: Vec<(usize, usize)> = snapshot
            .edges
            .iter()
            .map(|edge| (edge.before, edge.after))
            .collect();
        let tree = tree_of(&snapshot);
        let titles: Vec<f32> = snapshot
            .sets
            .iter()
            .map(|set| label_width(&set.name, &font, SET_LABEL_SIZE, window))
            .collect();
        let layout = layout::layout(&sizes, &edges, &tree, &titles);
        // Each box under the boxes in it, which are smaller.
        let area = |set: usize| layout.sets[set].map_or(0.0, |b| b.size.width * b.size.height);
        let mut boxes: Vec<usize> = (0..snapshot.sets.len())
            .filter(|&set| layout.sets[set].is_some())
            .collect();
        boxes.sort_by(|&a, &b| area(b).total_cmp(&area(a)));

        let mut counts: Vec<(SharedString, usize)> = Vec::new();
        for system in &snapshot.systems {
            if system.kind == SystemKind::SyncPoint {
                continue;
            }
            match counts.iter_mut().find(|(name, _)| *name == system.krate) {
                Some((_, count)) => *count += 1,
                None => counts.push((system.krate.clone(), 1)),
            }
        }
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let crates: Vec<(SharedString, Hsla, usize)> = counts
            .into_iter()
            .enumerate()
            .map(|(i, (name, count))| {
                (
                    name,
                    rgb(CRATE_COLORS[i % CRATE_COLORS.len()]).into(),
                    count,
                )
            })
            .collect();
        let crate_of = snapshot
            .systems
            .iter()
            .map(|system| match system.kind {
                SystemKind::SyncPoint => None,
                _ => crates.iter().position(|(name, ..)| *name == system.krate),
            })
            .collect();

        let n = snapshot.systems.len();
        let mut incoming = vec![Vec::new(); n];
        let mut outgoing = vec![Vec::new(); n];
        for (e, edge) in snapshot.edges.iter().enumerate() {
            outgoing[edge.before].push(e);
            incoming[edge.after].push(e);
        }
        let mut ambiguous = vec![Vec::new(); n];
        for (a, ambiguity) in snapshot.ambiguities.iter().enumerate() {
            ambiguous[ambiguity.a].push(a);
            ambiguous[ambiguity.b].push(a);
        }
        Graph {
            snapshot,
            layout: Rc::new(layout),
            sizes,
            crates,
            crate_of,
            incoming,
            outgoing,
            ambiguous,
            tree,
            boxes,
            relations: Vec::new(),
        }
    }

    /// Which systems are in `set`.
    fn members_of(&self, set: usize) -> Vec<Relation> {
        let mut relations = vec![Relation::Unrelated; self.snapshot.systems.len()];
        for &system in &self.snapshot.sets[set].systems {
            relations[system] = Relation::InSet;
        }
        relations
    }

    /// How each system relates to `selected`: whether it runs before it,
    /// after it, or neither.
    fn relations_to(&self, selected: usize) -> Vec<Relation> {
        let edges = &self.snapshot.edges;
        let mut relations = vec![Relation::Unrelated; self.snapshot.systems.len()];
        relations[selected] = Relation::Selected;
        for (relation, toward) in [
            (Relation::Before, &self.incoming),
            (Relation::After, &self.outgoing),
        ] {
            let mut stack = vec![selected];
            while let Some(system) = stack.pop() {
                for &e in &toward[system] {
                    let edge = &edges[e];
                    let other = if relation == Relation::Before {
                        edge.before
                    } else {
                        edge.after
                    };
                    if relations[other] == Relation::Unrelated {
                        relations[other] = relation;
                        stack.push(other);
                    }
                }
            }
        }
        relations
    }
}

/// The set each system and set of `snapshot` is drawn in; a sync point Bevy
/// adds, in the set of the systems around it.
pub(super) fn tree_of(snapshot: &GraphSnapshot) -> Tree {
    let edges: Vec<(usize, usize)> = snapshot
        .edges
        .iter()
        .map(|edge| (edge.before, edge.after))
        .collect();
    layout::tree(
        &snapshot
            .systems
            .iter()
            .map(|system| system.parents.clone())
            .collect::<Vec<_>>(),
        &snapshot
            .sets
            .iter()
            .map(|set| set.parents.clone())
            .collect::<Vec<_>>(),
        &snapshot
            .sets
            .iter()
            .map(|set| set.systems.len())
            .collect::<Vec<_>>(),
        &snapshot
            .systems
            .iter()
            .map(|system| system.kind == SystemKind::SyncPoint)
            .collect::<Vec<_>>(),
        &edges,
    )
}

/// What the graph's canvas paints: the edges' lines, and the lines to the
/// systems the selected one is ambiguous with.
struct EdgePaint {
    layout: Rc<Layout>,
    /// How each of the layout's lines is drawn.
    styles: Vec<EdgeStyle>,
    /// The color of each style, in the order of [`EdgeStyle::ALL`].
    colors: [Hsla; 7],
    camera: Camera,
    ambiguities: Vec<(Point<f32>, Point<f32>)>,
    danger: Hsla,
}

impl EdgePaint {
    fn paint(&self, bounds: Bounds<Pixels>, window: &mut Window) {
        let Camera { zoom, offset } = self.camera;
        let origin = bounds.origin + offset;
        let to_window =
            |at: Point<f32>| point(origin.x + px(at.x * zoom), origin.y + px(at.y * zoom));
        let arrow = (6.0 * zoom).clamp(3.0, 8.0);
        for (style, &color) in EdgeStyle::ALL.into_iter().zip(&self.colors) {
            let mut lines = PathBuilder::stroke(px(style.width()));
            let mut heads = PathBuilder::fill();
            let mut count = 0;
            for (line, _) in self
                .layout
                .lines
                .iter()
                .zip(&self.styles)
                .filter(|&(line, &s)| s == style && line.points.len() >= 2)
            {
                let route = &line.points;
                let (low, high) = route.iter().fold(
                    (point(f32::MAX, f32::MAX), point(f32::MIN, f32::MIN)),
                    |(low, high), at| {
                        (
                            point(low.x.min(at.x), low.y.min(at.y)),
                            point(high.x.max(at.x), high.y.max(at.y)),
                        )
                    },
                );
                let (low, high) = (to_window(low), to_window(high));
                if high.x < bounds.left()
                    || high.y < bounds.top()
                    || low.x > bounds.right()
                    || low.y > bounds.bottom()
                {
                    continue;
                }
                let points: Vec<Point<Pixels>> = route.iter().map(|&at| to_window(at)).collect();
                lines.move_to(points[0]);
                for pair in points.windows(2) {
                    curve(&mut lines, pair[0], pair[1]);
                }
                if line.arrow {
                    let tip = points[points.len() - 1];
                    heads.add_polygon(
                        &[
                            tip,
                            point(tip.x - px(arrow), tip.y - px(arrow / 2.0)),
                            point(tip.x - px(arrow), tip.y + px(arrow / 2.0)),
                        ],
                        true,
                    );
                }
                count += 1;
                if count == EDGES_PER_PATH {
                    paint_paths(lines, heads, color, window);
                    lines = PathBuilder::stroke(px(style.width()));
                    heads = PathBuilder::fill();
                    count = 0;
                }
            }
            if count > 0 {
                paint_paths(lines, heads, color, window);
            }
        }
        if !self.ambiguities.is_empty() {
            let mut lines = PathBuilder::stroke(px(1.)).dash_array(&[px(4.), px(4.)]);
            for &(a, b) in &self.ambiguities {
                lines.move_to(to_window(a));
                lines.line_to(to_window(b));
            }
            if let Ok(path) = lines.build() {
                window.paint_path(path, self.danger.opacity(0.5));
            }
        }
    }
}

fn paint_paths(lines: PathBuilder, heads: PathBuilder, color: Hsla, window: &mut Window) {
    if let Ok(path) = lines.build() {
        window.paint_path(path, color);
    }
    if let Ok(path) = heads.build() {
        window.paint_path(path, color);
    }
}

/// A segment of an edge: straight where it passes through a column, and a
/// curve leaving and entering horizontally between columns.
fn curve(path: &mut PathBuilder, from: Point<Pixels>, to: Point<Pixels>) {
    if (from.y - to.y).abs() < px(0.5) {
        path.line_to(to);
        return;
    }
    let reach = (to.x - from.x) / 2.0;
    path.cubic_bezier_to(to, point(from.x + reach, from.y), point(to.x - reach, to.y));
}

/// The value of the drag that pans the graph, from the camera's offset
/// `from`.
struct Pan {
    from: Point<Pixels>,
    /// Where the mouse was when the drag first moved.
    origin: Cell<Option<Point<Pixels>>>,
}

/// What a drag panning the graph shows under the mouse: nothing, as the
/// graph itself moves.
struct PanPreview;

impl Render for PanPreview {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        gpui::Empty
    }
}

/// Whether the mouse moved between the press and the release of `event`,
/// panning the graph rather than clicking.
fn moved(event: &ClickEvent) -> bool {
    match event {
        ClickEvent::Mouse(click) => {
            (click.up.position - click.down.position).magnitude() > CLICK_SLOP
        }
        _ => false,
    }
}

/// The width `text` takes at `size` at 100%, room for it at each of
/// [`MEASURED_ZOOMS`].
fn label_width(text: &SharedString, font: &Font, size: f32, window: &Window) -> f32 {
    let widest = MEASURED_ZOOMS
        .iter()
        .map(|&zoom| {
            let run = TextRun {
                len: text.len(),
                font: font.clone(),
                color: gpui::black(),
                background_color: None,
                underline: None,
                strikethrough: None,
            };
            let line = window
                .text_system()
                .shape_line(text.clone(), px(size * zoom), &[run], None);
            f32::from(line.width) / zoom
        })
        .fold(0.0, f32::max);
    widest.ceil() + 1.0
}

/// Whether `word` is in the system's name, path, crate or sets, ignoring
/// ASCII case.
fn system_matches(system: &SystemInfo, sets: &[SetInfo], word: &str) -> bool {
    contains_ignore_case(&system.name, word)
        || contains_ignore_case(&system.path, word)
        || contains_ignore_case(&system.krate, word)
        || system
            .sets
            .iter()
            .any(|&set| contains_ignore_case(&sets[set].name, word))
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    needle.is_empty()
        || haystack
            .as_bytes()
            .windows(needle.len())
            .any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

/// The system's name, path, crate, kind and sets.
fn system_tooltip(
    graph: &Graph,
    system: usize,
) -> impl Fn(&mut Window, &mut App) -> gpui::AnyView + use<> {
    let info = &graph.snapshot.systems[system];
    let title = match info.kind {
        SystemKind::SyncPoint => SharedString::new_static("Sync point"),
        _ => info.name.clone(),
    };
    let path = info.path.clone();
    let kind: Option<&'static str> = match info.kind {
        SystemKind::Exclusive => Some("Exclusive: runs alone"),
        SystemKind::SyncPoint => Some("Applies the commands of the systems before it"),
        SystemKind::System => None,
    };
    let sets = (!info.sets.is_empty()).then(|| format!("In {}", set_names(graph, &info.sets)));
    move |window, cx| {
        let (title, path, sets) = (title.clone(), path.clone(), sets.clone());
        Tooltip::element(move |_, cx| {
            v_flex()
                .child(div().font_weight(FontWeight::SEMIBOLD).child(title.clone()))
                .child(
                    div()
                        .font_family(cx.theme().mono_font_family.clone())
                        .text_xs()
                        .child(path.clone()),
                )
                .children(kind)
                .children(sets.clone())
        })
        .build(window, cx)
    }
}

/// The set's name, how many systems are in it, and the sets it's in.
fn set_tooltip(
    graph: &Graph,
    set: usize,
) -> impl Fn(&mut Window, &mut App) -> gpui::AnyView + use<> {
    let info = &graph.snapshot.sets[set];
    let name = info.name.clone();
    let count = match info.systems.len() {
        1 => "1 system".to_owned(),
        count => format!("{count} systems"),
    };
    let parents =
        (!info.parents.is_empty()).then(|| format!("In {}", set_names(graph, &info.parents)));
    move |window, cx| {
        let (name, count, parents) = (name.clone(), count.clone(), parents.clone());
        Tooltip::element(move |_, cx| {
            v_flex()
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .font_family(cx.theme().mono_font_family.clone())
                        .child(name.clone()),
                )
                .child(count.clone())
                .children(parents.clone())
        })
        .build(window, cx)
    }
}

/// The names of `sets`, joined by commas.
fn set_names(graph: &Graph, sets: &[usize]) -> String {
    sets.iter()
        .map(|&set| graph.snapshot.sets[set].name.as_ref())
        .collect::<Vec<&str>>()
        .join(", ")
}

/// The system's context menu, which copies its name or path.
fn system_menu(
    info: &SystemInfo,
) -> impl Fn(gpui_component::menu::PopupMenu) -> gpui_component::menu::PopupMenu + use<> {
    let (name, path) = (info.name.clone(), info.path.clone());
    move |menu| {
        let (name, path) = (name.clone(), path.clone());
        menu.item(PopupMenuItem::new("Copy name").on_click(move |_, _, cx| {
            cx.write_to_clipboard(ClipboardItem::new_string(name.to_string()))
        }))
        .item(
            PopupMenuItem::new("Copy type path").on_click(move |_, _, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(path.to_string()))
            }),
        )
    }
}

/// Why an edge orders its systems: the orderings it comes from, as
/// `before → after`.
fn reasons(
    reasons: &[Reason],
    graph: &Graph,
    edge: &super::graph::Edge,
    theme: &Theme,
) -> impl IntoElement + use<> {
    let systems = &graph.snapshot.systems;
    let sync = [edge.before, edge.after]
        .iter()
        .any(|&s| systems[s].kind == SystemKind::SyncPoint);
    let mono = theme.mono_font_family.clone();
    let muted = theme.muted_foreground;
    let rows: Vec<AnyElement> = if reasons.is_empty() {
        vec![
            div()
                .child("Through a set with no systems")
                .into_any_element(),
        ]
    } else {
        reasons
            .iter()
            .map(|reason| match reason {
                Reason::Declared { before, after } => h_flex()
                    .flex_wrap()
                    .gap_x_1()
                    .child(div().font_family(mono.clone()).child(before.clone()))
                    .child("→")
                    .child(div().font_family(mono.clone()).child(after.clone()))
                    .into_any_element(),
                Reason::Added if sync => div()
                    .child("Added by Bevy for the sync point")
                    .into_any_element(),
                Reason::Added => div().child("Added by a build pass").into_any_element(),
            })
            .collect()
    };
    v_flex().pl_2().text_xs().text_color(muted).children(rows)
}

/// The components two ambiguous systems conflict over.
fn conflicts(components: &[SharedString], theme: &Theme) -> impl IntoElement + use<> {
    let text = if components.is_empty() {
        "Over the whole world".to_owned()
    } else {
        format!("Over {}", components.join(", "))
    };
    div()
        .pl_2()
        .text_xs()
        .text_color(theme.muted_foreground)
        .child(text)
}

/// A bold heading over `contents`, with a line explaining them.
fn section(
    title: String,
    description: Option<&'static str>,
    contents: impl IntoElement,
    theme: &Theme,
) -> gpui::Div {
    v_flex()
        .gap_1()
        .child(div().font_weight(FontWeight::SEMIBOLD).child(title))
        .children(description.map(|text| {
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(text)
        }))
        .child(contents)
}

fn weak(text: impl Into<SharedString>, theme: &Theme) -> gpui::Div {
    div().text_color(theme.muted_foreground).child(text.into())
}

fn chip(text: impl Into<SharedString>, theme: &Theme) -> gpui::Div {
    div()
        .px_1()
        .rounded_sm()
        .text_xs()
        .bg(theme.secondary)
        .text_color(theme.secondary_foreground)
        .child(text.into())
}

/// A chip with a type's name: a set's, or a component's.
fn code_chip(text: SharedString, theme: &Theme) -> impl IntoElement + use<> {
    chip(text, theme).font_family(theme.mono_font_family.clone())
}

fn message(text: &'static str, theme: &Theme) -> AnyElement {
    div()
        .p_2()
        .text_color(theme.muted_foreground)
        .child(text)
        .into_any_element()
}
