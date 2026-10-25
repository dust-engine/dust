use std::cell::Cell;
use std::time::Duration;

use bevy::ecs::component::ComponentId;
use bevy::math::EulerRot;
use bevy::platform::collections::HashMap;
use bevy::platform::time::Instant;
use bevy::prelude::{Entity, Quat};
use bevy::reflect::PartialReflect;
use gpui::{
    AnyElement, AnyView, App, AppContext as _, ClipboardItem, Context, CursorStyle, DragMoveEvent,
    ElementId, FocusHandle, FontStyle, FontWeight, HighlightStyle, Hsla, InteractiveElement as _,
    IntoElement, KeyBinding, MouseButton, NavigationDirection, ParentElement as _, Pixels, Render,
    ScrollHandle, ScrollStrategy, SharedString, StatefulInteractiveElement as _, Styled as _,
    StyledText, Subscription, UniformListScrollHandle, Window, actions, canvas, div,
    prelude::FluentBuilder as _, px, rgb, uniform_list,
};
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::checkbox::Checkbox;
use gpui_component::input::{Escape, Input, InputEvent, InputState};
use gpui_component::menu::{ContextMenuExt as _, DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_component::scroll::ScrollableElement as _;
use gpui_component::tooltip::Tooltip;
use gpui_component::{
    ActiveTheme as _, Disableable as _, Icon, IconName, Selectable as _, Sizable as _, Theme,
    h_flex, h_resizable, resizable_panel, v_flex, v_resizable,
};

use super::inspector::{Body, ComponentView, EntityView, InspectorSnapshot};
use super::tree::{RowView, TreeKey, TreeSnapshot};
use super::value::{Leaf, LeafValue, Link, Number, Piece, ValueRow, ValueRowKind, Variant};
use super::{Action, Edit, Label, Section, Snapshot, send, snapshot};
use crate::watch;

actions!(
    dust_devtools_entities,
    [
        SelectUp,
        SelectDown,
        SelectLeft,
        SelectRight,
        FocusFilter,
        GoBack,
        GoForward
    ]
);

/// Height of a row of the tree, and of a component's header.
const ROW_HEIGHT: Pixels = px(22.);

/// Width of a disclosure triangle's column, which is also the indentation of
/// each level of the tree and of a value.
const INDENT: Pixels = px(14.);

/// The most components a row of the tree lists after the entity's title; the
/// rest are counted.
const ROW_COMPONENTS: usize = 3;

/// How long a component's header stays highlighted after the component
/// changes.
const FLASH: Duration = Duration::from_millis(800);

pub(crate) struct EntitiesView {
    tree: Option<TreeSnapshot>,
    inspector: Option<InspectorSnapshot>,
    tree_filter: gpui::Entity<InputState>,
    component_filter: gpui::Entity<InputState>,
    tree_scroll: UniformListScrollHandle,
    /// The tree takes the keyboard focus when clicked, and then moves the
    /// selection with the arrow keys.
    tree_focus: FocusHandle,
    breadcrumbs_scroll: ScrollHandle,
    /// When each component of `flash_entity` was last seen to change, to
    /// highlight its header as Chrome highlights a changed attribute.
    flashes: HashMap<ComponentId, Instant>,
    flash_entity: Option<Entity>,
    /// The value being edited.
    editing: Option<Editing>,
    /// Whether the tab is wider than it's tall, which puts the tree beside
    /// the inspector rather than above it, as Chrome lays out the Elements
    /// panel.
    wide: bool,
    _subscriptions: Vec<Subscription>,
}

impl EntitiesView {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.bind_keys([
            KeyBinding::new("up", SelectUp, Some("EntityTree")),
            KeyBinding::new("down", SelectDown, Some("EntityTree")),
            KeyBinding::new("left", SelectLeft, Some("EntityTree")),
            KeyBinding::new("right", SelectRight, Some("EntityTree")),
            KeyBinding::new("cmd-f", FocusFilter, Some("Entities")),
            KeyBinding::new("cmd-[", GoBack, Some("Entities")),
            KeyBinding::new("cmd-]", GoForward, Some("Entities")),
        ]);
        let tree_filter = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Filter (⌘F)")
                .clean_on_escape()
        });
        let component_filter = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Filter components")
                .clean_on_escape()
        });
        let subscriptions = vec![
            cx.subscribe(&tree_filter, |_, filter, event: &InputEvent, cx| {
                if let InputEvent::Change = event {
                    send(cx, Action::Filter(filter.read(cx).value().to_string()));
                }
            }),
            cx.subscribe(&component_filter, |_, filter, event: &InputEvent, cx| {
                if let InputEvent::Change = event {
                    let text = filter.read(cx).value().to_string();
                    send(cx, Action::ComponentFilter(text));
                }
            }),
        ];
        EntitiesView {
            tree: None,
            inspector: None,
            tree_filter,
            component_filter,
            tree_scroll: UniformListScrollHandle::new(),
            tree_focus: cx.focus_handle(),
            breadcrumbs_scroll: ScrollHandle::new(),
            flashes: HashMap::default(),
            flash_entity: None,
            editing: None,
            wide: true,
            _subscriptions: subscriptions,
        }
    }

    fn receive(&mut self, snapshot: Snapshot) {
        if let Some(tree) = snapshot.tree {
            if let Some(index) = tree.reveal {
                reveal(&self.tree_scroll, index);
                // The selected entity is the last breadcrumb, after a
                // separator between each two.
                let last = tree.breadcrumbs.len().saturating_sub(1) * 2;
                self.breadcrumbs_scroll.scroll_to_item(last);
            }
            self.tree = Some(tree);
        }
        if let Some(inspector) = snapshot.inspector {
            let entity = inspector.entity.as_ref().map(|view| view.entity);
            if entity != self.flash_entity {
                self.flashes.clear();
                self.flash_entity = entity;
            }
            let now = Instant::now();
            for &id in &inspector.changed {
                self.flashes.insert(id, now);
            }
            if self
                .editing
                .as_ref()
                .is_some_and(|editing| Some(editing.target.place.entity) != entity)
            {
                self.editing = None;
            }
            self.inspector = Some(inspector);
        }
    }

    /// Shows `text` in the tree's filter field, and filters the tree by it.
    fn set_tree_filter(&mut self, text: SharedString, window: &mut Window, cx: &mut Context<Self>) {
        self.tree_filter
            .update(cx, |filter, cx| filter.set_value(text.clone(), window, cx));
        send(cx, Action::Filter(text.to_string()));
    }

    /// The filter field, the rows, and the breadcrumbs.
    fn tree_panel(&self, window: &Window, cx: &Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let tree = self.tree.as_ref();
        let toolbar = h_flex()
            .gap_2()
            .p_1()
            .border_b_1()
            .border_color(theme.border)
            .child(
                div()
                    .id("entities-filter")
                    .flex_1()
                    .min_w_0()
                    .tooltip(|window, cx| {
                        Tooltip::new(
                            "Shows the entities whose name, component names or id contain \
                             each word.",
                        )
                        .build(window, cx)
                    })
                    .child(Input::new(&self.tree_filter).xsmall().cleanable(true)),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .children(tree.map(|tree| tree.count.clone())),
            );

        let focused = self.tree_focus.is_focused(window);
        let view = cx.weak_entity();
        let count = tree.map_or(0, |tree| tree.rows.len());
        let rows = uniform_list("entities-tree", count, move |range, _, cx| {
            let Some(view) = view.upgrade() else {
                return Vec::new();
            };
            let Some(tree) = &view.read(cx).tree else {
                return Vec::new();
            };
            range
                .filter_map(|index| Some(tree_row(tree, tree.rows.get(index)?, focused, cx)))
                .collect()
        })
        .track_scroll(&self.tree_scroll)
        .size_full();
        let rows = div()
            .id("entities-tree-rows")
            .flex_1()
            .min_h_0()
            .track_focus(&self.tree_focus)
            .key_context("EntityTree")
            .on_action(|_: &SelectUp, _, cx| send(cx, Action::Key(TreeKey::Up)))
            .on_action(|_: &SelectDown, _, cx| send(cx, Action::Key(TreeKey::Down)))
            .on_action(|_: &SelectLeft, _, cx| send(cx, Action::Key(TreeKey::Left)))
            .on_action(|_: &SelectRight, _, cx| send(cx, Action::Key(TreeKey::Right)))
            // Clicks on the rows, and between and below them.
            .on_click(cx.listener(|this, _, window, cx| this.tree_focus.focus(window, cx)))
            .child(
                div()
                    .relative()
                    .size_full()
                    .child(rows)
                    .vertical_scrollbar(&self.tree_scroll),
            );

        let breadcrumbs = tree
            .filter(|tree| !tree.breadcrumbs.is_empty())
            .map(|tree| {
                let mut bar = h_flex()
                    .id("entities-breadcrumbs")
                    .flex_shrink_0()
                    .gap_0p5()
                    .px_1()
                    .py_0p5()
                    .border_t_1()
                    .border_color(theme.border)
                    .overflow_x_scroll()
                    .track_scroll(&self.breadcrumbs_scroll);
                for (i, (entity, label)) in tree.breadcrumbs.iter().enumerate() {
                    if i > 0 {
                        bar = bar.child(
                            Icon::new(IconName::ChevronRight)
                                .xsmall()
                                .flex_shrink_0()
                                .text_color(theme.muted_foreground),
                        );
                    }
                    let entity = *entity;
                    bar = bar.child(
                        div().flex_shrink_0().child(
                            Button::new(("entities-breadcrumb", i))
                                .label(label.short())
                                .ghost()
                                .xsmall()
                                .selected(tree.selected == Some(entity))
                                .when(!label.components.is_empty(), |crumb| {
                                    crumb.tooltip(label.components.join(", "))
                                })
                                .on_click(move |_, _, cx| send(cx, Action::Select(entity))),
                        ),
                    );
                }
                bar
            });

        v_flex()
            .size_full()
            .child(toolbar)
            .child(rows)
            .children(breadcrumbs)
    }

    /// Back and forward buttons, the entity's title and its ⋮ menu, then its
    /// details and components.
    fn inspector_panel(&self, cx: &Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let snapshot = self.inspector.as_ref();
        let entity = snapshot.and_then(|snapshot| snapshot.entity.as_ref());
        let title_bar = h_flex()
            .flex_shrink_0()
            .gap_1()
            .px_1()
            .py_0p5()
            .border_b_1()
            .border_color(theme.border)
            .child(
                Button::new("entities-back")
                    .icon(IconName::ChevronLeft)
                    .ghost()
                    .xsmall()
                    .disabled(!snapshot.is_some_and(|snapshot| snapshot.can_go_back))
                    .tooltip("Back (⌘[)")
                    .on_click(|_, _, cx| send(cx, Action::Back)),
            )
            .child(
                Button::new("entities-forward")
                    .icon(IconName::ChevronRight)
                    .ghost()
                    .xsmall()
                    .disabled(!snapshot.is_some_and(|snapshot| snapshot.can_go_forward))
                    .tooltip("Forward (⌘])")
                    .on_click(|_, _, cx| send(cx, Action::Forward)),
            )
            .children(entity.map(|view| title(view, theme)));

        let message = |text: String| {
            div()
                .p_2()
                .text_color(theme.muted_foreground)
                .child(text)
                .into_any_element()
        };
        let contents = match snapshot {
            None => div().into_any_element(),
            Some(snapshot) => match (&snapshot.entity, snapshot.selected) {
                (Some(view), _) => self.entity(view, cx).into_any_element(),
                (None, Some(entity)) => message(format!("{entity} was despawned.")),
                (None, None) => message("Select an entity in the tree.".to_owned()),
            },
        };
        v_flex().size_full().child(title_bar).child(contents)
    }

    /// The entity's details, the component filter, and the components.
    fn entity(&self, view: &EntityView, cx: &Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let mut ids = Ids::default();
        let mut components = v_flex();
        if view.components.is_empty() {
            let text = if view.filtered {
                "No components match the filter."
            } else {
                "No components."
            };
            components =
                components.child(div().p_2().text_color(theme.muted_foreground).child(text));
        }
        for component in &view.components {
            components = components.child(self.component(view, component, &mut ids, cx));
        }
        v_flex()
            .flex_1()
            .min_h_0()
            .child(details(view, theme))
            .child(
                div()
                    .flex_shrink_0()
                    .px_1()
                    .pb_1()
                    .child(Input::new(&self.component_filter).xsmall().cleanable(true)),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .border_t_1()
                    .border_color(theme.border)
                    .overflow_y_scrollbar()
                    .child(components),
            )
    }

    /// A component's header, and below it, while it's open, its value.
    fn component(
        &self,
        view: &EntityView,
        component: &ComponentView,
        ids: &mut Ids,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let id = component.id;
        let entity = view.entity;
        let tooltip = (
            component.type_path.clone(),
            component.storage.clone(),
            component.ticks.clone(),
        );
        let (name, type_path, removable) = (
            component.name.clone(),
            component.type_path.clone(),
            component.removable,
        );
        let this = cx.weak_entity();
        let header = h_flex()
            .id(("entities-component", id.index()))
            .h(ROW_HEIGHT)
            .w_full()
            .pr_2()
            .bg(self.flash(id, theme))
            .child(match component.has_value {
                true => disclosure(component.body.is_some(), theme).into_any_element(),
                false => div().w(INDENT).flex_shrink_0().into_any_element(),
            })
            .child(
                div()
                    .min_w_0()
                    .truncate()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(component.name.clone()),
            )
            .child(div().flex_1().min_w_2())
            .child(
                div()
                    .flex_shrink_0()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(component.krate.clone()),
            )
            .when(component.has_value, |header| {
                header
                    .cursor_pointer()
                    .on_click(move |_, _, cx| send(cx, Action::ToggleComponent(id)))
            })
            .tooltip(move |window, cx| {
                let (type_path, storage, ticks) = tooltip.clone();
                Tooltip::element(move |_, cx| {
                    v_flex()
                        .child(
                            div()
                                .font_family(cx.theme().mono_font_family.clone())
                                .child(type_path.clone()),
                        )
                        .child(storage.clone())
                        .children(ticks.clone())
                })
                .build(window, cx)
            })
            .context_menu(move |menu, _, cx| {
                let type_path = type_path.clone();
                let name = name.clone();
                let this = this.clone();
                let mut menu = menu
                    .item(
                        PopupMenuItem::new("Copy type path").on_click(move |_, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(type_path.to_string()))
                        }),
                    )
                    .item(
                        PopupMenuItem::new("Show entities with this component").on_click(
                            move |_, window, cx| {
                                let name = name.clone();
                                this.update(cx, |this, cx| this.set_tree_filter(name, window, cx))
                                    .ok();
                            },
                        ),
                    );
                if removable {
                    let danger = cx.theme().danger;
                    menu = menu.separator().item(
                        PopupMenuItem::element(move |_, _| {
                            div().text_color(danger).child("Remove component")
                        })
                        .on_click(move |_, _, cx| send(cx, Action::RemoveComponent(entity, id))),
                    );
                }
                menu
            });
        let body: Vec<AnyElement> = match &component.body {
            None => Vec::new(),
            Some(Body::Rows(rows)) => rows
                .iter()
                .map(|row| self.value_row((entity, id), row, ids, cx))
                .collect(),
            Some(Body::Name(text)) => {
                let place = Place {
                    entity,
                    component: id,
                    path: Vec::new(),
                    axis: None,
                };
                let field = self.editable(
                    EditTarget {
                        place,
                        kind: EditKind::Name,
                    },
                    format!("{text:?}"),
                    text.to_string(),
                    Syntax::Value,
                    ids,
                    cx,
                );
                vec![
                    h_flex()
                        .min_h(ROW_HEIGHT)
                        .pl(INDENT * 2.)
                        .pr_2()
                        .child(field)
                        .into_any_element(),
                ]
            }
            Some(Body::NotReflected) => vec![
                div()
                    .pl(INDENT * 2.)
                    .pr_2()
                    .pb_0p5()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(
                        "Not reflected. Derive `Reflect` for the type to see its value; a \
                         generic type also needs `App::register_type`.",
                    )
                    .into_any_element(),
            ],
        };
        v_flex()
            .pb_0p5()
            .border_b_1()
            .border_color(theme.border)
            .child(header)
            .children(body)
    }

    /// The background of the component's header: the selection color right
    /// after a change, fading out over [`FLASH`].
    fn flash(&self, id: ComponentId, theme: &Theme) -> Hsla {
        let fade = self.flashes.get(&id).map_or(0.0, |changed| {
            1.0 - changed.elapsed().as_secs_f32() / FLASH.as_secs_f32()
        });
        if fade > 0.0 {
            theme.selection.opacity(0.6 * fade)
        } else {
            gpui::transparent_black()
        }
    }

    /// A row of a component's value. A field's name lines up with the names
    /// of collapsible fields, which follow a disclosure triangle.
    fn value_row(
        &self,
        at: (Entity, ComponentId),
        row: &ValueRow,
        ids: &mut Ids,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let left = INDENT * (row.depth + 1) as f32;
        match &row.kind {
            ValueRowKind::Inline { name, pieces } => h_flex()
                .flex_wrap()
                .gap_x_1p5()
                .min_h(ROW_HEIGHT)
                .pl(left + INDENT)
                .pr_2()
                .children(name.as_ref().map(|name| field_name(name, theme)))
                .children(pieces.iter().map(|piece| self.piece(at, piece, ids, cx)))
                .into_any_element(),
            ValueRowKind::Nested {
                name,
                path,
                open,
                header,
            } => {
                let (component, path) = (at.1, path.clone());
                h_flex()
                    .id(ids.next())
                    .min_h(ROW_HEIGHT)
                    .pl(left)
                    .pr_2()
                    .cursor_pointer()
                    .on_click(move |_, _, cx| {
                        send(cx, Action::ToggleValue(component, path.clone()))
                    })
                    .child(disclosure(*open, theme))
                    .child(
                        h_flex()
                            .flex_1()
                            .min_w_0()
                            .gap_x_1p5()
                            .child(field_name(name, theme))
                            .children(header.iter().map(|piece| self.piece(at, piece, ids, cx))),
                    )
                    .into_any_element()
            }
            ValueRowKind::More(more) => div()
                .min_h(ROW_HEIGHT)
                .pl(left + INDENT)
                .text_color(theme.muted_foreground)
                .child(format!("… {more} more"))
                .into_any_element(),
        }
    }

    fn piece(
        &self,
        at: (Entity, ComponentId),
        piece: &Piece,
        ids: &mut Ids,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        match piece {
            Piece::Name(text) | Piece::Weak(text) => div()
                .text_color(theme.muted_foreground)
                .child(text.clone())
                .into_any_element(),
            Piece::Text(text) => div().child(text.clone()).into_any_element(),
            Piece::Preview(text) => div()
                .min_w_0()
                .truncate()
                .font_family(theme.mono_font_family.clone())
                .text_color(theme.muted_foreground)
                .child(text.clone())
                .into_any_element(),
            Piece::Entity(link) => entity_link(link, ids.next(), theme),
            Piece::Variant(variant) => variant_picker(at, variant, ids, theme),
            Piece::Leaf(leaf) => self.leaf(at, leaf, ids, cx),
        }
    }

    fn leaf(
        &self,
        (entity, component): (Entity, ComponentId),
        leaf: &Leaf,
        ids: &mut Ids,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let target = |kind, axis| EditTarget {
            place: Place {
                entity,
                component,
                path: leaf.path.clone(),
                axis,
            },
            kind,
        };
        match &leaf.value {
            &LeafValue::Number(number) if leaf.editable => self.editable(
                target(EditKind::Number(number), None),
                number.to_string(),
                number.to_string(),
                Syntax::Number,
                ids,
                cx,
            ),
            LeafValue::Number(number) => value_text(number.to_string(), Syntax::Number, theme),
            &LeafValue::Bool(value) if leaf.editable => {
                let path = leaf.path.clone();
                Checkbox::new(ids.next())
                    .checked(value)
                    .on_click(move |&checked, _, cx| {
                        let value = Edit::Value(Box::new(checked));
                        send(cx, edit_action(entity, component, path.clone(), value));
                    })
                    .into_any_element()
            }
            LeafValue::Bool(value) => value_text(value.to_string(), Syntax::Number, theme),
            LeafValue::String(value) if leaf.editable => self.editable(
                target(EditKind::Text, None),
                format!("{value:?}"),
                value.clone(),
                Syntax::Value,
                ids,
                cx,
            ),
            LeafValue::String(value) => value_text(format!("{value:?}"), Syntax::Value, theme),
            LeafValue::Quoted(text) => value_text(text.to_string(), Syntax::Value, theme),
            LeafValue::Duration(duration) if leaf.editable => self.editable(
                target(EditKind::Seconds, None),
                format!("{duration:?}"),
                duration.as_secs_f64().to_string(),
                Syntax::Number,
                ids,
                cx,
            ),
            LeafValue::Duration(duration) => {
                value_text(format!("{duration:?}"), Syntax::Number, theme)
            }
            &LeafValue::Rotation(rotation) => {
                // Euler angles in degrees are easier to read and edit than a
                // quaternion's components.
                let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
                let angles = [x, y, z].map(f32::to_degrees);
                let mut row = h_flex().id(ids.next()).gap_x_1p5().tooltip(|window, cx| {
                    Tooltip::new("Euler angles, XYZ order").build(window, cx)
                });
                for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
                    let angle = angles[axis];
                    let text = format!("{angle:.1}°");
                    let value = match leaf.editable {
                        true => self.editable(
                            target(EditKind::Angle(rotation), Some(axis)),
                            text,
                            angle.to_string(),
                            Syntax::Number,
                            ids,
                            cx,
                        ),
                        false => value_text(text, Syntax::Number, theme),
                    };
                    row = row
                        .child(div().text_color(theme.muted_foreground).child(name))
                        .child(value);
                }
                row.into_any_element()
            }
        }
    }

    /// An editable value, shown as `text`. Clicking it edits `edit_text` in a
    /// field in its place; dragging a number changes it.
    fn editable(
        &self,
        target: EditTarget,
        text: String,
        edit_text: String,
        syntax: Syntax,
        ids: &mut Ids,
        cx: &Context<Self>,
    ) -> AnyElement {
        let id = ids.next();
        if let Some(editing) = &self.editing
            && editing.target.place == target.place
        {
            return div()
                .id(id)
                .min_w(px(96.))
                .on_action(cx.listener(|this, _: &Escape, _, cx| {
                    this.editing = None;
                    cx.notify();
                }))
                .child(Input::new(&editing.input).xsmall())
                .into_any_element();
        }
        let scrub = target.scrub_start();
        let mut value = div()
            .id(id)
            .text_color(syntax.color(cx.theme()))
            .cursor(CursorStyle::IBeam)
            .child(text)
            .on_click(cx.listener({
                let target = target.clone();
                move |this, _, window, cx| {
                    // A field within a collapsible row leaves it as it is.
                    cx.stop_propagation();
                    this.start_editing(target.clone(), edit_text.clone(), window, cx);
                }
            }));
        if let Some((start, speed)) = scrub {
            value = value.cursor(CursorStyle::ResizeLeftRight).on_drag(
                Scrub {
                    target,
                    start,
                    speed,
                    origin: Cell::new(None),
                    last: Cell::new(start),
                },
                |_, _, _, cx| cx.new(|_| ScrubPreview),
            );
        }
        value.into_any_element()
    }

    fn start_editing(
        &mut self,
        target: EditTarget,
        text: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // Clicking a value doesn't take the focus from the field, which would
        // apply its edit, so the edit is applied here.
        self.commit(cx);
        let input = cx.new(|cx| InputState::new(window, cx).default_value(text));
        input.update(cx, |input, cx| {
            input.focus(window, cx);
            input.select_all(window, cx);
        });
        let subscription = cx.subscribe_in(&input, window, |this, _, event: &InputEvent, _, cx| {
            if let InputEvent::PressEnter { .. } | InputEvent::Blur = event {
                this.commit(cx);
            }
        });
        self.editing = Some(Editing {
            target,
            input,
            _subscription: subscription,
        });
        cx.notify();
    }

    /// Sends the edited value, if the text is one, and stops editing.
    fn commit(&mut self, cx: &mut Context<Self>) {
        let Some(editing) = self.editing.take() else {
            return;
        };
        let text = editing.input.read(cx).value();
        if let Some(action) = editing.target.action(&text) {
            send(cx, action);
        }
        cx.notify();
    }

    /// Changes the number dragged by how far it was dragged.
    fn scrub(&mut self, event: &DragMoveEvent<Scrub>, _: &mut Window, cx: &mut Context<Self>) {
        let scrub = event.drag(cx);
        let x = f64::from(f32::from(event.event.position.x));
        // From where the drag started moving the number, rather than where
        // the mouse went down, which is a few pixels back.
        let origin = match scrub.origin.get() {
            Some(origin) => origin,
            None => {
                scrub.origin.set(Some(x));
                x
            }
        };
        let value = scrub.start + (x - origin) * scrub.speed;
        if value == scrub.last.get() {
            return;
        }
        scrub.last.set(value);
        if let Some(action) = scrub.target.scrubbed(value) {
            send(cx, action);
        }
    }
}

impl Render for EntitiesView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        watch(window, cx, snapshot, |this, snapshot, _, _| {
            this.receive(snapshot)
        });
        if self
            .flashes
            .values()
            .any(|changed| changed.elapsed() < FLASH)
        {
            window.request_animation_frame();
        }
        let tree = self.tree_panel(window, cx);
        let inspector = self.inspector_panel(cx);
        let split = if self.wide {
            h_resizable("entities-split-wide")
                .child(
                    resizable_panel()
                        .size(px(360.))
                        .size_range(px(160.)..px(4096.))
                        .child(tree),
                )
                .child(resizable_panel().child(inspector))
                .into_any_element()
        } else {
            v_resizable("entities-split-tall")
                .child(
                    resizable_panel()
                        .size(px(280.))
                        .size_range(px(120.)..px(4096.))
                        .child(tree),
                )
                .child(resizable_panel().child(inspector))
                .into_any_element()
        };
        let view = cx.weak_entity();
        let measure = canvas(
            move |bounds, _, cx| {
                let wide = bounds.size.width > bounds.size.height;
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
            .key_context("Entities")
            .relative()
            .size_full()
            .on_action(cx.listener(|this, _: &FocusFilter, window, cx| {
                this.tree_filter
                    .update(cx, |filter, cx| filter.focus(window, cx))
            }))
            .on_action(|_: &GoBack, _, cx| send(cx, Action::Back))
            .on_action(|_: &GoForward, _, cx| send(cx, Action::Forward))
            // The mouse's back and forward buttons, as in a browser.
            .on_mouse_down(
                MouseButton::Navigate(NavigationDirection::Back),
                |_, _, cx| send(cx, Action::Back),
            )
            .on_mouse_down(
                MouseButton::Navigate(NavigationDirection::Forward),
                |_, _, cx| send(cx, Action::Forward),
            )
            .on_drag_move(cx.listener(Self::scrub))
            .child(split)
            .child(measure)
    }
}

/// Scrolls the tree to the row at `index` if it's out of view, centering it
/// so that its surroundings show too.
fn reveal(scroll: &UniformListScrollHandle, index: usize) {
    let state = scroll.0.borrow();
    let in_view = state.last_item_size.is_some_and(|size| {
        let top = -state.base_handle.offset().y;
        let height = state.base_handle.bounds().size.height;
        let row_top = size.item.height * index as f32;
        row_top >= top && row_top + size.item.height <= top + height
    });
    drop(state);
    if !in_view {
        scroll.scroll_to_item(index, ScrollStrategy::Center);
    }
}

/// A row of the tree: a section's title, or an entity's disclosure triangle,
/// its title followed by its rarest components, and its id at the right end.
fn tree_row(tree: &TreeSnapshot, row: &RowView, focused: bool, cx: &App) -> AnyElement {
    let theme = cx.theme();
    match *row {
        RowView::Section {
            section,
            roots,
            open,
        } => h_flex()
            .id(("entities-section", section as usize))
            .h(ROW_HEIGHT)
            .w_full()
            .pr_1p5()
            .bg(theme.secondary)
            .cursor_pointer()
            .child(disclosure(open, theme))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        div()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(section.title()),
                    )
                    .child(
                        div()
                            .text_color(theme.muted_foreground)
                            .child(roots.to_string()),
                    ),
            )
            .on_click(move |_, _, cx| send(cx, Action::ToggleSection(section)))
            .tooltip(move |window, cx| Tooltip::new(section.description()).build(window, cx))
            .into_any_element(),
        RowView::Entity {
            entity,
            depth,
            children,
            open,
            ref label,
            matched,
        } => {
            let selected = tree.selected == Some(entity);
            // Gray, as Chrome's, while the keyboard focus is elsewhere.
            let background = match (selected, focused) {
                (true, true) => theme.list_active,
                (true, false) => theme.muted,
                (false, _) => gpui::transparent_black(),
            };
            let hover = theme.list_hover;
            let toggle = move |recursive: bool, cx: &App| {
                send(
                    cx,
                    Action::SetExpanded {
                        entity,
                        open: !open,
                        recursive,
                    },
                )
            };
            let triangle = match children {
                0 => div().w(INDENT).flex_shrink_0().into_any_element(),
                _ => disclosure(open, theme)
                    .id(("entities-toggle", entity.to_bits()))
                    .cursor_pointer()
                    .on_click(move |event, _, cx| {
                        cx.stop_propagation();
                        // ⌥-click expands or collapses the whole subtree, as
                        // in Chrome.
                        toggle(event.modifiers().alt, cx);
                    })
                    .into_any_element(),
            };
            let (tooltip_label, menu_label) = (label.clone(), label.clone());
            h_flex()
                .id(("entities-row", entity.to_bits()))
                .h(ROW_HEIGHT)
                .w_full()
                .pl(INDENT * depth as f32)
                .pr_1p5()
                .bg(background)
                .when(!selected, |row| row.hover(move |row| row.bg(hover)))
                .child(triangle)
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .child(label_text(label, !matched, theme)),
                )
                .child(
                    div()
                        .flex_shrink_0()
                        .pl_2()
                        .text_xs()
                        .font_family(theme.mono_font_family.clone())
                        .text_color(theme.muted_foreground)
                        .child(entity.to_string()),
                )
                .on_click(move |event, _, cx| {
                    if children > 0 && event.click_count() == 2 {
                        toggle(event.modifiers().alt, cx);
                    } else {
                        send(cx, Action::Select(entity));
                    }
                })
                .tooltip(move |window, cx| label_tooltip(&tooltip_label, window, cx))
                .context_menu(move |mut menu, _, _| {
                    if children > 0 {
                        menu = menu
                            .item(
                                PopupMenuItem::new("Expand recursively")
                                    .on_click(move |_, _, cx| toggle_all(entity, true, cx)),
                            )
                            .item(
                                PopupMenuItem::new("Collapse children")
                                    .on_click(move |_, _, cx| toggle_all(entity, false, cx)),
                            )
                            .separator();
                    }
                    entity_menu(menu, entity, &menu_label)
                })
                .into_any_element()
        }
    }
}

/// Expands or collapses `entity` and all of its descendants.
fn toggle_all(entity: Entity, open: bool, cx: &App) {
    send(
        cx,
        Action::SetExpanded {
            entity,
            open,
            recursive: true,
        },
    );
}

/// The actions on `entity` offered both in the tree's context menu and in the
/// inspector's ⋮ menu.
fn entity_menu(mut menu: PopupMenu, entity: Entity, label: &Label) -> PopupMenu {
    menu = menu.item(PopupMenuItem::new("Copy ID").on_click(move |_, _, cx| {
        cx.write_to_clipboard(ClipboardItem::new_string(entity.to_string()))
    }));
    if let Some(title) = label.title.clone() {
        menu = menu.item(PopupMenuItem::new("Copy title").on_click(move |_, _, cx| {
            cx.write_to_clipboard(ClipboardItem::new_string(title.to_string()))
        }));
    }
    // Resources, observers and one-shot systems aren't meant to be despawned.
    if label.section != Section::Entities {
        return menu;
    }
    let disabled = label.has_disabled;
    menu.separator()
        .item(
            // Adding `Disabled` hides the entity and its descendants from
            // queries.
            PopupMenuItem::new(if disabled { "Enable" } else { "Disable" })
                .on_click(move |_, _, cx| send(cx, Action::SetDisabled(entity, !disabled))),
        )
        .item(
            PopupMenuItem::element(|_, cx| div().text_color(cx.theme().danger).child("Despawn"))
                .on_click(move |_, _, cx| send(cx, Action::Despawn(entity))),
        )
}

/// The inspector's title: the entity's title, or its id, and its ⋮ menu.
fn title(view: &EntityView, theme: &Theme) -> impl IntoElement + use<> {
    let title = match &view.label.title {
        Some(title) => div()
            .text_color(Syntax::Tag.color(theme))
            .child(title.clone()),
        None => div()
            .font_family(theme.mono_font_family.clone())
            .child(view.entity.to_string()),
    };
    let entity = view.entity;
    let label = view.label.clone();
    h_flex()
        .flex_1()
        .min_w_0()
        .gap_1()
        .child(
            div()
                .flex_1()
                .min_w_0()
                .truncate()
                .text_base()
                .font_weight(FontWeight::SEMIBOLD)
                .child(title),
        )
        .child(
            Button::new("entities-entity-menu")
                .icon(IconName::EllipsisVertical)
                .ghost()
                .xsmall()
                .tooltip("Entity actions")
                .dropdown_menu(move |menu, _, _| entity_menu(menu, entity, &label)),
        )
}

/// The entity's id and kind, where it's stored, and when it was spawned.
fn details(view: &EntityView, theme: &Theme) -> impl IntoElement + use<> {
    let label = &view.label;
    let badges: Vec<&'static str> = label
        .section
        .kind()
        .into_iter()
        .chain(label.disabled.then_some("Disabled"))
        .collect();
    v_flex()
        .flex_shrink_0()
        .gap_0p5()
        .px_2()
        .py_1()
        .text_xs()
        .text_color(theme.muted_foreground)
        .when(label.title.is_some() || !badges.is_empty(), |details| {
            details.child(
                h_flex()
                    .flex_wrap()
                    .gap_1()
                    // The title takes the id's place when there is no title.
                    .when(label.title.is_some(), |row| {
                        row.child(
                            div()
                                .font_family(theme.mono_font_family.clone())
                                .child(view.entity.to_string()),
                        )
                    })
                    .children(badges.into_iter().map(|badge| {
                        div()
                            .px_1()
                            .rounded_sm()
                            .bg(theme.secondary)
                            .text_color(theme.secondary_foreground)
                            .child(badge)
                    })),
            )
        })
        .child(view.archetype.clone())
        .child(
            div()
                .id("entities-spawned")
                .child(view.spawned.clone())
                .when(!view.spawned_by, |spawned| {
                    spawned.tooltip(|window, cx| {
                        Tooltip::new("Bevy's `track_location` feature records where, too.")
                            .build(window, cx)
                    })
                }),
        )
}

/// A link to an entity, labeled with its title or rarest component and its
/// id. Clicking it selects the entity.
fn entity_link(link: &Link, id: ElementId, theme: &Theme) -> AnyElement {
    let entity = link.entity;
    let Some(label) = link.label.clone() else {
        return div()
            .text_color(theme.danger)
            .child(format!("{entity} (despawned)"))
            .into_any_element();
    };
    div()
        .id(id)
        .text_color(theme.link)
        .cursor_pointer()
        .hover(|link| link.underline())
        .child(format!("{} {entity}", label.short()))
        .on_click(move |_, _, cx| {
            cx.stop_propagation();
            send(cx, Action::Select(entity));
        })
        .tooltip(move |window, cx| label_tooltip(&label, window, cx))
        .into_any_element()
}

/// The enum's variant, picked from a menu when editable. A variant whose
/// fields have no reflected `Default` can't be built, and is disabled.
fn variant_picker(
    (entity, component): (Entity, ComponentId),
    variant: &Variant,
    ids: &mut Ids,
    theme: &Theme,
) -> AnyElement {
    let Some(choices) = variant.choices.clone() else {
        return value_text(variant.current.to_string(), Syntax::Value, theme);
    };
    let current = variant.current.clone();
    let path = variant.path.clone();
    let picker = Button::new(ids.next())
        .label(current.clone())
        .ghost()
        .xsmall()
        .dropdown_menu(move |mut menu, _, _| {
            for (name, buildable) in &choices {
                let picked = *name == current;
                let mut item = PopupMenuItem::new(name.clone())
                    .checked(picked)
                    .disabled(!buildable && !picked);
                // Picking the variant shown would reset its fields.
                if !picked {
                    let (name, path) = (name.clone(), path.clone());
                    item = item.on_click(move |_, _, cx| {
                        let value = Edit::Variant(name.clone());
                        send(cx, edit_action(entity, component, path.clone(), value));
                    });
                }
                menu = menu.item(item);
            }
            if choices.iter().any(|&(_, buildable)| !buildable) {
                menu = menu
                    .separator()
                    .label("Some variants' fields don't reflect `Default`.");
            }
            menu
        });
    div()
        .id(ids.next())
        // A picker within a collapsible row leaves it as it is.
        .on_click(|_, _, cx| cx.stop_propagation())
        .child(picker)
        .into_any_element()
}

fn edit_action(entity: Entity, component: ComponentId, path: Vec<usize>, value: Edit) -> Action {
    Action::Edit {
        entity,
        component,
        path,
        value,
    }
}

/// ▶ when closed, ▼ when open, in a column of the indentation's width.
fn disclosure(open: bool, theme: &Theme) -> gpui::Div {
    h_flex()
        .w(INDENT)
        .h_full()
        .flex_shrink_0()
        .justify_center()
        .child(
            Icon::new(if open {
                IconName::ChevronDown
            } else {
                IconName::ChevronRight
            })
            .xsmall()
            .text_color(theme.muted_foreground),
        )
}

/// A field's name, before its value.
fn field_name(name: &SharedString, theme: &Theme) -> impl IntoElement + use<> {
    div()
        .flex_shrink_0()
        .text_color(Syntax::Attribute.color(theme))
        .child(format!("{name}:"))
}

fn value_text(text: String, syntax: Syntax, theme: &Theme) -> AnyElement {
    div()
        .text_color(syntax.color(theme))
        .child(text)
        .into_any_element()
}

/// The entity's title and all of its components.
fn label_tooltip(label: &Label, window: &mut Window, cx: &mut App) -> AnyView {
    let title = label.title.clone();
    let components = label.components.join(", ");
    Tooltip::element(move |_, _| {
        v_flex()
            .children(
                title
                    .clone()
                    .map(|title| div().font_weight(FontWeight::SEMIBOLD).child(title)),
            )
            .child(components.clone())
    })
    .build(window, cx)
}

/// The entity's title in the tag color, followed by its rarest components in
/// the attribute color, as Chrome shows an element's tag and attributes.
/// Disabled entities are in italics. With `dimmed`, all of it is weak.
fn label_text(label: &Label, dimmed: bool, theme: &Theme) -> StyledText {
    let weak = theme.muted_foreground;
    let (tag, attribute) = if dimmed || label.disabled {
        (weak, weak)
    } else {
        (Syntax::Tag.color(theme), Syntax::Attribute.color(theme))
    };
    let mut text = Spans {
        italic: label.disabled,
        ..Default::default()
    };
    if let Some(title) = &label.title {
        text.push(title, tag);
    }
    for name in label.components.iter().take(ROW_COMPONENTS) {
        text.push(name, attribute);
    }
    let more = label.components.len().saturating_sub(ROW_COMPONENTS);
    if more > 0 {
        text.push(&format!("+{more}"), weak);
    }
    if text.text.is_empty() {
        text.push("empty", weak);
    }
    text.build()
}

/// Words in colors of their own, separated by spaces.
#[derive(Default)]
struct Spans {
    text: String,
    highlights: Vec<(std::ops::Range<usize>, HighlightStyle)>,
    italic: bool,
}

impl Spans {
    fn push(&mut self, word: &str, color: Hsla) {
        if !self.text.is_empty() {
            self.text.push(' ');
        }
        let start = self.text.len();
        self.text.push_str(word);
        self.highlights.push((
            start..self.text.len(),
            HighlightStyle {
                color: Some(color),
                font_style: self.italic.then_some(FontStyle::Italic),
                ..Default::default()
            },
        ));
    }

    fn build(self) -> StyledText {
        StyledText::new(self.text).with_highlights(self.highlights)
    }
}

/// The colors of the DOM tree in Chrome's DevTools, which the tab borrows.
#[derive(Clone, Copy)]
enum Syntax {
    /// A tag's name; an entity's title here.
    Tag,
    /// An attribute's name; a component's or a field's here.
    Attribute,
    /// An attribute's value, which is a string; any string here.
    Value,
    /// A number or a boolean, as Chrome's console colors them.
    Number,
}

impl Syntax {
    fn color(self, theme: &Theme) -> Hsla {
        let hex = match (self, theme.mode.is_dark()) {
            (Syntax::Tag, true) => 0x5db0d7,
            (Syntax::Tag, false) => 0x881280,
            (Syntax::Attribute, true) => 0x9bbbdc,
            (Syntax::Attribute, false) => 0x994500,
            (Syntax::Value, true) => 0xf29766,
            (Syntax::Value, false) => 0x1a1aa6,
            (Syntax::Number, true) => 0x9980ff,
            (Syntax::Number, false) => 0x1c00cf,
        };
        rgb(hex).into()
    }
}

/// Element ids for the inspector's controls, numbered in the order they're
/// drawn, which stays the same from frame to frame while the inspector shows
/// the same rows.
#[derive(Default)]
struct Ids(usize);

impl Ids {
    fn next(&mut self) -> ElementId {
        self.0 += 1;
        ("entities-value", self.0).into()
    }
}

/// Where an edited value is.
#[derive(Clone, PartialEq)]
struct Place {
    entity: Entity,
    component: ComponentId,
    /// The path to the value within the component, as
    /// [`element_mut`](super::value::element_mut) follows it.
    path: Vec<usize>,
    /// Which Euler angle, for a rotation.
    axis: Option<usize>,
}

#[derive(Clone)]
struct EditTarget {
    place: Place,
    kind: EditKind,
}

/// How an edited value's text becomes the new value.
#[derive(Clone, Copy)]
enum EditKind {
    /// A number of the same type.
    Number(Number),
    /// A `String`.
    Text,
    /// A `Duration`, in seconds.
    Seconds,
    /// This rotation, with the angle at the place's axis set to the degrees
    /// in the text.
    Angle(Quat),
    /// The entity's `Name`.
    Name,
}

impl EditTarget {
    /// The edit setting the value to the one `text` spells; `None` if it
    /// spells none.
    fn action(&self, text: &str) -> Option<Action> {
        let text = text.trim();
        let value: Box<dyn PartialReflect> = match self.kind {
            EditKind::Number(number) => number.parse(text)?.into_reflect(),
            EditKind::Text => Box::new(text.to_owned()),
            EditKind::Seconds => {
                let seconds = text.trim_end_matches('s').trim().parse().ok()?;
                Box::new(Duration::try_from_secs_f64(seconds).ok()?)
            }
            EditKind::Angle(rotation) => {
                let degrees = text.trim_end_matches('°').trim().parse().ok()?;
                Box::new(self.with_angle(rotation, degrees))
            }
            EditKind::Name => return Some(Action::Rename(self.place.entity, text.to_owned())),
        };
        Some(self.edit(value))
    }

    /// The value a drag starts from, and how much a pixel of dragging changes
    /// it; `None` for a value that can't be dragged.
    fn scrub_start(&self) -> Option<(f64, f64)> {
        match self.kind {
            EditKind::Number(number) => {
                let value = number.to_f64();
                // A float moves by a hundredth of its magnitude, so that small
                // and large values both edit smoothly.
                let speed = match number.is_integer() {
                    true => 0.25,
                    false => (value.abs() * 0.01).max(0.001),
                };
                Some((value, speed))
            }
            EditKind::Angle(rotation) => {
                let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
                let angle = [x, y, z][self.place.axis?].to_degrees();
                Some((angle.into(), 0.5))
            }
            EditKind::Text | EditKind::Seconds | EditKind::Name => None,
        }
    }

    /// The edit setting the value to `value`, dragged to.
    fn scrubbed(&self, value: f64) -> Option<Action> {
        let value: Box<dyn PartialReflect> = match self.kind {
            EditKind::Number(number) => number.with_f64(value).into_reflect(),
            EditKind::Angle(rotation) => Box::new(self.with_angle(rotation, value as f32)),
            EditKind::Text | EditKind::Seconds | EditKind::Name => return None,
        };
        Some(self.edit(value))
    }

    /// `rotation` with the Euler angle at the place's axis set to `degrees`.
    fn with_angle(&self, rotation: Quat, degrees: f32) -> Quat {
        let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
        let mut angles = [x, y, z];
        if let Some(angle) = self.place.axis.and_then(|axis| angles.get_mut(axis)) {
            *angle = degrees.to_radians();
        }
        let [x, y, z] = angles;
        Quat::from_euler(EulerRot::XYZ, x, y, z)
    }

    fn edit(&self, value: Box<dyn PartialReflect>) -> Action {
        let place = &self.place;
        edit_action(
            place.entity,
            place.component,
            place.path.clone(),
            Edit::Value(value),
        )
    }
}

/// The value being edited in a field.
struct Editing {
    target: EditTarget,
    input: gpui::Entity<InputState>,
    _subscription: Subscription,
}

/// The value of a drag changing a number, from `start` by `speed` per pixel.
struct Scrub {
    target: EditTarget,
    start: f64,
    speed: f64,
    /// Where the mouse was when the drag first moved.
    origin: Cell<Option<f64>>,
    /// The value last sent.
    last: Cell<f64>,
}

/// What a drag changing a number shows under the mouse: nothing, as the
/// number itself changes.
struct ScrubPreview;

impl Render for ScrubPreview {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        gpui::Empty
    }
}
