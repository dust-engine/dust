//! The "Entities" DevTools tab, modeled on the Elements panel of Chrome's
//! DevTools: a tree of the world's entities, and an inspector listing the
//! components of the entity selected in it.
//!
//! The tree nests entities by `ChildOf` and groups the roots into sections:
//! resources, which Bevy stores on entities of their own; observers; one-shot
//! systems; and every other entity. The inspector reads component values
//! through reflection and edits the values of mutable components.
//!
//! Drawing borrows the world immutably, so that any entity can be looked up
//! while drawing, e.g. to label a link to it. What the user does is collected
//! as [`Action`]s, which [`apply`] carries out once drawing is done.

mod inspector;
#[cfg(test)]
mod tests;
mod tree;
mod value;

use bevy::ecs::archetype::Archetype;
use bevy::ecs::component::{ComponentId, Components};
use bevy::ecs::entity_disabling::{DefaultQueryFilters, Disabled};
use bevy::ecs::observer::Observer;
use bevy::ecs::resource::{IS_RESOURCE, IsResource};
use bevy::ecs::system::SystemIdMarker;
use bevy::prelude::*;
use bevy::reflect::{PartialReflect, ReflectFromPtr, TypeRegistry};
use pumicite_egui::egui;

use crate::devtools::DevToolsAppExt;

/// Adds the "Entities" DevTools tab.
#[derive(Default)]
pub struct EntitiesPlugin;

impl Plugin for EntitiesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EntityInspector>();
        app.add_devtools_tab("Entities", entities_ui);
    }
}

/// State of the "Entities" tab.
#[derive(Resource, Default)]
pub struct EntityInspector {
    /// The entity the inspector shows. Whenever it changes, from the tab or
    /// from elsewhere, the tree expands down to the entity and scrolls it into
    /// view.
    pub selected: Option<Entity>,
    /// `selected` when the tab was last drawn.
    previous: Option<Entity>,
    history: History,
    names: ComponentNames,
    tree: tree::TreeState,
    inspector: inspector::InspectorState,
}

fn entities_ui(InMut(ui): InMut<egui::Ui>, world: &mut World) {
    world.resource_scope(|world, mut state: Mut<EntityInspector>| {
        let state = &mut *state;
        let selection_changed = state.selected != state.previous;
        if selection_changed {
            if let Some(entity) = state.selected {
                state.history.visit(entity);
            }
            state.previous = state.selected;
        }
        state.names.update(world.components());
        // A handle of its own, so that the registry stays reachable while
        // `apply` mutates the world.
        let registry = world.get_resource::<AppTypeRegistry>().cloned();
        let read = registry.as_ref().map(|registry| registry.read());

        let mut actions = Vec::new();
        if ui.rect_contains_pointer(ui.max_rect()) {
            // The mouse's back and forward buttons, as in a browser.
            ui.input(|input| {
                if input.pointer.button_pressed(egui::PointerButton::Extra1) {
                    actions.push(Action::Back);
                }
                if input.pointer.button_pressed(egui::PointerButton::Extra2) {
                    actions.push(Action::Forward);
                }
            });
        }
        let info = WorldInfo::new(world, &state.names);
        // Side by side when there is more width than height, stacked
        // otherwise, as Chrome lays out the Elements panel.
        let tree_panel = if ui.available_width() > ui.available_height() {
            egui::Panel::left("devtools_entities_tree_left")
                .default_size(ui.available_width() * 0.45)
        } else {
            egui::Panel::top("devtools_entities_tree_top")
                .default_size(ui.available_height() * 0.45)
        };
        tree_panel
            .resizable(true)
            .size_range(120.0..=f32::INFINITY)
            .show(ui, |ui| {
                tree::tree_ui(
                    ui,
                    &info,
                    &mut state.selected,
                    &mut state.tree,
                    selection_changed,
                    &mut actions,
                );
            });
        egui::CentralPanel::default().show(ui, |ui| {
            inspector::inspector_ui(
                ui,
                &info,
                read.as_deref(),
                state.selected,
                &mut state.inspector,
                &state.history,
                &mut actions,
            );
        });
        // Despawning and inserting run hooks and observers, which may want to
        // write the registry.
        drop(read);
        apply(world, state, registry.as_ref(), actions);
    });
}

/// Something the user did while the tab was drawn, carried out by [`apply`]
/// afterwards.
enum Action {
    Select(Entity),
    Back,
    Forward,
    /// Show only the tree's entities matching this text.
    Filter(String),
    /// Set the value at `path` within `entity`'s `component`; `path` is as
    /// [`value::element_mut`] follows it.
    Edit {
        entity: Entity,
        component: ComponentId,
        path: Vec<usize>,
        value: Box<dyn PartialReflect>,
    },
    Rename(Entity, String),
    /// Add `Disabled` to (`true`) or remove it from (`false`) the entity and
    /// its descendants.
    SetDisabled(Entity, bool),
    Despawn(Entity),
    RemoveComponent(Entity, ComponentId),
}

fn apply(
    world: &mut World,
    state: &mut EntityInspector,
    registry: Option<&AppTypeRegistry>,
    actions: Vec<Action>,
) {
    for action in actions {
        match action {
            Action::Select(entity) => state.selected = Some(entity),
            Action::Back => {
                if let Some(entity) = state.history.back() {
                    state.selected = Some(entity);
                }
            }
            Action::Forward => {
                if let Some(entity) = state.history.forward() {
                    state.selected = Some(entity);
                }
            }
            Action::Filter(text) => state.tree.filter = text,
            Action::Edit {
                entity,
                component,
                path,
                value,
            } => {
                if let Some(registry) = registry {
                    edit(world, &registry.read(), entity, component, &path, &*value);
                }
            }
            Action::Rename(entity, name) => {
                if let Some(mut current) = world.get_mut::<Name>(entity) {
                    current.set(name);
                }
            }
            Action::SetDisabled(entity, disabled) => {
                if let Ok(mut entity) = world.get_entity_mut(entity) {
                    if disabled {
                        entity.insert_recursive::<Children>(Disabled);
                    } else {
                        entity.remove_recursive::<Children, Disabled>();
                    }
                }
            }
            Action::Despawn(entity) => {
                let _ = world.try_despawn(entity);
            }
            Action::RemoveComponent(entity, component) => {
                if let Ok(mut entity) = world.get_entity_mut(entity) {
                    entity.remove_by_id(component);
                }
            }
        }
    }
}

/// Applies `value` at `path` within `entity`'s `component`, and marks the
/// component changed if that worked.
fn edit(
    world: &mut World,
    registry: &TypeRegistry,
    entity: Entity,
    component: ComponentId,
    path: &[usize],
    value: &dyn PartialReflect,
) {
    let Some(type_id) = world
        .components()
        .get_info(component)
        .and_then(|info| info.type_id())
    else {
        return;
    };
    let Some(from_ptr) = registry
        .get_type_data::<ReflectFromPtr>(type_id)
        .filter(|from_ptr| from_ptr.type_id() == type_id)
    else {
        return;
    };
    let Ok(mut entity) = world.get_entity_mut(entity) else {
        return;
    };
    // Fails for immutable components.
    let Ok(mut component) = entity.get_mut_by_id(component) else {
        return;
    };
    // SAFETY: the pointer is to the component, of type `type_id`, which
    // `from_ptr` was checked to be for.
    let reflect =
        unsafe { from_ptr.ptr_as_reflect_mut(component.bypass_change_detection().reborrow()) };
    let applied = value::element_mut(reflect.as_partial_reflect_mut(), path)
        .is_some_and(|target| target.try_apply(value).is_ok());
    if applied {
        component.set_changed();
    }
}

/// The entities selected in turn, for the back and forward buttons.
#[derive(Default)]
struct History {
    entities: Vec<Entity>,
    /// Index into `entities` of the entry shown.
    position: usize,
}

impl History {
    /// Makes `entity` the entry shown, unless it already is, dropping the
    /// entries after the one shown as a browser does on following a link.
    fn visit(&mut self, entity: Entity) {
        if self.entities.get(self.position) == Some(&entity) {
            return;
        }
        self.entities.truncate(self.position + 1);
        self.entities.push(entity);
        self.position = self.entities.len() - 1;
    }

    fn back(&mut self) -> Option<Entity> {
        self.position = self.position.checked_sub(1)?;
        Some(self.entities[self.position])
    }

    fn forward(&mut self) -> Option<Entity> {
        let entity = *self.entities.get(self.position + 1)?;
        self.position += 1;
        Some(entity)
    }

    fn can_go_back(&self) -> bool {
        self.position > 0
    }

    fn can_go_forward(&self) -> bool {
        self.position + 1 < self.entities.len()
    }
}

/// Short type names of the registered components, such as `Vec<Entity>` for
/// `alloc::vec::Vec<bevy_ecs::entity::Entity>`, indexed by
/// `ComponentId::index`.
#[derive(Default)]
struct ComponentNames(Vec<String>);

impl ComponentNames {
    /// Adds the names of the components registered since the last call.
    /// Components are never unregistered, so the names already there stay
    /// valid.
    fn update(&mut self, components: &Components) {
        for index in self.0.len()..components.num_registered() {
            // A component queued for registration has an id but no info yet.
            // Its name, and the ones after it, are added once it's registered.
            let Some(info) = components.get_info(ComponentId::new(index)) else {
                break;
            };
            self.0.push(info.name().shortname().to_string());
        }
    }
}

/// The sections grouping the tree's root entities, in the order they show.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Section {
    Resources,
    Observers,
    Systems,
    Entities,
}

impl Section {
    const ALL: [Section; 4] = [
        Section::Resources,
        Section::Observers,
        Section::Systems,
        Section::Entities,
    ];

    fn title(self) -> &'static str {
        match self {
            Section::Resources => "Resources",
            Section::Observers => "Observers",
            Section::Systems => "One-shot systems",
            Section::Entities => "Entities",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Section::Resources => "Each resource is stored on an entity of its own.",
            Section::Observers => "Entities holding an `Observer`.",
            Section::Systems => "Systems registered with `World::register_system`.",
            Section::Entities => "All other entities, nested by `ChildOf`.",
        }
    }
}

/// Facts about the world that the tree and the inspector look up while
/// drawing.
struct WorldInfo<'w> {
    world: &'w World,
    names: &'w ComponentNames,
    /// How many entities have each component, indexed by
    /// `ComponentId::index`.
    counts: Vec<u32>,
    entity_count: u32,
    known: KnownComponents,
    /// The components that disable the entities they are on, as `Disabled`
    /// does.
    disabling: Vec<ComponentId>,
}

/// Ids of the components the tab treats specially; `None` for one that isn't
/// registered.
struct KnownComponents {
    name: Option<ComponentId>,
    child_of: Option<ComponentId>,
    children: Option<ComponentId>,
    observer: Option<ComponentId>,
    system: Option<ComponentId>,
}

impl<'w> WorldInfo<'w> {
    fn new(world: &'w World, names: &'w ComponentNames) -> Self {
        let mut counts = vec![0; world.components().num_registered()];
        let mut entity_count = 0;
        for archetype in world.archetypes().iter() {
            entity_count += archetype.len();
            for id in archetype.iter_components() {
                if let Some(count) = counts.get_mut(id.index()) {
                    *count += archetype.len();
                }
            }
        }
        Self {
            world,
            names,
            counts,
            entity_count,
            known: KnownComponents {
                name: world.component_id::<Name>(),
                child_of: world.component_id::<ChildOf>(),
                children: world.component_id::<Children>(),
                observer: world.component_id::<Observer>(),
                system: world.component_id::<SystemIdMarker>(),
            },
            disabling: world
                .get_resource::<DefaultQueryFilters>()
                .map(|filters| filters.disabling_ids().collect())
                .unwrap_or_default(),
        }
    }

    /// The component's short type name.
    fn name(&self, id: ComponentId) -> &'w str {
        self.names.0.get(id.index()).map_or("?", String::as_str)
    }

    fn section(&self, archetype: &Archetype) -> Section {
        let has = |id: Option<ComponentId>| id.is_some_and(|id| archetype.contains(id));
        if archetype.contains(IS_RESOURCE) {
            Section::Resources
        } else if has(self.known.observer) {
            Section::Observers
        } else if has(self.known.system) {
            Section::Systems
        } else {
            Section::Entities
        }
    }

    fn parent(&self, entity: Entity) -> Option<Entity> {
        self.world.get::<ChildOf>(entity).map(ChildOf::parent)
    }

    /// `entity`'s parent, grandparent, and so on. Gives up after a thousand
    /// levels, in case `ChildOf` forms a cycle.
    fn ancestors(&self, entity: Entity) -> impl Iterator<Item = Entity> + '_ {
        std::iter::successors(self.parent(entity), |&entity| self.parent(entity)).take(1000)
    }

    fn children(&self, entity: Entity) -> &'w [Entity] {
        self.world
            .get::<Children>(entity)
            .map_or(&[], |children| &**children)
    }

    /// How `entity` shows in the tree, in links, and in the inspector's title;
    /// `None` if it isn't spawned.
    fn label(&self, entity: Entity) -> Option<Label> {
        let entity = self.world.get_entity(entity).ok()?;
        let archetype = entity.archetype();
        let section = self.section(archetype);
        let resource = entity
            .get::<IsResource>()
            .map(IsResource::resource_component_id);
        let title = entity
            .get::<Name>()
            .filter(|name| !name.as_str().is_empty())
            .map(|name| name.as_str().to_owned())
            .or_else(|| resource.map(|id| self.name(id).to_owned()))
            .or_else(|| {
                let observer = entity.get::<Observer>()?;
                Some(observer.system_name().shortname().to_string())
            });
        // Without the components the title or the tree already shows.
        let shown = [
            self.known.name,
            self.known.child_of,
            self.known.children,
            self.known.observer,
            Some(IS_RESOURCE),
            resource,
        ];
        let mut components: Vec<ComponentId> = archetype
            .iter_components()
            .filter(|&id| !shown.contains(&Some(id)))
            .collect();
        components.sort_by_key(|&id| (self.counts.get(id.index()).copied().unwrap_or(0), id));
        Some(Label {
            title,
            components,
            section,
            disabled: self.disabling.iter().any(|&id| archetype.contains(id)),
        })
    }
}

/// What identifies an entity in the tree, in links to it, and in the
/// inspector's title.
struct Label {
    /// The entity's `Name`, or else the type of its resource or the system of
    /// its observer.
    title: Option<String>,
    /// The entity's other components, those on the fewest entities first:
    /// the rarer a component, the better it tells the entity apart.
    components: Vec<ComponentId>,
    section: Section,
    /// Whether the entity has a component that disables it, such as
    /// `Disabled`.
    disabled: bool,
}

impl Label {
    /// The title, or else the rarest component's name.
    fn short<'a>(&'a self, info: &WorldInfo<'a>) -> &'a str {
        match (&self.title, self.components.first()) {
            (Some(title), _) => title,
            (None, Some(&id)) => info.name(id),
            (None, None) => "empty",
        }
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
    fn color(self, visuals: &egui::Visuals) -> egui::Color32 {
        let [r, g, b] = match (self, visuals.dark_mode) {
            (Syntax::Tag, true) => [0x5d, 0xb0, 0xd7],
            (Syntax::Tag, false) => [0x88, 0x12, 0x80],
            (Syntax::Attribute, true) => [0x9b, 0xbb, 0xdc],
            (Syntax::Attribute, false) => [0x99, 0x45, 0x00],
            (Syntax::Value, true) => [0xf2, 0x97, 0x66],
            (Syntax::Value, false) => [0x1a, 0x1a, 0xa6],
            (Syntax::Number, true) => [0x99, 0x80, 0xff],
            (Syntax::Number, false) => [0x1c, 0x00, 0xcf],
        };
        egui::Color32::from_rgb(r, g, b)
    }
}

/// A link to `entity`, labeled with its title or rarest component and its id.
/// Clicking it selects the entity.
fn entity_link(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    entity: Entity,
    actions: &mut Vec<Action>,
) -> egui::Response {
    let Some(label) = info.label(entity) else {
        let error = ui.visuals().error_fg_color;
        return ui.label(egui::RichText::new(format!("{entity} (despawned)")).color(error));
    };
    let response = ui.link(format!("{} {entity}", label.short(info)));
    if response.clicked() {
        actions.push(Action::Select(entity));
    }
    response.on_hover_ui(|ui| label_tooltip(ui, info, &label))
}

/// The entity's title and all of its components.
fn label_tooltip(ui: &mut egui::Ui, info: &WorldInfo, label: &Label) {
    if let Some(title) = &label.title {
        ui.strong(title);
    }
    let components: Vec<&str> = label.components.iter().map(|&id| info.name(id)).collect();
    ui.label(components.join(", "));
}

/// The actions on `entity` offered both in the tree's context menu and in the
/// inspector's ⋮ menu.
fn entity_menu(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    entity: Entity,
    label: &Label,
    actions: &mut Vec<Action>,
) {
    if ui.button("Copy ID").clicked() {
        ui.ctx().copy_text(entity.to_string());
    }
    if let Some(title) = &label.title
        && ui.button("Copy title").clicked()
    {
        ui.ctx().copy_text(title.clone());
    }
    // Resources, observers and one-shot systems aren't meant to be despawned.
    if label.section != Section::Entities {
        return;
    }
    ui.separator();
    if info.world.get::<Disabled>(entity).is_some() {
        if ui
            .button("Enable")
            .on_hover_text("Removes `Disabled` from the entity and its descendants.")
            .clicked()
        {
            actions.push(Action::SetDisabled(entity, false));
        }
    } else if ui
        .button("Disable")
        .on_hover_text(
            "Adds `Disabled` to the entity and its descendants, hiding them from queries.",
        )
        .clicked()
    {
        actions.push(Action::SetDisabled(entity, true));
    }
    let error = ui.visuals().error_fg_color;
    if ui
        .button(egui::RichText::new("Despawn").color(error))
        .clicked()
    {
        actions.push(Action::Despawn(entity));
    }
}

/// Whether `needle` is part of `haystack`, ignoring ASCII case.
fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    needle.is_empty()
        || haystack
            .as_bytes()
            .windows(needle.len())
            .any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Width of a disclosure triangle's column, which is also the indentation of
/// each level of the tree.
const INDENT: f32 = 14.0;

/// ▶ when closed, ▼ when open, as the console's disclosure triangles.
fn paint_triangle(painter: &egui::Painter, center: egui::Pos2, open: bool, color: egui::Color32) {
    let v = egui::vec2;
    let points = if open {
        vec![
            center + v(-4.0, -2.0),
            center + v(4.0, -2.0),
            center + v(0.0, 3.0),
        ]
    } else {
        vec![
            center + v(-2.0, -4.0),
            center + v(3.0, 0.0),
            center + v(-2.0, 4.0),
        ]
    };
    painter.add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::NONE,
    ));
}

/// A row of the inspector's width, filled with `background`, with a
/// disclosure triangle on the left when `open` is `Some`, and `header` drawn
/// after the triangle. Clicks on the row, outside the header's own widgets,
/// go to the returned response.
fn header_row(
    ui: &mut egui::Ui,
    open: Option<bool>,
    background: egui::Color32,
    header: impl FnOnce(&mut egui::Ui),
) -> egui::Response {
    let height = ui.spacing().interact_size.y;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click(),
    );
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(2), background);
    if let Some(open) = open {
        let color = if response.hovered() {
            ui.visuals().strong_text_color()
        } else {
            ui.visuals().weak_text_color()
        };
        let center = egui::pos2(rect.left() + INDENT / 2.0, rect.center().y);
        paint_triangle(ui.painter(), center, open, color);
    }
    let mut header_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect.with_min_x(rect.left() + INDENT))
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    // Selectable labels would take the clicks meant to toggle the row.
    header_ui.style_mut().interaction.selectable_labels = false;
    header(&mut header_ui);
    response
}

/// A [`header_row`] whose clicks open and close it. Returns the row's response
/// and whether it's open, in which case the caller draws what the row holds
/// below it. The open state is kept in egui's memory under `id`, so it
/// persists while the row is out of view.
fn collapsible_row(
    ui: &mut egui::Ui,
    id: egui::Id,
    default_open: bool,
    background: egui::Color32,
    header: impl FnOnce(&mut egui::Ui),
) -> (egui::Response, bool) {
    let mut open = ui.data_mut(|data| *data.get_temp_mut_or(id, default_open));
    let response = header_row(ui, Some(open), background, header);
    if response.clicked() {
        open = !open;
        ui.data_mut(|data| data.insert_temp(id, open));
    }
    (response, open)
}
