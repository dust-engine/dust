//! The "Entities" DevTools tab, modeled on the Elements panel of Chrome's
//! DevTools: a tree of the world's entities, and an inspector listing the
//! components of the entity selected in it.
//!
//! The tree nests entities by `ChildOf` and groups the roots into sections:
//! resources, which Bevy stores on entities of their own; observers; one-shot
//! systems; and every other entity. The inspector reads component values
//! through reflection and edits the values of mutable components.
//!
//! Showing any of it takes the world, so the tab's state lives on the Bevy
//! thread, in the [`EntityInspector`] resource: the selection, the filters,
//! and what is expanded. While the tab is drawn, [`snapshot`] runs at the end
//! of every Bevy frame (see [`watch`](crate::watch)) and sends what the tab
//! shows, as plain data, to [`EntitiesView`] on the UI thread. What the user
//! does there goes back as [`Action`]s, which [`apply`] carries out on the
//! Bevy thread.

mod inspector;
#[cfg(test)]
mod tests;
mod tree;
mod ui;
mod value;

use std::time::Duration;

use bevy::ecs::archetype::{Archetype, ArchetypeId};
use bevy::ecs::component::{ComponentId, Components};
use bevy::ecs::entity_disabling::{DefaultQueryFilters, Disabled};
use bevy::ecs::observer::Observer;
use bevy::ecs::resource::{IS_RESOURCE, IsResource};
use bevy::ecs::system::SystemIdMarker;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::reflect::{PartialReflect, ReflectFromPtr, TypeRegistry};
use dust_ui::BevyChannel;
use gpui::SharedString;

pub(crate) use ui::EntitiesView;

/// Adds the [`EntityInspector`] the "Entities" DevTools tab shows the world
/// through. The tab itself is created on the UI thread, by
/// [`overlay`](crate::overlay).
#[derive(Default)]
pub struct EntitiesPlugin;

impl Plugin for EntitiesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EntityInspector>();
    }
}

/// State of the "Entities" tab.
#[derive(Resource, Default)]
pub struct EntityInspector {
    /// The entity the inspector shows. Whenever it changes, from the tab or
    /// from elsewhere, the tree expands down to the entity and scrolls it into
    /// view.
    pub selected: Option<Entity>,
    /// `selected` when the tab last showed it.
    previous: Option<Entity>,
    history: History,
    names: ComponentNames,
    tree: tree::TreeState,
    inspector: inspector::InspectorState,
}

/// Something the user did in the tab, carried out on the Bevy thread by
/// [`apply`].
enum Action {
    Select(Entity),
    Back,
    Forward,
    /// An arrow key, pressed while the tree has the keyboard focus.
    Key(tree::TreeKey),
    /// Show only the tree's entities matching this text.
    Filter(String),
    /// Show or hide a section's entities.
    ToggleSection(Section),
    /// Expand (`open`) or collapse the entity, and with `recursive`, its
    /// descendants.
    SetExpanded {
        entity: Entity,
        open: bool,
        recursive: bool,
    },
    /// Show only the inspector's components whose name contains this text.
    ComponentFilter(String),
    /// Show or hide a component's value.
    ToggleComponent(ComponentId),
    /// Open or close the value at a path within a component.
    ToggleValue(ComponentId, Vec<usize>),
    /// Set the value at `path` within `entity`'s `component`; `path` is as
    /// [`value::element_mut`] follows it.
    Edit {
        entity: Entity,
        component: ComponentId,
        path: Vec<usize>,
        value: Edit,
    },
    Rename(Entity, String),
    /// Add `Disabled` to (`true`) or remove it from (`false`) the entity and
    /// its descendants.
    SetDisabled(Entity, bool),
    Despawn(Entity),
    RemoveComponent(Entity, ComponentId),
}

/// A new value for [`Action::Edit`].
enum Edit {
    Value(Box<dyn PartialReflect>),
    /// The variant of this name of the enum edited, with its fields' default
    /// values.
    Variant(SharedString),
}

/// Has the Bevy thread carry out `action`.
fn send(cx: &gpui::App, action: Action) {
    cx.global::<BevyChannel>().update(move |world| {
        world.try_resource_scope(|world, mut state: Mut<EntityInspector>| {
            // A handle of its own, so that the registry stays reachable while
            // `apply` mutates the world.
            let registry = world.get_resource::<AppTypeRegistry>().cloned();
            apply(world, &mut state, registry.as_ref(), [action]);
        });
    });
}

fn apply(
    world: &mut World,
    state: &mut EntityInspector,
    registry: Option<&AppTypeRegistry>,
    actions: impl IntoIterator<Item = Action>,
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
            Action::Key(key) => {
                let info = WorldInfo::new(world, &state.names);
                state.tree.key(&info, key, &mut state.selected);
            }
            Action::Filter(text) => state.tree.filter = text,
            Action::ToggleSection(section) => state.tree.toggle_section(section),
            Action::SetExpanded {
                entity,
                open,
                recursive,
            } => {
                let info = WorldInfo::new(world, &state.names);
                state.tree.set_expanded(&info, entity, open, recursive);
            }
            Action::ComponentFilter(text) => state.inspector.filter = text,
            Action::ToggleComponent(component) => state.inspector.toggle_component(component),
            Action::ToggleValue(component, path) => state.inspector.toggle_value(component, path),
            Action::Edit {
                entity,
                component,
                path,
                value,
            } => {
                if let Some(registry) = registry {
                    edit(world, &registry.read(), entity, component, &path, &value);
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
    value: &Edit,
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
    let applied = value::element_mut(reflect.as_partial_reflect_mut(), path).is_some_and(
        |target| match value {
            Edit::Value(value) => target.try_apply(value.as_partial_reflect()).is_ok(),
            Edit::Variant(name) => value::named_variant(target, name, registry)
                .is_some_and(|variant| target.try_apply(&variant).is_ok()),
        },
    );
    if applied {
        component.set_changed();
    }
}

/// How often the tab is sent while nothing it watches changed, to catch up
/// with the changes it doesn't watch, such as entities spawned or renamed.
const REFRESH: Duration = Duration::from_millis(250);

/// What the tab shows, sent to the UI thread by [`snapshot`]. Each part is
/// `None` when it's unchanged since the last snapshot.
struct Snapshot {
    tree: Option<tree::TreeSnapshot>,
    inspector: Option<inspector::InspectorSnapshot>,
}

/// What [`snapshot`] last sent.
#[derive(Default)]
struct Sent {
    /// When the tree was last sent.
    tree: Option<Instant>,
    /// The entity the inspector last showed, and its archetype then.
    inspected: Option<(Entity, ArchetypeId)>,
}

/// The tab's contents, each Bevy frame while it's drawn: the tree when the
/// tab's state changed or [`REFRESH`] passed, and the inspector also when the
/// selected entity's components changed.
fn snapshot(world: &mut World, mut sent: Local<Sent>) -> Option<Snapshot> {
    let snapshot = world.try_resource_scope(|world, mut state: Mut<EntityInspector>| {
        let changed = state.is_changed();
        // What follows only records that the tab caught up, which shouldn't
        // make the next frame send again.
        let state = state.bypass_change_detection();
        let now = Instant::now();
        let refresh = sent.tree.is_none_or(|at| now.duration_since(at) >= REFRESH);
        // A selection changed from the tab or from elsewhere goes into the
        // history, and is revealed in the tree.
        let reveal = state.selected != state.previous;
        if reveal {
            if let Some(entity) = state.selected {
                state.history.visit(entity);
            }
            state.previous = state.selected;
        }

        let selected = state
            .selected
            .and_then(|entity| world.get_entity(entity).ok());
        let inspected = selected.map(|entity| (entity.id(), entity.archetype().id()));
        // None for an entity the inspector didn't show before, which would
        // otherwise all show as changed.
        let changed_components: Vec<ComponentId> = match selected {
            Some(entity) if inspected == sent.inspected => {
                let (last_run, this_run) = (world.last_change_tick(), world.read_change_tick());
                entity
                    .archetype()
                    .iter_components()
                    .filter(|&id| {
                        entity
                            .get_change_ticks_by_id(id)
                            .is_some_and(|ticks| ticks.is_changed(last_run, this_run))
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        let send_tree = changed || refresh || reveal;
        let send_inspector =
            send_tree || inspected != sent.inspected || !changed_components.is_empty();
        if !send_inspector {
            return None;
        }

        state.names.update(world.components());
        let registry = world.get_resource::<AppTypeRegistry>();
        let registry = registry.map(|registry| registry.read());
        let info = WorldInfo::new(world, &state.names);
        let tree = send_tree.then(|| state.tree.snapshot(&info, state.selected, reveal));
        let inspector = state.inspector.snapshot(
            &info,
            registry.as_deref(),
            state.selected,
            &state.history,
            changed_components,
        );
        if send_tree {
            sent.tree = Some(now);
        }
        sent.inspected = inspected;
        Some(Snapshot {
            tree,
            inspector: Some(inspector),
        })
    });
    if snapshot.is_none() {
        warn_once!("The Entities tab needs `EntitiesPlugin`.");
    }
    snapshot.flatten()
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
struct ComponentNames(Vec<SharedString>);

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
            self.0.push(info.name().shortname().to_string().into());
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

    /// What the inspector calls an entity of the section; `None` for a plain
    /// entity.
    fn kind(self) -> Option<&'static str> {
        match self {
            Section::Resources => Some("Resource"),
            Section::Observers => Some("Observer"),
            Section::Systems => Some("One-shot system"),
            Section::Entities => None,
        }
    }
}

/// Facts about the world that the tree and the inspector look up while
/// building a snapshot.
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
    disabled: Option<ComponentId>,
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
                disabled: world.component_id::<Disabled>(),
            },
            disabling: world
                .get_resource::<DefaultQueryFilters>()
                .map(|filters| filters.disabling_ids().collect())
                .unwrap_or_default(),
        }
    }

    /// The component's short type name.
    fn name(&self, id: ComponentId) -> SharedString {
        self.names
            .0
            .get(id.index())
            .cloned()
            .unwrap_or(SharedString::new_static("?"))
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
            .map(|name| SharedString::from(name.as_str().to_owned()))
            .or_else(|| resource.map(|id| self.name(id)))
            .or_else(|| {
                let observer = entity.get::<Observer>()?;
                Some(observer.system_name().shortname().to_string().into())
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
        let has = |id: Option<ComponentId>| id.is_some_and(|id| archetype.contains(id));
        Some(Label {
            title,
            components: components.into_iter().map(|id| self.name(id)).collect(),
            section,
            disabled: self.disabling.iter().any(|&id| archetype.contains(id)),
            has_disabled: has(self.known.disabled),
        })
    }
}

/// What identifies an entity in the tree, in links to it, and in the
/// inspector's title.
#[derive(Clone, Debug)]
struct Label {
    /// The entity's `Name`, or else the type of its resource or the system of
    /// its observer.
    title: Option<SharedString>,
    /// The names of the entity's other components, those on the fewest
    /// entities first: the rarer a component, the better it tells the entity
    /// apart.
    components: Vec<SharedString>,
    section: Section,
    /// Whether the entity has a component that disables it, such as
    /// `Disabled`.
    disabled: bool,
    /// Whether the entity has `Disabled` itself, which "Enable" removes.
    has_disabled: bool,
}

impl Label {
    /// The title, or else the rarest component's name.
    fn short(&self) -> SharedString {
        match (&self.title, self.components.first()) {
            (Some(title), _) => title.clone(),
            (None, Some(name)) => name.clone(),
            (None, None) => SharedString::new_static("empty"),
        }
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
