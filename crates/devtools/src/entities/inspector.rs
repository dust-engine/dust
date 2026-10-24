//! The inspector: the selected entity's title and details, and its
//! components, each with its value below its name, as Chrome's Styles pane
//! lists an element's rules.

use bevy::ecs::component::{ComponentId, ComponentInfo, StorageType};
use bevy::ecs::observer::Observer;
use bevy::ecs::ptr::Ptr;
use bevy::ecs::relationship::RelationshipAccessor;
use bevy::ecs::resource::IsResource;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::reflect::{ReflectFromPtr, TypeRegistry};
use gpui::SharedString;

use super::value::{Link, MAX_ELEMENTS, Piece, ValueRow, ValueRowKind, ValueRows};
use super::{History, Label, Section, WorldInfo, contains_ignore_case};

#[derive(Default)]
pub(super) struct InspectorState {
    /// Only the components whose name contains this show.
    pub(super) filter: String,
    /// The components whose value is hidden. Kept by component rather than
    /// by entity, as are `open_values`.
    closed: HashSet<ComponentId>,
    /// The values open within each component; see [`ValueRows`].
    open_values: HashSet<(ComponentId, Vec<usize>)>,
}

impl InspectorState {
    pub(super) fn toggle_component(&mut self, component: ComponentId) {
        if !self.closed.remove(&component) {
            self.closed.insert(component);
        }
    }

    pub(super) fn toggle_value(&mut self, component: ComponentId, path: Vec<usize>) {
        let key = (component, path);
        if !self.open_values.remove(&key) {
            self.open_values.insert(key);
        }
    }

    /// What the inspector shows. `changed` are the components of the
    /// selected entity that changed since the last snapshot.
    pub(super) fn snapshot(
        &self,
        info: &WorldInfo,
        registry: Option<&TypeRegistry>,
        selected: Option<Entity>,
        history: &History,
        changed: Vec<ComponentId>,
    ) -> InspectorSnapshot {
        let entity = selected.and_then(|entity| {
            let entity = info.world.get_entity(entity).ok()?;
            let label = info.label(entity.id())?;
            Some(self.entity(info, registry, entity, label))
        });
        InspectorSnapshot {
            selected,
            can_go_back: history.can_go_back(),
            can_go_forward: history.can_go_forward(),
            entity,
            changed,
        }
    }

    fn entity(
        &self,
        info: &WorldInfo,
        registry: Option<&TypeRegistry>,
        entity: EntityRef,
        label: Label,
    ) -> EntityView {
        let archetype = entity.archetype();
        let archetype_line = format!(
            "Archetype {} · {} {} · Table {}",
            archetype.id().index(),
            archetype.len(),
            if archetype.len() == 1 {
                "entity"
            } else {
                "entities"
            },
            archetype.table_id().as_usize(),
        );
        let tick = entity.spawn_tick().get();
        let location = entity.spawned_by().into_option();
        let spawned = match location {
            Some(location) => format!("Spawned at tick {tick} by {location}"),
            None => format!("Spawned at tick {tick}"),
        };

        // A resource's own component first, then the rest by name.
        let resource = entity
            .get::<IsResource>()
            .map(IsResource::resource_component_id);
        let filter = self.filter.trim();
        let mut components: Vec<(ComponentId, &ComponentInfo)> = archetype
            .iter_components()
            .filter(|&id| contains_ignore_case(&info.name(id), filter))
            .filter_map(|id| Some((id, info.world.components().get_info(id)?)))
            .collect();
        components
            .sort_by_cached_key(|&(id, _)| (Some(id) != resource, info.name(id).to_lowercase()));
        let components = components
            .into_iter()
            .map(|(id, component)| {
                self.component(info, registry, entity, label.section, id, component)
            })
            .collect();

        EntityView {
            entity: entity.id(),
            label,
            archetype: archetype_line.into(),
            spawned: spawned.into(),
            spawned_by: location.is_some(),
            filtered: !filter.is_empty(),
            components,
        }
    }

    fn component(
        &self,
        info: &WorldInfo,
        registry: Option<&TypeRegistry>,
        entity: EntityRef,
        section: Section,
        id: ComponentId,
        component: &ComponentInfo,
    ) -> ComponentView {
        let type_path = component.name().to_string();
        // Where the type comes from, as Chrome names a rule's stylesheet.
        let krate = type_path.split("::").next().unwrap_or_default().to_owned();
        let storage = match component.storage_type() {
            StorageType::Table => "Table",
            StorageType::SparseSet => "Sparse set",
        };
        let mutability = if component.mutable() {
            "mutable"
        } else {
            "immutable"
        };
        let storage = format!(
            "{storage} storage · {} bytes · {mutability}",
            component.layout().size()
        );
        let ticks = entity.get_change_ticks_by_id(id).map(|ticks| {
            format!(
                "Added at tick {} · changed at tick {}",
                ticks.added.get(),
                ticks.changed.get()
            )
            .into()
        });
        // A marker without data has no value to show.
        let has_value = component.layout().size() > 0;
        let open = has_value && !self.closed.contains(&id);
        ComponentView {
            id,
            name: info.name(id),
            type_path: type_path.into(),
            krate: krate.into(),
            storage: storage.into(),
            ticks,
            has_value,
            body: open.then(|| self.body(info, registry, entity, id, component)),
            removable: section == Section::Entities,
        }
    }

    /// The component's value: the entities a relationship links, the text of a
    /// `Name`, what an `Observer` watches, or else the reflected value, which
    /// can be edited if the component is mutable.
    fn body(
        &self,
        info: &WorldInfo,
        registry: Option<&TypeRegistry>,
        entity: EntityRef,
        id: ComponentId,
        component: &ComponentInfo,
    ) -> Body {
        let Ok(ptr) = entity.get_by_id(id) else {
            return Body::Rows(Vec::new());
        };
        // Relationships show through their accessor rather than as reflected
        // values: it reaches the linked entities without reflection, and
        // leaves no field to edit around the hooks that keep both sides in
        // sync.
        if let Some(accessor) = component.relationship_accessor() {
            return Body::Rows(relationship_rows(info, accessor, ptr));
        }
        if Some(id) == info.known.name
            && let Some(name) = entity.get::<Name>()
        {
            return Body::Name(name.as_str().to_owned().into());
        }
        if Some(id) == info.known.observer
            && let Some(observer) = entity.get::<Observer>()
        {
            return Body::Rows(observer_rows(info, observer));
        }
        let reflected = registry
            .zip(component.type_id())
            .and_then(|(registry, type_id)| {
                let from_ptr = registry
                    .get_type_data::<ReflectFromPtr>(type_id)
                    .filter(|from_ptr| from_ptr.type_id() == type_id)?;
                // SAFETY: `ptr` points to the component, of type `type_id`,
                // which `from_ptr` was checked to be for.
                Some((registry, unsafe { from_ptr.ptr_as_reflect(ptr) }))
            });
        let Some((registry, value)) = reflected else {
            return Body::NotReflected;
        };
        let mut rows = ValueRows::new(info, registry, component.mutable(), id, &self.open_values);
        rows.show(value.as_partial_reflect());
        Body::Rows(rows.into_rows())
    }
}

/// A row with a link to a relationship's target, or to each of a
/// relationship target's sources, such as a `ChildOf`'s parent or a
/// `Children`'s children.
fn relationship_rows(info: &WorldInfo, accessor: &RelationshipAccessor, ptr: Ptr) -> Vec<ValueRow> {
    let related: Vec<Entity> = match *accessor {
        RelationshipAccessor::Relationship {
            entity_field_offset,
            ..
        } => {
            // SAFETY: Bevy records `entity_field_offset` as the offset of the
            // target `Entity` within the relationship `ptr` points to.
            vec![unsafe { *ptr.byte_add(entity_field_offset).deref::<Entity>() }]
        }
        RelationshipAccessor::RelationshipTarget { iter, .. } => {
            // SAFETY: `iter` is the accessor of the component `ptr` points to.
            unsafe { iter(ptr) }.collect()
        }
    };
    let row = |kind| ValueRow { depth: 0, kind };
    let mut rows: Vec<ValueRow> = related
        .iter()
        .take(MAX_ELEMENTS)
        .map(|&entity| {
            row(ValueRowKind::Inline {
                name: None,
                pieces: vec![Piece::Entity(Link::new(info, entity))],
            })
        })
        .collect();
    let more = related.len().saturating_sub(MAX_ELEMENTS);
    if more > 0 {
        rows.push(row(ValueRowKind::More(more)));
    }
    rows
}

/// The observer's system, the events it observes, and the components and
/// entities it watches, if limited to some.
fn observer_rows(info: &WorldInfo, observer: &Observer) -> Vec<ValueRow> {
    let descriptor = observer.descriptor();
    let row = |name: &'static str, pieces| ValueRow {
        depth: 0,
        kind: ValueRowKind::Inline {
            name: Some(SharedString::new_static(name)),
            pieces,
        },
    };
    let text = |text: String| vec![Piece::Text(text.into())];
    let names = |ids: &mut dyn Iterator<Item = ComponentId>| {
        let names: Vec<SharedString> = ids.map(|id| info.name(id)).collect();
        text(names.join(", "))
    };
    let mut rows = vec![
        row("system", text(observer.system_name().to_string())),
        row(
            "events",
            names(&mut descriptor.event_keys().iter().map(|key| key.component_id())),
        ),
    ];
    if !descriptor.components().is_empty() {
        rows.push(row(
            "components",
            names(&mut descriptor.components().iter().copied()),
        ));
    }
    if !descriptor.entities().is_empty() {
        let links = descriptor
            .entities()
            .iter()
            .map(|&entity| Piece::Entity(Link::new(info, entity)))
            .collect();
        rows.push(row("entities", links));
    }
    rows
}

/// What the inspector shows, sent to the UI thread.
pub(super) struct InspectorSnapshot {
    pub(super) selected: Option<Entity>,
    pub(super) can_go_back: bool,
    pub(super) can_go_forward: bool,
    /// The selected entity; `None` if there is none, or it was despawned.
    pub(super) entity: Option<EntityView>,
    /// The components of `entity` that changed since the last snapshot,
    /// which the inspector highlights.
    pub(super) changed: Vec<ComponentId>,
}

pub(super) struct EntityView {
    pub(super) entity: Entity,
    pub(super) label: Label,
    /// Its archetype and table.
    pub(super) archetype: SharedString,
    /// When it was spawned, and where from if Bevy tracked it.
    pub(super) spawned: SharedString,
    pub(super) spawned_by: bool,
    /// Whether the component filter leaves some components out.
    pub(super) filtered: bool,
    pub(super) components: Vec<ComponentView>,
}

pub(super) struct ComponentView {
    pub(super) id: ComponentId,
    /// The short type name.
    pub(super) name: SharedString,
    pub(super) type_path: SharedString,
    /// The crate the type is from.
    pub(super) krate: SharedString,
    /// How it's stored, its size, and whether it's mutable.
    pub(super) storage: SharedString,
    /// When it was added and last changed.
    pub(super) ticks: Option<SharedString>,
    /// `false` for a marker without data.
    pub(super) has_value: bool,
    /// The value, while it shows.
    pub(super) body: Option<Body>,
    /// Whether it can be removed; not from resources, observers and
    /// one-shot systems.
    pub(super) removable: bool,
}

pub(super) enum Body {
    Rows(Vec<ValueRow>),
    /// A `Name`'s text, which can be edited.
    Name(SharedString),
    NotReflected,
}
