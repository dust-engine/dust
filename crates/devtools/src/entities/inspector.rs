//! The inspector: the selected entity's title and details, and its
//! components, each with its value below its name, as Chrome's Styles pane
//! lists an element's rules.

use bevy::ecs::change_detection::Tick;
use bevy::ecs::component::{ComponentId, ComponentInfo, StorageType};
use bevy::ecs::observer::Observer;
use bevy::ecs::ptr::Ptr;
use bevy::ecs::relationship::RelationshipAccessor;
use bevy::ecs::resource::IsResource;
use bevy::platform::collections::HashMap;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use bevy::reflect::{ReflectFromPtr, TypeRegistry};
use pumicite_egui::egui;

use super::value::{MAX_ELEMENTS, ValueUi, field_name};
use super::{
    Action, History, INDENT, Label, Section, Syntax, WorldInfo, collapsible_row,
    contains_ignore_case, entity_link, entity_menu, header_row,
};
use crate::devtools::{icon_button, paint_kebab};

/// How long a component's header stays highlighted after the component
/// changes.
const FLASH_SECONDS: f32 = 0.8;

#[derive(Default)]
pub(super) struct InspectorState {
    /// Only the components whose name contains this show.
    filter: String,
    /// When each component of `flash_entity` was last seen to change, to
    /// highlight its header as Chrome highlights a changed attribute.
    flashes: HashMap<ComponentId, Instant>,
    flash_entity: Option<Entity>,
    /// The world's change tick when the inspector was last drawn.
    last_tick: Option<Tick>,
}

impl InspectorState {
    /// Records which of `entity`'s components changed since the inspector was
    /// last drawn.
    fn update_flashes(&mut self, entity: EntityRef, this_tick: Tick) {
        if self.flash_entity != Some(entity.id()) {
            self.flashes.clear();
            self.flash_entity = Some(entity.id());
        } else if let Some(last_tick) = self.last_tick {
            let now = Instant::now();
            for id in entity.archetype().iter_components() {
                let changed = entity
                    .get_change_ticks_by_id(id)
                    .is_some_and(|ticks| ticks.is_changed(last_tick, this_tick));
                if changed {
                    self.flashes.insert(id, now);
                }
            }
        }
        self.last_tick = Some(this_tick);
    }

    /// The background of the component's header: the selection color right
    /// after a change, fading out over `FLASH_SECONDS`.
    fn flash(&self, id: ComponentId, visuals: &egui::Visuals) -> egui::Color32 {
        let fade = self.flashes.get(&id).map_or(0.0, |changed| {
            1.0 - changed.elapsed().as_secs_f32() / FLASH_SECONDS
        });
        if fade > 0.0 {
            visuals.selection.bg_fill.gamma_multiply(0.6 * fade)
        } else {
            egui::Color32::TRANSPARENT
        }
    }
}

pub(super) fn inspector_ui(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    registry: Option<&TypeRegistry>,
    selected: Option<Entity>,
    state: &mut InspectorState,
    history: &History,
    actions: &mut Vec<Action>,
) {
    let shown = selected
        .and_then(|entity| Some((info.world.get_entity(entity).ok()?, info.label(entity)?)));
    title_bar(ui, info, shown.as_ref(), history, actions);
    let Some(entity) = selected else {
        ui.weak("Select an entity in the tree.");
        return;
    };
    let Some((entity, label)) = shown else {
        ui.weak(format!("{entity} was despawned."));
        return;
    };
    details(ui, entity, &label);
    ui.add(
        egui::TextEdit::singleline(&mut state.filter)
            .hint_text("Filter components")
            .desired_width(f32::INFINITY),
    );
    state.update_flashes(entity, info.world.read_change_tick());

    // A resource's own component first, then the rest by name.
    let resource = entity
        .get::<IsResource>()
        .map(IsResource::resource_component_id);
    let mut components: Vec<(ComponentId, &ComponentInfo)> = entity
        .archetype()
        .iter_components()
        .filter(|&id| contains_ignore_case(info.name(id), state.filter.trim()))
        .filter_map(|id| Some((id, info.world.components().get_info(id)?)))
        .collect();
    components.sort_by_cached_key(|&(id, _)| (Some(id) != resource, info.name(id).to_lowercase()));

    egui::ScrollArea::vertical()
        .id_salt("devtools_entities_inspector")
        .auto_shrink(false)
        .show(ui, |ui| {
            if components.is_empty() {
                ui.weak("No components match the filter.");
            }
            for (i, &(id, component)) in components.iter().enumerate() {
                if i > 0 {
                    ui.separator();
                }
                let background = state.flash(id, ui.visuals());
                component_ui(
                    ui,
                    info,
                    registry,
                    entity,
                    (id, component),
                    label.section,
                    background,
                    actions,
                );
            }
        });
}

/// Back and forward buttons, the entity's title, and its ⋮ menu.
fn title_bar(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    shown: Option<&(EntityRef, Label)>,
    history: &History,
    actions: &mut Vec<Action>,
) {
    ui.horizontal(|ui| {
        ui.add_enabled_ui(history.can_go_back(), |ui| {
            if icon_button(ui, paint_back).on_hover_text("Back").clicked() {
                actions.push(Action::Back);
            }
        });
        ui.add_enabled_ui(history.can_go_forward(), |ui| {
            if icon_button(ui, paint_forward)
                .on_hover_text("Forward")
                .clicked()
            {
                actions.push(Action::Forward);
            }
        });
        let Some((entity, label)) = shown else {
            return;
        };
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let menu = icon_button(ui, paint_kebab).on_hover_text("Entity actions");
            egui::Popup::menu(&menu).show(|ui| entity_menu(ui, info, entity.id(), label, actions));
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let title = match &label.title {
                    Some(title) => {
                        egui::RichText::new(title).color(Syntax::Tag.color(ui.visuals()))
                    }
                    None => egui::RichText::new(entity.id().to_string()).monospace(),
                };
                ui.add(egui::Label::new(title.heading()).truncate());
            });
        });
    });
}

/// The entity's id and kind, where it's stored, and when it was spawned.
fn details(ui: &mut egui::Ui, entity: EntityRef, label: &Label) {
    let kind = match label.section {
        Section::Resources => Some("Resource"),
        Section::Observers => Some("Observer"),
        Section::Systems => Some("One-shot system"),
        Section::Entities => None,
    };
    if label.title.is_some() || kind.is_some() || label.disabled {
        ui.horizontal_wrapped(|ui| {
            // The title takes the id's place when there is no title.
            if label.title.is_some() {
                ui.label(
                    egui::RichText::new(entity.id().to_string())
                        .monospace()
                        .weak(),
                );
            }
            for badge in kind.into_iter().chain(label.disabled.then_some("Disabled")) {
                egui::Frame::new()
                    .fill(ui.visuals().widgets.inactive.weak_bg_fill)
                    .corner_radius(egui::CornerRadius::same(3))
                    .inner_margin(egui::Margin::symmetric(4, 0))
                    .show(ui, |ui| ui.label(egui::RichText::new(badge).small()));
            }
        });
    }
    let archetype = entity.archetype();
    ui.label(
        egui::RichText::new(format!(
            "Archetype {} · {} {} · Table {}",
            archetype.id().index(),
            archetype.len(),
            if archetype.len() == 1 {
                "entity"
            } else {
                "entities"
            },
            archetype.table_id().as_usize(),
        ))
        .small()
        .weak(),
    );
    let tick = entity.spawn_tick().get();
    match entity.spawned_by().into_option() {
        Some(location) => {
            let text = format!("Spawned at tick {tick} by {location}");
            ui.label(egui::RichText::new(text).small().weak());
        }
        None => {
            let text = format!("Spawned at tick {tick}");
            ui.label(egui::RichText::new(text).small().weak())
                .on_hover_text("Bevy's `track_location` feature records where, too.");
        }
    }
}

/// A component's header, and below it, unless the component is a marker
/// without data, its value.
#[expect(clippy::too_many_arguments, reason = "the inspector's drawing state")]
fn component_ui(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    registry: Option<&TypeRegistry>,
    entity: EntityRef,
    (id, component): (ComponentId, &ComponentInfo),
    section: Section,
    background: egui::Color32,
    actions: &mut Vec<Action>,
) {
    let name = info.name(id);
    let header = |ui: &mut egui::Ui| {
        ui.label(egui::RichText::new(name).strong());
        // Where the type comes from, as Chrome names a rule's stylesheet.
        let path = component.name();
        let krate = path.split("::").next().unwrap_or_default().to_owned();
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(4.0);
            ui.weak(krate);
        });
    };
    let response = if component.layout().size() > 0 {
        let section_id = egui::Id::new(("devtools_entities_component", id));
        let (response, open) = collapsible_row(ui, section_id, true, background, header);
        if open {
            ui.indent(section_id, |ui| {
                component_value(ui, info, registry, entity, (id, component), actions);
            });
        }
        response
    } else {
        header_row(ui, None, background, header)
    };
    response
        .on_hover_ui(|ui| component_tooltip(ui, entity, id, component))
        .context_menu(|ui| {
            if ui.button("Copy type path").clicked() {
                ui.ctx().copy_text(component.name().to_string());
            }
            if ui.button("Show entities with this component").clicked() {
                actions.push(Action::Filter(name.to_owned()));
            }
            if section == Section::Entities {
                ui.separator();
                let error = ui.visuals().error_fg_color;
                if ui
                    .button(egui::RichText::new("Remove component").color(error))
                    .clicked()
                {
                    actions.push(Action::RemoveComponent(entity.id(), id));
                }
            }
        });
}

/// The component's full type path, how it's stored, and when it last
/// changed.
fn component_tooltip(
    ui: &mut egui::Ui,
    entity: EntityRef,
    id: ComponentId,
    component: &ComponentInfo,
) {
    ui.label(egui::RichText::new(component.name().to_string()).monospace());
    let storage = match component.storage_type() {
        StorageType::Table => "Table",
        StorageType::SparseSet => "Sparse set",
    };
    let mutability = if component.mutable() {
        "mutable"
    } else {
        "immutable"
    };
    ui.label(format!(
        "{storage} storage · {} bytes · {mutability}",
        component.layout().size()
    ));
    if let Some(ticks) = entity.get_change_ticks_by_id(id) {
        ui.label(format!(
            "Added at tick {} · changed at tick {}",
            ticks.added.get(),
            ticks.changed.get()
        ));
    }
}

/// The component's value: the entities a relationship links, the text of a
/// `Name`, what an `Observer` watches, or else the reflected value, which can
/// be edited if the component is mutable.
fn component_value(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    registry: Option<&TypeRegistry>,
    entity: EntityRef,
    (id, component): (ComponentId, &ComponentInfo),
    actions: &mut Vec<Action>,
) {
    let Ok(ptr) = entity.get_by_id(id) else {
        return;
    };
    // Relationships show through their accessor rather than as reflected
    // values: it reaches the linked entities without reflection, and leaves
    // no field to edit around the hooks that keep both sides in sync.
    if let Some(accessor) = component.relationship_accessor() {
        relationship_ui(ui, info, accessor, ptr, actions);
        return;
    }
    if Some(id) == info.known.name
        && let Some(name) = entity.get::<Name>()
    {
        ui.horizontal(|ui| {
            ui.add_space(INDENT);
            let mut text = name.as_str().to_owned();
            let field = egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY);
            if ui.add(field).changed() {
                actions.push(Action::Rename(entity.id(), text));
            }
        });
        return;
    }
    if Some(id) == info.known.observer
        && let Some(observer) = entity.get::<Observer>()
    {
        observer_ui(ui, info, observer, actions);
        return;
    }
    let reflected = registry
        .zip(component.type_id())
        .and_then(|(registry, type_id)| {
            let from_ptr = registry
                .get_type_data::<ReflectFromPtr>(type_id)
                .filter(|from_ptr| from_ptr.type_id() == type_id)?;
            // SAFETY: `ptr` points to the component, of type `type_id`, which
            // `from_ptr` was checked to be for.
            Some((registry, unsafe { from_ptr.ptr_as_reflect(ptr) }))
        });
    let Some((registry, value)) = reflected else {
        ui.horizontal_wrapped(|ui| {
            ui.add_space(INDENT);
            ui.weak(
                "Not reflected. Derive `Reflect` for the type to see its value; \
                 a generic type also needs `App::register_type`.",
            );
        });
        return;
    };
    let value_id = egui::Id::new(("devtools_entities_value", id));
    let mut value_ui = ValueUi::new(info, registry, component.mutable(), value_id, actions);
    value_ui.show(ui, value.as_partial_reflect());
    for (path, value) in value_ui.into_edits() {
        actions.push(Action::Edit {
            entity: entity.id(),
            component: id,
            path,
            value,
        });
    }
}

/// A link to a relationship's target, or to each of a relationship target's
/// sources, such as a `ChildOf`'s parent or a `Children`'s children.
fn relationship_ui(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    accessor: &RelationshipAccessor,
    ptr: Ptr,
    actions: &mut Vec<Action>,
) {
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
    for &entity in related.iter().take(MAX_ELEMENTS) {
        ui.horizontal(|ui| {
            ui.add_space(INDENT);
            entity_link(ui, info, entity, actions);
        });
    }
    let more = related.len().saturating_sub(MAX_ELEMENTS);
    if more > 0 {
        ui.horizontal(|ui| {
            ui.add_space(INDENT);
            ui.weak(format!("… {more} more"));
        });
    }
}

/// The observer's system, the events it observes, and the components and
/// entities it watches, if limited to some.
fn observer_ui(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    observer: &Observer,
    actions: &mut Vec<Action>,
) {
    let descriptor = observer.descriptor();
    let row = |ui: &mut egui::Ui, name: &str, value: String| {
        ui.horizontal_wrapped(|ui| {
            ui.add_space(INDENT);
            field_name(ui, name);
            ui.label(value);
        });
    };
    let system = observer.system_name();
    row(ui, "system", system.to_string());
    let events: Vec<&str> = descriptor
        .event_keys()
        .iter()
        .map(|key| info.name(key.component_id()))
        .collect();
    row(ui, "events", events.join(", "));
    if !descriptor.components().is_empty() {
        let components: Vec<&str> = descriptor
            .components()
            .iter()
            .map(|&id| info.name(id))
            .collect();
        row(ui, "components", components.join(", "));
    }
    if !descriptor.entities().is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.add_space(INDENT);
            field_name(ui, "entities");
            for &entity in descriptor.entities() {
                entity_link(ui, info, entity, actions);
            }
        });
    }
}

/// "<", for the back button.
fn paint_back(painter: &egui::Painter, c: egui::Pos2, stroke: egui::Stroke) {
    let tip = c + egui::vec2(-2.5, 0.0);
    painter.line_segment([tip + egui::vec2(4.5, -4.5), tip], stroke);
    painter.line_segment([tip + egui::vec2(4.5, 4.5), tip], stroke);
}

/// ">", for the forward button.
fn paint_forward(painter: &egui::Painter, c: egui::Pos2, stroke: egui::Stroke) {
    let tip = c + egui::vec2(2.5, 0.0);
    painter.line_segment([tip + egui::vec2(-4.5, -4.5), tip], stroke);
    painter.line_segment([tip + egui::vec2(-4.5, 4.5), tip], stroke);
}
