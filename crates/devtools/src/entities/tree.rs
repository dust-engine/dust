//! The entity tree: each section's root entities, with their children nested
//! below them by `ChildOf` as Chrome nests the DOM, a filter field above, and
//! the selected entity's ancestors as breadcrumbs below.

use bevy::ecs::entity::EntityHashSet;
use bevy::ecs::observer::Observer;
use bevy::ecs::resource::IsResource;
use bevy::prelude::*;
use pumicite_egui::egui;

use super::{
    Action, INDENT, Label, Section, Syntax, WorldInfo, contains_ignore_case, entity_menu,
    label_tooltip, paint_triangle,
};

const FIND_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::F);

/// The most components a row lists after the entity's title; the rest are
/// counted.
const ROW_COMPONENTS: usize = 3;

pub(super) struct TreeState {
    /// Entities whose children show.
    expanded: EntityHashSet,
    /// Whether each section's entities show, indexed by `Section as usize`.
    open_sections: [bool; Section::ALL.len()],
    /// Only the entities matching this show, with their ancestors; see
    /// [`Filter::new`].
    pub(super) filter: String,
}

impl Default for TreeState {
    fn default() -> Self {
        Self {
            expanded: EntityHashSet::default(),
            open_sections: Section::ALL.map(|section| section == Section::Entities),
            filter: String::new(),
        }
    }
}

impl TreeState {
    /// Expands the ancestors of `entity`, and opens the section of its root,
    /// so that its row shows.
    pub(super) fn expand_to(&mut self, info: &WorldInfo, entity: Entity) {
        let mut root = entity;
        for ancestor in info.ancestors(entity) {
            self.expanded.insert(ancestor);
            root = ancestor;
        }
        if let Ok(root) = info.world.get_entity(root) {
            self.open_sections[info.section(root.archetype()) as usize] = true;
        }
    }

    /// Expands or collapses `entity`, and with `recursive`, its descendants.
    pub(super) fn set_expanded(
        &mut self,
        info: &WorldInfo,
        entity: Entity,
        open: bool,
        recursive: bool,
    ) {
        let mut visited = EntityHashSet::default();
        let mut stack = vec![entity];
        while let Some(entity) = stack.pop() {
            if !visited.insert(entity) {
                continue;
            }
            if open {
                self.expanded.insert(entity);
            } else {
                self.expanded.remove(&entity);
            }
            if recursive {
                stack.extend_from_slice(info.children(entity));
            }
        }
    }

    /// The rows to show, in order: each section, followed by its entities
    /// while it's open.
    pub(super) fn rows(&self, info: &WorldInfo, filter: Option<&Filter>) -> Vec<Row> {
        let shows = |entity: &Entity| filter.is_none_or(|filter| filter.shown.contains(entity));
        let mut rows = Vec::new();
        for (section, roots) in Section::ALL.into_iter().zip(roots(info)) {
            let roots: Vec<Entity> = roots.into_iter().filter(shows).collect();
            if filter.is_some() && roots.is_empty() {
                continue;
            }
            // A filter shows all of its matches.
            let open = filter.is_some() || self.open_sections[section as usize];
            rows.push(Row {
                kind: RowKind::Section(section),
                depth: 0,
                children: roots.len(),
                open,
            });
            if !open {
                continue;
            }
            // Depth first. The stack holds the rows still to push, last
            // first, so that the first root pops first.
            let mut stack: Vec<(Entity, usize)> =
                roots.iter().rev().map(|&root| (root, 1)).collect();
            let mut visited = EntityHashSet::default();
            while let Some((entity, depth)) = stack.pop() {
                // `ChildOf` shouldn't form cycles, but a tree of one would
                // never end.
                if !visited.insert(entity) {
                    continue;
                }
                let children: Vec<Entity> = info
                    .children(entity)
                    .iter()
                    .copied()
                    .filter(shows)
                    .collect();
                let open =
                    !children.is_empty() && (filter.is_some() || self.expanded.contains(&entity));
                rows.push(Row {
                    kind: RowKind::Entity(entity),
                    depth,
                    children: children.len(),
                    open,
                });
                if open {
                    stack.extend(children.iter().rev().map(|&child| (child, depth + 1)));
                }
            }
        }
        rows
    }
}

/// Each section's root entities, those without a parent, indexed by
/// `Section as usize`. Resources are sorted by type name, the rest by index,
/// which follows the order entities were first spawned in.
pub(super) fn roots(info: &WorldInfo) -> [Vec<Entity>; Section::ALL.len()] {
    let mut roots: [Vec<Entity>; Section::ALL.len()] = Default::default();
    for archetype in info.world.archetypes().iter() {
        let child = info.known.child_of.is_some_and(|id| archetype.contains(id));
        if archetype.is_empty() || child {
            continue;
        }
        roots[info.section(archetype) as usize]
            .extend(archetype.entities().iter().map(|entity| entity.id()));
    }
    for (section, roots) in Section::ALL.into_iter().zip(&mut roots) {
        if section == Section::Resources {
            roots.sort_by_cached_key(|&entity| {
                info.world
                    .get::<IsResource>(entity)
                    .map(|resource| info.name(resource.resource_component_id()).to_lowercase())
            });
        } else {
            roots.sort_unstable_by_key(|entity| entity.index_u32());
        }
    }
    roots
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum RowKind {
    Section(Section),
    Entity(Entity),
}

pub(super) struct Row {
    pub(super) kind: RowKind,
    pub(super) depth: usize,
    /// How many rows open below this one: a section's roots, or an entity's
    /// children.
    children: usize,
    open: bool,
}

/// The entities matching the tree's filter, and their ancestors, which show
/// so that the matches keep their place in the tree.
pub(super) struct Filter {
    pub(super) matches: EntityHashSet,
    pub(super) shown: EntityHashSet,
}

impl Filter {
    /// `None` for a filter of only whitespace, which shows every entity.
    /// Otherwise each of the filter's words has to be part of the entity's
    /// title or of one of its components' names, or be the entity's id, with
    /// or without the generation.
    pub(super) fn new(info: &WorldInfo, text: &str) -> Option<Self> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return None;
        }
        let mut matches = EntityHashSet::default();
        for archetype in info.world.archetypes().iter() {
            if archetype.is_empty() {
                continue;
            }
            // Names of the archetype's components are the same for each of
            // its entities, so they're only checked once.
            let by_component: Vec<bool> = words
                .iter()
                .map(|word| {
                    archetype
                        .iter_components()
                        .any(|id| contains_ignore_case(info.name(id), word))
                })
                .collect();
            let observers = info.known.observer.is_some_and(|id| archetype.contains(id));
            for entity in archetype.entities().iter().map(|entity| entity.id()) {
                let all = words
                    .iter()
                    .zip(&by_component)
                    .all(|(word, &by_component)| {
                        by_component || title_or_id_matches(info, entity, word, observers)
                    });
                if all {
                    matches.insert(entity);
                }
            }
        }
        let mut shown = EntityHashSet::default();
        for &entity in &matches {
            // An ancestor already shown has its own ancestors shown too.
            let mut next = Some(entity);
            while let Some(entity) = next
                && shown.insert(entity)
            {
                next = info.parent(entity);
            }
        }
        Some(Self { matches, shown })
    }
}

/// Whether `word` is part of `entity`'s `Name` or of its observer's system
/// name, or is the entity's id.
fn title_or_id_matches(info: &WorldInfo, entity: Entity, word: &str, observer: bool) -> bool {
    let id_matches = match word.split_once(['v', 'V']) {
        Some((index, _)) => {
            index.parse() == Ok(entity.index_u32()) && entity.to_string().eq_ignore_ascii_case(word)
        }
        None => word.parse() == Ok(entity.index_u32()),
    };
    id_matches
        || info
            .world
            .get::<Name>(entity)
            .is_some_and(|name| contains_ignore_case(name.as_str(), word))
        || observer
            && info.world.get::<Observer>(entity).is_some_and(|observer| {
                contains_ignore_case(&observer.system_name().shortname().to_string(), word)
            })
}

pub(super) fn tree_ui(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    selected: &mut Option<Entity>,
    state: &mut TreeState,
    selection_changed: bool,
    actions: &mut Vec<Action>,
) {
    // Revealed on any change of the selection, including one from outside
    // the tree. Within the tree, the row is already in view, which makes
    // revealing it do nothing.
    let reveal = selection_changed.then_some(*selected).flatten();
    state
        .expanded
        .retain(|&entity| info.world.entities().contains_spawned(entity));
    if let Some(entity) = reveal {
        state.expand_to(info, entity);
    }
    // From the text as it was before this frame's typing, which shows a
    // frame later.
    let filter = Filter::new(info, &state.filter);
    let rows = state.rows(info, filter.as_ref());

    toolbar(ui, info, state, filter.as_ref());
    if let Some(entity) = *selected
        && info.world.get_entity(entity).is_ok()
    {
        egui::Panel::bottom("devtools_entities_breadcrumbs").show(ui, |ui| {
            breadcrumbs(ui, info, entity, actions);
        });
    }
    rows_ui(
        ui,
        info,
        &rows,
        selected,
        state,
        reveal.is_some(),
        filter.as_ref(),
        actions,
    );
}

/// The filter field, and how many entities there are or match.
fn toolbar(ui: &mut egui::Ui, info: &WorldInfo, state: &mut TreeState, filter: Option<&Filter>) {
    let id = egui::Id::new("devtools_entities_filter");
    if ui.input_mut(|input| input.consume_shortcut(&FIND_SHORTCUT)) {
        ui.memory_mut(|memory| memory.request_focus(id));
    }
    ui.horizontal(|ui| {
        // Right-to-left first, so the count keeps its place at the end and the
        // field takes the width left.
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let count = match filter {
                Some(filter) => format!("{} found", filter.matches.len()),
                None => format!("{} entities", info.entity_count),
            };
            ui.weak(count);
            let shortcut =
                FIND_SHORTCUT.format(&egui::ModifierNames::NAMES, ui.ctx().os().is_mac());
            let field = ui
                .add(
                    egui::TextEdit::singleline(&mut state.filter)
                        .id(id)
                        .hint_text(format!("Filter ({shortcut})"))
                        .desired_width(f32::INFINITY),
                )
                .on_hover_text(
                    "Shows the entities whose name, component names or id contain each word.",
                );
            if field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                state.filter.clear();
            }
        });
    });
}

/// The selected entity and its ancestors, from the root down. Clicking one
/// selects it.
fn breadcrumbs(ui: &mut egui::Ui, info: &WorldInfo, selected: Entity, actions: &mut Vec<Action>) {
    let mut path: Vec<Entity> = info.ancestors(selected).collect();
    path.reverse();
    path.push(selected);
    egui::ScrollArea::horizontal()
        .id_salt("devtools_entities_breadcrumbs")
        .stick_to_right(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                for (i, &entity) in path.iter().enumerate() {
                    if i > 0 {
                        let height = ui.spacing().interact_size.y;
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(6.0, height), egui::Sense::hover());
                        paint_separator(
                            ui.painter(),
                            rect.center(),
                            ui.visuals().weak_text_color(),
                        );
                    }
                    let Some(label) = info.label(entity) else {
                        continue;
                    };
                    let crumb = ui
                        .selectable_label(entity == selected, label.short(info))
                        .on_hover_ui(|ui| label_tooltip(ui, info, &label));
                    if crumb.clicked() {
                        actions.push(Action::Select(entity));
                    }
                }
            });
        });
}

/// ">", between breadcrumbs.
fn paint_separator(painter: &egui::Painter, c: egui::Pos2, color: egui::Color32) {
    let stroke = egui::Stroke::new(1.0, color);
    let tip = c + egui::vec2(1.5, 0.0);
    painter.line_segment([tip + egui::vec2(-3.0, -3.5), tip], stroke);
    painter.line_segment([tip + egui::vec2(-3.0, 3.5), tip], stroke);
}

/// The rows in view. Rows all have the same height, so the ones out of view
/// are skipped without being laid out.
#[expect(clippy::too_many_arguments, reason = "the tree's drawing state")]
fn rows_ui(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    rows: &[Row],
    selected: &mut Option<Entity>,
    state: &mut TreeState,
    reveal: bool,
    filter: Option<&Filter>,
    actions: &mut Vec<Action>,
) {
    // The tree takes the keyboard focus when clicked, and then moves the
    // selection with the arrow keys.
    let tree_id = egui::Id::new("devtools_entities_tree");
    let focused = ui.memory(|memory| memory.has_focus(tree_id));
    if focused {
        keyboard(ui, info, rows, selected, state, tree_id);
    }
    let selected_index = selected.and_then(|selected| {
        rows.iter()
            .position(|row| row.kind == RowKind::Entity(selected))
    });
    let row_height = ui.spacing().interact_size.y;
    egui::ScrollArea::vertical()
        .id_salt(tree_id)
        .auto_shrink(false)
        .show_viewport(ui, |ui, viewport| {
            let content = ui.max_rect();
            ui.set_height(row_height * rows.len() as f32);
            // Behind the rows, taking the clicks between and below them.
            let background = ui.interact(
                viewport.translate(content.min.to_vec2()),
                tree_id,
                egui::Sense::click(),
            );
            if background.clicked() {
                background.request_focus();
            }
            let row_rect = |index: usize| {
                egui::Rect::from_min_size(
                    egui::pos2(content.left(), content.top() + index as f32 * row_height),
                    egui::vec2(content.width(), row_height),
                )
            };
            let first = (viewport.min.y / row_height).floor().max(0.0) as usize;
            let end = ((viewport.max.y / row_height).ceil() as usize).min(rows.len());
            if reveal && let Some(index) = selected_index {
                // A row out of view is centered, rather than brought to the
                // nearest edge, so that its surroundings show too.
                let in_view = (first..end).contains(&index);
                ui.scroll_to_rect(row_rect(index), (!in_view).then_some(egui::Align::Center));
            }
            for (index, row) in rows.iter().enumerate().take(end).skip(first) {
                let rect = row_rect(index);
                match row.kind {
                    RowKind::Section(section) => section_row(ui, rect, row, section, state),
                    RowKind::Entity(entity) => {
                        let row = EntityRow {
                            entity,
                            row,
                            focused,
                            matched: filter.is_none_or(|filter| filter.matches.contains(&entity)),
                        };
                        entity_row(ui, info, rect, row, selected, state, tree_id, actions);
                    }
                }
            }
        });
}

/// ↑ and ↓ select the entity above or below; → expands the selected entity,
/// or moves to its first child; ← collapses it, or moves to its parent. As in
/// Chrome's Elements panel.
fn keyboard(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    rows: &[Row],
    selected: &mut Option<Entity>,
    state: &mut TreeState,
    tree_id: egui::Id,
) {
    ui.memory_mut(|memory| {
        // Keeps egui from moving the focus to another widget on arrow keys.
        memory.set_focus_lock_filter(
            tree_id,
            egui::EventFilter {
                horizontal_arrows: true,
                vertical_arrows: true,
                ..Default::default()
            },
        );
    });
    let [up, down, left, right] = ui.input_mut(|input| {
        [
            egui::Key::ArrowUp,
            egui::Key::ArrowDown,
            egui::Key::ArrowLeft,
            egui::Key::ArrowRight,
        ]
        .map(|key| input.consume_key(egui::Modifiers::NONE, key))
    });
    let entity_at = |index: usize| match rows.get(index)?.kind {
        RowKind::Entity(entity) => Some(entity),
        RowKind::Section(_) => None,
    };
    let current = selected.and_then(|selected| {
        rows.iter()
            .position(|row| row.kind == RowKind::Entity(selected))
    });
    let Some(index) = current else {
        if down {
            *selected = (0..rows.len()).find_map(entity_at);
        }
        return;
    };
    let row = &rows[index];
    let RowKind::Entity(entity) = row.kind else {
        return;
    };
    if up && let Some(above) = (0..index).rev().find_map(entity_at) {
        *selected = Some(above);
    } else if down && let Some(below) = (index + 1..rows.len()).find_map(entity_at) {
        *selected = Some(below);
    } else if right && row.children > 0 {
        if row.open {
            *selected = entity_at(index + 1);
        } else {
            state.set_expanded(info, entity, true, false);
        }
    } else if left {
        if row.open {
            state.set_expanded(info, entity, false, false);
        } else if let Some(parent) = info.parent(entity) {
            *selected = Some(parent);
        }
    }
}

fn section_row(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    row: &Row,
    section: Section,
    state: &mut TreeState,
) {
    let id = egui::Id::new(("devtools_entities_section", section as usize));
    let response = ui
        .interact(rect, id, egui::Sense::click())
        .on_hover_text(section.description());
    if response.clicked() {
        state.open_sections[section as usize] ^= true;
    }
    let visuals = ui.visuals();
    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::ZERO, visuals.faint_bg_color);
    let triangle = egui::pos2(rect.left() + INDENT / 2.0, rect.center().y);
    let color = if response.hovered() {
        visuals.strong_text_color()
    } else {
        visuals.weak_text_color()
    };
    paint_triangle(painter, triangle, row.open, color);
    let font = egui::TextStyle::Body.resolve(ui.style());
    let mut job = egui::text::LayoutJob::default();
    job.append(
        section.title(),
        0.0,
        egui::TextFormat::simple(font.clone(), visuals.strong_text_color()),
    );
    job.append(
        &row.children.to_string(),
        8.0,
        egui::TextFormat::simple(font, visuals.weak_text_color()),
    );
    let galley = painter.layout_job(job);
    let pos = egui::pos2(
        rect.left() + INDENT,
        rect.center().y - galley.size().y / 2.0,
    );
    painter.galley(pos, galley, visuals.text_color());
}

struct EntityRow<'a> {
    entity: Entity,
    row: &'a Row,
    /// Whether the tree has the keyboard focus, which colors the selection.
    focused: bool,
    /// `false` for an ancestor shown only to keep a match in place.
    matched: bool,
}

/// An entity's row: its disclosure triangle, its title followed by its
/// rarest components, and its id at the right end.
#[expect(clippy::too_many_arguments, reason = "the tree's drawing state")]
fn entity_row(
    ui: &mut egui::Ui,
    info: &WorldInfo,
    rect: egui::Rect,
    row: EntityRow,
    selected: &mut Option<Entity>,
    state: &mut TreeState,
    tree_id: egui::Id,
    actions: &mut Vec<Action>,
) {
    let EntityRow {
        entity,
        row,
        focused,
        matched,
    } = row;
    let Some(label) = info.label(entity) else {
        return;
    };
    let id = egui::Id::new(("devtools_entities_row", entity));
    let response = ui.interact(rect, id, egui::Sense::click());
    let visuals = ui.visuals().clone();
    let painter = ui.painter();
    let background = if *selected == Some(entity) {
        // Gray, as Chrome's, while the keyboard focus is elsewhere.
        if focused {
            visuals.selection.bg_fill
        } else {
            visuals.widgets.inactive.weak_bg_fill
        }
    } else if response.hovered() {
        visuals.widgets.hovered.weak_bg_fill
    } else {
        egui::Color32::TRANSPARENT
    };
    painter.rect_filled(rect, egui::CornerRadius::ZERO, background);

    let triangle_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left() + row.depth as f32 * INDENT, rect.top()),
        egui::vec2(INDENT, rect.height()),
    );
    // Registered after the row, so that it takes the clicks on it.
    let toggled = row.children > 0 && {
        let triangle = ui.interact(triangle_rect, id.with("toggle"), egui::Sense::click());
        let color = if triangle.hovered() {
            visuals.strong_text_color()
        } else {
            visuals.weak_text_color()
        };
        paint_triangle(ui.painter(), triangle_rect.center(), row.open, color);
        triangle.clicked()
    };
    if toggled || (row.children > 0 && response.double_clicked()) {
        // ⌥-click expands or collapses the whole subtree, as in Chrome.
        let recursive = ui.input(|input| input.modifiers.alt);
        state.set_expanded(info, entity, !row.open, recursive);
    } else if response.clicked() {
        *selected = Some(entity);
        ui.memory_mut(|memory| memory.request_focus(tree_id));
    }

    let painter = ui.painter();
    let id_galley = painter.layout_no_wrap(
        entity.to_string(),
        egui::TextStyle::Monospace.resolve(ui.style()),
        visuals.weak_text_color(),
    );
    let id_pos = egui::pos2(
        rect.right() - 6.0 - id_galley.size().x,
        rect.center().y - id_galley.size().y / 2.0,
    );
    let text_left = triangle_rect.right() + 2.0;
    let mut job = label_job(ui.style(), info, &label, !matched);
    job.wrap = egui::text::TextWrapping::truncate_at_width((id_pos.x - 8.0 - text_left).max(0.0));
    let galley = painter.layout_job(job);
    let text_pos = egui::pos2(text_left, rect.center().y - galley.size().y / 2.0);
    painter.galley(text_pos, galley, visuals.text_color());
    painter.galley(id_pos, id_galley, visuals.weak_text_color());

    response
        .on_hover_ui(|ui| label_tooltip(ui, info, &label))
        .context_menu(|ui| {
            if row.children > 0 {
                if ui.button("Expand recursively").clicked() {
                    state.set_expanded(info, entity, true, true);
                }
                if ui.button("Collapse children").clicked() {
                    state.set_expanded(info, entity, false, true);
                }
                ui.separator();
            }
            entity_menu(ui, info, entity, &label, actions);
        });
}

/// The entity's title in the tag color, followed by its rarest components in
/// the attribute color, as Chrome shows an element's tag and attributes.
/// Disabled entities are in italics. With `dimmed`, all of it is weak.
fn label_job(
    style: &egui::Style,
    info: &WorldInfo,
    label: &Label,
    dimmed: bool,
) -> egui::text::LayoutJob {
    let visuals = &style.visuals;
    let weak = visuals.weak_text_color();
    let (tag, attribute) = if dimmed || label.disabled {
        (weak, weak)
    } else {
        (Syntax::Tag.color(visuals), Syntax::Attribute.color(visuals))
    };
    let font = egui::TextStyle::Body.resolve(style);
    let format = |color| egui::TextFormat {
        font_id: font.clone(),
        color,
        italics: label.disabled,
        ..Default::default()
    };
    let mut job = egui::text::LayoutJob::default();
    if let Some(title) = &label.title {
        job.append(title, 0.0, format(tag));
    }
    for &id in label.components.iter().take(ROW_COMPONENTS) {
        let space = if job.is_empty() { 0.0 } else { 6.0 };
        job.append(info.name(id), space, format(attribute));
    }
    let more = label.components.len().saturating_sub(ROW_COMPONENTS);
    if more > 0 {
        job.append(&format!("+{more}"), 6.0, format(weak));
    }
    if job.is_empty() {
        job.append("empty", 0.0, format(weak));
    }
    job
}
