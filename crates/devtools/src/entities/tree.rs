//! The entity tree: each section's root entities, with their children nested
//! below them by `ChildOf` as Chrome nests the DOM, a filter field above, and
//! the selected entity's ancestors as breadcrumbs below.

use bevy::ecs::entity::EntityHashSet;
use bevy::ecs::observer::Observer;
use bevy::ecs::resource::IsResource;
use bevy::prelude::*;
use gpui::SharedString;

use super::{Label, Section, WorldInfo, contains_ignore_case};

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

/// The arrow keys, which move through the tree as in Chrome's Elements panel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TreeKey {
    Up,
    Down,
    Left,
    Right,
}

impl TreeState {
    pub(super) fn toggle_section(&mut self, section: Section) {
        self.open_sections[section as usize] ^= true;
    }

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

    /// ↑ and ↓ select the entity above or below; → expands the selected entity,
    /// or moves to its first child; ← collapses it, or moves to its parent.
    pub(super) fn key(&mut self, info: &WorldInfo, key: TreeKey, selected: &mut Option<Entity>) {
        let filter = Filter::new(info, &self.filter);
        let rows = self.rows(info, filter.as_ref());
        let entity_at = |index: usize| match rows.get(index)?.kind {
            RowKind::Entity(entity) => Some(entity),
            RowKind::Section(_) => None,
        };
        let current = selected.and_then(|selected| {
            rows.iter()
                .position(|row| row.kind == RowKind::Entity(selected))
        });
        let Some(index) = current else {
            if key == TreeKey::Down {
                *selected = (0..rows.len()).find_map(entity_at);
            }
            return;
        };
        let row = &rows[index];
        let RowKind::Entity(entity) = row.kind else {
            return;
        };
        match key {
            TreeKey::Up => {
                if let Some(above) = (0..index).rev().find_map(entity_at) {
                    *selected = Some(above);
                }
            }
            TreeKey::Down => {
                if let Some(below) = (index + 1..rows.len()).find_map(entity_at) {
                    *selected = Some(below);
                }
            }
            TreeKey::Right if row.children > 0 => {
                if row.open {
                    *selected = entity_at(index + 1);
                } else {
                    self.set_expanded(info, entity, true, false);
                }
            }
            TreeKey::Right => {}
            TreeKey::Left => {
                if row.open {
                    self.set_expanded(info, entity, false, false);
                } else if let Some(parent) = info.parent(entity) {
                    *selected = Some(parent);
                }
            }
        }
    }

    /// What the tree shows. With `reveal`, the selected entity's ancestors are
    /// expanded, and the snapshot asks for its row to be scrolled into view.
    pub(super) fn snapshot(
        &mut self,
        info: &WorldInfo,
        selected: Option<Entity>,
        reveal: bool,
    ) -> TreeSnapshot {
        self.expanded
            .retain(|&entity| info.world.entities().contains_spawned(entity));
        if reveal && let Some(entity) = selected {
            self.expand_to(info, entity);
        }
        let filter = Filter::new(info, &self.filter);
        let rows: Vec<RowView> = self
            .rows(info, filter.as_ref())
            .into_iter()
            .filter_map(|row| {
                Some(match row.kind {
                    RowKind::Section(section) => RowView::Section {
                        section,
                        roots: row.children,
                        open: row.open,
                    },
                    RowKind::Entity(entity) => RowView::Entity {
                        entity,
                        depth: row.depth,
                        children: row.children,
                        open: row.open,
                        label: info.label(entity)?,
                        matched: filter
                            .as_ref()
                            .is_none_or(|filter| filter.matches.contains(&entity)),
                    },
                })
            })
            .collect();
        let count = match &filter {
            Some(filter) => format!("{} found", filter.matches.len()),
            None => format!("{} entities", info.entity_count),
        };
        let breadcrumbs = match selected.filter(|&entity| info.world.get_entity(entity).is_ok()) {
            Some(entity) => {
                let mut path: Vec<Entity> = info.ancestors(entity).collect();
                path.reverse();
                path.push(entity);
                path.into_iter()
                    .filter_map(|entity| Some((entity, info.label(entity)?)))
                    .collect()
            }
            None => Vec::new(),
        };
        let reveal = reveal
            .then(|| {
                let selected = selected?;
                rows.iter().position(
                    |row| matches!(row, RowView::Entity { entity, .. } if *entity == selected),
                )
            })
            .flatten();
        TreeSnapshot {
            rows,
            count: count.into(),
            selected,
            breadcrumbs,
            reveal,
        }
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
                        .any(|id| contains_ignore_case(&info.name(id), word))
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

/// What the tree shows, sent to the UI thread.
pub(super) struct TreeSnapshot {
    pub(super) rows: Vec<RowView>,
    /// How many entities there are, or match the filter.
    pub(super) count: SharedString,
    pub(super) selected: Option<Entity>,
    /// The selected entity and its ancestors, from the root down.
    pub(super) breadcrumbs: Vec<(Entity, Label)>,
    /// The row to scroll into view: the selected entity's, after the
    /// selection changed.
    pub(super) reveal: Option<usize>,
}

pub(super) enum RowView {
    Section {
        section: Section,
        /// How many root entities the section has, or match the filter.
        roots: usize,
        open: bool,
    },
    Entity {
        entity: Entity,
        /// 1 for a root.
        depth: usize,
        children: usize,
        /// Whether its children show below it.
        open: bool,
        label: Label,
        /// `false` for an ancestor shown only to keep a match in place.
        matched: bool,
    },
}
