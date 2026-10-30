//! The graph the tab draws, read from a schedule: its systems, what orders
//! them, and which of them conflict with nothing ordering them.
//!
//! Bevy's executor runs a graph of systems alone: an ordering between two
//! sets orders every system of the first before every system of the second,
//! and orderings that others imply are dropped. Bevy keeps that graph to
//! itself, so [`read`] builds it again the way Bevy does, from the
//! schedule's hierarchy of sets and the orderings declared between systems
//! and sets. The edges Bevy's build passes add while building the schedule,
//! such as those of the sync points Bevy adds, are only in Bevy's graph;
//! [`ScheduleRecords`](super::ScheduleRecords) keeps them.
//!
//! One difference remains: Bevy drops the orderings declared with
//! `chain_weak`, `before_weak` and `after_weak` between systems that don't
//! conflict, and doesn't say which orderings those are, so they are kept
//! here. Nothing in dust declares one.
//!
//! The sets read are the named ones: a system's own set, which
//! `after(system)` orders against, and an anonymous set, which a tuple of
//! systems gets for conditions run once for all of them, only stand for
//! their systems.

use std::collections::VecDeque;

use bevy::ecs::component::{ComponentId, Components};
use bevy::ecs::query::ComponentAccessKind;
use bevy::ecs::schedule::graph::{DagGroups, Direction};
use bevy::ecs::schedule::{
    ApplyDeferred, InternedScheduleLabel, NodeId, Schedule, ScheduleGraph, SystemKey, SystemSetKey,
    SystemWithAccess,
};
use bevy::ecs::system::{System as _, SystemAccess};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::utils::prelude::ShortName;
use gpui::SharedString;

/// The most declared orderings an edge lists as its reasons.
const MAX_REASONS: usize = 8;

/// A schedule's systems and the orderings between them, as Bevy's executor
/// runs them.
pub(super) struct GraphSnapshot {
    pub label: InternedScheduleLabel,
    /// The systems, in the order the executor would run them one at a time.
    pub systems: Vec<SystemInfo>,
    /// `before` runs before `after`: indices into `systems`. Without the
    /// orderings that others imply.
    pub edges: Vec<Edge>,
    /// The named sets, each after the sets it's in.
    pub sets: Vec<SetInfo>,
    /// The systems that conflict, with nothing ordering them; Bevy runs them
    /// in either order.
    pub ambiguities: Vec<Ambiguity>,
    pub state: BuildState,
    /// Why the orderings couldn't be read: a cycle, which keeps Bevy from
    /// building the schedule, too. `edges` is then empty.
    pub error: Option<SharedString>,
}

/// How much of a schedule Bevy has built, which is what tells the systems'
/// access and the sync points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum BuildState {
    Built,
    /// Built, then given systems or orderings that it'll be rebuilt with
    /// when it next runs.
    Changed,
    /// Not run yet, so not built.
    NotBuilt,
}

pub(super) struct SystemInfo {
    pub key: SystemKey,
    /// The short type name, such as `propagate_parent_transforms`.
    pub name: SharedString,
    pub path: SharedString,
    /// The crate the system is defined in.
    pub krate: SharedString,
    pub kind: SystemKind,
    /// Whether the system has `Commands` or another buffer that a sync
    /// point applies.
    pub deferred: bool,
    /// Whether the system has to run on the main thread.
    pub non_send: bool,
    /// The named sets the system is in, directly or through other sets,
    /// nearest first, as indices into `sets`.
    pub sets: Vec<usize>,
    /// The named sets it's in directly, or through sets that only stand for
    /// their systems.
    pub parents: Vec<usize>,
    /// What the system reads and writes; `None` until Bevy has built the
    /// schedule since the system was added.
    pub access: Option<AccessInfo>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SystemKind {
    System,
    /// Has `&mut World`, so runs alone.
    Exclusive,
    /// An `ApplyDeferred`: applies the commands of the systems before it.
    SyncPoint,
}

#[derive(Clone, Default, Debug)]
pub(super) struct AccessInfo {
    /// What the system accesses when it isn't a list of components: the
    /// whole world, or all components.
    pub all: Option<&'static str>,
    /// The components and resources the system reads and doesn't write.
    pub reads: Vec<SharedString>,
    pub writes: Vec<SharedString>,
}

/// A named set.
pub(super) struct SetInfo {
    pub key: SystemSetKey,
    /// As Bevy prints it: `Propagate` for `TransformSystems::Propagate`.
    pub name: SharedString,
    /// The named sets it's in directly, or through anonymous sets.
    pub parents: Vec<usize>,
    /// The systems in it, directly or through other sets, in the order of
    /// `systems`.
    pub systems: Vec<usize>,
    /// What it's declared to run after, and before, with `before`, `after`
    /// or `chain`; not what the sets it's in are.
    pub after: Vec<Neighbor>,
    pub before: Vec<Neighbor>,
}

/// A system or set that a set is ordered against.
pub(super) struct Neighbor {
    pub name: SharedString,
    /// What selecting it shows: `None` for a set that stands for several
    /// systems.
    pub target: Option<Target>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Target {
    System(usize),
    Set(usize),
}

pub(super) struct Edge {
    pub before: usize,
    pub after: usize,
    /// The orderings the edge comes from.
    pub reasons: Vec<Reason>,
}

#[derive(Clone, PartialEq, Debug)]
pub(super) enum Reason {
    /// `before` was ordered before `after`, each a system or a set holding
    /// the edge's system, by `before`, `after` or `chain`.
    Declared {
        before: SharedString,
        after: SharedString,
    },
    /// Bevy added the edge while building the schedule: to a sync point it
    /// added to apply the commands of the systems before it, or from one to
    /// the systems after it, unless the app adds build passes of its own.
    Added,
}

pub(super) struct Ambiguity {
    pub a: usize,
    pub b: usize,
    /// The components and resources one of them writes and the other
    /// accesses; empty when they conflict over the whole world.
    pub components: Vec<SharedString>,
}

/// Reads `schedule`'s graph. `added_edges` are the edges Bevy's build passes
/// added when it last built the schedule.
pub(super) fn read(
    schedule: &Schedule,
    components: &Components,
    added_edges: &[(SystemKey, SystemKey)],
) -> GraphSnapshot {
    let graph = schedule.graph();
    // Bevy moves the systems out of the graph when it builds the schedule:
    // the ones built are the executor's, in its order, and the ones added
    // since are still in the graph.
    let built = schedule.systems_with_access();
    let state = match (&built, schedule.is_changed()) {
        (Err(_), _) => BuildState::NotBuilt,
        (Ok(_), true) => BuildState::Changed,
        (Ok(_), false) => BuildState::Built,
    };
    let mut entries: Vec<(SystemKey, &SystemWithAccess, bool)> = Vec::new();
    if let Ok(built) = built {
        entries.extend(built.map(|(key, system)| (key, system, true)));
    }
    for (key, _, _) in graph.systems.iter() {
        if let Some(system) = graph.systems.get(key) {
            entries.push((key, system, false));
        }
    }
    let index: HashMap<SystemKey, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, &(key, ..))| (key, i))
        .collect();

    let mut snapshot = GraphSnapshot {
        label: schedule.label(),
        systems: entries
            .iter()
            .map(|&(key, system, initialized)| system_info(key, system, initialized, components))
            .collect(),
        edges: Vec::new(),
        sets: Vec::new(),
        ambiguities: Vec::new(),
        state,
        error: None,
    };
    let hierarchy = graph.hierarchy().graph();
    let toposort = match hierarchy.toposort(Vec::new()) {
        Ok(toposort) => toposort,
        Err(error) => {
            snapshot.error = Some(format!("The sets contain each other: {error}").into());
            return snapshot;
        }
    };
    let set_keys: Vec<SystemSetKey> = toposort
        .iter()
        .filter_map(|&node| match node {
            NodeId::Set(key) if named(graph, key) => Some(key),
            _ => None,
        })
        .collect();
    let set_index: HashMap<SystemSetKey, usize> = set_keys
        .iter()
        .enumerate()
        .map(|(i, &key)| (key, i))
        .collect();
    for system in &mut snapshot.systems {
        let node = NodeId::System(system.key);
        system.sets = sets_of(graph, node, &set_index, true);
        system.parents = sets_of(graph, node, &set_index, false);
    }
    let names = Names {
        graph,
        systems: &snapshot.systems,
        index: &index,
    };

    // Each set's systems, from the sets in it too.
    let groups: DagGroups<SystemSetKey, SystemKey> = DagGroups::new(hierarchy, &toposort);
    let target = |node: NodeId| match node {
        NodeId::System(key) => index.get(&key).map(|&i| Target::System(i)),
        NodeId::Set(key) => match set_index.get(&key) {
            Some(&i) => Some(Target::Set(i)),
            // A system's own set stands for its system, if there's one.
            None => match groups.get(&key) {
                Some(systems) if systems.len() == 1 => {
                    index.get(&systems[0]).map(|&i| Target::System(i))
                }
                _ => None,
            },
        },
    };
    let dependency = graph.dependency().graph();
    let neighbors = |key: SystemSetKey, direction: Direction| {
        let mut neighbors: Vec<Neighbor> = dependency
            .neighbors_directed(NodeId::Set(key), direction)
            .map(|node| Neighbor {
                name: names.node(node),
                target: target(node),
            })
            .collect();
        neighbors.sort_by_cached_key(|neighbor| neighbor.name.to_lowercase());
        neighbors
    };
    let sets: Vec<SetInfo> = set_keys
        .iter()
        .map(|&key| {
            let mut systems: Vec<usize> = groups
                .get(&key)
                .into_iter()
                .flatten()
                .filter_map(|system| index.get(system).copied())
                .collect();
            systems.sort_unstable();
            SetInfo {
                key,
                name: names.node(NodeId::Set(key)),
                parents: sets_of(graph, NodeId::Set(key), &set_index, false),
                systems,
                after: neighbors(key, Direction::Incoming),
                before: neighbors(key, Direction::Outgoing),
            }
        })
        .collect();
    // Each ordering of a set applies to its systems.
    let mut flat = groups.flatten(graph.dependency().clone(), |_, _, _, _| {});
    for &key in index.keys() {
        flat.add_node(key);
    }
    let added: HashSet<(SystemKey, SystemKey)> = added_edges
        .iter()
        .copied()
        .filter(|(a, b)| index.contains_key(a) && index.contains_key(b))
        .collect();
    for &(a, b) in &added {
        flat.add_edge(a, b);
    }
    // A sync point replaces the edge it splits, which the reduction drops.
    let analysis = match flat.analyze() {
        Ok(analysis) => analysis,
        Err(error) => {
            snapshot.error = Some(format!("The orderings form a cycle: {error}").into());
            return snapshot;
        }
    };
    let mut edge_index: HashMap<(SystemKey, SystemKey), usize> = HashMap::default();
    for (a, b) in analysis.transitive_reduction().all_edges() {
        let (Some(&before), Some(&after)) = (index.get(&a), index.get(&b)) else {
            continue;
        };
        edge_index.insert((a, b), snapshot.edges.len());
        snapshot.edges.push(Edge {
            before,
            after,
            reasons: if added.contains(&(a, b)) {
                vec![Reason::Added]
            } else {
                Vec::new()
            },
        });
    }

    // The orderings each edge comes from: the declared orderings between
    // nodes holding its two systems. An ordering whose edge a sync point
    // split goes to the two edges through the sync point.
    let systems_of = |node: NodeId| -> Vec<SystemKey> {
        match node {
            NodeId::System(key) => vec![key],
            NodeId::Set(key) => groups
                .get(&key)
                .map(|systems| systems.iter().copied().collect())
                .unwrap_or_default(),
        }
    };
    let mut added_after: HashMap<SystemKey, Vec<SystemKey>> = HashMap::default();
    for &(a, b) in &added {
        added_after.entry(a).or_default().push(b);
    }
    for (from, to) in graph.dependency().graph().all_edges() {
        let (befores, afters) = (systems_of(from), systems_of(to));
        // Named once it's known to order an edge.
        let mut declared = None;
        for &a in &befores {
            for &b in &afters {
                let mut ordered = Vec::new();
                match edge_index.get(&(a, b)) {
                    Some(&e) => ordered.push(e),
                    None => {
                        for &sync in added_after.get(&a).into_iter().flatten() {
                            if let (Some(&into), Some(&out)) =
                                (edge_index.get(&(a, sync)), edge_index.get(&(sync, b)))
                            {
                                ordered.extend([into, out]);
                            }
                        }
                    }
                }
                for e in ordered {
                    let declared = declared.get_or_insert_with(|| Reason::Declared {
                        before: names.node(from),
                        after: names.node(to),
                    });
                    let reasons = &mut snapshot.edges[e].reasons;
                    if reasons.len() < MAX_REASONS && !reasons.contains(declared) {
                        reasons.push(declared.clone());
                    }
                }
            }
        }
    }

    snapshot.sets = sets;
    snapshot.ambiguities = graph
        .conflicting_systems()
        .iter()
        .filter_map(|(a, b, conflicts)| {
            Some(Ambiguity {
                a: *index.get(a)?,
                b: *index.get(b)?,
                components: conflicts
                    .iter()
                    .map(|&id| component_name(components, id))
                    .collect(),
            })
        })
        .collect();
    snapshot
}

fn system_info(
    key: SystemKey,
    system: &SystemWithAccess,
    initialized: bool,
    components: &Components,
) -> SystemInfo {
    let path = system.name().to_string();
    let kind = if system.system_type() == std::any::TypeId::of::<ApplyDeferred>() {
        SystemKind::SyncPoint
    } else if system.access().is_exclusive() {
        SystemKind::Exclusive
    } else {
        SystemKind::System
    };
    SystemInfo {
        key,
        name: short_name(&path).into(),
        krate: crate_of(&path).to_owned().into(),
        path: path.into(),
        kind,
        deferred: initialized && system.has_deferred(),
        non_send: initialized && !system.is_send(),
        sets: Vec::new(),
        parents: Vec::new(),
        access: initialized.then(|| access_info(system.access(), components)),
    }
}

fn access_info(access: &SystemAccess, components: &Components) -> AccessInfo {
    let mut info = AccessInfo::default();
    match access {
        SystemAccess::None => {}
        SystemAccess::Exclusive => info.all = Some("The whole world, exclusively"),
        SystemAccess::Shared(set) => {
            let combined = set.combined_access();
            match combined.try_iter_access() {
                Ok(access) => {
                    for kind in access {
                        match kind {
                            ComponentAccessKind::Shared(id) => {
                                info.reads.push(component_name(components, id))
                            }
                            ComponentAccessKind::Exclusive(id) => {
                                info.writes.push(component_name(components, id))
                            }
                            // Only filters by the component, as `With` does.
                            ComponentAccessKind::Archetypal(_) => {}
                        }
                    }
                }
                Err(_) => {
                    info.all = Some(if combined.has_write_all() {
                        "All components, mutably"
                    } else if combined.has_read_all() {
                        "All components"
                    } else {
                        "All components but some"
                    })
                }
            }
        }
    }
    for names in [&mut info.reads, &mut info.writes] {
        names.sort();
        names.dedup();
    }
    info
}

fn component_name(components: &Components, id: ComponentId) -> SharedString {
    match components.get_name(id) {
        Some(name) => name.shortname().to_string().into(),
        None => format!("{id:?}").into(),
    }
}

/// The type name without its path. A closure's is that of the function it's
/// in, as `build_app::{{closure}}`, for something to tell it apart by.
fn short_name(path: &str) -> String {
    let short = ShortName(path).to_string();
    if !short.starts_with("{{closure}}") {
        return short;
    }
    match path.rsplit("::").nth(1) {
        Some(function) => format!("{function}::{short}"),
        None => short,
    }
}

/// The crate a type path starts in: `bevy_time` for `bevy_time::time_system`,
/// and the first system's for a combination such as `Pipe(a, b)`.
fn crate_of(path: &str) -> &str {
    let mut rest = path;
    loop {
        rest = rest.trim_start_matches(['<', '&', ' ']);
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        match rest[end..].strip_prefix('(') {
            Some(inner) => rest = inner,
            None => return &rest[..end],
        }
    }
}

/// How the systems and sets of a schedule's graph are named.
struct Names<'a> {
    graph: &'a ScheduleGraph,
    systems: &'a [SystemInfo],
    index: &'a HashMap<SystemKey, usize>,
}

impl Names<'_> {
    /// A system's short name, or a set's: as it's declared, by its systems for
    /// a system's own set, which `after(system)` orders against, and by its
    /// members for an anonymous set, which a tuple of systems gets for
    /// conditions run once for all of them.
    fn node(&self, node: NodeId) -> SharedString {
        match node {
            NodeId::System(key) => self.system(key),
            NodeId::Set(key) => {
                let Some(set) = self.graph.system_sets.get(key) else {
                    return SharedString::new_static("?");
                };
                if named(self.graph, key) {
                    return format!("{set:?}").into();
                }
                let members: Vec<SharedString> = self
                    .graph
                    .hierarchy()
                    .graph()
                    .neighbors_directed(node, Direction::Outgoing)
                    .map(|member| self.node(member))
                    .collect();
                match (set.is_anonymous(), &members[..]) {
                    (false, [system]) => system.clone(),
                    (false, _) => {
                        let name = format!("{set:?}");
                        let name = name.trim_start_matches("SystemTypeSet:");
                        ShortName(name).to_string().into()
                    }
                    (true, members) if members.len() > 3 => {
                        format!("({}, …)", members[..3].join(", ")).into()
                    }
                    (true, members) => format!("({})", members.join(", ")).into(),
                }
            }
        }
    }

    fn system(&self, key: SystemKey) -> SharedString {
        match self.index.get(&key) {
            Some(&i) => self.systems[i].name.clone(),
            None => SharedString::new_static("?"),
        }
    }
}

/// Whether `key` is a named set, rather than one that only stands for its
/// systems.
fn named(graph: &ScheduleGraph, key: SystemSetKey) -> bool {
    graph
        .system_sets
        .get(key)
        .is_some_and(|set| set.system_type().is_none() && !set.is_anonymous())
}

/// The named sets `node` is in, nearest first, as their indices in
/// `set_index`: directly or through sets that only stand for their systems,
/// and with `through_named`, through named sets too.
fn sets_of(
    graph: &ScheduleGraph,
    node: NodeId,
    set_index: &HashMap<SystemSetKey, usize>,
    through_named: bool,
) -> Vec<usize> {
    let hierarchy = graph.hierarchy().graph();
    let mut seen: HashSet<NodeId> = HashSet::default();
    let mut queue: VecDeque<NodeId> = VecDeque::from([node]);
    let mut sets = Vec::new();
    while let Some(node) = queue.pop_front() {
        for parent in hierarchy.neighbors_directed(node, Direction::Incoming) {
            if !seen.insert(parent) {
                continue;
            }
            let named = match parent {
                NodeId::Set(key) => set_index.get(&key).copied(),
                NodeId::System(_) => None,
            };
            if let Some(set) = named {
                sets.push(set);
            }
            if named.is_none() || through_named {
                queue.push_back(parent);
            }
        }
    }
    sets
}
