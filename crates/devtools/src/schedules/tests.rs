use bevy::ecs::schedule::graph::DiGraph;
use bevy::ecs::schedule::{
    ApplyDeferred, FlattenedDependencies, NodeId, ScheduleBuildError, ScheduleBuildPass,
    ScheduleGraph, ScheduleLabel, SystemKey, SystemSetKey,
};
use bevy::ecs::system::ScheduleSystem;
use bevy::platform::hash::FixedHasher;
use gpui::{Bounds, Point, Size, size};
use indexmap::IndexSet;

use super::graph::{BuildState, GraphSnapshot, Reason, SystemKind, Target};
use super::layout::{GRID_HEADING, Layout, SET_PADDING, Tree, layout, tree};
use super::ui::tree_of;
use super::*;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum Stage {
    Early,
    Late,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct Outer;

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
struct Other;

#[derive(Resource, Default)]
struct Score(u32);

fn a() {}
fn b() {}
fn c() {}
fn d() {}

fn spawn(mut commands: Commands) {
    commands.spawn_empty();
}

fn count(entities: Query<Entity>) {
    let _ = entities.iter().count();
}

fn raise(mut score: ResMut<Score>) {
    score.0 += 1;
}

fn lower(mut score: ResMut<Score>) {
    score.0 = score.0.saturating_sub(1);
}

fn exclusive(_: &mut World) {}

fn begin() {}

/// A build pass like pumicite's, which Bevy runs as it builds the schedule:
/// it orders `begin`, which is in `Outer` like any other system, before the
/// set's other systems as Bevy collapses the set.
#[derive(Debug)]
struct BeginPass {
    begin: SystemKey,
}

impl ScheduleBuildPass for BeginPass {
    type EdgeOptions = ();

    fn add_dependency(&mut self, _: NodeId, _: NodeId, _: Option<&()>) {}

    fn collapse_set(
        &mut self,
        set: SystemSetKey,
        systems: &IndexSet<SystemKey, FixedHasher>,
        _: &mut World,
        graph: &mut ScheduleGraph,
        _: &DiGraph<NodeId>,
    ) -> impl Iterator<Item = (NodeId, NodeId)> {
        let outer = graph.system_sets.get(set) == Some(Outer.intern());
        systems
            .iter()
            .filter(|&&system| outer && system != self.begin)
            .map(|&system| (self.begin.into(), system.into()))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn build(
        &mut self,
        _: &mut World,
        _: &mut ScheduleGraph,
        _: FlattenedDependencies<'_>,
    ) -> Result<(), ScheduleBuildError> {
        Ok(())
    }
}

/// A world with the tab's records, as `SchedulesPlugin` adds them.
fn world() -> World {
    let mut world = World::new();
    world.init_resource::<ScheduleRecords>();
    world.init_resource::<Schedules>();
    world.init_resource::<Score>();
    world.add_observer(record_build);
    world
}

fn add<M>(world: &mut World, systems: impl IntoScheduleConfigs<ScheduleSystem, M>) {
    world
        .resource_mut::<Schedules>()
        .add_systems(Update, systems);
}

/// `Update`'s graph, after running it once to build it.
fn built(world: &mut World) -> GraphSnapshot {
    world.run_schedule(Update);
    graph(world)
}

fn graph(world: &World) -> GraphSnapshot {
    snapshot(world, Some(Update.intern()))
        .graph
        .expect("the world has `Update`")
}

/// Index of the system named `name`.
fn find(graph: &GraphSnapshot, name: &str) -> usize {
    graph
        .systems
        .iter()
        .position(|system| system.name == name)
        .unwrap_or_else(|| panic!("no system named {name}"))
}

/// The edges by the names of their systems, sorted.
fn edges(graph: &GraphSnapshot) -> Vec<(String, String)> {
    let name = |i: usize| match graph.systems[i].kind {
        SystemKind::SyncPoint => "sync".to_owned(),
        _ => graph.systems[i].name.to_string(),
    };
    let mut edges: Vec<(String, String)> = graph
        .edges
        .iter()
        .map(|edge| (name(edge.before), name(edge.after)))
        .collect();
    edges.sort();
    edges
}

/// The names of `sets`, sets of `graph`.
fn set_names(graph: &GraphSnapshot, sets: &[usize]) -> Vec<String> {
    sets.iter()
        .map(|&set| graph.sets[set].name.to_string())
        .collect()
}

/// Index of the set named `name`.
fn find_set(graph: &GraphSnapshot, name: &str) -> usize {
    graph
        .sets
        .iter()
        .position(|set| set.name == name)
        .unwrap_or_else(|| panic!("no set named {name}"))
}

fn pair(a: &str, b: &str) -> (String, String) {
    (a.to_owned(), b.to_owned())
}

fn declared(before: &str, after: &str) -> Reason {
    Reason::Declared {
        before: before.to_owned().into(),
        after: after.to_owned().into(),
    }
}

#[test]
fn orderings_of_sets_order_their_systems() {
    let mut world = world();
    world
        .resource_mut::<Schedules>()
        .configure_sets(Update, (Stage::Early, Stage::Late).chain());
    add(
        &mut world,
        (a.in_set(Stage::Early), b.in_set(Stage::Late), c.after(b)),
    );
    let graph = built(&mut world);

    assert_eq!(graph.state, BuildState::Built);
    assert_eq!(edges(&graph), [pair("a", "b"), pair("b", "c")]);
    let reasons = |before: &str, after: &str| {
        let (before, after) = (find(&graph, before), find(&graph, after));
        let edge = graph
            .edges
            .iter()
            .find(|edge| edge.before == before && edge.after == after)
            .unwrap();
        edge.reasons.clone()
    };
    assert_eq!(reasons("a", "b"), [declared("Early", "Late")]);
    // `after(b)` orders against `b`'s own set, which is named by `b`.
    assert_eq!(reasons("b", "c"), [declared("b", "c")]);
    assert_eq!(
        set_names(&graph, &graph.systems[find(&graph, "a")].sets),
        ["Early"]
    );
    assert!(graph.ambiguities.is_empty());
}

#[test]
fn orderings_implied_by_others_are_dropped() {
    let mut world = world();
    world
        .resource_mut::<Schedules>()
        .configure_sets(Update, (Stage::Early, Stage::Late).chain());
    add(
        &mut world,
        ((a, b).chain().in_set(Stage::Early), c.in_set(Stage::Late)),
    );
    let graph = built(&mut world);
    // `Early` before `Late` orders `a` before `c` too, through `b`.
    assert_eq!(edges(&graph), [pair("a", "b"), pair("b", "c")]);
}

#[test]
fn sync_points_split_the_edges_they_apply_commands_on() {
    let mut world = world();
    add(&mut world, (spawn, count).chain());
    let graph = built(&mut world);

    assert_eq!(
        edges(&graph),
        [pair("spawn", "sync"), pair("sync", "count")]
    );
    let sync = graph
        .systems
        .iter()
        .position(|system| system.kind == SystemKind::SyncPoint)
        .unwrap();
    assert!(graph.systems[find(&graph, "spawn")].deferred);
    // Both edges come from Bevy, for the chain.
    for edge in &graph.edges {
        assert!(edge.before == sync || edge.after == sync);
        assert_eq!(edge.reasons, [Reason::Added, declared("spawn", "count")]);
    }
}

#[test]
fn ambiguities_name_what_the_systems_conflict_over() {
    let mut world = world();
    add(&mut world, (raise, lower, a));
    let graph = built(&mut world);

    assert_eq!(graph.ambiguities.len(), 1);
    let ambiguity = &graph.ambiguities[0];
    let mut pair = [
        graph.systems[ambiguity.a].name.to_string(),
        graph.systems[ambiguity.b].name.to_string(),
    ];
    pair.sort();
    assert_eq!(pair, ["lower", "raise"]);
    assert_eq!(ambiguity.components, ["Score"]);
    let access = graph.systems[find(&graph, "raise")].access.clone().unwrap();
    assert!(access.writes.contains(&"Score".into()), "{access:?}");
    assert!(!access.reads.contains(&"Score".into()), "{access:?}");
}

#[test]
fn kinds_of_systems() {
    let mut world = world();
    add(&mut world, (exclusive, a, ApplyDeferred).chain());
    let graph = built(&mut world);
    assert_eq!(
        graph.systems[find(&graph, "exclusive")].kind,
        SystemKind::Exclusive
    );
    assert_eq!(graph.systems[find(&graph, "a")].kind, SystemKind::System);
    assert!(
        graph
            .systems
            .iter()
            .any(|system| system.kind == SystemKind::SyncPoint)
    );
    let access = graph.systems[find(&graph, "exclusive")]
        .access
        .clone()
        .unwrap();
    assert!(access.all.is_some());
}

#[test]
fn nested_sets_show_nearest_first() {
    let mut world = world();
    world
        .resource_mut::<Schedules>()
        .configure_sets(Update, Stage::Early.in_set(Outer));
    add(&mut world, a.in_set(Stage::Early));
    let graph = built(&mut world);
    let a = &graph.systems[find(&graph, "a")];
    assert_eq!(set_names(&graph, &a.sets), ["Early", "Outer"]);
    assert_eq!(set_names(&graph, &a.parents), ["Early"]);
}

#[test]
fn sets_are_read_with_their_systems_and_orderings() {
    let mut world = world();
    world.resource_mut::<Schedules>().configure_sets(
        Update,
        (
            Stage::Early.in_set(Outer),
            Stage::Late.in_set(Outer).after(Stage::Early).before(d),
        ),
    );
    add(
        &mut world,
        (
            a.in_set(Stage::Early),
            (b, c).run_if(|| true).in_set(Stage::Late),
            d,
        ),
    );
    let graph = built(&mut world);
    let (outer, early, late) = (
        find_set(&graph, "Outer"),
        find_set(&graph, "Early"),
        find_set(&graph, "Late"),
    );
    // Only the named sets, each after the sets it's in.
    assert_eq!(graph.sets.len(), 3);
    assert!(outer < early && outer < late);
    assert_eq!(graph.sets[late].parents, [outer]);
    let names = |systems: &[usize]| -> Vec<String> {
        systems
            .iter()
            .map(|&system| graph.systems[system].name.to_string())
            .collect()
    };
    let mut in_outer = names(&graph.sets[outer].systems);
    in_outer.sort();
    assert_eq!(in_outer, ["a", "b", "c"]);
    // In `Late` through the anonymous set the condition gives them, too.
    assert_eq!(graph.systems[find(&graph, "b")].parents, [late]);
    let targets = |neighbors: &[super::graph::Neighbor]| -> Vec<Option<Target>> {
        neighbors.iter().map(|neighbor| neighbor.target).collect()
    };
    assert_eq!(targets(&graph.sets[late].after), [Some(Target::Set(early))]);
    // `before(d)` orders against `d`'s own set, which stands for `d`.
    assert_eq!(
        targets(&graph.sets[late].before),
        [Some(Target::System(find(&graph, "d")))]
    );
    assert_eq!(graph.sets[late].before[0].name, "d");
}

#[test]
fn edges_build_passes_add_are_shown() {
    let mut world = world();
    add(&mut world, ((a, b).in_set(Outer), d.before(Outer)));
    let mut schedules = world.resource_mut::<Schedules>();
    let schedule = schedules.get_mut(Update).unwrap();
    let begin_key = schedule
        .graph_mut()
        .process_configs(begin.in_set(Outer).into_configs(), true)
        .nodes[0]
        .as_system()
        .unwrap();
    schedule.add_build_pass(BeginPass { begin: begin_key });
    let graph = built(&mut world);

    // The edges the pass returned as Bevy collapsed `Outer`, and `d`'s, which
    // `d.before(Outer)` gives `begin` as it does the set's other systems.
    assert_eq!(
        edges(&graph),
        [pair("begin", "a"), pair("begin", "b"), pair("d", "begin")]
    );
    for edge in &graph.edges {
        let reasons = match graph.systems[edge.before].name.as_ref() {
            "d" => vec![declared("d", "Outer")],
            _ => vec![Reason::Added],
        };
        assert_eq!(edge.reasons, reasons);
    }
}

#[test]
fn sync_points_are_drawn_in_the_set_of_the_systems_around_them() {
    let mut world = world();
    add(&mut world, (spawn, count).chain().in_set(Outer));
    let graph = built(&mut world);
    let sync = graph
        .systems
        .iter()
        .position(|system| system.kind == SystemKind::SyncPoint)
        .unwrap();
    assert!(graph.systems[sync].parents.is_empty());
    assert_eq!(tree_of(&graph).homes[sync], Some(find_set(&graph, "Outer")));
}

#[test]
fn schedules_not_run_yet_are_not_built() {
    let mut world = world();
    add(&mut world, (spawn, count).chain());
    let graph = graph(&world);

    assert_eq!(graph.state, BuildState::NotBuilt);
    // Without the sync point Bevy adds as it builds.
    assert_eq!(edges(&graph), [pair("spawn", "count")]);
    assert!(graph.systems.iter().all(|system| system.access.is_none()));
}

#[test]
fn systems_added_after_the_build_are_shown() {
    let mut world = world();
    add(&mut world, (a, b).chain());
    world.run_schedule(Update);
    add(&mut world, c.after(b));
    let graph = graph(&world);

    assert_eq!(graph.state, BuildState::Changed);
    assert_eq!(edges(&graph), [pair("a", "b"), pair("b", "c")]);
    assert!(graph.systems[find(&graph, "a")].access.is_some());
    assert!(graph.systems[find(&graph, "c")].access.is_none());
}

#[test]
fn running_schedules_are_listed_but_not_shown() {
    let mut world = world();
    add(&mut world, a);
    world.resource_mut::<Schedules>().add_systems(Other, b);
    let other = world
        .resource_mut::<Schedules>()
        .remove_temporarily(Other)
        .unwrap();

    let snapshot = snapshot(&world, Some(Other.intern()));
    let entry = snapshot
        .schedules
        .iter()
        .find(|entry| entry.label == Other.intern())
        .unwrap();
    assert_eq!(entry.systems, None);
    // `Update` instead.
    assert_eq!(snapshot.graph.unwrap().label, Update.intern());
    world.resource_mut::<Schedules>().reinsert(other);
}

#[test]
fn schedules_are_listed_in_the_order_they_run() {
    let mut world = world();
    add(&mut world, a);
    {
        let mut schedules = world.resource_mut::<Schedules>();
        schedules.add_systems(Other, b);
        schedules.add_systems(First, c);
        schedules.add_systems(Startup, d);
    }
    world.resource_mut::<ScheduleRecords>().order = vec![
        (Startup.intern(), Group::Startup),
        (First.intern(), Group::Main),
        (Update.intern(), Group::Main),
    ];

    let snapshot = snapshot(&world, None);
    let listed: Vec<(String, Group)> = snapshot
        .schedules
        .iter()
        .map(|entry| (entry.name.to_string(), entry.group))
        .collect();
    assert_eq!(
        listed,
        [
            ("Startup".to_owned(), Group::Startup),
            ("First".to_owned(), Group::Main),
            ("Update".to_owned(), Group::Main),
            ("Other".to_owned(), Group::Other),
        ]
    );
    // Nothing picked shows `Update`.
    assert_eq!(snapshot.graph.unwrap().label, Update.intern());
}

/// Node sizes for `layout`.
fn sizes(count: usize) -> Vec<Size<f32>> {
    (0..count)
        .map(|i| size(40.0 + 10.0 * (i % 3) as f32, 22.0))
        .collect()
}

/// Whether two nodes of `layout` overlap.
fn overlapping(nodes: &[gpui::Point<f32>], sizes: &[Size<f32>]) -> Option<(usize, usize)> {
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            let (a, b) = (nodes[i], nodes[j]);
            let apart = a.x + sizes[i].width <= b.x
                || b.x + sizes[j].width <= a.x
                || a.y + sizes[i].height <= b.y
                || b.y + sizes[j].height <= a.y;
            if !apart {
                return Some((i, j));
            }
        }
    }
    None
}

#[test]
fn layout_draws_each_system_after_the_systems_it_waits_for() {
    // A diamond with a long edge across it, a second part, and two systems
    // nothing orders.
    let edges = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (3, 4), (5, 6)];
    let sizes = sizes(9);
    let tree = Tree::flat(9);
    let layout = layout(&sizes, &edges, &tree, &[]);
    check_drawing(&layout, &tree, &sizes, &edges, "flat");

    for &(a, b) in &edges {
        assert!(
            layout.nodes[a].x + sizes[a].width < layout.nodes[b].x,
            "{a} isn't left of {b}"
        );
    }
    assert_eq!(overlapping(&layout.nodes, &sizes), None);
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    for (e, &(a, b)) in edges.iter().enumerate() {
        let route = route(&layout, e);
        let (first, last) = (route[0], route[route.len() - 1]);
        assert!(near(first.x, layout.nodes[a].x + sizes[a].width));
        assert!(near(first.y, layout.nodes[a].y + sizes[a].height / 2.0));
        assert!(near(last.x, layout.nodes[b].x));
        assert!(near(last.y, layout.nodes[b].y + sizes[b].height / 2.0));
        // From left to right.
        assert!(route.windows(2).all(|pair| pair[0].x <= pair[1].x));
    }
    // The edge from 0 to 3 passes through the column of 1 and 2.
    assert_eq!(route(&layout, 4).len(), 4);

    // The larger part on top, then the smaller, then the grid.
    let grid = layout.grid.as_ref().expect("7 and 8 are unordered").at;
    assert!(layout.nodes[0].y < layout.nodes[5].y);
    for node in 0..7 {
        assert!(layout.nodes[node].y + sizes[node].height <= grid.y);
    }
    for node in [7, 8] {
        assert!(layout.nodes[node].y >= grid.y + GRID_HEADING);
    }
    assert!(layout.size.width >= layout.nodes[4].x + sizes[4].width);
    assert!(layout.size.height >= layout.nodes[8].y + sizes[8].height);
}

#[test]
fn layout_keeps_columns_apart_at_any_width() {
    // Many systems waiting for one, and one waiting for many.
    let mut edges: Vec<(usize, usize)> = (1..20).map(|i| (0, i)).collect();
    edges.extend((1..20).map(|i| (i, 20)));
    let sizes = sizes(21);
    let layout = layout(&sizes, &edges, &Tree::flat(21), &[]);
    assert_eq!(overlapping(&layout.nodes, &sizes), None);
    // The fan stays centered on the system it leaves.
    let middle = |node: usize| layout.nodes[node].y + sizes[node].height / 2.0;
    let (top, bottom) = (middle(1).min(middle(19)), middle(1).max(middle(19)));
    assert!((middle(0) - (top + bottom) / 2.0).abs() < sizes[0].height);
}

#[test]
fn layout_of_nothing() {
    let layout = layout(&[], &[], &Tree::flat(0), &[]);
    assert!(layout.nodes.is_empty());
    assert!(layout.grid.is_none());
}

#[test]
fn tree_draws_each_system_and_set_in_the_deepest_set_it_is_in() {
    // Set 1 is in set 0, and set 2 beside them.
    let tree = tree(
        &[vec![1, 0], vec![0, 2], vec![], vec![0], vec![]],
        &[vec![], vec![0], vec![]],
        &[3, 2, 5],
        &[false, false, true, false, true],
        &[(0, 2), (2, 3)],
    );
    assert_eq!(tree.parents, [None, Some(0), None]);
    // 0 in the deeper of its sets, and 1 in the larger of its. 2 adopts the
    // innermost set holding 0 and 3, and 4, joined to nothing, none.
    assert_eq!(tree.homes, [Some(1), Some(2), Some(0), Some(0), None]);
}

#[test]
fn layout_draws_sets_around_their_systems() {
    // Set 1 in set 0, and set 2 beside them. 0 to 2 in set 1, 3 in set 0, 4
    // and 5 in set 2, and 6 and 7 in none.
    let tree = Tree {
        homes: vec![
            Some(1),
            Some(1),
            Some(1),
            Some(0),
            Some(2),
            Some(2),
            None,
            None,
        ],
        parents: vec![None, Some(0), None],
    };
    let edges = [(0, 1), (1, 2), (2, 3), (2, 6), (3, 4), (4, 5)];
    let sizes = sizes(8);
    let layout = layout(&sizes, &edges, &tree, &[30.0, 30.0, 300.0]);
    check_drawing(&layout, &tree, &sizes, &edges, "sets");

    assert!(layout.sets.iter().all(Option::is_some));
    // Wide enough for its name.
    assert!(layout.sets[2].unwrap().size.width >= 300.0 + 2.0 * SET_PADDING);
    // Out of set 1 into set 0, and out of both: the edges from 2 share the
    // line to set 1's border.
    assert_eq!(layout.edges[2].len(), 2);
    assert_eq!(layout.edges[3].len(), 3);
    assert_eq!(layout.edges[2][0], layout.edges[3][0]);
    // Out of set 0 into set 2.
    assert_eq!(layout.edges[4].len(), 3);
    assert_eq!(layout.edges[0].len(), 1);
    let grid = layout.grid.as_ref().expect("7 is unordered");
    assert_eq!(grid.count, 1);
}

#[test]
fn layout_draws_a_set_without_a_box_if_a_system_outside_runs_between_its_systems() {
    // 0 and 2 in set 0, and 1, in set 1, between them: set 1 keeps its box,
    // though the two sets wait for each other.
    let tree = Tree {
        homes: vec![Some(0), Some(1), Some(0)],
        parents: vec![None, None],
    };
    let edges = [(0, 1), (1, 2)];
    let sizes = sizes(3);
    let layout = layout(&sizes, &edges, &tree, &[]);
    assert!(layout.sets[0].is_none());
    assert!(layout.sets[1].is_some());
    check_drawing(&layout, &tree, &sizes, &edges, "between");
}

#[test]
fn layout_draws_one_of_two_sets_waiting_for_each_other_without_a_box() {
    // 0 and 1 in set 0, and 2 and 3 in set 1: 0 before 2, and 3 before 1.
    let tree = Tree {
        homes: vec![Some(0), Some(0), Some(1), Some(1)],
        parents: vec![None, None],
    };
    let edges = [(0, 2), (3, 1)];
    let sizes = sizes(4);
    let layout = layout(&sizes, &edges, &tree, &[]);
    assert_eq!(layout.sets.iter().filter(|set| set.is_some()).count(), 1);
    check_drawing(&layout, &tree, &sizes, &edges, "crossed");
}

/// The points edge `e` of `layout` is drawn through, over all its lines.
fn route(layout: &Layout, e: usize) -> Vec<Point<f32>> {
    let mut points: Vec<Point<f32>> = Vec::new();
    for &line in &layout.edges[e] {
        // Each line starts where the one before ends.
        let skip = usize::from(!points.is_empty());
        points.extend(&layout.lines[line].points[skip..]);
    }
    points
}

/// Checks that `layout` draws nodes of `sizes`, joined by `edges`, in the
/// boxes of `tree`: each node in the boxes of the sets it's drawn in and
/// apart from the others, each box likewise, no two nodes overlapping, and
/// each edge as lines joined end to start, left to right, from the right
/// side of the node it leaves to the left side of the node it enters, with
/// an arrowhead there only.
fn check_drawing(
    layout: &Layout,
    tree: &Tree,
    sizes: &[Size<f32>],
    edges: &[(usize, usize)],
    name: &str,
) {
    let node = |n: usize| Bounds {
        origin: layout.nodes[n],
        size: sizes[n],
    };
    let within = |inner: Bounds<f32>, outer: Bounds<f32>| {
        inner.origin.x >= outer.origin.x
            && inner.origin.y >= outer.origin.y
            && inner.origin.x + inner.size.width <= outer.origin.x + outer.size.width
            && inner.origin.y + inner.size.height <= outer.origin.y + outer.size.height
    };
    let apart = |a: Bounds<f32>, b: Bounds<f32>| {
        a.origin.x + a.size.width <= b.origin.x
            || b.origin.x + b.size.width <= a.origin.x
            || a.origin.y + a.size.height <= b.origin.y
            || b.origin.y + b.size.height <= a.origin.y
    };
    // The sets holding a node or set, as `tree` draws them.
    let holding = |mut set: Option<usize>| {
        let mut sets = Vec::new();
        while let Some(s) = set {
            sets.push(s);
            set = tree.parents[s];
        }
        sets
    };
    let boxes: Vec<(usize, Bounds<f32>)> = layout
        .sets
        .iter()
        .enumerate()
        .filter_map(|(set, bounds)| Some((set, (*bounds)?)))
        .collect();
    for (n, &home) in tree.homes.iter().enumerate() {
        let sets = holding(home);
        for &(set, bounds) in &boxes {
            if sets.contains(&set) {
                assert!(within(node(n), bounds), "{name}: {n} is outside {set}");
            } else {
                assert!(apart(node(n), bounds), "{name}: {n} is inside {set}");
            }
        }
    }
    for &(a, a_bounds) in &boxes {
        for &(b, b_bounds) in &boxes {
            if holding(tree.parents[a]).contains(&b) {
                assert!(within(a_bounds, b_bounds), "{name}: {a} is outside {b}");
            } else if a != b && !holding(tree.parents[b]).contains(&a) {
                assert!(apart(a_bounds, b_bounds), "{name}: {a} overlaps {b}");
            }
        }
    }
    assert_eq!(overlapping(&layout.nodes, sizes), None, "{name}");

    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    for (e, &(a, b)) in edges.iter().enumerate() {
        let lines = &layout.edges[e];
        for pair in lines.windows(2) {
            let end = *layout.lines[pair[0]].points.last().unwrap();
            let start = layout.lines[pair[1]].points[0];
            assert!(
                near(end.x, start.x) && near(end.y, start.y),
                "{name}: the lines of {a} to {b} aren't joined"
            );
        }
        for (i, &line) in lines.iter().enumerate() {
            let line = &layout.lines[line];
            assert!(
                line.points.windows(2).all(|pair| pair[0].x <= pair[1].x),
                "{name}: {a} to {b} goes back"
            );
            assert_eq!(line.arrow, i + 1 == lines.len(), "{name}: {a} to {b}");
        }
        let route = route(layout, e);
        let (first, last) = (route[0], route[route.len() - 1]);
        assert!(near(first.x, layout.nodes[a].x + sizes[a].width), "{name}");
        assert!(
            near(first.y, layout.nodes[a].y + sizes[a].height / 2.0),
            "{name}"
        );
        assert!(near(last.x, layout.nodes[b].x), "{name}");
        assert!(
            near(last.y, layout.nodes[b].y + sizes[b].height / 2.0),
            "{name}"
        );
    }
}

/// Every schedule of an app built from Bevy's plugins reads, each ordering
/// with the declared orderings it comes from, and lays out with each system
/// after the systems it waits for and none overlapping.
#[test]
fn real_schedules_read_and_lay_out() {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        bevy::transform::TransformPlugin,
        bevy::state::app::StatesPlugin,
        bevy::input::InputPlugin,
        SchedulesPlugin,
    ));
    app.finish();
    app.cleanup();
    app.update();
    let world = app.world();

    let listed = snapshot(world, None).schedules;
    let group = |name: &str| {
        listed
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.group)
    };
    assert_eq!(group("PostStartup"), Some(Group::Startup));
    assert_eq!(group("PostUpdate"), Some(Group::Main));
    // Run at startup too, but listed with the frame's schedules.
    assert_eq!(group("StateTransition"), Some(Group::Main));
    assert_eq!(group("FixedPostUpdate"), Some(Group::Fixed));
    assert_eq!(group("Main"), Some(Group::Other));
    for entry in &listed {
        let graph = snapshot(world, Some(entry.label)).graph.unwrap();
        assert_eq!(graph.label, entry.label);
        assert_eq!(graph.error, None, "{}", entry.name);
        for edge in &graph.edges {
            assert!(!edge.reasons.is_empty(), "{}", entry.name);
        }
        let sizes: Vec<Size<f32>> = graph
            .systems
            .iter()
            .map(|system| size(20.0 + 7.0 * system.name.len() as f32, 22.0))
            .collect();
        let edges: Vec<(usize, usize)> = graph
            .edges
            .iter()
            .map(|edge| (edge.before, edge.after))
            .collect();
        let tree = tree_of(&graph);
        let titles: Vec<f32> = graph
            .sets
            .iter()
            .map(|set| 7.0 * set.name.len() as f32)
            .collect();
        let layout = layout(&sizes, &edges, &tree, &titles);
        check_drawing(&layout, &tree, &sizes, &edges, &entry.name);
        for &(a, b) in &edges {
            assert!(layout.nodes[a].x + sizes[a].width < layout.nodes[b].x);
        }
        if entry.name == "PostUpdate" {
            let propagate = find_set(&graph, "Propagate");
            assert!(layout.sets[propagate].is_some());
        }
    }
}
