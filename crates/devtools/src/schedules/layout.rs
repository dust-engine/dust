//! The graph's drawing, left to right: each system in a column after the
//! systems it waits for, as Graphviz's `dot` draws a graph (Sugiyama's
//! method). An edge spanning several columns passes through each column
//! between, at a height of its own, so that it doesn't run through the
//! systems there.
//!
//! A set is drawn as a box around its systems and the sets in it, each
//! system and set in one box, picked by [`tree`]. A box's insides are laid
//! out first, on their own, and the box is then laid out as one node among
//! the systems and boxes beside it. An edge between a system in a box and
//! one outside it crosses the box's border at a port: on its left side
//! where it enters, and on its right side where it leaves, shared by the
//! edges entering or leaving at the same system. The insides are laid out
//! with the ports in a column before and after all others. As a box is
//! drawn after all that its systems wait for, a set that a system outside
//! it runs between two systems of is drawn without a box: what's in it is
//! drawn in the box it would be in.
//!
//! Each weakly connected part of the graph, or of a box's insides, is laid
//! out on its own, and the parts are stacked, the largest first, above a
//! grid of the systems nothing orders.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use gpui::{Bounds, Point, Size, point, size};

/// Room between two columns, for the edges.
const COLUMN_GAP: f32 = 56.0;

/// Room between a column of a box's ports and the column next to it.
const PORT_GAP: f32 = 28.0;

/// Least room between two systems in a column.
const NODE_GAP: f32 = 8.0;

/// Least room between an edge passing through a column and what's next to it.
const LANE: f32 = 6.0;

/// Room between two parts of the graph, and above the grid's heading.
const PART_GAP: f32 = 40.0;

/// Room between two parts of a box's insides, and above its grid.
const INNER_PART_GAP: f32 = 16.0;

/// Height of the grid's heading, above its systems.
pub(super) const GRID_HEADING: f32 = 24.0;

/// Room between two systems of the grid.
const GRID_GAP: Size<f32> = Size {
    width: 12.0,
    height: 8.0,
};

/// The grid is at least this wide, so that a small graph doesn't stack its
/// unordered systems in a single column.
const GRID_MIN_WIDTH: f32 = 640.0;

/// Room between a box's border and what's in it, at its sides and bottom.
pub(super) const SET_PADDING: f32 = 10.0;

/// Height of the top of a box, which shows the set's name.
pub(super) const SET_HEADER: f32 = 24.0;

/// How many times the order of the columns is swept to reduce crossings.
const ORDER_SWEEPS: usize = 24;

/// How many times the columns' heights are swept to straighten the edges.
const PLACE_SWEEPS: usize = 8;

/// Where everything goes, in a space whose origin is the top left corner of
/// the drawing.
pub(super) struct Layout {
    /// Top left corner of each node.
    pub nodes: Vec<Point<f32>>,
    /// Each set's box; `None` for a set drawn without one.
    pub sets: Vec<Option<Bounds<f32>>>,
    /// The lines the edges are drawn with.
    pub lines: Vec<Line>,
    /// The lines each edge is drawn with, in order: one, and one more for
    /// each box border it crosses, each line ending at the port the next
    /// starts at.
    pub edges: Vec<Vec<usize>>,
    pub size: Size<f32>,
    /// The grid of the nodes nothing orders, if there are any.
    pub grid: Option<Grid>,
}

/// A line of the drawing, from the right side of a node, or of a box at a
/// port, to the left side of one.
pub(super) struct Line {
    /// The points the line passes through, from left to right. Two points at
    /// the same height are joined by a straight line, and two at different
    /// heights by a curve leaving and entering horizontally.
    pub points: Vec<Point<f32>>,
    /// Whether the line ends at a node, with an arrowhead, rather than at a
    /// port.
    pub arrow: bool,
}

pub(super) struct Grid {
    /// Top left corner of its heading.
    pub at: Point<f32>,
    /// How many nodes it holds.
    pub count: usize,
}

/// The set each node and set is drawn in, if any.
pub(super) struct Tree {
    pub homes: Vec<Option<usize>>,
    /// Each set's, which is a set listed before it.
    pub parents: Vec<Option<usize>>,
}

impl Tree {
    /// No sets.
    #[cfg(test)]
    pub fn flat(nodes: usize) -> Self {
        Tree {
            homes: vec![None; nodes],
            parents: Vec::new(),
        }
    }

    /// Whether `node` is drawn in `set`, or in a set drawn in it.
    pub fn holds(&self, set: usize, node: usize) -> bool {
        let mut home = self.homes[node];
        while let Some(s) = home {
            if s == set {
                return true;
            }
            home = self.parents[s];
        }
        false
    }
}

/// Whether a node outside `inside` runs between two nodes inside: whether a
/// path of the edges `after` each node leaves them and comes back.
fn runs_between(inside: &[bool], after: &[Vec<usize>]) -> bool {
    let mut seen = vec![false; inside.len()];
    let mut stack: Vec<usize> = (0..inside.len()).filter(|&v| inside[v]).collect();
    while let Some(v) = stack.pop() {
        for &w in &after[v] {
            if inside[w] {
                if !inside[v] {
                    return true;
                }
            } else if !std::mem::replace(&mut seen[w], true) {
                stack.push(w);
            }
        }
    }
    false
}

/// Picks the set each node and set is drawn in, of those it's in directly
/// (`node_sets`, `set_parents`): the deepest, then the one with the most
/// systems (`set_sizes`), as a set that systems are in besides those they're
/// drawn in tends to hold systems from all over. Each set is listed after the
/// sets it's in.
///
/// A node in no set that `adopts` is drawn in the innermost set holding the
/// nodes joined to it, as the sync points Bevy adds are drawn with the
/// systems they apply the commands of.
pub(super) fn tree(
    node_sets: &[Vec<usize>],
    set_parents: &[Vec<usize>],
    set_sizes: &[usize],
    adopts: &[bool],
    edges: &[(usize, usize)],
) -> Tree {
    let sets = set_parents.len();
    let mut parents: Vec<Option<usize>> = Vec::with_capacity(sets);
    let mut depth: Vec<usize> = Vec::with_capacity(sets);
    for (set, options) in set_parents.iter().enumerate() {
        let parent = options
            .iter()
            .copied()
            .filter(|&parent| parent < set)
            .max_by_key(|&parent| (depth[parent], set_sizes[parent], Reverse(parent)));
        depth.push(parent.map_or(0, |parent| depth[parent] + 1));
        parents.push(parent);
    }
    let mut homes: Vec<Option<usize>> = node_sets
        .iter()
        .map(|options| {
            options
                .iter()
                .copied()
                .filter(|&set| set < sets)
                .max_by_key(|&set| (depth[set], set_sizes[set], Reverse(set)))
        })
        .collect();

    let mut joined: Vec<Vec<usize>> = vec![Vec::new(); node_sets.len()];
    for &(a, b) in edges {
        joined[a].push(b);
        joined[b].push(a);
    }
    let adopting = |node: usize| adopts[node] && node_sets[node].is_empty();
    for node in 0..node_sets.len() {
        if !adopting(node) {
            continue;
        }
        let home = {
            let mut around = joined[node]
                .iter()
                .filter(|&&other| !adopting(other))
                .map(|&other| homes[other]);
            around
                .next()
                .and_then(|first| around.fold(first, |a, b| common(a, b, &parents, &depth)))
        };
        homes[node] = home;
    }
    Tree { homes, parents }
}

/// The innermost set holding both `a` and `b`, which are sets or, as `None`,
/// the top level.
fn common(
    a: Option<usize>,
    b: Option<usize>,
    parents: &[Option<usize>],
    depth: &[usize],
) -> Option<usize> {
    let (mut a, mut b) = (a?, b?);
    while a != b {
        if depth[a] >= depth[b] {
            a = parents[a]?;
        } else {
            b = parents[b]?;
        }
    }
    Some(a)
}

/// Lays out nodes of the given sizes, joined by `edges`, from the first node
/// of each pair to the second, in the boxes of the sets of `tree`, whose
/// names are `titles` wide. The edges form no cycle: Bevy refuses to build a
/// schedule whose orderings do.
pub(super) fn layout(
    sizes: &[Size<f32>],
    edges: &[(usize, usize)],
    tree: &Tree,
    titles: &[f32],
) -> Layout {
    let (n, sets) = (sizes.len(), tree.parents.len());
    // A set that a node outside it runs between two nodes of can't be a box
    // drawn after all that its nodes wait for, nor can both of two sets
    // whose nodes wait for each other's: of those, the one holding the
    // fewest nodes goes, until none are left.
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        after[a].push(b);
    }
    let mut kept: Vec<bool> = (0..sets)
        .map(|set| {
            let inside: Vec<bool> = (0..n).map(|node| tree.holds(set, node)).collect();
            !runs_between(&inside, &after)
        })
        .collect();
    let nesting = loop {
        let nesting = Nesting::new(tree, &kept);
        match nesting.on_cycle(edges) {
            Some(set) => kept[set] = false,
            None => break nesting,
        }
    };

    // What each box holds, and then the top level: the nodes and boxes in
    // the order of their first nodes, which is the order Bevy would run the
    // nodes in one at a time.
    let top = sets;
    let level_of = |set: Option<usize>| set.unwrap_or(top);
    let mut levels: Vec<LevelBuilder> = (0..=sets).map(|_| LevelBuilder::default()).collect();
    for node in 0..n {
        levels[level_of(nesting.home[node])]
            .items
            .push(Item::Node(node));
    }
    for set in (0..sets).filter(|&set| nesting.boxed[set]) {
        levels[level_of(nesting.parent[set])]
            .items
            .push(Item::Set(set));
    }
    for level in &mut levels {
        level.items.sort_by_key(|&item| match item {
            Item::Node(node) => node,
            Item::Set(set) => nesting.first[set],
        });
    }

    // The lines of each edge: out of each box it leaves, across the level
    // holding both its nodes, and into each box it enters.
    let mut paths: Vec<Vec<(usize, usize)>> = Vec::with_capacity(edges.len());
    for &(a, b) in edges {
        let (leaves, across, enters) = nesting.route(a, b);
        let mut path = Vec::with_capacity(leaves.len() + enters.len() + 1);
        let mut start = Start::Node(a);
        for &set in &leaves {
            path.push((set, levels[set].link(start, End::Port(a))));
            start = Start::Box(set, a);
        }
        let end = enters.last().map_or(End::Node(b), |&set| End::Box(set, b));
        let level = level_of(across);
        path.push((level, levels[level].link(start, end)));
        for (i, &set) in enters.iter().enumerate().rev() {
            let end = match i.checked_sub(1) {
                Some(inner) => End::Box(enters[inner], b),
                None => End::Node(b),
            };
            path.push((set, levels[set].link(Start::Port(b), end)));
        }
        paths.push(path);
    }

    // Each box's insides before the box it's in, and the top level last.
    let mut order: Vec<usize> = (0..sets).filter(|&set| nesting.boxed[set]).collect();
    order.sort_by_key(|&set| Reverse(nesting.depth[set]));
    order.push(top);
    let mut arranged: Vec<Option<Arranged>> = (0..=sets).map(|_| None).collect();
    let mut boxes = Boxes {
        sizes: vec![size(0.0, 0.0); sets],
        ports_in: vec![HashMap::new(); sets],
        ports_out: vec![HashMap::new(); sets],
        count: &nesting.count,
    };
    for &level in &order {
        let builder = &levels[level];
        let result = arrange(&builder.level(sizes, &boxes), level == top);
        if level != top {
            let title = titles.get(level).copied().unwrap_or(0.0);
            boxes.sizes[level] = size(
                result.size.width.max(title) + 2.0 * SET_PADDING,
                SET_HEADER + result.size.height + SET_PADDING,
            );
            let ports = builder.items.len();
            for (i, &node) in builder.ins.iter().enumerate() {
                boxes.ports_in[level].insert(node, SET_HEADER + result.positions[ports + i].y);
            }
            let ports = ports + builder.ins.len();
            for (i, &node) in builder.outs.iter().enumerate() {
                boxes.ports_out[level].insert(node, SET_HEADER + result.positions[ports + i].y);
            }
        }
        arranged[level] = Some(result);
    }

    // Then where each level's insides go, from the top level in.
    let mut layout = Layout {
        nodes: vec![point(0.0, 0.0); n],
        sets: vec![None; sets],
        lines: Vec::new(),
        edges: Vec::with_capacity(edges.len()),
        size: size(0.0, 0.0),
        grid: None,
    };
    let mut origins = vec![point(0.0, 0.0); sets + 1];
    // The line of each link of each level.
    let mut lines: Vec<Vec<usize>> = vec![Vec::new(); sets + 1];
    for &level in order.iter().rev() {
        let (builder, result) = (&levels[level], arranged[level].as_ref().unwrap());
        let origin = origins[level];
        for (&item, &at) in builder.items.iter().zip(&result.positions) {
            let at = origin + at;
            match item {
                Item::Node(node) => layout.nodes[node] = at,
                Item::Set(set) => {
                    layout.sets[set] = Some(Bounds {
                        origin: at,
                        size: boxes.sizes[set],
                    });
                    origins[set] = at + point(SET_PADDING, SET_HEADER);
                }
            }
        }
        // A port is inside the box's padding, and its lines go on to the
        // border.
        let border = layout.sets.get(level).copied().flatten();
        for (&(start, end), route) in builder.links.iter().zip(&result.routes) {
            let mut points: Vec<Point<f32>> = route.iter().map(|&at| origin + at).collect();
            if let (Start::Port(_), Some(border)) = (start, border) {
                points.insert(0, point(border.origin.x, points[0].y));
            }
            if let (End::Port(_), Some(border)) = (end, border) {
                let y = points[points.len() - 1].y;
                points.push(point(border.origin.x + border.size.width, y));
            }
            lines[level].push(layout.lines.len());
            layout.lines.push(Line {
                points,
                arrow: matches!(end, End::Node(_)),
            });
        }
    }
    layout.edges = paths
        .iter()
        .map(|path| {
            path.iter()
                .map(|&(level, link)| lines[level][link])
                .collect()
        })
        .collect();
    let top = arranged[top].take().unwrap();
    layout.size = top.size;
    layout.grid = top.grid;
    layout
}

/// A node, or a set's box, as one of the things a level holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Item {
    Node(usize),
    Set(usize),
}

/// Which sets the layout draws a box for, and which box each node and box
/// is drawn in: those of a tree, less the sets left out, which hold what
/// they would in the box they'd be in, and the sets holding no node.
struct Nesting {
    home: Vec<Option<usize>>,
    parent: Vec<Option<usize>>,
    boxed: Vec<bool>,
    /// How many boxes each box is in.
    depth: Vec<usize>,
    /// How many nodes each box holds.
    count: Vec<usize>,
    /// The first node each box holds.
    first: Vec<usize>,
}

impl Nesting {
    /// Draws `tree` with boxes for the sets `kept` keeps.
    fn new(tree: &Tree, kept: &[bool]) -> Self {
        let sets = tree.parents.len();
        let nearest = |mut set: Option<usize>| {
            while let Some(s) = set {
                if kept[s] {
                    return Some(s);
                }
                set = tree.parents[s];
            }
            None
        };
        let home: Vec<Option<usize>> = tree.homes.iter().map(|&set| nearest(set)).collect();
        let parent: Vec<Option<usize>> = tree.parents.iter().map(|&set| nearest(set)).collect();
        let mut count = vec![0; sets];
        let mut first = vec![usize::MAX; sets];
        for (node, &home) in home.iter().enumerate() {
            let mut set = home;
            while let Some(s) = set {
                count[s] += 1;
                first[s] = first[s].min(node);
                set = parent[s];
            }
        }
        let mut depth = vec![0; sets];
        for set in 0..sets {
            depth[set] = parent[set].map_or(0, |parent| depth[parent] + 1);
        }
        Nesting {
            boxed: (0..sets).map(|set| kept[set] && count[set] > 0).collect(),
            home,
            parent,
            depth,
            count,
            first,
        }
    }

    /// The boxes holding `node`, innermost first.
    fn boxes(&self, node: usize) -> Vec<usize> {
        let mut boxes = Vec::new();
        let mut set = self.home[node];
        while let Some(s) = set {
            boxes.push(s);
            set = self.parent[s];
        }
        boxes
    }

    /// The boxes an edge from `a` to `b` leaves, innermost first; the box it
    /// crosses, holding both, or `None` for the top level; and the boxes it
    /// enters, innermost first.
    fn route(&self, a: usize, b: usize) -> (Vec<usize>, Option<usize>, Vec<usize>) {
        let (mut leaves, mut enters) = (self.boxes(a), self.boxes(b));
        let shared = leaves.iter().position(|set| enters.contains(set));
        let across = shared.map(|i| leaves[i]);
        leaves.truncate(shared.unwrap_or(leaves.len()));
        if let Some(across) = across
            && let Some(i) = enters.iter().position(|&set| set == across)
        {
            enters.truncate(i);
        }
        (leaves, across, enters)
    }

    /// A box on a cycle of the boxes and nodes beside it, waiting for what
    /// waits for it, which it can't be drawn after: the one holding the
    /// fewest nodes, if any.
    fn on_cycle(&self, edges: &[(usize, usize)]) -> Option<usize> {
        // The edges between the items of each level.
        let mut ids: HashMap<(Option<usize>, Item), usize> = HashMap::new();
        let mut items: Vec<Item> = Vec::new();
        let mut joined: Vec<(usize, usize)> = Vec::with_capacity(edges.len());
        for &(a, b) in edges {
            let (leaves, across, enters) = self.route(a, b);
            let mut id = |item: Item| {
                *ids.entry((across, item)).or_insert_with(|| {
                    items.push(item);
                    items.len() - 1
                })
            };
            let from = id(leaves.last().map_or(Item::Node(a), |&set| Item::Set(set)));
            let to = id(enters.last().map_or(Item::Node(b), |&set| Item::Set(set)));
            joined.push((from, to));
        }
        items
            .iter()
            .zip(on_cycles(items.len(), &joined))
            .filter_map(|(&item, cyclic)| match item {
                Item::Set(set) if cyclic => Some(set),
                _ => None,
            })
            .min_by_key(|&set| (self.count[set], Reverse(self.depth[set]), set))
    }
}

/// Which of `count` vertices are on a cycle of `edges`, or between two: those
/// left once the vertices nothing enters, or nothing leaves, are taken away,
/// again and again.
fn on_cycles(count: usize, edges: &[(usize, usize)]) -> Vec<bool> {
    let mut entering = vec![0usize; count];
    let mut leaving = vec![0usize; count];
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut before: Vec<Vec<usize>> = vec![Vec::new(); count];
    for &(a, b) in edges {
        leaving[a] += 1;
        entering[b] += 1;
        after[a].push(b);
        before[b].push(a);
    }
    let mut left = vec![true; count];
    let mut stack: Vec<usize> = (0..count)
        .filter(|&v| entering[v] == 0 || leaving[v] == 0)
        .collect();
    while let Some(v) = stack.pop() {
        if !std::mem::replace(&mut left[v], false) {
            continue;
        }
        for &w in &after[v] {
            entering[w] -= 1;
            if left[w] && entering[w] == 0 {
                stack.push(w);
            }
        }
        for &u in &before[v] {
            leaving[u] -= 1;
            if left[u] && leaving[u] == 0 {
                stack.push(u);
            }
        }
    }
    left
}

/// Where a line of a level starts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Start {
    Node(usize),
    /// A box of the level, at its port for the edges leaving it from `node`.
    Box(usize, usize),
    /// The level's own box's port for the edges entering it at `node`.
    Port(usize),
}

/// Where a line of a level ends.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum End {
    Node(usize),
    /// A box of the level, at its port for the edges entering it at `node`.
    Box(usize, usize),
    /// The level's own box's port for the edges leaving it from `node`.
    Port(usize),
}

/// The boxes laid out so far.
struct Boxes<'a> {
    sizes: Vec<Size<f32>>,
    /// Where each box's ports are, from its top, by the node whose edges
    /// enter or leave the box there.
    ports_in: Vec<HashMap<usize, f32>>,
    ports_out: Vec<HashMap<usize, f32>>,
    /// How many nodes each box holds.
    count: &'a [usize],
}

/// What a box holds, or the top level, gathered for [`arrange`].
#[derive(Default)]
struct LevelBuilder {
    items: Vec<Item>,
    /// The nodes the box's ports are for, where edges enter it and leave it.
    ins: Vec<usize>,
    outs: Vec<usize>,
    links: Vec<(Start, End)>,
    index: HashMap<(Start, End), usize>,
}

impl LevelBuilder {
    /// The index of the line from `start` to `end`, which the edges going
    /// the same way share.
    fn link(&mut self, start: Start, end: End) -> usize {
        if let Start::Port(node) = start
            && !self.ins.contains(&node)
        {
            self.ins.push(node);
        }
        if let End::Port(node) = end
            && !self.outs.contains(&node)
        {
            self.outs.push(node);
        }
        if let Some(&link) = self.index.get(&(start, end)) {
            return link;
        }
        self.links.push((start, end));
        self.index.insert((start, end), self.links.len() - 1);
        self.links.len() - 1
    }

    /// The level to arrange: its items, then its ports in, then out.
    fn level(&self, sizes: &[Size<f32>], boxes: &Boxes) -> Level {
        let index: HashMap<Item, usize> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, &item)| (item, i))
            .collect();
        let ports = self.ins.len() + self.outs.len();
        let mut level = Level {
            sizes: Vec::with_capacity(self.items.len() + ports),
            ports: Vec::with_capacity(self.items.len() + ports),
            weights: Vec::with_capacity(self.items.len() + ports),
            gridded: Vec::with_capacity(self.items.len() + ports),
            links: Vec::with_capacity(self.links.len()),
        };
        for &item in &self.items {
            let (size, weight) = match item {
                Item::Node(node) => (sizes[node], 1),
                Item::Set(set) => (boxes.sizes[set], boxes.count[set]),
            };
            level.sizes.push(size);
            level.ports.push(None);
            level.weights.push(weight);
            level.gridded.push(matches!(item, Item::Node(_)));
        }
        for port in self
            .ins
            .iter()
            .map(|_| Port::In)
            .chain(self.outs.iter().map(|_| Port::Out))
        {
            level.sizes.push(size(0.0, 0.0));
            level.ports.push(Some(port));
            level.weights.push(0);
            level.gridded.push(false);
        }
        let port_in =
            |node: usize| self.items.len() + self.ins.iter().position(|&n| n == node).unwrap();
        let port_out = |node: usize| {
            self.items.len() + self.ins.len() + self.outs.iter().position(|&n| n == node).unwrap()
        };
        for &(start, end) in &self.links {
            let (from, from_y) = match start {
                Start::Node(node) => (index[&Item::Node(node)], sizes[node].height / 2.0),
                Start::Box(set, node) => (index[&Item::Set(set)], boxes.ports_out[set][&node]),
                Start::Port(node) => (port_in(node), 0.0),
            };
            let (to, to_y) = match end {
                End::Node(node) => (index[&Item::Node(node)], sizes[node].height / 2.0),
                End::Box(set, node) => (index[&Item::Set(set)], boxes.ports_in[set][&node]),
                End::Port(node) => (port_out(node), 0.0),
            };
            level.links.push(Link {
                from,
                from_y,
                to,
                to_y,
            });
        }
        level
    }
}

/// What [`arrange`] lays out: the nodes and boxes drawn in a box, or at the
/// top level, and the box's ports.
struct Level {
    /// The size of each vertex: those of the nodes and boxes, and none for
    /// the ports.
    sizes: Vec<Size<f32>>,
    /// Which vertices are ports, and on which side of the box.
    ports: Vec<Option<Port>>,
    /// How many nodes each vertex holds, which orders the parts.
    weights: Vec<usize>,
    /// Whether each vertex goes in the grid when no line joins it: a node,
    /// whereas a box goes with the parts.
    gridded: Vec<bool>,
    links: Vec<Link>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Port {
    /// On the box's left side, where edges enter it.
    In,
    /// On its right side, where edges leave it.
    Out,
}

/// A line of a level, from the right side of `from`, `from_y` below its top,
/// to the left side of `to`, `to_y` below its top.
struct Link {
    from: usize,
    from_y: f32,
    to: usize,
    to_y: f32,
}

/// A level laid out, with its top left corner at the origin.
struct Arranged {
    /// Top left corner of each vertex.
    positions: Vec<Point<f32>>,
    /// The points each link passes through.
    routes: Vec<Vec<Point<f32>>>,
    size: Size<f32>,
    grid: Option<Grid>,
}

/// Lays out a level: each weakly connected part on its own, the parts
/// stacked, the largest first, above a grid of the nodes no line joins, with
/// a heading at the `top` level.
fn arrange(level: &Level, top: bool) -> Arranged {
    let n = level.sizes.len();
    let mut parts = Parts::new(n);
    for link in &level.links {
        parts.join(link.from, link.to);
    }
    // The members and links of each part, by the part's first vertex.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
    for v in 0..n {
        members[parts.find(v)].push(v);
    }
    let mut part_links: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (l, link) in level.links.iter().enumerate() {
        part_links[parts.find(link.from)].push(l);
    }
    let gridded = |v: usize| members[v].len() == 1 && level.gridded[v];
    // Largest first, then in the order of their first vertices.
    let mut roots: Vec<usize> = (0..n)
        .filter(|&root| !members[root].is_empty() && !gridded(root))
        .collect();
    roots.sort_by_key(|&root| {
        Reverse(
            members[root]
                .iter()
                .map(|&v| level.weights[v])
                .sum::<usize>(),
        )
    });

    let mut arranged = Arranged {
        positions: vec![point(0.0, 0.0); n],
        routes: vec![Vec::new(); level.links.len()],
        size: size(0.0, 0.0),
        grid: None,
    };
    let gap = if top { PART_GAP } else { INNER_PART_GAP };
    let mut y = 0.0;
    for root in roots {
        let part = Part::new(level, &members[root], &part_links[root]);
        let offset = point(0.0, y);
        for (&v, &at) in members[root].iter().zip(&part.nodes) {
            arranged.positions[v] = at + offset;
        }
        for (&l, route) in part_links[root].iter().zip(part.routes) {
            arranged.routes[l] = route.into_iter().map(|at| at + offset).collect();
        }
        arranged.size.width = arranged.size.width.max(part.size.width);
        y += part.size.height + gap;
    }

    let unordered: Vec<usize> = (0..n).filter(|&v| gridded(v)).collect();
    if unordered.is_empty() {
        arranged.size.height = (y - gap).max(0.0);
        return arranged;
    }
    let heading = if top {
        arranged.grid = Some(Grid {
            at: point(0.0, y),
            count: unordered.len(),
        });
        GRID_HEADING
    } else {
        0.0
    };
    let width = arranged.size.width.max(GRID_MIN_WIDTH);
    let (mut x, mut y, mut row) = (0.0, y + heading, 0.0f32);
    for v in unordered {
        let v_size = level.sizes[v];
        if x > 0.0 && x + v_size.width > width {
            (x, y, row) = (0.0, y + row + GRID_GAP.height, 0.0);
        }
        arranged.positions[v] = point(x, y);
        x += v_size.width + GRID_GAP.width;
        row = row.max(v_size.height);
        arranged.size.width = arranged.size.width.max(x - GRID_GAP.width);
    }
    arranged.size.height = y + row;
    arranged
}

/// Union-find over the vertices, joining the ends of each link.
struct Parts(Vec<usize>);

impl Parts {
    fn new(n: usize) -> Self {
        Parts((0..n).collect())
    }

    fn find(&mut self, mut v: usize) -> usize {
        while self.0[v] != v {
            self.0[v] = self.0[self.0[v]];
            v = self.0[v];
        }
        v
    }

    /// Joins the parts of `a` and `b`, into the part of whichever comes
    /// first, so that a part's root is its first vertex.
    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        let (first, second) = if a < b { (a, b) } else { (b, a) };
        self.0[second] = first;
    }
}

/// One weakly connected part of a level, laid out with its top left corner
/// at the origin.
struct Part {
    /// Top left corner of each of the part's vertices, in the order of its
    /// `members`.
    nodes: Vec<Point<f32>>,
    /// The route of each of the part's links, in the order of its links.
    routes: Vec<Vec<Point<f32>>>,
    size: Size<f32>,
}

/// A vertex of a part's drawing: one of the level's, or a point a line
/// passes through in a column between its ends.
struct Vertex {
    column: usize,
    size: Size<f32>,
    /// The vertices joined to this one in the column before, and in the one
    /// after, each with how far below the other's middle this one's middle
    /// is when the line between them is straight.
    before: Vec<(usize, f32)>,
    after: Vec<(usize, f32)>,
    /// Whether the vertex is a point of a line: a port, or one passing
    /// through.
    lane: bool,
}

impl Part {
    /// Lays out the level's vertices `members`, sorted, and the links between
    /// them, which are `part_links` of the level's.
    fn new(level: &Level, members: &[usize], part_links: &[usize]) -> Self {
        let local = |v: usize| members.binary_search(&v).unwrap();
        let count = members.len();
        let port = |v: usize| level.ports[members[v]];
        let mut after: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut before: Vec<Vec<usize>> = vec![Vec::new(); count];
        for &l in part_links {
            let link = &level.links[l];
            let (a, b) = (local(link.from), local(link.to));
            after[a].push(b);
            before[b].push(a);
        }

        // Each node or box in the column after the latest of those it waits
        // for. One waiting for none, nor for a port, goes in the column
        // before the earliest of those waiting for it, rather than the
        // first column, so that its edges stay short. The ports go in a
        // column of their own on each side.
        let order = topological_order(&after, &before);
        let inside = |v: usize| port(v).is_none();
        let mut columns = vec![0usize; count];
        for &v in order.iter().filter(|&&v| inside(v)) {
            columns[v] = before[v]
                .iter()
                .filter(|&&b| inside(b))
                .map(|&b| columns[b] + 1)
                .max()
                .unwrap_or(0);
        }
        for &v in order.iter().rev() {
            if inside(v)
                && before[v].is_empty()
                && let Some(earliest) = after[v]
                    .iter()
                    .filter(|&&a| inside(a))
                    .map(|&a| columns[a])
                    .min()
            {
                columns[v] = earliest.saturating_sub(1);
            }
        }
        // Without columns left empty, which only a cycle could leave.
        let mut used: Vec<usize> = (0..count)
            .filter(|&v| inside(v))
            .map(|v| columns[v])
            .collect();
        used.sort_unstable();
        used.dedup();
        let ports_in = (0..count).any(|v| port(v) == Some(Port::In));
        let ports_out = (0..count).any(|v| port(v) == Some(Port::Out));
        let first = usize::from(ports_in);
        let last = first + used.len();
        for (v, column) in columns.iter_mut().enumerate() {
            *column = match port(v) {
                None => first + used.binary_search(column).unwrap(),
                Some(Port::In) => 0,
                Some(Port::Out) => last,
            };
        }

        // The vertices: the members, then the points of each link spanning
        // more than the gap between two columns.
        let mut vertices: Vec<Vertex> = (0..count)
            .map(|v| Vertex {
                column: columns[v],
                size: level.sizes[members[v]],
                before: Vec::new(),
                after: Vec::new(),
                lane: !inside(v),
            })
            .collect();
        // Each link's vertices, and how far below the middles of its ends it
        // leaves and enters them.
        let mut chains: Vec<(Vec<usize>, f32, f32)> = Vec::with_capacity(part_links.len());
        for &l in part_links {
            let link = &level.links[l];
            let (a, b) = (local(link.from), local(link.to));
            let leaves = link.from_y - vertices[a].size.height / 2.0;
            let enters = link.to_y - vertices[b].size.height / 2.0;
            let mut chain = vec![a];
            if columns[b] > columns[a] {
                for column in columns[a] + 1..columns[b] {
                    vertices.push(Vertex {
                        column,
                        size: size(0.0, 0.0),
                        before: Vec::new(),
                        after: Vec::new(),
                        lane: true,
                    });
                    chain.push(vertices.len() - 1);
                }
                chain.push(b);
                let last = chain.len() - 2;
                for (i, pair) in chain.windows(2).enumerate() {
                    let from = if i == 0 { leaves } else { 0.0 };
                    let to = if i == last { enters } else { 0.0 };
                    vertices[pair[0]].after.push((pair[1], to - from));
                    vertices[pair[1]].before.push((pair[0], from - to));
                }
            } else {
                // Only were the links to form a cycle; drawn straight back.
                chain.push(b);
            }
            chains.push((chain, leaves, enters));
        }

        let column_count = columns.iter().max().map_or(0, |&last| last + 1);
        let mut layers = initial_order(&vertices, &order, column_count);
        reduce_crossings(&vertices, &mut layers);

        // Each column as wide as its widest vertex, its vertices on its left.
        let mut lefts = Vec::with_capacity(column_count);
        let mut widths = Vec::with_capacity(column_count);
        let mut left = 0.0;
        for (column, layer) in layers.iter().enumerate() {
            let width = layer
                .iter()
                .map(|&v| vertices[v].size.width)
                .fold(0.0, f32::max);
            lefts.push(left);
            widths.push(width);
            let beside_ports = (ports_in && column == 0) || (ports_out && column + 1 == last);
            left += width + if beside_ports { PORT_GAP } else { COLUMN_GAP };
        }
        let centers = place(&vertices, &layers);

        let top = (0..vertices.len())
            .map(|v| centers[v] - vertices[v].size.height / 2.0)
            .fold(f32::INFINITY, f32::min);
        let top = if top.is_finite() { top } else { 0.0 };
        let bottom = (0..vertices.len())
            .map(|v| centers[v] + vertices[v].size.height / 2.0 - top)
            .fold(0.0, f32::max);
        let nodes: Vec<Point<f32>> = (0..count)
            .map(|v| {
                let vertex = &vertices[v];
                point(
                    lefts[vertex.column],
                    centers[v] - vertex.size.height / 2.0 - top,
                )
            })
            .collect();
        let routes = chains
            .iter()
            .map(|(chain, leaves, enters)| {
                let mut route = Vec::with_capacity(chain.len() * 2 + 1);
                let (first, last) = (chain[0], chain[chain.len() - 1]);
                let start = point(
                    nodes[first].x + vertices[first].size.width,
                    centers[first] + leaves - top,
                );
                route.push(start);
                // Straight on to its column's right side, so that the curve
                // to the next column stays between the two, clear of what's
                // in them.
                let column = vertices[first].column;
                let right = lefts[column] + widths[column];
                if start.x < right {
                    route.push(point(right, start.y));
                }
                for &lane in &chain[1..chain.len() - 1] {
                    let column = vertices[lane].column;
                    let y = centers[lane] - top;
                    route.push(point(lefts[column], y));
                    route.push(point(lefts[column] + widths[column], y));
                }
                route.push(point(nodes[last].x, centers[last] + enters - top));
                route
            })
            .collect();
        let width = match (lefts.last(), widths.last()) {
            (Some(&left), Some(&width)) => left + width,
            _ => 0.0,
        };
        Part {
            nodes,
            routes,
            size: size(width, bottom),
        }
    }
}

/// The part's vertices in an order where each comes after the vertices it
/// waits for, earlier vertices first where the links leave a choice.
/// Vertices on a cycle, were there one, go last.
fn topological_order(after: &[Vec<usize>], before: &[Vec<usize>]) -> Vec<usize> {
    let mut waiting: Vec<usize> = before.iter().map(Vec::len).collect();
    let mut ready: BinaryHeap<Reverse<usize>> = (0..after.len())
        .filter(|&v| waiting[v] == 0)
        .map(Reverse)
        .collect();
    let mut order = Vec::with_capacity(after.len());
    while let Some(Reverse(v)) = ready.pop() {
        order.push(v);
        for &next in &after[v] {
            waiting[next] -= 1;
            if waiting[next] == 0 {
                ready.push(Reverse(next));
            }
        }
    }
    if order.len() < after.len() {
        let placed: HashSet<usize> = order.iter().copied().collect();
        order.extend((0..after.len()).filter(|v| !placed.contains(v)));
    }
    order
}

/// Each column's vertices in the order a depth-first walk from the members
/// meets them, which keeps the vertices of a line, and vertices joined to
/// each other, together.
fn initial_order(vertices: &[Vertex], order: &[usize], columns: usize) -> Vec<Vec<usize>> {
    let mut layers = vec![Vec::new(); columns];
    let mut seen = vec![false; vertices.len()];
    let mut stack = Vec::new();
    for &start in order {
        if seen[start] {
            continue;
        }
        stack.push(start);
        while let Some(v) = stack.pop() {
            if std::mem::replace(&mut seen[v], true) {
                continue;
            }
            layers[vertices[v].column].push(v);
            // Reversed, so that the first is walked first.
            stack.extend(
                vertices[v]
                    .after
                    .iter()
                    .rev()
                    .map(|&(a, _)| a)
                    .filter(|&a| !seen[a]),
            );
        }
    }
    layers
}

/// Reorders each column to cross fewer lines: sweeps the columns forwards
/// and backwards, sorting each by the average position of the vertices
/// it's joined to in the column it comes from, then swapping neighbors
/// while that crosses fewer lines, and keeps the best order found.
fn reduce_crossings(vertices: &[Vertex], layers: &mut [Vec<usize>]) {
    if layers.len() < 2 {
        return;
    }
    let mut positions = vec![0usize; vertices.len()];
    let index = |layers: &[Vec<usize>], positions: &mut [usize]| {
        for layer in layers {
            for (i, &v) in layer.iter().enumerate() {
                positions[v] = i;
            }
        }
    };
    index(layers, &mut positions);
    let mut best = layers.to_vec();
    let mut best_crossings = crossings(vertices, layers, &positions);
    for sweep in 0..ORDER_SWEEPS {
        if best_crossings == 0 {
            break;
        }
        let forwards = sweep % 2 == 0;
        let columns: Vec<usize> = if forwards {
            (1..layers.len()).collect()
        } else {
            (0..layers.len() - 1).rev().collect()
        };
        for column in columns {
            let from = if forwards { column - 1 } else { column + 1 };
            let span = layers[from].len().max(1) as f32;
            let own = layers[column].len().max(1) as f32;
            let keys: Vec<f32> = layers[column]
                .iter()
                .map(|&v| {
                    let joined = if forwards {
                        &vertices[v].before
                    } else {
                        &vertices[v].after
                    };
                    match joined.len() {
                        // Where it is now, scaled to the other column.
                        0 => positions[v] as f32 / own * span,
                        len => {
                            joined
                                .iter()
                                .map(|&(u, _)| positions[u] as f32)
                                .sum::<f32>()
                                / len as f32
                        }
                    }
                })
                .collect();
            let mut sorted: Vec<usize> = (0..layers[column].len()).collect();
            sorted.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
            layers[column] = sorted.iter().map(|&i| layers[column][i]).collect();
            for (i, &v) in layers[column].iter().enumerate() {
                positions[v] = i;
            }
        }
        transpose(vertices, layers, &mut positions);
        let count = crossings(vertices, layers, &positions);
        if count < best_crossings {
            best_crossings = count;
            best.clone_from_slice(layers);
        }
    }
    layers.clone_from_slice(&best);
}

/// Swaps neighbors in each column while that crosses fewer lines.
fn transpose(vertices: &[Vertex], layers: &mut [Vec<usize>], positions: &mut [usize]) {
    // Each pass swaps at least one pair, or stops; a few passes get most of
    // the gain.
    for _ in 0..4 {
        let mut improved = false;
        for layer in layers.iter_mut() {
            for i in 1..layer.len() {
                let (left, right) = (layer[i - 1], layer[i]);
                let now = pair_crossings(vertices, left, right, positions);
                let swapped = pair_crossings(vertices, right, left, positions);
                if swapped < now {
                    layer.swap(i - 1, i);
                    positions[left] = i;
                    positions[right] = i - 1;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// How many of the lines of `left` and `right`, neighbors with `left` first,
/// cross each other.
fn pair_crossings(vertices: &[Vertex], left: usize, right: usize, positions: &[usize]) -> usize {
    let count = |a: &[(usize, f32)], b: &[(usize, f32)]| {
        a.iter()
            .map(|&(x, _)| {
                b.iter()
                    .filter(|&&(y, _)| positions[x] > positions[y])
                    .count()
            })
            .sum::<usize>()
    };
    let (left, right) = (&vertices[left], &vertices[right]);
    count(&left.before, &right.before) + count(&left.after, &right.after)
}

/// How many pairs of lines cross between all the columns.
fn crossings(vertices: &[Vertex], layers: &[Vec<usize>], positions: &[usize]) -> usize {
    let mut total = 0;
    for pair in layers.windows(2) {
        let mut ends: Vec<(usize, usize)> = pair[0]
            .iter()
            .flat_map(|&v| {
                vertices[v]
                    .after
                    .iter()
                    .map(move |&(a, _)| (positions[v], positions[a]))
            })
            .collect();
        ends.sort_unstable();
        // Lines sorted by where they leave cross the earlier lines entering
        // further down, which a Fenwick tree counts.
        let mut tree = vec![0usize; pair[1].len() + 1];
        for (seen, &(_, end)) in ends.iter().enumerate() {
            let mut at_or_above = 0;
            let mut i = end + 1;
            while i > 0 {
                at_or_above += tree[i];
                i &= i - 1;
            }
            total += seen - at_or_above;
            let mut i = end + 1;
            while i < tree.len() {
                tree[i] += 1;
                i += i.isolate_lowest_one();
            }
        }
    }
    total
}

/// The height of each vertex's center: as close to where the lines joining
/// it would be straight as the order and the gaps of its column allow,
/// which keeps the lines short and straight.
fn place(vertices: &[Vertex], layers: &[Vec<usize>]) -> Vec<f32> {
    let mut centers = vec![0.0f32; vertices.len()];
    let gaps: Vec<Vec<f32>> = layers
        .iter()
        .map(|layer| {
            layer
                .windows(2)
                .map(|pair| {
                    let (a, b) = (&vertices[pair[0]], &vertices[pair[1]]);
                    let gap = if a.lane || b.lane { LANE } else { NODE_GAP };
                    (a.size.height + b.size.height) / 2.0 + gap
                })
                .collect()
        })
        .collect();
    for (layer, gaps) in layers.iter().zip(&gaps) {
        let mut y = 0.0;
        for (i, &v) in layer.iter().enumerate() {
            centers[v] = y;
            y += gaps.get(i).copied().unwrap_or(0.0);
        }
    }
    let mut desired = Vec::new();
    for sweep in 0..PLACE_SWEEPS * 2 + 1 {
        let forwards = sweep % 2 == 0;
        let columns: Vec<usize> = if forwards {
            (0..layers.len()).collect()
        } else {
            (0..layers.len()).rev().collect()
        };
        // Toward the column swept from, and in the last sweep, toward both.
        let last = sweep == PLACE_SWEEPS * 2;
        for column in columns {
            desired.clear();
            desired.extend(layers[column].iter().map(|&v| {
                let vertex = &vertices[v];
                let joined: [&[(usize, f32)]; 2] = match (last, forwards) {
                    (true, _) => [&vertex.before, &vertex.after],
                    (false, true) => [&vertex.before, &[]],
                    (false, false) => [&[], &vertex.after],
                };
                let count = joined[0].len() + joined[1].len();
                if count == 0 {
                    centers[v]
                } else {
                    joined
                        .iter()
                        .flat_map(|joined| joined.iter())
                        .map(|&(u, below)| centers[u] + below)
                        .sum::<f32>()
                        / count as f32
                }
            }));
            for (&v, y) in layers[column]
                .iter()
                .zip(closest_spaced(&desired, &gaps[column]))
            {
                centers[v] = y;
            }
        }
    }
    centers
}

/// The positions closest to `desired`, in least squares, that keep their
/// order and at least `gaps[i]` between the `i`th and the next: the
/// positions less the gaps before them may not decrease, so pooling the
/// neighbors that violate that into their mean solves it.
fn closest_spaced(desired: &[f32], gaps: &[f32]) -> Vec<f32> {
    let mut offsets = Vec::with_capacity(desired.len());
    let mut offset = 0.0;
    for i in 0..desired.len() {
        offsets.push(offset);
        offset += gaps.get(i).copied().unwrap_or(0.0);
    }
    // (mean, count) of each pool.
    let mut pools: Vec<(f32, usize)> = Vec::with_capacity(desired.len());
    for (i, &want) in desired.iter().enumerate() {
        pools.push((want - offsets[i], 1));
        while let [.., (a, a_count), (b, b_count)] = pools[..] {
            if a <= b {
                break;
            }
            let count = a_count + b_count;
            let mean = (a * a_count as f32 + b * b_count as f32) / count as f32;
            pools.pop();
            *pools.last_mut().unwrap() = (mean, count);
        }
    }
    pools
        .iter()
        .flat_map(|&(mean, count)| std::iter::repeat_n(mean, count))
        .zip(offsets)
        .map(|(position, offset)| position + offset)
        .collect()
}
