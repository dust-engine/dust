//! The "Schedules" DevTools tab: the systems of a schedule picked from a
//! menu, drawn as a graph of what runs before what, with each set as a box
//! around its systems, and an inspector for the system or set selected in
//! it. For a system: what it waits for and what waits for it, and why; the
//! sets it's in; what it accesses; and the systems it conflicts with that
//! nothing orders it against. For a set: the sets it's in and that are in
//! it, what it's ordered against, and its systems.
//!
//! The tab reads the world when it's shown, and again when the user picks
//! another schedule or refreshes the one shown; it doesn't follow the
//! schedules from frame to frame. [`snapshot`] reads the list of schedules
//! and the graph of the one picked (see [`graph`]), and [`SchedulesView`]
//! lays the graph out (see [`layout`]) and draws it, on the UI thread.
//!
//! DevTools reads the world in `Last`, while `Main` runs it, and both are
//! out of the world then: their graphs can't be shown.

mod graph;
mod layout;
#[cfg(test)]
mod tests;
mod ui;

use bevy::app::{FixedMainScheduleOrder, MainScheduleOrder};
use bevy::ecs::schedule::{
    InternedScheduleLabel, ScheduleBuilt, ScheduleLabel as _, Schedules, SystemKey,
};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use gpui::SharedString;

pub(crate) use ui::SchedulesView;

use graph::GraphSnapshot;

/// Records what the "Schedules" DevTools tab can't read from the schedules
/// themselves: the edges Bevy adds while it builds a schedule, and the order
/// `Main` runs the schedules in. The tab itself is created on the UI thread,
/// by [`overlay`](crate::overlay).
#[derive(Default)]
pub struct SchedulesPlugin;

impl Plugin for SchedulesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScheduleRecords>()
            .add_observer(record_build);
    }

    fn cleanup(&self, app: &mut App) {
        // `Main` takes its order out of the world while it runs the
        // schedules, which it is doing whenever the tab reads the world, so
        // it's read here, once all plugins have added their schedules.
        let world = app.world_mut();
        let mut order = Vec::new();
        if let Some(main) = world.get_resource::<MainScheduleOrder>() {
            // A schedule run at startup and each frame, as `StateTransition`
            // is, is listed with the frame's.
            order.extend(
                main.startup_labels
                    .iter()
                    .filter(|label| !main.labels.contains(label))
                    .map(|&label| (label, Group::Startup)),
            );
            order.extend(main.labels.iter().map(|&label| (label, Group::Main)));
        }
        if let Some(fixed) = world.get_resource::<FixedMainScheduleOrder>() {
            order.extend(fixed.labels.iter().map(|&label| (label, Group::Fixed)));
        }
        world.resource_mut::<ScheduleRecords>().order = order;
    }
}

#[derive(Resource, Default)]
struct ScheduleRecords {
    /// The edges Bevy's build passes added when each schedule was last
    /// built: those of the sync points it adds to apply commands, unless the
    /// app adds passes of its own.
    added_edges: HashMap<InternedScheduleLabel, Vec<(SystemKey, SystemKey)>>,
    /// The schedules `Main` runs, in the order it runs them: at startup, then
    /// each frame, then those `FixedMain` runs.
    order: Vec<(InternedScheduleLabel, Group)>,
}

/// Keeps the edges a schedule's build added, which Bevy reports only to
/// `ScheduleBuilt` observers.
fn record_build(built: On<ScheduleBuilt>, mut records: ResMut<ScheduleRecords>) {
    let edges = &built.build_metadata.edges_added_by_build_passes;
    records
        .added_edges
        .insert(built.label, edges.iter().copied().collect());
}

/// Where the schedule picker lists a schedule.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Group {
    /// Run by `Main` once, before the first frame.
    Startup,
    /// Run by `Main` each frame.
    Main,
    /// Run by `FixedMain`, once for each fixed timestep that passed.
    Fixed,
    /// Run by a system, as `RunFixedMainLoop` runs `FixedMain`, or on a state
    /// transition.
    Other,
}

impl Group {
    fn title(self) -> &'static str {
        match self {
            Group::Startup => "Startup",
            Group::Main => "Each frame",
            Group::Fixed => "Fixed timestep",
            Group::Other => "Other",
        }
    }
}

/// A schedule, as the picker lists it.
struct ScheduleEntry {
    label: InternedScheduleLabel,
    name: SharedString,
    group: Group,
    /// How many systems the schedule has; `None` for one running while the
    /// tab reads the world, which can't be shown.
    systems: Option<usize>,
}

/// What the tab shows, read by [`snapshot`].
struct Snapshot {
    /// The world's schedules, in the order the picker lists them.
    schedules: Vec<ScheduleEntry>,
    /// The picked schedule's graph; `None` if there's no schedule to show.
    graph: Option<GraphSnapshot>,
}

/// The world's schedules, and the graph of `picked`, or of `Update` if
/// `picked` is `None` or can't be shown.
fn snapshot(world: &World, picked: Option<InternedScheduleLabel>) -> Snapshot {
    let Some(schedules) = world.get_resource::<Schedules>() else {
        return Snapshot {
            schedules: Vec::new(),
            graph: None,
        };
    };
    let records = world.get_resource::<ScheduleRecords>();
    let order = records.map_or(&[][..], |records| &records.order);
    let entry = |label: InternedScheduleLabel, systems| {
        let position = order.iter().position(|&(other, _)| other == label);
        let group = position.map_or(Group::Other, |i| order[i].1);
        let entry = ScheduleEntry {
            label,
            name: format!("{label:?}").into(),
            group,
            systems,
        };
        (position, entry)
    };
    let mut entries: Vec<(Option<usize>, ScheduleEntry)> = schedules
        .iter()
        .map(|(_, schedule)| entry(schedule.label(), Some(schedule.systems_len())))
        .chain(
            schedules
                .get_temporarily_removed()
                .into_iter()
                .map(|label| entry(label, None)),
        )
        .collect();
    // In the order they run, and the others by name.
    entries.sort_by_cached_key(|(position, entry)| {
        (entry.group, *position, entry.name.to_lowercase())
    });
    let entries: Vec<ScheduleEntry> = entries.into_iter().map(|(_, entry)| entry).collect();

    let shown = |label: &InternedScheduleLabel| schedules.contains(*label);
    let label = picked
        .filter(shown)
        .or_else(|| Some(Update.intern()).filter(shown))
        .or_else(|| entries.iter().map(|entry| entry.label).find(shown));
    let graph = label.and_then(|label| {
        let added = records
            .and_then(|records| records.added_edges.get(&label))
            .map_or(&[][..], Vec::as_slice);
        Some(graph::read(
            schedules.get(label)?,
            world.components(),
            added,
        ))
    });
    Snapshot {
        schedules: entries,
        graph,
    }
}
