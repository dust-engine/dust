use bevy::reflect::Typed;

use super::tree::{Filter, Row, RowKind, TreeState, roots};
use super::*;

#[derive(Component, Reflect, Default, Clone, Copy)]
#[reflect(Component)]
struct Health {
    current: f32,
    max: f32,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
#[component(immutable)]
struct Team(u8);

#[derive(Component, Reflect, Default, PartialEq, Debug)]
#[reflect(Component)]
enum Stance {
    #[default]
    Idle,
    Moving(f32),
    Hiding {
        spot: Entity,
    },
}

#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
struct Score(u32);

/// Neither reflected nor holding data.
#[derive(Component)]
struct Marker;

fn on_add_health(_: On<Add<Health>>) {}

/// A world with the test types registered for reflection.
fn world() -> World {
    let mut world = World::new();
    world.init_resource::<AppTypeRegistry>();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let mut registry = registry.write();
    registry.register::<Health>();
    registry.register::<Team>();
    registry.register::<Stance>();
    registry.register::<Score>();
    registry.register::<Name>();
    registry.register::<Transform>();
    world
}

fn names(world: &World) -> ComponentNames {
    let mut names = ComponentNames::default();
    names.update(world.components());
    names
}

fn sorted(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort();
    entities
}

#[test]
fn history() {
    let [a, b, c, d] = [0, 1, 2, 3].map(|index| Entity::from_raw_u32(index).unwrap());
    let mut history = History::default();
    assert_eq!(history.back(), None);
    history.visit(a);
    history.visit(b);
    history.visit(c);
    assert_eq!(history.back(), Some(b));
    assert_eq!(history.back(), Some(a));
    assert_eq!(history.back(), None);
    assert_eq!(history.forward(), Some(b));
    // The tab visits the entry shown after going back, which keeps the
    // entries after it.
    history.visit(b);
    assert!(history.can_go_forward());
    history.visit(d);
    assert!(!history.can_go_forward());
    assert_eq!(history.back(), Some(b));
}

#[test]
fn sections() {
    let mut world = world();
    world.insert_resource(Score(3));
    let observer = world.add_observer(on_add_health).id();
    let system = world.register_system(|| {}).entity();
    let parent = world.spawn(Health::default()).id();
    let child = world.spawn((Health::default(), ChildOf(parent))).id();
    let names = names(&world);
    let info = WorldInfo::new(&world, &names);
    let [resources, observers, systems, entities] = roots(&info);

    let resources: Vec<String> = resources
        .iter()
        .map(|&entity| info.label(entity).unwrap().title.unwrap())
        .collect();
    assert_eq!(
        resources,
        ["AppTypeRegistry", "DefaultQueryFilters", "Score"]
    );
    assert_eq!(observers, [observer]);
    assert_eq!(systems, [system]);
    // Only roots: the child is below its parent.
    assert_eq!(entities, [parent]);
    assert!(!entities.contains(&child));
}

#[test]
fn labels() {
    let mut world = world();
    world.spawn_batch([Health::default(); 3]);
    let named = world
        .spawn((Name::new("teapot"), Health::default(), Marker))
        .id();
    let unnamed = world.spawn((Health::default(), Marker)).id();
    let despawned = world.spawn(Marker).id();
    world.despawn(despawned);
    world.insert_resource(Score(1));
    let observer = world.add_observer(on_add_health).id();
    let names = names(&world);
    let info = WorldInfo::new(&world, &names);

    let label = info.label(named).unwrap();
    assert_eq!(label.title.as_deref(), Some("teapot"));
    // `Marker` is on two entities and `Health` on five, so `Marker` comes
    // first. `Name` is the title, so it isn't listed.
    let components: Vec<&str> = label.components.iter().map(|&id| info.name(id)).collect();
    assert_eq!(components, ["Marker", "Health"]);
    assert_eq!(info.label(unnamed).unwrap().short(&info), "Marker");

    let score = world.resource_entity::<Score>().unwrap();
    let label = info.label(score).unwrap();
    assert_eq!(label.title.as_deref(), Some("Score"));
    assert_eq!(label.section, Section::Resources);
    let label = info.label(observer).unwrap();
    assert_eq!(label.title.as_deref(), Some("on_add_health"));
    assert_eq!(label.section, Section::Observers);
    assert!(info.label(despawned).is_none());
}

#[test]
fn filter() {
    let mut world = world();
    let castle = world.spawn(Name::new("Castle")).id();
    let tower = world.spawn((Name::new("Tower"), ChildOf(castle))).id();
    let guard = world
        .spawn((Name::new("Guard"), Health::default(), ChildOf(tower)))
        .id();
    let teapot = world.spawn((Name::new("Teapot"), Health::default())).id();
    let names = names(&world);
    let info = WorldInfo::new(&world, &names);
    let matches = |text: &str| {
        sorted(
            Filter::new(&info, text)
                .unwrap()
                .matches
                .into_iter()
                .collect(),
        )
    };

    // A name, ignoring case.
    assert_eq!(matches("teap"), [teapot]);
    // A component's name.
    assert_eq!(matches("health"), sorted(vec![guard, teapot]));
    // Each word has to match.
    assert_eq!(matches("health  GUARD"), [guard]);
    // The id, with or without the generation.
    assert_eq!(matches(&guard.to_string()), [guard]);
    assert_eq!(matches(&guard.index_u32().to_string()), [guard]);
    // Ancestors show, to keep a match in place.
    let filter = Filter::new(&info, "guard").unwrap();
    assert!(filter.shown.contains(&castle) && filter.shown.contains(&tower));
    assert!(!filter.matches.contains(&castle));
    assert!(Filter::new(&info, "  ").is_none());
}

#[test]
fn rows() {
    let mut world = world();
    let castle = world.spawn(Name::new("Castle")).id();
    let tower = world.spawn((Name::new("Tower"), ChildOf(castle))).id();
    let guard = world.spawn((Name::new("Guard"), ChildOf(tower))).id();
    let names = names(&world);
    let info = WorldInfo::new(&world, &names);
    let entities = |rows: Vec<Row>| -> Vec<(Entity, usize)> {
        rows.into_iter()
            .filter_map(|row| match row.kind {
                RowKind::Entity(entity) => Some((entity, row.depth)),
                RowKind::Section(_) => None,
            })
            .collect()
    };
    let all = [(castle, 1), (tower, 2), (guard, 3)];

    // At first, only the Entities section is open, and its roots closed.
    let mut state = TreeState::default();
    assert_eq!(entities(state.rows(&info, None)), [(castle, 1)]);
    state.set_expanded(&info, castle, true, true);
    assert_eq!(entities(state.rows(&info, None)), all);
    state.set_expanded(&info, castle, false, false);
    assert_eq!(entities(state.rows(&info, None)), [(castle, 1)]);

    // Revealing an entity expands its ancestors.
    let mut state = TreeState::default();
    state.expand_to(&info, guard);
    assert_eq!(entities(state.rows(&info, None)), all);

    // A filter shows each match, whatever is expanded.
    let filter = Filter::new(&info, "guard");
    let state = TreeState::default();
    assert_eq!(entities(state.rows(&info, filter.as_ref())), all);
}

#[test]
fn edits() {
    let mut world = world();
    let entity = world
        .spawn((
            Health {
                current: 1.0,
                max: 10.0,
            },
            Team(1),
            Stance::Idle,
        ))
        .id();
    let [health, team, stance] = [
        world.component_id::<Health>(),
        world.component_id::<Team>(),
        world.component_id::<Stance>(),
    ]
    .map(Option::unwrap);
    let registry = world.resource::<AppTypeRegistry>().clone();
    let changed = |world: &World| {
        world
            .entity(entity)
            .get_change_ticks::<Health>()
            .unwrap()
            .changed
    };
    let edit = |component, path: &[usize], value: Box<dyn PartialReflect>| Action::Edit {
        entity,
        component,
        path: path.to_vec(),
        value,
    };
    let enum_info = Stance::type_info().as_enum().unwrap();
    let variant = |name| value::default_variant(enum_info.variant(name).unwrap(), &registry.read());
    let mut state = EntityInspector::default();

    world.increment_change_tick();
    let before = changed(&world);
    // A path to no field changes nothing.
    apply(
        &mut world,
        &mut state,
        Some(&registry),
        vec![edit(health, &[7], Box::new(5.0f32))],
    );
    assert_eq!(changed(&world), before);
    apply(
        &mut world,
        &mut state,
        Some(&registry),
        vec![
            edit(health, &[0], Box::new(5.0f32)),
            // Immutable, so not edited in place.
            edit(team, &[0], Box::new(2u8)),
            edit(stance, &[], Box::new(variant("Moving").unwrap())),
        ],
    );
    assert_eq!(world.get::<Health>(entity).unwrap().current, 5.0);
    assert_ne!(changed(&world), before);
    assert_eq!(world.get::<Team>(entity).unwrap().0, 1);
    assert_eq!(world.get::<Stance>(entity), Some(&Stance::Moving(0.0)));
    // `Entity` has no `Default`, so the variant can't be built.
    assert!(variant("Hiding").is_none());
}

#[test]
fn entity_actions() {
    let mut world = world();
    let parent = world.spawn(Name::new("a")).id();
    let child = world.spawn(ChildOf(parent)).id();
    let other = world.spawn_empty().id();
    let mut state = EntityInspector::default();
    apply(
        &mut world,
        &mut state,
        None,
        vec![
            Action::SetDisabled(parent, true),
            Action::Rename(parent, "b".to_owned()),
            Action::Despawn(other),
            Action::Select(child),
            Action::Filter("x".to_owned()),
        ],
    );
    assert!(world.get::<Disabled>(child).is_some());
    assert_eq!(world.get::<Name>(parent).unwrap().as_str(), "b");
    assert!(world.get_entity(other).is_err());
    assert_eq!(state.selected, Some(child));
    assert_eq!(state.tree.filter, "x");
    apply(
        &mut world,
        &mut state,
        None,
        vec![Action::SetDisabled(parent, false)],
    );
    assert!(world.get::<Disabled>(child).is_none());
}

#[test]
fn element_paths() {
    let mut value = (
        1u8,
        vec![
            Health::default(),
            Health {
                current: 2.0,
                max: 3.0,
            },
        ],
    );
    let max = value::element_mut(&mut value, &[1, 1, 1]).unwrap();
    assert_eq!(max.try_downcast_ref::<f32>(), Some(&3.0));
    assert!(value::element_mut(&mut value, &[2]).is_none());
    assert!(value::element_mut(&mut value, &[1, 5]).is_none());
}

#[test]
fn preview_stops_at_max() {
    let long: Vec<u32> = (0..100_000).collect();
    let text = value::preview(&long, 20);
    assert!(text.ends_with('…'));
    assert!(text.len() <= 20 + '…'.len_utf8());
    assert_eq!(value::preview(&3.5f32, 20), "3.5");
}

/// Draws the tab with `selected` selected and `filter` in the tree's filter.
/// Twice, as egui sizes some widgets in the pass after they first show.
fn draw(world: &mut World, selected: Option<Entity>, filter: &str) {
    world.init_resource::<EntityInspector>();
    let mut state = world.resource_mut::<EntityInspector>();
    state.selected = selected;
    state.tree.filter = filter.to_owned();
    let system = world.register_system(entities_ui);
    let ctx = egui::Context::default();
    for _ in 0..2 {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| {
            world.run_system_with(system, ui).unwrap();
        });
        // No renderer takes the font atlas; egui asserts that something did.
        output.textures_delta.clear();
    }
}

#[test]
fn draws_each_kind_of_entity() {
    let mut world = world();
    world.insert_resource(Score(7));
    let observer = world.add_observer(on_add_health).id();
    let teapot = world
        .spawn((
            Name::new("teapot"),
            Health::default(),
            Stance::Moving(1.0),
            Team(3),
            Marker,
        ))
        .id();
    let lid = world.spawn((ChildOf(teapot), Transform::default())).id();
    let score = world.resource_entity::<Score>().unwrap();
    let despawned = world.spawn_empty().id();
    world.despawn(despawned);
    for selected in [
        None,
        Some(teapot),
        Some(lid),
        Some(observer),
        Some(score),
        Some(despawned),
    ] {
        for filter in ["", "teapot", "health 0v"] {
            draw(&mut world, selected, filter);
        }
    }
}
