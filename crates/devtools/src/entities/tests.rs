use bevy::ecs::system::RunSystemOnce as _;
use bevy::reflect::Typed;

use super::inspector::{Body, EntityView};
use super::tree::{Filter, Row, RowKind, RowView, TreeKey, TreeState, roots};
use super::value::{LeafValue, Number, Piece, ValueRowKind};
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

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
struct Inventory {
    items: Vec<u32>,
}

#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
struct Score(u32);

/// Neither reflected nor holding data.
#[derive(Component)]
struct Marker;

fn on_add_health(_: On<Add<Health>>) {}

/// A world with the test types registered for reflection, and the tab's
/// state.
fn world() -> World {
    let mut world = World::new();
    world.init_resource::<AppTypeRegistry>();
    world.init_resource::<EntityInspector>();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let mut registry = registry.write();
    registry.register::<Health>();
    registry.register::<Team>();
    registry.register::<Stance>();
    registry.register::<Inventory>();
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

/// Carries out `actions` as the tab does.
fn act(world: &mut World, actions: impl IntoIterator<Item = Action>) {
    world.resource_scope(|world, mut state: Mut<EntityInspector>| {
        let registry = world.get_resource::<AppTypeRegistry>().cloned();
        apply(world, &mut state, registry.as_ref(), actions);
    });
}

/// The inspector's view of the selected entity, from a snapshot taken
/// afresh, as when the tab is first shown.
fn inspect(world: &mut World, entity: Entity) -> EntityView {
    world.resource_mut::<EntityInspector>().selected = Some(entity);
    let snapshot = world.run_system_once(snapshot).unwrap().unwrap();
    snapshot.inspector.unwrap().entity.unwrap()
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

    let resources: Vec<SharedString> = resources
        .iter()
        .map(|&entity| info.label(entity).unwrap().title.unwrap())
        .collect();
    assert_eq!(
        resources,
        [
            "AppTypeRegistry",
            "DefaultQueryFilters",
            "EntityInspector",
            "Score"
        ]
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
    let disabled = world.spawn((Marker, Disabled)).id();
    let names = names(&world);
    let info = WorldInfo::new(&world, &names);

    let label = info.label(named).unwrap();
    assert_eq!(label.title.as_deref(), Some("teapot"));
    // `Marker` is on three entities and `Health` on five, so `Marker` comes
    // first. `Name` is the title, so it isn't listed.
    assert_eq!(label.components, ["Marker", "Health"]);
    assert_eq!(info.label(unnamed).unwrap().short(), "Marker");

    let score = world.resource_entity::<Score>().unwrap();
    let label = info.label(score).unwrap();
    assert_eq!(label.title.as_deref(), Some("Score"));
    assert_eq!(label.section, Section::Resources);
    let label = info.label(observer).unwrap();
    assert_eq!(label.title.as_deref(), Some("on_add_health"));
    assert_eq!(label.section, Section::Observers);
    assert!(info.label(despawned).is_none());
    let label = info.label(disabled).unwrap();
    assert!(label.disabled && label.has_disabled);
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
fn keys() {
    let mut world = world();
    let castle = world.spawn(Name::new("Castle")).id();
    let tower = world.spawn((Name::new("Tower"), ChildOf(castle))).id();
    world.spawn((Name::new("Guard"), ChildOf(tower)));
    let names = names(&world);
    let info = WorldInfo::new(&world, &names);
    let mut state = TreeState::default();
    let mut selected = None;
    let mut press = |key, selected: &mut Option<Entity>| state.key(&info, key, selected);

    // ↓ with nothing selected selects the first entity; sections are skipped.
    press(TreeKey::Down, &mut selected);
    assert_eq!(selected, Some(castle));
    // → first expands, then moves to the first child.
    press(TreeKey::Right, &mut selected);
    assert_eq!(selected, Some(castle));
    press(TreeKey::Right, &mut selected);
    assert_eq!(selected, Some(tower));
    // ← on a closed entity moves to its parent, then collapses it.
    press(TreeKey::Left, &mut selected);
    assert_eq!(selected, Some(castle));
    press(TreeKey::Left, &mut selected);
    press(TreeKey::Down, &mut selected);
    assert_eq!(selected, Some(castle));
    press(TreeKey::Up, &mut selected);
    assert_eq!(selected, Some(castle));
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
    let changed = |world: &World| {
        world
            .entity(entity)
            .get_change_ticks::<Health>()
            .unwrap()
            .changed
    };
    let edit = |component, path: &[usize], value: Edit| Action::Edit {
        entity,
        component,
        path: path.to_vec(),
        value,
    };

    world.increment_change_tick();
    let before = changed(&world);
    // A path to no field changes nothing.
    act(
        &mut world,
        [edit(health, &[7], Edit::Value(Box::new(5.0f32)))],
    );
    assert_eq!(changed(&world), before);
    act(
        &mut world,
        [
            edit(health, &[0], Edit::Value(Box::new(5.0f32))),
            // Immutable, so not edited in place.
            edit(team, &[0], Edit::Value(Box::new(2u8))),
            edit(stance, &[], Edit::Variant("Moving".into())),
        ],
    );
    assert_eq!(world.get::<Health>(entity).unwrap().current, 5.0);
    assert_ne!(changed(&world), before);
    assert_eq!(world.get::<Team>(entity).unwrap().0, 1);
    assert_eq!(world.get::<Stance>(entity), Some(&Stance::Moving(0.0)));

    // `Entity` has no `Default`, so the variant can't be built.
    act(
        &mut world,
        [edit(stance, &[], Edit::Variant("Hiding".into()))],
    );
    assert_eq!(world.get::<Stance>(entity), Some(&Stance::Moving(0.0)));
    let registry = world.resource::<AppTypeRegistry>().read();
    let enum_info = Stance::type_info().as_enum().unwrap();
    assert!(value::default_variant(enum_info.variant("Hiding").unwrap(), &registry).is_none());
}

#[test]
fn entity_actions() {
    let mut world = world();
    let parent = world.spawn(Name::new("a")).id();
    let child = world.spawn(ChildOf(parent)).id();
    let other = world.spawn_empty().id();
    act(
        &mut world,
        [
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
    let state = world.resource::<EntityInspector>();
    assert_eq!(state.selected, Some(child));
    assert_eq!(state.tree.filter, "x");
    act(&mut world, [Action::SetDisabled(parent, false)]);
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

#[test]
fn numbers() {
    assert_eq!(Number::U8(3).parse(" 200 "), Some(Number::U8(200)));
    assert_eq!(Number::U8(3).parse("300"), None);
    assert_eq!(Number::F32(0.0).parse("1.5"), Some(Number::F32(1.5)));
    // Dragged to, an integer rounds and stays within its type.
    assert_eq!(Number::I8(0).with_f64(2.6), Number::I8(3));
    assert_eq!(Number::U8(0).with_f64(-4.0), Number::U8(0));
    assert_eq!(Number::F64(0.0).with_f64(2.6), Number::F64(2.6));
}

#[test]
fn snapshots_each_kind_of_entity() {
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
            act(
                &mut world,
                [
                    Action::Filter(filter.to_owned()),
                    Action::ComponentFilter(filter.to_owned()),
                ],
            );
            world.resource_mut::<EntityInspector>().selected = selected;
            let snapshot = world.run_system_once(snapshot).unwrap().unwrap();
            let tree = snapshot.tree.unwrap();
            let inspector = snapshot.inspector.unwrap();
            assert_eq!(inspector.selected, selected);
            let shown = selected.filter(|&entity| entity != despawned);
            assert_eq!(inspector.entity.map(|view| view.entity), shown);
            // The tree reveals the selection when it shows.
            if let Some(index) = tree.reveal {
                assert!(matches!(
                    tree.rows[index],
                    RowView::Entity { entity, .. } if Some(entity) == selected
                ));
            }
        }
    }

    act(&mut world, [Action::Filter(String::new())]);
    let snapshot = world.run_system_once(snapshot).unwrap().unwrap();
    let tree = snapshot.tree.unwrap();
    // Selecting the lid expanded the teapot to show it.
    let row = |entity| {
        tree.rows.iter().find_map(|row| match row {
            RowView::Entity {
                entity: shown,
                depth,
                open,
                ..
            } if *shown == entity => Some((*depth, *open)),
            _ => None,
        })
    };
    assert_eq!(row(teapot), Some((1, true)));
    assert_eq!(row(lid), Some((2, false)));
}

#[test]
fn inspector_values() {
    let mut world = world();
    let teapot = world
        .spawn((
            Name::new("teapot"),
            Health {
                current: 1.0,
                max: 2.0,
            },
            Stance::Moving(1.0),
            Team(3),
            Inventory {
                items: vec![1, 2, 3],
            },
            Marker,
        ))
        .id();
    act(&mut world, [Action::ComponentFilter(String::new())]);
    let view = inspect(&mut world, teapot);
    fn component<'a>(view: &'a EntityView, name: &str) -> (ComponentId, Option<&'a Body>) {
        let component = view
            .components
            .iter()
            .find(|component| component.name == name)
            .unwrap();
        (component.id, component.body.as_ref())
    }

    let (_, name) = component(&view, "Name");
    assert!(matches!(name, Some(Body::Name(text)) if text == "teapot"));
    // A marker has no value to show.
    let (_, marker) = component(&view, "Marker");
    assert!(marker.is_none());

    // A struct of two numbers fits on one row, and can be edited.
    let (_, Some(Body::Rows(rows))) = component(&view, "Health") else {
        panic!("Health has rows");
    };
    let [row] = &rows[..] else {
        panic!("Health fits on one row");
    };
    let ValueRowKind::Inline { name: None, pieces } = &row.kind else {
        panic!("Health is inline");
    };
    let numbers: Vec<(&[usize], Number, bool)> = pieces
        .iter()
        .filter_map(|piece| match piece {
            Piece::Leaf(leaf) => match leaf.value {
                LeafValue::Number(number) => Some((&leaf.path[..], number, leaf.editable)),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(
        numbers,
        [
            (&[0][..], Number::F32(1.0), true),
            (&[1][..], Number::F32(2.0), true)
        ]
    );

    // An immutable component's value can't be edited.
    let (_, Some(Body::Rows(rows))) = component(&view, "Team") else {
        panic!("Team has rows");
    };
    assert!(matches!(
        &rows[0].kind,
        ValueRowKind::Inline { pieces, .. }
            if matches!(&pieces[..], [Piece::Leaf(leaf)] if !leaf.editable)
    ));

    // An enum's variant can be picked from all of its variants.
    let (_, Some(Body::Rows(rows))) = component(&view, "Stance") else {
        panic!("Stance has rows");
    };
    let ValueRowKind::Inline { pieces, .. } = &rows[0].kind else {
        panic!("Moving(f32) is inline");
    };
    let Piece::Variant(variant) = &pieces[0] else {
        panic!("the variant comes first");
    };
    assert_eq!(variant.current, "Moving");
    assert_eq!(
        variant.choices.as_deref(),
        Some(
            &[
                ("Idle".into(), true),
                ("Moving".into(), true),
                ("Hiding".into(), false)
            ][..]
        )
    );

    // A list is closed until opened, which shows its elements.
    let (inventory, Some(Body::Rows(rows))) = component(&view, "Inventory") else {
        panic!("Inventory has rows");
    };
    assert!(matches!(
        &rows[..],
        [row] if matches!(&row.kind, ValueRowKind::Nested { open: false, .. })
    ));
    act(&mut world, [Action::ToggleValue(inventory, vec![0])]);
    let view = inspect(&mut world, teapot);
    let (_, Some(Body::Rows(rows))) = component(&view, "Inventory") else {
        panic!("Inventory has rows");
    };
    let depths: Vec<usize> = rows.iter().map(|row| row.depth).collect();
    assert_eq!(depths, [0, 1, 1, 1]);

    // A closed component sends no value.
    act(&mut world, [Action::ToggleComponent(inventory)]);
    let view = inspect(&mut world, teapot);
    assert!(component(&view, "Inventory").1.is_none());

    // The component filter leaves the others out.
    act(&mut world, [Action::ComponentFilter("heal".to_owned())]);
    let view = inspect(&mut world, teapot);
    let names: Vec<&str> = view
        .components
        .iter()
        .map(|component| &*component.name)
        .collect();
    assert_eq!(names, ["Health"]);
    assert!(view.filtered);
}

#[test]
fn snapshot_sends_changes() {
    let mut world = world();
    let teapot = world.spawn((Name::new("teapot"), Health::default())).id();
    let other = world.spawn(Health::default()).id();
    let health = world.component_id::<Health>().unwrap();
    world.resource_mut::<EntityInspector>().selected = Some(teapot);
    let system = world.register_system(snapshot);
    let run = |world: &mut World| {
        // As the systems run between two frames do.
        world.increment_change_tick();
        world.run_system(system).unwrap()
    };

    // Everything when first shown, then nothing while nothing changes.
    let first = run(&mut world).unwrap();
    assert!(first.tree.is_some());
    assert!(first.inspector.unwrap().changed.is_empty());
    assert!(run(&mut world).is_none());

    // The inspector when the selected entity changes, with what changed.
    world.get_mut::<Health>(teapot).unwrap().current = 2.0;
    let snapshot = run(&mut world).unwrap();
    assert!(snapshot.tree.is_none());
    assert_eq!(snapshot.inspector.unwrap().changed, [health]);
    // Not when another entity changes.
    world.get_mut::<Health>(other).unwrap().current = 2.0;
    assert!(run(&mut world).is_none());

    // Both when the tab's state changes.
    act(&mut world, [Action::Select(other)]);
    let snapshot = run(&mut world).unwrap();
    assert!(snapshot.tree.unwrap().reveal.is_some());
    // A newly selected entity doesn't show as all changed.
    assert!(snapshot.inspector.unwrap().changed.is_empty());
    assert!(run(&mut world).is_none());
    // And when a component is added to the selected entity.
    world.entity_mut(other).insert(Marker);
    assert!(run(&mut world).unwrap().inspector.is_some());
}
