//! Changes that a system queues on its flecs stage while it runs.

use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering::Relaxed;

use crate::sys::{ecs_delete, ecs_is_alive, ecs_new, ecs_remove_id, ecs_world_t};
use crate::world::{assert_can_delete, registered_id_in};
use crate::{Component, Entity};

/// Changes to the world that a system queues while it runs. flecs defers the world then, so they
/// go on the flecs stage the system runs on, and flecs applies them when it next merges the stage:
/// [`World::run_system`](crate::world::World::run_system) once the system has run, and
/// [`World::progress`](crate::world::World::progress) once the frame's systems have run, and also
/// after the [`OnStart`](crate::system::Phase::OnStart) systems in the first frame.
///
/// flecs applies them in the order they were queued, and skips any whose entity is dead by then.
/// It applies them even if the system panics after queueing them, as it still ends the run.
pub struct Commands<'w> {
    /// The stage the system runs on, which queues what's done to it until flecs merges it.
    stage: *mut ecs_world_t,
    /// The world the stage is of, to read.
    world: *const ecs_world_t,
    _run: PhantomData<&'w ()>,
}

impl Commands<'_> {
    /// # Safety
    ///
    /// `stage` must be the stage that a system is running on, which flecs defers, of `world`,
    /// which isn't multi-threaded. Both must stay valid, and only used by this thread, for as long
    /// as this exists.
    pub(crate) unsafe fn new(stage: *mut ecs_world_t, world: *const ecs_world_t) -> Self {
        Self {
            stage,
            world,
            _run: PhantomData,
        }
    }

    /// Makes an entity now, rather than when the commands are applied, so that commands can use
    /// it. It has no components.
    pub fn new_entity(&mut self) -> Entity {
        // SAFETY: flecs makes an entity at once even while the world is deferred, as long as it
        // isn't multi-threaded, which `new`'s caller upholds. It appends the entity to the root
        // table, which has no components, so no row points into what moves: a query with a
        // required term, the only kind that can read a row's `Entity`, never matches that table.
        let id = unsafe { ecs_new(self.stage) };
        Entity(NonZeroU64::new(id).expect("ecs_new returned 0"))
    }

    /// Queues removing `T` from `entity` and dropping it, as [`World::remove`] does. Does nothing
    /// if `entity` is dead or has no `T` by the time flecs applies it.
    ///
    /// [`World::remove`]: crate::world::World::remove
    pub fn remove<T: Component>(&mut self, entity: Entity) {
        if !self.is_alive(entity) {
            return;
        }
        let id = unsafe { crate::sys::flecs_component_ids_get_alive(self.world, T::slot().load(Relaxed)) };
        let Some(component) = NonZeroU64::new(id).map(Entity) else {
            return;
        };
        // SAFETY: the entity is alive and `T` is registered, which flecs only checks in debug
        // builds. The stage is deferred, so this only queues the removal.
        unsafe { ecs_remove_id(self.stage, entity.id(), component.id()) };
    }

    /// Queues deleting `entity` and dropping its components, as [`World::delete`] does. Does
    /// nothing if `entity` is dead by the time flecs applies it.
    ///
    /// # Panics
    ///
    /// Now, rather than when flecs applies it, if `entity` is a component's, a query's or a
    /// system's own, as [`World::delete`] does.
    ///
    /// [`World::delete`]: crate::world::World::delete
    pub fn delete(&mut self, entity: Entity) {
        if !self.is_alive(entity) {
            return;
        }
        // SAFETY: `new`'s caller keeps the world valid, and the entity is alive.
        unsafe { assert_can_delete(self.world, entity) };
        // SAFETY: an `Entity` never names a built-in, and what it names doesn't become a
        // component, a query or a system: a recycled id gets a new generation. The stage is
        // deferred, so this only queues the deletion.
        unsafe { ecs_delete(self.stage, entity.id()) };
    }

    fn is_alive(&self, entity: Entity) -> bool {
        // SAFETY: `new`'s caller keeps the world valid, and `ecs_is_alive` accepts any non-zero
        // id.
        unsafe { ecs_is_alive(self.world, entity.id()) }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::query::With;
    use crate::system::{Phase, Run};
    use crate::world::World;

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    /// Counts its drops.
    #[derive(Component)]
    struct Tracked(Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Relaxed);
        }
    }

    /// An entity with each `Count`.
    fn counts(world: &mut World, values: impl IntoIterator<Item = u32>) -> Vec<Entity> {
        values
            .into_iter()
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Count(value));
                entity
            })
            .collect()
    }

    /// A row can still be read after its system queues a change to it.
    #[test]
    fn run_system_applies_commands_once_the_system_has_run() {
        let mut world = World::new();
        let entities = counts(&mut world, 0..3);
        let system = world.register_system(
            |Run {
                 iter, mut commands, ..
             }: Run<(Entity, &Count)>| {
                for (entity, count) in iter {
                    let old = count.0;
                    match old {
                        0 => commands.remove::<Count>(entity),
                        1 => commands.delete(entity),
                        _ => {}
                    }
                    assert_eq!(count.0, old);
                }
            },
        );
        world.run_system(system);

        assert!(world.is_alive(entities[0]));
        assert!(!world.has::<Count>(entities[0]));
        assert!(!world.is_alive(entities[1]));
        assert_eq!(world.get::<Count>(entities[2]), Some(&Count(2)));
    }

    #[test]
    fn progress_applies_commands_once_the_frames_systems_have_run() {
        let mut world = World::new();
        let entities = counts(&mut world, 0..2);
        world.add_system(
            Phase::OnUpdate,
            |Run {
                 iter, mut commands, ..
             }: Run<Entity, With<Count>>| {
                for entity in iter {
                    commands.delete(entity);
                }
            },
        );
        let rows_seen = Arc::new(Mutex::new(Vec::new()));
        world.add_system(Phase::PostUpdate, {
            let rows_seen = rows_seen.clone();
            move |Run { iter, .. }: Run<&Count>| rows_seen.lock().unwrap().push(iter.count())
        });

        world.progress(1.0);
        assert!(entities.iter().all(|&entity| !world.is_alive(entity)));
        world.progress(1.0);
        assert_eq!(*rows_seen.lock().unwrap(), [2, 0]);
    }

    /// flecs runs the `OnStart` systems as a pipeline of their own, and merges at its end.
    #[test]
    fn on_start_commands_apply_before_the_first_frames_other_phases() {
        let mut world = World::new();
        counts(&mut world, 0..2);
        world.add_system(
            Phase::OnStart,
            |Run {
                 iter, mut commands, ..
             }: Run<Entity, With<Count>>| {
                for entity in iter {
                    commands.remove::<Count>(entity);
                }
            },
        );
        let rows_seen = Arc::new(Mutex::new(Vec::new()));
        world.add_system(Phase::OnLoad, {
            let rows_seen = rows_seen.clone();
            move |Run { iter, .. }: Run<&Count>| rows_seen.lock().unwrap().push(iter.count())
        });

        world.progress(1.0);
        assert_eq!(*rows_seen.lock().unwrap(), [0]);
    }

    /// `run_system` defers the world, and `progress` also makes it read-only.
    #[test]
    fn new_entity_makes_an_entity_at_once() {
        let mut world = World::new();
        counts(&mut world, [0]);
        let made = Arc::new(Mutex::new(Vec::new()));
        let system = world.add_system(Phase::OnUpdate, {
            let made = made.clone();
            move |Run { mut commands, .. }: Run<&Count>| {
                let entity = commands.new_entity();
                let doomed = commands.new_entity();
                commands.delete(doomed);
                made.lock().unwrap().push((entity, doomed));
            }
        });
        world.run_system(system);
        world.progress(1.0);

        let made = made.lock().unwrap();
        assert_eq!(made.len(), 2);
        for &(entity, doomed) in made.iter() {
            assert!(world.is_alive(entity));
            assert!(!world.has::<Count>(entity));
            assert!(!world.is_alive(doomed));
        }
    }

    /// flecs skips a command whose entity is dead when it applies it.
    #[test]
    fn a_command_on_an_entity_deleted_earlier_in_the_queue_does_nothing() {
        let mut world = World::new();
        let entities = counts(&mut world, 0..2);
        let system = world.register_system(
            |Run {
                 iter, mut commands, ..
             }: Run<(Entity, &Count)>| {
                for (entity, _) in iter {
                    commands.delete(entity);
                    commands.remove::<Count>(entity);
                    commands.delete(entity);
                }
            },
        );
        world.run_system(system);
        assert!(entities.iter().all(|&entity| !world.is_alive(entity)));
    }

    #[test]
    fn commands_on_a_dead_entity_or_an_unstored_component_do_nothing() {
        #[derive(Component)]
        struct NeverStored;

        let mut world = World::new();
        let entities = counts(&mut world, 0..2);
        let dead = world.new_entity();
        world.delete(dead);
        let system = world.register_system(
            move |Run {
                      iter, mut commands, ..
                  }: Run<(Entity, &Count)>| {
                commands.remove::<Count>(dead);
                commands.delete(dead);
                for (entity, _) in iter {
                    commands.remove::<NeverStored>(entity);
                }
            },
        );
        world.run_system(system);
        assert!(!world.is_alive(dead));
        assert!(entities.iter().all(|&entity| world.has::<Count>(entity)));
    }

    #[test]
    fn removing_and_deleting_drop_each_value_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let entities = counts(&mut world, 0..2);
        for &entity in &entities {
            world.set(entity, Tracked(drops.clone()));
        }
        let system = world.register_system(
            |Run {
                 iter, mut commands, ..
             }: Run<(Entity, &Count)>| {
                for (entity, count) in iter {
                    if count.0 == 0 {
                        commands.remove::<Tracked>(entity);
                    } else {
                        commands.delete(entity);
                    }
                }
            },
        );
        world.run_system(system);
        assert_eq!(drops.load(Relaxed), 2);
        drop(world);
        assert_eq!(drops.load(Relaxed), 2);
    }

    #[test]
    fn commands_queued_before_a_panic_still_apply() {
        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let system = world.register_system(
            |Run {
                 iter, mut commands, ..
             }: Run<(Entity, &Count)>| {
                for (entity, _) in iter {
                    commands.remove::<Count>(entity);
                    panic!("the system panicked");
                }
            },
        );
        let payload = catch_unwind(AssertUnwindSafe(|| world.run_system(system))).unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"the system panicked"));
        assert!(!world.has::<Count>(entities[0]));
    }

    #[test]
    #[should_panic = "a component can't be deleted"]
    fn deleting_a_component_through_commands_panics() {
        let mut world = World::new();
        counts(&mut world, [0]);
        let component = world.register::<Count>();
        let system = world.register_system(move |Run { mut commands, .. }: Run<&Count>| {
            commands.delete(component);
        });
        world.run_system(system);
    }
}
