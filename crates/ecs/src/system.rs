//! Systems: functions that flecs runs over the rows of a query, like Bevy's one-shot systems.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::num::NonZeroU64;

use crate::Entity;
use crate::query::{QueryData, QueryFilter, QueryIter};
use crate::state::describe;
use crate::sys::{ecs_iter_t, ecs_run, ecs_system_desc_t, ecs_system_init};
use crate::world::World;

/// A system that [`World::register_system`] stored, to run with [`World::run_system`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SystemId {
    entity: Entity,
    /// [`World::id`] of the world that registered it.
    world: u64,
}

/// What a system gets each time it runs: the rows of a query of `D` whose rows also pass the
/// filter `F`, such as `With<T>`. A system's parameter type states its query, so a closure
/// annotates it, as in `|Run { rows, .. }: Run<&mut Position>|`. The world is deferred while a
/// system runs, so a system gets this rather than the world.
///
/// It has a private field, and more fields may be added, so take it apart with `..`.
pub struct Run<'w, D: QueryData, F: QueryFilter = ()> {
    /// The rows of the system's query.
    pub iter: QueryIter<'w, D>,
    _filter: PhantomData<fn() -> F>,
}

impl World {
    /// Stores `system` to run over the rows of the query its [`Run`] parameter states, and gives
    /// the id that [`Self::run_system`] runs it by. What it captures persists between runs. Fails
    /// to compile if the query's filter has [`DetectChanges`](crate::query::DetectChanges), as a
    /// system can't ask whether its query changed.
    ///
    /// It runs inside flecs, so a panic in it aborts the process. It's dropped with the world,
    /// under the same lock as a component's `Drop`, so its captures mustn't create or drop a
    /// world when dropped, and a panic there aborts too.
    ///
    /// ```
    /// use dust_ecs::Component;
    /// use dust_ecs::query::With;
    /// use dust_ecs::system::Run;
    /// use dust_ecs::world::World;
    ///
    /// #[derive(Component)]
    /// struct Position(f32);
    ///
    /// #[derive(Component)]
    /// struct Velocity(f32);
    ///
    /// #[derive(Component)]
    /// struct Moving;
    ///
    /// fn movement(Run { rows, .. }: Run<(&mut Position, &Velocity), With<Moving>>) {
    ///     for (position, velocity) in rows {
    ///         position.0 += velocity.0;
    ///     }
    /// }
    ///
    /// let mut world = World::new();
    /// let moving = world.new_entity();
    /// world.set(moving, Position(1.0));
    /// world.set(moving, Velocity(2.0));
    /// world.set(moving, Moving);
    /// let still = world.new_entity();
    /// world.set(still, Position(1.0));
    /// world.set(still, Velocity(2.0));
    ///
    /// let system = world.register_system(movement);
    /// world.run_system(system);
    /// world.run_system(system);
    ///
    /// assert_eq!(world.get::<Position>(moving).unwrap().0, 5.0);
    /// assert_eq!(world.get::<Position>(still).unwrap().0, 1.0);
    /// ```
    pub fn register_system<D, F, S>(&mut self, system: S) -> SystemId
    where
        D: QueryData,
        F: QueryFilter,
        S: for<'w> FnMut(Run<'w, D, F>) + Send + 'static,
    {
        const {
            assert!(
                !F::ACCESS.detects_changes,
                "a system can't ask whether its query changed, so it can't use `DetectChanges`"
            );
        }
        // SAFETY: flecs documents all zeroes as the default, and it's valid here.
        let mut desc: ecs_system_desc_t = unsafe { std::mem::zeroed() };
        desc.query = describe::<D, F>(self);
        desc.run = Some(run::<D, F, S>);
        desc.run_ctx = Box::into_raw(Box::new(system)).cast();
        desc.run_ctx_free = Some(drop_system::<S>);
        // SAFETY: the world is valid, `&mut self` makes this access exclusive, and `describe`
        // registered the query's components. flecs keeps the box, and frees it with `run_ctx_free`.
        let entity = unsafe { ecs_system_init(self.ptr.as_ptr(), &desc) };
        SystemId {
            entity: Entity(NonZeroU64::new(entity).unwrap()),
            world: self.id,
        }
    }

    /// Runs a system that this world registered.
    ///
    /// # Panics
    ///
    /// If another world registered it.
    pub fn run_system(&mut self, id: SystemId) {
        assert_eq!(
            id.world, self.id,
            "a system can only be run by the world that registered it"
        );
        // SAFETY: this world registered the system, as just checked, and `delete` refuses its
        // entity, so it's alive. `&mut self` makes this access exclusive.
        unsafe { ecs_run(self.ptr.as_ptr(), id.entity.id(), 0.0, std::ptr::null_mut()) };
    }
}

/// flecs's `run` callback for a system `S` over the rows of `D` that pass `F`.
unsafe extern "C" fn run<D, F, S>(it: *mut ecs_iter_t)
where
    D: QueryData,
    F: QueryFilter,
    S: for<'w> FnMut(Run<'w, D, F>),
{
    let it = unsafe { it.read() };
    let system = it.run_ctx.cast::<S>();
    // SAFETY: `register_system` gave flecs this callback with a query from `describe::<D, F>`, and
    // nothing has advanced the iterator. Once this returns, flecs only reads its own copy's
    // `interrupted_by`, which is still 0, as the system can't interrupt the run. It doesn't finish
    // its copy, as it only does that for a query whose terms all have no source, and this query's
    // terms all have one, the row's entity or a singleton's. Only `run_system` runs a system, which
    // borrows the world exclusively until this returns.
    let rows = unsafe { QueryIter::from_iter(it) };
    // SAFETY: `register_system` put an `S` here, which only `drop_system` drops. Nothing else uses it
    // while this runs, as `run_system` borrows the world exclusively, and the system can't reach
    // the world to run itself again.
    unsafe {
        (*system)(Run {
            iter: rows,
            _filter: PhantomData,
        })
    };
}

/// flecs's callback to free a system `S`.
unsafe extern "C" fn drop_system<S>(system: *mut c_void) {
    // SAFETY: flecs calls this once, when it deletes the system, with the box `register_system`
    // made.
    drop(unsafe { Box::from_raw(system.cast::<S>()) });
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::Component;
    use crate::query::With;
    use crate::traits::{CanToggle, DontFragment, Singleton, Sparse};

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Tag;

    #[derive(Component, Debug, PartialEq)]
    #[component(Singleton)]
    struct One(u32);

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

    fn values(world: &World, entities: &[Entity]) -> Vec<u32> {
        entities
            .iter()
            .map(|&entity| world.get::<Count>(entity).unwrap().0)
            .collect()
    }

    #[test]
    fn runs_over_its_querys_rows_each_time() {
        fn add_ten(Run { iter: rows, .. }: Run<&mut Count>) {
            for count in rows {
                count.0 += 10;
            }
        }

        let mut world = World::new();
        let entities = counts(&mut world, 0..3);
        world.set(entities[2], Tag);

        let all = world.register_system(add_ten);
        let tagged = world.register_system(|Run { iter: rows, .. }: Run<&mut Count, With<Tag>>| {
            for count in rows {
                count.0 += 100;
            }
        });
        world.run_system(all);
        world.run_system(tagged);
        world.run_system(all);
        assert_eq!(values(&world, &entities), [20, 21, 122]);
    }

    #[test]
    fn runs_with_no_rows_and_sees_tables_made_after_it() {
        let rows_seen = Arc::new(Mutex::new(Vec::new()));
        let mut world = World::new();
        let system = world.register_system({
            let rows_seen = rows_seen.clone();
            move |Run { iter: rows, .. }: Run<&Count>| rows_seen.lock().unwrap().push(rows.count())
        });

        world.run_system(system);
        let entities = counts(&mut world, 0..2);
        world.run_system(system);
        world.set(entities[1], Tag);
        world.run_system(system);
        assert_eq!(*rows_seen.lock().unwrap(), [0, 2, 2]);
    }

    #[test]
    fn keeps_its_captures_between_runs() {
        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let mut runs = 0;
        let system = world.register_system(move |Run { iter: mut rows, .. }: Run<&mut Count>| {
            runs += 1;
            rows.fetch_next().unwrap().0 = runs;
        });
        for _ in 0..3 {
            world.run_system(system);
        }
        assert_eq!(values(&world, &entities), [3]);
    }

    #[test]
    fn leaving_early_finishes_the_iterator() {
        let mut world = World::new();
        let entities = counts(&mut world, 0..5);
        for &entity in &entities[..2] {
            world.set(entity, Tag);
        }

        let ignores = world.register_system(|_: Run<&Count>| {});
        let reads_one = world.register_system(|Run { iter: mut rows, .. }: Run<&Count>| {
            assert!(rows.fetch_next().is_some());
        });
        let breaks = world.register_system(|Run { iter: rows, .. }: Run<&Count>| {
            for _count in rows {
                break;
            }
        });
        for system in [ignores, reads_one, breaks] {
            world.run_system(system);
            world.run_system(system);
        }

        let rows_seen = Arc::new(AtomicUsize::new(0));
        let counts_rows = world.register_system({
            let rows_seen = rows_seen.clone();
            move |Run { iter: rows, .. }: Run<&Count>| rows_seen.store(rows.count(), Relaxed)
        });
        world.run_system(counts_rows);
        assert_eq!(rows_seen.load(Relaxed), 5);
        // A debug build of flecs aborts when the world is dropped if any were left unfinished.
        drop(world);
    }

    #[test]
    fn fetch_next_can_write_to_a_singleton() {
        let mut world = World::new();
        world.set_singleton(One(0));
        counts(&mut world, 1..=3);

        let sum = world.register_system(|Run { iter: mut rows, .. }: Run<(&Count, &mut One)>| {
            while let Some((count, one)) = rows.fetch_next() {
                one.0 += count.0;
            }
        });
        world.run_system(sum);
        assert_eq!(world.singleton::<One>(), Some(&One(6)));

        // With nothing else to match on, a result has one row and no table.
        let only_one = world.register_system(|Run { iter: mut rows, .. }: Run<&mut One>| {
            rows.fetch_next().unwrap().0 += 1;
            assert!(rows.fetch_next().is_none());
        });
        world.run_system(only_one);
        assert_eq!(world.singleton::<One>(), Some(&One(7)));
    }

    #[test]
    fn reads_rows_however_they_are_stored() {
        #[derive(Component, Debug, PartialEq)]
        #[component(CanToggle)]
        struct Toggled(u32);
        #[derive(Component, Debug, PartialEq)]
        #[component(Sparse)]
        struct OutsideTables(u32);
        #[derive(Component, Debug, PartialEq)]
        #[component(DontFragment)]
        struct Unfragmented(u32);

        let mut world = World::new();
        let entities = counts(&mut world, 0..4);
        for (value, &entity) in (0..).zip(&entities) {
            world.set(entity, Toggled(value));
            world.enable_component::<Toggled>(entity);
            world.set(entity, OutsideTables(value));
            world.set(entity, Unfragmented(value));
        }
        world.disable_component::<Toggled>(entities[1]);

        type Stored<'a> = (Entity, &'a Toggled, &'a OutsideTables, &'a mut Unfragmented);
        let rows_seen = Arc::new(Mutex::new(Vec::new()));
        let system = world.register_system({
            let rows_seen = rows_seen.clone();
            move |Run { iter: rows, .. }: Run<Stored>| {
                for (entity, toggled, outside, unfragmented) in rows {
                    assert_eq!([toggled.0, outside.0], [unfragmented.0; 2]);
                    unfragmented.0 += 10;
                    rows_seen.lock().unwrap().push(entity);
                }
            }
        });
        world.run_system(system);

        let mut rows_seen = rows_seen.lock().unwrap().clone();
        rows_seen.sort_by_key(|&entity| world.get::<Count>(entity).unwrap().0);
        assert_eq!(rows_seen, [entities[0], entities[2], entities[3]]);
        assert_eq!(
            world.get::<Unfragmented>(entities[2]),
            Some(&Unfragmented(12))
        );
        assert_eq!(
            world.get::<Unfragmented>(entities[1]),
            Some(&Unfragmented(1))
        );
    }

    #[test]
    fn drops_each_system_with_the_world() {
        struct CountsDrops(Arc<AtomicUsize>);

        impl Drop for CountsDrops {
            fn drop(&mut self) {
                self.0.fetch_add(1, Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        counts(&mut world, 0..2);
        let systems = [(); 2].map(|()| {
            let captured = CountsDrops(drops.clone());
            world.register_system(move |_: Run<&Count>| {
                let _captured = &captured;
            })
        });
        for system in systems {
            world.run_system(system);
            world.run_system(system);
        }
        assert_eq!(drops.load(Relaxed), 0);
        drop(world);
        assert_eq!(drops.load(Relaxed), 2);
    }

    #[test]
    #[should_panic = "only be run by the world that registered it"]
    fn running_another_worlds_system_panics() {
        let mut first = World::new();
        let system = first.register_system(|_: Run<&Count>| {});
        let mut second = World::new();
        second.run_system(system);
    }

    #[test]
    #[should_panic = "a system can't be deleted"]
    fn deleting_a_system_panics() {
        let mut world = World::new();
        let system = world.register_system(|_: Run<&Count>| {});
        world.delete(system.entity);
    }

    /// A system only has to be `Send`, as only `&mut World` reaches it.
    #[test]
    fn a_world_with_systems_moves_between_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SystemId>();

        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let calls = Cell::new(0);
        let system = world.register_system(move |Run { iter: rows, .. }: Run<&mut Count>| {
            calls.set(calls.get() + 1);
            for count in rows {
                count.0 = calls.get();
            }
        });
        let mut world = std::thread::spawn(move || {
            world.run_system(system);
            world
        })
        .join()
        .unwrap();
        world.run_system(system);
        assert_eq!(values(&world, &entities), [2]);
        std::thread::spawn(move || drop(world)).join().unwrap();
    }
}
