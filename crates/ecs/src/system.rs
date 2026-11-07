//! Systems: functions that flecs runs over the rows of a query, by hand like Bevy's one-shot
//! systems, or each frame in a [`Phase`].

use std::any::Any;
use std::cell::RefCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use crate::Entity;
use crate::query::{QueryData, QueryFilter, QueryIter};
use crate::state::describe;
use crate::sys::{
    EcsOnLoad, EcsOnStart, EcsOnStore, EcsOnUpdate, EcsOnValidate, EcsPostLoad, EcsPostUpdate,
    EcsPreStore, EcsPreUpdate, ecs_delete, ecs_entity_t, ecs_iter_t, ecs_progress, ecs_run,
    ecs_system_desc_t, ecs_system_get, ecs_system_init,
};
use crate::world::World;

/// A system that [`World::register_system`] or [`World::add_system`] stored.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SystemId {
    entity: Entity,
    /// [`World::id`] of the world that registered it.
    world: u64,
}

/// When [`World::progress`] runs a system that [`World::add_system`] stored. A frame runs the
/// phases in this order, each phase's systems after all of the phase before.
///
/// These are flecs's phases. What each is for is flecs's convention, which nothing enforces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    /// Only in the first frame, before the other phases. A system added after the first frame
    /// never runs in this phase.
    OnStart,
    /// Loads data into the world, such as input.
    OnLoad,
    /// Processes what was loaded, such as turning key presses into actions.
    PostLoad,
    /// Prepares the frame for the game logic.
    PreUpdate,
    /// The game logic.
    OnUpdate,
    /// Checks what the game logic did, such as for collisions.
    OnValidate,
    /// Corrects what the checks found, such as by moving entities apart.
    PostUpdate,
    /// Prepares the frame for rendering, such as by computing transforms.
    PreStore,
    /// Renders the frame.
    OnStore,
}

impl Phase {
    /// flecs's id for the phase.
    fn id(self) -> ecs_entity_t {
        match self {
            Self::OnStart => EcsOnStart,
            Self::OnLoad => EcsOnLoad,
            Self::PostLoad => EcsPostLoad,
            Self::PreUpdate => EcsPreUpdate,
            Self::OnUpdate => EcsOnUpdate,
            Self::OnValidate => EcsOnValidate,
            Self::PostUpdate => EcsPostUpdate,
            Self::PreStore => EcsPreStore,
            Self::OnStore => EcsOnStore,
        }
    }
}

thread_local! {
    /// A panic from a system that this thread is running, kept while flecs finishes the run or the
    /// frame, as unwinding through flecs would abort, and returning early would leave the world
    /// deferred. While it's kept, systems don't run.
    static PANIC: RefCell<Option<Box<dyn Any + Send>>> = const { RefCell::new(None) };
}

/// What a system gets each time it runs: the rows of a query of `D` whose rows also pass the
/// filter `F`, such as `With<T>`. A system's parameter type states its query, so a closure
/// annotates it, as in `|Run { iter, .. }: Run<&mut Position>|`. The world is deferred while a
/// system runs, so a system gets this rather than the world.
///
/// It has a private field, and more fields may be added, so take it apart with `..`.
pub struct Run<'w, D: QueryData, F: QueryFilter = ()> {
    /// The rows of the system's query.
    pub iter: QueryIter<'w, D>,
    /// The seconds since the last frame, as [`World::progress`] was given or measured them. 0
    /// when [`World::run_system`] runs the system.
    pub delta_time: f32,
    _filter: PhantomData<fn() -> F>,
}

impl World {
    /// Stores `system` to run over the rows of the query its [`Run`] parameter states, and gives
    /// the id that [`Self::run_system`] runs it by. What it captures persists between runs. Fails
    /// to compile if the query's filter has [`DetectChanges`](crate::query::DetectChanges), as a
    /// system can't ask whether its query changed.
    ///
    /// If it panics, the panic reaches whoever ran it once flecs has finished the run, and it stays
    /// registered, with its captures as the panic left them. It's dropped by
    /// [`Self::unregister_system`] or with the world. A panic in its captures' `Drop` aborts the
    /// process, as it runs inside flecs. The world drops it under the same lock as a component's
    /// `Drop`, so its captures mustn't create or drop a world when dropped.
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
    /// fn movement(Run { iter, .. }: Run<(&mut Position, &Velocity), With<Moving>>) {
    ///     for (position, velocity) in iter {
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
    ///
    /// world.unregister_system(system);
    /// ```
    pub fn register_system<D, F, S>(&mut self, system: S) -> SystemId
    where
        D: QueryData,
        F: QueryFilter,
        S: for<'w> FnMut(Run<'w, D, F>) + Send + 'static,
    {
        self.store_system(None, system)
    }

    /// Stores `system` as [`Self::register_system`] does, to also run in `phase` each time
    /// [`Self::progress`] runs a frame. As flecs does, `progress` skips it until a table first
    /// matches its query, and then runs it every frame, even when it has no rows.
    pub fn add_system<D, F, S>(&mut self, phase: Phase, system: S) -> SystemId
    where
        D: QueryData,
        F: QueryFilter,
        S: for<'w> FnMut(Run<'w, D, F>) + Send + 'static,
    {
        self.store_system(Some(phase), system)
    }

    fn store_system<D, F, S>(&mut self, phase: Option<Phase>, system: S) -> SystemId
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
        desc.phase = phase.map_or(0, Phase::id);
        desc.run = Some(run::<D, F, S>);
        desc.run_ctx = Box::into_raw(Box::new(system)).cast();
        desc.run_ctx_free = Some(drop_system::<S>);
        let world = self.ptr.as_ptr();
        // SAFETY: the world is valid, `&mut self` makes this access exclusive, and `describe`
        // registered the query's components. flecs keeps the box, and frees it with `run_ctx_free`.
        let entity = unsafe { ecs_system_init(world, &desc) };
        SystemId {
            entity: Entity(NonZeroU64::new(entity).unwrap()),
            world: self.id,
        }
    }

    /// Runs a frame: each system that [`Self::add_system`] stored, phase by phase. Within a phase,
    /// the order is unspecified, as in Bevy, so order systems by their phases.
    ///
    /// `delta_time` is the seconds since the last frame, which systems get in [`Run::delta_time`].
    /// Pass 0 to have flecs measure it.
    ///
    /// # Panics
    ///
    /// If a system panics, once flecs has finished the frame. The frame's later systems don't run.
    pub fn progress(&mut self, delta_time: f32) {
        // SAFETY: the world is valid, and `&mut self` makes this access exclusive, so the world
        // isn't deferred, nor in a frame, nor being iterated.
        unsafe { ecs_progress(self.ptr.as_ptr(), delta_time) };
        if let Some(payload) = PANIC.take() {
            resume_unwind(payload);
        }
    }

    /// Runs a system that this world registered. If it was unregistered, this logs an error and
    /// does nothing else.
    ///
    /// # Panics
    ///
    /// If another world registered it, or if it panics.
    pub fn run_system(&mut self, id: SystemId) {
        if !self.is_registered(id) {
            tracing::error!("{id:?} can't be run, as it was unregistered");
            return;
        }
        // SAFETY: the system is registered in this world, as just checked, and `&mut self` makes
        // this access exclusive.
        unsafe { ecs_run(self.ptr.as_ptr(), id.entity.id(), 0.0, std::ptr::null_mut()) };
        if let Some(payload) = PANIC.take() {
            resume_unwind(payload);
        }
    }

    /// Removes a system that this world registered, and drops it. If it was already unregistered,
    /// this logs an error and does nothing else.
    ///
    /// # Panics
    ///
    /// If another world registered it.
    pub fn unregister_system(&mut self, id: SystemId) {
        if !self.is_registered(id) {
            tracing::error!("{id:?} can't be unregistered, as it already was");
            return;
        }
        // SAFETY: the entity is alive, as just checked, so it's still the one `store_system` made,
        // not a built-in. Deleting it frees the system's query and calls `drop_system`, and
        // `&mut self` means the system isn't running.
        unsafe { ecs_delete(self.ptr.as_ptr(), id.entity.id()) };
    }

    /// Whether `id`'s system is still registered.
    ///
    /// # Panics
    ///
    /// If another world registered it.
    fn is_registered(&self, id: SystemId) -> bool {
        assert_eq!(
            id.world, self.id,
            "a system can only be used with the world that registered it"
        );
        // A stale id stays dead when its entity is recycled, as the new entity's generation differs.
        // SAFETY: `ecs_system_get` needs a live entity, which `is_alive` checks first.
        self.is_alive(id.entity)
            && !unsafe { ecs_system_get(self.ptr.as_ptr(), id.entity.id()) }.is_null()
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
    let delta_time = it.delta_time;
    // SAFETY: `store_system` gave flecs this callback with a query from `describe::<D, F>`, and
    // nothing has advanced the iterator. Once this returns, flecs only reads its own copy's
    // `interrupted_by`, which is still 0, as the system can't interrupt the run. It doesn't finish
    // its copy, as it only does that for a query whose terms all have no source, and this query's
    // terms all have one, the row's entity or a singleton's. Only `run_system` and `progress` run a
    // system, on the thread that called them, as nothing gives flecs worker threads. Each borrows
    // the world exclusively until the run is over.
    let rows = unsafe { QueryIter::from_iter(it) };
    if PANIC.with_borrow(Option::is_some) {
        // A system panicked earlier in this frame, so as in Bevy, the frame's later systems don't
        // run. Dropping the rows finishes the iterator.
        return;
    }
    let run = Run {
        iter: rows,
        delta_time,
        _filter: PhantomData,
    };
    // The system stays registered after a panic, with its captures as the panic left them, as a
    // value does when a panic is caught anywhere else.
    // SAFETY: `store_system` put an `S` here, which only `drop_system` drops. Nothing else uses it
    // while this runs, as whatever runs it borrows the world exclusively, and the system can't
    // reach the world to run itself again.
    let result = catch_unwind(AssertUnwindSafe(|| unsafe { (*system)(run) }));
    if let Err(payload) = result {
        // Unwinding dropped the rows, which finished the iterator, so flecs can end the run.
        PANIC.set(Some(payload));
    }
}

/// flecs's callback to free a system `S`.
unsafe extern "C" fn drop_system<S>(system: *mut c_void) {
    // SAFETY: flecs calls this once, when it deletes the system, with the box `store_system` made.
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
    use crate::state::tests::errors_logged_by;
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

    struct CountsDrops(Arc<AtomicUsize>);

    impl Drop for CountsDrops {
        fn drop(&mut self) {
            self.0.fetch_add(1, Relaxed);
        }
    }

    #[test]
    fn drops_each_system_with_the_world() {
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
    fn unregistering_drops_the_system() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let captured = CountsDrops(drops.clone());
        let system = world.register_system(move |Run { iter, .. }: Run<&mut Count>| {
            let _captured = &captured;
            for count in iter {
                count.0 += 1;
            }
        });
        world.run_system(system);
        assert_eq!(errors_logged_by(|| world.unregister_system(system)), 0);
        assert_eq!(drops.load(Relaxed), 1);
        assert!(!world.is_alive(system.entity));

        assert_eq!(errors_logged_by(|| world.run_system(system)), 1);
        assert_eq!(errors_logged_by(|| world.unregister_system(system)), 1);
        assert_eq!(values(&world, &entities), [1]);
        drop(world);
        assert_eq!(drops.load(Relaxed), 1);
    }

    #[test]
    fn a_stale_id_stays_unregistered_when_its_entity_is_recycled() {
        let mut world = World::new();
        let stale = world.register_system(|_: Run<&Count>| {});
        world.unregister_system(stale);
        let runs = Arc::new(AtomicUsize::new(0));
        let fresh = world.register_system({
            let runs = runs.clone();
            move |_: Run<&Count>| {
                runs.fetch_add(1, Relaxed);
            }
        });
        assert_eq!(stale.entity.id() as u32, fresh.entity.id() as u32);

        assert_eq!(errors_logged_by(|| world.run_system(stale)), 1);
        assert_eq!(errors_logged_by(|| world.unregister_system(stale)), 1);
        assert_eq!(runs.load(Relaxed), 0);
        assert_eq!(errors_logged_by(|| world.run_system(fresh)), 0);
        assert_eq!(runs.load(Relaxed), 1);
    }

    #[test]
    fn a_panic_reaches_run_system_once_flecs_has_finished_the_run() {
        let mut world = World::new();
        let entities = counts(&mut world, 0..3);
        let mut runs = 0;
        let system = world.register_system(move |Run { iter, .. }: Run<&mut Count>| {
            runs += 1;
            for count in iter {
                // Panicking with a row in hand leaves the iterator in the middle of a result.
                if runs == 1 {
                    panic!("the system panicked");
                }
                count.0 += 10;
            }
        });
        let payload = catch_unwind(AssertUnwindSafe(|| world.run_system(system))).unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"the system panicked"));

        // flecs ended the run, so the world isn't deferred and this applies at once.
        world.set(entities[0], Tag);
        assert!(world.has::<Tag>(entities[0]));

        // The system is still registered, with its captures as the panic left them.
        world.run_system(system);
        assert_eq!(values(&world, &entities), [10, 11, 12]);
        // A debug build of flecs aborts when the world is dropped if an iterator was left
        // unfinished.
        drop(world);
    }

    /// Each `run_system` takes the panic of the run it made, so a panic can pass through systems of
    /// several worlds.
    #[test]
    fn a_panic_passes_through_a_system_that_runs_another_worlds_system() {
        let mut inner = World::new();
        let panics = inner.register_system(|_: Run<&Count>| panic!("the inner system panicked"));
        let mut outer = World::new();
        counts(&mut outer, [0]);
        let runs_inner = outer.register_system(move |_: Run<&Count>| {
            inner.run_system(panics);
        });

        let payload = catch_unwind(AssertUnwindSafe(|| outer.run_system(runs_inner))).unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"the inner system panicked")
        );
        let entity = outer.new_entity();
        outer.set(entity, Tag);
        assert!(outer.has::<Tag>(entity));
        // Dropping `outer` would drop `inner` inside `ecs_fini`, which deadlocks.
        outer.unregister_system(runs_inner);
    }

    #[test]
    #[should_panic = "only be used with the world that registered it"]
    fn running_another_worlds_system_panics() {
        let mut first = World::new();
        let system = first.register_system(|_: Run<&Count>| {});
        let mut second = World::new();
        second.run_system(system);
    }

    #[test]
    #[should_panic = "only be used with the world that registered it"]
    fn unregistering_another_worlds_system_panics() {
        let mut first = World::new();
        let system = first.register_system(|_: Run<&Count>| {});
        let mut second = World::new();
        second.unregister_system(system);
    }

    #[test]
    #[should_panic = "a system is removed by `World::unregister_system`"]
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

    const PHASES: [Phase; 9] = [
        Phase::OnStart,
        Phase::OnLoad,
        Phase::PostLoad,
        Phase::PreUpdate,
        Phase::OnUpdate,
        Phase::OnValidate,
        Phase::PostUpdate,
        Phase::PreStore,
        Phase::OnStore,
    ];

    #[test]
    fn progress_runs_each_phase_in_order() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut world = World::new();
        counts(&mut world, [0]);
        let add = |world: &mut World, phase| {
            let ran = ran.clone();
            world.add_system(phase, move |_: Run<&Count>| ran.lock().unwrap().push(phase));
        };
        // Added last to first, so the order can't come from the order they were added.
        for phase in PHASES.into_iter().rev() {
            add(&mut world, phase);
        }
        world.progress(1.0);
        assert_eq!(*ran.lock().unwrap(), PHASES);

        // `OnStart` only runs in the first frame, even for a system added after it.
        add(&mut world, Phase::OnStart);
        ran.lock().unwrap().clear();
        world.progress(1.0);
        assert_eq!(*ran.lock().unwrap(), PHASES[1..]);
    }

    /// flecs marks a system empty when no table matches its query yet, and stops when one does.
    #[test]
    fn progress_skips_a_system_until_a_table_matches_its_query() {
        let frames = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        world.add_system(Phase::OnUpdate, {
            let frames = frames.clone();
            move |_: Run<&Count>| {
                frames.fetch_add(1, Relaxed);
            }
        });
        world.progress(1.0);
        assert_eq!(frames.load(Relaxed), 0);

        let entities = counts(&mut world, [0]);
        world.progress(1.0);
        assert_eq!(frames.load(Relaxed), 1);

        // flecs doesn't mark it empty again once it has matched, even with no rows left.
        world.delete(entities[0]);
        world.progress(1.0);
        assert_eq!(frames.load(Relaxed), 2);
    }

    #[test]
    fn progress_only_runs_added_systems_and_run_system_runs_both() {
        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let by_hand = world.register_system(|Run { iter, .. }: Run<&mut Count>| {
            for count in iter {
                count.0 += 1;
            }
        });
        let each_frame = world.add_system(Phase::OnUpdate, |Run { iter, .. }: Run<&mut Count>| {
            for count in iter {
                count.0 += 10;
            }
        });
        world.progress(1.0);
        assert_eq!(values(&world, &entities), [10]);
        world.run_system(by_hand);
        world.run_system(each_frame);
        assert_eq!(values(&world, &entities), [21]);
    }

    #[test]
    fn adding_and_unregistering_between_frames_changes_what_runs() {
        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let adds_one = world.add_system(Phase::OnUpdate, |Run { iter, .. }: Run<&mut Count>| {
            for count in iter {
                count.0 += 1;
            }
        });
        world.progress(1.0);
        world.add_system(Phase::OnUpdate, |Run { iter, .. }: Run<&mut Count>| {
            for count in iter {
                count.0 += 10;
            }
        });
        world.progress(1.0);
        world.unregister_system(adds_one);
        world.progress(1.0);
        assert_eq!(values(&world, &entities), [22]);
    }

    #[test]
    fn a_later_phase_sees_what_an_earlier_one_wrote() {
        let mut world = World::new();
        counts(&mut world, [1]);
        let seen = Arc::new(AtomicUsize::new(0));
        world.add_system(Phase::PostUpdate, {
            let seen = seen.clone();
            move |Run { iter, .. }: Run<&Count>| {
                for count in iter {
                    seen.store(count.0 as usize, Relaxed);
                }
            }
        });
        world.add_system(Phase::OnUpdate, |Run { iter, .. }: Run<&mut Count>| {
            for count in iter {
                count.0 *= 10;
            }
        });
        world.progress(1.0);
        assert_eq!(seen.load(Relaxed), 10);
    }

    #[test]
    fn systems_get_the_frames_delta_time() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut world = World::new();
        counts(&mut world, [0]);
        let system = world.add_system(Phase::OnUpdate, {
            let seen = seen.clone();
            move |Run { delta_time, .. }: Run<&Count>| seen.lock().unwrap().push(delta_time)
        });
        world.progress(0.25);
        world.progress(0.5);
        world.run_system(system);
        // flecs measures it when given 0.
        world.progress(0.0);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[..3], [0.25, 0.5, 0.0]);
        assert!(seen[3] > 0.0);
    }

    #[test]
    fn a_panic_in_a_frame_skips_its_later_systems_and_reaches_progress() {
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut world = World::new();
        let entities = counts(&mut world, [0]);
        let mut frames = 0;
        world.add_system(Phase::OnUpdate, {
            let ran = ran.clone();
            move |Run { iter, .. }: Run<&Count>| {
                frames += 1;
                ran.lock().unwrap().push("panics");
                for _count in iter {
                    // Panicking with a row in hand leaves the iterator in the middle of a result.
                    if frames == 1 {
                        panic!("the system panicked");
                    }
                }
            }
        });
        world.add_system(Phase::PostUpdate, {
            let ran = ran.clone();
            move |_: Run<&Count>| ran.lock().unwrap().push("runs after")
        });

        let payload = catch_unwind(AssertUnwindSafe(|| world.progress(1.0))).unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"the system panicked"));
        assert_eq!(*ran.lock().unwrap(), ["panics"]);

        // flecs finished the frame, so the world isn't deferred and this applies at once.
        world.set(entities[0], Tag);
        assert!(world.has::<Tag>(entities[0]));

        world.progress(1.0);
        assert_eq!(*ran.lock().unwrap(), ["panics", "panics", "runs after"]);
        // A debug build of flecs aborts when the world is dropped if an iterator was left
        // unfinished.
        drop(world);
    }
}
