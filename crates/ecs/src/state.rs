//! What a query keeps between runs.

use std::any::type_name;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::rows::QueryIter;
use crate::sys::{
    EcsQueryCacheAuto, ecs_query_desc_t, ecs_query_fini, ecs_query_init, ecs_query_t,
};
use crate::terms::{QueryData, assert_entities_are_matched};
use crate::world::World;

/// A query, and what flecs keeps for it, such as which tables match. flecs updates that as tables
/// come and go. Making one walks the matching tables, so keep it to run a query more than once.
///
/// The world holds the query itself, and freeing it takes exclusive access to the world, so give
/// this back with [`World::return_query`]. Dropping this instead logs an error and leaves the
/// query in the world until the world is dropped. That's safe on any thread, even if the world
/// has gone.
#[must_use = "give this back with `World::return_query`"]
pub struct QueryState<D: QueryData> {
    /// Valid until this is returned or its world is dropped, as [`World::delete`] refuses a
    /// query's entity.
    query: NonNull<ecs_query_t>,
    /// [`World::id`] of the world this is in.
    world: u64,
    _data: PhantomData<fn() -> D>,
}

// SAFETY: this only uses `query` with `&mut self` and the `&mut World` that holds it, on whichever
// thread has those, and flecs only ties a world to a thread in a mode that nothing here turns on.
unsafe impl<D: QueryData> Send for QueryState<D> {}

// SAFETY: as above, and `&self` gives no access to `query`.
unsafe impl<D: QueryData> Sync for QueryState<D> {}

impl<D: QueryData> QueryState<D> {
    fn new(world: &mut World) -> Self {
        let () = D::CHECK;
        const { assert_entities_are_matched(&D::ACCESS) };
        // SAFETY: flecs documents all zeroes as the default, and it's valid here.
        let mut desc: ecs_query_desc_t = unsafe { std::mem::zeroed() };
        D::describe(world, &mut desc, &mut 0);
        // flecs caches what it can, and leaves the query uncached if that's nothing.
        desc.cache_kind = EcsQueryCacheAuto;
        // SAFETY: the world is valid, `&mut` makes this access exclusive, and `describe`
        // registered the terms' components in it.
        let query = unsafe { ecs_query_init(world.ptr.as_ptr(), &desc) };
        Self {
            query: NonNull::new(query).expect("flecs refused the query"),
            world: world.id,
            _data: PhantomData,
        }
    }

    /// The query's rows.
    ///
    /// # Panics
    ///
    /// If `world` didn't make this.
    pub fn iter_mut<'w>(&mut self, world: &'w mut World) -> QueryIter<'w, D> {
        self.assert_in(world);
        // SAFETY: the query is in this world, as just checked, so it's alive, with the terms that
        // `D::describe` gives. `'w` keeps the world borrowed, so nothing can free the query until
        // this is dropped, even if `self` is dropped first.
        unsafe { QueryIter::new(world, self.query.as_ptr()) }
    }

    fn assert_in(&self, world: &World) {
        assert_eq!(
            self.world, world.id,
            "a query can only be used with the world that made it"
        );
    }
}

impl<D: QueryData> Drop for QueryState<D> {
    fn drop(&mut self) {
        tracing::error!(
            "`{}` was dropped instead of given to `World::return_query`, so its world keeps the \
             query until the world is dropped",
            type_name::<Self>()
        );
    }
}

impl World {
    /// Makes a query, which runs on this world alone. Give it back with [`Self::return_query`].
    ///
    /// ```
    /// use dust_ecs::Component;
    /// use dust_ecs::world::World;
    ///
    /// #[derive(Component)]
    /// struct Position(f32);
    ///
    /// #[derive(Component)]
    /// struct Velocity(f32);
    ///
    /// let mut world = World::new();
    /// let entity = world.new_entity();
    /// world.set(entity, Position(1.0));
    /// world.set(entity, Velocity(2.0));
    ///
    /// let mut query = world.query::<(&mut Position, &Velocity)>();
    /// for (position, velocity) in query.iter_mut(&mut world) {
    ///     position.0 += velocity.0;
    /// }
    /// world.return_query(query);
    ///
    /// assert_eq!(world.get::<Position>(entity).unwrap().0, 3.0);
    /// ```
    pub fn query<D: QueryData>(&mut self) -> QueryState<D> {
        QueryState::new(self)
    }

    /// Frees a query that [`Self::query`] made.
    ///
    /// # Panics
    ///
    /// If this world didn't make it.
    pub fn return_query<D: QueryData>(&mut self, state: QueryState<D>) {
        state.assert_in(self);
        // SAFETY: the query is in this world, as just checked, so it's alive, and `&mut self` means
        // nothing is iterating it. This takes the state, so nothing can use it again.
        unsafe { ecs_query_fini(state.query.as_ptr()) };
        // Nothing in it needs dropping, and its `Drop` would report that it wasn't returned.
        std::mem::forget(state);
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;

    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    use super::*;
    use crate::traits::Sparse;
    use crate::{Component, Entity};

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Tag;

    /// flecs doesn't cache a query that only reads this.
    #[derive(Component, Debug, PartialEq)]
    #[component(Sparse)]
    struct OutsideTables(u32);

    /// The entity flecs gave the query, which it only does if it caches the query.
    fn query_entity<D: QueryData>(state: &QueryState<D>) -> Option<Entity> {
        // SAFETY: the query hasn't been returned, and these tests don't use it after its world.
        NonZeroU64::new(unsafe { state.query.as_ref() }.entity).map(Entity)
    }

    struct CountErrors(Arc<AtomicUsize>);

    impl Subscriber for CountErrors {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            if *event.metadata().level() == Level::ERROR {
                self.0.fetch_add(1, Relaxed);
            }
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }

    /// How many errors `f` logs on this thread.
    fn errors_logged_by(f: impl FnOnce()) -> usize {
        let errors = Arc::new(AtomicUsize::new(0));
        tracing::subscriber::with_default(CountErrors(errors.clone()), f);
        errors.load(Relaxed)
    }

    #[test]
    fn a_state_sees_tables_made_after_it() {
        let mut world = World::new();
        let first = world.new_entity();
        world.set(first, Count(1));

        let mut state = world.query::<&mut Count>();
        assert_eq!(state.iter_mut(&mut world).count(), 1);

        let second = world.new_entity();
        world.set(second, Count(2));
        world.set(second, Tag);
        for count in state.iter_mut(&mut world) {
            count.0 += 10;
        }
        assert_eq!(world.get::<Count>(first), Some(&Count(11)));
        assert_eq!(world.get::<Count>(second), Some(&Count(12)));
        world.return_query(state);
    }

    #[test]
    fn returning_a_query_frees_it_and_logs_nothing() {
        let mut world = World::new();
        let state = world.query::<&Count>();
        let entity = query_entity(&state).unwrap();
        assert_eq!(errors_logged_by(|| world.return_query(state)), 0);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn a_query_can_be_made_and_returned_each_time() {
        fn check<D: QueryData>() {
            let mut world = World::new();
            let entity = world.new_entity();
            world.set(entity, Count(0));
            world.set(entity, OutsideTables(0));
            for _ in 0..100 {
                let mut state = world.query::<D>();
                let mut rows = state.iter_mut(&mut world);
                assert!(rows.fetch_next().is_some());
                drop(rows);
                world.return_query(state);
            }
        }
        check::<&Count>();
        check::<&OutsideTables>();
    }

    #[test]
    fn dropping_a_state_logs_an_error_and_leaves_its_query_to_the_world() {
        let mut world = World::new();
        let state = world.query::<&Count>();
        let entity = query_entity(&state).unwrap();
        assert_eq!(errors_logged_by(|| drop(state)), 1);
        assert!(world.is_alive(entity));
    }

    #[test]
    fn a_state_can_outlive_its_world() {
        let mut world = World::new();
        let state = world.query::<&Count>();
        drop(world);
        assert_eq!(errors_logged_by(|| drop(state)), 1);
    }

    #[test]
    #[should_panic = "only be used with the world that made it"]
    fn running_on_another_world_panics() {
        let mut first = World::new();
        let mut state = first.query::<&Count>();
        let mut second = World::new();
        state.iter_mut(&mut second);
    }

    #[test]
    #[should_panic = "only be used with the world that made it"]
    fn returning_to_another_world_panics() {
        let mut first = World::new();
        let state = first.query::<&Count>();
        let mut second = World::new();
        second.return_query(state);
    }

    #[test]
    #[should_panic = "a query is freed by `World::return_query`"]
    fn deleting_a_query_panics() {
        let mut world = World::new();
        let state = world.query::<&Count>();
        world.delete(query_entity(&state).unwrap());
    }

    #[test]
    fn rows_can_outlive_their_state() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));

        let mut state = world.query::<&mut Count>();
        let rows = state.iter_mut(&mut world);
        drop(state);
        for count in rows {
            count.0 += 1;
        }
        assert_eq!(world.get::<Count>(entity), Some(&Count(2)));
    }

    #[test]
    fn a_state_can_move_between_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QueryState<&mut Count>>();

        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));

        let mut state = world.query::<&Count>();
        let (mut world, mut state) = std::thread::spawn(move || {
            assert_eq!(state.iter_mut(&mut world).count(), 1);
            (world, state)
        })
        .join()
        .unwrap();
        assert_eq!(state.iter_mut(&mut world).count(), 1);
        world.return_query(state);

        let state = world.query::<&Count>();
        std::thread::spawn(move || drop(state)).join().unwrap();
    }
}
