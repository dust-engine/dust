//! What a query keeps between runs.

use std::any::type_name;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::Entity;
use crate::rows::QueryIter;
use crate::sys::{
    EcsQueryCacheAuto, EcsQueryMatchThis, ecs_iter_fini, ecs_iter_t, ecs_query_desc_t,
    ecs_query_fini, ecs_query_has, ecs_query_init, ecs_query_is_true, ecs_query_t,
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
///
/// Its methods panic if given a world other than the one that made it.
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
    pub fn iter_mut<'w>(&mut self, world: &'w mut World) -> QueryIter<'w, D> {
        self.assert_in(world);
        // SAFETY: the query is in this world, as just checked, so it's alive, with the terms that
        // `D::describe` gives. `'w` keeps the world borrowed, so nothing can free the query until
        // this is dropped, even if `self` is dropped first.
        unsafe { QueryIter::new(world, self.query.as_ptr()) }
    }

    /// `entity`'s row, or why it has none.
    pub fn get_mut<'w>(
        &mut self,
        world: &'w mut World,
        entity: Entity,
    ) -> Result<D::Item<'w>, QueryEntityError> {
        let mut it = self.match_entity(world, entity)?;
        // SAFETY: `it` is on a result of this query that holds just `entity`'s row, and `new`
        // compiled, so `D`'s checks hold. The row points into the world, not into `it`, and `'w`
        // keeps the world borrowed for as long as it exists.
        let row = unsafe { D::fetch(&D::init_fetch(&it, &mut 0), &it, 0) };
        // SAFETY: flecs hasn't finished `it`, as it's on a result.
        unsafe { ecs_iter_fini(&mut it) };
        Ok(row)
    }

    /// Whether `entity` has a row.
    pub fn contains(&mut self, world: &mut World, entity: Entity) -> bool {
        let Ok(mut it) = self.match_entity(world, entity) else {
            return false;
        };
        // SAFETY: flecs hasn't finished `it`, as it's on a result.
        unsafe { ecs_iter_fini(&mut it) };
        true
    }

    /// The only row, or why there isn't exactly one.
    pub fn single_mut<'w>(
        &mut self,
        world: &'w mut World,
    ) -> Result<D::Item<'w>, QuerySingleError> {
        self.iter_mut(world).single()
    }

    /// Whether the query has no rows.
    pub fn is_empty(&mut self, world: &mut World) -> bool {
        self.assert_in(world);
        // SAFETY: the query is in this world, as just checked, so it's alive, and `&mut World`
        // makes this access exclusive.
        !unsafe { ecs_query_is_true(self.query.as_ptr()) }
    }

    /// As C++'s `query::has` does, an iterator on the result that holds just `entity`'s row. It
    /// has to be finished.
    fn match_entity(
        &mut self,
        world: &mut World,
        entity: Entity,
    ) -> Result<ecs_iter_t, QueryEntityError> {
        self.assert_in(world);
        if !world.is_alive(entity) {
            return Err(QueryEntityError::EntityDoesNotExist(entity));
        }
        let query = self.query.as_ptr();
        // flecs only checks this in debug builds. A query with no terms on `$this`, such as one
        // of only singletons, has no entity's row.
        // SAFETY: the query is in this world, as just checked, so it's alive.
        if unsafe { (*query).flags } & EcsQueryMatchThis == 0 {
            return Err(QueryEntityError::QueryDoesNotMatch(entity));
        }
        let mut it = MaybeUninit::uninit();
        // SAFETY: the query is alive and has terms on `$this`, the entity is alive, and
        // `&mut World` makes this access exclusive.
        if unsafe { ecs_query_has(query, entity.id(), it.as_mut_ptr()) } {
            // SAFETY: flecs sets the iterator when it returns true.
            Ok(unsafe { it.assume_init() })
        } else {
            Err(QueryEntityError::QueryDoesNotMatch(entity))
        }
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

/// Why [`QueryState::get_mut`] has no row for an entity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryEntityError {
    /// The entity has been deleted.
    EntityDoesNotExist(Entity),
    /// The entity isn't one of the query's rows.
    QueryDoesNotMatch(Entity),
}

impl fmt::Display for QueryEntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntityDoesNotExist(entity) => write!(f, "{entity:?} does not exist"),
            Self::QueryDoesNotMatch(entity) => write!(f, "the query does not match {entity:?}"),
        }
    }
}

impl std::error::Error for QueryEntityError {}

/// Why [`QueryState::single_mut`] doesn't have exactly one row. Each holds the query's
/// [`QueryData`] type, by name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuerySingleError {
    NoEntities(&'static str),
    MultipleEntities(&'static str),
}

impl fmt::Display for QuerySingleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoEntities(query) => write!(f, "no entities fit the query {query}"),
            Self::MultipleEntities(query) => write!(f, "multiple entities fit the query {query}"),
        }
    }
}

impl std::error::Error for QuerySingleError {}

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
    use crate::Component;
    use crate::traits::{CanToggle, DontFragment, Singleton, Sparse};

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Tag;

    /// flecs doesn't cache a query that only reads this.
    #[derive(Component, Debug, PartialEq)]
    #[component(Sparse)]
    struct OutsideTables(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(CanToggle)]
    struct Toggled(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(DontFragment)]
    struct Unfragmented(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(Singleton)]
    struct One(u32);

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

    #[test]
    fn get_mut_reads_and_writes_one_entity() {
        let mut world = World::new();
        let entities: Vec<_> = (0..3)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Count(value));
                entity
            })
            .collect();
        world.set(entities[2], Tag);
        let bare = world.new_entity();
        let deleted = world.new_entity();
        world.delete(deleted);

        let mut query = world.query::<(&mut Count, Option<&Tag>)>();
        let (count, tag) = query.get_mut(&mut world, entities[1]).unwrap();
        assert_eq!(tag, None);
        count.0 += 10;
        let counts: Vec<_> = entities
            .iter()
            .map(|&entity| world.get::<Count>(entity).unwrap().0)
            .collect();
        assert_eq!(counts, [0, 11, 2]);
        assert_eq!(
            query.get_mut(&mut world, entities[2]).unwrap().1,
            Some(&Tag)
        );

        assert_eq!(
            query.get_mut(&mut world, bare).err(),
            Some(QueryEntityError::QueryDoesNotMatch(bare))
        );
        assert_eq!(
            query.get_mut(&mut world, deleted).err(),
            Some(QueryEntityError::EntityDoesNotExist(deleted))
        );
        world.return_query(query);
    }

    #[test]
    fn get_mut_finds_values_however_they_are_stored() {
        let mut world = World::new();
        let entities: Vec<_> = (0..3)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Toggled(value));
                world.enable_component::<Toggled>(entity);
                world.set(entity, Unfragmented(value));
                entity
            })
            .collect();
        world.disable_component::<Toggled>(entities[1]);

        let mut toggled = world.query::<(Entity, &Toggled)>();
        assert_eq!(
            toggled.get_mut(&mut world, entities[2]).unwrap(),
            (entities[2], &Toggled(2))
        );
        assert_eq!(
            toggled.get_mut(&mut world, entities[1]).err(),
            Some(QueryEntityError::QueryDoesNotMatch(entities[1]))
        );
        world.return_query(toggled);

        // With nothing else to match on, a result has rows but no table.
        let mut unfragmented = world.query::<(Entity, &mut Unfragmented)>();
        let (entity, value) = unfragmented.get_mut(&mut world, entities[2]).unwrap();
        assert_eq!(entity, entities[2]);
        value.0 += 10;
        assert_eq!(
            world.get::<Unfragmented>(entities[2]),
            Some(&Unfragmented(12))
        );
        world.return_query(unfragmented);
    }

    #[test]
    fn contains_says_whether_an_entity_has_a_row() {
        let mut world = World::new();
        let tagged = world.new_entity();
        world.set(tagged, Count(1));
        world.set(tagged, Tag);
        let untagged = world.new_entity();
        world.set(untagged, Count(2));
        let deleted = world.new_entity();
        world.delete(deleted);

        let mut query = world.query::<(&Count, &Tag)>();
        assert!(query.contains(&mut world, tagged));
        assert!(!query.contains(&mut world, untagged));
        assert!(!query.contains(&mut world, deleted));
        world.return_query(query);
    }

    #[test]
    fn single_mut_needs_exactly_one_row() {
        let mut world = World::new();
        world.set_singleton(One(0));
        let mut query = world.query::<(&Count, &mut One)>();
        assert!(matches!(
            query.single_mut(&mut world),
            Err(QuerySingleError::NoEntities(_))
        ));

        let first = world.new_entity();
        world.set(first, Count(5));
        let (count, one) = query.single_mut(&mut world).unwrap();
        one.0 += count.0;
        assert_eq!(world.singleton::<One>(), Some(&One(5)));

        let second = world.new_entity();
        world.set(second, Count(6));
        assert!(matches!(
            query.single_mut(&mut world),
            Err(QuerySingleError::MultipleEntities(_))
        ));
        world.return_query(query);
    }

    #[test]
    fn is_empty_says_whether_there_are_rows() {
        let mut world = World::new();
        let mut query = world.query::<&Count>();
        assert!(query.is_empty(&mut world));
        let entity = world.new_entity();
        world.set(entity, Count(1));
        assert!(!query.is_empty(&mut world));
        world.return_query(query);
    }

    #[test]
    fn a_query_of_only_singletons_has_no_entitys_row() {
        let mut world = World::new();
        world.set_singleton(One(1));
        let entity = world.new_entity();

        let mut query = world.query::<&One>();
        assert!(!query.is_empty(&mut world));
        assert_eq!(query.single_mut(&mut world).ok(), Some(&One(1)));
        assert!(!query.contains(&mut world, entity));
        assert_eq!(
            query.get_mut(&mut world, entity).err(),
            Some(QueryEntityError::QueryDoesNotMatch(entity))
        );
        world.return_query(query);
    }
}
