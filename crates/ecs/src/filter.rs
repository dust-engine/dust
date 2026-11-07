//! What a query's rows must have or lack, beyond what they read.

use std::marker::PhantomData;

use crate::Component;
use crate::sys::{EcsAnd, EcsInOutNone, EcsNot, EcsOr, ecs_query_desc_t};
use crate::terms::{Accesses, describe_term};
use crate::traits::{has_dont_fragment, has_singleton};
use crate::world::World;

/// What a query's rows must have or lack, beyond its [`QueryData`](crate::query::QueryData):
/// [`With`], [`Without`], [`Or`], or a tuple of these, which can nest. `()` keeps every row.
///
/// # Safety
///
/// Only `dust_ecs` implements this. `describe` must add the terms that `ACCESS` lists, none of
/// which read or write, as `assert_entities_are_matched` relies on which of them are required.
pub unsafe trait QueryFilter {
    #[doc(hidden)]
    const ACCESS: Accesses;
    /// Fails to compile if this can't be used.
    #[doc(hidden)]
    const CHECK: () = ();

    #[doc(hidden)]
    /// Adds this filter's terms to `desc` from `index` on, registering their components, and
    /// moves `index` past them.
    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize);
}

/// Keeps the rows whose entity has a `T`, without reading it. For a
/// [`Singleton`](crate::traits::Singleton), keeps every row while the world has one.
///
/// This isn't the [`traits::With`](crate::traits::With) trait, which adds a tag to entities.
pub struct With<T>(PhantomData<T>);

/// Keeps the rows whose entity has no `T`. For a [`Singleton`](crate::traits::Singleton), keeps
/// every row while the world has none.
pub struct Without<T>(PhantomData<T>);

/// Keeps the rows whose entity has at least one of the components that `T` lists, as a tuple of
/// [`With`] filters, such as `Or<(With<A>, With<B>)>`.
///
/// Unlike Bevy's, this only holds `With` filters, as flecs can only join single terms with `Or`.
/// It fails to compile for a singleton, as all the terms must be on the row's entity, and for a
/// [`DontFragment`](crate::traits::DontFragment) component, which flecs doesn't support here.
pub struct Or<T>(PhantomData<T>);

// SAFETY: `ACCESS` lists the term, which reads nothing.
unsafe impl<T: Component> QueryFilter for With<T> {
    const ACCESS: Accesses = Accesses::of::<T>(EcsInOutNone, EcsAnd);

    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
        describe_term::<T>(world, desc, index, EcsInOutNone, EcsAnd);
    }
}

// SAFETY: `ACCESS` lists the term, which reads nothing.
unsafe impl<T: Component> QueryFilter for Without<T> {
    const ACCESS: Accesses = Accesses::of::<T>(EcsInOutNone, EcsNot);

    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
        describe_term::<T>(world, desc, index, EcsInOutNone, EcsNot);
    }
}

// SAFETY: this has no terms.
unsafe impl QueryFilter for () {
    const ACCESS: Accesses = Accesses::NONE;

    fn describe(_: &mut World, _: &mut ecs_query_desc_t, _: &mut usize) {}
}

const fn assert_can_be_in_or<T: Component>() {
    assert!(
        !has_singleton(T::TRAITS),
        "an `Or` can't hold a singleton, as flecs needs all its terms to be on the row's entity"
    );
    assert!(
        !has_dont_fragment(T::TRAITS),
        "flecs doesn't support `Or` on a `DontFragment` component"
    );
}

macro_rules! impl_query_filter {
    ($($filter:ident),+) => {
        // SAFETY: `ACCESS` lists each element's terms, which `describe` adds.
        unsafe impl<$($filter: QueryFilter),+> QueryFilter for ($($filter,)+) {
            const ACCESS: Accesses = Accesses::NONE$(.with($filter::ACCESS))+;
            const CHECK: () = {
                $(let () = $filter::CHECK;)+
            };

            fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
                $($filter::describe(world, desc, index);)+
            }
        }

        // SAFETY: `ACCESS` lists the chain's terms, which read nothing. It marks each as required,
        // as every row has at least one of them, which `CHECK` keeps on the row's entity.
        unsafe impl<$($filter: Component),+> QueryFilter for Or<($(With<$filter>,)+)> {
            const ACCESS: Accesses = Accesses::NONE$(.with(With::<$filter>::ACCESS))+;
            const CHECK: () = {
                $(assert_can_be_in_or::<$filter>();)+
            };

            fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
                $(describe_term::<$filter>(world, desc, index, EcsInOutNone, EcsOr);)+
                // flecs ends a chain at its first term that isn't `EcsOr`.
                desc.terms[*index - 1].oper = EcsAnd;
            }
        }
    };
}

impl_query_filter!(A);
impl_query_filter!(A, B);
impl_query_filter!(A, B, C);
impl_query_filter!(A, B, C, D);
impl_query_filter!(A, B, C, D, E);
impl_query_filter!(A, B, C, D, E, F);
impl_query_filter!(A, B, C, D, E, F, G);
impl_query_filter!(A, B, C, D, E, F, G, H);
impl_query_filter!(A, B, C, D, E, F, G, H, I);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J, K);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_query_filter!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entity;
    use crate::traits::{CanToggle, DontFragment, Singleton, Sparse};

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Tag;

    #[derive(Component, Debug, PartialEq)]
    struct Marked;

    #[derive(Component, Debug, PartialEq)]
    #[component(CanToggle)]
    struct Toggled(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(Sparse)]
    struct OutsideTables(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(DontFragment)]
    struct Unfragmented(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(Singleton)]
    struct One(u32);

    /// Entities with `Count(0)` to `Count(count - 1)`.
    fn counts(world: &mut World, count: u32) -> Vec<Entity> {
        (0..count)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Count(value));
                entity
            })
            .collect()
    }

    /// The `Count`s that pass `F`, in order.
    fn kept_by<F: QueryFilter>(world: &mut World) -> Vec<u32> {
        let mut query = world.query_filtered::<&Count, F>();
        let mut kept: Vec<_> = query.iter_mut(world).map(|count| count.0).collect();
        world.return_query(query);
        kept.sort();
        kept
    }

    #[test]
    fn with_and_without_choose_rows() {
        let mut world = World::new();
        let entities = counts(&mut world, 4);
        for value in [0, 2] {
            world.set(entities[value], Tag);
        }
        for value in [2, 3] {
            world.set(entities[value], Marked);
        }

        assert_eq!(kept_by::<()>(&mut world), [0, 1, 2, 3]);
        assert_eq!(kept_by::<With<Tag>>(&mut world), [0, 2]);
        assert_eq!(kept_by::<Without<Tag>>(&mut world), [1, 3]);
        assert_eq!(kept_by::<(With<Tag>, Without<Marked>)>(&mut world), [0]);
        assert_eq!(kept_by::<((With<Tag>,), (With<Marked>,))>(&mut world), [2]);
    }

    /// As in Bevy, a filter lets a query read just the entities.
    #[test]
    fn a_filter_lets_a_query_read_only_entities() {
        let mut world = World::new();
        let tagged = world.new_entity();
        world.set(tagged, Tag);
        let untagged = world.new_entity();
        world.set(untagged, Count(1));

        let mut query = world.query_filtered::<Entity, With<Tag>>();
        let entities: Vec<_> = query.iter_mut(&mut world).collect();
        assert_eq!(entities, [tagged]);
        assert!(query.contains(&mut world, tagged));
        assert!(!query.contains(&mut world, untagged));
        world.return_query(query);
    }

    #[test]
    fn a_disabled_component_counts_as_missing() {
        let mut world = World::new();
        let entities = counts(&mut world, 4);
        for &entity in &entities[..3] {
            world.set(entity, Toggled(0));
            world.enable_component::<Toggled>(entity);
        }
        world.disable_component::<Toggled>(entities[1]);

        assert_eq!(kept_by::<With<Toggled>>(&mut world), [0, 2]);
        assert_eq!(kept_by::<Without<Toggled>>(&mut world), [1, 3]);
    }

    #[test]
    fn filters_see_components_stored_outside_tables() {
        let mut world = World::new();
        let entities = counts(&mut world, 4);
        for value in [0, 2] {
            world.set(entities[value], OutsideTables(0));
        }
        for value in [2, 3] {
            world.set(entities[value], Unfragmented(0));
        }

        assert_eq!(kept_by::<With<OutsideTables>>(&mut world), [0, 2]);
        assert_eq!(kept_by::<Without<OutsideTables>>(&mut world), [1, 3]);
        assert_eq!(kept_by::<With<Unfragmented>>(&mut world), [2, 3]);
        assert_eq!(kept_by::<Without<Unfragmented>>(&mut world), [0, 1]);
    }

    #[test]
    fn or_keeps_rows_with_any_of_its_components() {
        let mut world = World::new();
        let entities = counts(&mut world, 4);
        for value in [0, 2] {
            world.set(entities[value], Tag);
        }
        for value in [2, 3] {
            world.set(entities[value], Marked);
        }

        assert_eq!(
            kept_by::<Or<(With<Tag>, With<Marked>)>>(&mut world),
            [0, 2, 3]
        );
        assert_eq!(kept_by::<Or<(With<Tag>,)>>(&mut world), [0, 2]);
        assert_eq!(
            kept_by::<(Or<(With<Tag>, With<Marked>)>, Without<Tag>)>(&mut world),
            [3]
        );

        let mut query = world.query_filtered::<Entity, Or<(With<Tag>, With<Marked>)>>();
        assert!(query.contains(&mut world, entities[3]));
        assert!(!query.contains(&mut world, entities[1]));
        world.return_query(query);
    }

    #[test]
    fn or_sees_disabled_and_sparse_components() {
        let mut world = World::new();
        let entities = counts(&mut world, 4);
        for &entity in &entities[..2] {
            world.set(entity, Toggled(0));
            world.enable_component::<Toggled>(entity);
        }
        world.disable_component::<Toggled>(entities[1]);
        world.set(entities[2], OutsideTables(0));
        world.set(entities[3], Tag);

        assert_eq!(
            kept_by::<Or<(With<Toggled>, With<Tag>)>>(&mut world),
            [0, 3]
        );
        assert_eq!(
            kept_by::<Or<(With<OutsideTables>, With<Tag>)>>(&mut world),
            [2, 3]
        );
    }

    #[test]
    fn a_singleton_filter_keeps_every_row_or_none() {
        let mut world = World::new();
        counts(&mut world, 2);

        assert_eq!(kept_by::<With<One>>(&mut world), []);
        assert_eq!(kept_by::<Without<One>>(&mut world), [0, 1]);
        world.set_singleton(One(0));
        assert_eq!(kept_by::<With<One>>(&mut world), [0, 1]);
        assert_eq!(kept_by::<Without<One>>(&mut world), []);
    }
}
