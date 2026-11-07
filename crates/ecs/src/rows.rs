//! Reading a query's rows, one at a time.

use std::marker::PhantomData;

use crate::sys::{ecs_iter_fini, ecs_iter_t, ecs_query_iter, ecs_query_next, ecs_query_t};
use crate::terms::{QueryData, assert_entities_are_matched, assert_rows_can_coexist};
use crate::world::World;

/// A query's rows. [`fetch_next`](Self::fetch_next) reads them for any terms. As an [`Iterator`]
/// they can all exist at once, so its `next` fails to compile for terms where two rows could
/// hold a `&mut` to the same value, which is what a `&mut` to a singleton does.
///
/// This borrows the world exclusively because iterating writes to the world's stage and to the
/// query, even if the query only reads components.
///
/// `mem::forget` skips finishing the flecs iterator, which is safe, but a debug build of flecs
/// then aborts when the world is dropped.
pub struct QueryIter<'w, D: QueryData> {
    it: ecs_iter_t,
    /// flecs finishes the iterator itself when `ecs_query_next` returns false, and crashes if
    /// it's then advanced or finished again.
    finished: bool,
    /// For the iterator's current result, so `None` until it has one.
    fetch: Option<D::Fetch>,
    next_row: usize,
    rows: usize,
    _world: PhantomData<&'w mut World>,
}

impl<'w, D: QueryData> QueryIter<'w, D> {
    /// # Safety
    ///
    /// `query` must be in `world`, with `D::describe`'s terms as its first terms. It must stay
    /// live until this is dropped.
    pub(crate) unsafe fn new(world: &'w mut World, query: *const ecs_query_t) -> Self {
        let () = D::CHECK;
        const { assert_entities_are_matched(&D::ACCESS) };
        Self {
            // SAFETY: the caller upholds this for the query, and the world is borrowed exclusively.
            it: unsafe { ecs_query_iter(world.ptr.as_ptr(), query) },
            finished: false,
            fetch: None,
            next_row: 0,
            rows: 0,
            _world: PhantomData,
        }
    }

    /// The next row, which borrows this, so it's gone before the one after it is read.
    pub fn fetch_next(&mut self) -> Option<D::Item<'_>> {
        while self.next_row == self.rows {
            if self.finished {
                return None;
            }
            // SAFETY: flecs hasn't finished the iterator, and `new`'s caller keeps the query live.
            if !unsafe { ecs_query_next(&mut self.it) } {
                self.finished = true;
                return None;
            }
            self.next_row = 0;
            self.rows = self.it.num_rows();
            // SAFETY: the iterator is on a result, and `new`'s caller upholds the rest.
            self.fetch = Some(unsafe { D::init_fetch(&self.it, &mut 0) });
        }
        let row = self.next_row;
        self.next_row += 1;
        let fetch = self.fetch.as_ref().expect("this row is in a result");
        // SAFETY: the loop stopped on a result, whose fetch this is, with `row` below its rows.
        // `new` compiled, so its checks hold. The row borrows `self`, so it's gone before this can
        // be called again, and the world stays borrowed that long.
        unsafe { Some(D::fetch(fetch, &self.it, row)) }
    }
}

impl<D: QueryData> Drop for QueryIter<'_, D> {
    fn drop(&mut self) {
        if !self.finished {
            // SAFETY: flecs hasn't finished the iterator, and `new`'s caller keeps the query live.
            unsafe { ecs_iter_fini(&mut self.it) };
        }
    }
}

impl<'w, D: QueryData + 'w> Iterator for QueryIter<'w, D> {
    type Item = D::Item<'w>;

    fn next(&mut self) -> Option<D::Item<'w>> {
        const { assert_rows_can_coexist(&D::ACCESS) };
        // SAFETY: this compiled, so rows can coexist. Rows point into the world, not into `self`,
        // so they don't need `self` to stay borrowed, and `this` isn't used after this call. `'w`
        // keeps the world borrowed for as long as a row exists, even once this is dropped.
        let this: &'w mut Self = unsafe { &mut *(self as *mut Self) };
        this.fetch_next()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

    use super::*;
    use crate::traits::{CanToggle, DontFragment, Singleton, Sparse};
    use crate::{Component, Entity};

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Text(String);

    #[derive(Component, Debug, PartialEq)]
    struct Tag;

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

    /// Calls `f` with the rows of a new query.
    fn with_rows<D: QueryData, R>(world: &mut World, f: impl FnOnce(QueryIter<'_, D>) -> R) -> R {
        let mut state = world.query::<D>();
        let output = f(state.iter_mut(world));
        world.return_query(state);
        output
    }

    /// Five rows, in two tables.
    fn five_counts(world: &mut World) -> [Entity; 5] {
        std::array::from_fn(|index| {
            let entity = world.new_entity();
            world.set(entity, Count(index as u32));
            if index % 2 == 0 {
                world.set(entity, Tag);
            }
            entity
        })
    }

    #[test]
    fn fetch_next_reads_each_row_once() {
        let mut world = World::new();
        let entities = five_counts(&mut world);

        let visits = with_rows::<&mut Count, _>(&mut world, |mut rows| {
            let mut visits = 0;
            while let Some(count) = rows.fetch_next() {
                count.0 += 10;
                visits += 1;
            }
            // flecs would crash if these reached it.
            assert!(rows.fetch_next().is_none());
            assert!(rows.next().is_none());
            visits
        });
        assert_eq!(visits, 5);
        for (index, &entity) in entities.iter().enumerate() {
            assert_eq!(world.get::<Count>(entity), Some(&Count(index as u32 + 10)));
        }
    }

    #[test]
    fn iterator_rows_can_exist_together_and_outlive_it() {
        let mut world = World::new();
        let entities = five_counts(&mut world);
        for &entity in &entities {
            world.set(entity, Text(String::from("old")));
        }

        with_rows::<(&mut Text, &Count), _>(&mut world, |rows| {
            // `collect` runs the iterator to its end and drops it.
            let mut all: Vec<_> = rows.collect();
            assert_eq!(all.len(), 5);
            // Assigning drops the old value, which would go wrong if two rows aliased.
            all.reverse();
            for (text, count) in all {
                *text = Text(count.0.to_string());
            }
        });
        for (index, &entity) in entities.iter().enumerate() {
            assert_eq!(world.get::<Text>(entity).unwrap().0, index.to_string());
        }
    }

    #[test]
    fn iterator_works_with_for_and_adapters() {
        let mut world = World::new();
        five_counts(&mut world);

        with_rows::<&mut Count, _>(&mut world, |rows| {
            for count in rows {
                count.0 += 1;
            }
        });
        let sum = with_rows::<(&Count, Option<&Tag>), _>(&mut world, |rows| {
            rows.filter(|(_, tag)| tag.is_some())
                .map(|(count, _)| count.0)
                .sum::<u32>()
        });
        // 1 + 3 + 5, from the rows with a tag.
        assert_eq!(sum, 9);
    }

    #[test]
    fn leaving_early_finishes_the_iterator() {
        let mut world = World::new();
        five_counts(&mut world);

        with_rows::<&Count, _>(&mut world, |_| {});
        with_rows::<&Count, _>(&mut world, |mut rows| {
            assert!(rows.fetch_next().is_some());
        });
        with_rows::<&Count, _>(&mut world, |rows| {
            for _count in rows {
                break;
            }
        });
        let unwound = catch_unwind(AssertUnwindSafe(|| {
            with_rows::<&Count, _>(&mut world, |rows| {
                for _ in rows {
                    resume_unwind(Box::new(()));
                }
            });
        }));
        assert!(unwound.is_err());

        // A debug build of flecs aborts when the world is dropped if any were left unfinished.
        let rows = with_rows::<&Count, _>(&mut world, |rows| rows.count());
        assert_eq!(rows, 5);
    }

    #[test]
    fn fetch_next_can_write_to_a_singleton_on_each_row() {
        let mut world = World::new();
        world.set_singleton(One(0));
        five_counts(&mut world);

        with_rows::<(&Count, &mut One), _>(&mut world, |mut rows| {
            while let Some((count, one)) = rows.fetch_next() {
                one.0 += count.0;
            }
        });
        with_rows::<(&Count, Option<&mut One>), _>(&mut world, |mut rows| {
            while let Some((_, one)) = rows.fetch_next() {
                one.unwrap().0 += 1;
            }
        });
        // 0 + 1 + 2 + 3 + 4, then 1 for each row.
        assert_eq!(world.singleton::<One>(), Some(&One(15)));

        // Reading one needs no such care.
        with_rows::<(&Count, &One), _>(&mut world, |rows| {
            let all: Vec<_> = rows.collect();
            assert_eq!(all.len(), 5);
            assert!(all.iter().all(|(_, one)| std::ptr::eq(*one, all[0].1)));
        });
    }

    #[test]
    fn a_query_of_only_singletons_has_one_row() {
        let mut world = World::new();
        let rows = with_rows::<&One, _>(&mut world, |rows| rows.count());
        assert_eq!(rows, 0);

        world.set_singleton(One(7));
        with_rows::<&mut One, _>(&mut world, |mut rows| {
            let one = rows.fetch_next().unwrap();
            one.0 += 1;
            assert!(rows.fetch_next().is_none());
        });
        assert_eq!(world.singleton::<One>(), Some(&One(8)));
    }

    #[test]
    fn each_kind_of_storage_can_be_collected() {
        let mut world = World::new();
        let entities = five_counts(&mut world);
        for (index, &entity) in entities.iter().enumerate() {
            let value = index as u32;
            world.set(entity, Toggled(value));
            world.enable_component::<Toggled>(entity);
            world.set(entity, OutsideTables(value));
            if index != 2 {
                world.set(entity, Unfragmented(value));
            }
        }
        world.disable_component::<Toggled>(entities[1]);

        with_rows::<(&mut Toggled, &mut OutsideTables, &mut Unfragmented), _>(&mut world, |rows| {
            let all: Vec<_> = rows.collect();
            // Row 1 is disabled, and row 2 has no `Unfragmented`.
            assert_eq!(all.len(), 3);
            for (toggled, outside, unfragmented) in all {
                toggled.0 += 10;
                outside.0 += 20;
                unfragmented.0 += 30;
            }
        });
        let value = |world: &World, index: usize| {
            let entity = entities[index];
            (
                world.get::<Toggled>(entity).unwrap().0,
                world.get::<OutsideTables>(entity).unwrap().0,
                world.get::<Unfragmented>(entity).map(|value| value.0),
            )
        };
        assert_eq!(value(&world, 0), (10, 20, Some(30)));
        assert_eq!(value(&world, 1), (1, 1, Some(1)));
        assert_eq!(value(&world, 2), (2, 2, None));
        assert_eq!(value(&world, 3), (13, 23, Some(33)));
        assert_eq!(value(&world, 4), (14, 24, Some(34)));
    }

    /// Reads each row, moving the rows between reads, as an `Iterator` adapter can.
    fn read_while_moving<D: QueryData, T>(
        world: &mut World,
        read: impl for<'r> Fn(D::Item<'r>) -> T,
    ) -> Vec<T> {
        with_rows::<D, _>(world, |mut rows| {
            let mut seen = Vec::new();
            while let Some(value) = rows.fetch_next().map(&read) {
                seen.push(value);
                rows = std::hint::black_box(rows);
            }
            seen
        })
    }

    #[test]
    fn rows_can_be_moved_between_reads() {
        let mut world = World::new();
        let entities = five_counts(&mut world);
        for (index, &entity) in entities.iter().enumerate() {
            world.set(entity, Unfragmented(index as u32));
            world.set(entity, Toggled(index as u32));
            world.enable_component::<Toggled>(entity);
        }
        world.disable_component::<Toggled>(entities[1]);

        let expected: Vec<_> = (0..).zip(entities).collect();
        let sorted = |mut rows: Vec<(u32, Entity)>| {
            rows.sort_by_key(|&(value, _)| value);
            rows
        };

        // A value stored outside tables is looked up through the flecs iterator on each row.
        let rows =
            read_while_moving::<(Entity, &Unfragmented), _>(&mut world, |(entity, value)| {
                (value.0, entity)
            });
        assert_eq!(sorted(rows), expected);
        let rows = read_while_moving::<(Entity, &Unfragmented, &Count), _>(
            &mut world,
            |(entity, value, _)| (value.0, entity),
        );
        assert_eq!(sorted(rows), expected);

        // flecs splits this table into ranges around the disabled row.
        let rows = read_while_moving::<(&Count, &Toggled), _>(&mut world, |(count, _)| count.0);
        assert_eq!(rows.len(), 4);
    }

    /// An [`Entity`] borrows nothing, so rows can give up their entities to use on the world.
    #[test]
    fn entities_from_rows_can_outlive_them() {
        let mut world = World::new();
        five_counts(&mut world);

        let found: Vec<Entity> = with_rows::<(Entity, &Count), _>(&mut world, |rows| {
            rows.filter(|(_, count)| count.0 < 2)
                .map(|(entity, _)| entity)
                .collect()
        });
        assert_eq!(found.len(), 2);
        for entity in found {
            world.delete(entity);
        }
        let rows = with_rows::<&Count, _>(&mut world, |rows| rows.count());
        assert_eq!(rows, 3);
    }
}
