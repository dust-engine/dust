//! A query's terms, written as the types its rows are read as:
//!
//! | Rust             | flecs C++        | Term                 |
//! |------------------|------------------|----------------------|
//! | `&T`             | `const T`        | `EcsIn`, `EcsAnd`      |
//! | `&mut T`         | `T`              | `EcsInOut`, `EcsAnd`   |
//! | `Option<&T>`     | `const T*`       | `EcsIn`, `EcsOptional` |
//! | `Option<&mut T>` | `T*`             | `EcsInOut`, `EcsOptional` |
//!
//! A tuple of these reads all of them, and tuples can nest. [`Entity`] reads a row's entity. It
//! isn't a term, so it doesn't use up a field.

use std::any::TypeId;
use std::num::NonZeroU64;
use std::ptr::NonNull;

use crate::sys::{
    EcsAnd, EcsIn, EcsInOut, EcsIterComponentInheritance, EcsOptional, FLECS_TERM_COUNT_MAX,
    ecs_base_field_w_size, ecs_field_at_w_size, ecs_field_is_set, ecs_field_stride,
    ecs_field_w_size, ecs_iter_t, ecs_query_desc_t,
};
use crate::traits::{has_dont_fragment, has_singleton};
use crate::world::World;
use crate::{Component, Entity};

/// Where a query's current result keeps a term's values, like `field_ptr` in flecs's C++ API. It
/// doesn't borrow the iterator, so that [`QueryIter`](crate::rows::QueryIter) can hold both,
/// which makes it `unsafe` to read.
pub struct Field<T> {
    /// The first row's value, or `None` if no row has one, as when an optional term is missing.
    /// Unused if `is_row`.
    ptr: Option<NonNull<T>>,
    /// Bytes between rows' values. A term can match a derived component larger than `T`.
    stride: usize,
    index: i8,
    /// Every row has the one value at `ptr`, which is on another entity, as a singleton's is.
    is_ref: bool,
    /// Stored outside tables, so flecs has to look up each row.
    is_row: bool,
}

impl<T: Component> Field<T> {
    /// # Safety
    ///
    /// `it` must be on a result, and field `index` must exist and hold `T`s.
    unsafe fn new(it: &ecs_iter_t, index: i8) -> Self {
        let mut field = Self {
            ptr: None,
            stride: size_of::<T>(),
            index,
            is_ref: false,
            is_row: false,
        };
        // SAFETY: the caller upholds this.
        unsafe {
            if !ecs_field_is_set(it, index) {
                // Asked first, unlike in C++, as flecs crashes looking up a sparse component that
                // no entity has had.
            } else if size_of::<T>() == 0 {
                // flecs has no storage for a tag and `ecs_field_w_size` assumes there is some.
                field.ptr = Some(NonNull::dangling());
            } else if it.ref_fields | it.up_fields != 0
                || it.flags & EcsIterComponentInheritance != 0
            {
                // C++'s `populate`.
                if it.row_fields & (1 << index) != 0 {
                    field.is_row = true;
                    field.is_ref = true;
                } else {
                    field.ptr =
                        NonNull::new(ecs_base_field_w_size(it, size_of::<T>(), index).cast());
                    field.is_ref = *it.sources.add(index as usize) != 0;
                }
                field.stride = ecs_field_stride(it, index);
            } else {
                // C++'s `populate_self`.
                field.ptr = NonNull::new(ecs_field_w_size(it, size_of::<T>(), index).cast());
            }
        }
        field
    }

    /// Where `row`'s value is, or `None` if it has none. The pointer is valid until the world
    /// next changes, even once the iterator moves on or is dropped.
    ///
    /// # Safety
    ///
    /// `it` must still be on the result this came from, and `row` must be below its
    /// [`num_rows`](ecs_iter_t::num_rows).
    unsafe fn get(&self, it: &ecs_iter_t, row: usize) -> Option<NonNull<T>> {
        debug_assert!(row < it.num_rows(), "row {row} is out of range");
        let offset = if self.is_ref { 0 } else { row * self.stride };
        let ptr = if self.is_row {
            // SAFETY: the caller upholds this, and the field is in range and stores `T`s.
            NonNull::new(
                unsafe { ecs_field_at_w_size(it, size_of::<T>(), self.index, row as i32) }.cast(),
            )
        } else {
            self.ptr
        };
        // SAFETY: the column has a value `stride` apart for each of the result's rows, and the
        // caller keeps `row` among them.
        ptr.map(|ptr| unsafe { ptr.byte_add(offset) })
    }
}

/// What a query reads from each row: `&T`, `&mut T`, `Option<&T>`, `Option<&mut T>`, [`Entity`], or
/// a tuple of these, which can nest, up to the 32 terms that flecs allows.
///
/// Reading [`Entity`] fails to compile unless the query also has `&T`, `&mut T`, a
/// [`With<T>`](crate::query::With) filter for a `T` that isn't a singleton, or an
/// [`Or`](crate::query::Or). Without one the query would also visit flecs's own entities, which an
/// [`Entity`] must never name, or would have no entities to read.
///
/// # Safety
///
/// Only `dust_ecs` implements this. `describe` must add the terms that `init_fetch` and `fetch`
/// read, in the same order, and `ACCESS` must list them. `fetch` can only write to a component if
/// `ACCESS` says it does, as `CHECK` relies on that to rule out aliasing, and can only read the
/// row's entity if it says it does, as `assert_entities_are_matched` relies on that.
pub unsafe trait QueryData {
    #[doc(hidden)]
    type Fetch;
    /// What a row is read as.
    type Item<'r>;
    #[doc(hidden)]
    const ACCESS: Accesses;
    /// Fails to compile if this can't be used.
    #[doc(hidden)]
    const CHECK: () = ();

    #[doc(hidden)]
    /// Adds this data's terms to `desc` from `index` on, registering their components, and
    /// moves `index` past them.
    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize);

    #[doc(hidden)]
    /// Reads this data's fields from `index` on, and moves `index` past them.
    ///
    /// # Safety
    ///
    /// `it` must be on a result, of a query where `describe` added these terms at `index`.
    unsafe fn init_fetch(it: &ecs_iter_t, index: &mut i8) -> Self::Fetch;

    #[doc(hidden)]
    /// `row`'s values.
    ///
    /// # Safety
    ///
    /// `fetch` must come from `it`'s current result, and `row` must be below its
    /// [`num_rows`](ecs_iter_t::num_rows). For the whole query, `CHECK` and
    /// `assert_entities_are_matched` must compile. The row must be gone before the world is next
    /// used, and before another row is read unless `assert_rows_can_coexist` compiles for the whole
    /// query, as it can hold a `&mut` to the same singleton.
    unsafe fn fetch<'r>(fetch: &Self::Fetch, it: &ecs_iter_t, row: usize) -> Self::Item<'r>;
}

const fn assert_can_write<T>() {
    assert!(
        size_of::<T>() != 0,
        "a zero-sized component has no value to change"
    );
}

const fn assert_can_be_optional<T: Component>() {
    // flecs accepts this, and then drops the rows that lack the value if others in their
    // table have it.
    assert!(
        !has_dont_fragment(T::TRAITS),
        "flecs doesn't support an optional term for a `DontFragment` component"
    );
}

// SAFETY: this only reads.
unsafe impl<T: Component> QueryData for &T {
    type Fetch = Field<T>;
    type Item<'r> = &'r T;
    const ACCESS: Accesses = Accesses::of::<T>(EcsIn, EcsAnd);

    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
        describe_term::<T>(world, desc, index, EcsIn, EcsAnd);
    }

    unsafe fn init_fetch(it: &ecs_iter_t, index: &mut i8) -> Field<T> {
        // SAFETY: the caller upholds this, and `describe` added a term for `T` here.
        unsafe { next_field(it, index) }
    }

    unsafe fn fetch<'r>(field: &Field<T>, it: &ecs_iter_t, row: usize) -> &'r T {
        // SAFETY: the caller upholds this, and flecs only matches a row with this term's value.
        unsafe { field.get(it, row).unwrap_unchecked().as_ref() }
    }
}

// SAFETY: `ACCESS` says this writes.
unsafe impl<T: Component> QueryData for &mut T {
    type Fetch = Field<T>;
    type Item<'r> = &'r mut T;
    const ACCESS: Accesses = Accesses::of::<T>(EcsInOut, EcsAnd);
    const CHECK: () = assert_can_write::<T>();

    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
        describe_term::<T>(world, desc, index, EcsInOut, EcsAnd);
    }

    unsafe fn init_fetch(it: &ecs_iter_t, index: &mut i8) -> Field<T> {
        // SAFETY: the caller upholds this, and `describe` added a term for `T` here.
        unsafe { next_field(it, index) }
    }

    unsafe fn fetch<'r>(field: &Field<T>, it: &ecs_iter_t, row: usize) -> &'r mut T {
        // SAFETY: the caller upholds this, and flecs only matches a row with this term's value.
        unsafe { field.get(it, row).unwrap_unchecked().as_mut() }
    }
}

// SAFETY: this only reads.
unsafe impl<T: Component> QueryData for Option<&T> {
    type Fetch = Field<T>;
    type Item<'r> = Option<&'r T>;
    const ACCESS: Accesses = Accesses::of::<T>(EcsIn, EcsOptional);
    const CHECK: () = assert_can_be_optional::<T>();

    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
        describe_term::<T>(world, desc, index, EcsIn, EcsOptional);
    }

    unsafe fn init_fetch(it: &ecs_iter_t, index: &mut i8) -> Field<T> {
        // SAFETY: the caller upholds this, and `describe` added a term for `T` here.
        unsafe { next_field(it, index) }
    }

    unsafe fn fetch<'r>(field: &Field<T>, it: &ecs_iter_t, row: usize) -> Option<&'r T> {
        // SAFETY: the caller upholds this.
        unsafe { field.get(it, row).map(|value| value.as_ref()) }
    }
}

// SAFETY: `ACCESS` says this writes.
unsafe impl<T: Component> QueryData for Option<&mut T> {
    type Fetch = Field<T>;
    type Item<'r> = Option<&'r mut T>;
    const ACCESS: Accesses = Accesses::of::<T>(EcsInOut, EcsOptional);
    const CHECK: () = {
        assert_can_write::<T>();
        assert_can_be_optional::<T>();
    };

    fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
        describe_term::<T>(world, desc, index, EcsInOut, EcsOptional);
    }

    unsafe fn init_fetch(it: &ecs_iter_t, index: &mut i8) -> Field<T> {
        // SAFETY: the caller upholds this, and `describe` added a term for `T` here.
        unsafe { next_field(it, index) }
    }

    unsafe fn fetch<'r>(field: &Field<T>, it: &ecs_iter_t, row: usize) -> Option<&'r mut T> {
        // SAFETY: the caller upholds this.
        unsafe { field.get(it, row).map(|mut value| value.as_mut()) }
    }
}

// SAFETY: this has no terms, and `ACCESS` says it reads the row's entity.
unsafe impl QueryData for Entity {
    type Fetch = ();
    type Item<'r> = Entity;
    const ACCESS: Accesses = Accesses {
        entity: true,
        ..Accesses::NONE
    };

    fn describe(_: &mut World, _: &mut ecs_query_desc_t, _: &mut usize) {}

    unsafe fn init_fetch(_: &ecs_iter_t, _: &mut i8) {}

    unsafe fn fetch<'r>(_: &(), it: &ecs_iter_t, row: usize) -> Self::Item<'r> {
        // SAFETY: `assert_entities_are_matched` compiled, so flecs lists this result's entities,
        // which are alive and so not 0. The caller keeps `row` among them.
        Entity(unsafe { NonZeroU64::new_unchecked(*it.entities.add(row)) })
    }
}

/// What the checks on a query need to know about a term.
#[derive(Clone, Copy)]
struct Access {
    component: TypeId,
    writes: bool,
    /// A row has to have this, or another term in its `Or` chain, rather than it being optional.
    required: bool,
    singleton: bool,
}

/// The terms of some [`QueryData`] or filter. This has a fixed size so that consts can add these up
/// through nested tuples, as the length of an array can't depend on a generic type.
#[derive(Clone, Copy)]
pub struct Accesses {
    /// Only those below `len` mean anything.
    terms: [Access; FLECS_TERM_COUNT_MAX],
    len: usize,
    /// Reads the row's [`Entity`], which isn't a term.
    entity: bool,
}

impl Accesses {
    pub(crate) const NONE: Self = Self {
        terms: [Access {
            component: TypeId::of::<()>(),
            writes: false,
            required: false,
            singleton: false,
        }; FLECS_TERM_COUNT_MAX],
        len: 0,
        entity: false,
    };

    pub(crate) const fn of<T: Component>(inout: i16, oper: i16) -> Self {
        let mut accesses = Self::NONE;
        accesses.terms[0] = Access {
            component: TypeId::of::<T>(),
            writes: inout == EcsInOut,
            required: oper == EcsAnd,
            singleton: has_singleton(T::TRAITS),
        };
        accesses.len = 1;
        accesses
    }

    pub(crate) const fn with(mut self, other: Self) -> Self {
        assert!(
            self.len + other.len <= FLECS_TERM_COUNT_MAX,
            "flecs allows a query at most 32 terms"
        );
        let mut i = 0;
        while i < other.len {
            self.terms[self.len] = other.terms[i];
            self.len += 1;
            i += 1;
        }
        self.entity |= other.entity;
        self
    }

    const fn as_slice(&self) -> &[Access] {
        self.terms.split_at(self.len).0
    }
}

/// Two terms for one component get the same pointer from flecs, so a `&mut` for one of them
/// would alias the other.
const fn assert_no_aliasing(terms: &[Access]) {
    let mut a = 0;
    while a < terms.len() {
        let mut b = a + 1;
        while b < terms.len() {
            assert!(
                !(terms[a].component == terms[b].component && (terms[a].writes || terms[b].writes)),
                "a query can't write to a component that another of its terms uses"
            );
            b += 1;
        }
        a += 1;
    }
}

/// `Iterator` lets a query's rows exist at once, so each row's `&mut` must be to a different
/// value. A singleton's is the same value on every row.
pub(crate) const fn assert_rows_can_coexist(accesses: &Accesses) {
    let terms = accesses.as_slice();
    let mut i = 0;
    while i < terms.len() {
        assert!(
            !(terms[i].writes && terms[i].singleton),
            "a `&mut` to a singleton is the same value on every row, so use `fetch_next`"
        );
        i += 1;
    }
}

/// A query with no required term visits every table, so flecs's own entities too, and an
/// [`Entity`] must never name those. A singleton's term doesn't count: with nothing else to match,
/// a query has no entities to read.
///
/// This can't be part of [`QueryData::CHECK`], which a tuple runs on each element, as another
/// element, or the query's filter, can be the required term.
pub(crate) const fn assert_entities_are_matched(accesses: &Accesses) {
    if !accesses.entity {
        return;
    }
    let terms = accesses.as_slice();
    let mut i = 0;
    while i < terms.len() {
        if terms[i].required && !terms[i].singleton {
            return;
        }
        i += 1;
    }
    panic!(
        "reading `Entity` needs a required term, such as `&T` or `With<T>`, that isn't a singleton"
    );
}

pub(crate) fn describe_term<T: Component>(
    world: &mut World,
    desc: &mut ecs_query_desc_t,
    index: &mut usize,
    inout: i16,
    oper: i16,
) {
    let term = &mut desc.terms[*index];
    term.id = world.register::<T>().id();
    term.inout = inout;
    term.oper = oper;
    *index += 1;
}

/// # Safety
///
/// `it` must be on a result, and field `index` must exist and hold `T`s.
unsafe fn next_field<T: Component>(it: &ecs_iter_t, index: &mut i8) -> Field<T> {
    let field = *index;
    *index += 1;
    // SAFETY: the caller upholds this.
    unsafe { Field::new(it, field) }
}

macro_rules! impl_query_data {
    ($($index:tt $data:ident),+) => {
        // SAFETY: `describe`, `init_fetch` and `fetch` all go through the elements in order, as a
        // tuple's expressions run left to right, and `CHECK` rules out aliasing between them.
        unsafe impl<$($data: QueryData),+> QueryData for ($($data,)+) {
            type Fetch = ($($data::Fetch,)+);
            type Item<'r> = ($($data::Item<'r>,)+);
            const ACCESS: Accesses = Accesses::NONE$(.with($data::ACCESS))+;
            const CHECK: () = {
                $(let () = $data::CHECK;)+
                assert_no_aliasing(Self::ACCESS.as_slice());
            };

            fn describe(world: &mut World, desc: &mut ecs_query_desc_t, index: &mut usize) {
                $($data::describe(world, desc, index);)+
            }

            unsafe fn init_fetch(it: &ecs_iter_t, index: &mut i8) -> Self::Fetch {
                // SAFETY: the caller upholds this, and `describe` added the elements in this order.
                unsafe { ($($data::init_fetch(it, index),)+) }
            }

            unsafe fn fetch<'r>(
                fetch: &Self::Fetch,
                it: &ecs_iter_t,
                row: usize,
            ) -> Self::Item<'r> {
                // SAFETY: the caller upholds this, and `init_fetch` kept the elements' order.
                unsafe { ($($data::fetch(&fetch.$index, it, row),)+) }
            }
        }
    };
}

impl_query_data!(0 A);
impl_query_data!(0 A, 1 B);
impl_query_data!(0 A, 1 B, 2 C);
impl_query_data!(0 A, 1 B, 2 C, 3 D);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N);
impl_query_data!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N, 14 O);
impl_query_data!(
    0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H, 8 I, 9 J, 10 K, 11 L, 12 M, 13 N, 14 O, 15 P
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{CanToggle, DontFragment, Singleton, Sparse};

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Big(u64);

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

    #[derive(Component, Debug, PartialEq)]
    #[component(Singleton, Sparse)]
    struct OneOutsideTables(u32);

    /// Visits each row of a new query.
    fn for_each<D: QueryData>(world: &mut World, mut f: impl for<'r> FnMut(D::Item<'r>)) {
        let mut state = world.query::<D>();
        let mut rows = state.iter_mut(world);
        while let Some(row) = rows.fetch_next() {
            f(row);
        }
        drop(rows);
        world.return_query(state);
    }

    #[test]
    fn an_entity_uses_no_field() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));
        world.set(entity, Big(2));
        world.set(entity, Text(String::from("text")));

        let mut visits = 0;
        for_each::<(Entity, &Count, (&Big, Entity), &mut Text)>(
            &mut world,
            |(first, count, (big, second), text)| {
                assert_eq!([first, second], [entity; 2]);
                assert_eq!((count, big), (&Count(1), &Big(2)));
                assert_eq!(text.0, "text");
                visits += 1;
            },
        );
        assert_eq!(visits, 1);
    }

    /// Whatever stores a row's value, the row's entity is the one that has it.
    #[test]
    fn a_row_has_the_entity_its_values_are_on() {
        let mut world = World::new();
        let entities: Vec<_> = (0..6)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Count(value));
                world.set(entity, Toggled(value));
                world.enable_component::<Toggled>(entity);
                world.set(entity, OutsideTables(value));
                world.set(entity, Unfragmented(value));
                if value % 2 == 0 {
                    world.set(entity, Tag);
                }
                entity
            })
            .collect();
        world.disable_component::<Toggled>(entities[0]);
        world.disable_component::<Toggled>(entities[3]);
        world.set_singleton(One(7));

        let expect = |values: &[usize], rows: &mut Vec<(u32, Entity)>| {
            rows.sort_by_key(|&(value, _)| value);
            let expected: Vec<_> = values
                .iter()
                .map(|&value| (value as u32, entities[value]))
                .collect();
            assert_eq!(*rows, expected);
        };
        let all = [0, 1, 2, 3, 4, 5];

        let mut rows = Vec::new();
        for_each::<(Entity, &Count)>(&mut world, |(entity, value)| rows.push((value.0, entity)));
        expect(&all, &mut rows);

        let mut rows = Vec::new();
        for_each::<(Entity, &Toggled)>(&mut world, |(entity, value)| rows.push((value.0, entity)));
        expect(&[1, 2, 4, 5], &mut rows);

        let mut rows = Vec::new();
        for_each::<(Entity, &OutsideTables)>(&mut world, |(entity, value)| {
            rows.push((value.0, entity));
        });
        expect(&all, &mut rows);

        let mut rows = Vec::new();
        for_each::<(Entity, &Unfragmented)>(&mut world, |(entity, value)| {
            rows.push((value.0, entity));
        });
        expect(&all, &mut rows);

        // Terms that aren't about the row's entity don't change what it is.
        let mut rows = Vec::new();
        for_each::<(Entity, &Count, &One, Option<&Tag>)>(
            &mut world,
            |(entity, value, one, tag)| {
                assert_eq!(one, &One(7));
                assert_eq!(tag.is_some(), value.0 % 2 == 0);
                rows.push((value.0, entity));
            },
        );
        expect(&all, &mut rows);

        // A tag has no value to compare, so this asks the world.
        let mut tagged = Vec::new();
        for_each::<(Entity, &Tag)>(&mut world, |(entity, _)| tagged.push(entity));
        let mut rows: Vec<_> = tagged
            .into_iter()
            .map(|entity| (world.get::<Count>(entity).unwrap().0, entity))
            .collect();
        expect(&[0, 2, 4], &mut rows);
    }

    #[test]
    fn a_table_with_no_rows_left_gives_no_entities() {
        let mut world = World::new();
        let kept = world.new_entity();
        world.set(kept, Count(1));
        let deleted = world.new_entity();
        world.set(deleted, Count(2));
        world.set(deleted, Tag);
        world.delete(deleted);

        let mut rows = Vec::new();
        for_each::<(Entity, &Count)>(&mut world, |(entity, _)| rows.push(entity));
        assert_eq!(rows, [kept]);

        let mut rows = 0;
        for_each::<(Entity, &Tag)>(&mut world, |_| rows += 1);
        assert_eq!(rows, 0);
    }

    #[test]
    fn a_term_needs_no_tuple() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));

        for_each::<&mut Count>(&mut world, |count| count.0 += 1);
        assert_eq!(world.get::<Count>(entity), Some(&Count(2)));

        let mut values = Vec::new();
        for_each::<&Count>(&mut world, |count| values.push(count.0));
        assert_eq!(values, [2]);

        // With nothing required, these also visit every entity without a `Count`.
        for_each::<Option<&mut Count>>(&mut world, |count| {
            if let Some(count) = count {
                count.0 += 1;
            }
        });
        assert_eq!(world.get::<Count>(entity), Some(&Count(3)));

        let mut values = Vec::new();
        for_each::<Option<&Count>>(&mut world, |count| {
            values.extend(count.map(|count| count.0));
        });
        assert_eq!(values, [3]);
    }

    #[test]
    fn tuples_can_nest() {
        let mut world = World::new();
        let full = world.new_entity();
        world.set(full, Count(1));
        world.set(full, Big(20));
        world.set(full, Text(String::from("old")));
        world.set(full, Tag);
        let bare = world.new_entity();
        world.set(bare, Count(2));
        world.set(bare, Big(30));
        world.set(bare, Tag);

        let mut rows = Vec::new();
        for_each::<(
            (&mut Count, &Big),
            (Option<&mut Text>, (&Tag,)),
            Option<&Toggled>,
        )>(&mut world, |((count, big), (text, (tag,)), toggled)| {
            assert_eq!(tag, &Tag);
            assert_eq!(toggled, None);
            count.0 += big.0 as u32;
            if let Some(text) = &text {
                assert_eq!(text.0, "old");
            }
            rows.push((count.0, text.is_some()));
        });
        rows.sort();
        assert_eq!(rows, [(21, true), (32, false)]);
    }

    #[test]
    fn writes_reach_the_world() {
        let mut world = World::new();
        let entities: Vec<_> = (0..3)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Count(value));
                world.set(entity, Text(String::from("old")));
                entity
            })
            .collect();

        for_each::<(&mut Text, &Count)>(&mut world, |(text, count)| {
            // This drops the old value, which would go wrong if `Text` was misread.
            *text = Text(count.0.to_string());
        });
        for (value, &entity) in entities.iter().enumerate() {
            assert_eq!(world.get::<Text>(entity).unwrap().0, value.to_string());
        }
    }

    #[test]
    fn optional_terms_are_none_for_rows_without_them() {
        let mut world = World::new();
        let bare = world.new_entity();
        world.set(bare, Count(1));
        let full = world.new_entity();
        world.set(full, Count(2));
        world.set(full, Big(20));
        world.set(full, Text(String::from("old")));
        world.set(full, Tag);

        let mut rows = Vec::new();
        for_each::<(&Count, Option<&Big>, Option<&mut Text>, Option<&Tag>)>(
            &mut world,
            |(count, big, text, tag)| {
                if let Some(text) = text {
                    text.0 = String::from("new");
                }
                rows.push((count.0, big.map(|big| big.0), tag.is_some()));
            },
        );
        rows.sort();
        assert_eq!(rows, [(1, None, false), (2, Some(20), true)]);
        assert_eq!(world.get::<Text>(full).unwrap().0, "new");
    }

    #[test]
    fn a_required_tag_limits_the_rows() {
        let mut world = World::new();
        let tagged = world.new_entity();
        world.set(tagged, Count(1));
        world.set(tagged, Tag);
        let untagged = world.new_entity();
        world.set(untagged, Count(2));

        let mut rows = Vec::new();
        for_each::<(&Count, &Tag)>(&mut world, |(count, tag)| {
            assert_eq!(tag, &Tag);
            rows.push(count.0);
        });
        assert_eq!(rows, [1]);
    }

    #[test]
    fn disabled_components_are_skipped_or_none() {
        let mut world = World::new();
        let entities: Vec<_> = (0..4)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Count(value));
                world.set(entity, Toggled(value));
                world.enable_component::<Toggled>(entity);
                entity
            })
            .collect();
        world.disable_component::<Toggled>(entities[1]);
        world.disable_component::<Toggled>(entities[2]);

        for_each::<&mut Toggled>(&mut world, |toggled| toggled.0 += 10);
        let values: Vec<_> = entities
            .iter()
            .map(|&entity| world.get::<Toggled>(entity).unwrap().0)
            .collect();
        assert_eq!(values, [10, 1, 2, 13]);

        let mut rows = Vec::new();
        for_each::<(&Count, Option<&Toggled>)>(&mut world, |(count, toggled)| {
            rows.push((count.0, toggled.map(|toggled| toggled.0)));
        });
        rows.sort();
        assert_eq!(rows, [(0, Some(10)), (1, None), (2, None), (3, Some(13))]);
    }

    #[test]
    fn values_outside_tables_are_read_and_written() {
        let mut world = World::new();
        let both = world.new_entity();
        world.set(both, Count(1));
        world.set(both, OutsideTables(10));
        world.set(both, Unfragmented(100));
        let also_both = world.new_entity();
        world.set(also_both, Count(2));
        world.set(also_both, OutsideTables(20));
        world.set(also_both, Unfragmented(200));
        let neither = world.new_entity();
        world.set(neither, Count(3));

        for_each::<(&mut OutsideTables, &Count)>(&mut world, |(value, count)| {
            value.0 += count.0;
        });
        assert_eq!(world.get::<OutsideTables>(both), Some(&OutsideTables(11)));
        assert_eq!(
            world.get::<OutsideTables>(also_both),
            Some(&OutsideTables(22))
        );

        let mut rows = Vec::new();
        for_each::<(&Count, Option<&mut OutsideTables>)>(&mut world, |(count, value)| {
            rows.push((count.0, value.map(|value| value.0)));
        });
        rows.sort();
        assert_eq!(rows, [(1, Some(11)), (2, Some(22)), (3, None)]);

        // With nothing else to match on, a result has rows but no table.
        let mut visits = 0;
        for_each::<&mut Unfragmented>(&mut world, |value| {
            value.0 += 1;
            visits += 1;
        });
        assert_eq!(visits, 2);
        assert_eq!(world.get::<Unfragmented>(both), Some(&Unfragmented(101)));
        assert_eq!(
            world.get::<Unfragmented>(also_both),
            Some(&Unfragmented(201))
        );
    }

    /// flecs crashes if asked for a row's value here, as no entity has had this component yet.
    #[test]
    fn an_optional_term_no_entity_has_had_is_none() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));

        let mut rows = Vec::new();
        for_each::<(&Count, Option<&OutsideTables>)>(&mut world, |(count, value)| {
            rows.push((count.0, value.map(|value| value.0)));
        });
        assert_eq!(rows, [(1, None)]);

        let mut rows = Vec::new();
        for_each::<(&Count, Option<&mut OutsideTables>)>(&mut world, |(count, value)| {
            rows.push((count.0, value.map(|value| value.0)));
        });
        assert_eq!(rows, [(1, None)]);
    }

    #[test]
    fn a_singleton_is_the_same_value_on_every_row() {
        let mut world = World::new();
        world.set_singleton(One(0));
        for value in 1..=3 {
            let entity = world.new_entity();
            world.set(entity, Count(value));
        }

        for_each::<(&Count, &mut One)>(&mut world, |(count, one)| one.0 += count.0);
        assert_eq!(world.singleton::<One>(), Some(&One(6)));

        let mut visits = Vec::new();
        for_each::<&One>(&mut world, |one| visits.push(one.0));
        assert_eq!(visits, [6]);

        world.remove_singleton::<One>();
        let mut rows = 0;
        for_each::<&One>(&mut world, |_| rows += 1);
        assert_eq!(rows, 0);
        for_each::<(&Count, Option<&One>)>(&mut world, |(_, one)| {
            assert_eq!(one, None);
            rows += 1;
        });
        assert_eq!(rows, 3);
    }

    #[test]
    fn a_singleton_outside_tables_is_read() {
        let mut world = World::new();
        world.set_singleton(OneOutsideTables(7));
        let entity = world.new_entity();
        world.set(entity, Count(1));

        let mut visits = 0;
        for_each::<(&Count, &OneOutsideTables)>(&mut world, |(_, one)| {
            assert_eq!(one, &OneOutsideTables(7));
            visits += 1;
        });
        assert_eq!(visits, 1);
    }
}
