//! A safe wrapper over the flecs ECS. [`sys`] holds the raw C bindings.

use std::num::NonZeroU64;
use std::sync::atomic::AtomicI32;

pub use dust_ecs_macros::Component;

pub mod sys;
pub mod traits;
pub mod world;

extern crate self as dust_ecs;

/// A handle to an entity, which stays safe to use after the entity is deleted:
/// ids carry a generation, so a stale handle is dead even once its id is recycled.
///
/// [`world::World::new_entity`] is the only public way to get one, and `World` otherwise
/// only makes one privately for a component's own entity. So it's never 0 and never names
/// a flecs built-in. That holds even if it's passed to another world, because built-ins
/// have the same ids in every world. It then means whichever entity has that id in that
/// world, which can be a component's own entity, so doing this is a logic error,
/// though a memory-safe one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Entity(NonZeroU64);

impl Entity {
    /// The flecs id, for use with [`sys`].
    pub(crate) fn id(self) -> sys::ecs_entity_t {
        self.0.get()
    }
}

/// A type that can be stored on an entity, implemented with `#[derive(Component)]`.
/// `Send + Sync` because [`world::World`] is.
///
/// The derive only works on the type's own definition and rejects generic types. Wrap
/// a type from another crate, or a concrete instantiation of a generic one, in a newtype.
/// Its flecs [`traits`] are listed by path in `#[component(...)]`.
///
/// Storing one fails to compile if it's aligned above 16 bytes, which flecs's storage
/// can't hold, or if it's zero-sized and has `Drop`, as flecs has no value to drop.
///
/// Its `Drop` runs inside flecs, so a panic there aborts the process. When a world is
/// dropped it also runs under a process-wide lock, so it must not create or drop a
/// world, which would deadlock.
///
/// # Safety
///
/// `slot` must return a `static` that no other type uses, or the two types would read
/// each other's values. This also rules out implementing it generically, as a `static`
/// in a generic impl is shared by every instantiation.
pub unsafe trait Component: Send + Sync + 'static {
    /// This type's index in flecs's per-world table of component ids, or 0 if it has
    /// none yet. A non-generic `static` is one location however many crates use the
    /// type, unlike a generic one, which rustc can emit once per crate.
    ///
    /// # Safety
    ///
    /// Only `dust_ecs` may call this: changing the index would make this type read
    /// another type's values.
    #[doc(hidden)]
    unsafe fn slot() -> &'static AtomicI32;

    /// The traits listed in `#[component(...)]`, which registering this type applies.
    #[doc(hidden)]
    const TRAITS: &'static [traits::Declaration] = &[];
}
