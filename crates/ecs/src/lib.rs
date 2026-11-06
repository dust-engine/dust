//! A safe wrapper over the flecs ECS. [`sys`] holds the raw C bindings.

use std::num::NonZeroU64;

pub mod sys;
pub mod world;

/// A handle to an entity, which stays safe to use after the entity is deleted:
/// ids carry a generation, so a stale handle is dead even once its id is recycled.
///
/// Only [`world::World::new_entity`] makes one, so it's never 0 and never names a flecs
/// built-in. That holds even if it's passed to another world, because every world
/// issues the same ids. It then means whichever entity has that id in that world,
/// so doing this is a logic error, though a memory-safe one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Entity(NonZeroU64);

impl Entity {
    /// The flecs id, for use with [`sys`].
    pub(crate) fn id(self) -> sys::ecs_entity_t {
        self.0.get()
    }
}
