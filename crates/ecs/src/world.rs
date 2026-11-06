use std::ffi::CStr;
use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::Entity;
use crate::sys::ecs_world_t;

/// Held across `ecs_init` and `ecs_fini`, which write process-wide state (the OS
/// API table, its init count, addon component ids) without synchronization.
/// Nothing can panic while it's held, so it can't be poisoned.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// A flecs world, destroyed on drop.
pub struct World {
    ptr: NonNull<ecs_world_t>,
}

// SAFETY: flecs has no thread-local state and only ties a world to a thread in
// exclusive-access mode, which nothing here turns on. `INIT_LOCK` covers the
// process-wide state.
unsafe impl Send for World {}

// SAFETY: the `&self` methods only read the world. Any new one needs the same
// check, by reading the flecs code and running it under ThreadSanitizer in debug
// and `NDEBUG` builds: a `const` world parameter isn't proof, as flecs casts it
// away in places.
unsafe impl Sync for World {}

impl World {
    pub fn new() -> Self {
        let ptr = {
            let _guard = INIT_LOCK.lock().unwrap();
            // SAFETY: the lock serializes this with every other init and fini.
            unsafe { crate::sys::ecs_init() }
        };
        Self {
            ptr: NonNull::new(ptr).expect("ecs_init returned null"),
        }
    }

    pub fn new_entity(&mut self) -> Entity {
        // SAFETY: the world is valid and `&mut self` makes this access exclusive.
        let id = unsafe { crate::sys::ecs_new(self.ptr.as_ptr()) };
        Entity(NonZeroU64::new(id).expect("ecs_new returned 0"))
    }

    /// Deletes `entity`. Does nothing if it's already dead.
    pub fn delete(&mut self, entity: Entity) {
        // SAFETY: `ecs_delete` ignores ids that aren't alive. It would corrupt the
        // world if given a built-in, but an `Entity` never names one.
        unsafe { crate::sys::ecs_delete(self.ptr.as_ptr(), entity.id()) }
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        // SAFETY: `ecs_is_alive` accepts any non-zero id.
        unsafe { crate::sys::ecs_is_alive(self.ptr.as_ptr(), entity.id()) }
    }

    /// The name of `entity`, or `None` if it's dead or has no name.
    pub fn name(&self, entity: Entity) -> Option<&CStr> {
        if !self.is_alive(entity) {
            return None;
        }
        // SAFETY: `ecs_get_name` needs a live entity, which was just checked.
        let name = unsafe { crate::sys::ecs_get_name(self.ptr.as_ptr(), entity.id()) };
        // SAFETY: flecs owns the string. Only renaming or deleting the entity frees
        // it, and both need `&mut self`, which this borrow of `self` rules out.
        (!name.is_null()).then(|| unsafe { CStr::from_ptr(name) })
    }

    /// Names `entity`, replacing any name it has. Returns `false`, changing nothing,
    /// if `entity` is dead. Names aren't checked for uniqueness.
    pub fn set_name(&mut self, entity: Entity, name: &CStr) {
        if !self.is_alive(entity) {
            return;
        }
        // SAFETY: `ecs_set_name` needs a live entity, which was just checked, and
        // copies `name`.
        unsafe { crate::sys::ecs_set_name(self.ptr.as_ptr(), entity.id(), name.as_ptr()) };
    }
}

impl Drop for World {
    fn drop(&mut self) {
        let _guard = INIT_LOCK.lock().unwrap();
        // SAFETY: the lock serializes this with every other init and fini, and
        // nothing can use the world after this.
        unsafe { crate::sys::ecs_fini(self.ptr.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_entities() {
        let mut world = World::new();
        let entity = world.new_entity();
        assert_eq!(world.name(entity), None);

        assert!(world.set_name(entity, c"player"));
        assert_eq!(world.name(entity), Some(c"player"));

        assert!(world.set_name(entity, c"player"));
        assert_eq!(world.name(entity), Some(c"player"));

        assert!(world.set_name(entity, c"renamed"));
        assert_eq!(world.name(entity), Some(c"renamed"));
    }

    #[test]
    fn dead_entity_has_no_name_and_cannot_take_one() {
        let mut world = World::new();
        let entity = world.new_entity();
        assert!(world.set_name(entity, c"player"));

        world.delete(entity);
        assert_eq!(world.name(entity), None);
        assert!(!world.set_name(entity, c"player"));
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn reads_from_many_threads() {
        let mut world = World::new();
        let named = world.new_entity();
        assert!(world.set_name(named, c"player"));
        let unnamed = world.new_entity();
        let dead = world.new_entity();
        world.delete(dead);

        let world = &world;
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(move || {
                    for _ in 0..1000 {
                        assert_eq!(world.name(named), Some(c"player"));
                        assert_eq!(world.name(unnamed), None);
                        assert_eq!(world.name(dead), None);
                        assert!(world.is_alive(named));
                        assert!(!world.is_alive(dead));
                    }
                });
            }
        });
    }

    #[test]
    fn moves_between_threads() {
        let mut world = World::new();
        let entity = world.new_entity();
        let world = std::thread::spawn(move || {
            world.delete(entity);
            world
        })
        .join()
        .unwrap();
        assert!(!world.is_alive(entity));
        std::thread::spawn(move || drop(world)).join().unwrap();
    }

    #[test]
    fn links_flecs_4_2_0() {
        let version = unsafe { CStr::from_ptr((*crate::sys::ecs_get_build_info()).version) };
        assert_eq!(version, c"4.2.0");
    }

    #[test]
    fn deletes_entities() {
        let mut world = World::new();
        let entity = world.new_entity();
        assert!(world.is_alive(entity));

        world.delete(entity);
        assert!(!world.is_alive(entity));

        world.delete(entity);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn stale_handle_stays_dead_when_its_id_is_recycled() {
        let mut world = World::new();
        let stale = world.new_entity();
        world.delete(stale);

        let fresh = world.new_entity();
        assert_eq!(stale.id() as u32, fresh.id() as u32);
        assert!(!world.is_alive(stale));
        assert!(world.is_alive(fresh));
    }

    /// `Entity` relies on this to rule out built-ins in another world.
    #[test]
    fn every_world_issues_the_same_ids() {
        assert_eq!(World::new().new_entity(), World::new().new_entity());
    }

    #[test]
    fn creates_worlds_on_many_threads() {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..4 {
                        let mut world = World::new();
                        let entity = world.new_entity();
                        assert!(world.is_alive(entity));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
