use std::any::type_name;
use std::ffi::{CStr, CString, c_void};
use std::mem::needs_drop;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;

use crate::sys::{ecs_component_desc_t, ecs_type_info_t, ecs_world_t};
use crate::{Component, Entity};

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
// process-wide state, and `Component` makes every stored value `Send`.
unsafe impl Send for World {}

// SAFETY: the `&self` methods only read the world, and `Component` makes every stored
// value `Sync`. Any new `&self` method needs the same check, by reading the flecs code
// and running it under ThreadSanitizer in debug and `NDEBUG` builds: a `const` world
// parameter isn't proof, as flecs casts it away in places.
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

    /// Deletes `entity` and drops its components. Does nothing if it's already dead.
    pub fn delete(&mut self, entity: Entity) {
        if !self.is_alive(entity) {
            return;
        }
        // SAFETY: `ecs_delete` would corrupt the world if given a built-in, but an
        // `Entity` never names one.
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

    /// Names `entity`, replacing any name it has. Does nothing if `entity` is dead.
    /// Names aren't checked for uniqueness.
    pub fn set_name(&mut self, entity: Entity, name: &CStr) {
        if !self.is_alive(entity) {
            return;
        }
        // SAFETY: `ecs_set_name` needs a live entity, which was just checked, and
        // copies `name`.
        unsafe { crate::sys::ecs_set_name(self.ptr.as_ptr(), entity.id(), name.as_ptr()) };
    }

    /// Stores `value` on `entity`, dropping any `T` it already has. If `entity` is dead,
    /// this just drops `value`.
    pub fn set<T: Component>(&mut self, entity: Entity, value: T) {
        if !self.is_alive(entity) {
            return;
        }
        let component = self.register::<T>().id();
        if size_of::<T>() == 0 {
            // SAFETY: the entity is live and `component` is registered. `T` is a tag
            // with nothing to store, and `register` rejects one that has `Drop`.
            unsafe { crate::sys::ecs_add_id(self.ptr.as_ptr(), entity.id(), component) };
            return;
        }
        let mut is_new = false;
        // SAFETY: the entity is live and `component` is registered with this size.
        let slot = unsafe {
            crate::sys::ecs_emplace_id(
                self.ptr.as_ptr(),
                entity.id(),
                component,
                size_of::<T>(),
                &mut is_new,
            )
        }
        .cast::<T>();
        if is_new {
            // SAFETY: `ecs_emplace_id` returned an aligned slot that nothing has
            // constructed, as `register` limits alignment to what flecs's storage has.
            unsafe { slot.write(value) };
        } else {
            // SAFETY: the slot holds the entity's current `T`, which this drops.
            unsafe { *slot = value };
        }
        // SAFETY: the entity is live and now has a `T`.
        unsafe { crate::sys::ecs_modified_id(self.ptr.as_ptr(), entity.id(), component) };
    }

    /// Whether `entity` has a `T`.
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        self.component_id::<T>(entity).is_some_and(|component| {
            // SAFETY: `component_id` checked that the entity is live.
            unsafe { crate::sys::ecs_has_id(self.ptr.as_ptr(), entity.id(), component.id()) }
        })
    }

    /// The `T` on `entity`, or `None` if it has none or is dead.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let component = self.component_id::<T>(entity)?;
        if size_of::<T>() == 0 {
            // flecs has no storage for a tag and `ecs_get_id` assumes there is some.
            // SAFETY: a dangling pointer is valid for a zero-sized type.
            return self.has::<T>(entity).then(|| unsafe { NonNull::dangling().as_ref() });
        }
        // SAFETY: `component_id` checked that the entity is live.
        let value =
            unsafe { crate::sys::ecs_get_id(self.ptr.as_ptr(), entity.id(), component.id()) };
        // SAFETY: if not null, this is the entity's live `T`. Only `&mut self` methods
        // move or drop it, which this borrow of `self` rules out.
        unsafe { value.cast::<T>().as_ref() }
    }

    /// The `T` on `entity`, or `None` if it has none or is dead. The world stays borrowed
    /// until the [`Mut`] is dropped, which tells flecs about any write through it. Fails to
    /// compile for a zero-sized `T`, which has no value to change.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<Mut<'_, T>> {
        const {
            assert!(size_of::<T>() != 0, "a zero-sized component has no value to change");
        }
        let component = self.component_id::<T>(entity)?;
        // SAFETY: `component_id` checked that the entity is live.
        let value =
            unsafe { crate::sys::ecs_get_mut_id(self.ptr.as_ptr(), entity.id(), component.id()) };
        // SAFETY: as in `get`, and this borrow of `self` is exclusive.
        let value = unsafe { value.cast::<T>().as_mut() }?;
        Some(Mut {
            world: self,
            value,
            entity,
            component,
            written: false,
        })
    }

    /// Removes `T` from `entity` and drops it. Does nothing if `entity` is dead or has
    /// no `T`.
    pub fn remove<T: Component>(&mut self, entity: Entity) {
        if let Some(component) = self.component_id::<T>(entity) {
            // SAFETY: `component_id` checked that the entity is live.
            unsafe { crate::sys::ecs_remove_id(self.ptr.as_ptr(), entity.id(), component.id()) };
        }
    }

    /// flecs's id for `T`, if `entity` is live and `T` has been stored in this world.
    /// flecs's own checks for these are compiled out in `-c opt`.
    fn component_id<T: Component>(&self, entity: Entity) -> Option<Entity> {
        if !self.is_alive(entity) {
            return None;
        }
        // Once a world runs out of low ids, its components take the same ids as
        // `new_entity`, so a handle from another world can delete one. Using a dead id
        // would crash flecs, so this treats `T` as unregistered, as flecs's C++ API does.
        // SAFETY: this crate may read the slot. flecs returns 0 for an index this world
        // hasn't set, which includes a slot's initial 0, as its indexes start at 1.
        let id = unsafe {
            crate::sys::flecs_component_ids_get_alive(self.ptr.as_ptr(), T::slot().load(Relaxed))
        };
        NonZeroU64::new(id).map(Entity)
    }

    /// flecs's id for `T`, registering `T` the first time it's stored in this world, and
    /// again if its component was deleted (see `component_id`).
    fn register<T: Component>(&mut self) -> Entity {
        const {
            // flecs allocates columns with plain `malloc` and ignores `alignment`.
            assert!(align_of::<T>() <= 16, "flecs only aligns component storage to 16 bytes");
            assert!(size_of::<T>() <= i32::MAX as usize, "flecs stores sizes as `i32`");
            assert!(
                size_of::<T>() != 0 || !needs_drop::<T>(),
                "flecs stores no value for a zero-sized component, so it can't be dropped"
            );
        }
        let world = self.ptr.as_ptr();
        let index = Self::component_index::<T>();
        // SAFETY: the world is valid, and an index this world hasn't set returns 0.
        let existing = unsafe { crate::sys::flecs_component_ids_get_alive(world, index) };
        if let Some(existing) = NonZeroU64::new(existing) {
            return Entity(existing);
        }
        // SAFETY: flecs documents all zeroes as the default, and it's valid here.
        let mut desc: ecs_component_desc_t = unsafe { std::mem::zeroed() };
        // A label for flecs's errors and explorer, which flecs copies. It needn't be unique:
        // same-named components stay distinct, and the newest just wins lookups by name.
        let name = CString::new(type_name::<T>()).expect("type names have no NUL");
        desc.type_.name = name.as_ptr();
        // flecs requires a size and alignment of 0 for a tag.
        if size_of::<T>() != 0 {
            desc.type_.size = size_of::<T>() as i32;
            desc.type_.alignment = align_of::<T>() as i32;
            // Zeroed or copied bytes needn't be a valid `T`, so flecs must abort if it ever
            // tries to make one that way, as it does for a C++ type that can't do either.
            desc.type_.hooks.flags = crate::sys::ECS_TYPE_HOOK_CTOR_ILLEGAL
                | crate::sys::ECS_TYPE_HOOK_COPY_ILLEGAL
                | crate::sys::ECS_TYPE_HOOK_COPY_CTOR_ILLEGAL;
            if needs_drop::<T>() {
                /// flecs's `dtor` hook for `T`.
                unsafe extern "C" fn drop_components<T>(
                    ptr: *mut c_void,
                    count: i32,
                    _type_info: *const ecs_type_info_t,
                ) {
                    let values = std::ptr::slice_from_raw_parts_mut(ptr.cast::<T>(), count as usize);
                    // SAFETY: flecs calls this with `count` live values that it no longer uses.
                    unsafe { std::ptr::drop_in_place(values) };
                }
                // Keep this the only hook. With no move hooks, flecs moves a value by
                // copying its bytes and never destroys the source, which is what a
                // Rust move is.
                desc.type_.hooks.dtor = Some(drop_components::<T>);
            }
        }
        // SAFETY: the world is valid and `&mut self` makes this access exclusive.
        unsafe {
            let id = crate::sys::ecs_component_init(world, &desc);
            crate::sys::flecs_component_ids_set(world, index, id);
            Entity(NonZeroU64::new(id).expect("ecs_component_init returned 0"))
        }
    }

    /// `T`'s process-wide index, assigning one the first time any world stores a `T`.
    fn component_index<T: Component>() -> i32 {
        // SAFETY: this crate may use the slot.
        let slot = unsafe { T::slot() };
        let index = slot.load(Relaxed);
        if index != 0 {
            return index;
        }
        // SAFETY: a world exists, so flecs's OS API is set and this increment is atomic.
        let new = unsafe { crate::sys::flecs_component_ids_index_get() };
        // Worlds on other threads can get here too, and all must use the same index. A
        // loser's index just goes unused.
        match slot.compare_exchange(0, new, Relaxed, Relaxed) {
            Ok(_) => new,
            Err(winner) => winner,
        }
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

/// A `T` borrowed from a [`World`] by [`World::get_mut`]. flecs can't see a write through
/// this, as it happens after `get_mut` returns, so dropping this tells flecs if there was
/// one, which fires `OnSet`. `mem::forget` skips that, which is safe but leaves observers
/// unaware of the write.
pub struct Mut<'a, T: Component> {
    world: &'a mut World,
    value: &'a mut T,
    entity: Entity,
    component: Entity,
    written: bool,
}

impl<T: Component> Deref for Mut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: Component> DerefMut for Mut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.written = true;
        self.value
    }
}

impl<T: Component> Drop for Mut<'_, T> {
    fn drop(&mut self) {
        if self.written {
            // SAFETY: `get_mut` found this component on this live entity, and this has
            // borrowed the world exclusively since.
            unsafe {
                crate::sys::ecs_modified_id(
                    self.world.ptr.as_ptr(),
                    self.entity.id(),
                    self.component.id(),
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI32, AtomicUsize};

    use super::*;

    /// Counts its drops. The `Box` makes a double drop or a zeroed value a crash.
    #[derive(Component)]
    struct Tracked {
        value: Box<u32>,
        drops: Arc<AtomicUsize>,
    }

    impl Tracked {
        fn new(value: u32, drops: &Arc<AtomicUsize>) -> Self {
            Self {
                value: Box::new(value),
                drops: drops.clone(),
            }
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Relaxed);
        }
    }

    #[derive(Component, Debug, PartialEq)]
    struct Tag;

    #[derive(Component, Debug, PartialEq)]
    struct Text(String);

    #[derive(Component, Debug, PartialEq)]
    struct Count(u32);

    #[derive(Component, Debug, PartialEq)]
    struct Big(u64);

    #[test]
    fn stores_replaces_and_removes_components() {
        let mut world = World::new();
        let entity = world.new_entity();
        assert!(!world.has::<Text>(entity));
        assert_eq!(world.get::<Text>(entity), None);

        world.set(entity, Text(String::from("a")));
        assert!(world.has::<Text>(entity));
        assert_eq!(world.get::<Text>(entity).unwrap().0, "a");

        world.get_mut::<Text>(entity).unwrap().0.push('b');
        assert_eq!(world.get::<Text>(entity).unwrap().0, "ab");

        world.set(entity, Text(String::from("c")));
        assert_eq!(world.get::<Text>(entity).unwrap().0, "c");

        world.remove::<Text>(entity);
        assert!(!world.has::<Text>(entity));
        assert_eq!(world.get::<Text>(entity), None);
        assert!(world.get_mut::<Text>(entity).is_none());
    }

    #[test]
    fn drops_each_value_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();

        let entity = world.new_entity();
        world.set(entity, Tracked::new(1, &drops));
        world.set(entity, Tracked::new(2, &drops));
        assert_eq!(drops.load(Relaxed), 1);

        world.remove::<Tracked>(entity);
        assert_eq!(drops.load(Relaxed), 2);

        world.set(entity, Tracked::new(3, &drops));
        world.delete(entity);
        assert_eq!(drops.load(Relaxed), 3);

        world.set(entity, Tracked::new(4, &drops));
        assert_eq!(drops.load(Relaxed), 4);

        let entity = world.new_entity();
        world.set(entity, Tracked::new(5, &drops));
        assert_eq!(drops.load(Relaxed), 4);
        drop(world);
        assert_eq!(drops.load(Relaxed), 5);
    }

    #[test]
    fn values_survive_table_moves() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let entities: Vec<_> = (0..1000)
            .map(|value| {
                let entity = world.new_entity();
                world.set(entity, Tracked::new(value, &drops));
                if value % 3 == 0 {
                    world.set(entity, Big(u64::from(value)));
                    world.set(entity, Tag);
                }
                entity
            })
            .collect();
        assert_eq!(drops.load(Relaxed), 0);

        for (value, &entity) in entities.iter().enumerate() {
            if value % 5 == 0 {
                world.delete(entity);
            } else if value % 7 == 0 {
                world.remove::<Big>(entity);
            }
        }
        assert_eq!(drops.load(Relaxed), 200);

        for (value, &entity) in entities.iter().enumerate() {
            if value % 5 != 0 {
                assert_eq!(*world.get::<Tracked>(entity).unwrap().value, value as u32);
                let has_big = value % 3 == 0 && value % 7 != 0;
                let big = world.get::<Big>(entity).map(|big| big.0);
                assert_eq!(big, has_big.then_some(value as u64));
            }
        }
        drop(world);
        assert_eq!(drops.load(Relaxed), 1000);
    }

    #[test]
    fn dead_entities_hold_nothing() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));
        world.set(entity, Tag);
        world.delete(entity);

        assert!(!world.has::<Count>(entity));
        assert_eq!(world.get::<Count>(entity), None);
        assert!(world.get_mut::<Count>(entity).is_none());
        assert_eq!(world.get::<Tag>(entity), None);
        world.remove::<Count>(entity);
        world.set(entity, Count(2));
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn tags_work_without_storage() {
        let mut world = World::new();
        let entity = world.new_entity();
        let other = world.new_entity();
        assert!(!world.has::<Tag>(entity));
        assert_eq!(world.get::<Tag>(entity), None);

        world.set(entity, Tag);
        assert!(world.has::<Tag>(entity));
        assert_eq!(world.get::<Tag>(entity), Some(&Tag));
        assert!(!world.has::<Tag>(other));
        assert_eq!(world.get::<Tag>(other), None);

        world.remove::<Tag>(entity);
        assert!(!world.has::<Tag>(entity));
    }

    #[test]
    fn keeps_16_byte_alignment_as_columns_grow() {
        #[derive(Component)]
        #[repr(align(16))]
        struct Aligned(u8);

        let mut world = World::new();
        for value in 0..1000 {
            let entity = world.new_entity();
            world.set(entity, Aligned(value as u8));
            let stored = world.get::<Aligned>(entity).unwrap();
            assert!(std::ptr::from_ref(stored).is_aligned());
            assert_eq!(stored.0, value as u8);
        }
    }

    #[test]
    fn worlds_track_components_separately() {
        let mut first = World::new();
        let mut second = World::new();
        let in_first = first.new_entity();
        let in_second = second.new_entity();

        // Registered in a different order, so `InBoth` has a different id in each world.
        #[derive(Component, Debug, PartialEq)]
        struct OnlyInSecond(u8);
        #[derive(Component, Debug, PartialEq)]
        struct InBoth(u32);

        first.set(in_first, InBoth(1));
        assert!(!second.has::<InBoth>(in_second));
        assert!(!first.has::<OnlyInSecond>(in_first));

        second.set(in_second, OnlyInSecond(7));
        second.set(in_second, InBoth(2));
        assert_eq!(first.get::<InBoth>(in_first), Some(&InBoth(1)));
        assert_eq!(second.get::<InBoth>(in_second), Some(&InBoth(2)));
        assert_eq!(second.get::<OnlyInSecond>(in_second), Some(&OnlyInSecond(7)));
        assert_eq!(first.get::<OnlyInSecond>(in_first), None);
    }

    /// A world runs out of low ids after 237 components, and later ones take the same
    /// ids as `new_entity`, so a handle from another world can name one.
    fn use_up_low_ids(world: &mut World) {
        struct Filler<const A: usize, const B: usize>;

        // The derive rejects generic types, but each of these impls has its own `static`.
        macro_rules! set_all {
            (@row $world:ident, $entity:ident, $a:literal, ($($b:literal)*)) => {
                $(
                    unsafe impl Component for Filler<$a, $b> {
                        unsafe fn slot() -> &'static AtomicI32 {
                            static SLOT: AtomicI32 = AtomicI32::new(0);
                            &SLOT
                        }
                    }
                    $world.set($entity, Filler::<$a, $b>);
                )*
            };
            ($world:ident, $entity:ident, ($($a:literal)*), $b:tt) => {
                $(set_all!(@row $world, $entity, $a, $b);)*
            };
        }

        let entity = world.new_entity();
        set_all!(
            world,
            entity,
            (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15),
            (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
        );
    }

    /// A handle from another world that names `component`.
    fn handle_naming(component: Entity) -> Entity {
        let mut other = World::new();
        (0..1000)
            .map(|_| other.new_entity())
            .find(|&handle| handle == component)
            .expect("`component` should be past the low ids")
    }

    #[test]
    fn set_and_writes_through_get_mut_fire_on_set() {
        #[derive(Component, Debug, PartialEq)]
        struct Watched(u32);

        static CALLS: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn on_set(_iter: *mut c_void) {
            CALLS.fetch_add(1, Relaxed);
        }

        // The two id ranges take different paths through `ecs_modified_id`.
        for past_low_ids in [false, true] {
            let mut world = World::new();
            if past_low_ids {
                use_up_low_ids(&mut world);
            }
            // Registers `Watched` as `register` does, but with a hook to count.
            let mut desc: ecs_component_desc_t = unsafe { std::mem::zeroed() };
            desc.type_.size = size_of::<Watched>() as i32;
            desc.type_.alignment = align_of::<Watched>() as i32;
            desc.type_.hooks.on_set = on_set as *const c_void;
            unsafe {
                let id = crate::sys::ecs_component_init(world.ptr.as_ptr(), &desc);
                let index = World::component_index::<Watched>();
                crate::sys::flecs_component_ids_set(world.ptr.as_ptr(), index, id);
            }
            CALLS.store(0, Relaxed);

            let entity = world.new_entity();
            world.set(entity, Watched(1));
            assert_eq!(CALLS.load(Relaxed), 1);
            world.set(entity, Watched(2));
            assert_eq!(CALLS.load(Relaxed), 2);

            // Reading through the guard fires nothing.
            assert_eq!(world.get_mut::<Watched>(entity).unwrap().0, 2);
            assert_eq!(CALLS.load(Relaxed), 2);

            // Writes fire it once, when the guard is dropped and they're all done.
            let mut value = world.get_mut::<Watched>(entity).unwrap();
            value.0 = 3;
            value.0 += 1;
            assert_eq!(CALLS.load(Relaxed), 2);
            drop(value);
            assert_eq!(CALLS.load(Relaxed), 3);
            assert_eq!(world.get::<Watched>(entity), Some(&Watched(4)));
        }
    }

    #[test]
    fn writing_through_get_mut_drops_the_old_value() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Tracked::new(1, &drops));

        *world.get_mut::<Tracked>(entity).unwrap() = Tracked::new(2, &drops);
        assert_eq!(drops.load(Relaxed), 1);
        assert_eq!(*world.get::<Tracked>(entity).unwrap().value, 2);

        drop(world);
        assert_eq!(drops.load(Relaxed), 2);
    }

    #[test]
    fn a_deleted_component_registers_again() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        use_up_low_ids(&mut world);
        let entity = world.new_entity();
        world.set(entity, Tracked::new(1, &drops));

        let handle = handle_naming(world.register::<Tracked>());
        world.delete(handle);
        assert!(!world.is_alive(handle));
        assert_eq!(drops.load(Relaxed), 1);
        assert!(!world.has::<Tracked>(entity));
        assert!(world.get::<Tracked>(entity).is_none());
        assert!(world.get_mut::<Tracked>(entity).is_none());
        world.remove::<Tracked>(entity);

        world.set(entity, Tracked::new(2, &drops));
        assert_eq!(*world.get::<Tracked>(entity).unwrap().value, 2);
        assert_eq!(drops.load(Relaxed), 1);
        drop(world);
        assert_eq!(drops.load(Relaxed), 2);
    }

    #[test]
    fn a_deleted_tag_registers_again_after_its_id_is_recycled() {
        let mut world = World::new();
        use_up_low_ids(&mut world);
        let entity = world.new_entity();
        world.set(entity, Tag);

        let handle = handle_naming(world.register::<Tag>());
        world.delete(handle);
        let recycled = world.new_entity();
        assert!(!world.has::<Tag>(entity));
        assert!(!world.has::<Tag>(recycled));
        assert_eq!(world.get::<Tag>(entity), None);

        world.set(entity, Tag);
        assert!(world.has::<Tag>(entity));
        assert!(!world.has::<Tag>(recycled));
    }

    #[test]
    fn deleting_a_component_that_holds_itself_drops_every_value() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        use_up_low_ids(&mut world);
        let entity = world.new_entity();
        world.set(entity, Tracked::new(1, &drops));

        let handle = handle_naming(world.register::<Tracked>());
        world.set(handle, Tracked::new(2, &drops));
        world.delete(handle);
        assert!(!world.is_alive(handle));
        assert_eq!(drops.load(Relaxed), 2);
        assert!(!world.has::<Tracked>(entity));

        world.set(entity, Tracked::new(3, &drops));
        assert_eq!(*world.get::<Tracked>(entity).unwrap().value, 3);
    }

    /// flecs's table has an entry for every index up to the highest set, so a type this
    /// world skipped over reads an empty entry, and one with no index yet reads entry 0.
    #[test]
    fn types_this_world_never_stored_read_as_missing() {
        #[derive(Component, Debug, PartialEq)]
        struct Skipped;
        #[derive(Component, Debug, PartialEq)]
        struct Stored;
        #[derive(Component, Debug, PartialEq)]
        struct NoIndexYet;

        let mut first = World::new();
        let in_first = first.new_entity();
        first.set(in_first, Skipped);

        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Stored);
        assert!(!world.has::<Skipped>(entity));
        assert_eq!(world.get::<Skipped>(entity), None);
        assert!(!world.has::<NoIndexYet>(entity));
        assert_eq!(world.get::<NoIndexYet>(entity), None);
    }

    #[test]
    fn names_components_after_their_types() {
        let mut world = World::new();
        let sized = world.register::<Count>();
        let tag = world.register::<Tag>();
        for (component, expected) in [(sized, type_name::<Count>()), (tag, type_name::<Tag>())] {
            assert_eq!(world.name(component).unwrap().to_str().unwrap(), expected);
        }
    }

    #[test]
    fn entities_can_share_a_components_name() {
        #[derive(Component, Debug, PartialEq)]
        struct Shared(u32);

        let name = CString::new(type_name::<Shared>()).unwrap();
        let mut world = World::new();
        let before = world.new_entity();
        world.set_name(before, &name);
        world.set(before, Shared(1));
        assert_eq!(world.get::<Shared>(before), Some(&Shared(1)));

        let after = world.new_entity();
        world.set_name(after, &name);
        world.delete(before);
        world.delete(after);

        let entity = world.new_entity();
        world.set(entity, Shared(2));
        assert_eq!(world.get::<Shared>(entity), Some(&Shared(2)));
    }

    #[test]
    fn names_entities() {
        let mut world = World::new();
        let entity = world.new_entity();
        assert_eq!(world.name(entity), None);

        world.set_name(entity, c"player");
        assert_eq!(world.name(entity), Some(c"player"));

        world.set_name(entity, c"player");
        assert_eq!(world.name(entity), Some(c"player"));

        world.set_name(entity, c"renamed");
        assert_eq!(world.name(entity), Some(c"renamed"));
    }

    #[test]
    fn dead_entity_has_no_name_and_cannot_take_one() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set_name(entity, c"player");

        world.delete(entity);
        assert_eq!(world.name(entity), None);
        world.set_name(entity, c"player");
        assert_eq!(world.name(entity), None);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn reads_from_many_threads() {
        let mut world = World::new();
        let named = world.new_entity();
        world.set_name(named, c"player");
        world.set(named, Text(String::from("a")));
        world.set(named, Tag);
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
                        assert_eq!(world.get::<Text>(named).unwrap().0, "a");
                        assert_eq!(world.get::<Text>(unnamed), None);
                        assert_eq!(world.get::<Tag>(named), Some(&Tag));
                        assert!(world.has::<Tag>(named));
                        assert!(!world.has::<Tag>(unnamed));
                    }
                });
            }
        });
    }

    #[test]
    fn moves_between_threads() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        let entity = world.new_entity();
        let kept = world.new_entity();
        world.set(kept, Tracked::new(1, &drops));
        let world = std::thread::spawn(move || {
            world.delete(entity);
            world
        })
        .join()
        .unwrap();
        assert!(!world.is_alive(entity));
        assert_eq!(*world.get::<Tracked>(kept).unwrap().value, 1);
        std::thread::spawn(move || drop(world)).join().unwrap();
        assert_eq!(drops.load(Relaxed), 1);
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
        // Only used here, so the threads race to assign its index.
        #[derive(Component, Debug, PartialEq)]
        struct FirstUse(u32);

        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for value in 0..4 {
                        let mut world = World::new();
                        let entity = world.new_entity();
                        assert!(world.is_alive(entity));
                        world.set(entity, FirstUse(value));
                        assert_eq!(world.get::<FirstUse>(entity), Some(&FirstUse(value)));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
