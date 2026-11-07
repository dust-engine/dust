use std::any::type_name;
use std::ffi::{CStr, CString, c_void};
use std::mem::needs_drop;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use crate::sys::{ecs_component_desc_t, ecs_id_t, ecs_type_info_t, ecs_world_t};
use crate::traits::{CanToggle, Singleton};
use crate::{Component, Entity};

/// Held across `ecs_init` and `ecs_fini`, which write process-wide state (the OS
/// API table, its init count, addon component ids) without synchronization.
/// Nothing can panic while it's held, so it can't be poisoned.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// A flecs world, destroyed on drop.
pub struct World {
    pub(crate) ptr: NonNull<ecs_world_t>,
    /// Unique in this process, unlike `ptr`, which a later world can reuse. This is how a
    /// [`QueryState`](crate::query::QueryState) or a [`SystemId`](crate::system::SystemId) knows
    /// which world it's in.
    pub(crate) id: u64,
}

// SAFETY: flecs has no thread-local state and only ties a world to a thread in
// exclusive-access mode, which nothing here turns on. `INIT_LOCK` covers the
// process-wide state, and `Component` and `register_system` make every stored value `Send`.
unsafe impl Send for World {}

// SAFETY: the `&self` methods only read the world, and `Component` makes every stored
// component `Sync`. A system needn't be, as no `&self` method reaches it. A `const` world
// parameter doesn't prove a new one reads only, as flecs casts it away in places.
unsafe impl Sync for World {}


impl World {
    pub fn new() -> Self {
        let ptr = {
            let _guard = INIT_LOCK.lock().unwrap();
            // SAFETY: the lock serializes this with every other init and fini.
            unsafe { crate::sys::ecs_init() }
        };
        static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            ptr: NonNull::new(ptr).expect("ecs_init returned null"),
            id: NEXT_WORLD_ID.fetch_add(1, Relaxed),
        }
    }

    pub fn new_entity(&mut self) -> Entity {
        // SAFETY: the world is valid and `&mut self` makes this access exclusive.
        let id = unsafe { crate::sys::ecs_new(self.ptr.as_ptr()) };
        Entity(NonZeroU64::new(id).expect("ecs_new returned 0"))
    }

    /// Deletes `entity` and drops its components. Does nothing if it's already dead.
    ///
    /// # Panics
    ///
    /// If `entity` is a component's, a query's or a system's own, which takes an [`Entity`] from
    /// another world. A component can't be deleted, a query is freed by [`Self::return_query`], and
    /// a system is removed by [`Self::unregister_system`].
    pub fn delete(&mut self, entity: Entity) {
        if !self.is_alive(entity) {
            return;
        }
        // SAFETY: the world is valid, and the entity is alive, as just checked.
        unsafe { assert_can_delete(self.ptr.as_ptr(), entity) };
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
    /// this just drops `value`. Fails to compile for a [`Singleton`], which goes through
    /// [`set_singleton`](Self::set_singleton).
    pub fn set<T: Component>(&mut self, entity: Entity, value: T) {
        const {
            assert!(
                !crate::traits::has_singleton(T::TRAITS),
                "a `Singleton` is stored with `World::set_singleton`"
            );
        }
        if !self.is_alive(entity) {
            return;
        }
        let component = self.register::<T>();
        self.store(entity, component, value);
    }

    /// Stores the world's `T`, dropping any it already has.
    pub fn set_singleton<T: Singleton>(&mut self, value: T) {
        let component = self.register::<T>();
        self.store(component, component, value);
    }

    /// Stores `value` on `entity`, which must be live. `component` must be `T`'s.
    fn store<T: Component>(&mut self, entity: Entity, component: Entity, value: T) {
        let component = component.id();
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

    /// The world's `T`, or `None` if it has none.
    pub fn singleton<T: Singleton>(&self) -> Option<&T> {
        self.get::<T>(self.registered_id::<T>()?)
    }

    /// The world's `T`, or `None` if it has none. As with [`get_mut`](Self::get_mut), this
    /// fails to compile for a zero-sized `T`.
    pub fn singleton_mut<T: Singleton>(&mut self) -> Option<Mut<'_, T>> {
        let component = self.registered_id::<T>()?;
        self.get_mut::<T>(component)
    }

    /// Removes the world's `T` and drops it, if it has one.
    pub fn remove_singleton<T: Singleton>(&mut self) {
        if let Some(component) = self.registered_id::<T>() {
            self.remove::<T>(component);
        }
    }

    /// Enables `entity`, undoing [`disable`](Self::disable). Does nothing if it's dead.
    pub fn enable(&mut self, entity: Entity) {
        self.set_enabled(entity, true);
    }

    /// Disables `entity`, so that queries skip it, though its components stay readable. Does
    /// nothing if it's dead.
    pub fn disable(&mut self, entity: Entity) {
        self.set_enabled(entity, false);
    }

    fn set_enabled(&mut self, entity: Entity, enabled: bool) {
        if self.is_alive(entity) {
            // SAFETY: `ecs_enable` needs a live entity, which was just checked.
            unsafe { crate::sys::ecs_enable(self.ptr.as_ptr(), entity.id(), enabled) };
        }
    }

    /// Enables `T` on `entity`, undoing [`disable_component`](Self::disable_component). Does
    /// nothing if `entity` is dead.
    pub fn enable_component<T: CanToggle>(&mut self, entity: Entity) {
        self.set_component_enabled::<T>(entity, true);
    }

    /// Disables `T` on `entity`, so that queries treat it as missing, though [`get`](Self::get)
    /// and [`has`](Self::has) still find it. It stays disabled through [`set`](Self::set) and
    /// [`remove`](Self::remove) until it's enabled, and `entity` needn't have a `T` yet. Does
    /// nothing if `entity` is dead.
    pub fn disable_component<T: CanToggle>(&mut self, entity: Entity) {
        self.set_component_enabled::<T>(entity, false);
    }

    fn set_component_enabled<T: CanToggle>(&mut self, entity: Entity, enabled: bool) {
        if !self.is_alive(entity) {
            return;
        }
        let component = self.register::<T>();
        // SAFETY: the entity is live, and `T: CanToggle` means `register` gave the component
        // that trait, which flecs only checks in debug builds.
        unsafe {
            crate::sys::ecs_enable_id(self.ptr.as_ptr(), entity.id(), component.id(), enabled);
        }
    }

    /// Whether `entity` has a `T` that isn't disabled.
    pub fn is_component_enabled<T: Component>(&self, entity: Entity) -> bool {
        self.component_id::<T>(entity).is_some_and(|component| {
            let world = self.ptr.as_ptr();
            // SAFETY: `component_id` checked that the entity is live. This checks for the
            // component too, as `ecs_is_enabled_id` also says yes if it was only enabled.
            unsafe {
                crate::sys::ecs_has_id(world, entity.id(), component.id())
                    && crate::sys::ecs_is_enabled_id(world, entity.id(), component.id())
            }
        })
    }

    /// flecs's id for `T`, if `entity` is live and `T` has been stored in this world.
    /// flecs's own checks for these are compiled out in `-c opt`.
    fn component_id<T: Component>(&self, entity: Entity) -> Option<Entity> {
        if !self.is_alive(entity) {
            return None;
        }
        self.registered_id::<T>()
    }

    /// flecs's id for `T`, if `T` has been stored in this world.
    fn registered_id<T: Component>(&self) -> Option<Entity> {
        // SAFETY: this crate may read the slot. flecs returns 0 for an index this world
        // hasn't set, which includes a slot's initial 0, as its indexes start at 1.
        let id = unsafe {
            crate::sys::flecs_component_ids_get_alive(self.ptr.as_ptr(), T::slot().load(Relaxed))
        };
        NonZeroU64::new(id).map(Entity)
    }

    /// flecs's id for `T`, registering `T` the first time it's stored in this world.
    pub(crate) fn register<T: Component>(&mut self) -> Entity {
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
        // A label for flecs's errors and explorer, which flecs copies. It needn't be unique.
        let name = CString::new(type_name::<T>()).expect("type names have no NUL");
        desc.type_.name = name.as_ptr();
        // flecs requires a size and alignment of 0 for a tag.
        if size_of::<T>() != 0 {
            desc.type_.size = size_of::<T>() as i32;
            desc.type_.alignment = align_of::<T>() as i32;
            // Zeroed or copied bytes needn't be a valid `T`, so flecs must abort if it ever
            // tries to make one that way.
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
        let component = unsafe {
            let id = crate::sys::ecs_component_init(world, &desc);
            crate::sys::flecs_component_ids_set(world, index, id);
            Entity(NonZeroU64::new(id).expect("ecs_component_init returned 0"))
        };
        for declaration in T::TRAITS {
            (declaration.apply)(self, component);
        }
        component
    }

    /// Adds a trait to a component that `register` just made.
    pub(crate) fn add_trait(&mut self, component: Entity, flecs_trait: ecs_id_t) {
        // SAFETY: the component is live, and nothing has used it yet, as flecs requires.
        unsafe { crate::sys::ecs_add_id(self.ptr.as_ptr(), component.id(), flecs_trait) };
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

/// flecs's id for `T` in `world`, if `T` has been stored there.
///
/// # Safety
///
/// `world` must be valid.
pub(crate) unsafe fn registered_id_in<T: Component>(world: *const ecs_world_t) -> Option<Entity> {
    // SAFETY: the caller upholds this, and this crate may read the slot. flecs returns 0 for an
    // index this world hasn't set, which includes a slot's initial 0, as its indexes start at 1.
    let id = unsafe { crate::sys::flecs_component_ids_get_alive(world, T::slot().load(Relaxed)) };
    NonZeroU64::new(id).map(Entity)
}

/// Panics if `entity` is a component's, a query's or a system's own, which an [`Entity`] from
/// another world can name, and which [`World::delete`] and
/// [`Commands::delete`](crate::system::Commands::delete) refuse.
///
/// # Safety
///
/// `world` must be valid, and `entity` alive in it.
pub(crate) unsafe fn assert_can_delete(world: *const ecs_world_t, entity: Entity) {
    // SAFETY: the caller upholds this.
    let (is_component, is_system, is_query) = unsafe {
        (
            crate::sys::ecs_has_id(world, entity.id(), crate::sys::FLECS_IDEcsComponentID_),
            !crate::sys::ecs_system_get(world, entity.id()).is_null(),
            !crate::sys::ecs_query_get(world, entity.id()).is_null(),
        )
    };
    assert!(!is_component, "a component can't be deleted");
    // Checked before `is_query`, as a system's query is on the system's entity.
    assert!(!is_system, "a system is removed by `World::unregister_system`");
    assert!(!is_query, "a query is freed by `World::return_query`");
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
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::traits::{DontFragment, Sparse, With};

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

    #[test]
    fn set_and_writes_through_get_mut_fire_on_set() {
        #[derive(Component, Debug, PartialEq)]
        struct Watched(u32);

        static CALLS: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "C" fn on_set(_iter: *mut c_void) {
            CALLS.fetch_add(1, Relaxed);
        }

        let mut world = World::new();
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

    #[derive(Component, Debug, PartialEq)]
    #[component(CanToggle)]
    struct Toggled(u32);

    #[derive(Component, Debug, PartialEq)]
    #[component(Singleton)]
    struct Clock(u32);

    #[test]
    fn toggles_a_component() {
        let mut world = World::new();
        let entity = world.new_entity();
        let other = world.new_entity();
        assert!(!world.is_component_enabled::<Toggled>(entity));

        world.set(entity, Toggled(1));
        world.set(other, Toggled(2));
        assert!(world.is_component_enabled::<Toggled>(entity));

        world.disable_component::<Toggled>(entity);
        assert!(!world.is_component_enabled::<Toggled>(entity));
        assert!(world.is_component_enabled::<Toggled>(other));
        assert!(world.has::<Toggled>(entity));
        assert_eq!(world.get::<Toggled>(entity), Some(&Toggled(1)));

        // It stays disabled through writes and table moves.
        world.set(entity, Toggled(3));
        world.set(entity, Count(4));
        assert!(!world.is_component_enabled::<Toggled>(entity));
        assert_eq!(world.get::<Toggled>(entity), Some(&Toggled(3)));

        world.enable_component::<Toggled>(entity);
        assert!(world.is_component_enabled::<Toggled>(entity));
    }

    #[test]
    fn toggles_a_tag() {
        #[derive(Component, Debug, PartialEq)]
        #[component(CanToggle)]
        struct ToggledTag;

        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, ToggledTag);
        world.disable_component::<ToggledTag>(entity);
        assert!(!world.is_component_enabled::<ToggledTag>(entity));
        assert_eq!(world.get::<ToggledTag>(entity), Some(&ToggledTag));
        world.enable_component::<ToggledTag>(entity);
        assert!(world.is_component_enabled::<ToggledTag>(entity));
    }

    #[test]
    fn disabling_outlasts_the_component() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Toggled(1));
        world.disable_component::<Toggled>(entity);

        world.remove::<Toggled>(entity);
        assert!(!world.is_component_enabled::<Toggled>(entity));
        world.set(entity, Toggled(2));
        assert!(!world.is_component_enabled::<Toggled>(entity));

        // Disabling can also come first.
        let early = world.new_entity();
        world.disable_component::<Toggled>(early);
        assert!(!world.has::<Toggled>(early));
        world.set(early, Toggled(3));
        assert!(world.has::<Toggled>(early));
        assert!(!world.is_component_enabled::<Toggled>(early));
    }

    #[test]
    fn only_a_component_the_entity_has_is_enabled() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.enable_component::<Toggled>(entity);
        assert!(!world.has::<Toggled>(entity));
        assert!(!world.is_component_enabled::<Toggled>(entity));

        // This doesn't need `CanToggle`, as a component that can't toggle is always enabled.
        assert!(!world.is_component_enabled::<Count>(entity));
        world.set(entity, Count(1));
        assert!(world.is_component_enabled::<Count>(entity));
    }

    #[test]
    fn toggling_a_dead_entity_does_nothing() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Toggled(1));
        world.delete(entity);

        world.disable_component::<Toggled>(entity);
        world.enable_component::<Toggled>(entity);
        world.disable(entity);
        world.enable(entity);
        assert!(!world.is_component_enabled::<Toggled>(entity));
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn queries_skip_a_disabled_entity_whose_components_stay_readable() {
        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Count(1));
        let mut query = world.query::<&Count>();

        world.disable(entity);
        assert_eq!(query.iter_mut(&mut world).count(), 0);
        assert_eq!(world.get::<Count>(entity), Some(&Count(1)));
        world.set(entity, Count(2));
        assert_eq!(world.get::<Count>(entity), Some(&Count(2)));

        world.enable(entity);
        assert_eq!(query.iter_mut(&mut world).count(), 1);
        world.return_query(query);
    }

    #[test]
    fn stored_outside_tables_keeps_its_address() {
        #[derive(Component)]
        #[component(Sparse)]
        struct InSparse(Tracked);
        #[derive(Component)]
        #[component(DontFragment)]
        struct Unfragmented(Tracked);

        fn check<T: Component>(new: fn(Tracked) -> T, value: fn(&T) -> u32) {
            let drops = Arc::new(AtomicUsize::new(0));
            let mut world = World::new();
            let entity = world.new_entity();
            world.set(entity, new(Tracked::new(1, &drops)));
            let address = std::ptr::from_ref(world.get::<T>(entity).unwrap());

            // These would move a value stored in a table.
            world.set(entity, Count(1));
            for number in 0..2000 {
                let other = world.new_entity();
                world.set(other, new(Tracked::new(number, &drops)));
            }
            assert_eq!(std::ptr::from_ref(world.get::<T>(entity).unwrap()), address);
            assert_eq!(value(world.get::<T>(entity).unwrap()), 1);

            world.set(entity, new(Tracked::new(2, &drops)));
            assert_eq!(drops.load(Relaxed), 1);
            assert_eq!(std::ptr::from_ref(world.get::<T>(entity).unwrap()), address);
            assert_eq!(value(&world.get_mut::<T>(entity).unwrap()), 2);

            *world.get_mut::<T>(entity).unwrap() = new(Tracked::new(3, &drops));
            assert_eq!(drops.load(Relaxed), 2);
            assert_eq!(value(world.get::<T>(entity).unwrap()), 3);

            world.remove::<T>(entity);
            assert_eq!(drops.load(Relaxed), 3);
            assert!(!world.has::<T>(entity));
            assert!(world.get::<T>(entity).is_none());
            assert!(world.get_mut::<T>(entity).is_none());

            world.set(entity, new(Tracked::new(4, &drops)));
            world.delete(entity);
            assert_eq!(drops.load(Relaxed), 4);

            drop(world);
            assert_eq!(drops.load(Relaxed), 2004);
        }

        check(InSparse, |stored| *stored.0.value);
        check(Unfragmented, |stored| *stored.0.value);
    }

    #[test]
    fn toggles_components_stored_outside_tables() {
        #[derive(Component, Debug, PartialEq)]
        #[component(CanToggle, Sparse)]
        struct ToggledSparse(u32);
        #[derive(Component, Debug, PartialEq)]
        #[component(CanToggle, DontFragment)]
        struct ToggledUnfragmented(u32);

        fn check<T: CanToggle + std::fmt::Debug + PartialEq>(new: fn(u32) -> T) {
            let mut world = World::new();
            let entity = world.new_entity();
            world.set(entity, new(1));
            world.disable_component::<T>(entity);
            assert!(!world.is_component_enabled::<T>(entity));
            assert_eq!(world.get::<T>(entity), Some(&new(1)));

            world.set(entity, Count(1));
            assert!(!world.is_component_enabled::<T>(entity));
            assert_eq!(world.get::<T>(entity), Some(&new(1)));

            world.enable_component::<T>(entity);
            assert!(world.is_component_enabled::<T>(entity));

            let early = world.new_entity();
            world.disable_component::<T>(early);
            world.set(early, new(2));
            assert!(!world.is_component_enabled::<T>(early));
            assert_eq!(world.get::<T>(early), Some(&new(2)));
        }

        check(ToggledSparse);
        check(ToggledUnfragmented);
    }

    #[test]
    fn stores_one_singleton_per_world() {
        let mut world = World::new();
        let mut other = World::new();
        assert_eq!(world.singleton::<Clock>(), None);
        assert!(world.singleton_mut::<Clock>().is_none());
        world.remove_singleton::<Clock>();

        world.set_singleton(Clock(1));
        assert_eq!(world.singleton::<Clock>(), Some(&Clock(1)));
        assert_eq!(other.singleton::<Clock>(), None);

        world.set_singleton(Clock(2));
        assert_eq!(world.singleton::<Clock>(), Some(&Clock(2)));
        world.singleton_mut::<Clock>().unwrap().0 = 3;
        assert_eq!(world.singleton::<Clock>(), Some(&Clock(3)));

        other.set_singleton(Clock(4));
        assert_eq!(world.singleton::<Clock>(), Some(&Clock(3)));
        assert_eq!(other.singleton::<Clock>(), Some(&Clock(4)));

        world.remove_singleton::<Clock>();
        assert_eq!(world.singleton::<Clock>(), None);
        world.set_singleton(Clock(5));
        assert_eq!(world.singleton::<Clock>(), Some(&Clock(5)));
    }

    #[test]
    fn drops_a_singleton_once() {
        #[derive(Component)]
        #[component(Singleton)]
        struct Owned(Tracked);

        let drops = Arc::new(AtomicUsize::new(0));
        let mut world = World::new();
        world.set_singleton(Owned(Tracked::new(1, &drops)));
        world.set_singleton(Owned(Tracked::new(2, &drops)));
        assert_eq!(drops.load(Relaxed), 1);
        assert_eq!(*world.singleton::<Owned>().unwrap().0.value, 2);

        world.remove_singleton::<Owned>();
        assert_eq!(drops.load(Relaxed), 2);

        world.set_singleton(Owned(Tracked::new(3, &drops)));
        drop(world);
        assert_eq!(drops.load(Relaxed), 3);
    }

    #[test]
    fn stores_a_singleton_tag() {
        #[derive(Component, Debug, PartialEq)]
        #[component(Singleton)]
        struct Paused;

        let mut world = World::new();
        assert_eq!(world.singleton::<Paused>(), None);
        world.set_singleton(Paused);
        assert_eq!(world.singleton::<Paused>(), Some(&Paused));
        world.remove_singleton::<Paused>();
        assert_eq!(world.singleton::<Paused>(), None);
    }

    #[test]
    fn with_adds_its_targets() {
        #[derive(Component, Debug, PartialEq)]
        struct First;
        #[derive(Component, Debug, PartialEq)]
        #[component(With<First>)]
        struct Second;
        #[derive(Component, Debug, PartialEq)]
        #[component(With<Second>, With<Tag>)]
        struct Holder(u32);

        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Holder(1));
        assert_eq!(world.get::<Holder>(entity), Some(&Holder(1)));
        assert!(world.has::<Second>(entity));
        assert!(world.has::<First>(entity));
        assert!(world.has::<Tag>(entity));

        world.set(entity, Holder(2));
        assert_eq!(world.get::<Holder>(entity), Some(&Holder(2)));

        world.remove::<Holder>(entity);
        assert!(!world.has::<Holder>(entity));
        assert!(world.has::<Second>(entity));

        let plain = world.new_entity();
        world.set(plain, Second);
        assert!(world.has::<First>(plain));
        assert!(!world.has::<Holder>(plain));
    }

    #[test]
    fn with_targets_can_have_traits_of_their_own() {
        #[derive(Component, Debug, PartialEq)]
        #[component(Sparse)]
        struct SparseTag;
        #[derive(Component, Debug, PartialEq)]
        #[component(With<SparseTag>, Sparse, CanToggle)]
        struct Holder(u32);

        let mut world = World::new();
        let entity = world.new_entity();
        world.set(entity, Holder(1));
        world.disable_component::<Holder>(entity);
        assert_eq!(world.get::<Holder>(entity), Some(&Holder(1)));
        assert!(world.has::<SparseTag>(entity));
    }

    #[test]
    #[should_panic = "a component can't be deleted"]
    fn deleting_a_component_panics() {
        let mut world = World::new();
        let component = world.register::<Count>();
        world.delete(component);
    }

    /// `Skipped` is only stored in another world, and `NoIndexYet` in none.
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

    /// `sys` is written by hand for this version, so an upgrade must check it again.
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

    #[test]
    fn a_singleton_can_list_its_traits_in_any_order() {
        #[derive(Component, Debug, PartialEq)]
        #[component(Singleton, Sparse, CanToggle)]
        struct ListedFirst(u32);

        #[derive(Component, Debug, PartialEq)]
        #[component(Sparse, CanToggle, Singleton)]
        struct ListedLast(u32);

        let mut world = World::new();
        world.set_singleton(ListedFirst(1));
        world.set_singleton(ListedLast(2));
        assert_eq!(world.singleton::<ListedFirst>(), Some(&ListedFirst(1)));
        assert_eq!(world.singleton::<ListedLast>(), Some(&ListedLast(2)));
    }
}
