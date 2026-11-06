//! The flecs traits a component can have, listed by path in `#[component(...)]`:
//!
//! ```no_run
//! use dust_ecs::Component;
//! use dust_ecs::traits::{CanToggle, Sparse, With};
//!
//! #[derive(Component)]
//! struct Visible;
//!
//! #[derive(Component)]
//! #[component(CanToggle, Sparse, With<Visible>)]
//! struct Health(u32);
//! ```
//!
//! They belong to the type, so they're the same in every world, because flecs needs them
//! before a component's first use. Added any later, debug builds abort, and in `-c opt`
//! `Sparse` and `DontFragment` lose the stored value.
//!
//! # Safety
//!
//! Every trait here is `unsafe` to implement by hand, because `World` relies on it also being
//! in [`Component::TRAITS`]. `#[component(...)]` does both.

use crate::world::World;
use crate::{Component, Entity, sys};

/// What listing a trait does to a component's entity, which `World::register` runs.
#[doc(hidden)]
pub struct Declaration {
    pub(crate) apply: fn(&mut World, Entity),
    pub(crate) singleton: bool,
}

/// Whether `traits` includes [`Singleton`].
pub(crate) const fn has_singleton(traits: &[Declaration]) -> bool {
    let mut i = 0;
    while i < traits.len() {
        if traits[i].singleton {
            return true;
        }
        i += 1;
    }
    false
}

/// Lets [`World::disable_component`] and [`World::enable_component`] toggle this component,
/// which is cheaper than removing and adding it again. Queries treat a disabled component
/// as missing, but [`World::get`] and [`World::has`] still find it.
///
/// Queries on a component with this trait cost slightly more, even for entities that never
/// toggled it, so it's only for components that need it.
///
/// # Safety
///
/// See the [module docs](self).
pub unsafe trait CanToggle: Component {
    #[doc(hidden)]
    const DECLARATION: Declaration = Declaration {
        apply: |world, component| world.add_trait(component, sys::EcsCanToggle),
        singleton: false,
    };
}

/// Stores this component outside tables, so its address stays the same until it's removed.
/// Queries on it are slower.
///
/// # Safety
///
/// See the [module docs](self).
pub unsafe trait Sparse: Component {
    #[doc(hidden)]
    const DECLARATION: Declaration = Declaration {
        apply: |world, component| world.add_trait(component, sys::EcsSparse),
        singleton: false,
    };
}

/// Stores this component as [`Sparse`] does, and adding it doesn't move an entity to another
/// table. This suits a component on few entities, as it doesn't multiply tables.
///
/// # Safety
///
/// See the [module docs](self).
pub unsafe trait DontFragment: Component {
    #[doc(hidden)]
    const DECLARATION: Declaration = Declaration {
        apply: |world, component| world.add_trait(component, sys::EcsDontFragment),
        singleton: false,
    };
}

/// Limits a world to one of this component, which lives on the component's own entity. Use
/// it through [`World::set_singleton`], [`World::singleton`], [`World::singleton_mut`] and
/// [`World::remove_singleton`]. [`World::set`] fails to compile for it, as flecs only
/// checks the entity in debug builds.
///
/// # Safety
///
/// See the [module docs](self).
pub unsafe trait Singleton: Component {
    #[doc(hidden)]
    const DECLARATION: Declaration = Declaration {
        apply: |world, component| world.add_trait(component, sys::EcsSingleton),
        singleton: true,
    };
}

/// Adds the tag `U` to any entity this component is added to. Removing this component leaves
/// `U` on the entity.
///
/// Fails to compile if `U` isn't zero-sized, as flecs would have to construct it, or if `With`
/// traits form a cycle, which would overflow the stack in flecs.
///
/// # Safety
///
/// See the [module docs](self).
pub unsafe trait With<U: Component>: Component {
    #[doc(hidden)]
    const DECLARATION: Declaration = {
        // Makes `U`'s traits depend on this one's, so a cycle is a cycle in const evaluation.
        let _ = U::TRAITS;
        assert!(size_of::<U>() == 0, "the target of `With` must be zero-sized");
        Declaration {
            apply: |world, component| {
                let target = world.register::<U>();
                // SAFETY: neither id is a pair, the only thing flecs checks here.
                let pair = unsafe { sys::ecs_make_pair(sys::EcsWith, target.id()) };
                world.add_trait(component, pair);
            },
            singleton: false,
        }
    };
}
