//! Calls between the Bevy thread and the UI thread.
//!
//! Neither thread can touch the other's state: GPUI's `App` lives in a
//! `RefCell` on the main thread, and Bevy's `World` is borrowed by whichever
//! system is running. So each side queues closures for the other to run with
//! its own state:
//!
//! - [`GPUIChannel`], a resource of the Bevy app, queues closures for the UI
//!   thread. A GPUI task runs them with `&mut gpui::App`, in the order they were
//!   queued, between UI frames.
//! - [`BevyChannel`], a global of the GPUI app, queues closures for the Bevy
//!   thread. The app's runner runs them with `&mut World`, in the order they
//!   were queued, before the next frame.
//!
//! Neither side ever waits for the other: a result goes back as another call.
//!
//! [`BevyChannel::run_every_frame`] runs a system on the Bevy thread every
//! frame, for as long as the UI thread keeps its `SystemHandle`.

use std::sync::mpsc::Sender;

use bevy::ecs::system::{SystemHandle, SystemId};
use bevy::prelude::{Component, Entity, IntoSystem, QueryState, Resource, With, World, warn_once};
use futures_channel::mpsc::UnboundedSender;

/// A closure queued for the UI thread.
pub type UiCall = Box<dyn FnOnce(&mut gpui::App) + Send>;

/// A closure queued for the Bevy thread.
pub type BevyCall = Box<dyn FnOnce(&mut World) + Send>;

/// Runs closures on the UI thread; a resource of the Bevy app. Clones queue to
/// the same thread, from any thread.
#[derive(Resource, Clone)]
pub struct GPUIChannel(pub UnboundedSender<UiCall>);

impl GPUIChannel {
    /// Runs `call` with the GPUI app, after the calls queued before it.
    pub fn update(&self, call: impl FnOnce(&mut gpui::App) + Send + 'static) {
        // Fails only once the UI has quit.
        self.0.unbounded_send(Box::new(call)).ok();
    }
}

/// Runs closures on the Bevy thread; a global of the GPUI app.
#[derive(Clone)]
pub struct BevyChannel(pub Sender<BevyCall>);

impl gpui::Global for BevyChannel {}

impl BevyChannel {
    /// Runs `call` with the Bevy world, after the calls queued before it.
    pub fn update(&self, call: impl FnOnce(&mut World) + Send + 'static) {
        // Fails only once the Bevy app has exited.
        self.0.send(Box::new(call)).ok();
    }

    /// Runs `system` at the end of every Bevy frame, in `Last`, and passes its
    /// [`SystemHandle`] to `then` on the UI thread.
    ///
    /// The system is registered with `World::register_tracked_system`: it runs
    /// until the last clone of the handle is dropped, on either thread, and
    /// Bevy then despawns it. It starts running before `then` gets the handle,
    /// so a `then` that drops the handle still lets it run for a frame or two.
    pub fn run_every_frame<M>(
        &self,
        system: impl IntoSystem<(), (), M> + Send + 'static,
        then: impl FnOnce(SystemHandle, &mut gpui::App) + Send + 'static,
    ) {
        self.update(move |world| {
            let handle = world.register_tracked_system(system);
            world.entity_mut(handle.entity()).insert(EveryFrame);
            world
                .resource::<GPUIChannel>()
                .update(move |cx| then(handle, cx));
        });
    }
}

/// Marks a system registered by [`BevyChannel::run_every_frame`].
#[derive(Component)]
pub(super) struct EveryFrame;

/// Runs the systems marked [`EveryFrame`]. Added to `Last`, after Bevy
/// despawns the systems whose handles were dropped.
pub(super) fn run_every_frame(
    world: &mut World,
    systems: &mut QueryState<Entity, With<EveryFrame>>,
) {
    let systems: Vec<Entity> = systems.iter(world).collect();
    for system in systems {
        if let Err(error) = world.run_system(SystemId::<(), ()>::from_entity(system)) {
            warn_once!("A system run every frame failed: {error}");
        }
    }
}
