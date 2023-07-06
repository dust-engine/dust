use std::marker::PhantomData;

use bevy_app::{Plugin, PostUpdate};
use bevy_ecs::{
    prelude::Component,
    query::{Changed, Or},
    schedule::IntoSystemConfigs,
    system::{Query, ResMut, Resource},
};
use crevice::std430::AsStd430;
use rhyolite::{ash::vk, ManagedBufferVec};
use rhyolite_bevy::{Allocator, RenderSystems};

use crate::TLASIndex;

/// Plugin that collects instance-related
pub struct InstanceVecPlugin<Item: InstanceVecItem, Marker: Component>
where
    <Item::Data as AsStd430>::Output: Send + Sync,
{
    buffer_usage_flags: vk::BufferUsageFlags,
    alignment: u32,
    _marker: PhantomData<(Item, Marker)>,
}
impl<Item: InstanceVecItem, Marker: Component> InstanceVecPlugin<Item, Marker>
where
    <Item::Data as AsStd430>::Output: Send + Sync,
{
    pub fn new(usage_flags: vk::BufferUsageFlags, alignment: u32) -> Self {
        Self {
            buffer_usage_flags: usage_flags,
            alignment,
            _marker: PhantomData,
        }
    }
}
impl<Item: InstanceVecItem, Marker: Component> Plugin for InstanceVecPlugin<Item, Marker>
where
    <Item::Data as AsStd430>::Output: Send + Sync,
{
    fn build(&self, app: &mut bevy_app::App) {
        let allocator: Allocator = app.world.resource::<Allocator>().clone();
        app.insert_resource(InstanceVecStore::<Item> {
            buffer: ManagedBufferVec::new(
                allocator.into_inner(),
                self.buffer_usage_flags,
                self.alignment,
            ),
        });
        app.add_systems(
            PostUpdate,
            (bevy_ecs::schedule::apply_deferred, collect::<Item, Marker>)
                .chain()
                .after(super::tlas::tlas_system::<Marker>)
                .in_set(RenderSystems::SetUp),
        );
    }
}

#[derive(Resource)]
pub struct InstanceVecStore<Item: InstanceVecItem> {
    pub buffer: ManagedBufferVec<<Item::Data as AsStd430>::Output>,
}

pub trait InstanceVecItem: Component {
    type Data: AsStd430 + Send + Sync;
    fn data(&self) -> Self::Data;
}

fn collect<Item: InstanceVecItem, Marker: Component>(
    mut store: ResMut<InstanceVecStore<Item>>,
    query: Query<(&TLASIndex<Marker>, &Item), Or<(Changed<Item>, Changed<TLASIndex<Marker>>)>>,
) where
    <Item::Data as AsStd430>::Output: Send + Sync,
{
    for (tlas_index, data) in query.iter() {
        store
            .buffer
            .set(tlas_index.index as usize, data.data().as_std430())
    }
    // TODO: implement removal.
}
