use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use bevy_app::{Plugin, Update};
use bevy_asset::{AssetEvent, AssetServer, Assets, Handle};
use bevy_ecs::{
    prelude::{Entity, EventReader},
    query::Changed,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_reflect::TypeUuid;

use crate::{
    pipeline::RayTracingPipeline, pipeline::RayTracingPipelineBuilder, sbt::SbtIndex,
    shader::SpecializedShader,
};

pub type MaterialType = rhyolite::RayTracingHitGroupType;
// Handle<Material> is a component
pub trait Material: Send + Sync + 'static + TypeUuid {
    type Pipeline: RayTracingPipeline;
    const TYPE: MaterialType;
    fn rahit_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn rchit_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn intersection_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader>;
    type ShaderParameters;
    fn parameters(&self, ray_type: u32) -> Self::ShaderParameters;
}

pub struct MaterialPlugin<M: Material> {
    _marker: PhantomData<M>,
}
impl<M: Material> Default for MaterialPlugin<M> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<M: Material> Plugin for MaterialPlugin<M> {
    fn build(&self, app: &mut bevy_app::App) {
        app.world
            .resource_scope::<RayTracingPipelineBuilder<M::Pipeline>, ()>(
                |world, mut pipeline_builder| {
                    pipeline_builder.register_material::<M>(world.resource());
                },
            );
        app.add_systems(Update, material_system::<M>);
    }
}

struct MaterialStore<T: Material> {
    sbt_indices: HashMap<Handle<T>, SbtIndex>,
    entitites: HashMap<Handle<T>, HashSet<Entity>>,
}
impl<T: Material> Default for MaterialStore<T> {
    fn default() -> Self {
        Self {
            sbt_indices: Default::default(),
            entitites: Default::default(),
        }
    }
}

fn material_system<T: Material>(
    mut commands: Commands,
    mut store: Local<MaterialStore<T>>,
    mut pipeline: ResMut<T::Pipeline>,
    materials: Res<Assets<T>>,
    mut events: EventReader<AssetEvent<T>>,
    query: Query<(Entity, &Handle<T>), Changed<Handle<T>>>,
) {
    for (entity, handle) in query.iter() {
        store
            .entitites
            .entry(handle.clone_weak())
            .or_default()
            .insert(entity);
    }
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                let material = materials.get(handle).unwrap();
                let sbt_index = pipeline.material_instance_added(material);
                // Now, for all entities with Handle<T>, add SbtIndex.
                if let Some(old_sbt_index) = store.sbt_indices.get(handle) && old_sbt_index == &sbt_index {

                } else {
                    store.sbt_indices.insert(handle.clone_weak(), sbt_index);
                    for entity in store.entitites.get(handle).unwrap().iter() {
                        commands.entity(*entity).insert(sbt_index);
                    }
                }
            }
            AssetEvent::Removed { handle: _ } => {
                todo!()
            }
        }
    }
}
