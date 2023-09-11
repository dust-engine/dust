use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use bevy_app::{Plugin, PostUpdate};
use bevy_asset::{Asset, AssetEvent, AssetId, AssetServer, Assets, Handle};
use bevy_ecs::{
    prelude::{Entity, EventReader},
    query::Changed,
    system::{Commands, Local, Query, Res, ResMut, SystemParam, SystemParamItem},
};
use bevy_reflect::TypePath;

use crate::{
    pipeline::RayTracingPipeline, pipeline::RayTracingPipelineBuilder, sbt::SbtIndex,
    shader::SpecializedShader,
};

pub type MaterialType = rhyolite::RayTracingHitGroupType;
// Handle<Material> is a component
pub trait Material: Send + Sync + 'static + Asset + TypePath {
    type Pipeline: RayTracingPipeline;
    const TYPE: MaterialType;
    fn rahit_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn rchit_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn intersection_shader(ray_type: u32, asset_server: &AssetServer) -> Option<SpecializedShader>;

    type ShaderParameters;
    type ShaderParameterParams: SystemParam;
    fn parameters(
        &self,
        ray_type: u32,
        params: &mut SystemParamItem<Self::ShaderParameterParams>,
    ) -> Self::ShaderParameters;
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
        app.add_systems(PostUpdate, material_system::<M>);
    }
}

struct MaterialStore<T: Material> {
    sbt_indices: HashMap<AssetId<T>, SbtIndex>,
    entitites: HashMap<AssetId<T>, HashSet<Entity>>,
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

    mut params: bevy_ecs::system::StaticSystemParam<T::ShaderParameterParams>,
) {
    for (entity, handle) in query.iter() {
        store
            .entitites
            .entry(handle.id())
            .or_default()
            .insert(entity);
        if let Some(sbt_index) = store.sbt_indices.get(&handle.id()) {
            // If this returns Some, it means `AssetEvent::Created` was already received,
            // and the SBT entry was already created. Add that to the entity.
            // If it does not already exist, do nothing. The SbtIndex will be added to the
            // entity later when `AssetEvent::Created` was called.
            commands.entity(entity).insert(*sbt_index);
        }
    }
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let material = materials.get(*id).unwrap();
                let sbt_index = pipeline.material_instance_added(material, &mut params);
                // Now, for all entities with Handle<T>, add SbtIndex.
                if let Some(old_sbt_index) = store.sbt_indices.get(id) && old_sbt_index == &sbt_index {

                } else {
                    store.sbt_indices.insert(*id, sbt_index);
                    if let Some(entities) = store.entitites.get(id) {
                        for entity in entities.iter() {
                            commands.entity(*entity).insert(sbt_index);
                        }
                    }
                }
            }
            AssetEvent::Removed { id: _ } => {
                todo!()
            }
            _ => (),
        }
    }
}
