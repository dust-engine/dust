use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    marker::PhantomData,
};

use bevy::asset::{Asset, AssetEvent, AssetId, AssetServer, Assets, Handle};
use bevy::ecs::{
    prelude::{Entity, EventReader},
    query::Changed,
    system::{Commands, Local, Query, Res, ResMut, SystemParam, SystemParamItem},
};
use bevy::reflect::TypePath;
use bevy::{
    app::{Plugin, PostUpdate},
    asset::UntypedAssetId,
    core::Pod,
    ecs::{component::Component, query::Added, system::Resource},
};
use rhyolite::{
    ash::vk,
    pipeline::{PipelineCache},
    shader::SpecializedShader,
};
use rhyolite_rtx::{HitGroup, HitgroupHandle, SbtMarker};

use crate::pipeline::{RayTracingPipeline, RayTracingPipelineBuilder};

pub enum MaterialType {
    Triangles,
    Procedural,
}
pub trait Material: Send + Sync + 'static + Asset + TypePath {
    type Pipeline: RayTracingPipeline;
    fn hitgroup(asset_server: &AssetServer) -> HitGroup;

    type ShaderParameters: Pod;
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
impl<M: Material> Plugin for MaterialPlugin<M>
where
    [(); M::Pipeline::NUM_RAYTYPES]: Sized,
{
    fn build(&self, app: &mut bevy::app::App) {
        app.world
            .resource_scope::<RayTracingPipelineBuilder<M::Pipeline>, ()>(
                |world, mut pipeline_builder| {
                    pipeline_builder.register_material::<M>(world.resource());
                },
            );
        app.add_systems(PostUpdate, material_system::<M>);
    }
}

#[derive(Resource)]
pub struct MaterialStore<P: RayTracingPipeline> {
    entitites: BTreeMap<UntypedAssetId, MaterialData>,
    hitgroup_handle: Option<HitgroupHandle>,
    _marker: PhantomData<P>,
}
impl<P: RayTracingPipeline> Default for MaterialStore<P> {
    fn default() -> Self {
        Self {
            entitites: Default::default(),
            hitgroup_handle: None,
            _marker: PhantomData,
        }
    }
}

fn material_system<T: Material>(
    mut commands: Commands,
    mut store: ResMut<MaterialStore<T::Pipeline>>,
    mut pipeline: ResMut<T::Pipeline>,
    materials: Res<Assets<T>>,
    mut events: EventReader<AssetEvent<T>>,
    added_handles: Query<(Entity, &Handle<T>), Added<Handle<T>>>,
    mut params: bevy::ecs::system::StaticSystemParam<T::ShaderParameterParams>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) where
    [(); T::Pipeline::NUM_RAYTYPES]: Sized,
{
    for (entity, handle) in added_handles.iter() {
        let handle_untyped = handle.id().untyped();
        let entry = store.entitites.entry(handle_untyped).or_default();
        entry.entities.insert(entity);
        if entry.hitgroup_params.is_some() {
            // If this returns Some, it means `AssetEvent::Created` was already received,
            // Add that to the entity.
            // If it does not already exist, do nothing. The SbtIndex will be added to the
            // entity later when `AssetEvent::Created` was called.
            commands.entity(entity).insert(MaterialKey::<T::Pipeline> {
                asset_id: handle_untyped,
                _marker: PhantomData,
            });
        }
    }
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let material = materials.get(*id).unwrap();
                let entry = store.entitites.entry(id.untyped()).or_default();
                for entity in entry.entities.iter() {
                    let key = MaterialKey::<T::Pipeline> {
                        asset_id: id.untyped(),
                        _marker: PhantomData,
                    };
                    // Even if the entity already has the key component, this would trigger the
                    // `Changed` event, causing the SbtManager to update.
                    commands.entity(*entity).insert(key);
                }
                // Update params
                let mut data: Vec<u8> = vec![
                    0;
                    pipeline.manager().hitgroup_layout().inline_params_size
                        * T::Pipeline::NUM_RAYTYPES
                ];
                for ray_type in 0..T::Pipeline::NUM_RAYTYPES {
                    let item = material.parameters(ray_type as u32, &mut params);
                    let dst_slice = &mut data[ray_type
                        * pipeline.manager().hitgroup_layout().inline_params_size
                        ..(ray_type + 1) * pipeline.manager().hitgroup_layout().inline_params_size];
                    dst_slice.copy_from_slice(bevy::core::cast_slice(&[item]));
                }
                entry.hitgroup_params = Some(data.into_boxed_slice());

                // Add material
                if store.hitgroup_handle.is_none() {
                    let hitgroup = T::hitgroup(asset_server.as_ref());
                    store.hitgroup_handle = Some(
                        pipeline
                            .manager_mut()
                            .add_hitgroup(hitgroup, pipeline_cache.as_ref()),
                    );
                }
            }
            AssetEvent::Removed { id: _ } => {
                pipeline
                    .manager_mut()
                    .remove_hitgroup(store.hitgroup_handle.unwrap());
            }
            _ => (),
        }
    }
}

/// This component will be automatically added or removed based on asset states.
#[derive(Component)]
pub struct MaterialKey<P: RayTracingPipeline> {
    asset_id: UntypedAssetId,
    _marker: PhantomData<P>,
}

#[derive(Default)]
struct MaterialData {
    entities: BTreeSet<Entity>,
    /// When this is None, the asset hasn't been loaded yet.
    hitgroup_params: Option<Box<[u8]>>,
}

pub struct MaterialSbtMarker<P: RayTracingPipeline> {
    _marker: PhantomData<P>,
}

impl<P: RayTracingPipeline> SbtMarker for MaterialSbtMarker<P> {
    type HitgroupKey = UntypedAssetId;

    type Marker = MaterialKey<P>;

    type QueryData = &'static MaterialKey<P>;

    type QueryFilter = ();
    type Params = Res<'static, MaterialStore<P>>;

    fn hitgroup_param(
        params: &mut Res<MaterialStore<P>>,
        key: &bevy::ecs::query::QueryItem<Self::QueryData>,
        ret: &mut [u8],
    ) {
        let data = params.entitites.get(&key.asset_id).unwrap();
        let slice = data.hitgroup_params.as_ref().unwrap();
        ret.copy_from_slice(slice);
    }

    fn hitgroup_handle(
        params: &mut Res<MaterialStore<P>>,
        key: &bevy::ecs::query::QueryItem<Self::QueryData>,
    ) -> HitgroupHandle {
        params.hitgroup_handle.unwrap()
    }

    fn hitgroup_key(
        _params: &mut Res<MaterialStore<P>>,
        data: &bevy::ecs::query::QueryItem<Self::QueryData>,
    ) -> Self::HitgroupKey {
        data.asset_id
    }
}
