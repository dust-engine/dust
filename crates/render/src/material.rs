use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::geometry::{GPUGeometry, Geometry};
use crate::pipeline::{HitGroup, HitGroupType, HitGroups};
use crate::render_asset::{
    BindlessAssetsSystem, GPURenderAsset, RenderAsset, RenderAssetPlugin, RenderAssetStore,
};
use crate::shader::SpecializedShader;

use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, Handle, HandleId};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;

use bevy_ecs::schedule::ParallelSystemDescriptorCoercion;
use bevy_ecs::system::{Commands, Query, Res, StaticSystemParam, SystemParam, SystemParamItem};

use dustash::resources::alloc::MemBuffer;

pub trait Material: RenderAsset {
    type Geometry: Geometry;

    fn anyhit_shader(asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn closest_hit_shader(asset_server: &AssetServer) -> Option<SpecializedShader>;
}

pub trait GPUMaterial<T: Material>: GPURenderAsset<T> {
    type SbtData: Copy;

    type MaterialInfoParams: SystemParam;
    fn material_info(
        &self,
        handle: &Handle<T>,
        params: &mut SystemParamItem<Self::MaterialInfoParams>,
    ) -> Self::SbtData;
}

#[derive(Component, Clone)]
pub struct GPUGeometryMaterial {
    pub geometry_handle: HandleId,
    pub material_handle: HandleId,
    // None if the geometry hasn't been loaded yet. This tells the BLAS systems to not build the BLAS yet.
    pub blas_input_primitives: Option<Arc<MemBuffer>>,
    pub blas_input_layout: Layout,
    pub sbt_data: Option<[u8; 32]>, // 32 byte
    pub hitgroup_index: u32,
}

pub struct MaterialPlugin<T: Material> {
    _marker: PhantomData<T>,
}

impl<T: Material> Default for MaterialPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// Insert Handle<T> in the render world for all entities with Handle<T>
fn extract_primitives<T: Material>(mut commands: Commands, query: Query<(Entity, &Handle<T>)>) {
    for (entity, geometry_handle) in query.iter() {
        commands
            .get_or_spawn(entity)
            .insert(geometry_handle.as_weak::<T>());
    }
}

impl<T: Material> Plugin for MaterialPlugin<T>
where
    <T::Geometry as RenderAsset>::GPUAsset: GPUGeometry<T::Geometry>,
    T::GPUAsset: GPUMaterial<T>,
{
    fn build(&self, app: &mut App) {
        app.add_plugin(RenderAssetPlugin::<T>::default());

        let asset_server = app.world.get_resource::<AssetServer>().unwrap();
        let hitgroup = HitGroup {
            intersection_shader: Some(T::Geometry::intersection_shader(asset_server)),
            anyhit_shader: T::anyhit_shader(asset_server),
            closest_hit_shader: T::closest_hit_shader(asset_server),
            ty: HitGroupType::Procedural,
        };
        let mut hitgroups = app
            .world
            .get_resource_mut::<HitGroups>()
            .expect("MaterialPlugin must be registered after RenderPlugin");
        let hitgroup_index = hitgroups.len() as u32;
        hitgroups.push(hitgroup);

        let prepare_primitives =
            move |mut commands: Commands,
                  geometry_store: Res<RenderAssetStore<T::Geometry>>,
                  material_store: Res<RenderAssetStore<T>>,
                  mut material_params: StaticSystemParam<
                    <T::GPUAsset as GPUMaterial<T>>::MaterialInfoParams,
                >,
            mut geometry_params: StaticSystemParam<
                <<T::Geometry as RenderAsset>::GPUAsset as GPUGeometry<T::Geometry>>::GeometryInfoParams,
            >,
                  query: Query<(Entity, &Handle<T::Geometry>, &Handle<T>)>| {
                for (entity, geometry_handle, material_handle) in query.iter() {
                    if let Some(geometry) = geometry_store.get(geometry_handle) {
                        if let Some(material) = material_store.get(material_handle) {
                            let buf = geometry.blas_input_buffer().clone();
                            let sbt_data = (
                                geometry.geometry_info(&geometry_handle, &mut geometry_params),
                                material.material_info(&material_handle, &mut material_params),
                            );
                            let sbt_data = unsafe {
                                let mut sbt_data_raw: [u8; 32] = [0; 32];
                                std::ptr::copy_nonoverlapping(
                                    &sbt_data as *const _ as *const u8,
                                    sbt_data_raw.as_mut_ptr(),
                                    std::mem::size_of_val(&sbt_data).min(32),
                                );
                                sbt_data_raw
                            };

                            commands.entity(entity).insert(GPUGeometryMaterial {
                                geometry_handle: geometry_handle.id,
                                material_handle: material_handle.id,
                                blas_input_primitives: Some(buf),
                                blas_input_layout: <<<T as Material>::Geometry as RenderAsset>::GPUAsset>::blas_input_layout(),
                                sbt_data: Some(sbt_data),
                                hitgroup_index,
                            });
                            continue;
                        }
                    }
                    commands.entity(entity).insert(GPUGeometryMaterial {
                        geometry_handle: geometry_handle.id,
                        material_handle: material_handle.id,
                        blas_input_primitives: None,
                        blas_input_layout: <<<T as Material>::Geometry as RenderAsset>::GPUAsset>::blas_input_layout(),
                        sbt_data: None,
                        hitgroup_index,
                    });
                }
            };

        // On each frame, Handle<Material> -> ExtractedMaterial
        app.sub_app_mut(crate::RenderApp)
            .add_system_to_stage(crate::RenderStage::Extract, extract_primitives::<T>)
            .add_system_to_stage(
                crate::RenderStage::Prepare,
                prepare_primitives.after(BindlessAssetsSystem::<T>::default()),
            );
        // TODO: maybe run prepare_material_info after prepare_materials to decrease frame delay?
    }
}
