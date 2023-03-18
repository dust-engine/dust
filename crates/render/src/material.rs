use std::marker::PhantomData;

use bevy_app::Plugin;
use bevy_asset::{Assets, Handle};
use bevy_ecs::{
    query::Added,
    system::{Query, Res, ResMut},
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
    fn rahit_shader(ray_type: u32) -> Option<SpecializedShader>;
    fn rchit_shader(ray_type: u32) -> Option<SpecializedShader>;
    fn intersection_shader(ray_type: u32) -> Option<SpecializedShader>;
    type ShaderParameters;
    fn parameters(&self, ray_type: u32) -> Self::ShaderParameters;

    type Object;
    // Add texture mapped object
    fn add_object(&mut self, _obj: &Self::Object) {}
}

pub struct MaterialPlugin<M: Material> {
    _marker: PhantomData<M>,
}
impl<M: Material> Plugin for MaterialPlugin<M> {
    fn build(&self, app: &mut bevy_app::App) {
        let pipeline_builder: &mut RayTracingPipelineBuilder<M::Pipeline> = &mut *app
            .world
            .get_resource_mut()
            .expect("MaterialPlugin must be inserted after the RayTracingPipeline plugin");
        pipeline_builder.register_material::<M>();
    }
}

fn material_system<T: Material>(
    mut pipeline: ResMut<T::Pipeline>,
    materials: Res<Assets<T>>,
    mut query: Query<(&Handle<T>, &mut SbtIndex), Added<Handle<T>>>,
) {
    for (material_handle, mut sbt_index) in query.iter_mut() {
        if let Some(material) = materials.get(material_handle) {
            *sbt_index = pipeline.material_instance_added(material);
        }
    }
}
