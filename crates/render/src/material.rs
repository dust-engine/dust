use std::marker::PhantomData;

use crate::geometry::Geometry;
use crate::pipeline::{HitGroup, HitGroupType};
use crate::shader::{Shader, SpecializedShader};
use bevy_app::{App, Plugin};
use bevy_asset::{AddAsset, Asset, AssetServer, Handle};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::system::{Commands, Query};

pub trait Material: Asset {
    type Geometry: Geometry;

    fn anyhit_shader(asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn closest_hit_shader(asset_server: &AssetServer) -> Option<SpecializedShader>;
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

impl<T: Material> Plugin for MaterialPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_asset::<T>();
        let asset_server = app.world.get_resource::<AssetServer>().unwrap();
        let hitgroup = HitGroup {
            intersection_shader: Some(T::Geometry::intersection_shader(asset_server)),
            anyhit_shader: T::anyhit_shader(asset_server),
            closest_hit_shader: T::closest_hit_shader(asset_server),
            ty: HitGroupType::Procedural,
        };
        let mut hitgroups = app
            .world
            .get_resource_mut::<Vec<HitGroup>>()
            .expect("MaterialPlugin must be registered after RenderPlugin");
        let hitgroup_index = hitgroups.len() as u32;
        hitgroups.push(hitgroup);

        // On each frame, Handle<Material> -> ExtractedMaterial
        app.sub_app_mut(crate::RenderApp).add_system_to_stage(
            crate::RenderStage::Extract,
            move |mut commands: Commands, query: Query<(Entity, &Handle<T>)>| {
                for (entity, material) in query.iter() {
                    commands.get_or_spawn(entity).insert(ExtractedMaterial {
                        hitgroup_index,
                        material_id: 0,
                    });
                }
            },
        );
    }
}

#[derive(Component, Clone)]
pub struct ExtractedMaterial {
    pub hitgroup_index: u32,
    /// The material_id in the SBT
    pub material_id: u32,
}
