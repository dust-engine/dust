use std::marker::PhantomData;
use std::sync::Arc;

use crate::geometry::Geometry;
use crate::pipeline::{HitGroup, HitGroupType};
use crate::shader::{Shader, SpecializedShader};
use ash::vk;
use bevy_app::{App, Plugin};
use bevy_asset::{AddAsset, Asset, AssetEvent, AssetServer, Assets, Handle};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::event::EventReader;
use bevy_ecs::prelude::FromWorld;
use bevy_ecs::system::{
    Commands, Query, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem,
};
use bevy_utils::HashMap;
use dustash::queue::semaphore::TimelineSemaphoreOp;
use dustash::queue::{QueueType, Queues};
use dustash::sync::{CommandsFuture, GPUFuture};
use dustash::Device;

pub trait Material: Asset + Sized {
    type Geometry: Geometry;

    fn anyhit_shader(asset_server: &AssetServer) -> Option<SpecializedShader>;
    fn closest_hit_shader(asset_server: &AssetServer) -> Option<SpecializedShader>;

    /// The geometry represented as an array of primitives
    /// This gets persisted in the render world.
    /// This is a GPU state.
    type GPUMaterial: GPUMaterial<Self>;

    /// The change in material.
    /// This gets extracted from the main world into the render world each frame.
    type ChangeSet: Send + Sync;
    type BuildSet: Send + Sync;

    type GenerateBuildsParam: SystemParam;
    fn generate_builds(
        &mut self,
        param: &mut SystemParamItem<Self::GenerateBuildsParam>,
    ) -> Self::BuildSet;

    type EmitChangesParam: SystemParam;
    fn emit_changes(
        &mut self,
        param: &mut SystemParamItem<Self::EmitChangesParam>,
    ) -> Self::ChangeSet;
}

pub trait GPUMaterial<T: Material>: Send + Sync {
    type BuildParam: SystemParam;
    fn build(
        build_set: T::BuildSet,
        commands_future: &mut CommandsFuture,
        params: &mut SystemParamItem<Self::BuildParam>,
    ) -> Self;

    type ApplyChangeParam: SystemParam;
    fn apply_change_set(
        &mut self,
        change_set: T::ChangeSet,
        commands_future: &mut CommandsFuture,
        params: &mut SystemParamItem<Self::ApplyChangeParam>,
    );
    fn material_binding(&self) -> dustash::descriptor::DescriptorVecBinding;
}

struct MaterialCarrier<T: Material> {
    updates: HashMap<Handle<T>, GPUMaterialUpdate<T>>,
    removed: Vec<Handle<T>>,
}
enum GPUMaterialUpdate<T: Material> {
    Rebuild(T::BuildSet),
    Update(T::ChangeSet),
}

/// This runs in the main world in the PostUpdate stage.
/// It listens to AssetEvents and creates a GeometryCarrier each frame.
fn generate_changes<A: Material>(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<A>>,
    mut materials: ResMut<Assets<A>>,
    mut generate_builds_param: StaticSystemParam<A::GenerateBuildsParam>,
    mut emit_changes_param: StaticSystemParam<A::EmitChangesParam>,
) {
    // handle, rebuild
    let mut changed_assets: HashMap<Handle<A>, bool> = HashMap::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } => {
                println!("Created a material asset");
                // Always rebuild this asset.
                if let Some(entry) = changed_assets.get_mut(handle) {
                    *entry = true;
                } else {
                    changed_assets.insert(handle.clone(), true);
                }
            }
            AssetEvent::Modified { handle } => {
                // If the asset is gonna be rebuilt, just do that. Otherwise, update only.
                if !changed_assets.contains_key(handle) {
                    changed_assets.insert(handle.clone(), false);
                }
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }
    if changed_assets.len() == 0 && removed.len() == 0 {
        return;
    }

    let mut updates: HashMap<Handle<A>, GPUMaterialUpdate<A>> = HashMap::default();
    for (handle, rebuild) in changed_assets.drain() {
        let material = materials.get_mut_untracked(&handle).unwrap();
        if rebuild {
            updates.insert(
                handle,
                GPUMaterialUpdate::Rebuild(material.generate_builds(&mut generate_builds_param)),
            );
        } else {
            updates.insert(
                handle,
                GPUMaterialUpdate::Update(material.emit_changes(&mut emit_changes_param)),
            );
        }
    }

    // Insert an Option<GeometryCarrier> so that it can be taken.
    commands.insert_resource(Some(MaterialCarrier { updates, removed }));
}

/// This runs in the Extract stage of the Render World.
/// It takes the GeometryCarrier from the App World into the Render World.
fn move_change_set_store_to_render_world<T: Material>(
    mut commands: Commands,
    mut geometery_carrier: ResMut<Option<MaterialCarrier<T>>>,
) {
    if let Some(carrier) = geometery_carrier.take() {
        commands.insert_resource(Some(carrier));
    }
}

struct IndexedGPUMaterial<T: Material> {
    material: T::GPUMaterial,
    descriptor_index: u32,
}
struct GPUMaterialStore<T: Material> {
    gpu_materials: HashMap<Handle<T>, IndexedGPUMaterial<T>>,
    pending_builds: Option<(
        Vec<(Handle<T>, Option<T::GPUMaterial>)>,
        TimelineSemaphoreOp,
    )>,
    buffered_builds: Option<MaterialCarrier<T>>,
}
pub struct GPUMaterialDescriptorVec {
    pub descriptor_vec: dustash::descriptor::DescriptorVec,
}
impl<T: Material> Default for GPUMaterialStore<T> {
    fn default() -> Self {
        Self {
            gpu_materials: HashMap::new(),
            pending_builds: None,
            buffered_builds: None,
        }
    }
}

impl FromWorld for GPUMaterialDescriptorVec {
    fn from_world(world: &mut bevy_ecs::prelude::World) -> Self {
        let device: &Arc<Device> = world.resource();
        Self {
            descriptor_vec: dustash::descriptor::DescriptorVec::new(
                device.clone(),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .unwrap(),
        }
    }
}

/// This runs in the Prepare stage of the Render world.
/// It takes the extracted BuildSet and ChangeSet and apply them to the Material
/// in the render world.
fn prepare_materials<T: Material>(
    mut material_carrier: ResMut<Option<MaterialCarrier<T>>>,
    mut material_store: ResMut<GPUMaterialStore<T>>,
    mut material_descriptor_vec: ResMut<GPUMaterialDescriptorVec>,
    queues: Res<Arc<Queues>>,
    mut build_params: StaticSystemParam<<T::GPUMaterial as GPUMaterial<T>>::BuildParam>,
    mut apply_change_params: StaticSystemParam<
        <T::GPUMaterial as GPUMaterial<T>>::ApplyChangeParam,
    >,
) {
    // Merge the new changes into the buffer. Incoming -> Buffered
    if let Some(buffered_builds) = material_store.buffered_builds.as_mut() {
        if let Some(new_builds) = material_carrier.take() {
            buffered_builds.updates.extend(new_builds.updates);
            buffered_builds.removed.extend(new_builds.removed);
        }
    } else {
        material_store.buffered_builds = material_carrier.take();
    }

    // Pending -> Existing
    if let Some((mut carrier, signal)) = material_store.pending_builds.take() {
        if signal.finished().unwrap() {
            // Finished, put it into the store.
            for (handle, material) in carrier.drain(..) {
                println!("Gotta rebuild BLAS for {:?}", handle);
                if let Some(material) = material {
                    // TODO: Batch this work.
                    let indices = material_descriptor_vec
                        .descriptor_vec
                        .extend(std::iter::once(material.material_binding()))
                        .unwrap();
                    material_store.gpu_materials.insert(
                        handle,
                        IndexedGPUMaterial {
                            material,
                            descriptor_index: indices[0],
                        },
                    );
                }
                // TODO: send rebuild BLAS signal.
            }
        } else {
            // put it back
            material_store.pending_builds = Some((carrier, signal));
            // Has pending work. return early
            return;
        }
    }
    assert!(material_store.pending_builds.is_none());

    // Buffered -> Pending
    if let Some(mut buffered_builds) = material_store.buffered_builds.take() {
        let mut future = dustash::sync::CommandsFuture::new(
            queues.clone(),
            queues.of_type(QueueType::Transfer).index(),
        );
        let mut pending_builds: Vec<(Handle<T>, Option<T::GPUMaterial>)> = Vec::new();
        for handle in buffered_builds.removed.drain(..) {
            material_store.gpu_materials.remove(&handle);
        }
        for (handle, update) in buffered_builds.updates.drain() {
            match update {
                GPUMaterialUpdate::Rebuild(build_set) => {
                    let geometry = <T::GPUMaterial as GPUMaterial<T>>::build(
                        build_set,
                        &mut future,
                        &mut build_params,
                    );
                    pending_builds.push((handle, Some(geometry)));
                }
                GPUMaterialUpdate::Update(change_set) => {
                    material_store
                        .gpu_materials
                        .get_mut(&handle)
                        .unwrap()
                        .material
                        .apply_change_set(change_set, &mut future, &mut apply_change_params);
                    pending_builds.push((handle, None));
                }
            }
        }
        if future.is_empty() {
            // If the future is empty, no commands were recorded. Transition to existing state directly.
            for (handle, material) in pending_builds.drain(..) {
                if let Some(material) = material {
                    // TODO: Batch this work.
                    let indices = material_descriptor_vec
                        .descriptor_vec
                        .extend(std::iter::once(material.material_binding()))
                        .unwrap();
                    material_store.gpu_materials.insert(
                        handle,
                        IndexedGPUMaterial {
                            material,
                            descriptor_index: indices[0],
                        },
                    );
                }
            }
        } else {
            let signal = future
                .stage(vk::PipelineStageFlags2::ALL_COMMANDS)
                .then_signal();
            material_store.pending_builds = Some((pending_builds, signal));
        }
    }
}

/// Insert GPUGeometryPrimitives for all entities with Handle<T>
fn prepare_material_info<T: Material>(
    mut commands: Commands,
    material_store: Res<GPUMaterialStore<T>>,
    mut query: Query<(Entity, &Handle<T>, &mut ExtractedMaterial)>,
) {
    for (entity, material_handle, mut extracted_material) in query.iter_mut() {
        if let Some(material) = material_store.gpu_materials.get(material_handle) {
            extracted_material.material_info = material.descriptor_index;
        } else {
            commands
                .entity(entity)
                .remove::<ExtractedMaterial>()
                .remove::<Handle<T>>();
        }
    }
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
        app.sub_app_mut(crate::RenderApp)
            .add_system_to_stage(
                crate::RenderStage::Extract,
                move |mut commands: Commands, query: Query<(Entity, &Handle<T>)>| {
                    for (entity, material_handle) in query.iter() {
                        commands
                            .get_or_spawn(entity)
                            .insert(ExtractedMaterial {
                                hitgroup_index,
                                material_info: 0,
                            })
                            .insert(material_handle.as_weak::<T>());
                    }
                },
            )
            .init_resource::<GPUMaterialStore<T>>()
            .insert_resource::<Option<MaterialCarrier<T>>>(None)
            .add_system_to_stage(
                crate::RenderStage::Extract,
                move_change_set_store_to_render_world::<T>,
            )
            .add_system_to_stage(crate::RenderStage::Prepare, prepare_materials::<T>)
            .add_system_to_stage(crate::RenderStage::Prepare, prepare_material_info::<T>);
        app.insert_resource::<Option<MaterialCarrier<T>>>(None)
            .add_system_to_stage(bevy_app::CoreStage::PostUpdate, generate_changes::<T>);
        // TODO: maybe run prepare_material_info after prepare_materials to decrease frame delay?
    }
}

#[derive(Component, Clone)]
pub struct ExtractedMaterial {
    pub hitgroup_index: u32,
    /// The material_info in the SBT
    pub material_info: u32,
}
