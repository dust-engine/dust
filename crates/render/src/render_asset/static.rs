use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use bevy_app::{App, Plugin};
use bevy_asset::{AddAsset, Asset, AssetEvent, Assets, Handle};

use bevy_ecs::event::{EventReader, EventWriter};

use bevy_ecs::schedule::ParallelSystemDescriptorCoercion;
use bevy_ecs::schedule::SystemLabel;
use bevy_ecs::system::{Commands, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem};
use bevy_utils::{HashMap, HashSet};
use dustash::queue::semaphore::TimelineSemaphoreOp;
use dustash::queue::{QueueType, Queues};
use dustash::sync::{CommandsFuture, GPUFuture};

/// Asset that can be send to the GPU.
pub trait RenderAsset: Asset + Sized + 'static {
    /// The geometry represented as an array of primitives
    /// This gets persisted in the render world.
    /// This is a GPU state.
    type GPUAsset: GPURenderAsset<Self>;

    /// Data needed to send this asset to the GPU.
    /// This is usually a GPU Resource such as a staging MemBuffer,
    /// or in the case of an integrated GPU, a DEVICE_VISIBLE MemBuffer.
    type BuildData: Send + Sync;

    type CreateBuildDataParam: SystemParam;

    /// Create build data by either copying data into the staging buffer
    /// or moving out the staging buffer.
    /// This is executed right after the asset was created.
    /// A mutable self reference is passed in so that the implementation
    /// can choose to delete the original buffer or moving out from self,
    /// if the data is supposed to be consumed by the GPU only.
    fn create_build_data(
        &mut self,
        param: &mut SystemParamItem<Self::CreateBuildDataParam>,
    ) -> Self::BuildData;
}

pub enum GPURenderAssetBuildResult<T: RenderAsset> {
    Success(T::GPUAsset),
    MissingDependency(T::BuildData)
}

/// The render asset on the GPU.
pub trait GPURenderAsset<T: RenderAsset>: Send + Sync + 'static + Sized {
    type BuildParam: SystemParam;

    /// Return None if not ready to build yet. The render asset system will attempt the build again
    /// at a later time, potentially in the next frame.
    fn build(
        build_set: T::BuildData,
        commands_future: &mut CommandsFuture,
        params: &mut SystemParamItem<Self::BuildParam>,
    ) -> GPURenderAssetBuildResult<T>;
}

struct ExtractedAssets<A: RenderAsset> {
    extracted: Vec<(Handle<A>, A::BuildData)>,
    removed: Vec<Handle<A>>,
}
impl<A: RenderAsset> ExtractedAssets<A> {
    pub fn merge(&mut self, other: Self) {
        self.extracted.extend(other.extracted);
        self.removed.extend(other.removed);
    }
}

/// This system calls the `create_build_data` method whenever a new render asset was created.
/// This system runs in PostUpdate in the app world.
fn create_build_data<A: RenderAsset>(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<A>>,
    mut assets: ResMut<Assets<A>>,
    mut params: StaticSystemParam<A::CreateBuildDataParam>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                println!("Create build data {}", std::any::type_name::<A>());
                changed_assets.insert(handle);
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }

    let mut extracted_assets = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(asset) = assets.get_mut_untracked(handle) {
            extracted_assets.push((handle.clone_weak(), asset.create_build_data(&mut params)));
        }
    }

    commands.insert_resource(Some(ExtractedAssets {
        extracted: extracted_assets,
        removed,
    }));
}

/// This runs in the Extract stage of the Render World.
/// It takes the GeometryCarrier from the App World into the Render World.
fn move_extracted_assets<A: RenderAsset>(
    mut commands: Commands,
    mut extracted_assets: ResMut<Option<ExtractedAssets<A>>>,
) {
    if let Some(carrier) = extracted_assets.take() {
        commands.insert_resource(Some(carrier));
    }
}

pub struct RenderAssetStore<A: RenderAsset> {
    /// Assets that were already built.
    assets: HashMap<Handle<A>, A::GPUAsset>,

    /// Assets that were submitted to GPU but still pending.
    pending_builds: Option<(Vec<(Handle<A>, A::GPUAsset)>, TimelineSemaphoreOp)>,

    /// Assets deferred to the next frame and not yet submitted to GPU.
    buffered_builds: Option<ExtractedAssets<A>>,
}
impl<A: RenderAsset> RenderAssetStore<A> {
    pub fn get(&self, handle: &Handle<A>) -> Option<&A::GPUAsset> {
        self.assets.get(handle)
    }
    pub fn get_mut(&mut self, handle: &Handle<A>) -> Option<&mut A::GPUAsset> {
        self.assets.get_mut(handle)
    }
}

impl<A: RenderAsset> Default for RenderAssetStore<A> {
    fn default() -> Self {
        Self {
            assets: HashMap::new(),
            pending_builds: None,
            buffered_builds: None,
        }
    }
}

#[derive(SystemLabel)] // TODO: Simplify this
#[system_label(ignore_fields)]
pub struct PrepareRenderAssetsSystem<T: RenderAsset> {
    _marker: PhantomData<T>,
}
impl<T: RenderAsset> Default for PrepareRenderAssetsSystem<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<T: RenderAsset> Clone for PrepareRenderAssetsSystem<T> {
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<T: RenderAsset> PartialEq for PrepareRenderAssetsSystem<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<T: RenderAsset> Eq for PrepareRenderAssetsSystem<T> {}
impl<T: RenderAsset> std::hash::Hash for PrepareRenderAssetsSystem<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ty = std::any::TypeId::of::<PrepareRenderAssetsSystem<T>>();
        ty.hash(state);
    }
}
impl<T: RenderAsset> Debug for PrepareRenderAssetsSystem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrepareRenderAssetsSystem")
    }
}

/// This runs in the Prepare stage of the Render world.
/// It takes the extracted BuildSet and ChangeSet and apply them to the Geometry
/// in the render world.
fn prepare_render_assets<T: RenderAsset>(
    mut extracted_assets: ResMut<Option<ExtractedAssets<T>>>,
    mut render_asset_store: ResMut<RenderAssetStore<T>>,
    queues: Res<Arc<Queues>>,
    mut event_writer: EventWriter<RenderAssetEvent<T>>,
    mut build_params: StaticSystemParam<<T::GPUAsset as GPURenderAsset<T>>::BuildParam>,
) {
    // Merge the new changes into the buffer. Incoming -> Buffered
    if let Some(buffered_builds) = render_asset_store.buffered_builds.as_mut() {
        if let Some(new_builds) = extracted_assets.take() {
            buffered_builds.merge(new_builds);
        }
    } else {
        render_asset_store.buffered_builds = extracted_assets.take();
    }

    // Pending -> Existing
    if let Some((mut carrier, signal)) = render_asset_store.pending_builds.take() {
        if signal.finished().unwrap() {
            // Finished, put it into the store.
            for (handle, gpu_asset) in carrier.drain(..) {
                event_writer.send(RenderAssetEvent::Created(handle.clone_weak()));
                render_asset_store.assets.insert(handle, gpu_asset);
                // TODO: send signal.
            }
        } else {
            // Has pending work. return early
            // put it back
            render_asset_store.pending_builds = Some((carrier, signal));
            return;
        }
    }
    assert!(render_asset_store.pending_builds.is_none());

    // Buffered -> Pending
    if let Some(mut buffered_builds) = render_asset_store.buffered_builds.take() {
        let mut future = dustash::sync::CommandsFuture::new(
            queues.clone(),
            queues.of_type(QueueType::Transfer).index(),
        );
        let mut pending_builds: Vec<(Handle<T>, T::GPUAsset)> = Vec::new();
        for handle in buffered_builds.removed.drain(..) {
            render_asset_store.assets.remove(&handle);
        }

        let mut rebuffered_builds: Vec<(Handle<T>, T::BuildData)> = Vec::new();
        for (handle, update) in buffered_builds.extracted.drain(..) {
            match
                <T::GPUAsset as GPURenderAsset<T>>::build(update, &mut future, &mut build_params) {
                GPURenderAssetBuildResult::Success(gpu_asset) => {
                    pending_builds.push((handle, gpu_asset));
                },
                GPURenderAssetBuildResult::MissingDependency(build_data) => {
                    rebuffered_builds.push((handle, build_data));
                }
            }
        }
        if rebuffered_builds.len() > 0 {
            render_asset_store.buffered_builds = Some(ExtractedAssets {
                extracted: rebuffered_builds,
                removed: Vec::new(),
            });
        }

        if future.is_empty() {
            // If the future is empty, no commands were recorded. Transition to existing state directly.
            for (handle, gpu_asset) in pending_builds.drain(..) {
                event_writer.send(RenderAssetEvent::Created(handle.clone_weak()));
                render_asset_store.assets.insert(handle, gpu_asset);
            }
        } else {
            let signal = future
                .stage(vk::PipelineStageFlags2::ALL_COMMANDS)
                .then_signal();
            render_asset_store.pending_builds = Some((pending_builds, signal));
        }
    }
}

pub struct RenderAssetPlugin<T: RenderAsset> {
    _marker: PhantomData<T>,
}
impl<T: RenderAsset> Default for RenderAssetPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

/// Plugin should be added to the main world.
impl<T: RenderAsset> Plugin for RenderAssetPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_asset::<T>()
            .add_system_to_stage(bevy_app::CoreStage::PostUpdate, create_build_data::<T>);
        let render_app = app.sub_app_mut(crate::RenderApp);
        render_app
            .init_resource::<RenderAssetStore<T>>()
            .add_event::<RenderAssetEvent<T>>()
            .add_system_to_stage(crate::RenderStage::Extract, move_extracted_assets::<T>)
            .add_system_to_stage(
                crate::RenderStage::Prepare,
                prepare_render_assets::<T>.label(PrepareRenderAssetsSystem::<T>::default()),
            );
    }
}

pub enum RenderAssetEvent<T: RenderAsset> {
    Created(Handle<T>),
}
