use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::{
    accel_struct::blas::BlasComponent,
    shader::{Shader, SpecializedShader},
    Allocator, RenderApp, RenderStage,
};
use crate::{Device, Queues, RayTracingLoader};
use bevy_app::Plugin;
use bevy_asset::{AssetEvent, AssetServer, Assets, Handle};
use bevy_ecs::{
    prelude::*,
    system::{StaticSystemParam, SystemParam, SystemParamItem},
};
use bevy_utils::{HashMap, HashSet};
use dustash::{
    queue::QueueType,
    ray_tracing::{
        pipeline::{PipelineLayout, RayTracingPipelineLayout},
        sbt::{Sbt, SbtLayout},
    },
    sync::CommandsFuture,
};

pub use dustash::ray_tracing::sbt::HitGroupType;
pub struct HitGroup {
    pub ty: HitGroupType,
    pub intersection_shader: Option<SpecializedShader>,
    pub anyhit_shader: Option<SpecializedShader>,
    pub closest_hit_shader: Option<SpecializedShader>,
}
impl HitGroup {
    fn try_extract_shaders(
        &self,
        device: &Arc<dustash::Device>,
        shaders: &Assets<Shader>,
    ) -> Option<dustash::ray_tracing::sbt::HitGroup> {
        use dustash::shader::SpecializedShader as SpecializedShaderModule;

        let build_shader = |specialized_shader: &SpecializedShader| {
            let raygen_shader = shaders.get(&specialized_shader.shader)?;
            Some(SpecializedShaderModule {
                shader: Arc::new(raygen_shader.create(device.clone())),
                specialization: specialized_shader.specialization.clone(),
            })
        };
        let hit_group = dustash::ray_tracing::sbt::HitGroup {
            ty: self.ty,
            intersection_shader: if let Some(shader) = self.intersection_shader.as_ref() {
                Some(build_shader(shader)?)
            } else {
                None
            },
            anyhit_shader: if let Some(shader) = self.anyhit_shader.as_ref() {
                Some(build_shader(shader)?)
            } else {
                None
            },
            closest_hit_shader: if let Some(shader) = self.closest_hit_shader.as_ref() {
                Some(build_shader(shader)?)
            } else {
                None
            },
        };
        Some(hit_group)
    }
}

pub struct RayTracingPipelineBuildJob {
    pub pipeline_layout: Arc<PipelineLayout>,
    pub raygen_shader: SpecializedShader,
    pub miss_shaders: Vec<SpecializedShader>,
    pub callable_shaders: Vec<SpecializedShader>,
    pub max_recursion_depth: u32,
}
impl RayTracingPipelineBuildJob {
    pub fn try_create_sbt_layout(
        &self,
        device: &Arc<dustash::Device>,
        shaders: &Assets<Shader>,
        hitgroups: &[dustash::ray_tracing::sbt::HitGroup],
    ) -> Option<SbtLayout> {
        use dustash::shader::SpecializedShader as SpecializedShaderModule;
        let build_shader = |specialized_shader: &SpecializedShader| {
            let raygen_shader = shaders.get(&specialized_shader.shader)?;
            Some(SpecializedShaderModule {
                shader: Arc::new(raygen_shader.create(device.clone())),
                specialization: specialized_shader.specialization.clone(),
            })
        };
        let raygen_shader = build_shader(&self.raygen_shader)?;
        let miss_shaders: Option<Box<[SpecializedShaderModule]>> =
            self.miss_shaders.iter().map(build_shader).collect();
        let miss_shaders = miss_shaders?;
        let callable_shaders: Option<Box<[SpecializedShaderModule]>> =
            self.callable_shaders.iter().map(build_shader).collect();
        let callable_shaders = callable_shaders?;
        let layout = SbtLayout::new(raygen_shader, miss_shaders, callable_shaders, hitgroups);
        Some(layout)
    }
}
pub trait RayTracingPipeline {
    fn max_recursion_depth(&self) -> u32;
    fn pipeline_layout(&self) -> Arc<PipelineLayout>;
    fn raygen_shader(&self, asset_server: &AssetServer) -> SpecializedShader;
    fn miss_shaders(&self, asset_server: &AssetServer) -> Vec<SpecializedShader>;
    fn callable_shaders(&self, asset_server: &AssetServer) -> Vec<SpecializedShader>;
    fn build(&self, asset_server: &AssetServer) -> RayTracingPipelineBuildJob {
        RayTracingPipelineBuildJob {
            pipeline_layout: self.pipeline_layout(),
            raygen_shader: self.raygen_shader(asset_server),
            miss_shaders: self.miss_shaders(asset_server),
            callable_shaders: self.callable_shaders(asset_server),
            max_recursion_depth: self.max_recursion_depth(),
        }
    }
}

pub trait RayTracingRenderer: Clone + Send + Sync + 'static + Resource {
    fn new(app: &mut bevy_app::App) -> Self;
    // Build the pipeline by calling self.add_pipeline()
    fn build(&self, index: PipelineIndex, asset_server: &AssetServer)
        -> RayTracingPipelineBuildJob;
    fn all_pipelines(&self) -> &[PipelineIndex];

    type RenderParam: SystemParam;
    fn render(&self, param: &mut SystemParamItem<Self::RenderParam>);
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct PipelineIndex(usize);
impl PipelineIndex {
    pub const fn new(index: usize) -> Self {
        PipelineIndex(index)
    }
}

#[derive(Resource, Default)]
pub struct HitGroups(Vec<HitGroup>);
impl Deref for HitGroups {
    type Target = Vec<HitGroup>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for HitGroups {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct RayTracingRendererPlugin<T: RayTracingRenderer> {
    _marker: PhantomData<T>,
}
impl<T: RayTracingRenderer> Default for RayTracingRendererPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: RayTracingRenderer> Plugin for RayTracingRendererPlugin<T> {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<HitGroups>();
        app.sub_app_mut(RenderApp)
            .init_resource::<ExtractedRayTracingPipelineLayoutsContainer>()
            .init_resource::<PipelineCache>()
            .init_resource::<crate::render_asset::BindlessGPUAssetDescriptors>()
            .add_system_to_stage(RenderStage::Extract, extract_pipeline_system::<T>)
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_pipeline_system::<T>.label(PreparePipelineSystem),
            )
            .add_system_to_stage(
                RenderStage::Render,
                prepare_sbt_system.label(PrepareSbtSystem),
            )
            .add_system_to_stage(
                RenderStage::Render,
                render_system::<T>
                    .label(RenderSystem)
                    .after(PrepareSbtSystem),
            );
        let renderer = T::new(app);
        app.insert_resource(renderer);
        // First, get the SBT layout
        // Then, create_many raytracing pipeilnes
        // Finally, use those pipelines to create SBTs
    }
}

pub struct ExtractedRayTracingPipelineLayout {
    index: PipelineIndex,
    max_recursion_depth: u32,
    pipeline_layout: Arc<PipelineLayout>,
    sbt_layout: SbtLayout,
}

#[derive(Default)]
pub struct PipelineShaders {
    /// In which pipelines were the shader actually used
    ray_shaders: HashMap<Handle<Shader>, HashSet<PipelineIndex>>,
    /// HitGroup Shaders are always used by all pipelines
    hitgroup_shaders: HashSet<Handle<Shader>>,

    /// Pipelines waiting to be built.
    /// We keep a list of RayTracingPipelineBuildJob here to retain a reference to the shaders,
    /// so that they don't get unloaded in unexpected ways
    queued_pipelines: HashMap<PipelineIndex, RayTracingPipelineBuildJob>,
}

fn extract_pipeline_system<T: RayTracingRenderer>(
    mut commands: Commands,
    mut events: EventReader<AssetEvent<Shader>>,
    mut pipeline_shaders: Local<PipelineShaders>,
    asset_server: Res<AssetServer>,
    device: Res<Device>,
    shaders: Res<Assets<Shader>>,
    hit_groups: Res<HitGroups>,
    renderer: Res<T>,
) {
    commands.insert_resource(renderer.clone());
    let pipeline_shaders = &mut *pipeline_shaders;
    // TODO: Preferably renderer and device resources should live in the render world,
    // since renderer contains PipelineLayout which is a render resource.
    // This would be blocked on bevy's asset rework.
    if renderer.is_changed() {
        pipeline_shaders.queued_pipelines = renderer
            .all_pipelines()
            .iter()
            .map(|index| {
                let job = renderer.build(*index, &asset_server);
                (*index, job)
            })
            .collect();
        let mut ray_shaders: HashMap<Handle<Shader>, HashSet<PipelineIndex>> = HashMap::new();
        for (index, job) in pipeline_shaders.queued_pipelines.iter() {
            for shader in job
                .miss_shaders
                .iter()
                .chain(job.callable_shaders.iter())
                .chain(std::iter::once(&job.raygen_shader))
            {
                ray_shaders
                    .entry(shader.shader.clone_weak())
                    .or_insert(HashSet::new())
                    .insert(*index);
            }
        }
        pipeline_shaders.ray_shaders = ray_shaders;
    }
    if hit_groups.is_changed() {
        // Recreate hitgroup shaders
        pipeline_shaders.hitgroup_shaders = hit_groups
            .iter()
            .flat_map(|hitgroup| {
                [
                    hitgroup.closest_hit_shader.as_ref(),
                    hitgroup.intersection_shader.as_ref(),
                    hitgroup.anyhit_shader.as_ref(),
                ]
                .into_iter()
                .filter_map(|item| item.map(|item| item.shader.clone_weak()))
            })
            .collect();
    }

    // Pipelines that should be considered for recreation due to asset events.
    let mut potentially_ready_pipelines: HashSet<PipelineIndex> = HashSet::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                if let Some(ids) = pipeline_shaders.ray_shaders.get(handle) {
                    potentially_ready_pipelines.extend(ids);
                    // Rebuild impacted pipelines
                    pipeline_shaders
                        .queued_pipelines
                        .extend(ids.iter().map(|index| {
                            let job = renderer.build(*index, &asset_server);
                            (*index, job)
                        }));
                } else if pipeline_shaders.hitgroup_shaders.contains(handle) {
                    potentially_ready_pipelines.extend(renderer.all_pipelines().iter());
                    // Rebuild all pipelines
                    pipeline_shaders.queued_pipelines = renderer
                        .all_pipelines()
                        .iter()
                        .map(|index| {
                            let job = renderer.build(*index, &asset_server);
                            (*index, job)
                        })
                        .collect();
                }
            }
            AssetEvent::Removed { handle: _ } => {}
        }
    }

    if potentially_ready_pipelines.is_empty() {
        return;
    }

    let hitgroups: Option<Vec<dustash::ray_tracing::sbt::HitGroup>> = hit_groups
        .iter()
        .map(|hitgroup| hitgroup.try_extract_shaders(&device, &shaders))
        .collect();
    if hitgroups.is_none() {
        return;
    }
    let hitgroups = hitgroups.unwrap();
    // Actually recreate the pipelines.
    let layouts: Vec<_> = potentially_ready_pipelines
        .iter()
        .flat_map(|pipeline_index| {
            let job = pipeline_shaders
                .queued_pipelines
                .get(pipeline_index)
                .unwrap();
            // If we can create sbt layout, record the created sbt layout.
            if let Some(sbt_layout) = job.try_create_sbt_layout(&device, &shaders, &hitgroups) {
                let job = pipeline_shaders
                    .queued_pipelines
                    .remove(pipeline_index)
                    .unwrap();
                Some(ExtractedRayTracingPipelineLayout {
                    index: *pipeline_index,
                    max_recursion_depth: job.max_recursion_depth,
                    pipeline_layout: job.pipeline_layout,
                    sbt_layout,
                })
            } else {
                None
            }
            // If we can't create sbt layout, there are shaders that aren't loaded yet. Try again later.
        })
        .collect();
    // These are the pipelines that we would like to rebuild this frame.
    commands.insert_resource(ExtractedRayTracingPipelineLayoutsContainer(Some(layouts)));
}

#[derive(Default, Resource)]
pub struct PipelineCache {
    pub generation: u64,
    pub pipelines: Vec<Option<Arc<dustash::ray_tracing::pipeline::RayTracingPipeline>>>,
    pub sbts: Vec<Option<Sbt>>,
    pub sbt_upload_future: Option<CommandsFuture>,
}

#[derive(Clone, Hash, Debug, Eq, PartialEq, SystemLabel)]
struct PreparePipelineSystem;

#[derive(Resource, Default)]
struct ExtractedRayTracingPipelineLayoutsContainer(Option<Vec<ExtractedRayTracingPipelineLayout>>);
impl Deref for ExtractedRayTracingPipelineLayoutsContainer {
    type Target = Option<Vec<ExtractedRayTracingPipelineLayout>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for ExtractedRayTracingPipelineLayoutsContainer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn prepare_pipeline_system<T: RayTracingRenderer>(
    mut layouts: ResMut<ExtractedRayTracingPipelineLayoutsContainer>,
    mut pipeline_cache: ResMut<PipelineCache>,
    rtx_loader: Res<RayTracingLoader>,
) {
    if layouts.is_none() {
        return;
    }
    let layouts = layouts.take().unwrap();
    if layouts.len() == 0 {
        return;
    }
    let pipelines = {
        let layouts: Vec<_> = layouts
            .iter()
            .map(|layout| RayTracingPipelineLayout {
                pipeline_layout: &layout.pipeline_layout,
                sbt_layout: &layout.sbt_layout,
                max_recursion_depth: layout.max_recursion_depth,
            })
            .collect();

        dustash::ray_tracing::pipeline::RayTracingPipeline::create_many(
            rtx_loader.clone(),
            layouts.as_slice(),
        )
    }
    .unwrap();

    let num_pipelines = layouts.iter().map(|layout| layout.index.0).max().unwrap() + 1;
    if num_pipelines > pipeline_cache.pipelines.len() {
        let len = pipeline_cache.pipelines.len();
        // Ensure that pipeline_cache.pipelines is large enough
        pipeline_cache
            .pipelines
            .extend(std::iter::repeat_with(|| None).take(num_pipelines - len));
    }

    let mut pipelines_updated: bool = false;
    layouts
        .iter()
        .zip(pipelines.into_iter())
        .for_each(|(layout, pipeline)| {
            // Record the render pipeline.
            pipeline_cache.pipelines[layout.index.0] = Some(Arc::new(pipeline));
            pipelines_updated = true;
            println!(
                "Created a render pipeline {}",
                pipeline_cache.pipelines.len()
            );
        });
    if pipelines_updated {
        pipeline_cache.generation += 1;
    }
}

#[derive(Clone, Hash, Debug, Eq, PartialEq, SystemLabel)]
struct PrepareSbtSystem;

fn prepare_sbt_system(
    mut pipeline_cache: ResMut<PipelineCache>,
    allocator: Res<Allocator>,
    queues: Res<Queues>,
    query: Query<&BlasComponent>,
) {
    // TODO: skip creating the SBT when there's no change
    // We always create a new SBT when there's a change. This is to avoid mutation while the SBT was being
    // read by GPU.
    let pipeline_cache = &mut *pipeline_cache;
    pipeline_cache.sbts = Vec::with_capacity(pipeline_cache.pipelines.len());
    pipeline_cache
        .sbts
        .extend(std::iter::repeat_with(|| None).take(pipeline_cache.pipelines.len()));
    let mut commands_future =
        CommandsFuture::new(queues.clone(), queues.of_type(QueueType::Transfer).index());
    for (index, pipeline) in
        pipeline_cache
            .pipelines
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                if let Some(pipeline) = value {
                    Some((index, pipeline))
                } else {
                    None
                }
            })
    {
        let rhit_data: Vec<_> = query
            .iter()
            .flat_map(|blas| {
                blas.geometry_materials.iter().map(|geometry_material| {
                    (
                        geometry_material.hitgroup_index as usize,
                        geometry_material.sbt_data.unwrap(),
                    )
                })
            })
            .collect();
        let sbt = Sbt::new(
            pipeline.clone(),
            (),
            std::iter::once(()),
            std::iter::once(()),
            rhit_data,
            &allocator,
            &mut commands_future,
        );
        pipeline_cache.sbts[index] = Some(sbt);
    }
    pipeline_cache.sbt_upload_future = Some(commands_future);
}

#[derive(SystemLabel, Hash, Clone, PartialEq, Eq, Debug)]
pub struct RenderSystem;
fn render_system<T: RayTracingRenderer>(
    renderer: Res<T>,
    mut param: StaticSystemParam<T::RenderParam>,
) {
    renderer.render(&mut param);
}
