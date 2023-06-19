use std::{alloc::Layout, future::Future, ops::Deref, sync::Arc};

use ash::{
    prelude::VkResult,
    vk::{self},
};

use crate::{
    shader::{ShaderModule, SpecializedShader},
    utils::send_marker::SendMarker,
    DeferredOperationTaskPool, Device, HasDevice, PipelineCache, PipelineLayout,
};

pub struct RayTracingPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
    sbt_handles: SbtHandles,
}
impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.layout.device().destroy_pipeline(self.pipeline, None);
        }
    }
}
impl RayTracingPipeline {
    pub fn sbt_handles(&self) -> &SbtHandles {
        &self.sbt_handles
    }
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RayTracingHitGroupType {
    Procedural,
    Triangle,
}
impl From<RayTracingHitGroupType> for vk::RayTracingShaderGroupTypeKHR {
    fn from(value: RayTracingHitGroupType) -> Self {
        match value {
            RayTracingHitGroupType::Triangle => {
                vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
            }
            RayTracingHitGroupType::Procedural => {
                vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
            }
        }
    }
}
impl HasDevice for RayTracingPipeline {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}

pub struct RayTracingPipelineLibrary {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
    num_raygen: u32,
    num_raymiss: u32,
    num_callable: u32,
    num_hitgroup: u32,
}
impl Drop for RayTracingPipelineLibrary {
    fn drop(&mut self) {
        unsafe {
            self.layout.device().destroy_pipeline(self.pipeline, None);
        }
    }
}
impl HasDevice for RayTracingPipelineLibrary {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}

#[derive(Clone)]
pub struct RayTracingPipelineLibraryCreateInfo {
    pub pipeline_create_flags: vk::PipelineCreateFlags,
    pub max_pipeline_ray_recursion_depth: u32,
    pub max_pipeline_ray_payload_size: u32,
    pub max_pipeline_ray_hit_attribute_size: u32,
}
impl Default for RayTracingPipelineLibraryCreateInfo {
    fn default() -> Self {
        Self {
            pipeline_create_flags: Default::default(),
            max_pipeline_ray_recursion_depth: 1,
            max_pipeline_ray_payload_size: 4,
            max_pipeline_ray_hit_attribute_size: 4,
        }
    }
}

impl RayTracingPipelineLibrary {
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
    fn create_one_deferred<'a>(
        layout: Arc<PipelineLayout>,
        stages: SendMarker<&'a [vk::PipelineShaderStageCreateInfo]>,
        groups: SendMarker<&'a [vk::RayTracingShaderGroupCreateInfoKHR]>,
        info: &'a RayTracingPipelineLibraryCreateInfo,
        pipeline_cache: Option<&'a PipelineCache>,
        pool: Arc<DeferredOperationTaskPool>,
    ) -> impl Future<Output = Result<Self, vk::Result>> + Send + 'a {
        unsafe {
            async move {
                let library_interface =
                    SendMarker::new(vk::RayTracingPipelineInterfaceCreateInfoKHR {
                        max_pipeline_ray_payload_size: info.max_pipeline_ray_payload_size,
                        max_pipeline_ray_hit_attribute_size: info
                            .max_pipeline_ray_hit_attribute_size,
                        ..Default::default()
                    });
                let info = SendMarker::new(vk::RayTracingPipelineCreateInfoKHR {
                    flags: vk::PipelineCreateFlags::LIBRARY_KHR | info.pipeline_create_flags,
                    stage_count: stages.len() as u32,
                    p_stages: stages.as_ptr(),
                    group_count: groups.len() as u32,
                    p_groups: groups.as_ptr(),
                    max_pipeline_ray_recursion_depth: info.max_pipeline_ray_recursion_depth,
                    p_library_interface: library_interface.deref(),
                    layout: layout.raw(),
                    ..Default::default()
                });
                let mut pipeline = vk::Pipeline::null();
                pool.schedule(|deferred_operation| {
                    (layout
                        .device()
                        .rtx_loader()
                        .fp()
                        .create_ray_tracing_pipelines_khr)(
                        layout.device().handle(),
                        deferred_operation.map(|d| d.raw()).unwrap_or_default(),
                        pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                        1,
                        info.deref(),
                        std::ptr::null(),
                        &mut pipeline,
                    )
                })
                .await?;
                drop(info);
                drop(library_interface);
                drop(stages);
                drop(groups);
                Ok(Self {
                    layout,
                    pipeline,
                    num_raygen: 0,
                    num_raymiss: 0,
                    num_callable: 0,
                    num_hitgroup: 0,
                })
            }
        }
    }
    pub fn create_for_hitgroups<'a, S: Deref<Target = ShaderModule>>(
        layout: Arc<PipelineLayout>,
        hitgroups: impl ExactSizeIterator<
            Item = (
                Option<SpecializedShader<'a, S>>, // rchit
                Option<SpecializedShader<'a, S>>, // rint
                Option<SpecializedShader<'a, S>>, // rahit
                RayTracingHitGroupType,
            ),
        >,
        info: &'a RayTracingPipelineLibraryCreateInfo,
        pipeline_cache: Option<&'a PipelineCache>,
        pool: Arc<DeferredOperationTaskPool>,
    ) -> impl Future<Output = VkResult<Self>> + Send + 'a {
        let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
            Vec::with_capacity(hitgroups.len() * 3);
        let mut specialization_infos: Vec<vk::SpecializationInfo> =
            Vec::with_capacity(hitgroups.len() * 3);
        let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> =
            Vec::with_capacity(hitgroups.len());

        unsafe {
            build_hitgroup_shaders(
                &mut specialization_infos,
                &mut stages,
                &mut groups,
                hitgroups,
            );
            let stages = SendMarker::new(stages);
            let groups = SendMarker::new(groups);
            let specialization_infos = SendMarker::new(specialization_infos);
            async move {
                let stages_ref = SendMarker::new(stages.as_slice());
                let groups_ref = SendMarker::new(groups.as_slice());
                let num_hitgroup = groups_ref.len() as u32;
                let mut result = Self::create_one_deferred(
                    layout,
                    stages_ref,
                    groups_ref,
                    info,
                    pipeline_cache,
                    pool,
                )
                .await?;
                result.num_hitgroup = num_hitgroup;
                drop(specialization_infos); // Make sure specialization info lives across await point
                drop(stages);
                drop(groups);
                Ok(result)
            }
        }
    }
    pub fn create_for_shaders<'a, S: Deref<Target = ShaderModule>>(
        layout: Arc<PipelineLayout>,
        shaders: &'a [SpecializedShader<'a, S>],
        info: &'a RayTracingPipelineLibraryCreateInfo,
        pipeline_cache: Option<&'a PipelineCache>,
        pool: Arc<DeferredOperationTaskPool>,
    ) -> impl Future<Output = VkResult<Self>> + Send + 'a {
        let (num_raygen, num_miss, num_callable) = verify_general_shader_orders(shaders);
        unsafe {
            let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
                Vec::with_capacity(shaders.len());
            let mut specialization_infos: Vec<vk::SpecializationInfo> =
                Vec::with_capacity(shaders.len());
            let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> =
                Vec::with_capacity(shaders.len());
            build_general_shaders(&mut specialization_infos, &mut stages, &mut groups, shaders);

            let stages = SendMarker::new(stages);
            let groups = SendMarker::new(groups);
            let specialization_infos = SendMarker::new(specialization_infos);
            async move {
                let stages_ref = SendMarker::new(stages.as_slice());
                let groups_ref = SendMarker::new(groups.as_slice());
                let mut result = Self::create_one_deferred(
                    layout,
                    stages_ref,
                    groups_ref,
                    info,
                    pipeline_cache,
                    pool,
                )
                .await?;
                drop(specialization_infos);
                drop(stages);
                drop(groups);
                result.num_raygen = num_raygen;
                result.num_raymiss = num_miss;
                result.num_callable = num_callable;
                Ok(result)
            }
        }
    }
}

impl RayTracingPipeline {
    pub fn create_for_shaders<'a, S: Deref<Target = ShaderModule>>(
        layout: Arc<PipelineLayout>,
        shaders: &'a [SpecializedShader<'a, S>],
        hitgroups: impl ExactSizeIterator<
            Item = (
                Option<SpecializedShader<'a, S>>, // rchit
                Option<SpecializedShader<'a, S>>, // rint
                Option<SpecializedShader<'a, S>>, // rahit
                RayTracingHitGroupType,
            ),
        >,
        info: &'a RayTracingPipelineLibraryCreateInfo,
        pipeline_cache: Option<&'a PipelineCache>,
        pool: Arc<DeferredOperationTaskPool>,
    ) -> impl Future<Output = VkResult<Self>> + Send + 'a {
        let (num_raygen, num_miss, num_callable) = verify_general_shader_orders(shaders);
        let num_hitgroups = hitgroups.len() as u32;
        unsafe {
            let mut stages: Vec<vk::PipelineShaderStageCreateInfo> =
                Vec::with_capacity(hitgroups.len() * 3 + shaders.len());
            let mut specialization_infos: Vec<vk::SpecializationInfo> =
                Vec::with_capacity(hitgroups.len() * 3 + shaders.len());
            let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> =
                Vec::with_capacity(hitgroups.len() + shaders.len());
            build_general_shaders(&mut specialization_infos, &mut stages, &mut groups, shaders);
            build_hitgroup_shaders(
                &mut specialization_infos,
                &mut stages,
                &mut groups,
                hitgroups,
            );
            let stages = SendMarker::new(stages);
            let groups = SendMarker::new(groups);
            let specialization_infos = SendMarker::new(specialization_infos);
            let device = layout.device().clone();
            let library_interface = SendMarker::new(vk::RayTracingPipelineInterfaceCreateInfoKHR {
                max_pipeline_ray_payload_size: info.max_pipeline_ray_payload_size,
                max_pipeline_ray_hit_attribute_size: info.max_pipeline_ray_hit_attribute_size,
                ..Default::default()
            });
            async move {
                let stages_ref = SendMarker::new(stages.as_slice());
                let groups_ref = SendMarker::new(groups.as_slice());
                let mut pipeline = vk::Pipeline::null();
                let info = SendMarker::new(vk::RayTracingPipelineCreateInfoKHR {
                    flags: info.pipeline_create_flags,
                    stage_count: stages_ref.len() as u32,
                    p_stages: stages_ref.as_ptr(),
                    group_count: groups_ref.len() as u32,
                    p_groups: groups_ref.as_ptr(),
                    max_pipeline_ray_recursion_depth: info.max_pipeline_ray_recursion_depth,
                    p_library_interface: library_interface.deref(),
                    layout: layout.raw(),
                    ..Default::default()
                });
                pool.schedule(|deferred_operation| {
                    (device.rtx_loader().fp().create_ray_tracing_pipelines_khr)(
                        device.handle(),
                        deferred_operation.map(|a| a.raw()).unwrap_or_default(),
                        pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                        1,
                        info.deref(),
                        std::ptr::null(),
                        &mut pipeline,
                    )
                })
                .await?;
                drop(specialization_infos);
                drop(stages);
                drop(groups);
                drop(info);
                drop(library_interface);

                let sbt_handles = SbtHandles::new(
                    &device,
                    pipeline,
                    num_raygen,
                    num_miss,
                    num_callable,
                    num_hitgroups,
                )?;
                Ok(Self {
                    layout,
                    pipeline,
                    sbt_handles,
                })
            }
        }
    }
    pub fn create_from_libraries<'a>(
        libs: impl Iterator<Item = &'a RayTracingPipelineLibrary>,
        info: &'a RayTracingPipelineLibraryCreateInfo,
        pipeline_cache: Option<&'a PipelineCache>,
        pool: Arc<DeferredOperationTaskPool>,
    ) -> impl Future<Output = VkResult<Self>> + Send + 'a {
        let mut device: Option<Arc<Device>> = None;
        let mut layout: Option<Arc<PipelineLayout>> = None;

        let mut num_raygen: u32 = 0;
        let mut num_raymiss: u32 = 0;
        let mut num_hitgroup: u32 = 0;
        let mut num_callable: u32 = 0;
        let raw_libs: Vec<vk::Pipeline> = libs
            .inspect(|lib| {
                if let Some(device) = device.as_mut() {
                    assert!(Arc::ptr_eq(device, lib.device()));
                } else {
                    device = Some(lib.device().clone());
                }

                if let Some(layout) = layout.as_mut() {
                    assert!(Arc::ptr_eq(layout, &lib.layout));
                } else {
                    layout = Some(lib.layout.clone());
                }

                if lib.num_raygen != 0 {
                    assert!(
                        num_callable == 0 && num_raymiss == 0 && num_hitgroup == 0,
                        "Ray Generation Shader must be specified before everything else"
                    );
                }
                if lib.num_raymiss != 0 {
                    assert!(
                        num_callable == 0 && num_hitgroup == 0,
                        "Miss Shader must be specified before callable and hitgroup shaders"
                    );
                }
                if lib.num_callable != 0 {
                    assert!(
                        num_hitgroup == 0,
                        "Callable Shader must be specified before hitgroup shaders"
                    );
                }
                num_raygen += lib.num_raygen;
                num_raymiss += lib.num_raymiss;
                num_hitgroup += lib.num_hitgroup;
                num_callable += lib.num_callable;
            })
            .map(|lib| lib.raw())
            .collect();
        let Some(device) = device else {
            panic!()
        };
        let layout = layout.unwrap();
        unsafe {
            async move {
                let library_info = SendMarker::new(vk::PipelineLibraryCreateInfoKHR {
                    library_count: raw_libs.len() as u32,
                    p_libraries: raw_libs.as_ptr(),
                    ..Default::default()
                });
                let library_interface =
                    SendMarker::new(vk::RayTracingPipelineInterfaceCreateInfoKHR {
                        max_pipeline_ray_payload_size: info.max_pipeline_ray_payload_size,
                        max_pipeline_ray_hit_attribute_size: info
                            .max_pipeline_ray_hit_attribute_size,
                        ..Default::default()
                    });
                let info = SendMarker::new(vk::RayTracingPipelineCreateInfoKHR {
                    flags: info.pipeline_create_flags,
                    max_pipeline_ray_recursion_depth: info.max_pipeline_ray_recursion_depth,
                    layout: layout.raw(),
                    p_library_info: library_info.deref(),
                    p_library_interface: library_interface.deref(),
                    ..Default::default()
                });
                let mut pipeline: vk::Pipeline = vk::Pipeline::null();
                pool.schedule(|deferred_operation| {
                    let result = (device.rtx_loader().fp().create_ray_tracing_pipelines_khr)(
                        device.handle(),
                        deferred_operation.map(|a| a.raw()).unwrap_or_default(),
                        pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                        1,
                        info.deref(),
                        std::ptr::null(),
                        &mut pipeline,
                    );
                    result
                })
                .await?;
                drop(raw_libs);
                drop(info);
                drop(library_info);
                drop(library_interface);

                let sbt_handles = SbtHandles::new(
                    &device,
                    pipeline,
                    num_raygen,
                    num_raymiss,
                    num_callable,
                    num_hitgroup,
                )?;
                Ok(Self {
                    layout,
                    pipeline,
                    sbt_handles,
                })
            }
        }
    }
}

pub struct SbtHandles {
    data: Box<[u8]>,
    handle_layout: Layout,
    group_base_alignment: u32,
    num_raygen: u32,
    num_miss: u32,
    num_callable: u32,
    num_hitgroup: u32,
}
impl SbtHandles {
    pub fn handle_layout(&self) -> &Layout {
        &self.handle_layout
    }
    fn new(
        device: &Device,
        pipeline: vk::Pipeline,
        num_raygen: u32,
        num_miss: u32,
        num_callable: u32,
        num_hitgroup: u32,
    ) -> VkResult<SbtHandles> {
        let total_num_groups = num_hitgroup + num_miss + num_callable + num_raygen;
        let rtx_properties = &device.physical_device().properties().ray_tracing;
        let sbt_handles_host_vec = unsafe {
            device
                .rtx_loader()
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    total_num_groups,
                    // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
                    // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
                    rtx_properties.shader_group_handle_size as usize * total_num_groups as usize,
                )?
                .into_boxed_slice()
        };
        Ok(SbtHandles {
            data: sbt_handles_host_vec,
            handle_layout: Layout::from_size_align(
                rtx_properties.shader_group_handle_size as usize,
                rtx_properties.shader_group_handle_alignment as usize,
            )
            .unwrap(),
            group_base_alignment: rtx_properties.shader_group_base_alignment,
            num_raygen: num_raygen,
            num_miss: num_miss,
            num_callable: num_callable,
            num_hitgroup: num_hitgroup,
        })
    }

    pub fn rgen(&self, index: usize) -> &[u8] {
        // Note that
        // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
        // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
        // This implies all handles are tightly packed. No need to call `pad_to_align` here
        let start = self.handle_layout.size() * index;
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn rmiss(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size() * (index + self.num_raygen as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn callable(&self, index: usize) -> &[u8] {
        let start =
            self.handle_layout.size() * (index + self.num_raygen as usize + self.num_miss as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn hitgroup(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size()
            * (index
                + self.num_miss as usize
                + self.num_callable as usize
                + self.num_raygen as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
}

fn verify_general_shader_orders<'a, S: Deref<Target = ShaderModule>>(
    shaders: &'a [SpecializedShader<'a, S>],
) -> (u32, u32, u32) {
    let mut num_raygen: u32 = 0;
    let mut num_raymiss: u32 = 0;
    let mut num_callable: u32 = 0;
    for shader in shaders.iter() {
        match shader.stage {
            vk::ShaderStageFlags::RAYGEN_KHR => {
                assert!(
                    num_callable == 0 && num_raymiss == 0,
                    "Ray Generation Shader must be specified before everything else"
                );
                num_raygen += 1;
            }
            vk::ShaderStageFlags::MISS_KHR => {
                assert!(
                    num_callable == 0,
                    "Miss Shader must be specified before Callable shaders"
                );
                num_raymiss += 1;
            }
            vk::ShaderStageFlags::CALLABLE_KHR => {
                num_callable += 1;
            }
            vk::ShaderStageFlags::ANY_HIT_KHR
            | vk::ShaderStageFlags::CLOSEST_HIT_KHR
            | vk::ShaderStageFlags::INTERSECTION_KHR => {
                panic!("For hitgroup shaders, use `RayTracingPipelineLibrary::create_for_hitgroup`")
            }
            _ => {
                panic!()
            }
        }
    }
    (num_raygen, num_raymiss, num_callable)
}

unsafe fn build_general_shaders<'a, S: Deref<Target = ShaderModule>>(
    specialization_infos: &mut Vec<vk::SpecializationInfo>,
    stages: &mut Vec<vk::PipelineShaderStageCreateInfo>,
    groups: &mut Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    shaders: &'a [SpecializedShader<'a, S>],
) {
    let specialization_infos_ptr = specialization_infos.as_ptr();
    specialization_infos.extend(
        shaders
            .iter()
            .map(|shader| shader.specialization_info.raw_info()),
    );

    for (i, shader) in shaders.iter().enumerate() {
        stages.push(vk::PipelineShaderStageCreateInfo {
            flags: shader.flags,
            stage: shader.stage,
            module: shader.shader.raw(),
            p_name: shader.entry_point.as_ptr(),
            p_specialization_info: specialization_infos_ptr.add(i),
            ..Default::default()
        });
        groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
            general_shader: i as u32,
            closest_hit_shader: vk::SHADER_UNUSED_KHR,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        });
    }
}

fn build_hitgroup_shaders<'a, S: Deref<Target = ShaderModule>>(
    specialization_info: &mut Vec<vk::SpecializationInfo>,
    stages: &mut Vec<vk::PipelineShaderStageCreateInfo>,
    groups: &mut Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    hitgroups: impl ExactSizeIterator<
        Item = (
            Option<SpecializedShader<'a, S>>, // rchit
            Option<SpecializedShader<'a, S>>, // rint
            Option<SpecializedShader<'a, S>>, // rahit
            RayTracingHitGroupType,
        ),
    >,
) {
    let initial_capacity = (
        stages.capacity(),
        groups.capacity(),
        specialization_info.capacity(),
    );

    for (rchit, rint, rahit, ty) in hitgroups {
        let mut rchit_stage: u32 = vk::SHADER_UNUSED_KHR;
        let mut rint_stage: u32 = vk::SHADER_UNUSED_KHR;
        let mut rahit_stage: u32 = vk::SHADER_UNUSED_KHR;
        if let Some(shader) = rchit.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::CLOSEST_HIT_KHR);
            rchit_stage = stages.len() as u32;

            let p_specialization_info = specialization_info.as_ptr_range().end;
            specialization_info.push(unsafe { shader.specialization_info.raw_info() });
            stages.push(vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                module: shader.shader.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info,
                ..Default::default()
            });
        }

        if let Some(shader) = rint.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::INTERSECTION_KHR);
            rint_stage = stages.len() as u32;

            let p_specialization_info = specialization_info.as_ptr_range().end;
            specialization_info.push(unsafe { shader.specialization_info.raw_info() });
            stages.push(vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: vk::ShaderStageFlags::INTERSECTION_KHR,
                module: shader.shader.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info,
                ..Default::default()
            });
        }
        if let Some(shader) = rahit.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::ANY_HIT_KHR);
            rahit_stage = stages.len() as u32;

            let p_specialization_info = specialization_info.as_ptr_range().end;
            specialization_info.push(unsafe { shader.specialization_info.raw_info() });
            stages.push(vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: vk::ShaderStageFlags::ANY_HIT_KHR,
                module: shader.shader.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info,
                ..Default::default()
            });
        }
        groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
            ty: ty.into(),
            closest_hit_shader: rchit_stage,
            any_hit_shader: rahit_stage,
            intersection_shader: rint_stage,
            general_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        })
    }

    let end_capacity = (
        stages.capacity(),
        groups.capacity(),
        specialization_info.capacity(),
    );
    assert_eq!(initial_capacity, end_capacity);
}
