use std::{alloc::Layout, collections::HashMap, marker::PhantomData, ops::Range, sync::Arc};

use bevy_ecs::{prelude::Component, system::SystemParamItem};
use crevice::std430::{AsStd430, Std430};

use rhyolite::{
    ash::vk,
    copy_buffer,
    future::{
        use_per_frame_state, use_shared_state, GPUCommandFuture, PerFrameContainer, PerFrameState,
        RenderData, RenderRes, SharedDeviceState, SharedDeviceStateHostContainer,
    },
    macros::commands,
    utils::either::Either,
    Allocator, BufferLike, HasDevice, ManagedBufferUnsized, PhysicalDeviceMemoryModel,
    ResidentBuffer,
};

use crate::{
    Material, RayTracingPipeline, RayTracingPipelineCharacteristics,
    RayTracingPipelineManagerSpecializedPipeline, Renderable,
};

#[derive(Clone, Copy)]
pub struct EmptyShaderRecords;
unsafe impl crevice::internal::bytemuck::Zeroable for EmptyShaderRecords {}
unsafe impl crevice::internal::bytemuck::Pod for EmptyShaderRecords {}
unsafe impl Std430 for EmptyShaderRecords {
    const ALIGNMENT: usize = 0;
}

// This is to be included on the component of entities.
#[derive(Component)]
pub struct SbtIndex<M = Renderable> {
    index: u32,
    _marker: PhantomData<M>,
}
impl<M> SbtIndex<M> {
    pub fn get_index(&self) -> u32 {
        self.index
    }
}

impl<M> Clone for SbtIndex<M> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            _marker: PhantomData,
        }
    }
}
impl<M> Copy for SbtIndex<M> {}
impl<M> PartialEq for SbtIndex<M> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<M> Eq for SbtIndex<M> {}

struct SbtLayout {
    /// The layout for one raytype.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--              size           -> | align   |
    one_raytype: Layout,

    // The layout for one entry with all its raytypes
    /// | Raytype 1                                    | Raytype 2                                    |
    /// | shader_handles | inline_parameters | padding | shader_handles | inline_parameters | padding |
    /// | <---                                      size                               ---> |  align  |
    one_entry: Layout,

    /// The size of the shader group handles, padded.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--- size ---> |
    handle_size: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Entry {
    material_id: std::any::TypeId,
    data: Box<[u8]>, // TODO: can we get rid of this Box?
}

pub struct SbtManager {
    allocator: Allocator,
    layout: SbtLayout,
    total_raytype: u32,
    buffer: ManagedBufferUnsized,

    /// Mapping from SBT Entry to index
    entries: HashMap<Entry, u32>,
    raytype_pipeline_handles: Vec<vk::Pipeline>,

    update_list: Vec<Entry>,
}
impl HasDevice for SbtManager {
    fn device(&self) -> &Arc<rhyolite::Device> {
        self.allocator.device()
    }
}

impl SbtManager {
    pub fn new(
        allocator: rhyolite_bevy::Allocator,
        pipeline_characteristics: &RayTracingPipelineCharacteristics,
    ) -> Self {
        let rtx_properties = allocator
            .device()
            .physical_device()
            .properties()
            .ray_tracing;
        let handle_layout = Layout::from_size_align(
            rtx_properties.shader_group_handle_size as usize,
            rtx_properties.shader_group_handle_alignment as usize,
        )
        .unwrap();
        let one_raytype = handle_layout
            .extend(pipeline_characteristics.sbt_param_layout)
            .unwrap()
            .0;
        let one_entry = one_raytype
            .repeat(pipeline_characteristics.num_raytype as usize)
            .unwrap()
            .0;
        let layout = SbtLayout {
            one_raytype,
            one_entry,
            handle_size: rtx_properties.shader_group_handle_size as usize,
        };
        Self {
            allocator: allocator.clone().into_inner(),
            total_raytype: pipeline_characteristics.num_raytype,
            layout,
            buffer: ManagedBufferUnsized::new(
                allocator.into_inner(),
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                one_raytype,
                rtx_properties.shader_group_base_alignment as usize,
            ),
            entries: Default::default(),
            raytype_pipeline_handles: vec![
                vk::Pipeline::null();
                pipeline_characteristics.num_raytype as usize
            ],
            update_list: Default::default(),
        }
    }
    /// Specify that raytype is using the specialized pipeline for rendering
    pub fn specify_pipelines(
        &mut self,
        pipelines: &[RayTracingPipelineManagerSpecializedPipeline],
    ) {
        assert_eq!(pipelines.len() as u32, self.total_raytype);
        let mut buffer: Box<[u8]> =
            vec![0; self.layout.one_raytype.pad_to_align().size()].into_boxed_slice();

        for (raytype, pipeline) in pipelines.iter().enumerate() {
            let raytype = raytype as u32;
            if self.raytype_pipeline_handles[raytype as usize] == pipeline.pipeline().raw() {
                for entry in self.update_list.iter() {
                    let index = self.entries.get(entry).unwrap();
                    let a = pipeline.get_sbt_handle_for_material(entry.material_id, raytype);
                    buffer[0..self.layout.handle_size].copy_from_slice(a);

                    let size_for_one = entry.data.len() / self.total_raytype as usize;
                    buffer[self.layout.handle_size..self.layout.handle_size + size_for_one]
                        .copy_from_slice(
                            &entry.data[size_for_one * raytype as usize
                                ..size_for_one * (raytype as usize + 1)],
                        );
                    self.buffer.set(
                        (*index * self.total_raytype + raytype) as usize,
                        &buffer[..self.layout.handle_size + size_for_one],
                    );
                }
            } else {
                self.raytype_pipeline_handles[raytype as usize] = pipeline.pipeline().raw();
                // Update all
                for (entry, index) in self.entries.iter() {
                    let a = pipeline.get_sbt_handle_for_material(entry.material_id, raytype);
                    buffer[0..self.layout.handle_size].copy_from_slice(a);

                    let size_for_one = entry.data.len() / self.total_raytype as usize;
                    buffer[self.layout.handle_size..self.layout.handle_size + size_for_one]
                        .copy_from_slice(
                            &entry.data[size_for_one * raytype as usize
                                ..size_for_one * (raytype as usize + 1)],
                        );
                    self.buffer.set(
                        (*index * self.total_raytype + raytype) as usize,
                        &buffer[..self.layout.handle_size + size_for_one],
                    );
                }
            }
        }
        self.update_list.clear();
    }
    pub fn hitgroup_sbt_buffer(
        &mut self,
    ) -> Option<impl GPUCommandFuture<Output = RenderRes<impl BufferLike + RenderData>>> {
        self.buffer.buffer()
    }
    pub fn hitgroup_stride(&self) -> usize {
        self.layout.one_raytype.pad_to_align().size()
    }
    pub fn add_instance<M: Material, A>(
        &mut self,
        material: &M,
        params: &mut SystemParamItem<M::ShaderParameterParams>,
    ) -> SbtIndex<A> {
        let mut data: Box<[u8]> =
            vec![0; self.total_raytype as usize * std::mem::size_of::<M::ShaderParameters>()]
                .into_boxed_slice();
        for i in 0..self.total_raytype {
            let params = material.parameters(i, params);

            let size = std::mem::size_of::<M::ShaderParameters>();
            data[size * i as usize..size * (i as usize + 1)].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &params as *const _ as *const u8,
                    std::mem::size_of_val(&params),
                )
            });
        }
        let entry = Entry {
            material_id: std::any::TypeId::of::<M>(),
            data,
        };
        if let Some(existing_index) = self.entries.get(&entry) {
            SbtIndex {
                index: *existing_index * self.total_raytype,
                _marker: PhantomData,
            }
        } else {
            let i = self.entries.len() as u32;
            self.entries.insert(entry.clone(), i);
            self.update_list.push(entry);
            SbtIndex {
                index: i * self.total_raytype,
                _marker: PhantomData,
            }
        }
    }
}

pub struct PipelineSbtManager {
    allocator: Allocator,
    buffer: Vec<u8>,
    num_raygen: u32,
    num_miss: u32,
    num_callable: u32,
    offset_strides: Vec<(u32, u32)>,
    upload_buffer: PerFrameState<ResidentBuffer>,
    device_buffer: Option<SharedDeviceStateHostContainer<ResidentBuffer>>,
}
pub struct PipelineSbtManagerInfo {
    buffer: Either<SharedDeviceState<ResidentBuffer>, PerFrameContainer<ResidentBuffer>>,
    num_raygen: u32,
    num_miss: u32,
    num_callable: u32,
    offset_strides: Vec<(u32, u32)>,
}
impl RenderData for PipelineSbtManagerInfo {
    fn tracking_feedback(&mut self, feedback: &rhyolite::future::TrackingFeedback) {
        self.buffer.tracking_feedback(feedback);
    }
}
impl BufferLike for PipelineSbtManagerInfo {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer.raw_buffer()
    }

    fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }

    fn device_address(&self) -> vk::DeviceAddress {
        self.buffer.device_address()
    }
    fn offset(&self) -> vk::DeviceSize {
        self.buffer.offset()
    }
}
impl PipelineSbtManagerInfo {
    pub fn rgen(&self, index: usize) -> vk::StridedDeviceAddressRegionKHR {
        assert!(index < self.num_raygen as usize);
        let (offset, size) = self.offset_strides[index];
        vk::StridedDeviceAddressRegionKHR {
            device_address: self.buffer.device_address() + offset as u64,
            stride: size as u64,
            size: size as u64,
        }
    }
    pub fn miss(&self) -> vk::StridedDeviceAddressRegionKHR {
        let index = Range {
            start: self.num_raygen as usize,
            end: self.num_miss as usize + self.num_raygen as usize,
        };
        let (offset, size) = self.offset_strides[index.start];
        for (_offset, stride) in self.offset_strides[index.clone()].iter() {
            debug_assert_eq!(*stride, size);
        }
        vk::StridedDeviceAddressRegionKHR {
            device_address: self.buffer.device_address() + offset as u64,
            stride: size as u64,
            size: size as u64 * index.len() as u64,
        }
    }
    pub fn callable(&self, index: Range<usize>) -> vk::StridedDeviceAddressRegionKHR {
        assert!(index.end <= self.num_callable as usize);
        let index = Range {
            start: index.start + self.num_raygen as usize + self.num_miss as usize,
            end: index.end + self.num_raygen as usize + self.num_miss as usize,
        };
        let (offset, size) = self.offset_strides[index.start];
        for (_offset, stride) in self.offset_strides[index.clone()].iter() {
            debug_assert_eq!(*stride, size);
        }
        vk::StridedDeviceAddressRegionKHR {
            device_address: self.buffer.device_address() + offset as u64,
            stride: size as u64,
            size: size as u64 * index.len() as u64,
        }
    }
}
impl PipelineSbtManager {
    pub fn new(allocator: Allocator) -> Self {
        Self {
            allocator,
            buffer: Vec::new(),
            num_callable: 0,
            num_miss: 0,
            num_raygen: 0,
            offset_strides: Vec::new(),
            upload_buffer: Default::default(),
            device_buffer: None,
        }
    }
    fn align(&mut self, first: bool) {
        let alignment = if first {
            self.allocator
                .device()
                .physical_device()
                .properties()
                .ray_tracing
                .shader_group_base_alignment
        } else {
            self.allocator
                .device()
                .physical_device()
                .properties()
                .ray_tracing
                .shader_group_handle_alignment
        };

        let new_len = self.buffer.len().next_multiple_of(alignment as usize);
        let additional_items = new_len - self.buffer.len();
        if additional_items > 0 {
            self.buffer
                .extend(std::iter::repeat(0).take(additional_items));
        }
    }
    /// Push the `index`th raygen shader in `pipeline` into the Sbt, with shader parameters P.
    pub fn push_raygen<P: AsStd430>(
        &mut self,
        pipeline: RayTracingPipelineManagerSpecializedPipeline,
        param: P,
        index: usize,
    ) {
        assert_eq!(self.num_miss, 0);
        assert_eq!(self.num_callable, 0);
        self.align(true);

        let offset = self.buffer.len();

        let handles_data = pipeline.pipeline().sbt_handles().rgen(index);
        self.buffer.extend(handles_data.iter().cloned());
        self.buffer.extend(param.as_std430().as_bytes());

        let stride = (handles_data.len() as u32 + P::std430_size_static() as u32).next_multiple_of(
            self.allocator
                .device()
                .physical_device()
                .properties()
                .ray_tracing
                .shader_group_handle_alignment,
        );
        self.offset_strides.push((offset as u32, stride));
        self.num_raygen += 1;
    }
    /// Push the `index`th miss shader in `pipeline` into the Sbt, with shader parameters P.
    pub fn push_miss<P: AsStd430>(
        &mut self,
        pipeline: RayTracingPipelineManagerSpecializedPipeline,
        param: P,
        index: usize,
    ) {
        assert_eq!(self.num_callable, 0);
        self.align(self.num_miss == 0);
        let offset = self.buffer.len();

        let handles_data = pipeline.pipeline().sbt_handles().rmiss(index);
        self.buffer.extend(handles_data.iter().cloned());
        self.buffer.extend(param.as_std430().as_bytes());
        let stride = (handles_data.len() as u32 + P::std430_size_static() as u32).next_multiple_of(
            self.allocator
                .device()
                .physical_device()
                .properties()
                .ray_tracing
                .shader_group_handle_alignment,
        );
        self.offset_strides.push((offset as u32, stride));
        self.num_miss += 1;
    }
    /// Push the `index`th callable shader in `pipeline` into the Sbt, with shader parameters P.
    pub fn push_callable<P: AsStd430>(
        &mut self,
        pipeline: RayTracingPipelineManagerSpecializedPipeline,
        param: P,
        index: usize,
    ) {
        self.align(self.num_callable == 0);
        let offset = self.buffer.len();

        let handles_data = pipeline.pipeline().sbt_handles().callable(index);
        self.buffer.extend(handles_data.iter().cloned());
        self.buffer.extend(param.as_std430().as_bytes());
        let stride = (handles_data.len() as u32 + P::std430_size_static() as u32).next_multiple_of(
            self.allocator
                .device()
                .physical_device()
                .properties()
                .ray_tracing
                .shader_group_handle_alignment,
        );
        self.offset_strides.push((offset as u32, stride));
        self.num_callable += 1;
    }

    pub fn build(&mut self) -> impl GPUCommandFuture<Output = RenderRes<PipelineSbtManagerInfo>> {
        self.align(false);
        let base_alignment = self
            .allocator
            .device()
            .physical_device()
            .properties()
            .ray_tracing
            .shader_group_base_alignment;

        let needs_copy = matches!(
            self.allocator.device().physical_device().memory_model(),
            PhysicalDeviceMemoryModel::Discrete
        );

        let create_upload_buffer = || {
            if needs_copy {
                self.allocator
                    .create_dynamic_buffer_uninit(
                        self.buffer.len() as u64,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .unwrap()
            } else {
                self.allocator
                    .create_dynamic_buffer_uninit_aligned(
                        self.buffer.len() as u64,
                        vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        base_alignment as u64,
                    )
                    .unwrap()
            }
        };
        let upload_buffer = use_per_frame_state(&mut self.upload_buffer, create_upload_buffer)
            .reuse(|old| {
                if old.size() < self.buffer.len() as u64 {
                    *old = create_upload_buffer();
                }
            });
        upload_buffer.contents_mut().unwrap()[0..self.buffer.len()].copy_from_slice(&self.buffer);

        let device_buffer = if needs_copy {
            let device_buffer = use_shared_state(
                &mut self.device_buffer,
                |_prev| {
                    self.allocator
                        .create_device_buffer_uninit_aligned(
                            self.buffer.len() as u64,
                            vk::BufferUsageFlags::TRANSFER_DST
                                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                            base_alignment as u64,
                        )
                        .unwrap()
                },
                |prev| prev.size() != self.buffer.len() as u64,
            );
            Some(device_buffer)
        } else {
            None
        };

        let num_raygen = std::mem::take(&mut self.num_raygen);
        let num_miss = std::mem::take(&mut self.num_miss);
        let num_callable = std::mem::take(&mut self.num_callable);
        let offset_strides = std::mem::take(&mut self.offset_strides);
        self.buffer.clear();

        commands! { move
            let upload_buffer = RenderRes::new(upload_buffer);
            let buffer = if let Some(device_buffer) = device_buffer {
                let mut device_buffer = device_buffer;
                copy_buffer(&upload_buffer, &mut device_buffer).await;
                retain!(upload_buffer);
                device_buffer.map(|a| Either::Left(a))
            }  else {
                upload_buffer.map(|a| Either::Right(a))
            };
            buffer.map(|a| PipelineSbtManagerInfo {
                buffer: a,
                num_callable,
                num_miss,
                num_raygen,
                offset_strides,
            })
        }
    }
}
