use crate::frame::Frame;
use crate::renderer::RenderState;
use crate::{back, hal};
use hal::prelude::*;

use crate::descriptor_pool::DescriptorPool;
use crate::shared_buffer::SharedBuffer;
use gfx_hal::window::Extent2D;
use std::io::Cursor;
use std::sync::Arc;

pub struct Raytracer {
    pub device: Arc<<back::Backend as hal::Backend>::Device>,
    pub shared_buffer: SharedBuffer,
    pub ray_pass: <back::Backend as hal::Backend>::RenderPass,
    pub framebuffer: <back::Backend as hal::Backend>::Framebuffer,
    pub pipeline: <back::Backend as hal::Backend>::GraphicsPipeline,
    pub pipeline_layout: <back::Backend as hal::Backend>::PipelineLayout,
    pub viewport: hal::pso::Viewport,
    pub frames: [Frame; 3],
    pub current_frame: u8,
    pub desc_pool: DescriptorPool,
    pub desc_set: <back::Backend as hal::Backend>::DescriptorSet,
    pub octree_desc_pool: DescriptorPool,
    pub octree_desc_set: <back::Backend as hal::Backend>::DescriptorSet,
}
const CUBE_INDICES: [u16; 14] = [3, 7, 1, 5, 4, 7, 6, 3, 2, 1, 0, 4, 2, 6];
const CUBE_POSITIONS: [(f32, f32, f32); 8] = [
    (-1.0, -1.0, -1.0), // 0
    (-1.0, -1.0, 1.0),  // 1
    (-1.0, 1.0, -1.0),  // 2
    (-1.0, 1.0, 1.0),   // 3
    (1.0, -1.0, -1.0),  // 4
    (1.0, -1.0, 1.0),   // 5
    (1.0, 1.0, 0.0),    // 6
    (1.0, 1.0, 1.0),    // 7
];

impl Raytracer {
    pub fn new(
        state: &mut RenderState,
        memory_properties: &hal::adapter::MemoryProperties,
        framebuffer_attachment: hal::image::FramebufferAttachment,
        device_type: &hal::adapter::DeviceType,
    ) -> (Raytracer, Box<svo::alloc::ArenaBlockAllocator>) {
        let mut octree_desc_pool = DescriptorPool::new(
            &state.device,
            "SSBOs",
            std::iter::once(hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Buffer {
                    ty: hal::pso::BufferDescriptorType::Storage { read_only: true },
                    format: hal::pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    },
                },
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            }),
        );
        let (block_allocator, octree_desc_set): (Box<svo::alloc::ArenaBlockAllocator>, _) = {
            let mut desc_set = octree_desc_pool
                .allocate_one(&state.device, "Octree")
                .unwrap();

            use block_alloc::{DiscreteBlockAllocator, IntegratedBlockAllocator};
            use hal::adapter::DeviceType;
            const SIZE: usize = svo::alloc::CHUNK_SIZE;
            let device_type = DeviceType::IntegratedGpu;
            let allocator: Box<svo::alloc::ArenaBlockAllocator> = match device_type {
                DeviceType::DiscreteGpu | DeviceType::VirtualGpu | DeviceType::Other => {
                    let allocator: DiscreteBlockAllocator<back::Backend, SIZE> =
                        DiscreteBlockAllocator::new(
                            state.device.clone(),
                            state.transfer_binding_queue_group.queues.pop().unwrap(),
                            state.transfer_binding_queue_group.family,
                            memory_properties,
                        )
                        .unwrap();
                    unsafe {
                        state
                            .device
                            .write_descriptor_set(hal::pso::DescriptorSetWrite {
                                set: &mut desc_set,
                                binding: 0,
                                array_offset: 0,
                                descriptors: std::iter::once(hal::pso::Descriptor::Buffer(
                                    &allocator.device_buf,
                                    hal::buffer::SubRange {
                                        offset: 0,
                                        size: None,
                                    },
                                )),
                            })
                    };
                    Box::new(allocator)
                }
                DeviceType::IntegratedGpu | DeviceType::Cpu => {
                    let allocator: IntegratedBlockAllocator<back::Backend, SIZE> =
                        IntegratedBlockAllocator::new(
                            state.device.clone(),
                            state.transfer_binding_queue_group.queues.pop().unwrap(),
                            memory_properties,
                        )
                        .unwrap();
                    unsafe {
                        state
                            .device
                            .write_descriptor_set(hal::pso::DescriptorSetWrite {
                                set: &mut desc_set,
                                binding: 0,
                                array_offset: 0,
                                descriptors: std::iter::once(hal::pso::Descriptor::Buffer(
                                    &allocator.buf,
                                    hal::buffer::SubRange {
                                        offset: 0,
                                        size: None,
                                    },
                                )),
                            })
                    };
                    Box::new(allocator)
                }
            };
            (allocator, desc_set)
        };

        let shared_buffer = SharedBuffer::new(
            state.device.clone(),
            &mut state.transfer_binding_queue_group,
            &CUBE_POSITIONS,
            &CUBE_INDICES,
            memory_properties,
        )
        .unwrap();
        let ray_pass = unsafe {
            state.device.create_render_pass(
                std::iter::once(hal::pass::Attachment {
                    format: Some(state.surface_format),
                    samples: 1,
                    ops: hal::pass::AttachmentOps::new(
                        hal::pass::AttachmentLoadOp::Clear,
                        hal::pass::AttachmentStoreOp::Store,
                    ),
                    stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                    // The pass renders from Nothing to Present
                    layouts: hal::image::Layout::Undefined..hal::image::Layout::Present,
                }),
                std::iter::once(hal::pass::SubpassDesc {
                    colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                    depth_stencil: None,
                    inputs: &[],
                    resolves: &[],
                    preserves: &[],
                }),
                std::iter::empty(),
            )
        }
        .unwrap();
        let framebuffer = unsafe {
            state
                .device
                .create_framebuffer(
                    &ray_pass,
                    std::iter::once(framebuffer_attachment),
                    state.extent.to_extent(),
                )
                .unwrap()
        };

        let mut desc_pool = DescriptorPool::new(
            &state.device,
            "Uniform",
            std::iter::once(
                // Camera
                hal::pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: hal::pso::DescriptorType::Buffer {
                        ty: hal::pso::BufferDescriptorType::Uniform,
                        format: hal::pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: hal::pso::ShaderStageFlags::VERTEX
                        | hal::pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
            ),
        );
        let desc_set = unsafe {
            let mut desc_set = desc_pool.allocate_one(&state.device, "Camera").unwrap();
            state
                .device
                .write_descriptor_set(hal::pso::DescriptorSetWrite {
                    set: &mut desc_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: std::iter::once(hal::pso::Descriptor::Buffer(
                        &shared_buffer.buffer,
                        hal::buffer::SubRange {
                            offset: 128,
                            size: Some(128),
                        },
                    )),
                });
            desc_set
        };

        let pipeline_layout = unsafe {
            state
                .device
                .create_pipeline_layout(
                    std::iter::once(&desc_pool.layout)
                        .chain(std::iter::once(&octree_desc_pool.layout)),
                    std::iter::empty(),
                )
                .unwrap()
        };
        let (pipeline_layout, pipeline) = unsafe {
            let vertex_module = {
                let spirv =
                    gfx_auxil::read_spirv(Cursor::new(&include_bytes!("./ray.vert.spv"))).unwrap();
                state.device.create_shader_module(&spirv).unwrap()
            };
            let fragment_module = {
                let spirv =
                    gfx_auxil::read_spirv(Cursor::new(&include_bytes!("./ray.frag.spv"))).unwrap();
                state.device.create_shader_module(&spirv).unwrap()
            };
            let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                hal::pso::PrimitiveAssemblerDesc::Vertex {
                    buffers: &[hal::pso::VertexBufferDesc {
                        binding: 0,
                        stride: std::mem::size_of::<(f32, f32, f32)>() as u32,
                        rate: hal::pso::VertexInputRate::Vertex,
                    }],
                    attributes: &[hal::pso::AttributeDesc {
                        location: 0,
                        binding: 0,
                        element: hal::pso::Element {
                            format: hal::format::Format::Rgb32Sfloat,
                            offset: 0,
                        },
                    }],
                    input_assembler: hal::pso::InputAssemblerDesc {
                        primitive: hal::pso::Primitive::TriangleStrip,
                        with_adjacency: false,
                        restart_index: None,
                    },
                    vertex: hal::pso::EntryPoint {
                        entry: "main",
                        module: &vertex_module,
                        specialization: Default::default(),
                    },
                    tessellation: None,
                    geometry: None,
                },
                hal::pso::Rasterizer {
                    front_face: hal::pso::FrontFace::Clockwise,
                    cull_face: hal::pso::Face::NONE,
                    ..hal::pso::Rasterizer::FILL
                },
                Some(hal::pso::EntryPoint {
                    entry: "main",
                    module: &fragment_module,
                    specialization: Default::default(),
                }),
                &pipeline_layout,
                hal::pass::Subpass {
                    index: 0,
                    main_pass: &ray_pass,
                },
            );
            pipeline_desc
                .blender
                .targets
                .push(hal::pso::ColorBlendDesc {
                    mask: hal::pso::ColorMask::ALL,
                    blend: Some(hal::pso::BlendState::ALPHA),
                });
            let pipeline = state
                .device
                .create_graphics_pipeline(&pipeline_desc, None)
                .unwrap();
            state.device.destroy_shader_module(vertex_module);
            state.device.destroy_shader_module(fragment_module);
            (pipeline_layout, pipeline)
        };
        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: state.extent.width as i16,
                h: state.extent.height as i16,
            },
            depth: 0.0..1.0,
        };
        let frames = [
            Frame::new(&state.device, state.graphics_queue_group.family),
            Frame::new(&state.device, state.graphics_queue_group.family),
            Frame::new(&state.device, state.graphics_queue_group.family),
        ];
        let raytracer = Self {
            device: state.device.clone(),
            shared_buffer,
            ray_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            viewport,
            frames,
            desc_pool,
            desc_set,
            octree_desc_pool,
            current_frame: 0,
            octree_desc_set,
        };
        (raytracer, block_allocator)
    }

    pub fn rebuild_framebuffer(
        &mut self,
        extent: Extent2D,
        framebuffer_attachment: hal::image::FramebufferAttachment,
    ) {
        self.viewport.rect.w = extent.width as i16;
        self.viewport.rect.h = extent.height as i16;
        self.device.wait_idle().unwrap();
        tracing::trace!(
            "Rebuilt framebuffer with size {}x{}",
            extent.width,
            extent.height
        );
        unsafe {
            let mut framebuffer = self
                .device
                .create_framebuffer(
                    &self.ray_pass,
                    std::iter::once(framebuffer_attachment),
                    extent.to_extent(),
                )
                .unwrap();
            std::mem::swap(&mut self.framebuffer, &mut framebuffer);
            self.device.destroy_framebuffer(framebuffer);
        }
    }

    pub unsafe fn update(
        &mut self,
        target: &<back::Backend as hal::Backend>::ImageView,
        graphics_queue: &mut <back::Backend as hal::Backend>::Queue,
        state: &crate::State,
    ) -> &mut <back::Backend as hal::Backend>::Semaphore {
        let aspect_ratio = self.viewport.rect.w as f32 / self.viewport.rect.h as f32;
        self.shared_buffer.update_camera(
            &state.camera_projection,
            &state.camera_transform,
            aspect_ratio,
        );

        let current_frame: &mut Frame = &mut self.frames[self.current_frame as usize];

        // First, wait for the previous submission to complete.
        self.device
            .wait_for_fence(&current_frame.submission_complete_fence, !0)
            .unwrap();
        self.device
            .reset_fence(&mut current_frame.submission_complete_fence)
            .unwrap();
        current_frame.command_pool.reset(false);

        let mut clear_value = hal::command::ClearValue::default();
        clear_value.color.float32 = [0.0, 0.0, 0.0, 1.0];

        let cmd_buffer = &mut current_frame.command_buffer;
        cmd_buffer.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        self.shared_buffer.record_cmd_buffer(cmd_buffer);
        cmd_buffer.set_viewports(0, std::iter::once(self.viewport.clone()));
        cmd_buffer.set_scissors(0, std::iter::once(self.viewport.rect));
        cmd_buffer.bind_graphics_descriptor_sets(
            &self.pipeline_layout,
            0,
            std::iter::once(&self.desc_set).chain(std::iter::once(&self.octree_desc_set)),
            std::iter::empty(),
        );
        cmd_buffer.bind_vertex_buffers(
            0,
            std::iter::once((
                &self.shared_buffer.buffer,
                hal::buffer::SubRange {
                    offset: 0,
                    size: Some(std::mem::size_of_val(&CUBE_POSITIONS) as u64),
                },
            )),
        );
        cmd_buffer.bind_index_buffer(
            &self.shared_buffer.buffer,
            hal::buffer::SubRange {
                offset: std::mem::size_of_val(&CUBE_POSITIONS) as u64,
                size: Some(std::mem::size_of_val(&CUBE_INDICES) as u64),
            },
            hal::IndexType::U16,
        );
        cmd_buffer.bind_graphics_pipeline(&self.pipeline);
        cmd_buffer.begin_render_pass(
            &self.ray_pass,
            &self.framebuffer,
            self.viewport.rect,
            std::iter::once(hal::command::RenderAttachmentInfo {
                image_view: target,
                clear_value,
            }),
            hal::command::SubpassContents::Inline,
        );
        cmd_buffer.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
        cmd_buffer.end_render_pass();
        cmd_buffer.finish();

        graphics_queue.submit(
            std::iter::once(&*cmd_buffer),
            std::iter::empty(),
            std::iter::once(&current_frame.submission_complete_semaphore),
            Some(&mut current_frame.submission_complete_fence),
        );
        self.current_frame += 1;
        if self.current_frame >= 3 {
            self.current_frame = 0;
        }
        &mut current_frame.submission_complete_semaphore
    }

    pub unsafe fn destroy(self) {
        self.shared_buffer.destroy(&self.device);
        self.device.destroy_render_pass(self.ray_pass);
        self.device.destroy_framebuffer(self.framebuffer);
        self.device.destroy_graphics_pipeline(self.pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        for frame in std::array::IntoIter::new(self.frames) {
            frame.destroy(&self.device);
        }
        self.desc_pool.destroy(&self.device);
    }
}
