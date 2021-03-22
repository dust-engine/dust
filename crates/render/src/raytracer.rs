use crate::frame::Frame;
use crate::hal;
use crate::renderer::RenderState;
use crate::{back, Renderer};
use hal::prelude::*;

use std::io::Cursor;
use tracing;
use crate::shared_buffer::SharedBuffer;

pub struct Raytracer {
    shared_buffer: SharedBuffer,
    ray_pass: <back::Backend as hal::Backend>::RenderPass,
    framebuffer: <back::Backend as hal::Backend>::Framebuffer,
    pipeline: <back::Backend as hal::Backend>::GraphicsPipeline,
    pipeline_layout: <back::Backend as hal::Backend>::PipelineLayout,
    viewport: hal::pso::Viewport,
    frames: [Frame; 3],
    current_frame: u8,
}
const CUBE_INDICES: [u16; 14] = [3, 7, 1, 5, 4, 7, 6, 3, 2, 1, 0, 4, 2, 6];
const CUBE_POSITIONS: [(f32, f32, f32); 8] = [
    (0.0, 0.0, 0.0), // 0
    (0.0, 0.0, 1.0), // 1
    (0.0, 1.0, 0.0), // 2
    (0.0, 1.0, 1.0), // 3
    (1.0, 0.0, 0.0), // 4
    (1.0, 0.0, 1.0), // 5
    (1.0, 1.0, 0.0), // 6
    (1.0, 1.0, 1.0), // 7
];

impl Raytracer {
    pub fn new(
        renderer: &Renderer,
        framebuffer_attachment: hal::image::FramebufferAttachment,
    ) -> Raytracer {
        let state = renderer.state.as_ref().unwrap();
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

        let pipeline_layout = unsafe {
            state
                .device
                .create_pipeline_layout(std::iter::empty(), std::iter::empty())
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
                    cull_face: hal::pso::Face::FRONT,
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
        let shared_buffer = SharedBuffer::new(
            &state.device,
            &CUBE_POSITIONS,
            &CUBE_INDICES,
            &renderer.memory_properties
        ).unwrap();
        Self {
            shared_buffer,
            ray_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            viewport,
            frames,
            current_frame: 0,
        }
    }

    pub fn rebuild_framebuffer(
        &mut self,
        state: &RenderState,
        framebuffer_attachment: hal::image::FramebufferAttachment,
    ) {
        self.viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: state.extent.width as i16,
                h: state.extent.height as i16,
            },
            depth: 0.0..1.0,
        };
        state.device.wait_idle().unwrap();
        tracing::trace!(
            "Rebuilt framebuffer with size {}x{}",
            state.extent.width,
            state.extent.height
        );
        unsafe {
            let mut framebuffer = state
                .device
                .create_framebuffer(
                    &self.ray_pass,
                    std::iter::once(framebuffer_attachment),
                    state.extent.to_extent(),
                )
                .unwrap();
            std::mem::swap(&mut self.framebuffer, &mut framebuffer);
            state.device.destroy_framebuffer(framebuffer);
        }
    }

    pub unsafe fn update(
        &mut self,
        state: &mut RenderState,
        target: &<back::Backend as hal::Backend>::ImageView,
    ) -> &mut <back::Backend as hal::Backend>::Semaphore {
        let current_frame: &mut Frame = &mut self.frames[self.current_frame as usize];

        // First, wait for the previous submission to complete.
        state
            .device
            .wait_for_fence(&current_frame.submission_complete_fence, !0)
            .unwrap();
        state
            .device
            .reset_fence(&mut current_frame.submission_complete_fence)
            .unwrap();
        current_frame.command_pool.reset(false);

        let mut clear_value = hal::command::ClearValue::default();
        clear_value.color.float32 = [0.0, 0.0, 0.0, 1.0];

        let cmd_buffer = &mut current_frame.command_buffer;
        cmd_buffer.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        cmd_buffer.set_viewports(0, std::iter::once(self.viewport.clone()));
        cmd_buffer.set_scissors(0, std::iter::once(self.viewport.rect));
        cmd_buffer.bind_vertex_buffers(
            0,
            std::iter::once((
                &self.shared_buffer.buffer,
                hal::buffer::SubRange {
                    offset: 0,
                    size: Some(std::mem::size_of_val(&CUBE_POSITIONS) as u64)
                }
            ))
        );
        cmd_buffer.bind_index_buffer(
            &self.shared_buffer.buffer,
            hal::buffer::SubRange {
                offset: std::mem::size_of_val(&CUBE_POSITIONS) as u64,
                size: Some(std::mem::size_of_val(&CUBE_INDICES) as u64),
            },
            hal::IndexType::U16
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
        cmd_buffer.draw_indexed(0..CUBE_INDICES.len() as u32, 0,0..1);
        cmd_buffer.end_render_pass();
        cmd_buffer.finish();

        state.graphics_queue_group.queues[0].submit(
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
}
