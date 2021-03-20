use crate::back;
use crate::hal;
use hal::prelude::*;
use crate::renderer::RenderState;
use tracing;
use std::io::Cursor;

pub struct Raytracer {
    ray_pass: <back::Backend as hal::Backend>::RenderPass,
    framebuffer: <back::Backend as hal::Backend>::Framebuffer,
    pipeline: <back::Backend as hal::Backend>::GraphicsPipeline,
}
const CUBE_INDICES: [u16; 14] = [
    3, 7, 1, 5, 4, 7, 6, 3, 2, 1, 0, 4, 2, 6,
];
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
        state: &RenderState,
        swap_config: &hal::window::SwapchainConfig,
        physical_device_properties: &hal::PhysicalDeviceProperties,
    ) -> Raytracer {
        let ray_pass = unsafe {
            state.device.create_render_pass(
                std::iter::once(
                    hal::pass::Attachment {
                        format: Some(state.surface_format),
                        samples: 1,
                        ops: hal::pass::AttachmentOps::new(
                            hal::pass::AttachmentLoadOp::Clear,
                            hal::pass::AttachmentStoreOp::Store,
                        ),
                        stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                        // The pass renders from Nothing to Present
                        layouts: hal::image::Layout::Undefined .. hal::image::Layout::Present
                    }
                ),
                std::iter::once(
                    hal::pass::SubpassDesc {
                        colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                        depth_stencil: None,
                        inputs: &[],
                        resolves: &[],
                        preserves: &[]
                    }
                ),
                std::iter::empty()
            )
        }.unwrap();
        let framebuffer = unsafe {
            state.device
                .create_framebuffer(
                    &ray_pass,
                    std::iter::once(
                        swap_config.framebuffer_attachment()
                    ),
                    swap_config.extent.to_extent()
                )
                .unwrap()
        };
        
        let pipeline_layout = unsafe {
            state.device
                .create_pipeline_layout(
                    std::iter::empty(),
                    std::iter::empty()
                )
                .unwrap()
        };
        let pipeline = unsafe {
            let vertex_module = {
                let spirv = gfx_auxil::read_spirv(
                    Cursor::new(&include_bytes!("./ray.vert.spv"))
                ).unwrap();
                state.device.create_shader_module(&spirv).unwrap()
            };
            let fragment_module = {
                let spirv = gfx_auxil::read_spirv(
                    Cursor::new(&include_bytes!("./ray.frag.spv"))
                ).unwrap();
                state.device.create_shader_module(&spirv).unwrap()
            };
            let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                hal::pso::PrimitiveAssemblerDesc::Vertex {
                    buffers: &[],
                    attributes: &[],
                    input_assembler: hal::pso::InputAssemblerDesc {
                        primitive: hal::pso::Primitive::TriangleStrip,
                        with_adjacency: false,
                        restart_index: None
                    },
                    vertex: hal::pso::EntryPoint {
                        entry: "main",
                        module: &vertex_module,
                        specialization: Default::default()
                    },
                    tessellation: None,
                    geometry: None
                },
                hal::pso::Rasterizer {
                    front_face: hal::pso::FrontFace::Clockwise,
                    cull_face: hal::pso::Face::FRONT,
                    ..hal::pso::Rasterizer::FILL
                },
                Some(hal::pso::EntryPoint {
                    entry: "main",
                    module: &fragment_module,
                    specialization: Default::default()
                }),
                &pipeline_layout,
                hal::pass::Subpass {
                    index: 0,
                    main_pass: &ray_pass
                }
            );
            pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc {
                mask: hal::pso::ColorMask::ALL,
                blend: Some(hal::pso::BlendState::ALPHA)
            });
            let pipeline = state.device.create_graphics_pipeline(
                &pipeline_desc,
            None,
            ).unwrap();
            state.device.destroy_shader_module(vertex_module);
            state.device.destroy_shader_module(fragment_module);
            pipeline
        };

        // Populate initial data
        Self {
            ray_pass,
            framebuffer,
            pipeline,
        }
    }

    pub fn rebuild_framebuffer(
        &mut self,
        state: &RenderState,
        swap_config: &hal::window::SwapchainConfig,
    ) {
        state.device.wait_idle().unwrap();
        tracing::trace!("Rebuilt framebuffer with size {}x{}", swap_config.extent.width, swap_config.extent.height);
        unsafe {
            let mut framebuffer = state.device
                .create_framebuffer(
                    &self.ray_pass,
                    std::iter::once(
                        swap_config.framebuffer_attachment()
                    ),
                    swap_config.extent.to_extent()
                )
                .unwrap();
            std::mem::swap(&mut self.framebuffer, &mut framebuffer);
            state.device
                .destroy_framebuffer(framebuffer);
        }

    }
}
