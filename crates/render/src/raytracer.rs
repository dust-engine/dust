use crate::back;
use crate::hal;
use hal::prelude::*;
use crate::renderer::RenderState;

pub struct Raytracer {
    ray_pass: <back::Backend as hal::Backend>::RenderPass,
    framebuffer: <back::Backend as hal::Backend>::Framebuffer,
}

impl Raytracer {
    pub fn new(
        state: &RenderState,
        swap_config: &hal::window::SwapchainConfig,
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
        Self {
            ray_pass,
            framebuffer,
        }
    }

    pub fn rebuild_framebuffer(
        &mut self,
        state: &RenderState,
        swap_config: &hal::window::SwapchainConfig,
    ) {
        state.device.wait_idle().unwrap();
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
