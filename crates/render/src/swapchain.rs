use std::{collections::HashMap, sync::Arc};

use crate::RenderWorld;
use ash::vk;
use bevy_app::{App, Plugin};
use bevy_ecs::{
    event::EventReader,
    schedule::{ParallelSystemDescriptorCoercion, SystemLabel},
    system::{Res, ResMut},
};
use bevy_window::WindowId;
pub use command_buffer::SwapchainCmdBufferState;
use dustash::{
    frames::AcquiredFrame,
    queue::Queues,
    surface::{Surface, SurfaceLoader},
    swapchain::SwapchainLoader,
    Device,
};

pub struct Window {
    frames: dustash::frames::FrameManager,

    // This is guaranteed to be Some in the Queue stage only.
    current_image: Option<AcquiredFrame>,
}
impl Window {
    pub fn current_image(&self) -> Option<&AcquiredFrame> {
        self.current_image.as_ref()
    }
    pub fn frames(&self) -> &dustash::frames::FrameManager {
        &self.frames
    }
}

struct ExtractedWindow {
    id: WindowId,
    physical_width: u32,
    physical_height: u32,
    handle: bevy_window::RawWindowHandleWrapper,
}

#[derive(Default)]
pub struct Windows {
    extracted_windows: Vec<ExtractedWindow>,
    windows: HashMap<bevy_window::WindowId, Window>,
}

impl Windows {
    pub fn primary(&self) -> Option<&Window> {
        self.windows.get(&bevy_window::WindowId::primary())
    }
}

/// Extract WindowCreated and WindowResized events into Windows.extracted_windows
fn extract_windows(
    mut render_world: ResMut<RenderWorld>,
    windows: Res<bevy_window::Windows>,
    mut window_created_events: EventReader<bevy_window::WindowCreated>,
    mut window_resized_events: EventReader<bevy_window::WindowResized>,
) {
    let mut render_windows = render_world.resource_mut::<Windows>();
    for created_event in window_created_events.iter() {
        let window = windows.get(created_event.id).unwrap();
        render_windows.extracted_windows.push(ExtractedWindow {
            id: window.id(),
            physical_width: window.physical_width(),
            physical_height: window.physical_height(),
            handle: window.raw_window_handle(),
        })
    }
    for created_event in window_resized_events.iter() {
        let window = windows.get(created_event.id).unwrap();
        render_windows.extracted_windows.push(ExtractedWindow {
            id: window.id(),
            physical_width: window.physical_width(),
            physical_height: window.physical_height(),
            handle: window.raw_window_handle(),
        })
    }
}

/// - Drain Windows.extracted_windows
/// - Create surface and swapchain for new windows
/// - Update swapchains for resized windows
/// - Acquire swapchain image for all windows
fn prepare_windows(
    mut windows: ResMut<Windows>,
    surface_loader: Res<Arc<SurfaceLoader>>,
    swapchain_loader: Res<Arc<SwapchainLoader>>,
) {
    let windows = &mut *windows;
    for extracted_window in windows.extracted_windows.iter() {
        if let Some(window) = windows.windows.get_mut(&extracted_window.id) {
            // Update the window
            window.frames.update(vk::Extent2D {
                width: extracted_window.physical_width,
                height: extracted_window.physical_height,
            });
        } else {
            // Create the swapchain
            let surface = Surface::create(surface_loader.clone(), unsafe {
                &extracted_window.handle.get_handle()
            })
            .unwrap();
            let frames = dustash::frames::FrameManager::new(
                swapchain_loader.clone(),
                Arc::new(surface),
                dustash::frames::Options {
                    frames_in_flight: 3,
                    format_preference: vec![vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_SRGB,
                        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    }],
                    present_mode_preference: vec![vk::PresentModeKHR::FIFO],
                    usage: vk::ImageUsageFlags::TRANSFER_DST,
                    ..Default::default()
                },
                vk::Extent2D {
                    width: extracted_window.physical_width,
                    height: extracted_window.physical_height,
                },
            )
            .unwrap();
            windows.windows.insert(
                extracted_window.id,
                Window {
                    frames,
                    current_image: None,
                },
            );
        }
    }
    windows.extracted_windows.clear();

    for window in windows.windows.values_mut() {
        let next_frame = window.frames.acquire(!0).unwrap();
        window.current_image = Some(next_frame);
    }
}

// Flush and present
fn queue_flush_and_present(mut windows: ResMut<Windows>, mut queues: ResMut<Queues>) {
    queues.flush().unwrap();
    for window in windows.windows.values_mut() {
        let frame = window.current_image.take().unwrap();
        queues.present(&mut window.frames, frame).unwrap();
    }
}

#[derive(SystemLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub enum SwapchainSystem {
    Extract,
    Prepare,
    FlushAndPresent,
}

pub struct SwapchainPlugin {
    /// When true, a `SwapchainCmdBufferState` will be available as a resource during RenderStage::Render.
    /// The application is required to attach a system that calls `SwapchainCmdBufferState::record` during
    /// RenderStage::Render. The callback function passed to `SwapchainCmdBufferState::record` will only be
    /// invoked when a command buffer reset and re-record is necessary, usually due to a window resize.
    managed_command_buffer: bool,
}
impl Default for SwapchainPlugin {
    fn default() -> Self {
        Self {
            managed_command_buffer: true,
        }
    }
}
impl Plugin for SwapchainPlugin {
    fn build(&self, render_app: &mut App) {
        let device = render_app.world.resource::<Arc<Device>>().clone();
        render_app
            .insert_resource(Arc::new(SurfaceLoader::new(device.instance().clone())))
            .insert_resource(Arc::new(SwapchainLoader::new(device)))
            .init_resource::<Windows>()
            .add_system_to_stage(
                crate::RenderStage::Extract,
                extract_windows.label(SwapchainSystem::Extract),
            )
            .add_system_to_stage(
                crate::RenderStage::Prepare,
                prepare_windows.label(SwapchainSystem::Prepare),
            )
            .add_system_to_stage(
                crate::RenderStage::Cleanup,
                queue_flush_and_present.label(SwapchainSystem::FlushAndPresent),
            );

        if self.managed_command_buffer {
            render_app
                .init_resource::<command_buffer::SwapchainCmdBufferState>()
                .init_resource::<command_buffer::SwapchainCmdBuffers>()
                .add_system_to_stage(
                    crate::RenderStage::Prepare,
                    command_buffer::swapchain_cmd_buffer_prepare.after(SwapchainSystem::Prepare),
                )
                .add_system_to_stage(
                    crate::RenderStage::Cleanup,
                    command_buffer::swapchain_cmd_buffer_submit
                        .before(SwapchainSystem::FlushAndPresent),
                );
        }
    }
}

pub mod command_buffer {
    use std::sync::Arc;

    use super::Windows;
    use ash::vk;
    use bevy_ecs::{
        system::{Res, ResMut},
        world::FromWorld,
    };
    use dustash::{
        command::{
            pool::{CommandBuffer, CommandPool},
            recorder::{CommandExecutable, CommandRecorder},
        },
        queue::{QueueType, Queues, SemaphoreOp},
        Device,
    };

    pub enum SwapchainCmdBufferState {
        Recorded(Arc<CommandExecutable>),
        Initial(CommandBuffer),
        None,
    }
    impl Default for SwapchainCmdBufferState {
        fn default() -> Self {
            Self::None
        }
    }
    impl SwapchainCmdBufferState {
        /// Record into the command buffer if it's not already recorded.
        pub fn record(
            &mut self,
            flags: vk::CommandBufferUsageFlags,
            record: impl FnOnce(&mut CommandRecorder),
        ) {
            let exec = match std::mem::take(self) {
                SwapchainCmdBufferState::Recorded(exec) => exec,
                SwapchainCmdBufferState::Initial(buffer) => {
                    buffer.record(flags, record).map(Arc::new).unwrap()
                }
                SwapchainCmdBufferState::None => return,
            };
            *self = SwapchainCmdBufferState::Recorded(exec);
        }
    }

    /// The command buffers used exclusively for rendering to the swapchain images
    pub(super) struct SwapchainCmdBuffers {
        // The command pool used for rendering to the swapchain
        command_pool: Arc<CommandPool>,
        command_buffers: Vec<SwapchainCmdBufferState>,
    }
    impl FromWorld for SwapchainCmdBuffers {
        fn from_world(world: &mut bevy_ecs::prelude::World) -> Self {
            let device = world.get_resource::<Arc<Device>>().unwrap();
            let queues = world.get_resource::<Queues>().unwrap();
            let pool = CommandPool::new(
                device.clone(),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queues
                    .of_type(dustash::queue::QueueType::Graphics)
                    .family_index(),
            )
            .unwrap();
            let pool = Arc::new(pool);
            Self {
                command_pool: pool,
                command_buffers: Vec::new(),
            }
        }
    }
    pub(super) fn swapchain_cmd_buffer_prepare(
        windows: Res<Windows>,
        mut state: ResMut<SwapchainCmdBuffers>,
        mut swapchain_command_buffer: ResMut<SwapchainCmdBufferState>,
    ) {
        // TODO: make this work for multiple windows
        let num_images = windows.primary().unwrap().frames().num_images();
        let current_image = windows.primary().unwrap().current_image().unwrap();

        if num_images != state.command_buffers.len() {
            let buffers = state.command_pool.allocate_n(num_images as u32).unwrap();
            state.command_buffers = buffers
                .into_iter()
                .map(|b| SwapchainCmdBufferState::Initial(b))
                .collect()
        }
        let buffer_state =
            std::mem::take(&mut state.command_buffers[current_image.image_index as usize]);

        match (buffer_state, current_image.invalidate_images) {
            (SwapchainCmdBufferState::Initial(buffer), _) => {
                *swapchain_command_buffer = SwapchainCmdBufferState::Initial(buffer);
            }
            (SwapchainCmdBufferState::Recorded(exec), true) => {
                // The Arc<CommandExecutable> was only cloned once to be referenced by the queue.
                // At this point the queue should've completed execution, so the Arc should only have one reference.
                let buffer = match Arc::try_unwrap(exec) {
                    Ok(exec) => exec.reset(false),
                    Err(exec) => {
                        // Fallback to re-allocating command buffer, but throw a warning for the potential leak.
                        drop(exec);
                        println!("Re-allocating command buffer");
                        state.command_pool.allocate_one().unwrap()
                    }
                };
                *swapchain_command_buffer = SwapchainCmdBufferState::Initial(buffer);
            }
            (SwapchainCmdBufferState::Recorded(exec), false) => {
                // Reuse the command buffer.
                *swapchain_command_buffer = SwapchainCmdBufferState::Recorded(exec);
            }
            (SwapchainCmdBufferState::None, _) => panic!(),
        };
    }
    pub(super) fn swapchain_cmd_buffer_submit(
        queues: Res<Queues>,
        windows: Res<Windows>,
        mut state: ResMut<SwapchainCmdBuffers>,
        mut swapchain_command_buffer: ResMut<SwapchainCmdBufferState>,
    ) {
        let exec = match std::mem::take(&mut *swapchain_command_buffer) {
            SwapchainCmdBufferState::Recorded(exec) => exec,
            _ => panic!("Expecting the swapchain command buffer to be recorded!"),
        };
        let current_image = windows.primary().unwrap().current_image().unwrap();
        queues
            .of_type(QueueType::Graphics)
            .submit(
                Box::new([SemaphoreOp {
                    semaphore: current_image.acquire_ready_semaphore.clone(),
                    stage_mask: vk::PipelineStageFlags2::CLEAR,
                    value: 0,
                }]),
                Box::new([exec.clone()]),
                Box::new([SemaphoreOp {
                    semaphore: current_image.render_complete_semaphore.clone(),
                    stage_mask: vk::PipelineStageFlags2::CLEAR,
                    value: 0,
                }]),
            )
            .fence(current_image.complete_fence.clone());

        state.command_buffers[current_image.image_index as usize] =
            SwapchainCmdBufferState::Recorded(exec);
    }
}
