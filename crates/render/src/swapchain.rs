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
    pub fn current_image_mut(&mut self) -> Option<&mut AcquiredFrame> {
        self.current_image.as_mut()
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

    pub fn primary_mut(&mut self) -> Option<&mut Window> {
        self.windows.get_mut(&bevy_window::WindowId::primary())
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
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
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
fn queue_flush_and_present(mut windows: ResMut<Windows>, mut queues: ResMut<Arc<Queues>>) {
    let queues: &mut Queues = Arc::get_mut(&mut queues)
        .expect("All references to Queues should be dropped by the end of the frame.");
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

#[derive(Default)]
pub struct SwapchainPlugin;
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
    }
}
