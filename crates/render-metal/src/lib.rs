use dust_core::svo::alloc::{AllocError, BlockAllocator, SystemBlockAllocator, CHUNK_SIZE};
use dust_core::{CameraProjection, SunLight};
use glam::TransformRT;
use std::ops::Range;
use std::os::raw::c_void;

extern "C" {
    fn RendererNew(window: *mut c_void, view: *mut c_void);
}

pub struct Renderer {}
impl Renderer {
    pub fn new(window_handle: &impl raw_window_handle::HasRawWindowHandle) -> Self {
        use raw_window_handle::RawWindowHandle;
        let window_handle = window_handle.raw_window_handle();
        match window_handle {
            RawWindowHandle::MacOS(window_handle) => {
                unsafe {
                    RendererNew(window_handle.ns_window as *mut c_void, window_handle.ns_view as *mut c_void);
                }
            }
            _ => {
                panic!("Unsupported Window Handle")
            }
        }
        Renderer {}
    }
    pub fn resize(&mut self) {}
    pub fn update(&mut self, state: &State) {}
    pub fn create_raytracer(&self) {}
    pub fn create_block_allocator(&self, size: u64) -> Box<dyn BlockAllocator> {
        unsafe { Box::new(SystemBlockAllocator::new(CHUNK_SIZE as u32)) }
    }
}

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
    pub sunlight: &'a SunLight,
}
