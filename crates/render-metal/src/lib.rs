use dust_core::svo::alloc::{AllocError, BlockAllocator};
use dust_core::{CameraProjection, SunLight};
use glam::TransformRT;
use std::ops::Range;

pub struct Renderer {}
impl Renderer {
    pub fn new(window_handle: &impl raw_window_handle::HasRawWindowHandle) -> Self {
        Renderer {}
    }
    pub fn resize(&mut self) {}
    pub fn update(&mut self, state: &State) {}
    pub fn create_raytracer(&self) {}
    pub fn create_block_allocator(&self, size: u64) -> Box<dyn BlockAllocator> {
        unsafe { Box::new(Allocator {}) }
    }
}
struct Allocator {}
impl BlockAllocator for Allocator {
    unsafe fn allocate_block(&mut self) -> Result<*mut u8, AllocError> {
        unimplemented!()
    }

    unsafe fn deallocate_block(&mut self, block: *mut u8) {
        unimplemented!()
    }

    unsafe fn flush(&mut self, ranges: &mut dyn Iterator<Item = (*mut u8, Range<u32>)>) {
        unimplemented!()
    }
}

pub struct State<'a> {
    pub camera_projection: &'a CameraProjection,
    pub camera_transform: &'a TransformRT,
    pub sunlight: &'a SunLight,
}
