mod buffer;
mod copy;
mod image;
pub use buffer::*;
pub use copy::*;
pub use image::*;
mod image_view;
pub use image_view::*;
mod managed_buffer_vec;
pub use managed_buffer_vec::*;
mod staging_ring_buffer;
pub use staging_ring_buffer::*;

#[derive(Clone)]
pub enum SharingMode<'a> {
    Exclusive,
    Concurrent { queue_family_indices: &'a [u32] },
}
impl<'a> Default for SharingMode<'a> {
    fn default() -> Self {
        Self::Exclusive
    }
}
