mod layout;
mod pool;

use ash::vk;
pub use layout::*;
pub use pool::*;

// descriptor pool should be a recycled resource.
// It doesn't have to be per unique descriptor layout, but it can be made against a list of descriptor layouts.
// It should then generate `all` of the descriptors for us to bind.
// It does not have to be per-frame, but it needs to have enough capacity,
// When writing descriptor, a comparison should be made first, If equal, skip writing desceritpro.
pub use crate::macros::PushConstants;
pub trait PushConstants {
    // TODO: Make this a const property, pending rust offset_of! macro
    fn ranges() -> Vec<vk::PushConstantRange>;
}
