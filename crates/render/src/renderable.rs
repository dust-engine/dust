use ash::vk;

use bevy_ecs::component::Component;
use bitflags::bitflags;

bitflags! {
    /// Defines the flags for a [`Renderable`]
    /// Corresponds to VkAccelerationStructureInstanceKHR::flags
    pub struct RenderableFlags: u8 {
        /// Disables face culling for this renderable.
        const TRIANGLE_FACING_CULL_DISABLE = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8;

        /// Indicates that the facing determination for geometry contained in this renderable is inverted.
        /// Because the facing is determined in object space, an instance transform does not change the winding,
        /// but a geometry transform does.
        const TRIANGLE_FLIP_FACING = vk::GeometryInstanceFlagsKHR::TRIANGLE_FLIP_FACING.as_raw() as u8;

        /// Causes all geometries contained in this renderable to act as opaque, and the any hit shader will not be executed.
        /// This behavior can be overridden by the SPIR-V NoOpaqueKHR ray flag.
        const FORCE_OPAQUE = vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8;

        /// Causes all geometries contained in this renderable to act as non-opaqued.
        /// This behavior can be overridden by the SPIR-V OpaqueKHR ray flag.
        const FORCE_NO_OPAQUE = vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8;
    }
}
impl Default for RenderableFlags {
    fn default() -> Self {
        RenderableFlags::empty()
    }
}

/// Marker component for instances in the scene.
#[derive(Component, Clone, Default)]
pub struct Renderable {
    /// Index into the BLASStore
    pub flags: RenderableFlags,
}
