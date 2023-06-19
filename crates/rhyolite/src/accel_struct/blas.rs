use std::{alloc::Layout, sync::Arc};

use ash::prelude::VkResult;
use ash::vk;

use crate::Allocator;
use crate::BufferLike;

use crate::HasDevice;
use crate::ResidentBuffer;

use super::build::AccelerationStructureBuild;
use super::AccelerationStructure;

/// Builds one AABB BLAS containing many geometries
pub struct AabbBlasBuilder {
    geometries: Vec<(Arc<ResidentBuffer>, usize, vk::GeometryFlagsKHR, u32)>, // data, stride, flags, num_primitives
    flags: vk::BuildAccelerationStructureFlagsKHR,
    num_primitives: u64,
    geometry_primitive_counts: Vec<u32>,
    primitive_datasize: usize,
}

impl AabbBlasBuilder {
    pub fn new(flags: vk::BuildAccelerationStructureFlagsKHR) -> Self {
        Self {
            geometries: Vec::new(),
            flags,
            num_primitives: 0,
            geometry_primitive_counts: Vec::new(),
            primitive_datasize: 0,
        }
    }
    pub fn add_geometry(
        &mut self,
        primitives: Arc<ResidentBuffer>,
        flags: vk::GeometryFlagsKHR,
        layout: Layout, // Layout for one AABB entry
    ) {
        // There might be two cases where vk::AabbPositionsKHR aren't layed out with a stride = 24
        // 1. The user wants to interleave some other metadata between vk::AabbPositionsKHR.
        //    Vulkan only guarantees that the intersection shader will be called for items within the AABB,
        //    so without raw f32 AABB data there might be visible artifacts.
        //    The primitive buffer likely needs to stay in device memory persistently for this, and the user might want to
        //    interleave some other metadata alongside the vk::AabbPositionsKHR.
        // 2. Using the same buffer for two or more geometries, interleaving the data. We assume that this use case
        //     would be very rare, so the design of the API does not consider this.
        let stride = {
            // verify that the layout is OK
            let padding_in_slice = layout.padding_needed_for(layout.align());
            // VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545: stride must be a multiple of 8
            let padding_in_buffer = layout.padding_needed_for(8);
            debug_assert_eq!(
                padding_in_slice, padding_in_buffer,
                "Type is incompatible. Stride between items must be a multiple of 8."
            );
            let stride = layout.size() + padding_in_buffer;
            debug_assert!(stride <= u32::MAX as usize);
            debug_assert!(stride % 8 == 0);
            stride
        };
        let num_primitives = primitives.size() / stride as u64;
        self.num_primitives += num_primitives;
        self.primitive_datasize += primitives.size() as usize;
        self.geometries
            .push((primitives, stride, flags, num_primitives as u32));
        self.geometry_primitive_counts.push(num_primitives as u32);
    }
    pub fn build(self, allocator: Allocator) -> VkResult<AccelerationStructureBuild> {
        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = self
            .geometries
            .iter()
            .map(
                |(_, stride, flags, _)| vk::AccelerationStructureGeometryKHR {
                    geometry_type: vk::GeometryTypeKHR::AABBS,
                    geometry: vk::AccelerationStructureGeometryDataKHR {
                        aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                            // No need to touch the data pointer here, since this VkAccelerationStructureGeometryKHR is
                            // used for VkgetAccelerationStructureBuildSizes only.
                            stride: *stride as u64,
                            ..Default::default()
                        },
                    },
                    flags: *flags,
                    ..Default::default()
                },
            )
            .collect();
        unsafe {
            let build_size = allocator
                .device()
                .accel_struct_loader()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &vk::AccelerationStructureBuildGeometryInfoKHR {
                        ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                        flags: self.flags,
                        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                        geometry_count: self.geometries.len() as u32,
                        p_geometries: geometries.as_ptr(),
                        ..Default::default()
                    },
                    &self.geometry_primitive_counts,
                );
            let accel_struct = AccelerationStructure::new_blas_aabb(
                &allocator,
                build_size.acceleration_structure_size,
            )?;
            Ok(AccelerationStructureBuild {
                accel_struct,
                build_size,
                geometries: self.geometries.into_boxed_slice(),
                primitive_datasize: self.primitive_datasize,
            })
        }
    }
}
