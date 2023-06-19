use std::{ops::Range, sync::Arc};

use super::AccelerationStructure;
use crate::debug::DebugObject;
use crate::future::{
    use_shared_state, DisposeContainer, RenderData, RenderRes, SharedDeviceState,
    SharedDeviceStateHostContainer,
};
use crate::HasDevice;
use crate::{future::GPUCommandFuture, Allocator, BufferLike, ResidentBuffer};
use ash::vk;
use pin_project::pin_project;

pub struct AccelerationStructureBuild {
    pub accel_struct: AccelerationStructure,
    pub build_size: vk::AccelerationStructureBuildSizesInfoKHR,
    pub geometries: Box<[(Arc<ResidentBuffer>, usize, vk::GeometryFlagsKHR, u32)]>, // data, stride, flags, num_primitives
    pub primitive_datasize: usize,
}

/// Builds many acceleration structures in batch.
pub struct AccelerationStructureBatchBuilder<T> {
    allocator: Allocator,
    builds: Vec<(T, AccelerationStructureBuild)>,
}

impl<T> AccelerationStructureBatchBuilder<T> {
    pub fn new(allocator: Allocator, builds: Vec<(T, AccelerationStructureBuild)>) -> Self {
        Self { allocator, builds }
    }
    // build on the device.
    /// Instead of calling VkCmdBuildAccelerationStructure multiple times, it calls VkCmdBuildAccelerationStructure
    /// in batch mode, once for BLAS and once for TLAS, with a pipeline barrier inbetween.  
    pub fn build(self) -> BLASBuildFuture<T> {
        // Calculate the total number of geometries
        let total_num_geometries = self
            .builds
            .iter()
            .map(|(_, build)| build.geometries.len())
            .sum();
        let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> =
            Vec::with_capacity(total_num_geometries);
        let mut build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(total_num_geometries);
        let mut build_range_ptrs: Box<[*const vk::AccelerationStructureBuildRangeInfoKHR]> =
            vec![std::ptr::null(); self.builds.len()].into_boxed_slice();

        let scratch_buffer_alignment =
            self.allocator
                .device()
                .physical_device()
                .properties()
                .acceleration_structure
                .min_acceleration_structure_scratch_offset_alignment as usize;

        let scratch_buffers = self
            .builds
            .iter()
            .map(|(_, build)| {
                let mut scratch_buffer = self
                    .allocator
                    .create_device_buffer_uninit_aligned(
                        build.build_size.build_scratch_size,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        scratch_buffer_alignment as u64,
                    )
                    .unwrap();
                scratch_buffer
                    .set_name(&format!(
                        "BLAS Build Scratch Buffer for BLAS {:?}",
                        build.accel_struct.raw()
                    ))
                    .unwrap();
                scratch_buffer
            })
            .collect::<Vec<_>>();

        let build_infos = self
            .builds
            .iter()
            .enumerate()
            .map(|(i, (_, as_build))| {
                build_range_ptrs[i] = unsafe { build_ranges.as_ptr().add(build_ranges.len()) };

                // Add geometries
                build_ranges.extend(as_build.geometries.iter().map(|(_, _, _, a)| {
                    vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: *a,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    }
                }));

                let geometry_range: Range<usize> =
                    geometries.len()..(geometries.len() + as_build.geometries.len());
                // Insert geometries
                geometries.extend(aabbs_to_geometry_infos(&as_build.geometries));
                vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    flags: as_build.accel_struct.flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    dst_acceleration_structure: as_build.accel_struct.raw,
                    geometry_count: as_build.geometries.len() as u32,
                    p_geometries: unsafe { geometries.as_ptr().add(geometry_range.start) },
                    scratch_data: vk::DeviceOrHostAddressKHR {
                        device_address: scratch_buffers[i].device_address(),
                    },
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        assert_eq!(geometries.len(), total_num_geometries);
        assert_eq!(build_ranges.len(), total_num_geometries);
        let info = self
            .builds
            .into_iter()
            .map(|(info, a)| (info, a.accel_struct));
        BLASBuildFuture {
            scratch_buffers,
            accel_structs: info.collect(),
            geometries: geometries.into_boxed_slice(),
            build_infos,
            build_range_infos: build_ranges.into_boxed_slice(),
            build_range_ptrs,
        }
    }
}

#[pin_project]
pub struct BLASBuildFuture<T> {
    scratch_buffers: Vec<ResidentBuffer>,
    accel_structs: Vec<(T, AccelerationStructure)>,
    geometries: Box<[vk::AccelerationStructureGeometryKHR]>,
    build_infos: Box<[vk::AccelerationStructureBuildGeometryInfoKHR]>,
    build_range_infos: Box<[vk::AccelerationStructureBuildRangeInfoKHR]>,
    build_range_ptrs: Box<[*const vk::AccelerationStructureBuildRangeInfoKHR]>,
}

impl<T> GPUCommandFuture for BLASBuildFuture<T> {
    type Output = Vec<(T, AccelerationStructure)>;

    type RetainedState = DisposeContainer<Vec<ResidentBuffer>>;

    type RecycledState = ();

    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut crate::future::CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        assert_eq!(this.build_range_ptrs.len(), this.build_infos.len());
        ctx.record(|ctx, cmd_buffer| unsafe {
            (ctx.device()
                .accel_struct_loader()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                cmd_buffer,
                this.build_infos.len() as u32,
                this.build_infos.as_ptr(),
                this.build_range_ptrs.as_ptr(),
            )
        });
        let futs = std::mem::replace(this.accel_structs, Vec::new());
        std::task::Poll::Ready((
            futs,
            DisposeContainer::new(std::mem::replace(this.scratch_buffers, Vec::new())),
        ))
    }

    fn context(self: std::pin::Pin<&mut Self>, _ctx: &mut crate::future::StageContext) {}
}

fn aabbs_to_geometry_infos(
    geometries: &[(Arc<ResidentBuffer>, usize, vk::GeometryFlagsKHR, u32)],
) -> impl IntoIterator<Item = vk::AccelerationStructureGeometryKHR> + '_ {
    geometries.iter().map(
        |(data, stride, flags, _)| vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::AABBS,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                    data: vk::DeviceOrHostAddressConstKHR {
                        device_address: data.device_address(),
                    },
                    stride: *stride as u64,
                    ..Default::default()
                },
            },
            flags: *flags,
            ..Default::default()
        },
    )
}

pub struct TLASBuildInfo {
    allocator: Allocator,
    geometry_info: vk::AccelerationStructureGeometryKHR,
    build_info: vk::AccelerationStructureBuildGeometryInfoKHR,
    num_instances: u32,
    build_size: vk::AccelerationStructureBuildSizesInfoKHR,
}

impl TLASBuildInfo {
    pub fn new(
        allocator: Allocator,
        num_instances: u32,
        geometry_flags: vk::GeometryFlagsKHR,
        build_flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> Self {
        let geometry_info = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                    array_of_pointers: vk::FALSE,
                    data: Default::default(),
                    ..Default::default()
                },
            },
            flags: geometry_flags,
            ..Default::default()
        };
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: build_flags,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            geometry_count: 1,
            p_geometries: &geometry_info,
            ..Default::default()
        };
        let build_size = unsafe {
            allocator
                .device()
                .accel_struct_loader()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[num_instances],
                )
        };
        Self {
            allocator: allocator,
            geometry_info,
            build_info,
            num_instances,
            build_size,
        }
    }
}
impl TLASBuildInfo {
    pub fn build_for<T: BufferLike + RenderData>(
        mut self,
        buffer: RenderRes<T>,
    ) -> TLASBuildFuture<T> {
        self.geometry_info.geometry = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                array_of_pointers: vk::FALSE,
                data: vk::DeviceOrHostAddressConstKHR {
                    device_address: buffer.inner().device_address(),
                },
                ..Default::default()
            },
        };

        let acceleration_structure = AccelerationStructure::new_tlas(
            &self.allocator,
            self.build_size.acceleration_structure_size,
        )
        .unwrap();

        self.build_info.dst_acceleration_structure = acceleration_structure.raw();
        TLASBuildFuture {
            allocator: self.allocator,
            input_buffer: Some(buffer),
            acceleration_structure: Some(RenderRes::new(acceleration_structure)),
            geometry_info: self.geometry_info,
            build_info: self.build_info,
            num_instances: self.num_instances,
            build_size: self.build_size,
        }
    }
}

#[pin_project]
pub struct TLASBuildFuture<T: BufferLike + RenderData> {
    allocator: Allocator,
    input_buffer: Option<RenderRes<T>>,
    acceleration_structure: Option<RenderRes<AccelerationStructure>>,
    geometry_info: vk::AccelerationStructureGeometryKHR,
    build_info: vk::AccelerationStructureBuildGeometryInfoKHR,
    num_instances: u32,
    build_size: vk::AccelerationStructureBuildSizesInfoKHR,
}

impl<T: BufferLike + Send + RenderData> GPUCommandFuture for TLASBuildFuture<T> {
    type Output = RenderRes<AccelerationStructure>;
    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut crate::future::CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        let alignment = this
            .allocator
            .device()
            .physical_device()
            .properties()
            .acceleration_structure
            .min_acceleration_structure_scratch_offset_alignment;
        let scratch_buffer = use_shared_state(
            recycled_state,
            |old| {
                let old_size = old.map(|a: &ResidentBuffer| a.size()).unwrap_or(0);
                this.allocator
                    .create_device_buffer_uninit_aligned(
                        this.build_size.build_scratch_size.max(old_size),
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                        alignment as u64,
                    )
                    .unwrap()
            },
            |old| old.size() < this.build_size.build_scratch_size,
        );
        this.build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.inner().device_address(),
        };
        this.build_info.p_geometries = this.geometry_info; // Fix link
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: *this.num_instances,
            primitive_offset: 0,
            ..Default::default()
        };
        let ptr = &build_range_info as *const vk::AccelerationStructureBuildRangeInfoKHR;

        ctx.record(|ctx, command_buffer| unsafe {
            (ctx.device()
                .accel_struct_loader()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                command_buffer, 1, this.build_info, &ptr
            );
        });
        std::task::Poll::Ready((
            this.acceleration_structure.take().unwrap(),
            (this.input_buffer.take().unwrap(), scratch_buffer),
        ))
    }

    type RetainedState = (
        RenderRes<T>,                                 // input buffer
        RenderRes<SharedDeviceState<ResidentBuffer>>, // scratch buffer
    );

    type RecycledState = Option<SharedDeviceStateHostContainer<ResidentBuffer>>;

    fn context(mut self: std::pin::Pin<&mut Self>, ctx: &mut crate::future::StageContext) {
        // TODO: what's the optimal access flags to use here?
        ctx.write(
            self.acceleration_structure.as_mut().unwrap(),
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
        );
        ctx.read(
            self.input_buffer.as_ref().unwrap(),
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::MEMORY_READ_KHR,
        );
        // TODO: indicate read&write on scratch buffer. Rn it's not tracked.
    }
}
