use ash::vk;

use crate::renderer::RenderContext;
use dust_core::CameraProjection;
use dust_core::SunLight;
use glam::{Mat3, TransformRT, Vec3};
use std::mem::size_of;
use std::sync::Arc;
use vk_mem as vma;

struct StagingState {
    buffer: vk::Buffer,
    allocation: vma::Allocation,
    allocation_info: vma::AllocationInfo,
    queue: vk::Queue,
    queue_family: u32,
}

const SHARED_BUFFER_FRAME_UPDATE_SIZE: u64 = 256;
#[repr(C)]
pub struct StagingStateLayout {
    pub camera_view_col0: [f32; 3],
    pub padding0: f32,
    pub camera_view_col1: [f32; 3],
    pub padding1: f32,
    pub camera_view_col2: [f32; 3],
    pub padding2: f32,

    pub camera_position: Vec3,
    pub tan_half_fov: f32,
}

/**
SharedBuffer Layout:
0-128: Vertex and Index buffer of the cube
128 - 256: Camera Data
256-512: Light Data

StagingBuffer Layout:
0 - 128: Camera Data
128-256: Light Data

Staging area is fixed 128 bytes
*/
pub struct SharedBuffer {
    context: Arc<RenderContext>,
    staging: Option<StagingState>,
    pub buffer: vk::Buffer,
    allocation: vma::Allocation,
    allocation_info: vma::AllocationInfo,
    layout: &'static mut StagingStateLayout,
}
unsafe impl Send for SharedBuffer {}
unsafe impl Sync for SharedBuffer {}

impl SharedBuffer {
    pub unsafe fn new(
        context: Arc<RenderContext>,
        allocator: &vma::Allocator,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue: vk::Queue,
        queue_family: u32,
    ) -> Self {
        let span = tracing::info_span!("shared_buffer_new");
        let _enter = span.enter();

        let _device = &context.device;
        let _needs_staging = !memory_properties.memory_types
            [0..memory_properties.memory_type_count as usize]
            .iter()
            .any(|ty| {
                ty.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            });
        let needs_staging = true;
        tracing::info!("SharedBuffer using staging: {:?}", needs_staging);
        let (buffer, allocation, allocation_info) = allocator
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .flags(vk::BufferCreateFlags::empty())
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size(size_of::<StagingStateLayout>() as u64)
                    .usage(
                        vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::VERTEX_BUFFER
                            | if needs_staging {
                                vk::BufferUsageFlags::TRANSFER_DST
                            } else {
                                vk::BufferUsageFlags::empty()
                            },
                    ),
                &vma::AllocationCreateInfo {
                    usage: if needs_staging {
                        vma::MemoryUsage::GpuOnly
                    } else {
                        vma::MemoryUsage::CpuToGpu
                    },
                    flags: vma::AllocationCreateFlags::MAPPED,
                    required_flags: vk::MemoryPropertyFlags::empty(),
                    preferred_flags: vk::MemoryPropertyFlags::empty(),
                    memory_type_bits: 0,
                    pool: None,
                    user_data: None,
                },
            )
            .unwrap();

        let staging = if needs_staging {
            let (staging_buffer, allocation, allocation_info) = allocator
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size_of::<StagingStateLayout>() as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .flags(vk::BufferCreateFlags::empty())
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vma::AllocationCreateInfo {
                        usage: vma::MemoryUsage::CpuToGpu,
                        flags: vma::AllocationCreateFlags::MAPPED,
                        ..Default::default()
                    },
                )
                .unwrap();

            Some(StagingState {
                buffer: staging_buffer,
                allocation,
                allocation_info,
                queue,
                queue_family,
            })
        } else {
            None
        };

        let ptr = if let Some(staging) = staging.as_ref() {
            staging.allocation_info.get_mapped_data()
        } else {
            allocation_info.get_mapped_data()
        } as *mut StagingStateLayout;
        assert_ne!(ptr, std::ptr::null_mut());

        let shared_buffer = Self {
            staging,
            buffer,
            allocation,
            context,
            layout: &mut *ptr,
            allocation_info,
        };
        shared_buffer
    }

    pub fn write_camera(&mut self, camera_projection: &CameraProjection, transform: &TransformRT) {
        let rotation_matrix = Mat3::from_quat(transform.rotation).to_cols_array_2d();
        self.layout.camera_view_col0 = rotation_matrix[0];
        self.layout.camera_view_col1 = rotation_matrix[1];
        self.layout.camera_view_col2 = rotation_matrix[2];
        self.layout.camera_position = transform.translation;
        self.layout.tan_half_fov = (camera_projection.fov / 2.0).tan();
    }

    pub fn write_light(&mut self, _sunlight: &SunLight) {
        //self.layout.sunlight = sunlight.clone();
    }

    pub unsafe fn record_cmd_buffer_copy(&mut self, cmd_buffer: vk::CommandBuffer) {
        let size = size_of::<StagingStateLayout>() as u64;
        if let Some(staging_buffer) = self.staging.as_ref() {
            self.context.device.cmd_copy_buffer(
                cmd_buffer,
                staging_buffer.buffer,
                self.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
        }
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // If a memory object is mapped at the time it is freed, it is implicitly unmapped.
        unsafe {
            if let Some(staging) = self.staging.as_ref() {
                self.context.device.destroy_buffer(staging.buffer, None);
            }
            self.context.device.destroy_buffer(self.buffer, None);
        }
    }
}
