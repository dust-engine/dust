use ash::vk;

use crate::renderer::RenderContext;
use ash::version::DeviceV1_0;
use dust_core::CameraProjection;
use dust_core::SunLight;
use glam::{Mat4, TransformRT, Vec3, Vec4};
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
pub struct StagingStateLayout {
    pub view_proj: Mat4,
    pub rotation_view_proj: Mat4,
    pub camera_position: Vec3,
    placeholder: f32,
    pub forward: Vec3,
    placeholder2: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect_ratio: f32,
    // -- 192
    pub sunlight: SunLight,
    _padding1: [f32; 8],
    // -- 256
    pub vertex_buffer: [(f32, f32, f32); 8],
    _padding3: f32,
    pub index_buffer: [u16; 14],
    // -- 384
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
    static_data_dirty: bool,
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

        let device = &context.device;
        let needs_staging = !memory_properties.memory_types
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
            static_data_dirty: true,
            allocation_info,
        };
        shared_buffer
    }

    pub unsafe fn copy_vertex_index(
        &mut self,
        vertex_buffer: &[(f32, f32, f32); 8],
        index_buffer: &[u16; 14],
    ) {
        self.layout.vertex_buffer = vertex_buffer.clone();
        self.layout.index_buffer = index_buffer.clone();
    }

    pub fn write_camera(
        &mut self,
        camera_projection: &CameraProjection,
        transform: &TransformRT,
        aspect_ratio: f32,
    ) {
        let rotation_only_proj_matrix = Mat4::from_rotation_translation(transform.rotation, Vec3::ZERO);
        let proj_matrix = Mat4::from_rotation_translation(transform.rotation, transform.translation);
        let forward_vector = transform.rotation * Vec3::new(0.0, 0.0, -1.0);

        let cubed_projection_matrix = CameraProjection {
            fov: camera_projection.fov,
            near: 0.5,
            far: 2.0,
        };
        self.layout.view_proj =
            camera_projection.get_projection_matrix(aspect_ratio) * proj_matrix.inverse(); // The normal ViewProj matrix
        self.layout.rotation_view_proj =
            cubed_projection_matrix.get_projection_matrix(aspect_ratio) * rotation_only_proj_matrix.inverse();
        self.layout.camera_position = transform.translation;
        self.layout.forward = forward_vector.normalize();
        self.layout.far = camera_projection.far;
        self.layout.near = camera_projection.near;
        self.layout.fov = camera_projection.fov;
        self.layout.aspect_ratio = aspect_ratio;
    }

    pub fn write_light(&mut self, sunlight: &SunLight) {
        self.layout.sunlight = sunlight.clone();
    }

    pub unsafe fn record_cmd_buffer_copy(&mut self, cmd_buffer: vk::CommandBuffer) {
        let size = if self.static_data_dirty {
            self.static_data_dirty = false;
            size_of::<StagingStateLayout>() as u64
        } else {
            SHARED_BUFFER_FRAME_UPDATE_SIZE
        };
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
