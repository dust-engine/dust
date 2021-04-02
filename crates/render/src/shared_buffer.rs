use ash::vk;

use ash::version::DeviceV1_0;
use dust_core::CameraProjection;
use dust_core::SunLight;
use glam::{Mat4, TransformRT, Vec3};
use std::mem::size_of;

struct StagingState {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    queue: vk::Queue,
    queue_family: u32,
}

const SHARED_BUFFER_FRAME_UPDATE_SIZE: u64 = 256;
pub struct StagingStateLayout {
    pub view_proj: Mat4,
    pub proj: Mat4,
    // -- 128
    pub sunlight: SunLight,
    _padding1: [f32; 8],
    _padding2: [f32; 16],
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
    device: ash::Device,
    memory: vk::DeviceMemory,
    staging: Option<StagingState>,
    pub buffer: vk::Buffer,
    layout: &'static mut StagingStateLayout,
    static_data_dirty: bool,
}
unsafe impl Send for SharedBuffer {}
unsafe impl Sync for SharedBuffer {}

impl SharedBuffer {
    pub unsafe fn new(
        device: ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue: vk::Queue,
        queue_family: u32,
    ) -> Self {
        let span = tracing::info_span!("shared_buffer_new");
        let _enter = span.enter();

        let needs_staging = !memory_properties.memory_types
            [0..memory_properties.memory_type_count as usize]
            .iter()
            .any(|ty| {
                ty.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
            });
        tracing::info!("SharedBuffer using staging: {:?}", needs_staging);
        let buffer = device
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
                None,
            )
            .unwrap();
        let buffer_requirements = device.get_buffer_memory_requirements(buffer);
        let buffer_memory_type: u32 = memory_properties.memory_types
            [0..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .position(|(i, ty)| {
                if buffer_requirements.memory_type_bits & (1 << i) == 0 {
                    return false;
                }
                if ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && (needs_staging
                        ^ ty.property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
                {
                    return true;
                }
                return false;
            })
            .unwrap() as u32;
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .memory_type_index(buffer_memory_type)
                    .allocation_size(buffer_requirements.size),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buffer, memory, 0).unwrap();

        let staging = if needs_staging {
            let staging_buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size_of::<StagingStateLayout>() as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .flags(vk::BufferCreateFlags::empty())
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let staging_buffer_requirements = device.get_buffer_memory_requirements(staging_buffer);
            let memory_type = memory_properties.memory_types
                [0..memory_properties.memory_type_count as usize]
                .iter()
                .enumerate()
                .position(|(i, ty)| {
                    if staging_buffer_requirements.memory_type_bits & (1 << i) == 0 {
                        return false;
                    }
                    if ty
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                        && !ty
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    {
                        return true;
                    }
                    return false;
                })
                .unwrap() as u32;
            let staging_mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(staging_buffer_requirements.size)
                        .memory_type_index(memory_type)
                        .build(),
                    None,
                )
                .unwrap();
            device
                .bind_buffer_memory(staging_buffer, staging_mem, 0)
                .unwrap();
            Some(StagingState {
                memory: staging_mem,
                buffer: staging_buffer,
                queue,
                queue_family,
            })
        } else {
            None
        };

        let mem_to_write = if let Some(staging) = staging.as_ref() {
            staging.memory
        } else {
            memory
        };
        let ptr = device
            .map_memory(
                mem_to_write,
                0,
                size_of::<StagingStateLayout>() as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap() as *mut StagingStateLayout;

        let shared_buffer = Self {
            memory,
            staging,
            buffer,
            device,
            layout: &mut *ptr,
            static_data_dirty: true,
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
        let rotation = Mat4::from_rotation_translation(transform.rotation, Vec3::ZERO);
        let transform = Mat4::from_rotation_translation(transform.rotation, transform.translation);
        let view_proj = camera_projection.get_projection_matrix(aspect_ratio) * rotation.inverse();
        self.layout.view_proj = view_proj;
        self.layout.proj = transform;
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
        let mem_to_write = if let Some(staging) = self.staging.as_ref() {
            staging.memory
        } else {
            self.memory
        };
        self.device
            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
                .memory(mem_to_write)
                .size(size)
                .offset(0)
                .build()])
            .unwrap();
        if let Some(staging_buffer) = self.staging.as_ref() {
            self.device.cmd_copy_buffer(
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
                self.device.destroy_buffer(staging.buffer, None);
                self.device.free_memory(staging.memory, None);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
