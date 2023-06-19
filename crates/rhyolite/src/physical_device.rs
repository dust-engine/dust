use crate::{QueueInfo, Queues, Surface, Version};

use super::{Device, Instance};
use ash::{prelude::VkResult, vk};
use core::ffi::{c_char, c_void};
use std::{
    ffi::CStr,
    ops::{Deref, DerefMut},
    sync::Arc,
};
pub struct PhysicalDevice {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    properties: Box<PhysicalDeviceProperties>,
    features: Box<PhysicalDeviceFeatures>,
    memory_properties: Box<vk::PhysicalDeviceMemoryProperties>,
    memory_model: PhysicalDeviceMemoryModel,
}

pub struct DeviceCreateInfo<'a, F: Fn(u32) -> Vec<f32>> {
    pub enabled_layer_names: &'a [*const c_char],
    pub enabled_extension_names: &'a [*const c_char],
    pub enabled_features: Box<PhysicalDeviceFeatures>,
    pub queue_create_callback: F,
}
impl<'a, F: Fn(u32) -> Vec<f32>> DeviceCreateInfo<'a, F> {
    pub fn with_queue_create_callback(callback: F) -> Self {
        Self {
            enabled_layer_names: &[],
            enabled_extension_names: &[],
            enabled_features: Default::default(),
            queue_create_callback: callback,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PhysicalDeviceMemoryModel {
    Discrete,
    Bar,
    ResizableBar,
    UMA,
}

impl PhysicalDevice {
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub fn enumerate(instance: &Arc<Instance>) -> VkResult<Vec<Self>> {
        // Safety: No Host Syncronization rules for vkEnumeratePhysicalDevices.
        // It should be OK to call this method and obtain multiple copies of VkPhysicalDevice,
        // because nothing except vkDestroyInstance require exclusive access to VkPhysicalDevice.
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let results = physical_devices
            .into_iter()
            .map(|pdevice| {
                let properties = PhysicalDeviceProperties::new(instance, pdevice);
                let features = PhysicalDeviceFeatures::new(instance, pdevice);
                let memory_properties =
                    unsafe { instance.get_physical_device_memory_properties(pdevice) };
                let types = &memory_properties.memory_types
                    [0..memory_properties.memory_type_count as usize];
                let heaps = &memory_properties.memory_heaps
                    [0..memory_properties.memory_heap_count as usize];

                let memory_model =
                    if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        PhysicalDeviceMemoryModel::UMA
                    } else {
                        let bar_heap = types
                            .iter()
                            .find(|ty| {
                                ty.property_flags.contains(
                                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                                        | vk::MemoryPropertyFlags::HOST_VISIBLE,
                                ) && heaps[ty.heap_index as usize]
                                    .flags
                                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                            })
                            .map(|a| &heaps[a.heap_index as usize]);
                        if let Some(bar_heap) = bar_heap {
                            if bar_heap.size <= 256 * 1024 * 1024 {
                                // regular 256MB bar
                                PhysicalDeviceMemoryModel::Bar
                            } else {
                                PhysicalDeviceMemoryModel::ResizableBar
                            }
                        } else {
                            // Can't find a BAR heap
                            PhysicalDeviceMemoryModel::Discrete
                        }
                    };
                PhysicalDevice {
                    instance: instance.clone(), // Retain reference to Instance here
                    physical_device: pdevice,   // Borrow VkPhysicalDevice from Instance
                    // Borrow is safe because we retain a reference to Instance here,
                    // ensuring that Instance wouldn't be dropped as long as the borrow is still there.
                    properties,
                    features,
                    memory_model,
                    memory_properties: Box::new(memory_properties),
                }
            })
            .collect();
        Ok(results)
    }
    pub fn properties(&self) -> &PhysicalDeviceProperties {
        &self.properties
    }
    pub fn features(&self) -> &PhysicalDeviceFeatures {
        &self.features
    }
    pub fn memory_model(&self) -> PhysicalDeviceMemoryModel {
        self.memory_model
    }
    pub fn memory_types(&self) -> &[vk::MemoryType] {
        &self.memory_properties.memory_types[0..self.memory_properties.memory_type_count as usize]
    }
    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory_properties.memory_heaps[0..self.memory_properties.memory_heap_count as usize]
    }
    pub fn get_surface_formats(&self, surface: &Surface) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        assert!(Arc::ptr_eq(surface.instance(), self.instance()));
        unsafe {
            surface
                .instance()
                .surface_loader()
                .get_physical_device_surface_formats(self.raw(), surface.raw())
        }
    }
    pub fn image_format_properties(
        &self,
        format_info: &vk::PhysicalDeviceImageFormatInfo2,
    ) -> VkResult<Option<vk::ImageFormatProperties2>> {
        let mut out = vk::ImageFormatProperties2::default();
        unsafe {
            match self.instance.get_physical_device_image_format_properties2(
                self.physical_device,
                format_info,
                &mut out,
            ) {
                Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(None),
                Ok(_) => Ok(Some(out)),
                Err(_) => panic!(),
            }
        }
    }
    pub(crate) fn get_queue_family_properties(&self) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.physical_device)
        }
    }
    pub fn create_device(
        self,
        mut infos: DeviceCreateInfo<'_, impl Fn(u32) -> Vec<f32>>,
    ) -> VkResult<(Arc<Device>, Queues)> {
        let mut num_queue_families: u32 = 0;
        unsafe {
            (self
                .instance
                .fp_v1_0()
                .get_physical_device_queue_family_properties)(
                self.physical_device,
                &mut num_queue_families,
                std::ptr::null_mut(),
            );
        }
        assert!(num_queue_families > 0);
        let mut list_priorities = Vec::new();
        let queue_create_infos: Vec<_> = (0..num_queue_families)
            .filter_map(|queue_family_index| {
                let priorities = (infos.queue_create_callback)(queue_family_index);
                if priorities.is_empty() {
                    return None;
                }
                let queue_count = priorities.len() as u32;
                let p_queue_priorities = priorities.as_ptr();
                list_priorities.push(priorities);
                Some(vk::DeviceQueueCreateInfo {
                    queue_family_index,
                    queue_count,
                    p_queue_priorities,
                    ..Default::default()
                })
            })
            .collect();
        infos.enabled_features.fix_links();
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_layer_names(infos.enabled_layer_names)
            .enabled_extension_names(infos.enabled_extension_names)
            .push_next(&mut infos.enabled_features.inner)
            .build();
        tracing::info!("Creating device with {:?}", create_info);
        let queue_info = QueueInfo::new(num_queue_families, &queue_create_infos);

        let device = Arc::new(Device::new(
            self.instance.clone(),
            self,
            create_info,
            queue_info,
        )?);
        drop(list_priorities);

        let queues = unsafe { Queues::new(device.clone()) };
        Ok((device, queues))
    }
}

pub struct PhysicalDeviceProperties {
    pub inner: vk::PhysicalDeviceProperties2,
    pub v11: vk::PhysicalDeviceVulkan11Properties,
    pub v12: vk::PhysicalDeviceVulkan12Properties,
    pub v13: vk::PhysicalDeviceVulkan13Properties,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    pub ray_tracing: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Box<PhysicalDeviceProperties> {
        let mut this = Box::pin(Self {
            inner: vk::PhysicalDeviceProperties2::default(),
            v11: vk::PhysicalDeviceVulkan11Properties::default(),
            v12: vk::PhysicalDeviceVulkan12Properties::default(),
            v13: vk::PhysicalDeviceVulkan13Properties::default(),
            acceleration_structure: vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default(),
            ray_tracing: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default(),
        });
        this.inner.p_next = &mut this.v11 as *mut _ as *mut c_void;
        this.v11.p_next = &mut this.v12 as *mut _ as *mut c_void;
        this.v12.p_next = &mut this.v13 as *mut _ as *mut c_void;
        this.v13.p_next = &mut this.acceleration_structure as *mut _ as *mut c_void;
        this.acceleration_structure.p_next = &mut this.ray_tracing as *mut _ as *mut c_void;
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut this.inner);
        }
        std::pin::Pin::into_inner(this)
    }
    pub fn device_name(&self) -> &CStr {
        unsafe {
            CStr::from_bytes_until_nul(std::slice::from_raw_parts(
                self.inner.properties.device_name.as_ptr() as *const _,
                self.inner.properties.device_name.len(),
            ))
            .unwrap()
        }
    }
    pub fn api_version(&self) -> Version {
        Version(self.inner.properties.api_version)
    }
    pub fn driver_version(&self) -> Version {
        Version(self.inner.properties.driver_version)
    }
}
impl Deref for PhysicalDeviceProperties {
    type Target = vk::PhysicalDeviceProperties;
    fn deref(&self) -> &Self::Target {
        &self.inner.properties
    }
}
impl DerefMut for PhysicalDeviceProperties {
    fn deref_mut(&mut self) -> &mut vk::PhysicalDeviceProperties {
        &mut self.inner.properties
    }
}

#[derive(Default, Clone)]
pub struct PhysicalDeviceFeatures {
    pub inner: vk::PhysicalDeviceFeatures2,
    pub v11: vk::PhysicalDeviceVulkan11Features,
    pub v12: vk::PhysicalDeviceVulkan12Features,
    pub v13: vk::PhysicalDeviceVulkan13Features,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    pub ray_tracing: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    pub shader_atomics: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT,
}
unsafe impl Send for PhysicalDeviceFeatures {}
unsafe impl Sync for PhysicalDeviceFeatures {}
impl PhysicalDeviceFeatures {
    fn fix_links(&mut self) {
        self.inner.p_next = &mut self.v11 as *mut _ as *mut c_void;
        self.v11.p_next = &mut self.v12 as *mut _ as *mut c_void;
        self.v12.p_next = &mut self.v13 as *mut _ as *mut c_void;
        self.v13.p_next = &mut self.acceleration_structure as *mut _ as *mut c_void;
        self.acceleration_structure.p_next = &mut self.ray_tracing as *mut _ as *mut c_void;
        self.ray_tracing.p_next = &mut self.shader_atomics as *mut _ as *mut c_void;
    }
    fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Box<PhysicalDeviceFeatures> {
        let mut this = Box::pin(Self {
            inner: vk::PhysicalDeviceFeatures2::default(),
            v11: vk::PhysicalDeviceVulkan11Features::default(),
            v12: vk::PhysicalDeviceVulkan12Features::default(),
            v13: vk::PhysicalDeviceVulkan13Features::default(),
            acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default(),
            ray_tracing: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default(),
            shader_atomics: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default(),
        });
        this.fix_links();
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut this.inner);
        }
        std::pin::Pin::into_inner(this)
    }
}
impl Deref for PhysicalDeviceFeatures {
    type Target = vk::PhysicalDeviceFeatures;
    fn deref(&self) -> &Self::Target {
        &self.inner.features
    }
}
impl DerefMut for PhysicalDeviceFeatures {
    fn deref_mut(&mut self) -> &mut vk::PhysicalDeviceFeatures {
        &mut self.inner.features
    }
}

pub struct MemoryType {
    pub property_flags: vk::MemoryPropertyFlags,
    pub heap_index: u32,
}

pub struct MemoryHeap {
    pub size: vk::DeviceSize,
    pub flags: vk::MemoryHeapFlags,
    pub budget: vk::DeviceSize,
    pub usage: vk::DeviceSize,
}
