use ash::prelude::VkResult;
use ash::vk;

use crate::utils::either::Either;
use crate::Instance;
use crate::PhysicalDevice;
use crate::QueueInfo;

use std::collections::BTreeSet;
use std::ffi::CStr;
use std::ops::Deref;
use std::sync::Arc;

pub trait HasDevice {
    fn device(&self) -> &Arc<Device>;
    fn physical_device(&self) -> &PhysicalDevice {
        &self.device().physical_device
    }
    fn instance(&self) -> &Arc<Instance> {
        self.device().physical_device.instance()
    }
}

impl<A: HasDevice, B: HasDevice> HasDevice for Either<A, B> {
    fn device(&self) -> &Arc<Device> {
        match self {
            Either::Left(a) => a.device(),
            Either::Right(a) => a.device(),
        }
    }
}

pub struct Device {
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
    device: ash::Device,

    swapchain_loader: Option<Box<ash::extensions::khr::Swapchain>>,
    rtx_loader: Option<Box<ash::extensions::khr::RayTracingPipeline>>,
    accel_struct_loader: Option<Box<ash::extensions::khr::AccelerationStructure>>,
    deferred_host_operation_loader: Option<Box<ash::extensions::khr::DeferredHostOperations>>,

    queue_info: QueueInfo,
}

impl Device {
    pub fn queue_info(&self) -> &QueueInfo {
        &self.queue_info
    }
    pub fn swapchain_loader(&self) -> &ash::extensions::khr::Swapchain {
        self.swapchain_loader
            .as_ref()
            .expect("VkSwapchainKHR extension was not enabled")
    }
    pub fn rtx_loader(&self) -> &ash::extensions::khr::RayTracingPipeline {
        self.rtx_loader
            .as_ref()
            .expect("VkRayTracingPipelineKHR extension was not enabled")
    }
    pub fn accel_struct_loader(&self) -> &ash::extensions::khr::AccelerationStructure {
        self.accel_struct_loader
            .as_ref()
            .expect("VkAccelerationStructureKHR extensions was not enabled")
    }
    pub fn deferred_host_operation_loader(&self) -> &ash::extensions::khr::DeferredHostOperations {
        self.deferred_host_operation_loader
            .as_ref()
            .expect("VkDeferredHostOperation extension was not enabled")
    }
    pub(crate) fn new(
        instance: Arc<Instance>,
        physical_device: PhysicalDevice,
        create_info: vk::DeviceCreateInfo,
        queue_info: QueueInfo,
    ) -> VkResult<Self> {
        // Safety: No Host Syncronization rules for VkCreateDevice.
        // Device retains a reference to Instance, ensuring that Instance is dropped later than Device.
        let device = unsafe { instance.create_device(physical_device.raw(), &create_info, None) }?;
        let extensions: BTreeSet<&CStr> = unsafe {
            std::slice::from_raw_parts(
                create_info.pp_enabled_extension_names,
                create_info.enabled_extension_count as usize,
            )
            .iter()
            .map(|a| CStr::from_ptr(*a))
            .collect()
        };
        let swapchain_loader = if extensions.contains(ash::extensions::khr::Swapchain::name()) {
            Some(Box::new(ash::extensions::khr::Swapchain::new(
                &instance, &device,
            )))
        } else {
            None
        };
        let rtx_loader = if extensions.contains(ash::extensions::khr::RayTracingPipeline::name()) {
            Some(Box::new(ash::extensions::khr::RayTracingPipeline::new(
                &instance, &device,
            )))
        } else {
            None
        };
        let accel_struct_loader =
            if extensions.contains(ash::extensions::khr::AccelerationStructure::name()) {
                Some(Box::new(ash::extensions::khr::AccelerationStructure::new(
                    &instance, &device,
                )))
            } else {
                None
            };

        let deferred_host_operation_loader =
            if extensions.contains(ash::extensions::khr::DeferredHostOperations::name()) {
                Some(Box::new(ash::extensions::khr::DeferredHostOperations::new(
                    &instance, &device,
                )))
            } else {
                None
            };

        Ok(Self {
            instance,
            physical_device,
            device,
            swapchain_loader,
            rtx_loader,
            accel_struct_loader,
            deferred_host_operation_loader,
            queue_info,
        })
    }
    pub fn instance(&self) -> &Arc<Instance> {
        self.physical_device.instance()
    }
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::info!(device = ?self.device.handle(), "drop device");
        // Safety: Host Syncronization rule for vkDestroyDevice:
        // - Host access to device must be externally synchronized.
        // - Host access to all VkQueue objects created from device must be externally synchronized
        // We have &mut self and therefore exclusive control on device.
        // VkQueue objects may not exist at this point, because Queue retains an Arc to Device.
        // If there still exist a Queue, the Device wouldn't be dropped.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
