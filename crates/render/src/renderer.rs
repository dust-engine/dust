use crate::device_info::DeviceInfo;

use ash::vk;
use std::ffi::CStr;

use std::sync::Arc;

use std::borrow::Cow;

pub struct RenderContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,

    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::extensions::khr::Surface,
    pub debug_messenger: Option<(ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT)>,

    pub graphics_queue: vk::Queue,
    pub transfer_binding_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub transfer_binding_queue_family: u32,
}
pub struct Renderer {
    pub context: Arc<RenderContext>,
    pub physical_device: vk::PhysicalDevice,

    pub graphics_queue: vk::Queue,
    pub transfer_binding_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub transfer_binding_queue_family: u32,
    pub info: DeviceInfo,
}

unsafe fn display_debug_utils_label_ext(
    label_structs: *mut vk::DebugUtilsLabelEXT,
    count: usize,
) -> Option<String> {
    if count == 0 {
        return None;
    }

    Some(
        std::slice::from_raw_parts::<vk::DebugUtilsLabelEXT>(label_structs, count)
            .iter()
            .flat_map(|dul_obj| {
                dul_obj
                    .p_label_name
                    .as_ref()
                    .map(|lbl| CStr::from_ptr(lbl).to_string_lossy().into_owned())
            })
            .collect::<Vec<String>>()
            .join(", "),
    )
}

unsafe fn display_debug_utils_object_name_info_ext(
    info_structs: *mut vk::DebugUtilsObjectNameInfoEXT,
    count: usize,
) -> Option<String> {
    if count == 0 {
        return None;
    }

    //TODO: use color field of vk::DebugUtilsLabelExt in a meaningful way?
    Some(
        std::slice::from_raw_parts::<vk::DebugUtilsObjectNameInfoEXT>(info_structs, count)
            .iter()
            .map(|obj_info| {
                let object_name = obj_info
                    .p_object_name
                    .as_ref()
                    .map(|name| CStr::from_ptr(name).to_string_lossy().into_owned());

                match object_name {
                    Some(name) => format!(
                        "(type: {:?}, hndl: 0x{:x}, name: {})",
                        obj_info.object_type, obj_info.object_handle, name
                    ),
                    None => format!(
                        "(type: {:?}, hndl: 0x{:x})",
                        obj_info.object_type, obj_info.object_handle
                    ),
                }
            })
            .collect::<Vec<String>>()
            .join(", "),
    )
}

unsafe extern "system" fn debug_utils_messenger_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if std::thread::panicking() {
        return vk::FALSE;
    }
    let callback_data = *p_callback_data;

    let message_severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Trace,
        _ => log::Level::Warn,
    };
    let message_type = &format!("{:?}", message_type);
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let additional_info: [(&str, Option<String>); 3] = [
        (
            "queue info",
            display_debug_utils_label_ext(
                callback_data.p_queue_labels as *mut _,
                callback_data.queue_label_count as usize,
            ),
        ),
        (
            "cmd buf info",
            display_debug_utils_label_ext(
                callback_data.p_cmd_buf_labels as *mut _,
                callback_data.cmd_buf_label_count as usize,
            ),
        ),
        (
            "object info",
            display_debug_utils_object_name_info_ext(
                callback_data.p_objects as *mut _,
                callback_data.object_count as usize,
            ),
        ),
    ];

    {
        let mut msg = format!(
            "\n{} [{} (0x{:x})] : {}",
            message_type, message_id_name, message_id_number, message
        );

        for &(info_label, ref info) in additional_info.iter() {
            if let Some(ref data) = *info {
                msg = format!("{}\n{}: {}", msg, info_label, data);
            }
        }
    }

    log!(message_severity, "{}\n", {
        let mut msg = format!(
            "\n{} [{} (0x{:x})] : {}",
            message_type, message_id_name, message_id_number, message
        );

        for &(info_label, ref info) in additional_info.iter() {
            if let Some(ref data) = *info {
                msg = format!("{}\n{}: {}", msg, info_label, data);
            }
        }
        msg
    });

    vk::FALSE
}

impl Renderer {
    /// Returns a Renderer and its associated BlockAllocator.
    /// Note that the two objects must be dropped at the same time.
    /// The application needs to ensure that when the Renderer was dropped,
    /// the BlockAllocator will not be used anymore.
    pub unsafe fn new(window_handle: &impl raw_window_handle::HasRawWindowHandle) -> Renderer {
        let entry = ash::Entry::new().unwrap();
        let instance_extensions = entry.enumerate_instance_extension_properties().unwrap();

        let mut extensions = ash_window::enumerate_required_extensions(window_handle).unwrap();
        extensions.push(ash::extensions::ext::DebugUtils::name());

        let layers: [&CStr; 0] = [];
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(
                        &vk::ApplicationInfo::builder()
                            .application_name(&CStr::from_bytes_with_nul_unchecked(
                                b"Dust Application\0",
                            ))
                            .application_version(0)
                            .engine_name(&CStr::from_bytes_with_nul_unchecked(b"Dust Engine\0"))
                            .engine_version(0)
                            .api_version(vk::make_version(1, 2, 0)),
                    )
                    .enabled_layer_names(&layers.map(|str| str.as_ptr() as *const i8))
                    .enabled_extension_names(
                        &extensions
                            .iter()
                            .map(|&str| str.as_ptr())
                            .collect::<Vec<_>>(),
                    ),
                None,
            )
            .unwrap();

        let debug_messenger = {
            // make sure VK_EXT_debug_utils is available
            if instance_extensions.iter().any(|props| unsafe {
                CStr::from_ptr(props.extension_name.as_ptr())
                    == ash::extensions::ext::DebugUtils::name()
            }) {
                let ext = ash::extensions::ext::DebugUtils::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                    .pfn_user_callback(Some(debug_utils_messenger_callback));
                let handle = ext.create_debug_utils_messenger(&info, None).unwrap();
                Some((ext, handle))
            } else {
                None
            }
        };

        let surface = ash_window::create_surface(&entry, &instance, window_handle, None).unwrap();
        let available_physical_devices: Vec<_> = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .map(|physical_device| {
                let device_info = DeviceInfo::new(&entry, &instance, physical_device);
                (physical_device, device_info)
            })
            .filter(|(_physical_device, device_info)| {
                device_info.features.sparse_residency_buffer != 0
                    && device_info.features.sparse_binding != 0
            })
            .collect();
        let (physical_device, device_info) = available_physical_devices
            .iter()
            .find(|(_physical_device, device_info)| {
                device_info.physical_device_properties.device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .or_else(|| {
                available_physical_devices
                    .iter()
                    .find(|(_physical_device, device_info)| {
                        device_info.physical_device_properties.device_type
                            == vk::PhysicalDeviceType::INTEGRATED_GPU
                    })
            })
            .expect("Unable to find a supported graphics card");
        let physical_device = *physical_device;
        let device_info = device_info.clone();
        println!(
            "Selected graphics card: {}",
            CStr::from_ptr(&device_info.physical_device_properties.device_name as *const _)
                .to_string_lossy()
        );
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        let available_queue_family =
            instance.get_physical_device_queue_family_properties(physical_device);
        let graphics_queue_family = available_queue_family
            .iter()
            .enumerate()
            .find(|&(i, family)| {
                family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && surface_loader
                        .get_physical_device_surface_support(physical_device, i as u32, surface)
                        .unwrap_or(false)
            })
            .unwrap()
            .0 as u32;
        let transfer_binding_queue_family = available_queue_family
            .iter()
            .enumerate()
            .find(|&(_, family)| {
                !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && family
                        .queue_flags
                        .contains(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING)
            })
            .or_else(|| {
                available_queue_family
                    .iter()
                    .enumerate()
                    .find(|&(_, family)| {
                        !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && family
                                .queue_flags
                                .contains(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING)
                    })
            })
            .or_else(|| {
                available_queue_family
                    .iter()
                    .enumerate()
                    .find(|&(_, family)| {
                        family
                            .queue_flags
                            .contains(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING)
                    })
            })
            .unwrap()
            .0 as u32;

        let extension_names = device_info.required_device_extensions();
        let mut ext = extension_names
            .into_iter()
            .map(|str| str.as_ptr())
            .collect::<Vec<_>>();
        ext.push(b"VK_KHR_shader_non_semantic_info" as *const u8 as *const i8);

        let device = instance
            .create_device(
                physical_device,
                &vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&[
                        vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(graphics_queue_family)
                            .queue_priorities(&[1.0])
                            .build(),
                        vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(transfer_binding_queue_family)
                            .queue_priorities(&[0.5])
                            .build(),
                    ])
                    .enabled_extension_names(&ext)
                    .enabled_features(&vk::PhysicalDeviceFeatures {
                        sparse_binding: 1,
                        sparse_residency_buffer: 1,
                        ..Default::default()
                    })
                    .push_next(
                        &mut vk::PhysicalDeviceShaderFloat16Int8Features::builder()
                            .shader_int8(false)
                            .build(),
                    )
                    .push_next(
                        &mut vk::PhysicalDevice16BitStorageFeatures::builder()
                            .storage_buffer16_bit_access(true)
                            .build(),
                    )
                    .push_next(
                        &mut vk::PhysicalDevice8BitStorageFeatures::builder()
                            .uniform_and_storage_buffer8_bit_access(true)
                            .build(),
                    ),
                None,
            )
            .unwrap();
        let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
        let transfer_binding_queue = device.get_device_queue(transfer_binding_queue_family, 0);
        let context = RenderContext {
            entry,
            device,
            surface,
            instance,
            surface_loader,
            debug_messenger,
            graphics_queue,
            transfer_binding_queue,
            graphics_queue_family,
            transfer_binding_queue_family,
        };
        let renderer = Self {
            context: Arc::new(context),
            physical_device,
            graphics_queue,
            transfer_binding_queue,
            graphics_queue_family,
            transfer_binding_queue_family,
            info: device_info,
        };

        renderer
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            if let Some((debug_utils, messenger)) = self.debug_messenger.take() {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
