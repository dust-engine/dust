// VK_LAYER_DUST_metal_objects: VK_EXT_metal_objects for the Vulkan SDK's KosmicKrisp,
// which doesn't have it.
//
// The driver offers no way to ask for the Metal objects behind its Vulkan objects. But
// its Vulkan handles are pointers to its own structs, and those hold the Metal objects.
// So this layer reads the Metal objects out of the structs. It imports a texture by
// storing it in the struct of the image it's for, which is the only thing it writes.
//
// Where in a struct the Metal object is depends on the build of the driver, and reading
// the wrong place would crash or corrupt memory. `kDrivers` lists the builds it's known
// for, by `driverUUID`, which KosmicKrisp takes from the UUID that the linker gives each
// build. On any other driver, the layer doesn't offer the extension.
//
// Importing buffers, IOSurfaces and shared events isn't implemented: their structs are
// ignored. Neither is exporting the shared event of a VkEvent, because KosmicKrisp doesn't
// implement those with one. An imported texture isn't made resident: the application has
// to add it to a residency set of the queue's.

#define VK_USE_PLATFORM_METAL_EXT
#include <CoreFoundation/CoreFoundation.h>
#include <objc/message.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

constexpr char kLayerName[] = "VK_LAYER_DUST_metal_objects";

// Where a build of KosmicKrisp keeps its Metal objects: byte offsets into its structs.
struct Driver {
    uint8_t uuid[VK_UUID_SIZE];
    size_t device;               // kk_device::mtl_handle
    size_t queue;                // kk_queue::mtl_handle
    size_t command_buffer;       // kk_cmd_buffer::metal.cmd_buf
    size_t render_encoder;       // kk_cmd_buffer::metal.render
    size_t compute_encoder;      // kk_cmd_buffer::metal.compute
    size_t memory_bo;            // kk_device_memory::bo
    size_t bo_buffer;            // kk_bo::map
    size_t image_texture;        // kk_image::planes[0].mtl_handle
    size_t image_plane_size;     // sizeof(kk_image::planes[0])
    size_t view_sampled;         // kk_image_view::planes[0].mtl_handle_sampled
    size_t view_storage;         // kk_image_view::planes[0].mtl_handle_storage
    size_t view_input;           // kk_image_view::planes[0].mtl_handle_input
    size_t view_plane_size;      // sizeof(kk_image_view::planes[0])
    size_t buffer_view_texture;  // kk_buffer_view::mtl_texel_buffer_handle
    size_t semaphore_type;       // vk_semaphore::type
    size_t binary_event;         // vk_semaphore::permanent, as a vk_sync_binary's kk_sync_timeline
    size_t timeline_event;       // vk_semaphore::permanent, as a kk_sync_timeline
};

// After updating the Vulkan SDK, add the output of
// `bazel run //layers/metal_objects:find_offsets`.
constexpr Driver kDrivers[] = {
    // vulkan-sdk-1.4.363 (git-269a955c89)
    {
        .uuid = {0xcc, 0xf7, 0xde, 0x1b, 0x1d, 0x36, 0x37, 0xc5,
                 0xa6, 0xa2, 0x57, 0x25, 0x6a, 0x11, 0x74, 0x7d},
        .device = 0x1b50,
        .queue = 0x218,
        .command_buffer = 0x17a0,
        .render_encoder = 0x17a8,
        .compute_encoder = 0x17b0,
        .memory_bo = 0x70,
        .bo_buffer = 0x8,
        .image_texture = 0x1b8,
        .image_plane_size = 0x130,
        .view_sampled = 0xb0,
        .view_storage = 0xb8,
        .view_input = 0xd0,
        .view_plane_size = 0x50,
        .buffer_view_texture = 0x68,
        .semaphore_type = 0x40,
        .binary_event = 0x78,
        .timeline_event = 0x60,
    },
};

struct Instance {
    VkInstance handle;
    PFN_vkGetInstanceProcAddr get_proc_addr;

    template <typename Function>
    Function next(const char *name) const {
        return reinterpret_cast<Function>(get_proc_addr(handle, name));
    }
};

struct Device {
    PFN_vkGetDeviceProcAddr get_proc_addr;
    PFN_vkDestroyDevice destroy_device;
    // Null if the layer leaves the device alone: the application didn't enable the
    // extension, or the driver implements it.
    const Driver *driver;
    PFN_vkCreateImage create_image;
    PFN_vkBindImageMemory2 bind_image_memory2;
    PFN_vkCmdBeginRendering cmd_begin_rendering;
    PFN_vkCmdEndRendering cmd_end_rendering;
};

// The layer's state for each instance and device, by the dispatch table that the loader
// puts at the start of every handle of theirs: an instance's physical devices, and a
// device's queues and command buffers, share its table.
std::shared_mutex mutex;
std::unordered_map<void *, Instance> instances;
std::unordered_map<void *, Device> devices;

template <typename Handle>
void *dispatch_table(Handle handle) {
    return *reinterpret_cast<void **>(handle);
}

template <typename Handle>
Instance instance_of(Handle handle) {
    std::shared_lock lock(mutex);
    return instances.at(dispatch_table(handle));
}

// References into an unordered_map stay valid while others are inserted and erased.
template <typename Handle>
const Device &device_of(Handle handle) {
    std::shared_lock lock(mutex);
    return devices.at(dispatch_table(handle));
}

template <typename T = void *, typename Handle>
T read(Handle handle, size_t offset) {
    T value;
    memcpy(&value, reinterpret_cast<const char *>(handle) + offset, sizeof(value));
    return value;
}

uint32_t plane_index(VkImageAspectFlags plane) {
    return plane == VK_IMAGE_ASPECT_PLANE_2_BIT ? 2 : plane == VK_IMAGE_ASPECT_PLANE_1_BIT ? 1 : 0;
}

size_t image_texture(const Driver &driver, VkImageAspectFlags plane) {
    return driver.image_texture + plane_index(plane) * driver.image_plane_size;
}

std::vector<VkExtensionProperties> driver_extensions(const Instance &instance,
                                                     VkPhysicalDevice physical_device) {
    auto enumerate = instance.next<PFN_vkEnumerateDeviceExtensionProperties>(
        "vkEnumerateDeviceExtensionProperties");
    uint32_t count = 0;
    enumerate(physical_device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    enumerate(physical_device, nullptr, &count, extensions.data());
    return extensions;
}

// The driver to implement the extension for, or null if its build isn't a known one or
// it has the extension itself.
const Driver *find_driver(const Instance &instance, VkPhysicalDevice physical_device) {
    for (const auto &extension : driver_extensions(instance, physical_device)) {
        if (extension.extensionName == std::string_view(VK_EXT_METAL_OBJECTS_EXTENSION_NAME)) {
            return nullptr;
        }
    }
    auto get_properties =
        instance.next<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2");
    if (!get_properties) {
        return nullptr;  // a Vulkan 1.0 instance
    }
    VkPhysicalDeviceIDProperties id{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &id};
    get_properties(physical_device, &properties);
    for (const auto &driver : kDrivers) {
        if (memcmp(driver.uuid, id.driverUUID, VK_UUID_SIZE) == 0) {
            return &driver;
        }
    }
    return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice handle, const VkImageCreateInfo *info,
                                           const VkAllocationCallbacks *allocator, VkImage *image) {
    const Device &device = device_of(handle);
    VkResult result = device.create_image(handle, info, allocator, image);
    if (result != VK_SUCCESS) {
        return result;
    }
    for (auto *s = static_cast<const VkBaseInStructure *>(info->pNext); s; s = s->pNext) {
        if (s->sType == VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT) {
            auto *import = reinterpret_cast<const VkImportMetalTextureInfoEXT *>(s);
            // The driver releases the image's textures when it destroys the image.
            const void *texture = CFRetain(import->mtlTexture);
            memcpy(reinterpret_cast<char *>(*image) + image_texture(*device.driver, import->plane),
                   &texture, sizeof(texture));
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2(VkDevice handle, uint32_t count,
                                                const VkBindImageMemoryInfo *infos) {
    const Device &device = device_of(handle);
    VkResult result = VK_SUCCESS;
    for (uint32_t i = 0; i < count; i++) {
        VkImageAspectFlags plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
        for (auto *s = static_cast<const VkBaseInStructure *>(infos[i].pNext); s; s = s->pNext) {
            if (s->sType == VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO) {
                plane = reinterpret_cast<const VkBindImagePlaneMemoryInfo *>(s)->planeAspect;
            }
        }
        // An image only has a texture before it's bound if it was imported. The driver
        // would replace that with one in the memory, so leave it alone.
        if (read(infos[i].image, image_texture(*device.driver, plane))) {
            continue;
        }
        VkResult bound = device.bind_image_memory2(handle, 1, &infos[i]);
        if (result == VK_SUCCESS) {
            result = bound;
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice device, VkImage image,
                                               VkDeviceMemory memory, VkDeviceSize offset) {
    VkBindImageMemoryInfo info{.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
                               .image = image,
                               .memory = memory,
                               .memoryOffset = offset};
    return BindImageMemory2(device, 1, &info);
}

// The structure type with which pumicite asks for a command buffer's `MTL4CommandBuffer`,
// which the extension has no structure for. The structure is a
// VkExportMetalCommandQueueInfoEXT: `queue` is the VkCommandBuffer, and `mtlCommandQueue`
// receives the Metal command buffer.
//
// The driver doesn't know about the Metal commands that the caller encodes, and ignores a
// Vulkan barrier that comes right after them. So the caller has to order them with Metal
// barriers.
constexpr uint32_t kExportMetalCommandBufferInfo = 1000311012;

void *metal_command_buffer(const Device &device, VkCommandBuffer command_buffer) {
    const Driver &driver = *device.driver;
    // The caller encodes into the Metal command buffer itself, for which Metal requires
    // the driver's encoder to be closed. The driver closes it at the end of rendering,
    // even if nothing was rendered.
    if (read(command_buffer, driver.compute_encoder) && device.cmd_begin_rendering) {
        VkRenderingInfo rendering{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                  .renderArea = {{}, {1, 1}},
                                  .layerCount = 1};
        device.cmd_begin_rendering(command_buffer, &rendering);
        device.cmd_end_rendering(command_buffer);
    }
    if (read(command_buffer, driver.compute_encoder) ||
        read(command_buffer, driver.render_encoder)) {
        return nullptr;
    }
    return read(command_buffer, driver.command_buffer);
}

void *metal_texture(const Driver &driver, const VkExportMetalTextureInfoEXT &info) {
    if (info.image) {
        return read(info.image, image_texture(driver, info.plane));
    }
    if (info.imageView) {
        // The view has a texture for each way it can be used, and null for the others.
        size_t plane = plane_index(info.plane) * driver.view_plane_size;
        for (size_t texture : {driver.view_sampled, driver.view_storage, driver.view_input}) {
            if (void *view = read(info.imageView, texture + plane)) {
                return view;
            }
        }
        return nullptr;
    }
    return info.bufferView ? read(info.bufferView, driver.buffer_view_texture) : nullptr;
}

VKAPI_ATTR void VKAPI_CALL ExportMetalObjectsEXT(VkDevice handle,
                                                 VkExportMetalObjectsInfoEXT *info) {
    const Device &device = device_of(handle);
    const Driver &driver = *device.driver;
    auto *next = static_cast<const VkBaseOutStructure *>(info->pNext);
    for (auto *s = const_cast<VkBaseOutStructure *>(next); s; s = s->pNext) {
        switch (static_cast<uint32_t>(s->sType)) {
            case VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT:
                reinterpret_cast<VkExportMetalDeviceInfoEXT *>(s)->mtlDevice =
                    read<MTLDevice_id>(handle, driver.device);
                break;
            case VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT: {
                auto *queue = reinterpret_cast<VkExportMetalCommandQueueInfoEXT *>(s);
                queue->mtlCommandQueue = read<MTLCommandQueue_id>(queue->queue, driver.queue);
                break;
            }
            case VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT: {
                auto *buffer = reinterpret_cast<VkExportMetalBufferInfoEXT *>(s);
                buffer->mtlBuffer = read<MTLBuffer_id>(read(buffer->memory, driver.memory_bo),
                                                       driver.bo_buffer);
                break;
            }
            case VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT: {
                auto *texture = reinterpret_cast<VkExportMetalTextureInfoEXT *>(s);
                texture->mtlTexture = static_cast<MTLTexture_id>(metal_texture(driver, *texture));
                break;
            }
            case VK_STRUCTURE_TYPE_EXPORT_METAL_IO_SURFACE_INFO_EXT: {
                auto *surface = reinterpret_cast<VkExportMetalIOSurfaceInfoEXT *>(s);
                auto send = reinterpret_cast<IOSurfaceRef (*)(void *, SEL)>(objc_msgSend);
                void *texture = read(surface->image, driver.image_texture);
                surface->ioSurface = send(texture, sel_registerName("iosurface"));
                break;
            }
            case VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT: {
                auto *event = reinterpret_cast<VkExportMetalSharedEventInfoEXT *>(s);
                event->mtlSharedEvent = nullptr;
                if (event->semaphore) {
                    auto type = read<VkSemaphoreType>(event->semaphore, driver.semaphore_type);
                    event->mtlSharedEvent = read<MTLSharedEvent_id>(
                        event->semaphore, type == VK_SEMAPHORE_TYPE_TIMELINE
                                              ? driver.timeline_event
                                              : driver.binary_event);
                }
                break;
            }
            case kExportMetalCommandBufferInfo: {
                auto *command_buffer = reinterpret_cast<VkExportMetalCommandQueueInfoEXT *>(s);
                command_buffer->mtlCommandQueue = static_cast<MTLCommandQueue_id>(
                    metal_command_buffer(device, (VkCommandBuffer)command_buffer->queue));
                break;
            }
        }
    }
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice handle, const VkAllocationCallbacks *allocator) {
    if (!handle) {
        return;
    }
    void *key = dispatch_table(handle);
    device_of(handle).destroy_device(handle, allocator);
    std::unique_lock lock(mutex);
    devices.erase(key);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice handle, const char *name) {
    const Device &device = device_of(handle);
    struct Intercept {
        std::string_view name;
        PFN_vkVoidFunction function;
    };
    static const Intercept always[] = {
        {"vkGetDeviceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(GetDeviceProcAddr)},
        {"vkDestroyDevice", reinterpret_cast<PFN_vkVoidFunction>(DestroyDevice)},
    };
    static const Intercept implementing[] = {
        {"vkCreateImage", reinterpret_cast<PFN_vkVoidFunction>(CreateImage)},
        {"vkBindImageMemory", reinterpret_cast<PFN_vkVoidFunction>(BindImageMemory)},
        {"vkBindImageMemory2", reinterpret_cast<PFN_vkVoidFunction>(BindImageMemory2)},
        {"vkBindImageMemory2KHR", reinterpret_cast<PFN_vkVoidFunction>(BindImageMemory2)},
        {"vkExportMetalObjectsEXT", reinterpret_cast<PFN_vkVoidFunction>(ExportMetalObjectsEXT)},
    };
    for (const auto &intercept : always) {
        if (intercept.name == name) {
            return intercept.function;
        }
    }
    if (device.driver) {
        for (const auto &intercept : implementing) {
            if (intercept.name == name) {
                return intercept.function;
            }
        }
    }
    return device.get_proc_addr(handle, name);
}

// The entry in a creation's pNext chain that links this layer to what's below it, the next
// layer or the driver. A layer has to move it on to the next link before calling down.
template <typename LayerCreateInfo, typename CreateInfo>
LayerCreateInfo *find_link(const CreateInfo *info, VkStructureType type) {
    for (auto *s = static_cast<const VkBaseInStructure *>(info->pNext); s; s = s->pNext) {
        auto *link = reinterpret_cast<const LayerCreateInfo *>(s);
        if (s->sType == type && link->function == VK_LAYER_LINK_INFO) {
            return const_cast<LayerCreateInfo *>(link);
        }
    }
    return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physical_device,
                                            const VkDeviceCreateInfo *info,
                                            const VkAllocationCallbacks *allocator,
                                            VkDevice *handle) {
    auto *link = find_link<VkLayerDeviceCreateInfo>(
        info, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO);
    if (!link) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    Instance instance = instance_of(physical_device);
    Device device{.get_proc_addr = link->u.pLayerInfo->pfnNextGetDeviceProcAddr};
    auto create_device = reinterpret_cast<PFN_vkCreateDevice>(
        link->u.pLayerInfo->pfnNextGetInstanceProcAddr(instance.handle, "vkCreateDevice"));
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    auto *enabled = info->ppEnabledExtensionNames;
    if (std::any_of(enabled, enabled + info->enabledExtensionCount, [](const char *name) {
            return name == std::string_view(VK_EXT_METAL_OBJECTS_EXTENSION_NAME);
        })) {
        device.driver = find_driver(instance, physical_device);
    }

    VkResult result = create_device(physical_device, info, allocator, handle);
    if (result != VK_SUCCESS) {
        return result;
    }
    auto next = [&]<typename Function>(Function &function, const char *name) {
        function = reinterpret_cast<Function>(device.get_proc_addr(*handle, name));
    };
    next(device.destroy_device, "vkDestroyDevice");
    next(device.create_image, "vkCreateImage");
    next(device.bind_image_memory2, "vkBindImageMemory2");
    next(device.cmd_begin_rendering, "vkCmdBeginRendering");
    next(device.cmd_end_rendering, "vkCmdEndRendering");
    std::unique_lock lock(mutex);
    devices[dispatch_table(*handle)] = device;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(
    VkPhysicalDevice physical_device, const char *layer, uint32_t *count,
    VkExtensionProperties *properties) {
    Instance instance = instance_of(physical_device);
    constexpr VkExtensionProperties kExtension{VK_EXT_METAL_OBJECTS_EXTENSION_NAME,
                                               VK_EXT_METAL_OBJECTS_SPEC_VERSION};
    std::vector<VkExtensionProperties> extensions;
    if (!layer) {
        extensions = driver_extensions(instance, physical_device);
        if (find_driver(instance, physical_device)) {
            extensions.push_back(kExtension);
        }
    } else if (layer == std::string_view(kLayerName)) {
        extensions.push_back(kExtension);
    } else {
        return instance.next<PFN_vkEnumerateDeviceExtensionProperties>(
            "vkEnumerateDeviceExtensionProperties")(physical_device, layer, count, properties);
    }
    if (!properties) {
        *count = extensions.size();
        return VK_SUCCESS;
    }
    *count = std::min<uint32_t>(*count, extensions.size());
    std::copy_n(extensions.begin(), *count, properties);
    return *count < extensions.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *info,
                                              const VkAllocationCallbacks *allocator,
                                              VkInstance *handle) {
    auto *link = find_link<VkLayerInstanceCreateInfo>(
        info, VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO);
    if (!link) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto get_proc_addr = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto create_instance =
        reinterpret_cast<PFN_vkCreateInstance>(get_proc_addr(nullptr, "vkCreateInstance"));
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    VkResult result = create_instance(info, allocator, handle);
    if (result == VK_SUCCESS) {
        std::unique_lock lock(mutex);
        instances[dispatch_table(*handle)] = {*handle, get_proc_addr};
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance handle,
                                           const VkAllocationCallbacks *allocator) {
    if (!handle) {
        return;
    }
    void *key = dispatch_table(handle);
    instance_of(handle).next<PFN_vkDestroyInstance>("vkDestroyInstance")(handle, allocator);
    std::unique_lock lock(mutex);
    instances.erase(key);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance handle, const char *name) {
    struct Intercept {
        std::string_view name;
        PFN_vkVoidFunction function;
    };
    static const Intercept intercepts[] = {
        {"vkGetInstanceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(GetInstanceProcAddr)},
        {"vkCreateInstance", reinterpret_cast<PFN_vkVoidFunction>(CreateInstance)},
        {"vkDestroyInstance", reinterpret_cast<PFN_vkVoidFunction>(DestroyInstance)},
        {"vkEnumerateDeviceExtensionProperties",
         reinterpret_cast<PFN_vkVoidFunction>(EnumerateDeviceExtensionProperties)},
        {"vkCreateDevice", reinterpret_cast<PFN_vkVoidFunction>(CreateDevice)},
        {"vkGetDeviceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(GetDeviceProcAddr)},
    };
    for (const auto &intercept : intercepts) {
        if (intercept.name == name) {
            return intercept.function;
        }
    }
    return handle ? instance_of(handle).get_proc_addr(handle, name) : nullptr;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *interface) {
    if (interface->loaderLayerInterfaceVersion < 2) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    interface->loaderLayerInterfaceVersion = 2;
    interface->pfnGetInstanceProcAddr = GetInstanceProcAddr;
    interface->pfnGetDeviceProcAddr = GetDeviceProcAddr;
    interface->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
