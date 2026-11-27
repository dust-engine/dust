// Finds where the linked build of KosmicKrisp keeps its Metal objects, and prints it as an
// entry for `kDrivers` in metal_objects.cc.
//
// KosmicKrisp makes its Metal objects by sending messages to other Metal objects, during
// the Vulkan call that needs them. So this replaces the methods it uses with ones that also
// record what they return, makes one Vulkan object of each kind, and searches the driver's
// struct for what was recorded. A field that's found in no place or in several is reported,
// and nothing is printed for it.

#define VK_USE_PLATFORM_METAL_EXT
#import <Metal/Metal.h>
#include <malloc/malloc.h>
#include <objc/runtime.h>
#include <vulkan/vulkan.h>

#include <cstring>
#include <functional>
#include <vector>

extern "C" PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);

// What the replaced methods have recorded: the selector, and the object it returned, or
// the receiver if it returns nothing.
static std::vector<std::pair<SEL, id>> recorded;

template <typename Result, typename... Arguments>
static void record(id object, SEL selector) {
    Class object_class = object_getClass(object);
    Method method = class_getInstanceMethod(object_class, selector);
    auto original = reinterpret_cast<Result (*)(id, SEL, Arguments...)>(
        method_getImplementation(method));
    Result (^replacement)(id, Arguments...) = [=](id self, Arguments... arguments) -> Result {
        if constexpr (std::is_void_v<Result>) {
            original(self, selector, arguments...);
            recorded.push_back({selector, self});
        } else {
            Result result = original(self, selector, arguments...);
            recorded.push_back({selector, result});
            return result;
        }
    };
    class_replaceMethod(object_class, selector, imp_implementationWithBlock(replacement),
                        method_getTypeEncoding(method));
}

// The objects recorded for `selector` while `call` ran.
static std::vector<id> during(SEL selector, const std::function<void()> &call) {
    recorded.clear();
    call();
    std::vector<id> objects;
    for (auto [recorded_selector, object] : recorded) {
        if (recorded_selector == selector) {
            objects.push_back(object);
        }
    }
    return objects;
}

static bool failed = false;

// Where `object` holds `wanted`. `size` is only for the queue: it's part of the device's
// allocation, so malloc can't tell its size.
static size_t find(const char *field, const void *object, id wanted, size_t size = 0) {
    size = size ? size : malloc_size(object);
    std::vector<size_t> offsets;
    for (size_t offset = 0; offset + sizeof(id) <= size; offset += sizeof(id)) {
        if (memcmp(static_cast<const char *>(object) + offset, &wanted, sizeof(id)) == 0) {
            offsets.push_back(offset);
        }
    }
    if (!wanted || offsets.size() != 1) {
        fprintf(stderr, "%s: found in %zu places\n", field, wanted ? offsets.size() : 0);
        failed = true;
        return 0;
    }
    return offsets[0];
}

int main() {
    @autoreleasepool {
        id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
        MTLHeapDescriptor *heap_descriptor = [MTLHeapDescriptor new];
        heap_descriptor.type = MTLHeapTypePlacement;
        heap_descriptor.storageMode = MTLStorageModeShared;
        heap_descriptor.size = 1 << 20;
        id<MTLHeap> heap = [metal newHeapWithDescriptor:heap_descriptor];
        MTLTextureDescriptor *texture_descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:16
                                                              height:16
                                                           mipmapped:NO];
        texture_descriptor.storageMode = MTLStorageModeShared;
        id<MTLTexture> texture = [heap newTextureWithDescriptor:texture_descriptor offset:0];
        id<MTLBuffer> buffer = [heap newBufferWithLength:4096
                                                 options:heap.resourceOptions
                                                  offset:1 << 19];
        id<MTL4CommandBuffer> command_buffer = [metal newCommandBuffer];

        record<id>(metal, @selector(newMTL4CommandQueue));
        record<id>(metal, @selector(newSharedEvent));
        record<id, MTLTextureDescriptor *, NSUInteger>(heap,
                                                       @selector(newTextureWithDescriptor:offset:));
        record<id, NSUInteger, MTLResourceOptions, NSUInteger>(
            heap, @selector(newBufferWithLength:options:offset:));
        SEL new_view = @selector(newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:);
        record<id, MTLPixelFormat, MTLTextureType, NSRange, NSRange, MTLTextureSwizzleChannels>(
            texture, new_view);
        record<id, MTLTextureDescriptor *, NSUInteger, NSUInteger>(
            buffer, @selector(newTextureWithDescriptor:offset:bytesPerRow:));
        record<void, id>(command_buffer, @selector(beginCommandBufferWithAllocator:));
        record<id>(command_buffer, @selector(computeCommandEncoder));
        record<id, MTL4RenderPassDescriptor *>(command_buffer,
                                               @selector(renderCommandEncoderWithDescriptor:));

        auto get_instance = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vk_icdGetInstanceProcAddr);
        VkApplicationInfo application{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                      .apiVersion = VK_API_VERSION_1_3};
        VkInstanceCreateInfo instance_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                           .pApplicationInfo = &application};
        VkInstance instance = VK_NULL_HANDLE;
        reinterpret_cast<PFN_vkCreateInstance>(get_instance(nullptr, "vkCreateInstance"))(
            &instance_info, nullptr, &instance);
#define INSTANCE(name) auto name = reinterpret_cast<PFN_##name>(get_instance(instance, #name))
        INSTANCE(vkEnumeratePhysicalDevices);
        INSTANCE(vkGetPhysicalDeviceProperties2);
        INSTANCE(vkCreateDevice);
        INSTANCE(vkGetDeviceProcAddr);
        uint32_t count = 1;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        vkEnumeratePhysicalDevices(instance, &count, &physical_device);
        VkPhysicalDeviceDriverProperties driver{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceIDProperties id_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, .pNext = &driver};
        VkPhysicalDeviceProperties2 properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &properties);

        float priority = 1;
        VkDeviceQueueCreateInfo queue_info{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                           .queueCount = 1,
                                           .pQueuePriorities = &priority};
        VkPhysicalDeviceVulkan11Features vulkan11{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .samplerYcbcrConversion = VK_TRUE};
        VkPhysicalDeviceVulkan13Features vulkan13{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &vulkan11,
            .dynamicRendering = VK_TRUE};
        VkDeviceCreateInfo device_info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                       .pNext = &vulkan13,
                                       .queueCreateInfoCount = 1,
                                       .pQueueCreateInfos = &queue_info};
        VkDevice device = VK_NULL_HANDLE;
        auto queues = during(@selector(newMTL4CommandQueue), [&] {
            vkCreateDevice(physical_device, &device_info, nullptr, &device);
        });
#define DEVICE(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name))
        DEVICE(vkGetDeviceQueue);
        DEVICE(vkCreateImage);
        DEVICE(vkGetImageMemoryRequirements);
        DEVICE(vkAllocateMemory);
        DEVICE(vkBindImageMemory);
        DEVICE(vkCreateImageView);
        DEVICE(vkCreateSamplerYcbcrConversion);
        DEVICE(vkCreateBuffer);
        DEVICE(vkBindBufferMemory);
        DEVICE(vkCreateBufferView);
        DEVICE(vkCreateSemaphore);
        DEVICE(vkCreateCommandPool);
        DEVICE(vkAllocateCommandBuffers);
        DEVICE(vkBeginCommandBuffer);
        DEVICE(vkCmdFillBuffer);
        DEVICE(vkCmdBeginRendering);

        printf("    // %s\n    {\n        .uuid = {", driver.driverInfo);
        for (uint32_t i = 0; i < VK_UUID_SIZE; i++) {
            printf("%s0x%02x", i == 8 ? ",\n                 " : i ? ", " : "",
                   id_properties.driverUUID[i]);
        }
        printf("},\n");
        auto print = [](const char *field, size_t offset) {
            printf("        .%s = %#zx,\n", field, offset);
        };

        VkQueue queue;
        vkGetDeviceQueue(device, 0, 0, &queue);
        print("device", find("device", device, metal));
        print("queue", find("queue", queue, queues.empty() ? nil : queues[0], 1024));

        auto make_image = [&](VkFormat format, VkImageUsageFlags usage, VkImage &image) {
            VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                   .imageType = VK_IMAGE_TYPE_2D,
                                   .format = format,
                                   .extent = {64, 64, 1},
                                   .mipLevels = 1,
                                   .arrayLayers = 1,
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .usage = usage};
            vkCreateImage(device, &info, nullptr, &image);
            VkMemoryRequirements requirements;
            vkGetImageMemoryRequirements(device, image, &requirements);
            VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                            .allocationSize = requirements.size};
            VkDeviceMemory memory;
            vkAllocateMemory(device, &allocation, nullptr, &memory);
            return during(@selector(newTextureWithDescriptor:offset:),
                          [&] { vkBindImageMemory(device, image, memory, 0); });
        };
        auto make_view = [&](VkImage image, VkFormat format, VkImageUsageFlags usage,
                             const void *conversion, VkImageView &view) {
            VkImageViewUsageCreateInfo usage_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                .pNext = conversion,
                .usage = usage};
            VkImageViewCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = &usage_info,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = format,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
            auto views = during(new_view,
                                [&] { vkCreateImageView(device, &info, nullptr, &view); });
            views.resize(std::max<size_t>(views.size(), 1));
            return views;
        };

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        VkImageUsageFlags usages[] = {VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_STORAGE_BIT,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        VkImage image;
        auto textures = make_image(format, usages[0] | usages[1] | usages[2], image);
        textures.resize(std::max<size_t>(textures.size(), 1));
        VkImageView views[3];
        size_t view_offsets[3];
        for (uint32_t i = 0; i < 3; i++) {
            auto made = make_view(image, format, usages[i], nullptr, views[i]);
            const char *fields[] = {"view_sampled", "view_storage", "view_input"};
            view_offsets[i] = find(fields[i], (void *)views[i], made[0]);
        }
        size_t image_texture = find("image_texture", (void *)image, textures[0]);

        VkFormat planar_format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        VkImage planar;
        auto plane_textures = make_image(planar_format, VK_IMAGE_USAGE_SAMPLED_BIT, planar);
        plane_textures.resize(std::max<size_t>(plane_textures.size(), 2));
        VkSamplerYcbcrConversionCreateInfo conversion_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
            .format = planar_format,
            .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709};
        VkSamplerYcbcrConversionInfo conversion{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO};
        vkCreateSamplerYcbcrConversion(device, &conversion_info, nullptr, &conversion.conversion);
        VkImageView planar_view;
        auto plane_views = make_view(planar, planar_format, VK_IMAGE_USAGE_SAMPLED_BIT,
                                     &conversion, planar_view);
        plane_views.resize(std::max<size_t>(plane_views.size(), 2));

        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 4096,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT};
        VkBuffer vulkan_buffer;
        vkCreateBuffer(device, &buffer_info, nullptr, &vulkan_buffer);
        VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                        .allocationSize = 1 << 16};
        VkDeviceMemory memory;
        auto buffers = during(@selector(newBufferWithLength:options:offset:), [&] {
            vkAllocateMemory(device, &allocation, nullptr, &memory);
        });
        vkBindBufferMemory(device, vulkan_buffer, memory, 0);

        // The memory's struct points to another allocation of the driver's, which holds
        // the buffer.
        size_t memory_bo = 0, bo_buffer = 0;
        for (size_t offset = 0; offset + 8 <= malloc_size((void *)memory); offset += 8) {
            void *bo;
            memcpy(&bo, reinterpret_cast<char *>(memory) + offset, sizeof(bo));
            for (size_t i = 0; bo && malloc_zone_from_ptr(bo) && i + 8 <= malloc_size(bo); i += 8) {
                if (!buffers.empty() && memcmp(static_cast<char *>(bo) + i, &buffers[0], 8) == 0) {
                    failed |= memory_bo != 0;
                    memory_bo = offset;
                    bo_buffer = i;
                }
            }
        }
        failed |= !memory_bo;

        VkBufferViewCreateInfo buffer_view_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .buffer = vulkan_buffer,
            .format = VK_FORMAT_R32_UINT,
            .range = VK_WHOLE_SIZE};
        VkBufferView buffer_view;
        auto buffer_textures = during(@selector(newTextureWithDescriptor:offset:bytesPerRow:), [&] {
            vkCreateBufferView(device, &buffer_view_info, nullptr, &buffer_view);
        });
        buffer_textures.resize(std::max<size_t>(buffer_textures.size(), 1));

        VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        VkCommandPool pool;
        vkCreateCommandPool(device, &pool_info, nullptr, &pool);
        VkCommandBufferAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .commandBufferCount = 1};
        VkCommandBuffer commands;
        vkAllocateCommandBuffers(device, &allocate_info, &commands);
        VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        auto begun = during(@selector(beginCommandBufferWithAllocator:),
                            [&] { vkBeginCommandBuffer(commands, &begin_info); });
        begun.resize(std::max<size_t>(begun.size(), 1));
        print("command_buffer", find("command_buffer", commands, begun[0]));
        auto compute = during(@selector(computeCommandEncoder),
                              [&] { vkCmdFillBuffer(commands, vulkan_buffer, 0, 64, 0); });
        compute.resize(std::max<size_t>(compute.size(), 1));
        size_t compute_encoder = find("compute_encoder", commands, compute[0]);
        VkRenderingAttachmentInfo attachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = views[2],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkRenderingInfo rendering{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                  .renderArea = {{}, {64, 64}},
                                  .layerCount = 1,
                                  .colorAttachmentCount = 1,
                                  .pColorAttachments = &attachment};
        auto render = during(@selector(renderCommandEncoderWithDescriptor:),
                             [&] { vkCmdBeginRendering(commands, &rendering); });
        render.resize(std::max<size_t>(render.size(), 1));
        print("render_encoder", find("render_encoder", commands, render[0]));
        print("compute_encoder", compute_encoder);
        print("memory_bo", memory_bo);
        print("bo_buffer", bo_buffer);
        print("image_texture", image_texture);
        print("image_plane_size",
              find("plane 1's texture", (void *)planar, plane_textures[1]) -
                  find("plane 0's texture", (void *)planar, plane_textures[0]));
        print("view_sampled", view_offsets[0]);
        print("view_storage", view_offsets[1]);
        print("view_input", view_offsets[2]);
        print("view_plane_size",
              find("plane 1's view", (void *)planar_view, plane_views[1]) -
                  find("plane 0's view", (void *)planar_view, plane_views[0]));
        print("buffer_view_texture",
              find("buffer_view_texture", (void *)buffer_view, buffer_textures[0]));

        VkSemaphore semaphores[2];
        size_t events[2];
        for (uint32_t timeline = 0; timeline < 2; timeline++) {
            VkSemaphoreTypeCreateInfo type{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY};
            VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                       .pNext = &type};
            auto made = during(@selector(newSharedEvent), [&] {
                vkCreateSemaphore(device, &info, nullptr, &semaphores[timeline]);
            });
            made.resize(std::max<size_t>(made.size(), 1));
            events[timeline] = find(timeline ? "timeline_event" : "binary_event",
                                    (void *)semaphores[timeline], made[0]);
        }
        // The type follows the header that every Vulkan object starts with, and is the
        // first thing that's 0 in a binary semaphore and 1 in a timeline semaphore.
        size_t semaphore_type = 0;
        for (size_t offset = 0; offset + 4 <= events[1] && !semaphore_type; offset += 4) {
            uint32_t values[2];
            for (uint32_t timeline = 0; timeline < 2; timeline++) {
                memcpy(&values[timeline], reinterpret_cast<char *>(semaphores[timeline]) + offset, 4);
            }
            if (values[0] == VK_SEMAPHORE_TYPE_BINARY && values[1] == VK_SEMAPHORE_TYPE_TIMELINE) {
                semaphore_type = offset;
            }
        }
        failed |= !semaphore_type;
        print("semaphore_type", semaphore_type);
        print("binary_event", events[0]);
        print("timeline_event", events[1]);
        printf("    },\n");
    }
    if (failed) {
        fprintf(stderr, "Some fields weren't found: don't use this.\n");
    }
    return failed;
}
