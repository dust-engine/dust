// Runs the layer on the linked KosmicKrisp: imports a texture, exports each kind of object
// that the layer can, and uses them from both Vulkan and Metal.
//
// The layer is found through VK_ADD_LAYER_PATH.

#define VK_USE_PLATFORM_METAL_EXT
#import <Metal/Metal.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

extern "C" PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);

static int failures = 0;

#define CHECK(condition)                                                 \
    do {                                                                 \
        if (!(condition)) {                                              \
            fprintf(stderr, "line %d: %s is false\n", __LINE__, #condition); \
            failures++;                                                  \
        }                                                                \
    } while (0)

constexpr uint32_t kSize = 16;

// The structure that says that `type` will be exported from the object being created. It's
// chained in front of `next`.
static VkExportMetalObjectCreateInfoEXT exporting(VkExportMetalObjectTypeFlagBitsEXT type,
                                                  const void *next = nullptr) {
    return {.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT,
            .pNext = next,
            .exportObjectType = type};
}

int main() {
    @autoreleasepool {
        VkDirectDriverLoadingInfoLUNARG driver{
            .sType = VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_INFO_LUNARG,
            .pfnGetInstanceProcAddr =
                reinterpret_cast<PFN_vkGetInstanceProcAddrLUNARG>(vk_icdGetInstanceProcAddr)};
        VkDirectDriverLoadingListLUNARG drivers{
            .sType = VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_LIST_LUNARG,
            .mode = VK_DIRECT_DRIVER_LOADING_MODE_EXCLUSIVE_LUNARG,
            .driverCount = 1,
            .pDrivers = &driver};
        VkApplicationInfo application{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                      .apiVersion = VK_API_VERSION_1_3};
        const char *layer = "VK_LAYER_DUST_metal_objects";
        const char *instance_extension = VK_LUNARG_DIRECT_DRIVER_LOADING_EXTENSION_NAME;
        auto device_exported = exporting(VK_EXPORT_METAL_OBJECT_TYPE_METAL_DEVICE_BIT_EXT, &drivers);
        auto queue_exported = exporting(VK_EXPORT_METAL_OBJECT_TYPE_METAL_COMMAND_QUEUE_BIT_EXT,
                                        &device_exported);
        VkInstanceCreateInfo instance_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                           .pNext = &queue_exported,
                                           .pApplicationInfo = &application,
                                           .enabledLayerCount = 1,
                                           .ppEnabledLayerNames = &layer,
                                           .enabledExtensionCount = 1,
                                           .ppEnabledExtensionNames = &instance_extension};
        VkInstance instance;
        if (VkResult result = vkCreateInstance(&instance_info, nullptr, &instance)) {
            fprintf(stderr, "vkCreateInstance: %d\n", result);
            return 1;
        }
        uint32_t count = 1;
        VkPhysicalDevice physical_device;
        vkEnumeratePhysicalDevices(instance, &count, &physical_device);

        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data());
        uint32_t offered = 0;
        for (const auto &extension : extensions) {
            offered += !strcmp(extension.extensionName, VK_EXT_METAL_OBJECTS_EXTENSION_NAME);
        }
        CHECK(offered == 1);
        if (!offered) {
            return 1;
        }
        count = 1;
        VkExtensionProperties one;
        CHECK(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, &one) ==
              VK_INCOMPLETE);

        float priority = 1;
        VkDeviceQueueCreateInfo queue_info{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                           .queueCount = 1,
                                           .pQueuePriorities = &priority};
        VkPhysicalDeviceVulkan12Features vulkan12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .timelineSemaphore = VK_TRUE};
        VkPhysicalDeviceVulkan13Features vulkan13{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &vulkan12,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE};
        const char *device_extension = VK_EXT_METAL_OBJECTS_EXTENSION_NAME;
        VkDeviceCreateInfo device_info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                       .pNext = &vulkan13,
                                       .queueCreateInfoCount = 1,
                                       .pQueueCreateInfos = &queue_info,
                                       .enabledExtensionCount = 1,
                                       .ppEnabledExtensionNames = &device_extension};
        VkDevice device;
        if (VkResult result = vkCreateDevice(physical_device, &device_info, nullptr, &device)) {
            fprintf(stderr, "vkCreateDevice: %d\n", result);
            return 1;
        }
        VkQueue queue;
        vkGetDeviceQueue(device, 0, 0, &queue);
        auto export_metal = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
            vkGetDeviceProcAddr(device, "vkExportMetalObjectsEXT"));
        CHECK(export_metal);
        auto export_one = [&](void *structure) {
            VkExportMetalObjectsInfoEXT info{.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
                                             .pNext = structure};
            export_metal(device, &info);
        };

        // The device and the queue, exported together.
        VkExportMetalCommandQueueInfoEXT metal_queue_info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT, .queue = queue};
        VkExportMetalDeviceInfoEXT metal_device_info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT, .pNext = &metal_queue_info};
        export_one(&metal_device_info);
        id<MTLDevice> metal = metal_device_info.mtlDevice;
        auto metal_queue = (id<MTL4CommandQueue>)metal_queue_info.mtlCommandQueue;
        CHECK(metal == MTLCreateSystemDefaultDevice());
        CHECK([metal_queue conformsToProtocol:@protocol(MTL4CommandQueue)]);
        CHECK(metal_queue.device == metal);

        VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        VkCommandPool pool;
        vkCreateCommandPool(device, &pool_info, nullptr, &pool);
        VkSemaphoreTypeCreateInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        auto event_exported =
            exporting(VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT, &timeline_info);
        VkSemaphoreCreateInfo semaphore_info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                             .pNext = &event_exported};
        VkSemaphore timeline;
        vkCreateSemaphore(device, &semaphore_info, nullptr, &timeline);
        uint64_t submitted = 0;
        // Records a command buffer, runs it, and waits for it.
        auto run = [&](auto record) {
            VkCommandBufferAllocateInfo allocate_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = pool,
                .commandBufferCount = 1};
            VkCommandBuffer commands;
            vkAllocateCommandBuffers(device, &allocate_info, &commands);
            VkCommandBufferBeginInfo begin_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(commands, &begin_info);
            record(commands);
            vkEndCommandBuffer(commands);
            VkCommandBufferSubmitInfo command_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = commands};
            VkSemaphoreSubmitInfo signal{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                         .semaphore = timeline,
                                         .value = ++submitted,
                                         .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
            VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                                 .commandBufferInfoCount = 1,
                                 .pCommandBufferInfos = &command_info,
                                 .signalSemaphoreInfoCount = 1,
                                 .pSignalSemaphoreInfos = &signal};
            CHECK(vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
            VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                     .semaphoreCount = 1,
                                     .pSemaphores = &timeline,
                                     .pValues = &submitted};
            CHECK(vkWaitSemaphores(device, &wait, 5'000'000'000) == VK_SUCCESS);
            vkFreeCommandBuffers(device, pool, 1, &commands);
        };
        auto barrier = [](VkCommandBuffer commands) {
            VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                    .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                                                     VK_ACCESS_2_MEMORY_WRITE_BIT};
            VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                        .memoryBarrierCount = 1,
                                        .pMemoryBarriers = &memory};
            vkCmdPipelineBarrier2(commands, &dependency);
        };
        VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto to_general = [&](VkCommandBuffer commands, VkImage image) {
            VkImageMemoryBarrier2 image_barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = image,
                .subresourceRange = whole};
            VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                        .imageMemoryBarrierCount = 1,
                                        .pImageMemoryBarriers = &image_barrier};
            vkCmdPipelineBarrier2(commands, &dependency);
        };

        // A buffer that's read back on the CPU, in memory exported as a Metal buffer.
        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 1 << 14,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT};
        VkBuffer buffer;
        vkCreateBuffer(device, &buffer_info, nullptr, &buffer);
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        auto buffer_exported = exporting(VK_EXPORT_METAL_OBJECT_TYPE_METAL_BUFFER_BIT_EXT);
        VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                        .pNext = &buffer_exported,
                                        .allocationSize = requirements.size};
        VkDeviceMemory buffer_memory;
        vkAllocateMemory(device, &allocation, nullptr, &buffer_memory);
        vkBindBufferMemory(device, buffer, buffer_memory, 0);
        uint32_t *mapped;
        vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&mapped));
        VkExportMetalBufferInfoEXT metal_buffer_info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT, .memory = buffer_memory};
        export_one(&metal_buffer_info);
        id<MTLBuffer> metal_buffer = metal_buffer_info.mtlBuffer;
        CHECK([metal_buffer conformsToProtocol:@protocol(MTLBuffer)]);
        mapped[0] = 0x12345678;
        CHECK(*static_cast<uint32_t *>(metal_buffer.contents) == 0x12345678);
        auto texture_exported = exporting(VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT);
        VkBufferViewCreateInfo buffer_view_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .pNext = &texture_exported,
            .buffer = buffer,
            .format = VK_FORMAT_R32_UINT,
            .range = 256};
        VkBufferView buffer_view;
        vkCreateBufferView(device, &buffer_view_info, nullptr, &buffer_view);
        VkExportMetalTextureInfoEXT buffer_texture{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
            .bufferView = buffer_view,
            .plane = VK_IMAGE_ASPECT_PLANE_0_BIT};
        export_one(&buffer_texture);
        CHECK(buffer_texture.mtlTexture.textureType == MTLTextureTypeTextureBuffer);
        CHECK(buffer_texture.mtlTexture.width == 64);

        // Importing a texture: Vulkan clears it, and both Metal and Vulkan read it.
        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGB10A2Unorm
                                                               width:kSize
                                                              height:kSize
                                                           mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                           MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
        id<MTLTexture> texture = [metal newTextureWithDescriptor:descriptor];
        id<MTLResidencySet> residency =
            [metal newResidencySetWithDescriptor:[[MTLResidencySetDescriptor new] autorelease]
                                           error:nil];
        [residency addAllocation:texture];
        [residency commit];
        [metal_queue addResidencySet:residency];
        NSUInteger references = texture.retainCount;
        auto surface_exported =
            exporting(VK_EXPORT_METAL_OBJECT_TYPE_METAL_IOSURFACE_BIT_EXT, &texture_exported);
        VkImportMetalTextureInfoEXT import{
            .sType = VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT,
            .pNext = &surface_exported,
            .plane = VK_IMAGE_ASPECT_PLANE_0_BIT,
            .mtlTexture = texture};
        VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &import,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
            .extent = {kSize, kSize, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VkImage imported;
        CHECK(vkCreateImage(device, &image_info, nullptr, &imported) == VK_SUCCESS);
        VkExportMetalTextureInfoEXT exported{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
            .image = imported,
            .plane = VK_IMAGE_ASPECT_PLANE_0_BIT};
        export_one(&exported);
        CHECK(exported.mtlTexture == texture);
        // Binding an imported image to memory leaves its texture alone.
        CHECK(vkBindImageMemory(device, imported, buffer_memory, 0) == VK_SUCCESS);
        export_one(&exported);
        CHECK(exported.mtlTexture == texture);
        VkImageViewCreateInfo view_info{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                        .image = imported,
                                        .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                        .format = image_info.format,
                                        .subresourceRange = whole};
        VkImageView imported_view;
        CHECK(vkCreateImageView(device, &view_info, nullptr, &imported_view) == VK_SUCCESS);
        run([&](VkCommandBuffer commands) {
            to_general(commands, imported);
            VkRenderingAttachmentInfo attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = imported_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {.color = {.float32 = {1, 0, 1, 1}}}};
            VkRenderingInfo rendering{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                      .renderArea = {{}, {kSize, kSize}},
                                      .layerCount = 1,
                                      .colorAttachmentCount = 1,
                                      .pColorAttachments = &attachment};
            vkCmdBeginRendering(commands, &rendering);
            vkCmdEndRendering(commands);
            barrier(commands);
            VkBufferImageCopy copy{.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                   .imageExtent = image_info.extent};
            vkCmdCopyImageToBuffer(commands, imported, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
        });
        constexpr uint32_t kMagenta = 0xfff003ff;  // red in the low 10 bits, blue, alpha
        uint32_t pixels[kSize * kSize] = {};
        [texture getBytes:pixels
              bytesPerRow:kSize * 4
               fromRegion:MTLRegionMake2D(0, 0, kSize, kSize)
              mipmapLevel:0];
        CHECK(pixels[0] == kMagenta && pixels[kSize * kSize - 1] == kMagenta);
        CHECK(mapped[0] == kMagenta && mapped[kSize * kSize - 1] == kMagenta);
        // The driver lets go of the texture.
        vkDestroyImageView(device, imported_view, nullptr);
        vkDestroyImage(device, imported, nullptr);
        CHECK(texture.retainCount == references);

        // An image of the driver's own, and views of it.
        image_info.pNext = &surface_exported;
        image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        image_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        VkImage image;
        vkCreateImage(device, &image_info, nullptr, &image);
        vkGetImageMemoryRequirements(device, image, &requirements);
        allocation.allocationSize = requirements.size;
        VkDeviceMemory image_memory;
        vkAllocateMemory(device, &allocation, nullptr, &image_memory);
        exported = {.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
                    .image = image,
                    .plane = VK_IMAGE_ASPECT_PLANE_0_BIT};
        export_one(&exported);
        CHECK(exported.mtlTexture == nil);  // it has no storage yet
        CHECK(vkBindImageMemory(device, image, image_memory, 0) == VK_SUCCESS);
        export_one(&exported);
        id<MTLTexture> image_texture = exported.mtlTexture;
        CHECK(image_texture.pixelFormat == MTLPixelFormatRGBA16Float);
        CHECK(image_texture.width == kSize);
        VkExportMetalIOSurfaceInfoEXT surface{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_IO_SURFACE_INFO_EXT, .image = image};
        export_one(&surface);
        CHECK(surface.ioSurface == nullptr);
        view_info.image = image;
        view_info.format = image_info.format;
        view_info.components.r = VK_COMPONENT_SWIZZLE_A;
        for (VkImageUsageFlags usage :
             {VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_USAGE_STORAGE_BIT,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}) {
            VkImageViewUsageCreateInfo usage_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                .pNext = &texture_exported,
                .usage = usage};
            view_info.pNext = &usage_info;
            view_info.components.r = usage == VK_IMAGE_USAGE_SAMPLED_BIT
                                         ? VK_COMPONENT_SWIZZLE_A
                                         : VK_COMPONENT_SWIZZLE_IDENTITY;
            VkImageView view;
            CHECK(vkCreateImageView(device, &view_info, nullptr, &view) == VK_SUCCESS);
            VkExportMetalTextureInfoEXT view_texture{
                .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
                .imageView = view,
                .plane = VK_IMAGE_ASPECT_PLANE_0_BIT};
            export_one(&view_texture);
            CHECK(view_texture.mtlTexture.pixelFormat == MTLPixelFormatRGBA16Float);
            CHECK(view_texture.mtlTexture.parentTexture == image_texture);
            if (usage == VK_IMAGE_USAGE_SAMPLED_BIT) {
                CHECK(view_texture.mtlTexture.swizzle.red == MTLTextureSwizzleAlpha);
            }
            vkDestroyImageView(device, view, nullptr);
        }

        // Encoding Metal work between the Vulkan commands of a command buffer. Each step
        // only gives the right result after the one before it.
        memset(mapped, 0, 1 << 14);
        run([&](VkCommandBuffer commands) {
            vkCmdFillBuffer(commands, buffer, 0, 256, 1);
            barrier(commands);
            VkExportMetalCommandQueueInfoEXT command_buffer_info{
                .sType = static_cast<VkStructureType>(1000311012),
                .queue = reinterpret_cast<VkQueue>(commands)};
            export_one(&command_buffer_info);
            auto metal_commands = (id<MTL4CommandBuffer>)command_buffer_info.mtlCommandQueue;
            CHECK([metal_commands conformsToProtocol:@protocol(MTL4CommandBuffer)]);
            id<MTL4ComputeCommandEncoder> encoder = [metal_commands computeCommandEncoder];
            [encoder barrierAfterQueueStages:MTLStageAll
                                beforeStages:MTLStageAll
                           visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder copyFromBuffer:metal_buffer
                       sourceOffset:0
                           toBuffer:metal_buffer
                  destinationOffset:256
                               size:256];
            [encoder barrierAfterStages:MTLStageAll
                      beforeQueueStages:MTLStageAll
                      visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder endEncoding];
            VkBufferCopy copy{.srcOffset = 256, .dstOffset = 512, .size = 256};
            vkCmdCopyBuffer(commands, buffer, buffer, 1, &copy);
            barrier(commands);
            vkCmdFillBuffer(commands, buffer, 0, 256, 2);
        });
        CHECK(mapped[0] == 2 && mapped[63] == 2);
        CHECK(mapped[64] == 1 && mapped[127] == 1);
        CHECK(mapped[128] == 1 && mapped[191] == 1);

        // Semaphores.
        VkExportMetalSharedEventInfoEXT event{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT, .semaphore = timeline};
        export_one(&event);
        CHECK([event.mtlSharedEvent conformsToProtocol:@protocol(MTLSharedEvent)]);
        CHECK(event.mtlSharedEvent.signaledValue == submitted);
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_BINARY;
        VkSemaphore binary;
        vkCreateSemaphore(device, &semaphore_info, nullptr, &binary);
        event.semaphore = binary;
        export_one(&event);
        CHECK([event.mtlSharedEvent conformsToProtocol:@protocol(MTLSharedEvent)]);

        vkDestroySemaphore(device, binary, nullptr);
        vkDestroySemaphore(device, timeline, nullptr);
        vkDestroyImage(device, image, nullptr);
        vkFreeMemory(device, image_memory, nullptr);
        vkDestroyBufferView(device, buffer_view, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, buffer_memory, nullptr);
        vkDestroyCommandPool(device, pool, nullptr);
        [metal_queue removeResidencySet:residency];
        vkDestroyDevice(device, nullptr);

        // Without the extension, the layer leaves the device alone.
        device_info.enabledExtensionCount = 0;
        vkCreateDevice(physical_device, &device_info, nullptr, &device);
        CHECK(!vkGetDeviceProcAddr(device, "vkExportMetalObjectsEXT"));
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
    printf("%d failures\n", failures);
    return failures != 0;
}
