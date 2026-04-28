// Thin C wrapper around the `static inline` helper macros declared in
// `nvsdk_ngx_helpers_dlssd_vk.h`. We can't call those directly from Rust
// since they aren't compiled into `nvsdk_ngx_d.lib`. Re-export them under
// stable symbols so the FFI in `dust_dlss::sys` can bind them.

#include <vulkan/vulkan.h>

#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

NVSDK_NGX_Result dust_ngx_vulkan_create_dlssd_ext1(
    VkDevice in_device,
    VkCommandBuffer in_cmd_list,
    unsigned int in_creation_node_mask,
    unsigned int in_visibility_node_mask,
    NVSDK_NGX_Handle **pp_out_handle,
    NVSDK_NGX_Parameter *p_in_params,
    NVSDK_NGX_DLSSD_Create_Params *p_in_dlssd_create_params)
{
    return NGX_VULKAN_CREATE_DLSSD_EXT1(
        in_device,
        in_cmd_list,
        in_creation_node_mask,
        in_visibility_node_mask,
        pp_out_handle,
        p_in_params,
        p_in_dlssd_create_params);
}
