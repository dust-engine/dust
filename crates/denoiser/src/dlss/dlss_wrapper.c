// Thin C wrapper around the `static inline` helper macros declared in
// `nvsdk_ngx_helpers_dlssd_vk.h`. We can't call those directly from Rust
// since they aren't compiled into `nvsdk_ngx_d.lib`. Re-export them under
// stable symbols so the FFI in `dust_denoiser::dlss::sys` can bind them.

#include <vulkan/vulkan.h>

#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssd.h"
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

NVSDK_NGX_Result dust_ngx_vulkan_evaluate_dlssd_ext(
    VkCommandBuffer in_cmd_list,
    NVSDK_NGX_Handle *p_in_handle,
    NVSDK_NGX_Parameter *p_in_params,
    NVSDK_NGX_VK_DLSSD_Eval_Params *p_in_dlssd_eval_params)
{
    return NGX_VULKAN_EVALUATE_DLSSD_EXT(
        in_cmd_list,
        p_in_handle,
        p_in_params,
        p_in_dlssd_eval_params);
}

// Thin wrapper around the `static inline` NGX_DLSSD_GET_OPTIMAL_SETTINGS helper
// from `nvsdk_ngx_helpers_dlssd.h`. The helper pulls the DLSS-RR optimal-settings
// callback (the `DLSSDOptimalSettingsCallback` key — distinct from the DLSS-SR
// `DLSSOptimalSettingsCallback` one) out of `p_in_params` (only populated when
// the map comes from NVSDK_NGX_VULKAN_GetCapabilityParameters *and* the DLSS-RR
// snippet is available), writes the user-selected target resolution +
// perf-quality value into the map, invokes the callback, and reads the
// recommended / min / max render resolution and sharpness back out.
NVSDK_NGX_Result dust_ngx_dlssd_get_optimal_settings(
    NVSDK_NGX_Parameter *p_in_params,
    unsigned int in_user_selected_width,
    unsigned int in_user_selected_height,
    NVSDK_NGX_PerfQuality_Value in_perf_quality_value,
    unsigned int *p_out_render_optimal_width,
    unsigned int *p_out_render_optimal_height,
    unsigned int *p_out_render_max_width,
    unsigned int *p_out_render_max_height,
    unsigned int *p_out_render_min_width,
    unsigned int *p_out_render_min_height,
    float *p_out_sharpness)
{
    return NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        p_in_params,
        in_user_selected_width,
        in_user_selected_height,
        in_perf_quality_value,
        p_out_render_optimal_width,
        p_out_render_optimal_height,
        p_out_render_max_width,
        p_out_render_max_height,
        p_out_render_min_width,
        p_out_render_min_height,
        p_out_sharpness);
}
