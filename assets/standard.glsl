
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_samplerless_texture_functions: require
#extension GL_EXT_control_flow_attributes: require

#extension GL_EXT_debug_printf : enable
// Illuminance: total luminous flux incident on a surface, per unit area.
// Unit: lux (lm / m^2)
layout(set = 0, binding = 0, rgba32f) uniform image2D u_illuminance;
layout(set = 0, binding = 1, rgb10_a2) uniform image2D u_albedo;
layout(set = 0, binding = 2, rgba16_snorm) uniform image2D u_normal;
layout(set = 0, binding = 3, r32f) uniform image2D u_depth;
layout(set = 0, binding = 4, rg16f) uniform image2D u_motion;

layout(set = 0, binding = 5) uniform accelerationStructureEXT accelerationStructure;
layout(set = 0, binding = 6) uniform texture2D blue_noise;



layout(set = 0, binding = 8, std430) uniform CameraSettingsLastFrame {
    mat4 view_proj;
} u_camera_last_frame;

#define RETENTION_FACTOR 0.95

// TODO: make this adaptable
//#define SHADER_INT_64 

layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;

struct Block
{
    u16vec4 position;
    #ifdef SHADER_INT_64
    uint64_t mask;
    #else
    uint32_t mask1;
    uint32_t mask2;
    #endif
    uint32_t material_ptr;
    uint32_t reserved;
};

layout(buffer_reference, buffer_reference_align = 8, scalar) buffer GeometryInfo {
    Block blocks[];
};
layout(buffer_reference, buffer_reference_align = 1, scalar) buffer MaterialInfo {
    uint8_t materials[];
};
layout(buffer_reference) buffer PaletteInfo {
    u8vec4 palette[];
};

struct IrradianceCacheFace {
    f16vec3 irradiance;
    uint16_t mask;
};
struct IrradianceCacheEntry {
    IrradianceCacheFace faces[6];
    uint16_t lastAccessedFrameIndex[6];
    uint32_t _reerved;
};
layout(buffer_reference, scalar) buffer IrradianceCache {
    IrradianceCacheEntry entries[];
};


vec3 CubedNormalize(vec3 dir) {
    vec3 dir_abs = abs(dir);
    float max_element = max(dir_abs.x, max(dir_abs.y, dir_abs.z));
    return sign(dir) * step(max_element, dir_abs);
}

vec2 intersectAABB(vec3 origin, vec3 dir, vec3 box_min, vec3 box_max) {
    vec3 tMin = (box_min - origin) / dir;
    vec3 tMax = (box_max - origin) / dir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float t_min = max(max(t1.x, t1.y), t1.z);
    float t_max = min(min(t2.x, t2.y), t2.z);
    return vec2(t_min, t_max);
}

uint8_t encode_index(u8vec3 position){
    return (position.x<<4) | (position.y << 2) | position.z;
}

struct ArHosekSkyModelChannelConfiguration {
    vec4 configs0;
    vec4 configs1;
    float configs2;
    float radiance;
    float ldCoefficient0;
    float ldCoefficient1;
    vec4 ldCoefficient2; // 2, 3, 4, 5
};

layout(set = 0, binding = 7, std430) uniform ArHosekSkyModelConfiguration{
    ArHosekSkyModelChannelConfiguration r;
    ArHosekSkyModelChannelConfiguration g;
    ArHosekSkyModelChannelConfiguration b;
    vec4 direction; // normalized.
    vec4 solar_intensity; // w is solar radius
} sunlight_config;


float ArHosekSkyModel_GetRadianceInternal(
        float[9]  configuration, 
        float                        cos_theta, 
        float                        gamma,
        float                        cos_gamma
        )
{
    float expM = exp(configuration[4] * gamma);
    float rayM = cos_gamma * cos_gamma;
    float mieM = (1.0 + rayM) / pow((1.0 + configuration[8]*configuration[8] - 2.0*configuration[8]*cos_gamma), 1.5);
    float zenith = sqrt(cos_theta);

    return (1.0 + configuration[0] * exp(configuration[1] / (cos_theta + 0.01))) *
            (configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith);
}

// dir: normalized view direction vector
vec3 arhosek_sky_radiance(vec3 dir)
{
    if (sunlight_config.direction.y <= 0) {
        // Avoid NaN problems.
        return vec3(0);
    }
    float cos_theta = clamp(dir.y, 0, 1);
    float cos_gamma = dot(dir, sunlight_config.direction.xyz);
    float gamma = acos(cos_gamma);


    float x =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.r.configs0.x,
            sunlight_config.r.configs0.y,
            sunlight_config.r.configs0.z,
            sunlight_config.r.configs0.w,
            sunlight_config.r.configs1.x,
            sunlight_config.r.configs1.y,
            sunlight_config.r.configs1.z,
            sunlight_config.r.configs1.w,
            sunlight_config.r.configs2
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.r.radiance;
    float y =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.g.configs0.x,
            sunlight_config.g.configs0.y,
            sunlight_config.g.configs0.z,
            sunlight_config.g.configs0.w,
            sunlight_config.g.configs1.x,
            sunlight_config.g.configs1.y,
            sunlight_config.g.configs1.z,
            sunlight_config.g.configs1.w,
            sunlight_config.g.configs2
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.g.radiance;
    float z =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.b.configs0.x,
            sunlight_config.b.configs0.y,
            sunlight_config.b.configs0.z,
            sunlight_config.b.configs0.w,
            sunlight_config.b.configs1.x,
            sunlight_config.b.configs1.y,
            sunlight_config.b.configs1.z,
            sunlight_config.b.configs1.w,
            sunlight_config.b.configs2
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.b.radiance;
    vec3 sky_color =  vec3(x, y, z) * 683.0;
    return sky_color;
}

vec3 arhosek_sun_radiance(
    vec3 dir
) {
    float cos_gamma = dot(dir, sunlight_config.direction.xyz);
    if (cos_gamma < 0.0 || dir.y < 0.0) {
        return vec3(0.0);
    }
    float sol_rad_sin = sin(sunlight_config.solar_intensity.w);
    float ar2 = 1.0 / (sol_rad_sin * sol_rad_sin);
    float singamma = 1.0 - (cos_gamma * cos_gamma);
    float sc2 = 1.0 - ar2 * singamma * singamma;
    if (sc2 <= 0.0) {
        return vec3(0.0);
    }
    float sampleCosine = sqrt(sc2);

    vec3 darkeningFactor = vec3(sunlight_config.r.ldCoefficient0, sunlight_config.g.ldCoefficient0, sunlight_config.b.ldCoefficient2);


    darkeningFactor += vec3(sunlight_config.r.ldCoefficient1, sunlight_config.g.ldCoefficient1, sunlight_config.b.ldCoefficient1) * sampleCosine;

    float currentSampleCosine = sampleCosine;
    [[unroll]]
    for (uint i = 0; i < 4; i++) {
        currentSampleCosine *= sampleCosine;
        darkeningFactor += vec3(
            sunlight_config.r.ldCoefficient2[i],
            sunlight_config.g.ldCoefficient2[i],
            sunlight_config.b.ldCoefficient2[i]
        ) * currentSampleCosine;
    }
    return sunlight_config.solar_intensity.xyz * darkeningFactor;
}

