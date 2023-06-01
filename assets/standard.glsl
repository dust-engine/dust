
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_samplerless_texture_functions: require

// Illuminance: total luminous flux incident on a surface, per unit area.
// Unit: lux (lm / m^2)
layout(set = 0, binding = 0, rgba32f) uniform image2D u_illuminance;
layout(set = 0, binding = 1, rgb10_a2) uniform image2D u_albedo;
layout(set = 0, binding = 2, rgba16_snorm) uniform image2D u_normal;
layout(set = 0, binding = 3, r32f) uniform image2D u_depth;

layout(set = 0, binding = 4) uniform accelerationStructureEXT accelerationStructure;
layout(set = 0, binding = 5) uniform texture2D blue_noise;

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
    float configuration[9];
    float radiance;
};

layout(set = 0, binding = 6) uniform ArHosekSkyModelConfiguration{
    float params[32];
    vec3 direction; // normalized
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

// dir: normalized direction vector
vec3 arhosek_sky_radiance(vec3 dir)
{
    float cos_theta = clamp(dir.y, 0, 1);
    float cos_gamma = dot(dir, sunlight_config.direction);
    float gamma = acos(cos_gamma);


    float x =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.params[0],
            sunlight_config.params[1],
            sunlight_config.params[2],
            sunlight_config.params[3],
            sunlight_config.params[4],
            sunlight_config.params[5],
            sunlight_config.params[6],
            sunlight_config.params[7],
            sunlight_config.params[8]
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.params[9];
    float y =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.params[10],
            sunlight_config.params[11],
            sunlight_config.params[12],
            sunlight_config.params[13],
            sunlight_config.params[14],
            sunlight_config.params[15],
            sunlight_config.params[16],
            sunlight_config.params[17],
            sunlight_config.params[18]
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.params[19];
    float z =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.params[20],
            sunlight_config.params[21],
            sunlight_config.params[22],
            sunlight_config.params[23],
            sunlight_config.params[24],
            sunlight_config.params[25],
            sunlight_config.params[26],
            sunlight_config.params[27],
            sunlight_config.params[28]
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.params[29];
    return vec3(x, y, z);
}
