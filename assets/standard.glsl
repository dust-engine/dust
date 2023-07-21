
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_samplerless_texture_functions: require
#extension GL_EXT_control_flow_attributes: require

#extension GL_EXT_debug_printf : enable
// Illuminance: total luminous flux incident on a surface, per unit area.
// Unit: lux (lm / m^2)
// Stores the incoming radiance at primary ray hit points.
layout(set = 0, binding = 0, rgba32f) uniform image2D u_illuminance;
layout(set = 0, binding = 1, rgb10_a2) uniform image2D u_albedo;
layout(set = 0, binding = 2, rgba16_snorm) uniform image2D u_normal;
layout(set = 0, binding = 3, r32f) uniform image2D u_depth;
layout(set = 0, binding = 4, rg16f) uniform image2D u_motion;

layout(set = 0, binding = 5) uniform accelerationStructureEXT accelerationStructure;
layout(set = 0, binding = 6) uniform texture2D blue_noise;


layout(set = 0, binding = 8, std430) uniform CameraSettingsLastFrame {
    mat4 view_proj;
    mat4 inverse_view_proj;
    vec3 camera_view_col0;
    float position_x;
    vec3 camera_view_col1;
    float position_y;
    vec3 camera_view_col2;
    float position_z;
    float tan_half_fov;
    float far;
    float near;
    float padding;
} u_camera_last_frame;
layout(set = 0, binding = 9, std430) uniform CameraSettings {
    mat4 view_proj;
    mat4 inverse_view_proj;
    vec3 camera_view_col0;
    float position_x;
    vec3 camera_view_col1;
    float position_y;
    vec3 camera_view_col2;
    float position_z;
    float tan_half_fov;
    float far;
    float near;
    float padding;
} u_camera;
layout(set = 0, binding = 10, std430) buffer InstanceData {
    mat4 last_frame_transforms[];
} s_instances;
vec3 camera_origin() {
    return vec3(u_camera.position_x, u_camera.position_y, u_camera.position_z);
}
vec3 camera_ray_dir() {
    const vec2 pixelNDC = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);

    vec2 pixelCamera = 2 * pixelNDC - 1;
    pixelCamera.y *= -1;
    pixelCamera.x *= float(gl_LaunchSizeEXT.x) / float(gl_LaunchSizeEXT.y);
    pixelCamera *= u_camera.tan_half_fov;

    const mat3 rotationMatrix = mat3(u_camera.camera_view_col0, u_camera.camera_view_col1, u_camera.camera_view_col2);

    const vec3 pixelCameraWorld = rotationMatrix * vec3(pixelCamera, -1);
    return pixelCameraWorld;
}

#define RETENTION_FACTOR 0.99

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

float SRGBToLinear(float color)
{
    // Approximately pow(color, 2.2)
    return color < 0.04045 ? color / 12.92 : pow(abs(color + 0.055) / 1.055, 2.4);
}

vec3 SRGBToXYZ(vec3 srgb) {
    mat3 transform = mat3(
        0.4124564, 0.2126729, 0.0193339,
        0.3575761, 0.7151522, 0.1191920,
        0.1804375, 0.0721750, 0.9503041
    );
    return transform * srgb;
}


uint hash1(uint x) {
	x += (x << 10u);
	x ^= (x >>  6u);
	x += (x <<  3u);
	x ^= (x >> 11u);
	x += (x << 15u);
	return x;
}

uint hash1_mut(inout uint h) {
    uint res = h;
    h = hash1(h);
    return res;
}

uint hash_combine2(uint x, uint y) {
    const uint M = 1664525u, C = 1013904223u;
    uint seed = (x * M + y + C) * M;

    // Tempering (from Matsumoto)
    seed ^= (seed >> 11u);
    seed ^= (seed << 7u) & 0x9d2c5680u;
    seed ^= (seed << 15u) & 0xefc60000u;
    seed ^= (seed >> 18u);
    return seed;
}

uint hash2(uvec2 v) {
	return hash_combine2(v.x, hash1(v.y));
}

uint hash3(uvec3 v) {
	return hash_combine2(v.x, hash2(v.yz));
}

uint hash4(uvec4 v) {
	return hash_combine2(v.x, hash3(v.yzw));
}


float uint_to_u01_float(uint h) {
	const uint mantissaMask = 0x007FFFFFu;
	const uint one = 0x3F800000u;

	h &= mantissaMask;
	h |= one;

	float  r2 = uintBitsToFloat( h );
	return r2 - 1.0;
}

struct Sample {
    vec3 visible_point; // The primary ray hit point
    vec3 visible_point_normal; // The normal at the primary ray hit point in world space
    vec3 sample_point; // The final gather ray / secondary ray hit point
    vec3 sample_point_normal; // The normal at the secondary ray hit point in world space
    vec3 outgoing_radiance; // Outgoing radiance at the sample point in XYZ color space
};

struct Reservoir {
    Sample current_sample;      // z
    float total_weight; // w
    uint sample_count; // M
};

void ReservoirUpdate(inout Reservoir self, Sample new_sample, float sample_weight, inout uint rng) {
    self.total_weight += sample_weight;
    self.sample_count += 1;
    const float dart = uint_to_u01_float(hash1_mut(rng));

    if ((self.sample_count == 1) || (dart < sample_weight / self.total_weight)) {
        self.current_sample = new_sample;
    }
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This function assumes the newReservoir has been normalized, so its weightSum means "1/g * 1/M * \sum{g/p}"
// and the targetPdf is a conversion factor from the newReservoir's space to the reservoir's space (integrand).
void ReservoirMerge(inout Reservoir self, Reservoir other, float target_pdf, inout uint rng) {
    uint total_sample_count = self.sample_count + other.sample_count;
    ReservoirUpdate(self, other.current_sample, target_pdf * other.total_weight * other.sample_count, rng);
    self.sample_count = total_sample_count;
}

layout(set = 0, binding = 11, std430) buffer ReservoirData {
    Reservoir reservoirs[];
} s_reservoirs;

layout(set = 0, binding = 12, std430) buffer ReservoirDataPrev {
    Reservoir reservoirs[];
} s_reservoirs_prev;