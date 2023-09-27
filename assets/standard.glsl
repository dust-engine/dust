
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
layout(set = 0, binding = 0, rgba16f) uniform image2D u_illuminance;
layout(set = 0, binding = 1, rgba16f) uniform image2D u_illuminance_denoised;
layout(set = 0, binding = 2, rgb10_a2) uniform image2D u_albedo;
layout(set = 0, binding = 3, rgb10_a2) uniform image2D u_normal;
layout(set = 0, binding = 4, r32f) uniform image2D u_depth;
layout(set = 0, binding = 5, rgba16f) uniform image2D u_motion;
layout(set = 0, binding = 6, r32ui) uniform uimage2D u_voxel_id;
layout(set = 0, binding = 7) uniform texture2D blue_noise[6]; // [1d, 2d, unit2d, 3d, unit3d, unit3d_cosine]

layout(set = 0, binding = 9, std430) uniform CameraSettingsLastFrame {
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
layout(set = 0, binding = 10, std430) uniform CameraSettings {
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
layout(set = 0, binding = 11, std430) buffer InstanceData {
    mat4 last_frame_transforms[];
} s_instances;
layout(set = 0, binding = 14) uniform accelerationStructureEXT accelerationStructure;
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

layout(set = 0, binding = 8, std430) uniform ArHosekSkyModelConfiguration{
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
vec2 _NRD_EncodeUnitVector( vec3 v, const bool bSigned )
{
    v /= dot( abs( v ), vec3(1.0) );

    vec2 octWrap = ( 1.0 - abs( v.yx ) ) * ( step( 0.0, v.xy ) * 2.0 - 1.0 );
    v.xy = v.z >= 0.0 ? v.xy : octWrap;

    return bSigned ? v.xy : v.xy * 0.5 + 0.5;
}

#define NRD_ROUGHNESS_ENCODING_SQ_LINEAR                                                0 // linearRoughness * linearRoughness
#define NRD_ROUGHNESS_ENCODING_LINEAR                                                   1 // linearRoughness
#define NRD_ROUGHNESS_ENCODING_SQRT_LINEAR                                              2 // sqrt( linearRoughness )

#define NRD_NORMAL_ENCODING_RGBA8_UNORM                                                 0
#define NRD_NORMAL_ENCODING_RGBA8_SNORM                                                 1
#define NRD_NORMAL_ENCODING_R10G10B10A2_UNORM                                           2 // supports material ID bits
#define NRD_NORMAL_ENCODING_RGBA16_UNORM                                                3
#define NRD_NORMAL_ENCODING_RGBA16_SNORM                                                4 // also can be used with FP formats


#define NRD_NORMAL_ENCODING NRD_NORMAL_ENCODING_R10G10B10A2_UNORM
#define NRD_ROUGHNESS_ENCODING NRD_ROUGHNESS_ENCODING_LINEAR
vec4 NRD_FrontEnd_PackNormalAndRoughness(vec3 N, float roughness, float materialID )
{
    vec4 p;

    #if( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR )
        roughness = sqrt( clamp( roughness, 0, 1 ) );
    #elif( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR )
        roughness *= roughness;
    #endif

    #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        p.xy = _NRD_EncodeUnitVector( N, false );
        p.z = roughness;
        p.w = clamp( materialID / 3.0, 0.0, 1.0 );
    #else
        // Best fit ( optional )
        N /= max( abs( N.x ), max( abs( N.y ), abs( N.z ) ) );

        #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM )
            N = N * 0.5 + 0.5;
        #endif

        p.xyz = N;
        p.w = roughness;
    #endif

    return p;
}

vec3 _NRD_DecodeUnitVector( vec2 p, const bool bSigned, const bool bNormalize )
{
    p = bSigned ? p : ( p * 2.0 - 1.0 );

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3( p.xy, 1.0 - abs( p.x ) - abs( p.y ) );
    float t = clamp( -n.z, 0.0, 1.0 );
    n.xy -= t * ( step( 0.0, n.xy ) * 2.0 - 1.0 );

    return bNormalize ? normalize( n ) : n;
}

vec4 NRD_FrontEnd_UnpackNormalAndRoughness( vec4 p, out float materialID )
{
    vec4 r;
    #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        r.xyz = _NRD_DecodeUnitVector( p.xy, false, false );
        r.w = p.z;

        materialID = p.w;
    #else
        #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM )
            p.xyz = p.xyz * 2.0 - 1.0;
        #endif

        r.xyz = p.xyz;
        r.w = p.w;

        materialID = 0;
    #endif

    r.xyz = normalize( r.xyz );

    #if( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR )
        r.w *= r.w;
    #elif( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR )
        r.w = sqrt( r.w );
    #endif

    return r;
}


#define NRD_FP16_MIN 1e-7 // min allowed hitDist (0 = no data)

vec3 _NRD_LinearToYCoCg( vec3 color )
{
    float Y = dot( color, vec3( 0.25, 0.5, 0.25 ) );
    float Co = dot( color, vec3( 0.5, 0.0, -0.5 ) );
    float Cg = dot( color, vec3( -0.25, 0.5, -0.25 ) );

    return vec3( Y, Co, Cg );
}


vec4 REBLUR_FrontEnd_PackRadianceAndNormHitDist( vec3 radiance, float normHitDist)
{
    /*
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0, NRD_FP16_MAX );
        normHitDist = ( isnan( normHitDist ) | isinf( normHitDist ) ) ? 0 : saturate( normHitDist );
    }
    */

    // "0" is reserved to mark "no data" samples, skipped due to probabilistic sampling
    if( normHitDist != 0 )
        normHitDist = max( normHitDist, NRD_FP16_MIN );

    radiance = _NRD_LinearToYCoCg( radiance );

    return vec4( radiance, normHitDist );
}

struct SpatialHashEntry {
    uint32_t fingerprint;
    uint16_t last_accessed_frame;
    uint16_t sample_count;
    f16vec3 radiance; // The amount of incoming radiance
    float16_t visual_importance;
};

struct SpatialHashKey {
    ivec3 position;
    uint8_t direction; // [0, 6) indicating one of the six faces of the cube
};


// https://www.pcg-random.org/
uint pcg(in uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

    return (word >> 22u) ^ word;
}

// xxhash (https://github.com/Cyan4973/xxHash)
//   From: https://www.shadertoy.com/view/Xt3cDn
uint xxhash32(in uint p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;

    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));

    return h32 ^ (h32 >> 16);
}

uint32_t SpatialHashKeyGetFingerprint(SpatialHashKey key) {
    uint hash = xxhash32(key.position.x);
    hash = xxhash32(key.position.y + hash);
    hash = xxhash32(key.position.z + hash);
    hash = xxhash32(key.direction + hash);
    hash = max(1, hash);
    return hash;
}
layout(constant_id = 0) const uint32_t SpatialHashCapacity = 32 * 1024 * 1024; // 512 MB
uint32_t SpatialHashKeyGetLocation(SpatialHashKey key) {
    uint hash = pcg(key.position.x);
    hash = pcg(key.position.y + hash);
    hash = pcg(key.position.z + hash);
    hash = pcg(key.direction + hash);
    return hash % SpatialHashCapacity;
}


layout(set = 0, binding = 12) buffer SpatialHash {
    SpatialHashEntry entries[];
} s_spatial_hash;

// input param: a vector with only one component being 1 or -1, the rest being 0
// +1 0 0 | 0b101
// -1 0 0 | 0b100
// 0 +1 0 | 0b011
// 0 -1 0 | 0b010
// 0 0 +1 | 0b001
// 0 0 -1 | 0b000
uint8_t normal2FaceID(vec3 normalObject) {
    float s = clamp(normalObject.x + normalObject.y + normalObject.z, 0.0, 1.0); // Sign of the nonzero component
    uint8_t faceId = uint8_t(s); // The lowest digit is 1 if the sign is positive, 0 otherwise

    // 4 (0b100) if z is the nonzero component, 2 (0b010) if y is the nonzero component, 0 if x is the nonzero component
    uint8_t index = uint8_t(abs(normalObject.z)) * uint8_t(4) + uint8_t(abs(normalObject.y)) * uint8_t(2);

    faceId += index;
    return faceId;
}

vec3 faceId2Normal(uint8_t faceId) {
    float s = float(faceId & 1) * 2.0 - 1.0; // Extract the lowest component and restore as the sign.

    vec3 normal = vec3(0);
    normal[faceId >> 1] = s;
    return normal;
}

// This function rotates `target` from the z axis by the same amount as `normal`.
// param: normal: a unit vector.
//        sample: the vector to be rotated
vec3 rotateVectorByNormal(vec3 normal, vec3 target) {
    vec4 quat = normalize(vec4(-normal.y, normal.x, 0.0, 1.0 + normal.z));
    if (normal.z < -0.99999) {
        quat = vec4(-1.0, 0.0, 0.0, 0.0);
    }
    return 2.0 * dot(quat.xyz, target) * quat.xyz + (quat.w * quat.w - dot(quat.xyz, quat.xyz)) * target + 2.0 * quat.w * cross(quat.xyz, target);
}

void SpatialHashInsert(SpatialHashKey key, vec3 value) {
    uint fingerprint = SpatialHashKeyGetFingerprint(key);
    uint location = SpatialHashKeyGetLocation(key);


    uint i_minFrameIndex;
    uint minFrameIndex;
    for (uint i = 0; i < 3; i++) {
        uint current_fingerprint = s_spatial_hash.entries[location + i].fingerprint;
        uint current_frame_index = s_spatial_hash.entries[location + i].last_accessed_frame;
        if (i == 0 || current_frame_index < minFrameIndex) {
            i_minFrameIndex = i;
            minFrameIndex = current_frame_index;
        }


        if (current_fingerprint == fingerprint || current_fingerprint == 0) {
            // Found.
            if (current_fingerprint == 0) {
                s_spatial_hash.entries[location + i].fingerprint = fingerprint;
            }

            vec3 current_radiance = vec3(0.0);
            uint current_sample_count = 0;
            if (current_fingerprint == fingerprint) {
                current_sample_count = s_spatial_hash.entries[location + i].sample_count;
                current_radiance = s_spatial_hash.entries[location + i].radiance;
            }
            #define MAX_SAMPLE_COUNT 256
            current_sample_count = min(current_sample_count, MAX_SAMPLE_COUNT - 1);
            uint next_sample_count = current_sample_count + 1;
            current_radiance = current_radiance * (float(current_sample_count) / float(next_sample_count)) + value * (1.0 / float(next_sample_count));
            
            s_spatial_hash.entries[location + i].radiance = f16vec3(current_radiance);
            s_spatial_hash.entries[location + i].last_accessed_frame = uint16_t(pushConstants.frameIndex);
            s_spatial_hash.entries[location + i].sample_count = uint16_t(next_sample_count);
            return;
        }
    }
    // Not found after 3 iterations. Evict the LRU entry.
    s_spatial_hash.entries[location + i_minFrameIndex].fingerprint = fingerprint;
    uint current_sample_count = s_spatial_hash.entries[location + i_minFrameIndex].sample_count;
    vec3 current_radiance = s_spatial_hash.entries[location + i_minFrameIndex].radiance;
    uint next_sample_count = current_sample_count + 1;
    current_radiance = current_radiance * (float(current_sample_count) / float(next_sample_count)) + value * (1.0 / float(next_sample_count));
    s_spatial_hash.entries[location + i_minFrameIndex].radiance = f16vec3(current_radiance);
    s_spatial_hash.entries[location + i_minFrameIndex].last_accessed_frame = uint16_t(pushConstants.frameIndex);
    s_spatial_hash.entries[location + i_minFrameIndex].sample_count = uint16_t(next_sample_count);
}


// Returns: found
// out vec3: value. The outgoing radiance at the voxel.
bool SpatialHashGet(SpatialHashKey key, out vec3 value, out uint sample_count) {
    uint fingerprint = SpatialHashKeyGetFingerprint(key);
    uint location = SpatialHashKeyGetLocation(key);
    value = vec3(0.0);
    sample_count = 0;
    for (uint i = 0; i < 3; i++) {
        uint current_fingerprint = s_spatial_hash.entries[location + i].fingerprint;
        if (current_fingerprint == 0) {
            return false; // Found an empty entry, so we terminate the search early.
        }

        if (current_fingerprint == fingerprint) {
            // Found.
            s_spatial_hash.entries[location + i].last_accessed_frame = uint16_t(pushConstants.frameIndex);
            value = s_spatial_hash.entries[location + i].radiance;
            sample_count = s_spatial_hash.entries[location + i].sample_count;
            return true;
        }
    }
    return false;
}



struct SurfelEntry { 
    ivec3 position;
    uint32_t direction; // [0, 6) indicating one of the six faces of the cube
};
layout(constant_id = 1) const uint32_t SurfelPoolSize = 720*480;

layout(set = 0, binding = 13) buffer SurfelPool {
    SurfelEntry entries[];
} s_surfel_pool;
