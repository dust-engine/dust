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

#define M_PI 3.1415926535897932384626433832795

// Eye -> Object -> Skylight
#define CONTRIBUTION_SECONDARY_SKYLIGHT
// Eye -> Object -> Object (sample spatial hash)
#define CONTRIBUTION_SECONDARY_SPATIAL_HASH
// Eye -> Object -> Sun
#define CONTRIBUTION_DIRECT

// Eye -> Object -> Surfel -> Sun
#define CONTRIBUTION_SECONDARY_SUNLIGHT

//#define DEBUG_VISUALIZE_SPATIAL_HASH

#define AMBIENT_OCCLUSION_THRESHOLD 8.0
