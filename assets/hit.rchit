#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 0) uniform writeonly image2D u_imgOutput;
layout(set = 0, binding = 1) uniform writeonly image2D u_diffuseOutput;
struct Block
{
    u16vec4 position;
    uint64_t mask;
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

layout(shaderRecordEXT) buffer sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
};
layout(location = 0) rayPayloadInEXT float hitT;

hitAttributeEXT uint8_t voxelId;


struct RadianceHashMapEntry {
    vec3 energy;
    uint32_t lastAccessedFrameIndex;
};
layout(set = 0, binding = 4) buffer RadianceHashMap {
    uint num_entries;
    RadianceHashMapEntry[] entries;
} radianceCache;

uvec3 pcg3d(uvec3 v)
{
    v = v * 1664525 + 1013904223;
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v = v ^ (v >> 16);
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    return v;
}
uint hashPayload(
    uint instanceId,
    uint primitiveId,
    uint voxelId,
    uint faceId
) {
    uvec3 result = pcg3d(uvec3(instanceId, primitiveId, voxelId << 16 | faceId));
    return result.x + result.y + result.z;
}



void main() {
    Block block = geometryInfo.blocks[gl_PrimitiveID];

    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << voxelId) - 1));
    uint32_t voxelMemoryOffset = bitCount(masked.x) + bitCount(masked.y);

    uint8_t palette_index = materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = paletteInfo.palette[palette_index];

    uint hash = hashPayload(gl_InstanceID, gl_PrimitiveID, voxelId, 0) % radianceCache.num_entries;
    vec3 energy = radianceCache.entries[hash].energy;

    vec3 diffuseColor = vec3(color) / 255.0;
    // The parameter 0.01 was derived from the 0.99 retention factor. It's not arbitrary.
    vec3 indirectContribution = 0.01 * energy * diffuseColor;

    // Store the contribution from photon maps
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(indirectContribution, 1.0));
    imageStore(u_diffuseOutput, ivec2(gl_LaunchIDEXT.xy), vec4(diffuseColor, 1.0));
    hitT = gl_HitTEXT;
}
