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
layout(push_constant) uniform PushConstants {
    // Indexed by block id
    uint rand;
    uint frameIndex;
} pushConstants;

layout(shaderRecordEXT) buffer sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
};
layout(location = 0) rayPayloadInEXT float hitT;
hitAttributeEXT HitAttribute {
    uint8_t voxelId;
    uint8_t faceId;
} hitAttributes;

struct RadianceHashMapEntry {
    vec3 energy;
    uint32_t lastAccessedFrameIndex;
};
layout(set = 0, binding = 4) buffer RadianceHashMap {
    uint num_entries;
    RadianceHashMapEntry[] entries;
} radianceCache;

uint pcg_hash(uint in_data)
{
    uint state = in_data * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}


uint hashPayload(
    uint instanceId,
    uint primitiveId,
    uint voxelId,
    uint faceId
) {
    uint id1 = instanceId * 70297021 + primitiveId * 256 + voxelId * 4 + faceId;
    return pcg_hash(id1);
}


void main() {
    Block block = geometryInfo.blocks[gl_PrimitiveID];

    u32vec2 masked = unpack32(block.mask & ((uint64_t(1) << hitAttributes.voxelId) - 1));
    uint32_t voxelMemoryOffset = bitCount(masked.x) + bitCount(masked.y);

    uint8_t palette_index = materialInfo.materials[block.material_ptr + voxelMemoryOffset];
    u8vec4 color = paletteInfo.palette[palette_index];

    uint hash = hashPayload(gl_InstanceID, gl_PrimitiveID, hitAttributes.voxelId, hitAttributes.faceId) % radianceCache.num_entries;
    RadianceHashMapEntry hashEntry = radianceCache.entries[hash];
    vec3 energy = hashEntry.energy * pow(0.999, pushConstants.frameIndex - hashEntry.lastAccessedFrameIndex);

    vec3 diffuseColor = vec3(color) / 255.0;
    // The parameter 0.01 was derived from the 0.999 retention factor. It's not arbitrary.
    vec3 indirectContribution = 0.002 * energy * diffuseColor;

    // Store the contribution from photon maps
    imageStore(u_imgOutput, ivec2(gl_LaunchIDEXT.xy), vec4(indirectContribution, 1.0));
    imageStore(u_diffuseOutput, ivec2(gl_LaunchIDEXT.xy), vec4(diffuseColor, 1.0));
    hitT = gl_HitTEXT;
}
