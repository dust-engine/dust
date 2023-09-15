#version 460
#include "standard.glsl"

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
} sbt;

hitAttributeEXT HitAttribute {
    uint8_t voxelId;
} hitAttributes;
layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


void main() {
    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(payload.illuminance, gl_HitTEXT);
    imageStore(u_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}
