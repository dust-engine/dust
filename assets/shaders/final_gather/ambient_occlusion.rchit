#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/nrd.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


void main() {
    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(payload.illuminance, gl_HitTEXT);
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}
