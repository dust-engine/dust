#include "../headers/standard.glsl"
#include "../headers/layout.playout"
#include "../headers/nrd.glsl"

layout(location = 0) rayPayloadInEXT struct RayPayload {
    vec3 illuminance;
} payload;


void main() {
    vec4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(vec3(0.0), 0.0);
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), packed);
}
