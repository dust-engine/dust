#include "../headers/header.glsl"
#include "../headers/layout.playout"


layout(location = 0) rayPayloadEXT uint _rayPayloadNotUsed;
layout(shaderRecordEXT) buffer Sbt {
    float color[4];
} sbt;

void main() {
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(abs(gl_WorldRayDirectionEXT), 1.0));
}
