#include "../headers/header.glsl"
#include "../headers/layout.playout"


void main() {
    imageStore(img_illuminance, ivec2(gl_LaunchIDEXT.xy), vec4(1.0, 1.0, 0.0, 1.0));
}
