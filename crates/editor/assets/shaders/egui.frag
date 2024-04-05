#version 460
#include "draw.playout"

layout(location = 0) in vec4 in_Color;
layout(location = 1) in vec2 in_UV;
layout(location = 0) out vec4 out_Color;

void main(){
    out_Color = in_Color * texture(u_img, in_UV);
}
