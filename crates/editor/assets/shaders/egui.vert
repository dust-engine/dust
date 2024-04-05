#version 460
#include "draw.playout"

layout(location = 0) in vec2 in_Position;
layout(location = 1) in vec2 in_UV;
layout(location = 2) in vec4 in_Color;
layout(location = 0) out vec4 out_Color;
layout(location = 1) out vec2 out_UV;

vec3 srgbToLinear(vec3 sRGB)
{
	bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
	vec3 higher = pow((sRGB + vec3(0.055))/vec3(1.055), vec3(2.4));
	vec3 lower = sRGB/vec3(12.92);

	return mix(higher, lower, cutoff);
}

void main(){
    vec2 size = vec2(1280.0, 720.0);
    gl_Position = vec4(2.0 * in_Position / u_transform - 1.0, 0.0, 1.0);
    out_Color = vec4(srgbToLinear(in_Color.xyz), in_Color.w);
    out_UV = in_UV;
}
