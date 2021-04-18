#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
struct Node {
    uint8_t _padding1;
    uint8_t freemask;
    uint16_t _padding2;
    uint children;
    uint16_t data[8];
};

struct Ray {
    vec3 origin;
    vec3 dir;
};
struct Box {
    vec3 origin;
    float extent;
};
struct Sunlight {
    vec3 color;
    float padding1;
    vec3 dir;
    float padding2;
};
uint MaskLocationNthOne(uint mask, uint location) {
    return bitCount(mask & ((1 << location) - 1));
}


layout(early_fragment_tests) in;
layout (constant_id = 0) const uint MAX_ITERATION_VALUE = 100;
const Box GlobalBoundingBox = { vec3(0,0,0), 512 };

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform u_Camera {
    mat4 ViewProj;
    mat4 RotationViewProj;
    vec3 position;
    float placeholder;
    vec3 forward;
    float placeholder2;
    float fov;
    float near;
    float far;
    float aspect_ratio;
} Camera;
layout(set = 0, binding = 1) uniform Lights {
    Sunlight Lights_Sunlight;
};
layout(set = 1, binding = 0) readonly buffer Chunk {
    Node Chunk_Nodes[];
};
struct Material {
    float scale;
    uint16_t diffuse;
    uint16_t normal;
    float _reserved1;
    float _reserved2;
};
struct ColoredMaterial {
    float scale;
    uint16_t diffuse;
    uint16_t normal;
    float _reserved1;
    float _reserved2;
    vec4 palette[256];
};
layout(set = 1, binding = 1) readonly buffer Materials {
    Material u_RegularMaterials[];
};
layout(set = 1, binding = 2) readonly buffer ColoredMaterials {
    ColoredMaterial u_ColoredMaterials[];
};
layout(set = 1, binding = 3) uniform sampler2DArray TextureRepoSampler;
layout (input_attachment_index = 0, set = 2, binding = 0) uniform subpassInput inputDepth;

Ray GenerateRay() {
    Ray ray;
    ray.origin = Camera.position;
    ray.dir = normalize(vWorldPosition);
    return ray;
}
vec2 IntersectAABB(vec3 origin, vec3 dir, Box box) {
    vec3 box_min = box.origin;
    vec3 box_max = box_min + box.extent;
    vec3 tMin = (box_min - origin) / dir;
    vec3 tMax = (box_max - origin) / dir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float t_min = max(max(t1.x, t1.y), t1.z);
    float t_max = min(min(t2.x, t2.y), t2.z);
    return vec2(t_min, t_max);
}
bool ContainsAABB(vec3 point, Box box) {
    vec3 min = box.origin;
    vec3 max = min + box.extent;

    vec3 s = step(min, point) - step(max, point);
    bvec3 bs = bvec3(s);
    return all(bs);
}
vec3 CubedNormalize(vec3 dir) {
    vec3 dir_abs = abs(dir);
    float max_element = max(dir_abs.x, max(dir_abs.y, dir_abs.z));
    return -sign(dir) * step(max_element, dir_abs);
}
uint MaterialAtPosition(inout Box box, vec3 position) {
    uint node_index = 0; // Assume root node

    while(true) {
        // start
        // Calculate new box location
        box.extent = box.extent / 2;
        vec3 box_midpoint = box.origin + box.extent;
        vec3 s = step(box_midpoint, position);
        box.origin = box.origin + s * box.extent;


        uint child_index = uint(dot(s, vec3(4,2,1)));
        uint freemask = uint(Chunk_Nodes[node_index].freemask);
        if ((freemask & (1 << child_index)) == 0) {
            // is a leaf node
            return uint(Chunk_Nodes[node_index].data[child_index]);
        } else {
            // has children
            uint child_offset = MaskLocationNthOne(freemask, child_index);
            node_index = Chunk_Nodes[node_index].children + child_offset;
        }
    }
}

uint RayMarch(Box initial_box, Ray ray, out vec3 hitpoint, out Box hitbox, out uint counter, float initialDistance) {
    hitbox = initial_box;
    vec3 entry_point = ray.origin + initialDistance * ray.dir;
    vec3 test_point = entry_point;
    uint material_id = 0;

    for(
    counter = 0;
    counter < MAX_ITERATION_VALUE && ContainsAABB(test_point, GlobalBoundingBox);
    counter++) {
        // TODO: change this so that entry point doesn't get too big
        hitbox = initial_box;
        material_id = MaterialAtPosition(hitbox, test_point);
        if (material_id > 0) {
            // Hit some materials
            break;
        }
        // calculate the next t_min
        vec2 new_intersection = IntersectAABB(entry_point, ray.dir, hitbox);

        entry_point = entry_point + ray.dir * new_intersection.y;
        test_point = entry_point + sign(ray.dir) * hitbox.extent * 0.001;
    }
    hitpoint = entry_point;
    return material_id;
}

//#define DEBUG_RENDERING

void main() {
    Ray ray = GenerateRay();
    float distance = subpassLoad(inputDepth).r;
    if (distance == 1.0) {
        distance = 0.0;
    } else {
        distance = Camera.near * Camera.far / (Camera.far - distance*(Camera.far - Camera.near));
        distance = distance / dot(Camera.forward, ray.dir);
    }

    float depth;
    vec3 hitpoint;
    Box hitbox;
    uint iteration_times;
    uint voxel_id = RayMarch(GlobalBoundingBox, ray, hitpoint, hitbox, iteration_times, distance);
    float iteration = float(iteration_times) / float(MAX_ITERATION_VALUE); // 0 to 1
    #ifdef DEBUG_RENDERING
    f_color = vec4(iteration, iteration, iteration, 1.0);
    #else
    vec3 normal = CubedNormalize(hitpoint - (hitbox.origin + hitbox.extent/2));
    vec2 texcoords = vec2(
        dot(vec3(hitpoint.z, hitpoint.x, -hitpoint.x), normal),
        dot(-sign(normal) * vec3(hitpoint.y, hitpoint.z, hitpoint.y), normal)
    );

    if (voxel_id == 0) {
        f_color = vec4(0.0, 1.0, 0.0, 1.0);
    } else {
        float sunLightFactor = min(1.0, dot(normal, Lights_Sunlight.dir));
        vec4 output_color = vec4(sunLightFactor, sunLightFactor, sunLightFactor, 1.0);
        vec4 texture_color = texture(
            TextureRepoSampler,
            vec3(
                texcoords * 10.0,
                uint(u_RegularMaterials[voxel_id - 1].diffuse)
            )
        );
        f_color = output_color * texture_color;
    }
    #endif
}
