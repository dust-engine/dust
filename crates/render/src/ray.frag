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
uint MaskLocationNthOne(uint mask, uint location) {
    return bitCount(mask & ((1 << location) - 1));
}


layout (constant_id = 0) const uint MAX_ITERATION_VALUE = 1000;
const Box GlobalBoundingBox = { vec3(0,0,0), 16 };

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 Proj;
};
layout(set = 1, binding = 0) readonly buffer Chunk {
    Node Chunk_Nodes[];
};

Ray GenerateRay() {
    Ray ray;
    ray.origin = Proj[3].xyz;
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

uint RayMarch(Box initial_box, Ray ray, out vec3 hitpoint, out Box hitbox, out uint iteration_times) {
    hitbox = initial_box;
    vec2 intersection = IntersectAABB(ray.origin, ray.dir, hitbox);
    vec3 entry_point = ray.origin + max(0, intersection.x) * ray.dir;
    vec3 test_point = entry_point + ray.dir * hitbox.extent * 0.000001;
    uint material_id = 0;

    uint counter;
    for(
    counter = 0;
    counter < MAX_ITERATION_VALUE && ContainsAABB(test_point, GlobalBoundingBox);
    counter++) {
        vec4 entry_point_camera_space = ViewProj * vec4(entry_point, 1.0);
        hitbox = initial_box;
        material_id = MaterialAtPosition(hitbox, test_point);
        if (material_id > 0) {
            // Hit some materials
            break;
        }
        // calculate the next t_min
        vec2 new_intersection = IntersectAABB(entry_point, ray.dir, hitbox);

        entry_point = entry_point + ray.dir * new_intersection.y;
        test_point = entry_point + sign(ray.dir) * hitbox.extent * 0.0001;
    }
    hitpoint = entry_point;
    iteration_times = counter;
    return material_id;
}


void main() {
    Ray ray = GenerateRay();

    float depth;
    vec3 hitpoint;
    Box hitbox;
    uint iteration_times;
    uint voxel_id = RayMarch(GlobalBoundingBox, ray, hitpoint, hitbox, iteration_times);
    float iteration = float(iteration_times) / float(MAX_ITERATION_VALUE) * 10.0; // 0 to 1
    f_color = vec4(iteration, iteration, iteration, 1.0);
}
