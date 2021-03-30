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


layout (constant_id = 0) const uint MAX_ITERATION_VALUE = 1000;
const vec4 bounding_box = vec4(0.0, 0.0, 0.0, 16.0);

layout(location=0) out vec4 f_color;
layout(location=0) in vec3 vWorldPosition;
layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
    mat4 transform;
};
layout(set = 1, binding = 0) readonly buffer Chunk {
    Node nodes[];
};

Ray generate_ray() {
    Ray ray;
    ray.origin = transform[3].xyz;
    ray.dir = normalize(vWorldPosition);
    return ray;
}

// Given a mask an a location, returns n where the given '1' on the location
// is the nth '1' counting from the Least Significant Bit.
uint mask_location_nth_one(uint mask, uint location) {
    return bitCount(mask & ((1 << location) - 1));
}

vec2 intersectAABB(vec3 origin, vec3 dir, vec4 box) {
    vec3 box_min = box.xyz;
    vec3 box_max = box_min + box.w;
    vec3 tMin = (box_min - origin) / dir;
    vec3 tMax = (box_max - origin) / dir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float t_min = max(max(t1.x, t1.y), t1.z);
    float t_max = min(min(t2.x, t2.y), t2.z);
    return vec2(t_min, t_max);
}

bool containsAABB(vec3 point, vec4 box) {
    vec3 min = box.xyz;
    vec3 max = min + box.w;

    vec3 s = step(min, point) - step(max, point);
    bvec3 bs = bvec3(s);
    return all(bs);
}

uint material_at_position(inout vec4 box, vec3 position) {
    uint node_index = 0; // Assume root node

    while(true) {
        // start
        // Calculate new box location
        box.w = box.w / 2;
        vec3 box_midpoint = box.xyz + box.w;
        vec3 s = step(box_midpoint, position);
        box.xyz = box.xyz + s * box.w;


        uint child_index = uint(dot(s, vec3(4,2,1)));
        uint freemask = uint(nodes[node_index].freemask);
        if ((freemask & (1 << child_index)) == 0) {
            // is a leaf node
            return uint(nodes[node_index].data[child_index]);
        } else {
            // has children
            uint child_offset = mask_location_nth_one(freemask, child_index);
            node_index = nodes[node_index].children + child_offset;
        }
    }
}


uint RayMarch(vec4 initial_box, Ray ray, out vec3 hitpoint, out vec4 hitbox, out uint iteration_times) {
    hitbox = initial_box;
    vec2 intersection = intersectAABB(ray.origin, ray.dir, hitbox);
    vec3 entry_point = ray.origin + max(0, intersection.x) * ray.dir;
    vec3 test_point = entry_point + ray.dir * hitbox.w * 0.000001;
    uint material_id = 0;

    uint counter;
    for(
    counter = 0;
    counter < MAX_ITERATION_VALUE && containsAABB(test_point, bounding_box);
    counter++) {
        hitbox = initial_box;
        material_id = material_at_position(hitbox, test_point);
        if (material_id > 0) {
            // Hit some materials
            break;
        }
        // calculate the next t_min
        vec2 new_intersection = intersectAABB(entry_point, ray.dir, hitbox);

        entry_point = entry_point + ray.dir * new_intersection.y;
        test_point = entry_point + sign(ray.dir) * hitbox.w * 0.0001;
    }
    hitpoint = entry_point;
    iteration_times = counter;
    return material_id;
}

void main() {
    f_color = vec4(0.0, 0.0, 1.0, 1.0);
/*

    vec4 output_color;
    float scale;
    if (voxel_id == 0) {
        discard;
        //output_color = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        output_color = vec4(1.0, 0.5, 1.0, 1.0);
        // colored
        //uint material_id = (voxel_id >> 8) & 0x7f;
        //uint color = voxel_id & 0xff;
        //output_color = coloredMaterials[material_id].palette[color];
    }
    f_color = output_color;
*/
}
