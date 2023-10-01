
struct SurfelEntry { 
    ivec3 position;
    uint32_t direction; // [0, 6) indicating one of the six faces of the cube
};
layout(constant_id = 1) const uint32_t SurfelPoolSize = 720*480;

layout(set = 0, binding = 13) buffer SurfelPool {
    SurfelEntry entries[];
} s_surfel_pool;

