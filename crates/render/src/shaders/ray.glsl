struct Node {
    uint16_t occupancy_freemask;
    uint16_t _padding2;
    uint children;
    uint8_t extended_occupancy[8];
    // uint16_t data[8];
};

struct Ray {
    vec3 origin;
    vec3 dir;
};
struct Box {
    vec3 origin;
    float extent;
};
