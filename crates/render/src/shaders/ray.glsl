struct Node {
    uint8_t freemask;
    uint8_t occupancy;
    uint16_t _padding2;
    uint children;
    uint8_t extended_occupancy[8];
    // uint16_t data[8];
};

struct RawNode {
    uint8_t raw_data[8];
};

struct Ray {
    vec3 origin;
    vec3 dir;
};
struct Box {
    vec3 origin;
    float extent;
};
