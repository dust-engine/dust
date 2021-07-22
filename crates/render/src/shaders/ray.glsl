// struct Node {
//     uint8_t _padding; 0
//     uint8_t occupancy; 1
//     uint16_t sizemask; 2, 3
//     uint children;
//     // uint16_t data[8];
// };
struct RawNode {
    uint8_t raw[8]; // Also extended_occupancy[8]
};

struct Ray {
    vec3 origin;
    vec3 dir;
};
struct Box {
    vec3 origin;
    float extent;
};
