

#ifdef SHADER_INT_64

#define GridType uint64_t[8]
bool GridCheck(GridType mask, uint32_t hit) {
    uint byteIndex = hit / 64;
    uint bitIndex = hit - byteIndex * 64;
    return (mask[byteIndex] & (1 << bitIndex)) == 0;
}
uint32_t GridCountOnesBefore(GridType grid, uint32_t hit) {
    return 0; // TODO
}

#else

#define GridType uint32_t[16]
bool GridCheck(GridType mask, uint32_t hit) {
    uint byteIndex = hit / 32;
    uint bitIndex = hit - byteIndex * 32;
    return (mask[byteIndex] & (1 << bitIndex)) == 0;
}
uint32_t GridCountOnesBefore(GridType grid, uint32_t hit) {
    uint byteIndex = hit / 32;
    uint bitIndex = hit - byteIndex * 32;
    uint32_t mask = (1 << bitIndex) - 1;

    uint sum = 0;
    uint i;
    for (i = 0; i < 15; i++) {
        if (i == byteIndex) {
            break;
        }
        sum += bitCount(grid[i]);
    }
    sum += bitCount(grid[byteIndex] & mask);
    return sum;
}
#endif

struct Block
{
    #ifdef SHADER_INT_64
    uint64_t mask[8];
    #else
    uint32_t mask[16];
    #endif
    vec3 min;
    vec3 max;
    uint32_t material_ptr;
    uint32_t reserved;
};


layout(buffer_reference, buffer_reference_align = 8, scalar) buffer GeometryInfo {
    Block blocks[];
};
layout(buffer_reference, buffer_reference_align = 1, scalar) buffer MaterialInfo {
    uint8_t materials[];
};
layout(buffer_reference) buffer PaletteInfo {
    u8vec4 palette[];
};

layout(shaderRecordEXT) buffer Sbt {
    GeometryInfo geometryInfo;
    MaterialInfo materialInfo;
    PaletteInfo paletteInfo;
} sbt;