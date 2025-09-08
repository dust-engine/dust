struct VdbLeaf
{
    uint mask1;
    uint mask2;
    uint coords;
    uint material_ptr;
}

struct VkAabbPositionsKHR
{
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
}

[shader("compute")]
[numthreads(32)]
void main(
    uint threadID : SV_DispatchThreadID,
    StructuredBuffer<VdbLeaf> src,
    RWStructuredBuffer<VkAabbPositionsKHR> dst
)
{
    uint coords = src[threadID].coords;

    VkAabbPositionsKHR aabbPositions;
    if (coords == !0) {
        aabbPositions.minX = 0.0 / 0.0;
        aabbPositions.minY = 0.0;
        aabbPositions.minZ = 0.0;
        aabbPositions.maxX = 0.0;
        aabbPositions.maxY = 0.0;
        aabbPositions.maxZ = 0.0;
    } else {
        uint3 actualCoords = uint3(
            coords >> 20,
            coords >> 10,
            coords
        ) & ((1 << 10) - 1);
        
        aabbPositions.minX = actualCoords.x;
        aabbPositions.minY = actualCoords.y;
        aabbPositions.minZ = actualCoords.z;
        aabbPositions.maxX = actualCoords.x + 1;
        aabbPositions.maxY = actualCoords.y + 1;
        aabbPositions.maxZ = actualCoords.z + 1;
    }
    dst[threadID] = aabbPositions;
}
