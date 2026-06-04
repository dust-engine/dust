module vox_common;

public struct Block
{
    public uint64_t mask;
    private uint32_t coords_packed;
    public uint32_t material_ptr;

    public property float3 coords {
        get {
            static const uint32_t mask = ((1 << 10) - 1);
            uint32_t3 min;
            min.x = coords_packed >> 20;
            min.y = (coords_packed >> 10) & mask;
            min.z = coords_packed & mask;
            return min << 2;
        }
    }
};

public struct HitAttribute {
    public uint8_t voxelId;
    private uint8_t normal;
    [mutating]
    public void setNormal(int8_t3 n) {
        int8_t axis = (n.y & 1) * 2 + (n.z & 1) * 4;
        int8_t sign = ((n.x | n.y | n.z) >> 7) & 1;
        this.normal = axis + sign;
    }

    public int8_t3 getNormal() {
        int axis = this.normal >> 1;            // 0, 1, or 2
        int val = 1 - ((this.normal & 1) << 1); // sign bit: 0 -> +1, 1 -> -1

        int8_t3 d;
        d.x = axis == 0 ? val : 0;
        d.y = axis == 1 ? val : 0;
        d.z = axis == 2 ? val : 0;
        return d;
    }
}

// Usually one would expect the `hitAttributeEXT` value in the closest hit shader to be the one set
// during the intersection shader invocation of the closest hit point. This is the case on NVIDIA GPUs.
// However, on my AMD GPU (6600XT), the `hitAttributeEXT` value seems to reflect the value set during
// the most recent intersection shader invocation instead.
// Meaning that if additional intersection tests were performed after the closest hit, wrong values
// would be reflected in the closest hit shader.
// Here, for NVIDIA GPUs we always set the voxelId. On AMD GPUs we manually make the comparison with gl_RayTmaxEXT.
public __generic<A>
[ForceInline]
[require(glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHitWithAMDWorkaround(float tHit, uint hitKind, A attributes)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "ReportHit($0, $1, $2)";
    case glsl:
    case spirv:
        if (tHit <= RayTCurrent()) {
            [__vulkanHitAttributes]
            static A a;
            a = attributes;
        }
        return __reportIntersection(tHit, hitKind);
    }
}

public struct VoxModelParams {
    private Block *geometry_info;
    private uint32_t *material_info;
    private uint32_t *palette_info;
    public float unit_size;
    private uint32_t _pad;

    public Block getCurrentPrimitive() {
        return geometry_info[PrimitiveIndex()];
    }

    public float4 getColorForVoxel(uint8_t voxelId) {
        if (!material_info || !palette_info) {
            return float4(1.0, 0.0, 1.0, 1.0);
        }
        Block currentPrimitive = getCurrentPrimitive();
        float3 offsetInBox = float3(voxelId >> 4, (voxelId >> 2) & 0b11, voxelId & 0b11);

        uint64_t masked = currentPrimitive.mask & ((uint64_t(1) << voxelId) - 1);
        uint32_t voxelMemoryOffset = countbits(masked);
        uint32_t paletteIndexPacked = material_info[(voxelMemoryOffset + currentPrimitive.material_ptr) / 4];
        uint8_t paletteIndex = (uint8_t)(paletteIndexPacked >> (((voxelMemoryOffset + currentPrimitive.material_ptr) & 3) * 8));

        // min(x, 255) is a NVIDIA workaround.
        uint8_t4 palette_color = bit_cast<uint8_t4>(palette_info[min(paletteIndex, 255)]);
        return float4(palette_color) / 255.0;
    }
}
