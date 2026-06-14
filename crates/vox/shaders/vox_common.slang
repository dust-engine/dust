module vox_common;

public struct Block
{
    public uint64_t mask;
    private uint32_t coords_packed;
    public uint32_t material_ptr;

    public property uint32_t3 coords {
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
    public uint8_t normal;
    [mutating]
    public void setNormal(int8_t3 n) {
        int8_t axis = (n.y & 1) * 2 + (n.z & 1) * 4;
        int8_t sign = ((n.x | n.y | n.z) >> 7) & 1;
        this.normal = axis + sign;
    }

    public int8_t3 getNormal() {
        uint8_t axis = this.normal >> 1;               // 0, 1, or 2
        int8_t val = 1 - ((this.normal & 1) << 1); // sign bit: 0 -> +1, 1 -> -1

        int8_t3 d;
        d.x = (axis == 0) ? val : int8_t(0);
        d.y = (axis == 1) ? val : int8_t(0);
        d.z = (axis == 2) ? val : int8_t(0);
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

public struct VoxMaterial {
    // Packed bytes, low → high: type, roughness, metalness, specular.
    private uint32_t packed;
    // sRGB color, packed RGBA8 (little-endian).
    private uint32_t _albedo;
    public half emission;
    public half ior;
    public half transparency;
    public half _padding;

    // 0 = diffuse, 1 = metal, 2 = glass, 3 = emit, 4 = media/cloud.
    public uint8_t materialType() { return uint8_t(packed & 0xFFu); }
    public float roughness() { return float((packed >> 8) & 0xFFu) / 255.0; }
    public float metalness() { return float((packed >> 16) & 0xFFu) / 255.0; }
    public float specular() { return float((packed >> 24) & 0xFFu) / 255.0; }

    // Decoded sRGB albedo (the palette color) as 0..1 RGBA
    public float4 albedo() {
        uint8_t4 c = bit_cast<uint8_t4>(_albedo);
        return float4(c) / 255.0;
    }

    // Default opaque diffuse (black), matching the Rust `VoxGpuMaterial::DEFAULT`.
    public static VoxMaterial diffuse() {
        VoxMaterial m;
        m.packed = 0u;
        m._albedo = 0u;
        m.emission = half(0.0);
        m.ior = half(1.0);
        m.transparency = half(0.0);
        m._padding = half(0.0);
        return m;
    }
}

public struct VoxModelParams {
    private Block *geometry_info;
    private uint32_t *material_info;
    // Merged 256-entry palette table (color + material), addressed by palette
    // index. Mirrors the Rust `VoxModelParams::palette`.
    private VoxMaterial *palette;
    public float unit_size;
    private uint32_t _pad;

    public Block getCurrentPrimitive() {
        return geometry_info[PrimitiveIndex()];
    }

    public VoxMaterial getMaterialForVoxel(uint8_t voxelId) {
        if (!material_info || !palette) {
            return VoxMaterial::diffuse();
        }
        Block currentPrimitive = getCurrentPrimitive();

        uint64_t masked = currentPrimitive.mask & ((uint64_t(1) << voxelId) - 1);
        uint32_t voxelMemoryOffset = countbits(masked);
        uint32_t paletteIndexPacked = material_info[(voxelMemoryOffset + currentPrimitive.material_ptr) / 4];
        uint8_t paletteIndex = (uint8_t)(paletteIndexPacked >> (((voxelMemoryOffset + currentPrimitive.material_ptr) & 3) * 8));

        // min(x, 255) is a NVIDIA workaround.
        return palette[min(paletteIndex, 255)];
    }

    // Discrete, jitter-invariant identity of the hit voxel face, for building a
    // SHARC cell key (see SharcParams::cellKey). All three outputs depend only
    // on *which* voxel/face the DDA resolved — never on the exact ray hit
    // point — so sub-pixel camera jitter can't perturb them.
    //   - `voxelCoord`: integer object-space voxel coordinate (block min corner
    //      + in-block offset).
    //   - `faceId`: 0..5, axis*2 + sign — distinguishes all six faces (SHARC's
    //      3-bit normal octant alone can't separate same-sign axes).
    //   - `cellCenterWorld`: world-space center of the voxel; used only to pick
    //      the distance LOD. Using the voxel center (not the hit face) keeps all
    //      six faces of a voxel on the same LOD, so they coarsen together rather
    //      than splitting across a level boundary. Must be called from a hit
    //      shader.
    public VoxModelCell voxelCellInfo(uint8_t voxelId) {
        Block block = getCurrentPrimitive();
        uint32_t3 offsetInBox = uint32_t3(voxelId >> 4, (voxelId >> 2) & 0b11, voxelId & 0b11);
        VoxModelCell cell;
        cell.coords = block.coords + offsetInBox;
        cell.unit_size = unit_size;
        return cell;
    }
}

public struct VoxModelCell {
    public uint32_t3 coords;
    public float unit_size;
    public float3 centerObjectSpace() {
        return (float3(coords) + 0.5) * unit_size;
    }
    public float3 centerWorldSpace() {
        float3x4 o2w = ObjectToWorld3x4();
        return mul(o2w, float4(centerObjectSpace(), 1.0));
    }
}
