// Closest-depth hierarchical Z buffer (HZB) build: the whole mip chain in one
// dispatch.
//
// Layout mirrors Unreal's HZB (`SceneTextureReductions.cpp`, `HZB.usf`): mip 0
// is half the power-of-two-rounded render extent and stores the closest
// (largest reverse-Z) depth of each 2x2 of depth texels; every further mip is
// the max of a 2x2 of its parent. A screen-space trace walking this chain can
// read the full-resolution depth G-buffer as "mip -1".
//
// Unreal issues one dispatch per four mips. Here the per-group reduction of
// `HZBBuildCS` (a tile folded through groupshared memory) is combined with the
// FidelityFX SPD last-workgroup election so the tail mips are produced by the
// last group to finish, with no barrier between dispatches:
//
//   Stage A, every group: a 64x64 mip-0 tile (128x128 depth texels) folded down
//   to mips 0..6 (mip 6 = one texel per group).
//   Stage B, the elected group only: reads the whole of mip 6 back and folds it
//   to mips 7..11.
//
// Coherence between the two stages follows SPD: mip 6 and the election counter
// are `globallycoherent`, every group issues a device-scope memory barrier
// before its atomic, and the elected group issues another before reading.

struct HzbConstants {
    uint2 renderExtent;   // depth G-buffer extent (texels)
    uint2 hzbExtent;      // mip 0 extent (texels)
    uint numMips;         // levels in the chain, at most kMaxMips
    uint numGroups;       // dispatch size, for the election
    float2 invRenderExtent;
};

static const uint kMaxMips = 12;
// Mip-0 texels per group tile, per axis. 256 threads x a 4x4 block each.
static const uint kTileSize = 64;

[vk::binding(0, 0)] Texture2D<float> depthTexture;
[vk::binding(1, 0)] RWTexture2D<float> hzbMips[kMaxMips];
// Mip 6 again, decorated coherent: written by every group in stage A and read
// back by the elected group in stage B. Loads and stores through this view
// bypass non-coherent caches (SPD's requirement for the cross-group hand-off).
[vk::binding(2, 0)] globallycoherent RWTexture2D<float> hzbMip6Coherent;
// Election counter. Incremented once per group; the group that observes
// `numGroups - 1` is the last to finish stage A and resets it for next frame.
[vk::binding(3, 0)] globallycoherent RWStructuredBuffer<uint> hzbCounter;
[vk::binding(4, 0)] ConstantBuffer<HzbConstants> cb;
// Clamp-to-border, transparent black: depth texels past the render extent
// (power-of-two padding) gather as 0, i.e. far.
[vk::binding(5, 0)] SamplerState depthSampler;

groupshared float sharedZ[256];
groupshared uint sharedElection;

uint2 mipExtent(uint level) {
    return max(cb.hzbExtent >> level, uint2(1, 1));
}

void storeMip(uint level, uint2 pos, float z) {
    if (level >= cb.numMips || any(pos >= mipExtent(level))) {
        return;
    }
    // Constant array indices only: dynamically uniform indexing of a storage
    // image array would need shaderStorageImageArrayDynamicIndexing.
    switch (level) {
    case 0: hzbMips[0][pos] = z; break;
    case 1: hzbMips[1][pos] = z; break;
    case 2: hzbMips[2][pos] = z; break;
    case 3: hzbMips[3][pos] = z; break;
    case 4: hzbMips[4][pos] = z; break;
    case 5: hzbMips[5][pos] = z; break;
    case 6: hzbMip6Coherent[pos] = z; break;
    case 7: hzbMips[7][pos] = z; break;
    case 8: hzbMips[8][pos] = z; break;
    case 9: hzbMips[9][pos] = z; break;
    case 10: hzbMips[10][pos] = z; break;
    default: hzbMips[11][pos] = z; break;
    }
}

uint signedShift(uint x, int shift) {
    if (shift > 0) {
        return x << uint(shift);
    } else if (shift < 0) {
        return x >> uint(-shift);
    }
    return x;
}

// Position of thread `id` in a 16x16 tile such that the threads
// `{t, t + n, t + 2n, t + 3n}` (n = the bank size of a reduction level) always
// own one 2x2 block of the level above. The low bits of x / y come from the
// high bits of `id`, so each fold halves the position and quarters the bank.
// Port of Unreal's `InitialTilePixelPositionForReduction2x2(4, id)`
// (ReductionCommon.ush).
uint2 tilePosition(uint id) {
    uint x = 0;
    uint y = 0;
    [ForceUnroll]
    for (uint i = 0; i < 4; i++) {
        const uint destBit = 3 - i;
        const uint destMask = 1u << destBit;
        x |= destMask & signedShift(id, int(destBit) - int(i * 2 + 0));
        y |= destMask & signedShift(id, int(destBit) - int(i * 2 + 1));
    }
    return uint2(x, y);
}

float max4(float a, float b, float c, float d) {
    return max(max(a, b), max(c, d));
}

// Closest depth of the 2x2 depth texels starting at `texel`, in one gather.
// UI pixels store exactly 1 (the near plane) and sky stores 0; both are "no
// surface" for the trace, so the sentinel maps to the far value. Out-of-range
// texels read 0 through the border sampler.
float closestDepth2x2(uint2 texel) {
    // The block's shared corner: the gather footprint is
    // floor(uv * size - 0.5) = texel, half a texel from any rounding boundary.
    float2 uv = float2(texel + 1) * cb.invRenderExtent;
    float4 z = depthTexture.GatherRed(depthSampler, uv);
    z = select(z >= 1.0, float4(0.0), z);
    return max4(z.x, z.y, z.z, z.w);
}

// Folds the 16x16 values in `sharedZ` (one per thread, `z` being this thread's)
// four times: level k (1..4) writes mip `firstMip + k - 1` at `pos >> k`.
// Port of the reduction loop in Unreal's `HZBBuildCS`.
void reduceTile(float z, uint gtid, uint2 pos, uint firstMip) {
    sharedZ[gtid] = z;
    [ForceUnroll]
    for (uint k = 1; k <= 4; k++) {
        const uint tileSize = 16u >> k;
        const uint bankSize = tileSize * tileSize;
        // Unreal skips this barrier once the bank fits in one wave; keep it
        // unconditionally, four barriers a group are noise next to the loads.
        GroupMemoryBarrierWithGroupSync();
        if (gtid < bankSize) {
            z = max4(z, sharedZ[gtid + bankSize], sharedZ[gtid + 2 * bankSize], sharedZ[gtid + 3 * bankSize]);
            pos >>= 1;
            storeMip(firstMip + k - 1, pos, z);
            sharedZ[gtid] = z;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupId: SV_GroupID, uint gtid: SV_GroupIndex) {
    uint2 tilePos = tilePosition(gtid);                    // 0..15 per axis
    uint2 blockOrigin = groupId.xy * kTileSize + tilePos * 4; // mip-0 texel

    // Stage A. Mip 0: max over 2x2 depth texels, a 4x4 block per thread.
    float m0[4][4];
    [ForceUnroll]
    for (uint j = 0; j < 4; j++) {
        [ForceUnroll]
        for (uint i = 0; i < 4; i++) {
            uint2 t = blockOrigin + uint2(i, j);
            float z = closestDepth2x2(t * 2);
            m0[j][i] = z;
            storeMip(0, t, z);
        }
    }
    // Mip 1: a 2x2 per thread, in registers.
    float m1[2][2];
    [ForceUnroll]
    for (uint j = 0; j < 2; j++) {
        [ForceUnroll]
        for (uint i = 0; i < 2; i++) {
            float z = max4(m0[2 * j][2 * i], m0[2 * j][2 * i + 1],
                           m0[2 * j + 1][2 * i], m0[2 * j + 1][2 * i + 1]);
            m1[j][i] = z;
            storeMip(1, blockOrigin / 2 + uint2(i, j), z);
        }
    }
    // Mip 2: one texel per thread, then the groupshared fold gives mips 3..6.
    float z = max4(m1[0][0], m1[0][1], m1[1][0], m1[1][1]);
    uint2 pos = groupId.xy * (kTileSize / 4) + tilePos;
    storeMip(2, pos, z);
    reduceTile(z, gtid, pos, 3);

    if (cb.numMips <= 7) {
        return;
    }

    // Stage B. Elect the last group to finish; it alone folds mip 6 further.
    // Release this group's mip-6 store device-wide before counting ourselves.
    DeviceMemoryBarrier();
    if (gtid == 0) {
        uint previous;
        InterlockedAdd(hzbCounter[0], 1, previous);
        sharedElection = previous;
    }
    GroupMemoryBarrierWithGroupSync();
    if (sharedElection != cb.numGroups - 1) {
        return;
    }
    // Acquire every other group's mip-6 store.
    DeviceMemoryBarrier();

    // Mip 7: a 2x2 of mip 6 per thread. Mip 6 is at most 32x32 (a 2048 mip 0).
    uint2 e6 = mipExtent(6);
    uint2 p = tilePos * 2;
    float z6[2][2];
    [ForceUnroll]
    for (uint j = 0; j < 2; j++) {
        [ForceUnroll]
        for (uint i = 0; i < 2; i++) {
            uint2 q = p + uint2(i, j);
            z6[j][i] = all(q < e6) ? hzbMip6Coherent[q] : 0.0;
        }
    }
    z = max4(z6[0][0], z6[0][1], z6[1][0], z6[1][1]);
    storeMip(7, tilePos, z);
    reduceTile(z, gtid, tilePos, 8);

    if (gtid == 0) {
        hzbCounter[0] = 0;
    }
}
