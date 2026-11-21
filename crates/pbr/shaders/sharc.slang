module sharc;

// The radiance cache, after NVIDIA's SHARC: outgoing radiance per voxel face,
// which the Update pass (sharc/sharc_update.slang) samples, the Resolve pass
// (sharc/sharc_resolve.slang) averages over frames, and the final gather reads.
//
// A `HashMap` maps each cell's key (`SharcParams::cellKey`) to a slot. The cell's
// state is at that slot in three buffers: `SharcAccumulation`, this frame's
// samples; `SharcResolvedHot`, their average over recent frames, which every
// cache query reads; and `SharcResolvedCold`, the bookkeeping only Resolve reads.
import hash_map;
__exported import emitters;

// Bounds on SharcShaderConstants::accumulationFrameNum and ::staleFrameNum.
public static const uint kSharcAccumulationFrameNumMin = 1;
public static const uint kSharcAccumulationFrameNumMax = 1024;
public static const uint kSharcStaleFrameNumMin = 8;
public static const uint kSharcStaleFrameNumMax = 1024;

// The slot of a cell that couldn't be opened (`openCellForUpdate`).
public static const uint kSharcInvalidSlot = 0xFFFFFFFFu;

// A cell's samples this frame: their radiance, summed with atomic adds, and how
// many there are. Resolve folds them into the cell's `SharcResolvedHot` and
// zeroes this. std430 stride 16 B (`SharcResources::STRIDE_ACCUMULATION`).
public struct SharcAccumulation {
    public float3 radiance;
    public uint sampleNum;
}

// A cell's radiance averaged over recent frames: what every cache query reads
// (`getCachedRadianceByKey`), kept apart from `SharcResolvedCold` so the final
// gather's random lookups fetch 8 B per cell. std430 stride 8 B
// (`SharcResources::STRIDE_RESOLVED_HOT`).
public struct SharcResolvedHot {
    // The average radiance and the (frame-weighted) number of samples in it.
    float16_t4 radianceAndSampleNum;

    public __init(float3 radiance, float sampleNum) {
        const float kFloat16Max = 65504.0;
        radianceAndSampleNum = float16_t4(min(float4(radiance, sampleNum), kFloat16Max));
    }

    public property float3 radiance {
        get { return float3(radianceAndSampleNum.xyz); }
    }
    public property float sampleNum {
        get { return float(radianceAndSampleNum.w); }
    }
}

// A cell's bookkeeping, which only Resolve reads. std430 stride 4 B
// (`SharcResources::STRIDE_RESOLVED_COLD`).
public struct SharcResolvedCold {
    // Frames accumulated in the low 16 bits; frames since the last sample in the
    // high 16. Both stay under 2^16: a cell is evicted after at most
    // kSharcStaleFrameNumMax frames without a sample, and every sample caps the
    // count at the accumulation window.
    uint frameNums;

    public __init(uint accumulatedFrameNum, uint staleFrameNum) {
        frameNums = accumulatedFrameNum | (staleFrameNum << 16);
    }

    public property uint accumulatedFrameNum {
        get { return frameNums & 0xFFFFu; }
    }
    public property uint staleFrameNum {
        get { return frameNums >> 16; }
    }
}

// Public mirror of `SharcConstantsRaw` in crates/pbr/src/sharc.rs. It's declared
// in this module so PbrPipelineParams and the Resolve pass can both carry a
// ConstantBuffer<SharcShaderConstants>.
public struct SharcShaderConstants {
    public float3 cameraPosition;
    public float sceneScale;
    public float3 cameraPositionPrev;
    public float roughnessMin;
    public uint accumulationFrameNum;
    public uint staleFrameNum;
    public uint entriesNum;
    public uint frameIndex;
    public uint poolCapacity;
    private uint _debugMode;
    public float debugBrightness;
    // SHARC_FLAG_*.
    public uint flags;
    // This frame's instances of models with emitters. Only the Update gets
    // them; elsewhere the count is 0.
    public EmitterInstance *emitterInstances;
    public uint emitterInstanceCount;
    // The instances' total power in world units.
    public float emitterPower;
    // Candidates `sampleEmitter` draws to keep one of.
    public uint emitterCandidates;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    public property SharcDebugMode debugMode {
        get {
            return SharcDebugMode(_debugMode);
        }
    }
    // Whether the Update samples emissive faces directly. When it does, a
    // bounce ray that hits one leaves out its emission, which the shadow ray
    // to an emitter already brings in.
    public property bool samplesEmitters {
        get {
            return (flags & SHARC_FLAG_SAMPLE_EMITTERS) != 0 && emitterInstanceCount > 0;
        }
    }
}

// SharcShaderConstants::flags, set from SharcConfig.
public static const uint SHARC_FLAG_SAMPLE_EMITTERS = 1;
// Resolve sums `SharcStats` for SharcConfig::log_stats.
public static const uint SHARC_FLAG_STATS = 2;
public enum SharcDebugMode
{
    None = 0,
    ColoredHash = 1,
    CachedRadiance = 2,
    HashCollisions = 3,
    HashOccupancy = 4,
    CandidatePool = 5,
}

/// Candidate-pool entry — one surface point the Update pass should populate
/// next frame.
///
/// `worldPos`/`normal` are the *raw* hit surface frame, kept for tracing the
/// Update rays (NEE + bounce) from this point — they don't address the cache,
/// so sub-pixel jitter in them is harmless. `key` is the precomputed cache
/// cell key (see `SharcParams::cellKey`); it's what actually addresses the
/// hash map, decoupled from position. `albedo` is linear, and rides along so the
/// Update raygen doesn't fire a probe ray to recover it from the voxel palette.
///
/// Layout matches `SharcResources::STRIDE_CANDIDATE` (std430 stride 64 B):
/// three `float3` at 16 B each + a `uint64_t` key.
public struct Candidate {
    public float3 worldPos;
    public float3 normal;
    public float3 albedo;
    public uint64_t key;
}

// Local hash used by pushCandidate — keeps `sharc.slang` self-contained
// without an `import random;` dependency.
uint pcgHash(uint x) {
    x = x * 747796405u + 2891336453u;
    uint word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (word >> 22u) ^ word;
}

// A stable color per 32-bit hash, for debug views.
float3 colorFromHash(uint hash) {
    return float3(float((hash >> 0) & 0x3FFu) / 1023.0,
                  float((hash >> 11) & 0x7FFu) / 2047.0,
                  float((hash >> 22) & 0x3FFu) / 1023.0);
}

// Generic pumicite shader-stage flags for the `[Stages(...)]` attribute. The
// pumicite codegen reads this mask off every ParameterBlock binding and only
// coalesces consecutive descriptor writes that share it, so each emitted
// VkWriteDescriptorSet stays stage-uniform. Values mirror VkShaderStageFlagBits.
// Declared in this base module so importers (e.g. pbr_common) can annotate their
// own bindings via `__exported import sharc`.
public static const int STAGE_COMPUTE = 0x20;
public static const int STAGE_RAYGEN = 0x100;
public static const int STAGE_ANY_HIT = 0x200;
public static const int STAGE_CLOSEST_HIT = 0x400;
public static const int STAGE_MISS = 0x800;
public static const int STAGE_INTERSECTION = 0x1000;

[__AttributeUsage(_AttributeTargets.Var)]
public struct StagesAttribute { int stages; }

public struct SharcParams {
    // The hash map: each slot's key tag, 0 when empty (`HashMap`).
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    RWStructuredBuffer<uint> g_sharcHashEntries;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT)]
    RWStructuredBuffer<SharcAccumulation> g_sharcAccumulation;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    // Only the hot half: `SharcResolvedCold` is bound to Resolve alone.
    RWStructuredBuffer<SharcResolvedHot> g_sharcResolvedHot;
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public ConstantBuffer<SharcShaderConstants> g_sharcConstants;
    // Candidate pool — read side feeds Update's primary vertices, write side
    // accumulates this frame's final_gather CHS hits and Update cascade
    // pushes for *next* frame to consume.
    [Stages(STAGE_RAYGEN | STAGE_MISS | STAGE_CLOSEST_HIT | STAGE_COMPUTE)]
    public RWStructuredBuffer<Candidate> g_sharcCandidatesRead;
    [Stages(STAGE_RAYGEN | STAGE_COMPUTE)]
    public RWStructuredBuffer<uint> g_sharcKeysRead;
    [Stages(STAGE_RAYGEN | STAGE_CLOSEST_HIT)]
    RWStructuredBuffer<Candidate> g_sharcCandidatesWrite;
    [Stages(STAGE_RAYGEN | STAGE_CLOSEST_HIT)]
    RWStructuredBuffer<uint> g_sharcKeysWrite;

    property HashMap map {
        get {
            return HashMap(g_sharcHashEntries, g_sharcConstants.entriesNum);
        }
    }

    // ── Cell keys ─────────────────────────────────────────────────────────
    // SHARC hashes a *continuous* surface position into a grid cell. Voxels are
    // already discrete, and a hit on a voxel face sits on a cell boundary, where
    // sub-pixel camera jitter would flip `floor()` to a neighbour. So the caller
    // keys the cache by the voxel's integer identity instead.

    // Pack a cache-cell key from discrete, jitter-invariant inputs.
    // `voxelCoord` is the integer (object-space) voxel coordinate, `faceId` in
    // [0,5], `instanceId` from InstanceIndex(). `cellCenterWorld` is used ONLY
    // to pick a distance LOD, so it never perturbs the key under jitter.
    //
    // Distance coarsening (discrete, jitter-free LOD): within `sceneScale`
    // world units of the camera the key is full-resolution (lod 0); every
    // doubling of distance past that right-shifts the coordinate one more bit,
    // merging 2^lod voxels per axis so the live-entry count stays bounded.
    //
    // 64-bit layout:
    //   [0..3) faceId | [3..7) lod | [7..23) x | [23..39) y | [39..55) z
    //   | [55..63) instanceId
    public uint64_t cellKey(uint instanceId, int3 voxelCoord, uint faceId,
                            float3 cellCenterWorld) {
        float dist = distance(g_sharcConstants.cameraPosition, cellCenterWorld);
        float t = max(dist / max(g_sharcConstants.sceneScale, 1e-6), 1.0);
        uint lod = min(uint(log2(t)), 15u);
        uint3 c = uint3(voxelCoord) >> lod;

        uint64_t key = uint64_t(faceId & 0x7u);
        key |= uint64_t(lod & 0xFu) << 3;
        key |= (uint64_t(c.x) & 0xFFFFu) << 7;
        key |= (uint64_t(c.y) & 0xFFFFu) << 23;
        key |= (uint64_t(c.z) & 0xFFFFu) << 39;
        key |= (uint64_t(instanceId) & 0xFFu) << 55;
        return key;
    }

    // ── Queries and updates ───────────────────────────────────────────────

    // Cache query by precomputed key. Returns true with the resolved radiance
    // when the cell has samples; always reports the cell's accumulated sample
    // count (0 on miss) so callers can weight a re-seed by 1/(1+n).
    public bool getCachedRadianceByKey(uint64_t key, out float3 radiance,
                                       out float sampleNum) {
        radiance = float3(0.0, 0.0, 0.0);
        sampleNum = 0.0;
        if (let slot = map.find(key)) {
            SharcResolvedHot cell = g_sharcResolvedHot[slot];
            sampleNum = cell.sampleNum;
            if (sampleNum > 0.0) {
                radiance = cell.radiance;
                return true;
            }
        }
        return false;
    }

    // Open the cell for this frame — find it or insert it — and count one
    // sample in it. Returns false, with `kSharcInvalidSlot`, when the cell's
    // bucket is full.
    public bool openCellForUpdate(uint64_t key, out uint slot) {
        if (let opened = map.findOrInsert(key)) {
            slot = opened;
            InterlockedAdd(g_sharcAccumulation[slot].sampleNum, 1u);
            return true;
        }
        slot = kSharcInvalidSlot;
        return false;
    }

    // Fold a lighting contribution into a cell already opened by
    // `openCellForUpdate`, scaled by the path throughput. The sample was
    // counted when the cell was opened, so this only adds radiance. No hash
    // lookup: `slot` was resolved once in the ray-gen and carried here via the
    // payload. No-ops on `kSharcInvalidSlot`.
    public void addRadianceToSlot(uint slot, float3 radiance, float3 throughput) {
        if (slot == kSharcInvalidSlot) return;
        float3 sample = radiance * throughput;
        // Greater than 0, not different from it, to keep NaN out: in the sum it
        // would reach the resolved radiance and spread to every cell that
        // gathers from this one.
        if (sample.x > 0.0) InterlockedAdd(g_sharcAccumulation[slot].radiance.x, sample.x);
        if (sample.y > 0.0) InterlockedAdd(g_sharcAccumulation[slot].radiance.y, sample.y);
        if (sample.z > 0.0) InterlockedAdd(g_sharcAccumulation[slot].radiance.z, sample.z);
    }

    // ── Debug overlays ────────────────────────────────────────────────────

    // Stable per-cell color from the key's hash.
    public float3 debugColoredHashByKey(uint64_t key) {
        return colorFromHash(HashMap::hash(key));
    }

    // How far into its bucket the cell is stored (blue = at its base slot …
    // red = 5 or more slots in, or not in the map).
    public float3 debugHashCollisionsByKey(uint64_t key) {
        uint probe = map.probeLength(key);
        if (probe == 0) return float3(0.0, 0.0, 1.0);
        else if (probe == 1) return float3(0.0, 0.5, 0.5);
        else if (probe == 2) return float3(0.0, 1.0, 0.0);
        else if (probe == 3) return float3(1.0, 1.0, 0.0);
        else if (probe == 4) return float3(0.75, 0.25, 0.0);
        else return float3(1.0, 0.0, 0.0);
    }

    // The table itself, one 7×7-pixel block per slot: green when occupied,
    // black when empty. Columns of kBucketSize slots fill the screen top to
    // bottom, then left to right.
    public float3 debugOccupancy(uint2 pixel, uint2 screenSize) {
        const uint kElementSize = 7;
        const uint kBlockSize = kElementSize + 1;
        const uint kColumns = HashMap::kBucketSize;

        uint rowNum = screenSize.y / kBlockSize;
        uint2 block = pixel / kBlockSize;
        uint slot = (block.x / kColumns) * (rowNum * kColumns) + block.y * kColumns
            + (block.x % kColumns);
        bool inElement = all(pixel % kBlockSize < kElementSize);
        HashMap m = map;
        if (inElement && slot < m.capacity && m.isOccupied(slot)) {
            return float3(0.0, 1.0, 0.0);
        }
        return float3(0.0, 0.0, 0.0);
    }

    // ── Candidate-pool API ────────────────────────────────────────────────
    // Streaming weighted-reservoir-sampling, one independent reservoir of
    // size 1 per pool slot. Each call hashes `seed` to a slot and competes
    // for it with `r = U(0,1)^(1/weight)` via `InterlockedMax` on the slot's
    // u32 key. The candidate write is non-atomic and may race with another
    // winner; we accept the inconsistency (the slot still holds a valid push
    // from this frame, just possibly not the strictly-maximum-weight one).
    public void pushCandidate(float3 worldPos, float3 normal, float3 albedo,
                              uint64_t key, float weight, uint seed) {
        if (weight <= 0.0) return;
        uint cap = g_sharcConstants.poolCapacity;
        if (cap == 0) return;

        // Two independent hashes: one for slot, one for U.
        uint h0 = pcgHash(seed ^ 0x9e3779b9u);
        uint h1 = pcgHash(h0 ^ 0x517cc1b7u);

        uint slot = h0 % cap;
        // Avoid U == 0 (yields r = 0, ties with empty sentinel). Map h1 to
        // (0, 1] by using the upper 24 bits and adding 1 LSB.
        float u = (float((h1 >> 8) & 0x00FFFFFFu) + 1.0) * (1.0 / 16777217.0);
        // r = U^(1/w). Higher weight ⇒ r closer to 1 ⇒ wins atomic_max.
        float r = pow(u, 1.0 / weight);
        uint rUi = asuint(r);
        // Guard against r being exactly 0 in the bit pattern (denormals etc.).
        if (rUi == 0u) rUi = 1u;

        uint old;
        InterlockedMax(g_sharcKeysWrite[slot], rUi, old);
        if (rUi > old) {
            Candidate c;
            c.worldPos = worldPos;
            c.normal = normal;
            c.albedo = albedo;
            c.key = key;
            g_sharcCandidatesWrite[slot] = c;
        }
    }

    // Read side: Update raygen calls these to consume the pool. `keyAt`
    // returns 0 for empty slots — caller must gate.
    public uint keyAt(uint slot) {
        return g_sharcKeysRead[slot];
    }
    public Candidate candidateAt(uint slot) {
        return g_sharcCandidatesRead[slot];
    }
}
