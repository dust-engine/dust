module sharc;

// SHARC SDK plumbing. Mirrors the prefix block in sharc_pt.slang so the cache
// types (HashGridKey, SharcAccumulationData, SharcPackedData, SharcParameters,
// SharcHitData, …) resolve when this module is compiled. Headers come from
// @sharc//:sharc_headers — see crates/pbr/shaders/BUILD.bazel.
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#define HASH_GRID_COMPACT 0
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 0
// Our keys are discrete voxel identities, not a spatial grid, so the
// camera-motion LOD reprojection in SharcResolveEntry (which reinterprets key
// bits as grid position/level) is meaningless and must be off. `cellKey` does
// its own discrete distance coarsening instead.
#define SHARC_BLEND_ADJACENT_LEVELS 0
#define RW_STRUCTURED_BUFFER(name, type) RWStructuredBuffer<type> name
#define BUFFER_AT_OFFSET(name, offset) name[offset]
#include "SharcCommon.h"

// Public mirror of `SharcConstantsRaw` in crates/pbr/src/sharc.rs and of the
// hand-rolled `SharcShaderConstants` in sharc_pt.slang. We re-declare it here
// (as opposed to forwarding) so PbrPipelineParams can carry a public
// ConstantBuffer<SharcShaderConstants> field — `#include`d struct types are
// module-private in Slang and can't appear in public APIs.
public struct SharcShaderConstants {
    public float3 cameraPosition;
    public float sceneScale;
    public float3 cameraPositionPrev;
    public float radianceScale;
    public float roughnessMin;
    public uint accumulationFrameNum;
    public uint staleFrameNum;
    public uint entriesNum;
    public uint frameIndex;
    public uint downscaleFactor;
    public uint poolCapacity;
    private uint _debugMode;
    public float debugBrightness;
    public property SharcDebugMode debugMode {
        get {
            return SharcDebugMode(_debugMode);
        }
    }
}
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
/// hash map, decoupled from position. `albedo` rides along so the Update
/// raygen doesn't fire a probe ray to recover it from the voxel palette.
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

public struct SharcParams {

    // SHARC working storage. Private because the SDK types are #include'd
    // from C headers and therefore module-private — making the fields public
    // would force their types to be public too, which Slang rejects. The
    // public accessor methods further down expose what callers actually need
    // (query, debug overlays) via public input/output types only.
    //
    // Slang auto-assigns these to set 0 bindings 14..16 once any accessor is
    // referenced; the constants UBO lands at binding 17. Bindings 18..21 are
    // the ping-ponged candidate pool. All match pbr.playout.ron and
    // sharc.playout.ron exactly. Importers of this module (e.g. sharc_pt.slang)
    // do NOT see these private fields, so their own standalone
    // `[vk::binding(14..17, 0)]` globals don't collide.
    RWStructuredBuffer<HashGridKey> g_sharcHashEntries;
    RWStructuredBuffer<SharcAccumulationData> g_sharcAccumulation;
    RWStructuredBuffer<SharcPackedData> g_sharcResolved;
    public ConstantBuffer<SharcShaderConstants> g_sharcConstants;
    // Candidate pool — read side feeds Update's primary vertices, write side
    // accumulates this frame's final_gather CHS misses and Update cascade
    // pushes for *next* frame to consume.
    public RWStructuredBuffer<Candidate> g_sharcCandidatesRead;
    public RWStructuredBuffer<uint> g_sharcKeysRead;
    RWStructuredBuffer<Candidate> g_sharcCandidatesWrite;
    RWStructuredBuffer<uint> g_sharcKeysWrite;

    // Private helpers used by the public accessor methods below.
    SharcParameters buildSharcParameters() {
        SharcParameters p;
        p.hashGridParameters.cameraPosition = g_sharcConstants.cameraPosition;
        p.hashGridParameters.sceneScale = g_sharcConstants.sceneScale;
        p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        p.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
        p.hashGridData.capacity = g_sharcConstants.entriesNum;
        p.hashGridData.hashEntriesBuffer = g_sharcHashEntries;
        p.accumulationBuffer = g_sharcAccumulation;
        p.resolvedBuffer = g_sharcResolved;
        p.radianceScale = g_sharcConstants.radianceScale;
        return p;
    }

    // ── Key-based SHARC API (voxel-native) ────────────────────────────────
    // SHARC's spatial hash exists to quantize a *continuous* surface position
    // into a discrete cell. Voxels are already discrete, so that round-trip
    // only adds floating-point boundary flicker (a hit on a voxel face sits on
    // a cell boundary; sub-pixel camera jitter flips `floor()` to a neighbour).
    // Instead the caller hands us a precomputed cell key built from stable
    // integer identity and we address the hash map with it directly.

    // Pack a cache-cell key from discrete, jitter-invariant inputs.
    // `voxelCoord` is the integer (object-space) voxel coordinate, `faceId` in
    // [0,5], `instanceId` from InstanceIndex(). `cellCenterWorld` is used ONLY
    // to pick a distance LOD, so it never perturbs the key under jitter.
    //
    // Distance coarsening (discrete, jitter-free LOD): within `sceneScale`
    // world units of the camera the key is full-resolution (lod 0); every
    // doubling of distance past that right-shifts the coordinate one more bit,
    // merging 2^lod voxels per axis so the live-entry count stays bounded —
    // the discrete replacement for SHARC's disabled adjacent-level blend.
    //
    // 64-bit layout — the top bit is forced to 1 so the key can never equal the
    // HASH_GRID_INVALID_HASH_KEY sentinel (0):
    //   [0..3) faceId | [3..7) lod | [7..23) x | [23..39) y | [39..55) z
    //   | [55..63) instanceId | [63] = 1
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
        key |= uint64_t(1) << 63;
        return key;
    }

    // Cache query by precomputed key. Returns true with the resolved radiance
    // when the cell has a usable entry; always reports the entry's accumulated
    // sample count (0 on miss) so callers can weight a re-seed by 1/(1+n).
    public bool getCachedRadianceByKey(uint64_t key, out float3 radiance,
                                       out float sampleNum) {
        SharcParameters p = buildSharcParameters();
        radiance = float3(0.0, 0.0, 0.0);
        sampleNum = 0.0;
        uint baseSlot = HashGridGetBaseSlot(key, p.hashGridData.capacity);
        HashGridIndex idx = HASH_GRID_INVALID_CACHE_INDEX;
        uint bucketOffset;
        if (!HashGridFind(p.hashGridData, key, baseSlot,
                          HASH_GRID_HASH_MAP_BUCKET_SIZE, idx, bucketOffset))
            return false;
        SharcVoxelData voxelData = SharcGetVoxelData(p.resolvedBuffer, idx);
        sampleNum = voxelData.accumulatedSampleNum;
        if (sampleNum > SHARC_SAMPLE_NUM_THRESHOLD) {
            // SH encoding is off, so SharcDecodeRadiance ignores the direction.
            radiance = SharcDecodeRadiance(voxelData.accumulatedRadiance,
                                           float3(0.0, 0.0, 1.0));
            return true;
        }
        return false;
    }

    // Open this candidate's cache cell for the frame.
    // Returns true if the cache cell was successfully opened;
    // Returns false if the bucket was full and the cache cell cannot be opened.
    public bool openCellForUpdate(uint64_t key, out uint slot) {
        SharcParameters p = buildSharcParameters();
        slot = HASH_GRID_INVALID_CACHE_INDEX;
        uint baseSlot = HashGridGetBaseSlot(key, p.hashGridData.capacity);
        uint bucketOffset;
        if (!HashGridInsert(p.hashGridData, key, baseSlot,
                            HASH_GRID_HASH_MAP_BUCKET_SIZE, slot, bucketOffset))
            return false;
        SharcAddVoxelData(p, slot, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0),
                          float3(0.0, 0.0, 1.0), 0.0, 1);
        return true;
    }

    // Fold a lighting contribution into a cell already opened by
    // `openCellForUpdate`, scaled by the path throughput. sampleData 0 — the
    // sample was already counted when the cell was opened, so this only adds
    // radiance. No hash lookup: `slot` was resolved once in the ray-gen and
    // carried here via the payload. No-ops on an invalid slot (bucket was full).
    public void addRadianceToSlot(uint slot, float3 radiance, float3 throughput) {
        if (slot == HASH_GRID_INVALID_CACHE_INDEX) return;
        SharcParameters p = buildSharcParameters();
        SharcAddVoxelData(p, slot, radiance, throughput, float3(0.0, 0.0, 1.0),
                          0.0, 0);
    }

    // ── Debug overlays ────────────────────────────────────────────────────
    // Stable per-cell color from the key hash (replaces the spatial
    // HashGridDebugColoredHash — same color-from-hash, fed our voxel key).
    public float3 debugColoredHashByKey(uint64_t key) {
        return HashGridGetColorFromHash32(HashGridHash32(key));
    }

    // Linear-probe bucket offset for `key` (blue = no collision … red = heavy
    // probing). Mirrors HashGridDebugHashCollisions but key-based.
    public float3 debugHashCollisionsByKey(uint64_t key) {
        SharcParameters p = buildSharcParameters();
        uint baseSlot = HashGridGetBaseSlot(key, p.hashGridData.capacity);
        HashGridIndex idx = HASH_GRID_INVALID_CACHE_INDEX;
        uint bucketOffset = 0;
        HashGridFind(p.hashGridData, key, baseSlot,
                     HASH_GRID_HASH_MAP_BUCKET_SIZE, idx, bucketOffset);
        if (bucketOffset == 0) return float3(0.0, 0.0, 1.0);
        else if (bucketOffset == 1) return float3(0.0, 0.5, 0.5);
        else if (bucketOffset == 2) return float3(0.0, 1.0, 0.0);
        else if (bucketOffset == 3) return float3(1.0, 1.0, 0.0);
        else if (bucketOffset == 4) return float3(0.75, 0.25, 0.0);
        else return float3(1.0, 0.0, 0.0);
    }

    // Table-occupancy view — screen-space, key-agnostic, so it stays as-is.
    public float3 debugOccupancy(uint2 pixel, uint2 screenSize) {
        HashGridData d;
        d.capacity = g_sharcConstants.entriesNum;
        d.hashEntriesBuffer = g_sharcHashEntries;
        return HashGridDebugOccupancy(pixel, screenSize, d);
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
