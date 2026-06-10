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

/// Candidate-pool entry — one surface point whose cache cell was a miss
/// this frame and should be populated by Update on the next frame.
///
/// `albedo` is included so the Update raygen doesn't have to fire a probe
/// ray to recover it from the voxel palette: both producers (the
/// final_gather CHS where the candidate is born on cache miss, and the
/// Update cascade after a bounce) already have it via vox CHS, so paying
/// 12 B in the buffer is cheaper than 1 ray per pool slot.
///
/// Layout matches `SharcResources::STRIDE_CANDIDATE` (std430 stride 48 B):
/// each `float3` aligns to 16 with a 4 B trailing pad per field.
public struct Candidate {
    public float3 worldPos;
    public float3 normal;
    public float3 albedo;
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

    HashGridParameters buildHashGridParameters() {
        HashGridParameters g;
        g.cameraPosition = g_sharcConstants.cameraPosition;
        g.sceneScale = g_sharcConstants.sceneScale;
        g.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        g.levelBias = SHARC_GRID_LEVEL_BIAS;
        return g;
    }

    // ── Public SHARC API ──────────────────────────────────────────────────
    // Cache query at a surface point. Returns true if the cache has a usable
    // resolved entry there. `radiance` is left untouched on miss; gate on the
    // return.
    public bool getCachedRadiance(float3 worldPos, float3 normal, out float3 radiance) {
        float sampleNum;
        return getCachedRadiance(worldPos, normal, radiance, sampleNum);
    }

    // Cache query that also reports the entry's accumulated sample count.
    // `sampleNum` is the number of samples already folded into this cell's
    // resolved value — 0 on a hard miss (no entry) and for an entry that
    // hasn't crossed SHARC_SAMPLE_NUM_THRESHOLD yet. The bool return still
    // gates the usable-radiance case exactly like the 3-arg overload.
    //
    // Reimplements `SharcGetCachedRadiance` rather than calling it because the
    // SDK function hides `accumulatedSampleNum`. The optional SDK paths
    // (responsive lighting, material demodulation, separate emissive, SH
    // encoding) are all compiled out in this configuration, so the find +
    // unpack + decode below is behaviorally identical to the SDK for the
    // usable case — it just additionally surfaces the sample count, which
    // callers use to weight a candidate push by 1/(1+sampleNum) and keep the
    // cell continuously refreshed instead of only re-seeding it on a miss.
    public bool getCachedRadiance(float3 worldPos, float3 normal,
                                  out float3 radiance, out float sampleNum) {
        SharcParameters p = buildSharcParameters();
        radiance = float3(0.0, 0.0, 0.0);
        sampleNum = 0.0;

        HashGridKey hashGridKey;
        HashGridIndex hashGridIndex = HashGridFindEntry(
            p.hashGridData, worldPos, normal, p.hashGridParameters, hashGridKey);
        if (hashGridIndex == HASH_GRID_INVALID_CACHE_INDEX)
            return false;

        SharcVoxelData voxelData = SharcGetVoxelData(p.resolvedBuffer, hashGridIndex);
        sampleNum = voxelData.accumulatedSampleNum;
        if (sampleNum > SHARC_SAMPLE_NUM_THRESHOLD) {
            // SH encoding is off, so SharcDecodeRadiance ignores the direction.
            radiance = SharcDecodeRadiance(voxelData.accumulatedRadiance,
                                           float3(0.0, 0.0, 1.0));
            return true;
        }
        return false;
    }

    // Hash-grid debug overlays. Useful for verifying scene-scale alignment,
    // occupancy, and collision rate. Cheap — call them in place of an
    // indirect bounce in a debug pixel shader.
    public float3 debugColoredHash(float3 worldPos, float3 normal) {
        return HashGridDebugColoredHash(worldPos, normal, buildHashGridParameters());
    }
    public float3 debugOccupancy(uint2 pixel, uint2 screenSize) {
        HashGridData d;
        d.capacity = g_sharcConstants.entriesNum;
        d.hashEntriesBuffer = g_sharcHashEntries;
        return HashGridDebugOccupancy(pixel, screenSize, d);
    }

    public float3 debugHashCollisions(float3 worldPos, float3 normal) {
        SharcParameters p = buildSharcParameters();
        return HashGridDebugHashCollisions(worldPos, normal, p.hashGridParameters, p.hashGridData);
    }

    // ── Candidate-pool API ────────────────────────────────────────────────
    // Streaming weighted-reservoir-sampling, one independent reservoir of
    // size 1 per pool slot. Each call hashes `seed` to a slot and competes
    // for it with `r = U(0,1)^(1/weight)` via `InterlockedMax` on the slot's
    // u32 key. The candidate write is non-atomic and may race with another
    // winner; we accept the inconsistency (the slot still holds a valid push
    // from this frame, just possibly not the strictly-maximum-weight one).
    public void pushCandidate(float3 worldPos, float3 normal, float3 albedo,
                              float weight, uint seed) {
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
