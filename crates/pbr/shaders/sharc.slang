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
    public uint debugMode;
    public uint _pad;
}

public struct SharcParams {

    // SHARC working storage. Private because the SDK types are #include'd
    // from C headers and therefore module-private — making the fields public
    // would force their types to be public too, which Slang rejects. The
    // public accessor methods further down expose what callers actually need
    // (query, debug overlays) via public input/output types only.
    //
    // Slang auto-assigns these to set 0 bindings 14..16 once any accessor is
    // referenced; the constants UBO lands at binding 17. The bindings match
    // pbr.playout.ron and sharc.playout.ron exactly. Importers of this module
    // (e.g. sharc_pt.slang) do NOT see these private fields, so their own
    // standalone `[vk::binding(14..17, 0)]` globals don't collide.
    RWStructuredBuffer<HashGridKey> g_sharcHashEntries;
    RWStructuredBuffer<SharcAccumulationData> g_sharcAccumulation;
    RWStructuredBuffer<SharcPackedData> g_sharcResolved;
    public ConstantBuffer<SharcShaderConstants> g_sharcConstants;

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
        SharcParameters p = buildSharcParameters();
        SharcHitData hd;
        hd.positionWorld = worldPos;
        hd.normalWorld = normal;
        return SharcGetCachedRadiance(p, hd, radiance, false);
    }

    // Hash-grid debug overlays. Useful for verifying scene-scale alignment,
    // occupancy, and collision rate. Cheap — call them in place of an
    // indirect bounce in a debug pixel shader.
    public float3 debugColoredHash(float3 worldPos, float3 normal) {
        return HashGridDebugColoredHash(worldPos, normal, buildHashGridParameters());
    }

    public float3 debugOccupancy(uint2 pixel, uint2 screenSize) {
        return HashGridDebugOccupancy(pixel, screenSize, buildSharcParameters().hashGridData);
    }

    public float3 debugHashCollisions(float3 worldPos, float3 normal) {
        SharcParameters p = buildSharcParameters();
        return HashGridDebugHashCollisions(worldPos, normal, p.hashGridParameters, p.hashGridData);
    }
}
