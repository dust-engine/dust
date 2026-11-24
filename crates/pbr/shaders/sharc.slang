// SHaRC, NVIDIA's spatially hashed radiance cache: a hash map from the cells of
// a grid around the camera to the light the surfaces in them reflect. The path
// tracer (path_tracer.slang) adds to it along a few paths and ends its other
// paths in it; between the two, the resolve pass (sharc_resolve.slang) averages
// it over the frames. This sets up the SDK, whose integration guide describes
// the passes:
// https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md

// Apple GPUs have no 64-bit atomics, so a lock per entry guards the insertions.
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 0
// A cell stores light without the albedo and emission of the surfaces in it.
// Voxels of different colors can share a cell, and averaging them would blur
// their colors. Every hit knows its own albedo and emission, so they are applied
// when the cell is read.
#define SHARC_MATERIAL_DEMODULATION 1
#define SHARC_SEPARATE_EMISSIVE 1
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_GLSL 0
#ifndef SHARC_UPDATE
#define SHARC_UPDATE 0
#endif
// This bias shifts positions slightly, so that rounding errors don't flip points
// on a cell boundary between two cells. It isn't needed, because sharcSurface
// already moves surfaces away from the boundaries.
#define HASH_GRID_POSITION_BIAS 0.0
// The buffers are used by address.
#define RW_STRUCTURED_BUFFER(name, type) type *name
#include "SharcCommon.h"

// A frame's samples are added up as integers, in steps of 1 / kRadianceScale.
// Radiance is in units of the sun's illuminance (AtmosphereParams's
// solar_illuminance is 1). Smaller steps would lose less of dim light, such as a
// house's interior's, and larger ones leave more room for bright emitters'.
static const float kRadianceScale = 1e5;

// The most light to add for a sample, so that 1024 of them fit in an entry.
static const float kMaxCachedRadiance = 4294967296.0 / (1024.0 * kRadianceScale);

// Layout MUST match `SharcConstants` in crates/pbr/src/sharc.rs.
struct SharcConstants {
    // Addresses of the buffers.
    uint64_t hashEntries;
    uint64_t locks;
    uint64_t accumulation;
    uint64_t resolved;
    float3 cameraPosition;
    float sceneScale;
    float3 previousCameraPosition;
    float levelBias;
    uint entries;
    // Frames the cells average.
    uint accumulationFrames;
    // Frames without a new sample after which a cell is dropped.
    uint staleFrames;
    // The update traces a path from one pixel of each block of this many
    // squared.
    uint updateBlock;

    SharcParameters parameters() {
        SharcParameters p;
        p.hashGridParameters.cameraPosition = cameraPosition;
        p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        p.hashGridParameters.sceneScale = sceneScale;
        p.hashGridParameters.levelBias = levelBias;
        p.hashGridData.capacity = entries;
        p.hashGridData.hashEntriesBuffer = (HashGridKey *)hashEntries;
        p.hashGridData.lockBuffer = (uint *)locks;
        p.radianceScale = kRadianceScale;
        p.accumulationBuffer = (SharcAccumulationData *)accumulation;
        p.resolvedBuffer = (SharcPackedData *)resolved;
        return p;
    }
}

// A surface as the cache takes it.
SharcHitData sharcSurface(SharcParameters cache, float3 position, float3 normal, float3 albedo, float3 emission) {
    SharcHitData surface;
    // The smallest cells (level 1) are as wide as a voxel, so the faces of voxels
    // lie exactly on cell boundaries, where rounding errors would decide which
    // cell a point falls in. Moving the point half a voxel inward from its face
    // puts it well inside its own voxel.
    float voxelSize = HashGridGetVoxelSize(1, cache.hashGridParameters);
    surface.positionWorld = position - 0.5 * voxelSize * normal;
    // SHaRC's key stores only the signs of the normal's components. A voxel's
    // +X, +Y and +Z faces have the same signs, so they would share an entry and
    // average their light. This rotation gives each of the six faces a different
    // combination of signs. Only the key uses this normal.
    surface.normalWorld = (normal.x * float3(2.0, 2.0, 1.0)
                        + normal.y * float3(-2.0, 1.0, 2.0)
                        + normal.z * float3(1.0, -2.0, 2.0)) / 3.0;
    surface.materialDemodulation = albedo;
    surface.emissive = emission;
    return surface;
}
