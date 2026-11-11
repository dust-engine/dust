// Radiance cascades: diffuse global illumination from radiance probes placed in
// world space.
//
// A probe holds the light arriving at a point from a set of directions, each
// over an interval of distance. The probes come in cascades: going up a cascade
// doubles the spacing between probes, quadruples the directions each one has
// and quadruples the length of their interval. Near light then comes from many
// probes with few directions, far light from few probes with many.
//
// Probes exist only where visible surfaces need them. Each cascade divides space
// into a grid of cells, one probe per cell, and keeps a hash table of the cells
// that need one. The cells double in size each time the distance from the camera
// doubles (the levels of detail, LODs), which keeps their size on screen, and
// the probe count, about constant.
//
// Each frame (rc_build.slang):
//   1. Insert the cascade 0 cell of every visible surface point.
//   2. Give each of a cascade's cells a probe, and insert each probe's parent:
//      the cell of the next cascade, at the same LOD, that contains it. Repeat
//      up the cascades.
//
// A key packs a cell's LOD and coordinates into 32 bits, since 64-bit atomics
// can't be assumed: 5 bits for the LOD plus one, so that no key is 0, and 9 per
// axis, wrapped. Wrapping loses nothing because the cells of a LOD span fewer
// than 512 per axis around the camera; RadianceCascadesConfig keeps it so.
module rc_common;
import pbr_common;

// No probe, or no slot.
public static const uint RC_INVALID = 0xFFFFFFFFu;
// Linear probing gives up after this many slots, dropping the insertion.
static const uint RC_MAX_PROBE_STEPS = 32;
// The highest LOD a key can hold. Everything farther shares it.
static const uint RC_MAX_LOD = 30;

// The push constants of the compute passes, after the ray generation stage's
// range (pbr.playout.ron).
public struct RcPush {
    // The cascade a pass works on.
    [vk::offset(4)] public uint cascade;
}

// ── Levels of detail ─────────────────────────────────────────────────────

// The LOD of a point: 0 within lodDistance of the camera, and one more each
// time the distance doubles. The distance is Chebyshev, so that a LOD's cells
// around the camera span a cube.
public uint rcLodOf(float3 position) {
    float3 offset = abs(position - params.rc.cameraPosition);
    float distance = max(offset.x, max(offset.y, offset.z)) / params.rc.lodDistance;
    return distance < 1.0 ? 0u : min(uint(floor(log2(distance))) + 1u, RC_MAX_LOD);
}

// World units between the probes of a cascade at a LOD: each LOD and each
// cascade up doubles them.
public float rcSpacing(uint lod, uint cascade) {
    return params.rc.spacing * exp2(float(lod + cascade));
}

// Where a surface point counts as being on the grid: a twentieth of a cell off
// the surface, so that voxel faces, which often lie exactly on a cell boundary,
// consistently land in the cell they face.
public float3 rcGridPoint(float3 position, float3 normal, uint lod) {
    return position + normal * 0.05 * rcSpacing(lod, 0);
}

// ── Hash tables ──────────────────────────────────────────────────────────

// The cell of a cascade's grid at a LOD that holds `position`.
public int3 rcCellOf(float3 position, uint lod, uint cascade) {
    return int3(floor(position / rcSpacing(lod, cascade)));
}

public uint rcKey(uint lod, int3 cell) {
    uint3 wrapped = uint3(cell) & 511u;
    return ((lod + 1u) << 27) | (wrapped.x << 18) | (wrapped.y << 9) | wrapped.z;
}

// The cell behind a key: the one with the wrapped coordinates nearest the camera.
public int3 rcCellOfKey(uint key, uint cascade, out uint lod) {
    lod = (key >> 27) - 1u;
    int3 wrapped = int3((key >> 18) & 511u, (key >> 9) & 511u, key & 511u);
    int3 center = rcCellOf(params.rc.cameraPosition, lod, cascade);
    int3 difference = (wrapped - center) & 511;
    return center + (difference ^ 256) - 256;
}

// Mixes a key's bits (MurmurHash3's finalizer), so that neighboring cells land
// in scattered slots.
uint rcHash(uint key) {
    key ^= key >> 16;
    key *= 0x85EBCA6Bu;
    key ^= key >> 13;
    key *= 0xC2B2AE35u;
    return key ^ (key >> 16);
}

// Adds a key to a cascade's table, unless it's there already.
public void rcInsert(uint cascade, uint key) {
    RcCascade c = params.rc.cascades[cascade];
    uint hash = rcHash(key);
    for (uint i = 0; i < RC_MAX_PROBE_STEPS; i++) {
        uint slot = c.tableOffset + ((hash + i) & c.tableMask);
        uint current = params.rc.keys[slot];
        if (current == 0) {
            InterlockedCompareExchange(params.rc.keys[slot], 0u, key, current);
            if (current == 0) {
                return;
            }
        }
        if (current == key) {
            return;
        }
    }
}

// The index of the probe of a key in a cascade, once its probes are allocated;
// RC_INVALID if it has none.
public uint rcFindProbe(uint cascade, uint key) {
    RcCascade c = params.rc.cascades[cascade];
    uint hash = rcHash(key);
    for (uint i = 0; i < RC_MAX_PROBE_STEPS; i++) {
        uint slot = c.tableOffset + ((hash + i) & c.tableMask);
        uint current = params.rc.keys[slot];
        if (current == key) {
            return params.rc.slots[slot];
        }
        if (current == 0) {
            break;
        }
    }
    return RC_INVALID;
}
