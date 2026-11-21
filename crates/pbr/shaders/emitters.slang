module emitters;

// Emitter tables (crates/pbr/src/emitters.rs): each model's emissive voxel
// faces, grouped into clusters by 32³ region and face direction, and this
// frame's instances of those models. `EmitterReservoir` resamples a point on
// one face to light a shading point with, from candidates drawn by power and a
// reservoir kept from earlier frames (ReSTIR).
//
// Every struct here is laid out the same under std430 and scalar rules, so it
// reads the bytes emitters.rs writes whichever the pointers use.

// Rec. 709 luminance, the weights emitters.rs computes power with.
public static const float3 kRec709Luminance = float3(0.2126, 0.7152, 0.0722);

// The faces of one region that point one way, in the model's object space.
public struct EmitterCluster {
    // Bounds of the faces themselves.
    public float3 min;
    public uint faceOffset;
    public float3 max;
    public uint faceCount;
    // The sum of the faces' luminance times area.
    public float power;
    // 0..6 for +x, -x, +y, -y, +z, -z.
    public uint direction;
    // Alias table over the table's clusters, by power.
    public float aliasThreshold;
    public uint alias;
}

// One emissive face and its alias-table entry.
public struct EmitterFace {
    // x | y << 8 | z << 16 | direction << 24
    public uint packed;
    // The emitted radiance, RGB9E5.
    public uint radiance;
    // Draw a face of the cluster uniformly, then keep it below this threshold
    // and take its alias (an index within the cluster) above it.
    public float aliasThreshold;
    public uint alias;

    public property uint3 coords {
        get { return uint3(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF); }
    }
    public property uint direction {
        get { return packed >> 24; }
    }
}

// An instance of a model with emitters, placed in the world this frame.
public struct EmitterInstance {
    // Rows of the 3×4 world-from-object and object-from-world matrices.
    public float4 objectToWorld[3];
    public float4 worldToObject[3];
    public EmitterCluster *clusters;
    public EmitterFace *faces;
    public uint clusterCount;
    // Object-space size of a voxel.
    public float unitSize;
    // World area per object area: the square of the instance's uniform scale.
    public float areaScale;
    // World-space power: the table's power times areaScale.
    public float power;
    // Alias table over this frame's instances, by power.
    public float aliasThreshold;
    public uint alias;
    uint _pad0;
    uint _pad1;
}

float3 transformPoint(float4 rows[3], float3 p) {
    float4 h = float4(p, 1.0);
    return float3(dot(rows[0], h), dot(rows[1], h), dot(rows[2], h));
}

float3 transformVector(float4 rows[3], float3 v) {
    return float3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v));
}

// Unpacks VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: three 9-bit mantissas and a 5-bit
// exponent biased by 15.
public float3 decodeRGB9E5(uint packed) {
    float scale = exp2(float(packed >> 27) - 24.0);
    return float3(float(packed & 0x1FF), float((packed >> 9) & 0x1FF),
                  float((packed >> 18) & 0x1FF)) * scale;
}

// The unit normal of a face direction.
float3 directionNormal(uint direction) {
    float3 n = float3(0.0, 0.0, 0.0);
    n[direction >> 1] = (direction & 1) == 0 ? 1.0 : -1.0;
    return n;
}

// A small PCG generator, so this module needs no imports: uniform in [0, 1)
// with 24 bits.
public float nextUniform(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8) * (1.0 / 16777216.0);
}

// Draws entry `i` of an alias table of `count` entries uniformly, from one
// random number, and returns that number's leftover, uniform in [0, 1), to
// decide between the entry and its alias.
uint drawAliasEntry(float u, uint count, out float leftover) {
    float slot = u * float(count);
    uint entry = min(uint(slot), count - 1);
    leftover = slot - float(entry);
    return entry;
}

// A point on an emissive face. `id` names it across frames; the rest is where
// it is this frame (`decodeEmitterSample`).
public struct EmitterSample {
    // instance (8 bits, into this frame's instance list) | face (24, into the
    // instance's table) | u (10) | v (10): the point's position across the
    // face, at the center of a 1024² grid.
    public uint64_t id;
    // World space.
    public float3 position;
    // World space, unit length, pointing out of the face.
    public float3 normal;
    public float3 radiance;
}

// The sample `id` names, placed by this frame's instances. False when the id
// names an instance the list no longer has.
public bool decodeEmitterSample(EmitterInstance *instances, uint instanceCount, uint64_t id,
                                out EmitterSample s) {
    s.id = id;
    s.position = float3(0.0, 0.0, 0.0);
    s.normal = float3(0.0, 0.0, 1.0);
    s.radiance = float3(0.0, 0.0, 0.0);
    uint instanceIndex = uint(id >> 56);
    if (instanceIndex >= instanceCount) {
        return false;
    }
    EmitterInstance instance = instances[instanceIndex];
    EmitterFace face = instance.faces[uint(id >> 32) & 0xFFFFFF];
    float u = (float(uint(id >> 10) & 0x3FF) + 0.5) / 1024.0;
    float v = (float(uint(id) & 0x3FF) + 0.5) / 1024.0;

    uint axis = face.direction >> 1;
    float3 corner = float3(face.coords);
    corner[axis] += (face.direction & 1) == 0 ? 1.0 : 0.0;
    corner[(axis + 1) % 3] += u;
    corner[(axis + 2) % 3] += v;
    s.position = transformPoint(instance.objectToWorld, corner * instance.unitSize);
    s.normal = normalize(transformVector(instance.objectToWorld, directionNormal(face.direction)));
    s.radiance = decodeRGB9E5(face.radiance);
    return true;
}

// A point on an emissive face drawn in proportion to its power: an instance by
// the instances' alias table, a cluster of its table by the clusters', a face
// by the cluster's, and a point on the face uniformly. Its density per unit of
// world area is the face's luminance over the instances' total power.
EmitterSample drawByPower(EmitterInstance *instances, uint instanceCount, inout uint rng) {
    float leftover;
    uint i = drawAliasEntry(nextUniform(rng), instanceCount, leftover);
    if (leftover >= instances[i].aliasThreshold) {
        i = instances[i].alias;
    }
    EmitterInstance instance = instances[i];
    uint c = drawAliasEntry(nextUniform(rng), instance.clusterCount, leftover);
    EmitterCluster cluster = instance.clusters[c];
    if (leftover >= cluster.aliasThreshold) {
        cluster = instance.clusters[cluster.alias];
    }
    uint f = drawAliasEntry(nextUniform(rng), cluster.faceCount, leftover);
    if (leftover >= instance.faces[cluster.faceOffset + f].aliasThreshold) {
        f = instance.faces[cluster.faceOffset + f].alias;
    }
    uint u = min(uint(nextUniform(rng) * 1024.0), 1023);
    uint v = min(uint(nextUniform(rng) * 1024.0), 1023);
    uint64_t id = (uint64_t(i) << 56) | (uint64_t(cluster.faceOffset + f) << 32)
        | uint64_t(u << 10) | uint64_t(v);
    EmitterSample s;
    decodeEmitterSample(instances, instanceCount, id, s);
    return s;
}

// The floor under both cosines of `emitterTarget`.
static const float kEmitterTargetCosFloor = 0.05;

// What the samples are resampled toward: about the light `s` would bring a
// point at `x` on a surface facing `n` if nothing blocked it, up to the
// receiver's albedo / π, luminance(L_e) · cosθ · cosθ' / d², but with both
// cosines held at least kEmitterTargetCosFloor.
//
// The floor gives every point the same support: a cell's Updates shade
// different points of it and resample each other's samples, and with exact
// cosines a sample one point sees may face away from another. ReSTIR's 1/M
// weights then count candidates that couldn't have been picked, which darkened
// the emissive room's cache by 23%. With the floor that can't happen, at the
// cost of sometimes picking a face that faces away, which lights nothing.
public float emitterTarget(EmitterSample s, float3 x, float3 n) {
    float3 toSample = s.position - x;
    float distanceSquared = dot(toSample, toSample);
    if (!(distanceSquared > 0.0)) {
        return 0.0;
    }
    float3 w = toSample * rsqrt(distanceSquared);
    float cosSource = max(dot(n, w), kEmitterTargetCosFloor);
    float cosEmitter = max(-dot(s.normal, w), kEmitterTargetCosFloor);
    return dot(s.radiance, kRec709Luminance) * cosSource * cosEmitter / distanceSquared;
}

// A weighted reservoir of one emitter sample (ReSTIR's Alg. 2), streamed into
// from fresh candidates (`addCandidates`) and from a reservoir kept from an
// earlier frame (`merge`).
public struct EmitterReservoir {
    public EmitterSample sample;
    // The kept sample's target at the point the reservoir is for.
    public float target;
    public float weightSum;
    // How many candidates the reservoir has seen.
    public float count;

    public __init() {
        sample.id = 0;
        sample.position = float3(0.0, 0.0, 0.0);
        sample.normal = float3(0.0, 0.0, 1.0);
        sample.radiance = float3(0.0, 0.0, 0.0);
        target = 0.0;
        weightSum = 0.0;
        count = 0.0;
    }

    // Streams in one sample with resampling weight `weight`, standing for
    // `count` candidates.
    [mutating]
    public void update(EmitterSample candidate, float candidateTarget, float weight,
                       float candidateCount, float u) {
        count += candidateCount;
        if (!(weight > 0.0)) {
            return;
        }
        weightSum += weight;
        if (u * weightSum < weight) {
            sample = candidate;
            target = candidateTarget;
        }
    }

    // Draws `candidates` points in proportion to their power and streams them
    // in, each weighted by its target over its density (Alg. 3).
    [mutating]
    public void addCandidates(EmitterInstance *instances, uint instanceCount, float totalPower,
                              uint candidates, float3 x, float3 n, inout uint rng) {
        for (uint k = 0; k < candidates; k++) {
            EmitterSample candidate = drawByPower(instances, instanceCount, rng);
            float candidateTarget = emitterTarget(candidate, x, n);
            float luminance = dot(candidate.radiance, kRec709Luminance);
            // target / density, with density = luminance / totalPower.
            float weight = luminance > 0.0 ? candidateTarget * totalPower / luminance : 0.0;
            update(candidate, candidateTarget, weight, 1.0, nextUniform(rng));
        }
    }

    // Streams in a reservoir kept from an earlier frame, given by its sample's
    // id, its weight W and how many candidates it saw (Alg. 4): its sample is
    // re-evaluated at `x`, where the instances are now. `maxCount` caps how
    // many candidates it stands for, so old samples fade out.
    [mutating]
    public void merge(EmitterInstance *instances, uint instanceCount, uint64_t id, float W,
                      float previousCount, float maxCount, float3 x, float3 n, inout uint rng) {
        EmitterSample previous;
        if (previousCount <= 0.0 || !decodeEmitterSample(instances, instanceCount, id, previous)) {
            return;
        }
        float previousTarget = emitterTarget(previous, x, n);
        float capped = min(previousCount, maxCount);
        update(previous, previousTarget, previousTarget * W * capped, capped, nextUniform(rng));
    }

    // The kept sample's weight W: its light estimate is its integrand times W,
    // as if W were 1 / density. 0 when it has no sample worth keeping.
    public property float W {
        get {
            return target > 0.0 && count > 0.0 ? weightSum / (count * target) : 0.0;
        }
    }
}
