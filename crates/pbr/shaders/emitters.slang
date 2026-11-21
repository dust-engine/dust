module emitters;

// Emitter tables (crates/pbr/src/emitters.rs): each model's emissive voxel
// faces, grouped into clusters by 32³ region and face direction, and this
// frame's instances of those models. `sampleEmitter` picks a point on one face
// to light a shading point with.
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
float nextUniform(inout uint state) {
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

// A point on an emissive face.
public struct EmitterSample {
    // World space.
    public float3 position;
    // World space, unit length, pointing out of the face.
    public float3 normal;
    public float3 radiance;
    // The resampling weight W: the point's contribution estimate is its
    // integrand f times W, as if W were 1 / pdf.
    public float weight;
}

// A point on an emissive face drawn in proportion to its power: an instance by
// the instances' alias table, a cluster of its table by the clusters', a face
// by the cluster's, and a point on the face uniformly. Its density per unit of
// world area is the face's luminance over the instances' total power.
EmitterSample drawByPower(EmitterInstance *instances, uint instanceCount, inout uint rng) {
    float leftover;
    uint i = drawAliasEntry(nextUniform(rng), instanceCount, leftover);
    EmitterInstance instance = instances[i];
    if (leftover >= instance.aliasThreshold) {
        instance = instances[instance.alias];
    }
    uint c = drawAliasEntry(nextUniform(rng), instance.clusterCount, leftover);
    EmitterCluster cluster = instance.clusters[c];
    if (leftover >= cluster.aliasThreshold) {
        cluster = instance.clusters[cluster.alias];
    }
    uint f = drawAliasEntry(nextUniform(rng), cluster.faceCount, leftover);
    EmitterFace face = instance.faces[cluster.faceOffset + f];
    if (leftover >= face.aliasThreshold) {
        face = instance.faces[cluster.faceOffset + face.alias];
    }

    uint axis = face.direction >> 1;
    float3 corner = float3(face.coords);
    corner[axis] += (face.direction & 1) == 0 ? 1.0 : 0.0;
    corner[(axis + 1) % 3] += nextUniform(rng);
    corner[(axis + 2) % 3] += nextUniform(rng);

    EmitterSample s;
    s.position = transformPoint(instance.objectToWorld, corner * instance.unitSize);
    s.normal = normalize(transformVector(instance.objectToWorld, directionNormal(face.direction)));
    s.radiance = decodeRGB9E5(face.radiance);
    s.weight = 0.0;
    return s;
}

// Picks a point on an emissive face to light a point at `x` (world space) on a
// surface facing `n`, by resampled importance sampling (ReSTIR's Alg. 3): draws
// `candidates` points in proportion to their power, and keeps one in
// proportion to the light it would bring `x` if nothing blocked it,
// luminance(L_e) · cosθ · cosθ' / d². `totalPower` is the instances' total
// power. `seed` seeds the draws. False when no candidate can light `x`.
public bool sampleEmitter(EmitterInstance *instances, uint instanceCount, float totalPower,
                          uint candidates, float3 x, float3 n, uint seed,
                          out EmitterSample s) {
    uint rng = seed;
    float weightSum = 0.0;
    float keptTarget = 0.0;
    s = drawByPower(instances, instanceCount, rng);
    for (uint k = 0; k < candidates; k++) {
        EmitterSample candidate = k == 0 ? s : drawByPower(instances, instanceCount, rng);
        float3 toCandidate = candidate.position - x;
        float distanceSquared = dot(toCandidate, toCandidate);
        float3 w = toCandidate * rsqrt(distanceSquared);
        float cosSource = dot(n, w);
        float cosEmitter = -dot(candidate.normal, w);
        if (cosSource <= 0.0 || cosEmitter <= 0.0) {
            continue;
        }
        float luminance = dot(candidate.radiance, kRec709Luminance);
        float target = luminance * cosSource * cosEmitter / distanceSquared;
        // target / pdf, with pdf = luminance / totalPower.
        float weight = totalPower * cosSource * cosEmitter / distanceSquared;
        weightSum += weight;
        if (nextUniform(rng) * weightSum < weight) {
            s = candidate;
            keptTarget = target;
        }
    }
    if (weightSum <= 0.0 || keptTarget <= 0.0) {
        return false;
    }
    s.weight = weightSum / (float(candidates) * keptTarget);
    return true;
}
