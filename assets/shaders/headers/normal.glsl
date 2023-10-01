
// input param: a vector with only one component being 1 or -1, the rest being 0
// +1 0 0 | 0b101
// -1 0 0 | 0b100
// 0 +1 0 | 0b011
// 0 -1 0 | 0b010
// 0 0 +1 | 0b001
// 0 0 -1 | 0b000
uint8_t normal2FaceID(vec3 normalObject) {
    float s = clamp(normalObject.x + normalObject.y + normalObject.z, 0.0, 1.0); // Sign of the nonzero component
    uint8_t faceId = uint8_t(s); // The lowest digit is 1 if the sign is positive, 0 otherwise

    // 4 (0b100) if z is the nonzero component, 2 (0b010) if y is the nonzero component, 0 if x is the nonzero component
    uint8_t index = uint8_t(abs(normalObject.z)) * uint8_t(4) + uint8_t(abs(normalObject.y)) * uint8_t(2);

    faceId += index;
    return faceId;
}

vec3 faceId2Normal(uint8_t faceId) {
    float s = float(faceId & 1) * 2.0 - 1.0; // Extract the lowest component and restore as the sign.

    vec3 normal = vec3(0);
    normal[faceId >> 1] = s;
    return normal;
}

// This function rotates `target` from the z axis by the same amount as `normal`.
// param: normal: a unit vector.
//        sample: the vector to be rotated
vec3 rotateVectorByNormal(vec3 normal, vec3 target) {
    vec4 quat = normalize(vec4(-normal.y, normal.x, 0.0, 1.0 + normal.z));
    if (normal.z < -0.99999) {
        quat = vec4(-1.0, 0.0, 0.0, 0.0);
    }
    return 2.0 * dot(quat.xyz, target) * quat.xyz + (quat.w * quat.w - dot(quat.xyz, quat.xyz)) * target + 2.0 * quat.w * cross(quat.xyz, target);
}

vec3 CubedNormalize(vec3 dir) {
    vec3 dir_abs = abs(dir);
    float max_element = max(dir_abs.x, max(dir_abs.y, dir_abs.z));
    return sign(dir) * step(max_element, dir_abs);
}
