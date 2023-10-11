float SRGBToLinear(float color)
{
    // Approximately pow(color, 2.2)
    return color < 0.04045 ? color / 12.92 : pow(abs(color + 0.055) / 1.055, 2.4);
}


vec3 sRGB2AECScg(vec3 srgb) {
    mat3 transform = mat3(
        0.6031065, 0.07011794, 0.022178888,
        0.32633433, 0.9199162, 0.11607823,
        0.047995567, 0.012763573, 0.94101846
    );
    return transform * srgb;
}
vec3 AECScg2sRGB(vec3 srgb) {
    mat3 transform = mat3(
        1.7312546, -0.131619, -0.024568284,
        -0.6040432, 1.1348418, -0.12575036,
        -0.08010775, -0.008679431, 1.0656371
    );
    return transform * srgb;
}
vec3 XYZ2ACEScg(vec3 srgb) {
    mat3 transform = mat3(
        1.6410228, -0.66366285, 0.011721907,
        -0.32480323, 1.6153315, -0.0082844375,
        -0.23642465, 0.016756356, 0.9883947
    );
    return transform * srgb;
}
