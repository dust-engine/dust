float ArHosekSkyModel_GetRadianceInternal(
        float[9]  configuration, 
        float                        cos_theta, 
        float                        gamma,
        float                        cos_gamma
        )
{
    float expM = exp(configuration[4] * gamma);
    float rayM = cos_gamma * cos_gamma;
    float mieM = (1.0 + rayM) / pow((1.0 + configuration[8]*configuration[8] - 2.0*configuration[8]*cos_gamma), 1.5);
    float zenith = sqrt(cos_theta);

    return (1.0 + configuration[0] * exp(configuration[1] / (cos_theta + 0.01))) *
            (configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith);
}

// dir: normalized view direction vector
vec3 arhosek_sky_radiance(vec3 dir)
{
    if (sunlight_config.direction.y <= 0) {
        // Avoid NaN problems.
        return vec3(0);
    }
    float cos_theta = clamp(dir.y, 0, 1);
    float cos_gamma = dot(dir, sunlight_config.direction.xyz);
    float gamma = acos(cos_gamma);


    float x =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.r.configs0.x,
            sunlight_config.r.configs0.y,
            sunlight_config.r.configs0.z,
            sunlight_config.r.configs0.w,
            sunlight_config.r.configs1.x,
            sunlight_config.r.configs1.y,
            sunlight_config.r.configs1.z,
            sunlight_config.r.configs1.w,
            sunlight_config.r.configs2
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.r.radiance;
    float y =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.g.configs0.x,
            sunlight_config.g.configs0.y,
            sunlight_config.g.configs0.z,
            sunlight_config.g.configs0.w,
            sunlight_config.g.configs1.x,
            sunlight_config.g.configs1.y,
            sunlight_config.g.configs1.z,
            sunlight_config.g.configs1.w,
            sunlight_config.g.configs2
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.g.radiance;
    float z =
    ArHosekSkyModel_GetRadianceInternal(
        float[](
            sunlight_config.b.configs0.x,
            sunlight_config.b.configs0.y,
            sunlight_config.b.configs0.z,
            sunlight_config.b.configs0.w,
            sunlight_config.b.configs1.x,
            sunlight_config.b.configs1.y,
            sunlight_config.b.configs1.z,
            sunlight_config.b.configs1.w,
            sunlight_config.b.configs2
        ), 
        cos_theta,
        gamma, cos_gamma
    ) * sunlight_config.b.radiance;
    vec3 sky_color =  vec3(x, y, z) * 683.0;
    return XYZ2ACEScg(sky_color);
}

vec3 arhosek_sun_radiance(
    vec3 dir
) {
    float cos_gamma = dot(dir, sunlight_config.direction.xyz);
    if (cos_gamma < 0.0 || dir.y < 0.0) {
        return vec3(0.0);
    }
    float sol_rad_sin = sin(sunlight_config.solar_intensity.w);
    float ar2 = 1.0 / (sol_rad_sin * sol_rad_sin);
    float singamma = 1.0 - (cos_gamma * cos_gamma);
    float sc2 = 1.0 - ar2 * singamma * singamma;
    if (sc2 <= 0.0) {
        return vec3(0.0);
    }
    float sampleCosine = sqrt(sc2);

    vec3 darkeningFactor = vec3(sunlight_config.r.ld_coefficient0, sunlight_config.g.ld_coefficient0, sunlight_config.b.ld_coefficient2);


    darkeningFactor += vec3(sunlight_config.r.ld_coefficient1, sunlight_config.g.ld_coefficient1, sunlight_config.b.ld_coefficient1) * sampleCosine;

    float currentSampleCosine = sampleCosine;
    [[unroll]]
    for (uint i = 0; i < 4; i++) {
        currentSampleCosine *= sampleCosine;
        darkeningFactor += vec3(
            sunlight_config.r.ld_coefficient2[i],
            sunlight_config.g.ld_coefficient2[i],
            sunlight_config.b.ld_coefficient2[i]
        ) * currentSampleCosine;
    }
    return XYZ2ACEScg(sunlight_config.solar_intensity.xyz * darkeningFactor);
}

