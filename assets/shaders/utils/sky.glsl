struct ArHosekSkyModelChannelConfiguration {
    float configuration[9];
    float radiance;
};

layout(set = 0, binding = 3) readonly buffer ArHosekSkyModelConfiguration{
	ArHosekSkyModelChannelConfiguration channels[3];
    vec3 direction; // normalized
} hosek_config;


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

// dir: normalized direction vector
vec3 arhosek_sky_radiance(vec3 dir)
{
    float cos_theta = clamp(dir.y, 0, 1);
    float cos_gamma = dot(dir, hosek_config.direction);
    float gamma = acos(cos_gamma);


    float x =
    ArHosekSkyModel_GetRadianceInternal(
        hosek_config.channels[0].configuration, 
        cos_theta,
        gamma, cos_gamma
    ) * hosek_config.channels[0].radiance;
    float y =
    ArHosekSkyModel_GetRadianceInternal(
        hosek_config.channels[1].configuration, 
        cos_theta,
        gamma, cos_gamma
    ) * hosek_config.channels[1].radiance;
    float z =
    ArHosekSkyModel_GetRadianceInternal(
        hosek_config.channels[2].configuration, 
        cos_theta,
        gamma, cos_gamma
    ) * hosek_config.channels[2].radiance;
    return vec3(x, y, z);
}