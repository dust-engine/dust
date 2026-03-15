// Sky Atmosphere Common - Shared structures and functions
// Based on "A Scalable and Production Ready Sky and Atmosphere Rendering Technique" (EGSR 2020)

static const float PI = 3.14159265358979323846;
static const float PLANET_RADIUS_OFFSET = 0.01;

// LUT dimensions
static const int TRANSMITTANCE_WIDTH = 256;
static const int TRANSMITTANCE_HEIGHT = 64;
static const int MULTI_SCATTERING_RES = 32;
static const int SKY_VIEW_WIDTH = 192;
static const int SKY_VIEW_HEIGHT = 108;

// Atmosphere parameters - matches Rust struct layout
struct AtmosphereParams {
    // Planet geometry (km)
    float bottom_radius;        // 6360.0 for Earth
    float top_radius;           // 6460.0 for Earth (100km atmosphere)
    float2 _pad0;

    // Rayleigh scattering (air molecules)
    float3 rayleigh_scattering;
    float rayleigh_density_exp_scale;  // -1/8 for Earth

    // Mie scattering (aerosols)
    float3 mie_scattering;
    float mie_density_exp_scale;       // -1/1.2 for Earth

    float3 mie_extinction;
    float mie_phase_g;                 // 0.8 for Earth

    float3 mie_absorption;
    float _pad1;

    // Ozone absorption layer
    float3 absorption_extinction;
    float absorption_density_0_layer_width;

    float absorption_density_0_constant_term;
    float absorption_density_0_linear_term;
    float absorption_density_1_constant_term;
    float absorption_density_1_linear_term;

    // Ground
    float3 ground_albedo;
    float _pad2;

    // Sun
    float3 sun_direction;
    float sun_angular_radius;

    float3 solar_irradiance;
    float _pad3;

    // Camera
    float3 camera_position;
    float _pad4;

    // View matrices (column_major to match glam/Bevy layout)
    column_major float4x4 inv_view_proj_mat;
    column_major float4x4 inv_proj_mat;
    column_major float4x4 inv_view_mat;

    // Resolution
    float2 resolution;
    float2 _pad5;
};

// Medium sample at a point in the atmosphere
struct MediumSample {
    float3 scattering;      // Total scattering
    float3 absorption;      // Total absorption
    float3 extinction;      // scattering + absorption

    float3 scattering_mie;
    float3 scattering_ray;

    float3 extinction_mie;
    float3 extinction_ray;
    float3 extinction_ozo;
};

// Ray-sphere intersection
// Returns distance to nearest intersection, or -1 if no intersection
float raySphereIntersectNearest(float3 r0, float3 rd, float3 s0, float sR) {
    float a = dot(rd, rd);
    float3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0 * a * c;

    if (delta < 0.0 || a == 0.0) {
        return -1.0;
    }

    float sol0 = (-b - sqrt(delta)) / (2.0 * a);
    float sol1 = (-b + sqrt(delta)) / (2.0 * a);

    if (sol0 < 0.0 && sol1 < 0.0) {
        return -1.0;
    }
    if (sol0 < 0.0) {
        return max(0.0, sol1);
    }
    if (sol1 < 0.0) {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

// Phase functions

// Rayleigh phase function
float rayleighPhase(float cosTheta) {
    float factor = 3.0 / (16.0 * PI);
    return factor * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks Mie phase function (better than Henyey-Greenstein)
float cornetteShanksMiePhase(float g, float cosTheta) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

// Henyey-Greenstein phase function
float hgPhase(float g, float cosTheta) {
    return cornetteShanksMiePhase(g, cosTheta);
}

// Uniform phase function
float uniformPhase() {
    return 1.0 / (4.0 * PI);
}

// Sample the atmosphere medium properties at a given world position
MediumSample sampleMedium(float3 worldPos, AtmosphereParams atm) {
    float viewHeight = length(worldPos) - atm.bottom_radius;

    // Density at current height
    float densityMie = exp(atm.mie_density_exp_scale * viewHeight);
    float densityRay = exp(atm.rayleigh_density_exp_scale * viewHeight);

    // Ozone density profile (tent function)
    float densityOzo = saturate(viewHeight < atm.absorption_density_0_layer_width ?
        atm.absorption_density_0_linear_term * viewHeight + atm.absorption_density_0_constant_term :
        atm.absorption_density_1_linear_term * viewHeight + atm.absorption_density_1_constant_term);

    MediumSample s;

    s.scattering_mie = densityMie * atm.mie_scattering;
    s.extinction_mie = densityMie * atm.mie_extinction;

    s.scattering_ray = densityRay * atm.rayleigh_scattering;
    s.extinction_ray = s.scattering_ray; // Rayleigh has no absorption

    s.extinction_ozo = densityOzo * atm.absorption_extinction;

    s.scattering = s.scattering_mie + s.scattering_ray;
    s.absorption = densityMie * atm.mie_absorption + s.extinction_ozo;
    s.extinction = s.extinction_mie + s.extinction_ray + s.extinction_ozo;

    return s;
}

// LUT parameterization functions

// Convert physical parameters to transmittance LUT UV
void transmittanceParamsToUv(AtmosphereParams atm, float viewHeight, float viewZenithCosAngle, out float2 uv) {
    float H = sqrt(max(0.0, atm.top_radius * atm.top_radius - atm.bottom_radius * atm.bottom_radius));
    float rho = sqrt(max(0.0, viewHeight * viewHeight - atm.bottom_radius * atm.bottom_radius));

    float discriminant = viewHeight * viewHeight * (viewZenithCosAngle * viewZenithCosAngle - 1.0) + atm.top_radius * atm.top_radius;
    float d = max(0.0, (-viewHeight * viewZenithCosAngle + sqrt(discriminant)));

    float d_min = atm.top_radius - viewHeight;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;

    uv = float2(x_mu, x_r);
}

// Convert transmittance LUT UV to physical parameters
void uvToTransmittanceParams(AtmosphereParams atm, out float viewHeight, out float viewZenithCosAngle, float2 uv) {
    float x_mu = uv.x;
    float x_r = uv.y;

    float H = sqrt(atm.top_radius * atm.top_radius - atm.bottom_radius * atm.bottom_radius);
    float rho = H * x_r;
    viewHeight = sqrt(rho * rho + atm.bottom_radius * atm.bottom_radius);

    float d_min = atm.top_radius - viewHeight;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    viewZenithCosAngle = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * viewHeight * d);
    viewZenithCosAngle = clamp(viewZenithCosAngle, -1.0, 1.0);
}

// Sky view LUT parameterization
// Non-linear mapping to allocate more precision near the horizon

float fromUnitToSubUvs(float u, float resolution) {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

float fromSubUvsToUnit(float u, float resolution) {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

void uvToSkyViewParams(AtmosphereParams atm, out float viewZenithCosAngle, out float lightViewCosAngle, float viewHeight, float2 uv) {
    uv = float2(fromSubUvsToUnit(uv.x, float(SKY_VIEW_WIDTH)), fromSubUvsToUnit(uv.y, float(SKY_VIEW_HEIGHT)));

    float Vhorizon = sqrt(viewHeight * viewHeight - atm.bottom_radius * atm.bottom_radius);
    float CosBeta = Vhorizon / viewHeight;
    float Beta = acos(CosBeta);
    float ZenithHorizonAngle = PI - Beta;

    if (uv.y < 0.5) {
        float coord = 2.0 * uv.y;
        coord = 1.0 - coord;
        coord *= coord;
        coord = 1.0 - coord;
        viewZenithCosAngle = cos(ZenithHorizonAngle * coord);
    } else {
        float coord = uv.y * 2.0 - 1.0;
        coord *= coord;
        viewZenithCosAngle = cos(ZenithHorizonAngle + Beta * coord);
    }

    float lightCoord = uv.x;
    lightCoord *= lightCoord;
    lightViewCosAngle = -(lightCoord * 2.0 - 1.0);
}

void skyViewParamsToUv(AtmosphereParams atm, bool intersectGround, float viewZenithCosAngle, float lightViewCosAngle, float viewHeight, out float2 uv) {
    float Vhorizon = sqrt(viewHeight * viewHeight - atm.bottom_radius * atm.bottom_radius);
    float CosBeta = Vhorizon / viewHeight;
    float Beta = acos(CosBeta);
    float ZenithHorizonAngle = PI - Beta;

    if (!intersectGround) {
        float coord = acos(viewZenithCosAngle) / ZenithHorizonAngle;
        coord = 1.0 - coord;
        coord = sqrt(coord);
        coord = 1.0 - coord;
        uv.y = coord * 0.5;
    } else {
        float coord = (acos(viewZenithCosAngle) - ZenithHorizonAngle) / Beta;
        coord = sqrt(coord);
        uv.y = coord * 0.5 + 0.5;
    }

    float coord = -lightViewCosAngle * 0.5 + 0.5;
    coord = sqrt(coord);
    uv.x = coord;

    uv = float2(fromUnitToSubUvs(uv.x, float(SKY_VIEW_WIDTH)), fromUnitToSubUvs(uv.y, float(SKY_VIEW_HEIGHT)));
}

// Move ray origin to top of atmosphere if outside
bool moveToTopAtmosphere(inout float3 worldPos, float3 worldDir, float topRadius) {
    float viewHeight = length(worldPos);
    if (viewHeight > topRadius) {
        float tTop = raySphereIntersectNearest(worldPos, worldDir, float3(0.0, 0.0, 0.0), topRadius);
        if (tTop >= 0.0) {
            float3 upVector = worldPos / viewHeight;
            float3 upOffset = upVector * -PLANET_RADIUS_OFFSET;
            worldPos = worldPos + worldDir * tTop + upOffset;
        } else {
            return false; // Ray doesn't intersect atmosphere
        }
    }
    return true;
}

// Get sun luminance if looking directly at sun
float3 getSunLuminance(float3 worldPos, float3 worldDir, float3 sunDir, float sunAngularRadius, float bottomRadius) {
    if (dot(worldDir, sunDir) > cos(sunAngularRadius)) {
        float t = raySphereIntersectNearest(worldPos, worldDir, float3(0.0, 0.0, 0.0), bottomRadius);
        if (t < 0.0) { // No intersection with ground
            return float3(1000000.0, 1000000.0, 1000000.0); // Bright sun disk
        }
    }
    return float3(0.0, 0.0, 0.0);
}
