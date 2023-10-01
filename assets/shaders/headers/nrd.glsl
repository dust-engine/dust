
vec2 _NRD_EncodeUnitVector( vec3 v, const bool bSigned )
{
    v /= dot( abs( v ), vec3(1.0) );

    vec2 octWrap = ( 1.0 - abs( v.yx ) ) * ( step( 0.0, v.xy ) * 2.0 - 1.0 );
    v.xy = v.z >= 0.0 ? v.xy : octWrap;

    return bSigned ? v.xy : v.xy * 0.5 + 0.5;
}

#define NRD_ROUGHNESS_ENCODING_SQ_LINEAR                                                0 // linearRoughness * linearRoughness
#define NRD_ROUGHNESS_ENCODING_LINEAR                                                   1 // linearRoughness
#define NRD_ROUGHNESS_ENCODING_SQRT_LINEAR                                              2 // sqrt( linearRoughness )

#define NRD_NORMAL_ENCODING_RGBA8_UNORM                                                 0
#define NRD_NORMAL_ENCODING_RGBA8_SNORM                                                 1
#define NRD_NORMAL_ENCODING_R10G10B10A2_UNORM                                           2 // supports material ID bits
#define NRD_NORMAL_ENCODING_RGBA16_UNORM                                                3
#define NRD_NORMAL_ENCODING_RGBA16_SNORM                                                4 // also can be used with FP formats


#define NRD_NORMAL_ENCODING NRD_NORMAL_ENCODING_R10G10B10A2_UNORM
#define NRD_ROUGHNESS_ENCODING NRD_ROUGHNESS_ENCODING_LINEAR
vec4 NRD_FrontEnd_PackNormalAndRoughness(vec3 N, float roughness, float materialID )
{
    vec4 p;

    #if( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR )
        roughness = sqrt( clamp( roughness, 0, 1 ) );
    #elif( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR )
        roughness *= roughness;
    #endif

    #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        p.xy = _NRD_EncodeUnitVector( N, false );
        p.z = roughness;
        p.w = clamp( materialID / 3.0, 0.0, 1.0 );
    #else
        // Best fit ( optional )
        N /= max( abs( N.x ), max( abs( N.y ), abs( N.z ) ) );

        #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM )
            N = N * 0.5 + 0.5;
        #endif

        p.xyz = N;
        p.w = roughness;
    #endif

    return p;
}

vec3 _NRD_DecodeUnitVector( vec2 p, const bool bSigned, const bool bNormalize )
{
    p = bSigned ? p : ( p * 2.0 - 1.0 );

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    vec3 n = vec3( p.xy, 1.0 - abs( p.x ) - abs( p.y ) );
    float t = clamp( -n.z, 0.0, 1.0 );
    n.xy -= t * ( step( 0.0, n.xy ) * 2.0 - 1.0 );

    return bNormalize ? normalize( n ) : n;
}

vec4 NRD_FrontEnd_UnpackNormalAndRoughness( vec4 p, out float materialID )
{
    vec4 r;
    #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
        r.xyz = _NRD_DecodeUnitVector( p.xy, false, false );
        r.w = p.z;

        materialID = p.w;
    #else
        #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM )
            p.xyz = p.xyz * 2.0 - 1.0;
        #endif

        r.xyz = p.xyz;
        r.w = p.w;

        materialID = 0;
    #endif

    r.xyz = normalize( r.xyz );

    #if( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQRT_LINEAR )
        r.w *= r.w;
    #elif( NRD_ROUGHNESS_ENCODING == NRD_ROUGHNESS_ENCODING_SQ_LINEAR )
        r.w = sqrt( r.w );
    #endif

    return r;
}


#define NRD_FP16_MIN 1e-7 // min allowed hitDist (0 = no data)

vec3 _NRD_LinearToYCoCg( vec3 color )
{
    float Y = dot( color, vec3( 0.25, 0.5, 0.25 ) );
    float Co = dot( color, vec3( 0.5, 0.0, -0.5 ) );
    float Cg = dot( color, vec3( -0.25, 0.5, -0.25 ) );

    return vec3( Y, Co, Cg );
}


vec4 REBLUR_FrontEnd_PackRadianceAndNormHitDist( vec3 radiance, float normHitDist)
{
    /*
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0, NRD_FP16_MAX );
        normHitDist = ( isnan( normHitDist ) | isinf( normHitDist ) ) ? 0 : saturate( normHitDist );
    }
    */

    // "0" is reserved to mark "no data" samples, skipped due to probabilistic sampling
    if( normHitDist != 0 )
        normHitDist = max( normHitDist, NRD_FP16_MIN );

    radiance = _NRD_LinearToYCoCg( radiance );

    return vec4( radiance, normHitDist );
}

