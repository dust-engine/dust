// Screen-space gather trace against the closest-depth HZB, plus the
// compaction that feeds the indirect hardware gather and a debug overlay.
//
// One thread per pixel marches the pixel's gather ray through the HZB
// (`traceHzb`, a port of Unreal's TraceHZB in HZBTracing.ush). On-screen hits
// read last frame's lit color at the reprojected hit and add it to the HDR
// output right here; every pixel writes a packed trace record
// (`encodeTraceHit`) saying whether a hardware ray is still needed and from
// what distance it can resume. `compact` turns those records into the
// dispatch list of final_gather.slang. Mirrors Lumen's
// ScreenProbeTraceScreenTexturesCS / ScreenProbeCompactTracesCS
// (LumenScreenProbeTracing.usf) with one ray per pixel instead of per probe
// texel.

import pbr_common;
import gather_common;

// ── TraceHZB port ────────────────────────────────────────────────────────

struct HzbTrace {
    bool hit;
    // The hit is behind a feature thinner than `numThicknessSteps` HZB texels
    // (foliage-style): treated as a miss whose resume point is the hit.
    bool uncertain;
    // Screen UV in [0, 1]^2 and device Z of the hit, or of the ray's furthest
    // unoccluded point on a miss.
    float3 hitUVZ;
    // Furthest unoccluded point along the ray.
    float3 lastVisibleUVZ;
};

// Entry / exit times of the segment `origin -> end` through the box, as
// fractions of the segment (Unreal's LineBoxIntersect, Common.ush).
float2 lineBoxIntersect(float3 origin, float3 end, float3 boxMin, float3 boxMax) {
    float3 invDir = 1.0 / (end - origin);
    float3 first = (boxMin - origin) * invDir;
    float3 second = (boxMax - origin) * invDir;
    float3 closest = min(first, second);
    float3 furthest = max(first, second);
    return float2(
        max(closest.x, max(closest.y, closest.z)),
        min(furthest.x, min(furthest.y, furthest.z)));
}

// Point sample of HZB level `mip` at `hzbUV`, clamped to the level's edge (a
// texel fetch; the layout has no room for a point sampler).
float sampleHzb(float2 hzbUV, float mip) {
    uint level = uint(mip);
    int2 extent = int2(max(params.screenTrace.hzbExtent >> level, uint2(1, 1)));
    int2 texel = clamp(int2(floor(hzbUV * float2(extent))), int2(0, 0), extent - 1);
    return params.hzbTexture.Load(int3(texel, int(level)));
}

// The depth G-buffer as "mip -1" of the HZB. UI pixels store 1 and count as
// no surface, the same rule hzb_build.slang applies.
float sampleFullResDepth(float2 hzbUV) {
    float2 renderExtent = params.screenTrace.renderExtent;
    float2 screenUV = hzbUV * params.screenTrace.hzbInvUvFactor;
    int2 texel = clamp(int2(screenUV * renderExtent), int2(0, 0), int2(renderExtent) - 1);
    float z = params.gbuffer.depthTexture[uint2(texel)];
    return z >= 1.0 ? 0.0 : z;
}

// Marches `rayOrigin + t * rayDir` (world space, unit direction) through the
// HZB. The origin must be on screen. Port of Unreal's TraceHZB with
// HZB_TRACE_INCLUDE_FULL_RES_DEPTH = 1.
HzbTrace traceHzb(float3 rayOrigin, float3 rayDir, float maxWorldDistance) {
    ScreenTraceConstants st = params.screenTrace;
    uint2 screenSize = uint2(st.renderExtent);
    float2 hzbUvFactor = st.hzbUvFactor;

    // Ray start / end in HZB UV + device Z.
    bool inFront;
    float3 startUVZ = params.uniforms.projectToScreenUVZ(rayOrigin, screenSize, inFront);
    float3 rayStart = float3(startUVZ.xy * hzbUvFactor, startUVZ.z);

    float3 rayEnd;
    {
        float3 viewDir = params.uniforms.worldDirToView(rayDir);
        float sceneDepth = -params.uniforms.worldToView(rayOrigin).z;
        // Clamp the end point in front of the camera plane (view Z < 0) so it
        // projects to a finite UV.
        float rayEndDistance = viewDir.z > 0.0
            ? min(0.99 * sceneDepth / viewDir.z, maxWorldDistance)
            : maxWorldDistance;
        float3 rayWorldEnd = rayOrigin + rayDir * rayEndDistance;
        float3 endUVZ = params.uniforms.projectToScreenUVZ(rayWorldEnd, screenSize, inFront);
        rayEnd = float3(endUVZ.xy * hzbUvFactor, endUVZ.z);

        // Cut the ray where it leaves the render area or the depth range.
        float2 screenEdge = lineBoxIntersect(rayStart, rayEnd, float3(0, 0, 0), float3(hzbUvFactor, 1));
        rayEnd = rayStart + (rayEnd - rayStart) * screenEdge.y;
    }

    const float baseMipLevel = -1.0;
    float mipLevel = baseMipLevel;

    float3 rayDirUV = rayEnd - rayStart;
    float currentTime = 0.0;
    float lastAboveSurfaceTime = 0.0;

    // Which XY boundary planes of a texel the ray can exit through.
    float2 floorOffset = select(rayDirUV.xy < 0.0, 0.0, 1.0);
    float3 rayUV = rayStart;

    // Step out of the starting texel without a hit test: the origin sits on
    // the surface that texel stores.
    {
        float2 mipTexelSize = exp2(mipLevel) * st.hzbBaseTexelSize;
        float2 mipResolution = 1.0 / mipTexelSize;
        float2 uvOffset = 0.005 * mipTexelSize;
        uvOffset = select(rayDirUV.xy < 0.0, -uvOffset, uvOffset);
        float2 xyPlane = floor(rayUV.xy * mipResolution) + floorOffset;
        xyPlane = xyPlane * mipTexelSize + uvOffset;
        float2 planeTimes = (xyPlane - rayStart.xy) / rayDirUV.xy;
        currentTime = min(planeTimes.x, planeTimes.y);
        rayUV = rayStart + currentTime * rayDirUV;
    }
    if (currentTime >= 1.0) {
        // The step-out consumed the whole ray; report the start as visible.
        rayUV = rayStart;
    }

    HzbTrace result;
    result.hit = false;
    result.uncertain = false;
    float numIterations = 0.0;

    // Stackless HZB traversal: skip whole tiles while the ray stays above
    // (in front of) the stored closest depth, refine when it dips below.
    // A wave whose lanes have mostly finished stops early rather than idle
    // on a few long rays (TERMINATE_ON_LOW_OCCUPANCY in HZBTracing.ush).
    while (mipLevel >= baseMipLevel && numIterations < st.maxIterations && currentTime < 1.0
        && (st.minimumOccupancy == 0 || WaveActiveCountBits(true) > st.minimumOccupancy)) {
        float2 mipTexelSize = exp2(mipLevel) * st.hzbBaseTexelSize;
        float2 mipResolution = 1.0 / mipTexelSize;
        float2 uvOffset = 0.005 * mipTexelSize;
        uvOffset = select(rayDirUV.xy < 0.0, -uvOffset, uvOffset);
        float2 xyPlane = floor(rayUV.xy * mipResolution) + floorOffset;
        xyPlane = xyPlane * mipTexelSize + uvOffset;

        float tileZ = mipLevel < 0.0 ? sampleFullResDepth(rayUV.xy) : sampleHzb(rayUV.xy, mipLevel);

        float3 boundaryPlanes = float3(xyPlane, tileZ);
        float3 planeTimes = (boundaryPlanes - rayStart) / rayDirUV;
        // Reverse Z: the ray only crosses the tile's depth plane while moving
        // away from the camera (device Z decreasing).
        planeTimes.z = rayDirUV.z < 0.0 ? planeTimes.z : 1.0;
        float intersectionTime = min(planeTimes.x, planeTimes.y);

        bool aboveSurface = rayUV.z > tileZ;
        bool skippedTile = aboveSurface;
        intersectionTime = min(intersectionTime, planeTimes.z);
        skippedTile = skippedTile && (intersectionTime != planeTimes.z);

        if (skippedTile) {
            lastAboveSurfaceTime = intersectionTime;
        }
        currentTime = aboveSurface ? intersectionTime : currentTime;
        rayUV = rayStart + min(currentTime, 1.0) * rayDirUV;
        mipLevel += skippedTile ? 1.0 : -1.0;
        numIterations += 1.0;
    }

    if (mipLevel < baseMipLevel && currentTime < 1.0) {
        float tileZ = sampleFullResDepth(rayUV.xy);
        if (tileZ > 0.0) {
            float hitSceneDepth = params.uniforms.near / tileZ;
            float raySceneDepth = params.uniforms.near / max(rayUV.z, 1e-12);
            float thickness = max(st.relativeDepthThickness * hitSceneDepth, st.minWorldThickness);
            result.hit = (raySceneDepth - hitSceneDepth) < thickness;
        }
        if (!result.hit) {
            // Went below the surface without a hit: rewind to the last time
            // the ray was known to be in front of everything.
            rayUV = rayStart + lastAboveSurfaceTime * rayDirUV;
        }
    }

    // Certainty search: linear steps past the hit along the ray. If the ray
    // would come back out in front of the HZB within a few texels the hit
    // was on something thin, and a hardware ray decides instead.
    if (result.hit && st.numThicknessSteps > 0) {
        const float searchMip = 0.0;
        float mipNumTexels = exp2(searchMip);
        float2 tileSize = mipNumTexels * st.hzbBaseTexelSize;
        float numSteps = float(st.numThicknessSteps) / mipNumTexels;
        float searchEndTime = min(
            length(rayDirUV.xy * tileSize * numSteps) / max(length(rayEnd.xy - rayUV.xy), 1e-8),
            1.0);
        for (float i = 0.0; i < numSteps; i += 1.0) {
            float3 sampleUV = rayUV + (i / numSteps) * searchEndTime * (rayEnd - rayUV);
            if (all(sampleUV.xy > 0.0) && all(sampleUV.xy < hzbUvFactor)) {
                float sampleTileZ = sampleHzb(sampleUV.xy, searchMip);
                if (sampleUV.z > sampleTileZ) {
                    result.uncertain = true;
                }
            }
        }
    }

    result.hitUVZ = float3(rayUV.xy * st.hzbInvUvFactor, rayUV.z);
    float3 lastVisible = rayStart + lastAboveSurfaceTime * rayDirUV;
    result.lastVisibleUVZ = float3(lastVisible.xy * st.hzbInvUvFactor, lastVisible.z);
    return result;
}

// ── Lumen screen-hit acceptance helpers ─────────────────────────────────

// Fades hits out toward the screen border (`screenPos` in [-1, 1]^2), so
// that a hit's color is never sampled from the clamped edge texels.
// ComputeHitVignetteFromScreenPos, SSRTRayCast.ush.
float hitVignette(float2 screenPos) {
    float2 vignette = saturate(abs(screenPos) * 5.0 - 4.0);
    return saturate(1.0 - dot(vignette, vignette));
}

// Jorge Jimenez's interleaved gradient noise, with Unreal's per-frame offset.
float interleavedGradientNoise(float2 uv, float frameId) {
    uv += frameId * (float2(47.0, 17.0) * 0.695);
    const float3 magic = float3(0.06711056, 0.00583715, 52.9829189);
    return frac(magic.z * frac(dot(uv, magic.xy)));
}

// Pushes the ray origin off the depth texel's surface along the normal by
// twice the texel's extent projected onto it, so the trace does not
// self-intersect the camera-facing texel it starts in. HZB traversal is
// point sampled, so the texel sticks out of the plane it stands for.
// ApplyScreenSpaceRayBias, LumenScreenTracing.ush.
float3 applyScreenSpaceRayBias(float3 worldPos, float3 N, uint2 pixel, float deviceZ, uint2 screenSize) {
    float2 cornerUV = (float2(pixel) + 1.0) / float2(screenSize);
    float3 cornerWorld = params.uniforms.worldPosFromScreenUVZ(cornerUV, deviceZ, screenSize);
    float normalBias = abs(dot(cornerWorld - worldPos, N)) * 2.0;
    return worldPos + normalBias * N;
}

// Whether the G-buffer holds a surface at `pixel`: sky stores depth 0 and
// UI stores 1 (the same pixels albedoUint.a marks with 0xFF).
bool isSurfacePixel(uint2 pixel) {
    float z = params.gbuffer.depthTexture[pixel];
    return z > 0.0 && z < 1.0;
}

// ── Screen trace: one thread per pixel ───────────────────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    ScreenTraceConstants st = params.screenTrace;
    uint2 screenSize = uint2(st.renderExtent);
    uint2 pixel = threadId.xy;
    if (any(pixel >= screenSize)) {
        return;
    }
    // Sky / UI pixels have no surface to gather for: mark them "done" so the
    // compaction skips them.
    if (!isSurfacePixel(pixel)) {
        params.traceHit[pixel] = encodeTraceHit(0.0, true, false, false, false);
        return;
    }
    bool historyValid = (st.flags & SCREEN_TRACE_FLAG_HISTORY_VALID) != 0;
    bool enabled = (st.flags & SCREEN_TRACE_FLAG_ENABLED) != 0;
    if (!enabled || !historyValid) {
        params.traceHit[pixel] = encodeTraceHit(0.0, false, false, false, false);
        return;
    }

    float3 worldPos = params.pixelPositionWorldSpace(pixel, screenSize);
    float3 N = params.gbuffer.normalTexture[pixel].xyz;
    float3 rayDir = gatherRayDirection(pixel, N, st.frameIndex);
    float deviceZ = params.gbuffer.depthTexture[pixel];

    float3 rayOrigin = applyScreenSpaceRayBias(worldPos, N, pixel, deviceZ, screenSize);
    // The bias can push the origin off screen, which the trace cannot start
    // from; leave the pixel to the hardware ray.
    bool inFront;
    float3 originUVZ = params.uniforms.projectToScreenUVZ(rayOrigin, screenSize, inFront);
    if (!inFront || any(originUVZ.xy < 0.0) || any(originUVZ.xy > 1.0)) {
        params.traceHit[pixel] = encodeTraceHit(0.0, false, false, false, false);
        return;
    }

    HzbTrace trace = traceHzb(rayOrigin, rayDir, params.uniforms.far);
    bool screenOnly = (st.flags & SCREEN_TRACE_FLAG_SCREEN_ONLY) != 0;

    bool accepted = false;
    bool rejected = false;
    if (trace.hit && (!trace.uncertain || screenOnly)) {
        uint2 hitPixel = min(uint2(trace.hitUVZ.xy * float2(screenSize)), screenSize - 1);
        // Where the hit pixel's surface was last frame: the motion-vector
        // G-buffer already folds camera and per-instance motion. It stores
        // `prevPixel - currPixel` on the unjittered grid where integer
        // coordinates are texel centers (see prevPixelFromWorld).
        float2 jitter = float2(params.uniforms.jitter_x, params.uniforms.jitter_y);
        float2 prevPixel = (float2(hitPixel) - jitter) + params.gbuffer.motionVectorTexture[hitPixel];
        float2 prevUV = (prevPixel + 0.5) / float2(screenSize);

        float2 hitScreenPos = trace.hitUVZ.xy * 2.0 - 1.0;
        float2 prevScreenPos = prevUV * 2.0 - 1.0;
        float vignette = min(hitVignette(hitScreenPos), hitVignette(prevScreenPos));
        float noise = interleavedGradientNoise(float2(pixel) + 0.5, float(st.frameIndex % 8));
        // Stochastic rejection near the edges: the denoiser averages the
        // partial coverage instead of seeing a hard cutoff.
        bool ok = vignette >= noise;
        if (ok) {
            // History depth test: the surface hit this frame must have been
            // at the reprojected position last frame too, otherwise the
            // color there belongs to something else (disocclusion).
            float3 hitWorld = params.uniforms.worldPosFromScreenUVZ(trace.hitUVZ.xy, trace.hitUVZ.z, screenSize);
            float expectedZ = params.uniforms.prevDeviceZ(hitWorld);
            int2 prevTexel = int2(floor(prevPixel + 0.5));
            ok = all(prevTexel >= 0) && all(prevTexel < int2(screenSize));
            if (ok) {
                float historyZ = params.prevDepthTexture.Load(int3(prevTexel, 0));
                // 0 = sky, 1 = UI last frame: nothing to sample either way.
                ok = historyZ > 0.0 && historyZ < 1.0
                    && abs(historyZ - expectedZ) < st.historyDepthThickness * lerp(0.5, 2.0, noise);
            }
        }
        if (ok) {
            float3 lighting = params.prevSceneColor.SampleLevel(params.sky.linearSampler, prevUV, 0).rgb;
            // Cosine-sampled Lambertian: BRDF / pdf leaves `albedo` (see the
            // gather miss shader).
            float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
            float4 current = params.outputTexture[pixel];
            params.outputTexture[pixel] = float4(current.rgb + albedo * lighting, current.a);
            accepted = true;
        } else {
            rejected = true;
        }
    } else if (!trace.hit && screenOnly) {
        // The ray left the screen unoccluded as far as the trace could see:
        // take the sky, as the gather miss shader would.
        float3 skyLuminance = params.sky.atmosphereParams.solar_illuminance
            * params.sky.sampleScatteredTransfer(rayDir);
        float3 albedo = params.gbuffer.albedoSRGB[pixel].rgb;
        float4 current = params.outputTexture[pixel];
        params.outputTexture[pixel] = float4(current.rgb + albedo * skyLuminance, current.a);
    }

    // Resume distance of the hardware ray: the hit (accepted, uncertain or
    // rejected) or, on a miss, the furthest point the trace could still see.
    float3 resumeWorld = params.uniforms.worldPosFromScreenUVZ(trace.hitUVZ.xy, trace.hitUVZ.z, screenSize);
    float resumeDistance = distance(worldPos, resumeWorld);
    params.traceHit[pixel] = encodeTraceHit(resumeDistance, accepted, trace.uncertain, rejected, false);
}

// ── Compaction: list of pixels that still need a hardware ray ────────────

// Wave-level allocation (WAVE_OPS path of ScreenProbeCompactTracesCS): one
// atomic per wave on the count, prefix bits give each lane its slot. Every
// lane stays live for the wave intrinsics; off-screen threads contribute no
// entry.
[shader("compute")]
[numthreads(8, 8, 1)]
void compact(uint3 threadId: SV_DispatchThreadID) {
    uint2 screenSize = uint2(params.screenTrace.renderExtent);
    uint2 pixel = threadId.xy;
    bool needsRay = false;
    if (all(pixel < screenSize)) {
        needsRay = !decodeTraceHit(params.traceHit[pixel]).hit;
    }
    uint waveCount = WaveActiveCountBits(needsRay);
    uint slotInWave = WavePrefixCountBits(needsRay);
    uint waveBase = 0;
    if (WaveIsFirstLane() && waveCount > 0) {
        InterlockedAdd(params.compactedTraceArgs[0], waveCount, waveBase);
    }
    waveBase = WaveReadLaneFirst(waveBase);
    if (needsRay) {
        params.compactedTraceTexels[waveBase + slotInWave] = pixel.x | (pixel.y << 16);
    }
}

// ── Debug overlay ────────────────────────────────────────────────────────

// Values of `ScreenTraceConstants::debugMode` (ScreenTraceDebugMode in
// screen_trace.rs).
static const uint DEBUG_OFF = 0;
static const uint DEBUG_TRACE_TYPES = 1;   // Lumen's DEBUG_VISUALIZE_TRACE_TYPES
static const uint DEBUG_HZB_MIP = 2;       // one HZB level, log-scaled
static const uint DEBUG_PREV_COLOR = 3;    // the history color the hits sample

// Dispatched at display resolution over the overlay layer (occlusionTexture,
// which tonemap composites), like the SHARC debug overlay.
[shader("compute")]
[numthreads(8, 8, 1)]
void debugOverlay(uint3 threadId: SV_DispatchThreadID) {
    ScreenTraceConstants st = params.screenTrace;
    uint2 px = threadId.xy;
    uint2 displayDims;
    params.gbuffer.occlusionTexture.GetDimensions(displayDims.x, displayDims.y);
    if (any(px >= displayDims)) {
        return;
    }
    if (params.gbuffer.occlusionTexture[px].a == 1.0) {
        return; // keep egui on top
    }
    float2 uv = (float2(px) + 0.5) / float2(displayDims);
    uint2 screenSize = uint2(st.renderExtent);
    uint2 pixel = min(uint2(uv * float2(screenSize)), screenSize - 1);

    float3 color = float3(0, 0, 0);
    if (st.debugMode == DEBUG_TRACE_TYPES) {
        if (!isSurfacePixel(pixel)) {
            return; // sky / UI: nothing traced
        }
        TraceHit t = decodeTraceHit(params.traceHit[pixel]);
        if (t.hit) {
            color = float3(0.1, 0.8, 0.1);        // green: screen hit accepted
        } else if (t.rejected) {
            color = float3(0.9, 0.9, 0.1);        // yellow: hit rejected (history / edge)
        } else if (t.uncertain) {
            color = float3(1.0, 0.5, 0.0);        // orange: uncertain, hardware retrace
        } else if (t.distance > 0.0) {
            color = float3(0.9, 0.1, 0.1);        // red: screen miss, hardware ray
        } else {
            color = float3(0.1, 0.2, 0.9);        // blue: trace skipped
        }
    } else if (st.debugMode == DEBUG_HZB_MIP) {
        float z = sampleHzb(uv * st.hzbUvFactor, float(st.debugMip));
        float v = z > 0.0 ? saturate((log2(z) + 14.0) / 14.0) : 0.0;
        color = float3(v, v, v);
    } else if (st.debugMode == DEBUG_PREV_COLOR) {
        color = params.prevSceneColor.SampleLevel(params.sky.linearSampler, uv, 0).rgb * st.debugScale;
    } else {
        return;
    }
    params.gbuffer.occlusionTexture[px] = float4(color, 1.0);
}
