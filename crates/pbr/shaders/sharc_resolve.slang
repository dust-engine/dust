// The radiance cache's resolve pass (sharc.slang), one thread per entry: adds
// this frame's samples to the average over the frames, and drops the cells that
// have gone without any for too long.

import sharc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform SharcConstants constants) {
    SharcResolveParameters resolve;
    resolve.cameraPositionPrev = constants.previousCameraPosition;
    resolve.accumulationFrameNum = constants.accumulationFrames;
    resolve.staleFrameNumMax = constants.staleFrames;
    // Read only by features that are off (sharc.slang).
    resolve.responsiveFrameNum = 0;
    resolve.frameIndex = 0;
    SharcResolveEntry(id.x, constants.parameters(), resolve);
}
