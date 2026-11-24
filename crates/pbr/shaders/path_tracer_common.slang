module path_tracer_common;

// What a hit group tells the path tracer (path_tracer.slang) about the surface
// its ray hit.
public struct [raypayload] PathTracerPayload {
    // Distance along the ray; negative if it missed.
    public float hitT : read(caller) : write(caller, closesthit, miss);
    // World space, facing the side the ray came from.
    public float3 normal : read(caller) : write(closesthit);
    // Linear.
    public float3 albedo : read(caller) : write(closesthit);
    // Radiance it emits.
    public float3 emission : read(caller) : write(closesthit);
}
