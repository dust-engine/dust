# Dust

A Rust-based 3D voxel rendering engine using physically-based ray tracing. Built on [Bevy](https://bevyengine.org/) (ECS), Vulkan ray tracing via [pumicite](https://github.com/dust-engine/pumicite).

## Features

- Hardware ray-traced rendering of sparse voxel scenes
- Hierarchical voxel spatial index (VDB-style) with const-generic configurable tree shape
- MagicaVoxel `.vox` file loading
- HDR output with tone-mapping and sky atmosphere
- DLSS upscaling on supported NVIDIA GPUs
- egui debug overlay

## Requirements

- A GPU and driver supporting `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure`
- Rust nightly
- [Bazelisk](https://github.com/bazelbuild/bazelisk) — Bazel is the primary build system, since it drives Slang shader compilation and asset bundling

## Building

```sh
# Build everything
bazelisk build -c opt //...

# Run the demo
bazelisk run -c opt //:dust
```

Cargo also works for code-only builds (`cargo build`, `cargo test`), but Bazel is required to run at least once to compile shaders and bundle runtime assets. Tests for the `dust_vdb` crate run under Cargo.

## Workspace Layout

| Crate | Purpose |
| --- | --- |
| `crates/vdb` (`dust_vdb`) | Hierarchical voxel spatial index. Generic tree with configurable depth via the `hierarchy!` macro. Bit-packed node storage and pooling. Compiled at `opt-level = 3` even in dev. |
| `crates/vox` (`dust_vox`) | Voxel geometry, materials, palettes, and `.vox` loading. Defines the tree as `hierarchy!(3, 3, 2, VoxLeafNode)` and provides hit-group shaders for ray intersection. |
| `crates/pbr` (`dust_pbr`) | PBR ray-tracing renderer: pipeline, Shader Binding Table, camera, sky atmosphere. Shaders in `crates/pbr/shaders/` (Slang). |
| `crates/denoiser` (`dust_denoiser`) | Pluggable denoiser/upscaler backends. Currently DLSS via NVIDIA NGX. |
| `crates/app` (`dust_app`) | Application-level Bevy plugin glue. |
| `src/main.rs` | Demo entry point — loads a castle scene, an orbiting teapot, and an animated rainbow. |

## Rendering Pipeline

1. PBR ray-tracing pass — HDR color output and G-buffers (albedo, normal, depth, motion vectors)
2. Shadow pass — visibility
3. Tone-mapping compute pass — HDR to LDR
4. egui overlay

## Shaders

Shaders are written in Slang (`.slang`) and compiled to SPIR-V through custom Bazel rules (`slang_shader`, `slang_playout` in `third_party/pumicite_cli.bzl`). Pipeline layouts are generated as Rust source. Compiled binaries are bundled as Bazel runfiles.

## Third Party

`third_party/` contains custom `BUILD.bazel` files for external C/C++ dependencies: VulkanMemoryAllocator, shader-slang Rust bindings, and the pumicite shader-compilation macros. Bevy is patched to the [`dust-engine/bevy`](https://github.com/dust-engine/bevy) `release-0.17.3` fork; `ash` is patched to a fork that exposes additional Vulkan metadata.
