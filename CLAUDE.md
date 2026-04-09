## Project Overview

Dust is a Rust-based 3D voxel rendering engine using physically-based ray tracing. Built on Bevy (ECS framework), Vulkan (GPU compute via pumicite/ash), and Slang (shader language compiled to SPIR-V).

## Build System

Hybrid Bazel + Cargo build. Bazel is the primary build system.

- **Build all:** `bazelisk build //...`
- **Run:** `bazelisk run //:dust`
- **Cargo build:** `cargo build` (alternative, but Bazel handles shader compilation and asset bundling)
- **Run tests:** `cargo test` (vdb crate has dev-dependency on rand for tests)

## Architecture

### Workspace Crates

- **`crates/vdb/`** (`dust_vdb`) — Hierarchical voxel spatial index. A generic tree structure with configurable depth using const generics. Handles node pooling, bit-packed storage, and tree traversal. Tree shape configurable via `hierarchy!` macro. Performance-critical: compiled with `opt-level = 3` even in dev profile.

- **`crates/pbr/`** (`dust_pbr`) — PBR ray-tracing renderer. Manages the Vulkan ray-tracing pipeline, Shader Binding Table (SBT), camera, and sky atmosphere rendering. Shaders in `crates/pbr/shaders/` are written in Slang.

- **`crates/vox/`** (`dust_vox`) — Voxel geometry, materials, and MagicaVoxel `.vox` file loading. Defines the voxel tree hierarchy as `hierarchy!(3, 3, 2, VoxLeafNode)`. Provides hit group shaders for voxel ray intersection and shadow rays.

- **`src/main.rs`** — Application entry point.

### Rendering Pipeline

1. PBR ray-tracing pass → HDR output + G-buffers (albedo, normal, depth)
2. Shadow pass → visibility
3. Tone-mapping compute pass → HDR to LDR
4. egui overlay

### Shader Compilation

Shaders are written in Slang (`.slang` files) and compiled to SPIR-V via custom Bazel rules (`slang_shader`, `slang_playout` in `//third_party:pumicite_cli.bzl`). Pipeline layouts are generated as Rust code. Shader binaries are bundled as Bazel runfiles.

### Key Dependencies

- **Bevy 0.17.0-dev** — Heavily patched from `dust-engine/bevy` fork (`release-0.17.3` branch). Multiple workspace patches in `Cargo.toml`.
- **pumicite / bevy_pumicite** — Custom Vulkan abstraction layer and Bevy integration.
- **ash** — Vulkan FFI (custom fork with patches).
- **shader-slang** — Slang compiler bindings.

### Third Party / Bazel

`third_party/` contains custom BUILD files for external C/C++ dependencies:
- `vma.BUILD.bazel` — VulkanMemoryAllocator
- `slang_rs.BUILD.bazel` — shader-slang Rust bindings
- `pumicite_cli.bzl` — Bazel macros for shader compilation rules
