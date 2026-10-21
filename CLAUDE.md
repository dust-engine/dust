## Project Overview

Dust is a Rust-based 3D voxel rendering engine using physically-based ray tracing. Built on Bevy (ECS framework), Vulkan (GPU compute via pumicite/ash), and Slang (shader language compiled to SPIR-V).

## Build System

Hybrid Bazel + Cargo build. Bazel is the primary build system.

- **Build all:** `bazel build -c opt  //...`
- **Run:** `bazel run -c opt //:dust`
- **Cargo build:** `cargo build` (alternative, but Bazel handles shader compilation and asset bundling)
  - Build rust dependencies with bazel first
    `bazel build //third_party/ffx:ffx_lpm_cpu //crates/pbr/shaders:pbr_module_layout_macro //:runtime_assets`
- **Run tests:** `cargo test` (vdb crate has dev-dependency on rand for tests)

### pumicite

We own pumicite (`dust-engine/pumicite`, local checkout at `../pumicite`), so fix bugs there rather than working around them in dust. Committed `Cargo.toml` must pin `pumicite` to a pushed commit hash (`rev`). To develop against local pumicite changes, add a `[patch]` section:

```toml
[patch."https://github.com/dust-engine/pumicite.git"]
pumicite = { path = "../pumicite" }
bevy_pumicite = { path = "../pumicite/crates/bevy" }
pumicite_egui = { path = "../pumicite/crates/egui" }
pumicite_types = { path = "../pumicite/crates/types" }
```

Keep `pumicite_types` in the list; Bazel's crate_universe splice fails without it. While the `[patch]` is in, build and test with cargo only, not Bazel: a Bazel repin against the local path rewrites `MODULE.bazel.lock` and writes generated `BUILD.bazel` files into `../pumicite`. Before committing dust, push the pumicite commit, bump the three `rev`s, and remove the `[patch]`.

## Architecture

### Workspace Crates

- **`crates/vdb/`** (`dust_vdb`) — Hierarchical voxel spatial index. A generic tree structure with configurable depth using const generics. Handles node pooling, bit-packed storage, and tree traversal. Tree shape configurable via `hierarchy!` macro. Performance-critical: compiled with `opt-level = 3` even in dev profile.

- **`crates/pbr/`** (`dust_pbr`) — PBR ray-tracing renderer. Manages the Vulkan ray-tracing pipeline, Shader Binding Table (SBT), camera, and sky atmosphere rendering. Shaders in `crates/pbr/shaders/` are written in Slang.

- **`crates/vox/`** (`dust_vox`) — Voxel geometry, materials, and MagicaVoxel `.vox` file loading. Defines the voxel tree hierarchy as `hierarchy!(3, 3, 2, VoxLeafNode)`. Provides hit group shaders for voxel ray intersection and shadow rays.

- **`crates/ui/`** (`dust_ui`) — `BevyView`, the GPUI view the game runs in, and the thread split. GPUI (`gpui-pre`) owns the main thread and draws the UI into its own `CAMetalLayer`; `BevyView` creates the Bevy app on a spawned thread, with the `Swapchain` resource, and has a closure build the rest. The app draws into a second `CAMetalLayer`, in a view under GPUI's. A `CAMetalDisplayLink` on that layer updates the Bevy app once per display frame, and the game renders straight into the link's drawables, imported into Vulkan (`VK_EXT_metal_objects`) as the frame's `SwapchainImage`; there is no Vulkan surface or swapchain. All macOS code is in `src/apple/`. Each thread runs code on the other by queueing closures: `UiHandle` (a Bevy resource) runs them with GPUI's `App`, `BevyHandle` (a GPUI global) with Bevy's `World`; `BevyHandle::run_every_frame` runs a system every frame until its handle is dropped (see `bridge.rs`). macOS only.

- **`crates/input/`** (`dust_input`) — The game's keyboard and mouse, read on the Bevy thread from the platform (GameController on macOS) and written as `bevy_input` messages, only while the `CursorMode` state is attached (for gameplay); while it is detached (for the UI) the backend doesn't read, and keys held at detach are released. `BevyView` sets it: a click on the game attaches the cursor, Escape and window deactivation detach it.

- **`crates/devtools/`** (`dust_devtools`) — DevTools dock and overlays as GPUI views. A view shows the world with `watch` (a system run every Bevy frame while the view is drawn) or `read` (once, when it is shown), and edits it through `dust_ui`'s `BevyHandle` (see `devtools.rs`).

- **`src/main.rs`** — Application entry point.

### Rendering Pipeline

1. PBR ray-tracing pass → HDR output + G-buffers (albedo, normal, depth)
2. Shadow pass → visibility
3. Tone-mapping compute pass → HDR to LDR
4. GPUI overlay, drawn and presented separately in its own layer

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
