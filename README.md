# Dust Engine

The Dust Engine tries to become a powerful tool for creating immersive voxel-based games with stunning graphics. We're building the Dust Engine to explore and address some of the [big problems for voxel games](https://dust.rs/posts/12#the-big-problems-for-voxel-games). Our current prioity is createing a real-time global illumination renderer. Please see this [blog post](https://dust.rs/posts/14) for details on how it's working.

It is built on top of the awesome [Bevy Engine](https://github.com/bevyengine/bevy).

Please note that this engine is currently under development and is not ready for production use. However, we encourage you to explore and experiment with it.

## Features
- Voxel-based worlds: The engine provides the infrastructure to create and manipulate voxel-based worlds. Voxel data can be efficiently stored and rendered to create immersive game environments.
- Vulkan hardware raytracing: It leverages the power of Vulkan hardware raytracing to achieve real-time global illumination on detailed voxel scenes.
- Future-based render graphs: The `rhyolite` library offers a simple async-like interface for GPU syncronization and resource lifetime management. The rendering pipeline is designed to be highly customizable.
- Spatial Data structures: The storage and management of the scene geometry was powered by a spatial database inspired by OpenVDB, tailored for hardware ray tracing.







## Requirements
### Nightly Rust compiler
### Compatible graphics card
Your graphics card must have Vulkan 1.3 and hardware raytracing support.
Please ensure that your hardware supports the required Vulkan extensions (`VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`).
That includes the following devices:
- NVIDIA GTX1060 and above
- AMD RX6600 and above
- Intel Arc A380 and above
- AMD RDNA2+ integrated graphics. That includes all Ryzen 6000 series laptop CPUs, all Ryzen 7000 series laptop CPUs except 7030 series, and potentially Steam Deck, PS5 and Xbox.

### Compatible operating system
The engine is only substantially tested on Windows

Should also work on Linux, but it's tested to a lesser degree. If you run into any problems, please let us know on our [Discord Server](https://discord.com/invite/7R26SXn8CT).

Windows Subsystem for Linux is unsupported.

macOS support is possible in the future through MoltenVK.

## How to run
First, download and install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home). We need the GLSL shader compiler from it.

Before cloning this repository, ensure that you have [Git LFS](https://git-lfs.com/) installed. If you cloned
the repository without Git LFS, manually pull the LFS files:
```bash
git lfs fetch --all
```

Finally, compile and run the `castle` example.
```bash
cd ..
cargo run --release --example castle
```
## What if this doesn't work
If you encounter any issues while running this program, we highly encourage you to enable the `sentry` feature
when compiling the program. [Sentry](https://sentry.io) helps us identify and debug issues by providing detailed crash reports.

If you are experiencing a DEVICE_LOST error while using an NVIDIA GPU, we recommend enabling NVIDIA Aftermath to help us diagnose the issue. NVIDIA Aftermath provides additional information about GPU crashes and can assist in debugging.

```sh
cargo run --release --example castle --features sentry

# For DEVICE_LOST crashes on NVIDIA GPUs, you may also send NVIDIA Aftermath Crash Reports
cargo run --release --example castle --features sentry --features aftermath

```


## Contributing
The Dust Engine is still in very early stages of development, and we welcome and appreciate contributions of any kind.
Your input, ideas, and expertise can make a real difference in the development of this project. If you encounter any issues, have suggestions, or would like to contribute, feel free to join our [Discord server](https://discord.com/invite/7R26SXn8CT).

## License
The Voxel Game Engine is released under the MPL-2.0 License. Please review the license file for more details.

