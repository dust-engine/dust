# Dust Engine

The Dust Engine tries to become a powerful tool for creating immersive voxel-based games with stunning graphics.

We are built on top of the awesome [Bevy Engine](https://github.com/bevyengine/bevy).

Please note that this engine is currently under development and is not ready for production use. However, we encourage you to explore and experiment with it.

## Features
- Voxel-based worlds: The engine provides the infrastructure to create and manipulate voxel-based worlds. Voxel data can be efficiently stored and rendered to create immersive game environments.
- Vulkan hardware raytracing: The Voxel Game Engine leverages the power of Vulkan hardware raytracing to achieve stunning visual effects.
- Future-based render graphs: The `rhyolite` library offers a simple async-like interface for GPU syncronization and resource lifetime
management. The rendering pipeline is designed to be highly customizable.


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

### Compatible operating system.
The engine currently supports Windows only.

Linux support is possible but untested. Users probably have to manually add the corresponding swapchain extensions.

macOS support is possible in the future through MoltenVK.

## How to run
First, download and install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

Before cloning the the repository, ensure that you have Git LFS installed. If you cloned
the repository without Git LFS, manually pull the LFS files:
```bash
git lfs fetch --all
```

Next, compile the example shaders. This shouldn't be needed once Bevy
completes their asset system [overhaul](https://github.com/bevyengine/bevy/discussions/3972)
and added asset preprocessing capabilities.

```bash
cd assets
./compile.sh
# or
./compile.bat
```

Finally, compile and run the `castle` example.
```bash
cd ..
cargo run --release --example castle
```

## Contributing
The Dust Engine is still in very early stages of development, and we welcome and appreciate contributions of any kind.
Your input, ideas, and expertise can make a real difference in the development of this project. If you encounter any issues, have suggestions, or would like to contribute, feel free to join our [Discord server](https://discord.gg/5eUGuQuX).

## License
The Voxel Game Engine is released under the MPL-2.0 License. Please review the license file for more details.

