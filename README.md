# Dust Engine

Dust engine is a voxel game engine built on top of the awesome [Bevy Engine](https://github.com/bevyengine/bevy).
It is still a work in progress.

We currently supports Windows only. Running on Linux is untested, but you will probably have to
manually add the corresponding swapchain extensions.

## How to run
```rs
# First, compile the example shaders. This shouldn't be needed once Bevy
completes their asset system [overhaul](https://github.com/bevyengine/bevy/discussions/3972)
and added asset preprocessing capabilities.

cd assets
./compile.sh

cd ..
cargo run --release --example castle
```
