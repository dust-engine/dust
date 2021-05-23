# dust-plugin

## Architecture

There are three different parts of the plugin system:

### Plugins

Each plugin is a crate, that may depend on other crates, as well as other plugins. It also depends as a build dependency on `dust-plugin`.

For example, one plugin definition may be:

```toml
[package]
name = "my-plugin"
version = "1.0.0"
edition = "2018"
...

[dependencies]
dust-engine = { version = "1.0.0", registry = "dust-plugins" }
other-plugin = { version = "1.0.0", registry = "dust-plugins" }
other-library = "1.0.0"

[build-dependencies]
dust-plugin = "1.0.0"
```

Build.rs:

```rust
fn main() {
    dust::plugin::create::link_dependencies(); // This does the trick to make all the rlibs work.
}
```

And then, call `dust plugin build`. This will run the following steps:

```bash
cargo fetch
# Copy the rlibs into a folder. Use cargo metadata to find the place the rlibs are.
# Create lib.rs, which contains extern crates for all the plugin dependencies, and a special export for the startup method.
cargo build
# Copy rlibs in the output to the output folder.
```

### Runtime

The runtime will use the same mechanism to resolve dependencies as plugins, but will also generate via build script some combining stuff, including declaring the voxel type (combined from all the plugins), starting up the engine, etc.

### Server

The server is a combined crate registry, that uses github actions to build the plugins.
