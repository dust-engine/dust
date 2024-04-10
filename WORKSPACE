load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_rust",
    integrity = "sha256-XT1YVJ6FHJTXBr1v3px2fV37/OCS3dQk3ul+XvfIIf8=",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.42.0/rules_rust-v0.42.0.tar.gz"],
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains()
