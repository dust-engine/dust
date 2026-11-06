// Cargo only: links the flecs library that Bazel built. The Bazel build ignores
// this file and links it through `link_deps`.
fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let flecs_dir = std::path::Path::new(&manifest_dir)
        .join("../../bazel-bin/external/+new_git_repository+flecs")
        .canonicalize()
        .expect("flecs not found: run `bazel build -c opt @flecs//:flecs` first");

    println!("cargo:rustc-link-search=native={}", flecs_dir.display());
    println!("cargo:rustc-link-lib=static=flecs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", flecs_dir.display());

    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        for lib in ["wsock32", "ws2_32", "dbghelp"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
