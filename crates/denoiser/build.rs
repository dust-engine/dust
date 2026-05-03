// Cargo build wires denoiser against the same C artifacts Bazel uses:
//   - dlss_helpers : built by `bazelisk build //crates/denoiser:dlss_helpers`
//                    (C shim around the static-inline DLSSD helper macros)
//   - nvsdk_ngx_d / libnvsdk_ngx : NVIDIA NGX static library, fetched by
//                    Bazel into the external dlss repo
//
// Both must be linked; the NGX lib supplies the
// `NVSDK_NGX_VULKAN_*` entry points referenced from `dlss::sys`.

fn main() {
    let bazel_bin_dir = std::env::var("BAZEL_BIN").unwrap();
    let bazel_bin_dir = std::path::Path::new(&bazel_bin_dir);

    // dlss_helpers (built by Bazel into bazel-bin/crates/denoiser)
    let helpers_dir = bazel_bin_dir
        .join("crates/denoiser")
        .canonicalize()
        .expect("bazel-bin/crates/denoiser not found — run `bazelisk build //crates/denoiser:dlss_helpers` first");
    println!("cargo:rustc-link-search=native={}", helpers_dir.display());
    println!("cargo:rustc-link-lib=static=dlss_helpers");
    let helpers_lib_name = if cfg!(target_os = "windows") {
        "dlss_helpers.lib"
    } else {
        "libdlss_helpers.a"
    };
    println!(
        "cargo:rerun-if-changed={}",
        helpers_dir.join(helpers_lib_name).display()
    );

    // NGX SDK static library — fetched by Bazel into the external dlss repo
    // under a Bzlmod-mangled directory name. Resolve via the bazel-dust
    // convenience symlink (workspace name is "dust").
    let dlss_repo = bazel_bin_dir
        .join("../bazel-dust/external/+new_git_repository+dlss")
        .canonicalize()
        .expect(
            "bazel-dust/external/+new_git_repository+dlss not found — \
             run a `bazelisk build` (e.g. //crates/denoiser:dlss_helpers) to fetch the DLSS repo",
        );

    #[cfg(target_os = "windows")]
    {
        let ngx_dir = dlss_repo.join("lib/Windows_x86_64/x64");
        println!("cargo:rustc-link-search=native={}", ngx_dir.display());
        // nvsdk_ngx_d.lib pairs with the dynamic CRT (matches Rust MSVC default).
        println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
        println!(
            "cargo:rerun-if-changed={}",
            ngx_dir.join("nvsdk_ngx_d.lib").display()
        );
    }

    #[cfg(target_os = "linux")]
    {
        let ngx_dir = dlss_repo.join("lib/Linux_x86_64");
        println!("cargo:rustc-link-search=native={}", ngx_dir.display());
        println!("cargo:rustc-link-lib=static=nvsdk_ngx");
        println!(
            "cargo:rerun-if-changed={}",
            ngx_dir.join("libnvsdk_ngx.a").display()
        );
    }
}
