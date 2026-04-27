fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bazel_bin = std::path::Path::new(&manifest_dir)
        .join("bazel-bin/third_party/ffx")
        .canonicalize()
        .expect("bazel-bin/third_party/ffx not found — run `bazelisk build //third_party/ffx:ffx_lpm_cpu` first");

    println!("cargo:rustc-link-search=native={}", bazel_bin.display());
    println!("cargo:rustc-link-lib=static=ffx_lpm_cpu");
    println!(
        "cargo:rerun-if-changed={}",
        bazel_bin.join("ffx_lpm_cpu.lib").display()
    );
}
