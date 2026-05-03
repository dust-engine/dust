fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = std::path::Path::new(&manifest_dir);

    let third_party_dir = manifest_dir
        .join("../../bazel-bin/crates/denoiser")
        .canonicalize()
        .expect("bazel-bin/crates/denoiser not found — run `bazelisk build //crates/denoiser:dlss_helpers` first");
    println!(
        "cargo:rustc-link-search=native={}",
        third_party_dir.display()
    );
    println!("cargo:rustc-link-lib=static=dlss_helpers");
    println!(
        "cargo:rerun-if-changed={}",
        third_party_dir.join("dlss_helpers.lib").display()
    );
}
