fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = std::path::Path::new(&manifest_dir);

    let third_party_dir = manifest_dir
        .join("bazel-bin/third_party")
        .canonicalize()
        .expect("bazel-bin/third_party not found — run `bazelisk build //third_party:dlss_helpers` first");
    println!("cargo:rustc-link-search=native={}", third_party_dir.display());
    println!("cargo:rustc-link-lib=static=dlss_helpers");
    println!(
        "cargo:rerun-if-changed={}",
        third_party_dir.join("dlss_helpers.lib").display()
    );
}
