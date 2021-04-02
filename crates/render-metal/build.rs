use std::env;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

const MACOS_TARGET_VERSION: &str = "10.15";

mod swift {
    use serde::Deserialize;
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SwiftRuntimePaths {
        pub runtime_library_paths: Vec<String>,
        pub runtime_library_import_paths: Vec<String>,
        pub runtime_resource_path: String,
    }
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SwiftTarget {
        triple: String,
        unversioned_triple: String,
        module_triple: String,
        swift_runtime_compatibility_version: String,
        libraries_require_r_path: bool,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SwiftTargetInfo {
        pub target: SwiftTarget,
        pub paths: SwiftRuntimePaths,
    }
}

/// Builds mac_ddc library Swift project
fn build_mac() {
    let build_profile = env::var("PROFILE").unwrap();
    let build_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let mut swift_build_dir = Path::new(&env::var("OUT_DIR").unwrap()).to_path_buf();
    swift_build_dir.push("swift");
    let swift_build_dir = swift_build_dir.into_os_string();

    if !Command::new("swift")
        .args(&[
            OsStr::new("build"),
            OsStr::new("-c"),
            OsStr::new(&build_profile),
            OsStr::new("--build-path"),
            &swift_build_dir,
        ])
        .current_dir(".")
        .status()
        .unwrap()
        .success()
    {
        panic!("Swift library mac_ddc compilation failed")
    }

    println!(
        "cargo:rustc-link-search=native={}/{}-apple-macosx/{}",
        swift_build_dir.to_string_lossy(),
        build_arch,
        build_profile
    ); // Add the swift output dir to Rust library search path
    println!("cargo:rustc-link-lib=static=DustRender"); // Link the DustRender Library
                                                        // Linking Swift dynamic libraries
    let target = format!(
        "{}-apple-macosx{}",
        env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
        MACOS_TARGET_VERSION
    );
    let swift_target_info_str = Command::new("swift")
        .args(&["-target", &target, "-print-target-info"])
        .output()
        .unwrap()
        .stdout;
    let swift_target_info: swift::SwiftTargetInfo =
        serde_json::from_slice(&swift_target_info_str).unwrap();
    for path in swift_target_info.paths.runtime_library_paths.iter() {
        println!("cargo:rustc-link-search=native={}", path);
    }
}

fn main() {
    println!("cargo:rerun-if-changed=./Sources");
    println!("cargo:rerun-if-changed=./Package.swift");
    let target = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target == "macos" {
        build_mac();
    }
}
