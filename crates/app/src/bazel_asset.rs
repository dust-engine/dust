use atomicow::CowArc;
use bevy::asset::io::AssetSource;
use bevy::asset::io::{AssetReader, AssetReaderError, PathStream, Reader, VecReader};
use runfiles::{Runfiles, rlocation};
use std::path::{Path, PathBuf};

/// An [`AssetReader`] that resolves Bazel label paths using runfiles
pub enum BazelAssetReader {
    /// Standard Bazel runfiles, used when launched via `bazel run`. Env vars or a
    /// neighboring `*.runfiles` directory tell the `runfiles` crate where to look.
    Runfiles(Runfiles),
    /// A macOS `.app` bundle produced by `macos_application` with the
    /// `@build_bazel_rules_apple//apple:use_runfiles` aspect hint. rules_apple
    /// copies data into `Contents/Resources` keyed by each file's *exec* path
    /// rather than its runfiles short-path, and drops the `_main/` repo dir and
    /// `_repo_mapping`. So source files sit at their package path while generated
    /// files sit under `bazel-out/<config>/bin/`.
    #[cfg(target_os = "macos")]
    Bundle { resources: PathBuf },
}

impl BazelAssetReader {
    pub fn new() -> Self {
        // `bazel run` provides runfiles. A bundled macOS `.app` does not, so fall
        // back to reading from the bundle's `Contents/Resources` directory.
        match Runfiles::create() {
            Ok(runfiles) => Self::Runfiles(runfiles),
            #[cfg(target_os = "macos")]
            Err(_) => Self::Bundle {
                resources: PathBuf::from(
                    objc2_foundation::NSBundle::mainBundle()
                        .resourcePath()
                        .expect("app bundle has no Resources directory")
                        .to_string(),
                ),
            },
            #[cfg(not(target_os = "macos"))]
            Err(err) => panic!("failed to locate Bazel runfiles: {err:?}"),
        }
    }

    /// Resolves a runfile-style path to a real file, or `None` if it doesn't exist.
    fn resolve(&self, path: &Path) -> Option<PathBuf> {
        match self {
            Self::Runfiles(runfiles) => rlocation!(runfiles, path),
            #[cfg(target_os = "macos")]
            Self::Bundle { resources } => {
                // Drop the leading apparent-repo component (e.g. `dust/`) that
                // rlocation paths carry; inside the bundle everything is rooted at
                // Resources with no repo prefix.
                let mut components = path.components();
                components.next();
                let rel = components.as_path();

                // Source files keep their package-relative path.
                let direct = resources.join(rel);
                if direct.exists() {
                    return Some(direct);
                }

                // Generated files live under bazel-out/<config>/bin/<package path>.
                for entry in std::fs::read_dir(resources.join("bazel-out"))
                    .ok()?
                    .flatten()
                {
                    let candidate = entry.path().join("bin").join(rel);
                    if candidate.exists() {
                        return Some(candidate);
                    }
                }
                None
            }
        }
    }
}

impl AssetReader for BazelAssetReader {
    async fn read<'a>(
        &'a self,
        path: CowArc<'a, Path>,
    ) -> Result<impl Reader + 'a, AssetReaderError> {
        let full_path = self
            .resolve(&path)
            .ok_or(AssetReaderError::NotFound(path.to_path_buf()))?;
        let bytes = std::fs::read(&full_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                AssetReaderError::NotFound(full_path)
            } else {
                e.into()
            }
        })?;
        Ok(VecReader::new(bytes))
    }

    async fn read_meta<'a>(
        &'a self,
        path: CowArc<'a, Path>,
    ) -> Result<impl Reader + 'a, AssetReaderError> {
        Err::<VecReader, _>(AssetReaderError::NotFound(path.to_path_buf()))
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<Box<PathStream>, AssetReaderError> {
        let full_path = self
            .resolve(path)
            .ok_or(AssetReaderError::NotFound(path.to_path_buf()))?;
        let entries: Vec<PathBuf> = std::fs::read_dir(&full_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    AssetReaderError::NotFound(full_path.clone())
                } else {
                    e.into()
                }
            })?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let file_name = entry.file_name();
                Some(PathBuf::from(file_name))
            })
            .collect();
        Ok(Box::new(futures_lite::stream::iter(entries)))
    }

    async fn is_directory<'a>(&'a self, path: &'a Path) -> Result<bool, AssetReaderError> {
        let full_path = self
            .resolve(path)
            .ok_or(AssetReaderError::NotFound(path.to_path_buf()))?;
        let metadata = full_path
            .metadata()
            .map_err(|_| AssetReaderError::NotFound(full_path))?;
        Ok(metadata.file_type().is_dir())
    }
}

/// Returns an [`AssetSource`] builder configured to read from a Bazel workspace's
/// `bazel-bin/` directory.
///
/// Register before `DefaultPlugins`:
/// ```ignore
/// app.register_asset_source("bazel", bazel_asset_source(workspace_root));
/// ```
pub fn bazel_asset_source() -> bevy::asset::io::AssetSourceBuilder {
    AssetSource::build().with_reader(move || Box::new(BazelAssetReader::new()))
}
