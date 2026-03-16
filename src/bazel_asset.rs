use atomicow::CowArc;
use bevy::asset::io::AssetSource;
use bevy::asset::io::{AssetReader, AssetReaderError, PathStream, Reader, VecReader};
use std::path::{Path, PathBuf};

/// An [`AssetReader`] that resolves Bazel label paths against `bazel-bin/`.
///
/// Paths use Bazel label syntax: `//package:target`. In bevy asset path form
/// this becomes `"bazel://package:target"` — the `bazel` source prefix
/// naturally supplies the leading `//`.
///
/// The `:` separates the package directory from the target (file) name:
/// - `bazel://crates/pbr/shaders:pbr.spv` → `bazel-bin/crates/pbr/shaders/pbr.spv`
pub struct BazelAssetReader {
    bazel_bin: PathBuf,
}

impl BazelAssetReader {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            bazel_bin: workspace_root.join("bazel-bin"),
        }
    }

    /// Convert a Bazel label path (`package:target`) into a filesystem path
    /// under `bazel-bin/` (`bazel-bin/package/target`).
    fn resolve(&self, label: &Path) -> PathBuf {
        let s = label.to_string_lossy();
        let rel = if let Some((package, target)) = s.split_once(':') {
            PathBuf::from(package).join(target)
        } else {
            label.to_path_buf()
        };
        self.bazel_bin.join(rel)
    }
}

impl AssetReader for BazelAssetReader {
    async fn read<'a>(
        &'a self,
        path: CowArc<'a, Path>,
    ) -> Result<impl Reader + 'a, AssetReaderError> {
        let full_path = self.resolve(&path);
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
        let full_path = self.resolve(path);
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
        let full_path = self.resolve(path);
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
pub fn bazel_asset_source(workspace_root: PathBuf) -> bevy::asset::io::AssetSourceBuilder {
    AssetSource::build()
        .with_reader(move || Box::new(BazelAssetReader::new(workspace_root.clone())))
}
