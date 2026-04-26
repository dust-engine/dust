use atomicow::CowArc;
use bevy::asset::io::AssetSource;
use bevy::asset::io::{AssetReader, AssetReaderError, PathStream, Reader, VecReader};
use runfiles::{Runfiles, rlocation};
use std::path::{Path, PathBuf};

/// An [`AssetReader`] that resolves Bazel label paths using runfiles
pub struct BazelAssetReader {
    runfiles: Runfiles,
}

impl BazelAssetReader {
    pub fn new() -> Self {
        Self {
            runfiles: Runfiles::create().unwrap(),
        }
    }
}

impl AssetReader for BazelAssetReader {
    async fn read<'a>(
        &'a self,
        path: CowArc<'a, Path>,
    ) -> Result<impl Reader + 'a, AssetReaderError> {
        let full_path = rlocation!(self.runfiles, &path)
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
        let full_path = rlocation!(self.runfiles, &path)
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
        let full_path = rlocation!(self.runfiles, &path)
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
