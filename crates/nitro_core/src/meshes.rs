//! Loading Rocket League arena collision meshes into RocketSim (once per process).

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Number of distinct soccar arena meshes RocketSim v3 recognises by hash.
pub const SOCCAR_MESH_COUNT: usize = 16;

const MESH_EXTENSION: &str = "cmf";
const SOCCAR_SUBFOLDER: &str = "soccar";
// RocketSim refuses (by panicking) larger counts; see `CollisionMeshFile::read_from_bytes`.
const MAX_MESH_ELEMENTS: usize = 1_000_000;

/// What was found in a collision-mesh folder and what RocketSim accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshReport {
    pub folder: PathBuf,
    /// `.cmf` files present in `<folder>/soccar`.
    pub soccar_files: usize,
    /// Files RocketSim recognised as genuine soccar meshes (unknown files are skipped).
    pub soccar_recognized: usize,
}

impl MeshReport {
    /// True when every soccar mesh RocketSim knows about was loaded.
    pub fn is_complete(&self) -> bool {
        self.soccar_recognized == SOCCAR_MESH_COUNT
    }
}

#[derive(Debug)]
pub enum MeshError {
    FolderMissing(PathBuf),
    NoSoccarMeshes(PathBuf),
    Unreadable { path: PathBuf, source: io::Error },
    Corrupt { path: PathBuf, reason: String },
    AlreadyLoaded { loaded: PathBuf, requested: PathBuf },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FolderMissing(path) => {
                write!(f, "collision mesh folder {} does not exist", path.display())
            }
            Self::NoSoccarMeshes(path) => write!(
                f,
                "no .{MESH_EXTENSION} files in {}; the folder must contain a '{SOCCAR_SUBFOLDER}' \
                 subfolder with the dumped soccar arena meshes",
                path.display()
            ),
            Self::Unreadable { path, source } => {
                write!(f, "could not read {}: {source}", path.display())
            }
            Self::Corrupt { path, reason } => {
                write!(
                    f,
                    "{} is not a valid collision mesh: {reason}",
                    path.display()
                )
            }
            Self::AlreadyLoaded { loaded, requested } => write!(
                f,
                "collision meshes were already loaded from {} in this process; cannot switch to {}",
                loaded.display(),
                requested.display()
            ),
        }
    }
}

impl std::error::Error for MeshError {}

// RocketSim's initialisation is process-global and one-shot, so the folder it was given is remembered.
static LOADED: Mutex<Option<MeshReport>> = Mutex::new(None);

/// Loads the meshes in `folder` into RocketSim, or returns the existing report if that folder is
/// already loaded. Every file is validated first: RocketSim panics on malformed files, and a failed
/// initialisation cannot be retried within the same process.
pub fn load(folder: &Path) -> Result<MeshReport, MeshError> {
    let folder = absolute_folder(folder)?;
    let soccar_files = mesh_files(&folder.join(SOCCAR_SUBFOLDER))?;
    if soccar_files.is_empty() {
        return Err(MeshError::NoSoccarMeshes(folder));
    }

    let mut loaded = LOADED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(report) = loaded.as_ref().filter(|report| report.folder == folder) {
        return Ok(report.clone());
    }

    for file in &soccar_files {
        let bytes = fs::read(file).map_err(|source| MeshError::Unreadable {
            path: file.clone(),
            source,
        })?;
        validate_mesh_bytes(&bytes).map_err(|reason| MeshError::Corrupt {
            path: file.clone(),
            reason,
        })?;
    }
    if let Some(report) = loaded.as_ref() {
        return Err(MeshError::AlreadyLoaded {
            loaded: report.folder.clone(),
            requested: folder,
        });
    }

    let report = MeshReport {
        soccar_recognized: init_simulator(&folder)?,
        soccar_files: soccar_files.len(),
        folder,
    };
    *loaded = Some(report.clone());
    Ok(report)
}

/// Hands the meshes to the simulator; returns how many soccar meshes it recognised.
fn init_simulator(folder: &Path) -> Result<usize, MeshError> {
    rocketsim::init(folder, true).map_err(|source| MeshError::Unreadable {
        path: folder.to_path_buf(),
        source,
    })?;
    Ok(rocketsim::get_arena_collision_mesh_files(rocketsim::GameMode::Soccar).len())
}

/// Checks the `.cmf` layout RocketSim expects: `u32 triangle count, u32 vertex count`, then
/// `3 × u32` vertex indices per triangle, then `3 × f32` per vertex, all little-endian.
fn validate_mesh_bytes(bytes: &[u8]) -> Result<(), String> {
    let read_u32 = |offset: usize| -> Option<usize> {
        let chunk = bytes.get(offset..offset + 4)?;
        Some(u32::from_le_bytes(chunk.try_into().ok()?) as usize)
    };
    let (Some(tris), Some(verts)) = (read_u32(0), read_u32(4)) else {
        return Err("file is shorter than its header".into());
    };
    if tris == 0 || verts == 0 || tris.max(verts) > MAX_MESH_ELEMENTS {
        return Err(format!(
            "implausible size ({tris} triangles, {verts} vertices)"
        ));
    }
    let expected_len = 8 + tris * 3 * 4 + verts * 3 * 4;
    if bytes.len() < expected_len {
        return Err(format!(
            "file is truncated ({} of {expected_len} bytes)",
            bytes.len()
        ));
    }
    let bad_index = (0..tris * 3)
        .map(|i| read_u32(8 + i * 4).unwrap_or(usize::MAX))
        .find(|&index| index >= verts);
    match bad_index {
        Some(index) => Err(format!("triangle refers to vertex {index} of {verts}")),
        None => Ok(()),
    }
}

// `std::path::absolute` rather than `fs::canonicalize`: the latter adds a `\\?\` prefix on Windows,
// which would leak into every message shown to the user.
fn absolute_folder(folder: &Path) -> Result<PathBuf, MeshError> {
    if !folder.is_dir() {
        return Err(MeshError::FolderMissing(folder.to_path_buf()));
    }
    std::path::absolute(folder).map_err(|source| MeshError::Unreadable {
        path: folder.to_path_buf(),
        source,
    })
}

fn mesh_files(dir: &Path) -> Result<Vec<PathBuf>, MeshError> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(dir).map_err(|source| MeshError::Unreadable {
        path: dir.to_path_buf(),
        source,
    })?;
    Ok(entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == MESH_EXTENSION))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{meshes_loaded, test_meshes};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("nitro_core_{name}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(path.join(SOCCAR_SUBFOLDER)).unwrap();
            Self(path)
        }

        fn write_mesh(&self, name: &str, bytes: &[u8]) {
            fs::write(self.0.join(SOCCAR_SUBFOLDER).join(name), bytes).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn mesh_bytes(tris: u32, verts: u32, indices: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(tris.to_le_bytes());
        bytes.extend(verts.to_le_bytes());
        for index in indices {
            bytes.extend(index.to_le_bytes());
        }
        bytes.extend(std::iter::repeat_n(0u8, verts as usize * 12));
        bytes
    }

    #[test]
    fn mesh_layout_validation() {
        assert_eq!(validate_mesh_bytes(&mesh_bytes(1, 3, &[0, 1, 2])), Ok(()));
        assert!(validate_mesh_bytes(b"short").is_err());
        assert!(validate_mesh_bytes(&mesh_bytes(0, 3, &[])).is_err());
        assert!(validate_mesh_bytes(&mesh_bytes(1, 3, &[0, 1, 3])).is_err());
        let truncated = mesh_bytes(1, 3, &[0, 1, 2]);
        assert!(validate_mesh_bytes(&truncated[..truncated.len() - 1]).is_err());
    }

    #[test]
    fn missing_folder_is_reported() {
        let err = load(Path::new("definitely/not/a/folder")).unwrap_err();
        assert!(matches!(err, MeshError::FolderMissing(_)), "{err}");
    }

    #[test]
    fn folder_without_soccar_meshes_is_reported() {
        let dir = TempDir::new("empty");
        fs::write(
            dir.0.join(SOCCAR_SUBFOLDER).join("readme.txt"),
            b"not a mesh",
        )
        .unwrap();
        let err = load(&dir.0).unwrap_err();
        assert!(matches!(err, MeshError::NoSoccarMeshes(_)), "{err}");
    }

    // Other tests may load the same meshes first; every step below holds in either order.
    #[test]
    fn rejects_corrupt_files_then_loads_once_and_refuses_a_different_folder() {
        let corrupt = TempDir::new("corrupt");
        corrupt.write_mesh("mesh_0.cmf", b"not a mesh at all");
        let err = load(&corrupt.0).unwrap_err();
        assert!(matches!(err, MeshError::Corrupt { .. }), "{err}");

        if !meshes_loaded() {
            return;
        }
        let meshes = test_meshes();
        let report = load(&meshes).unwrap();
        assert!(report.is_complete(), "{report:?}");
        assert_eq!(report.soccar_files, SOCCAR_MESH_COUNT);
        assert_eq!(load(&meshes).unwrap(), report);

        let other = TempDir::new("other");
        other.write_mesh("mesh_0.cmf", &mesh_bytes(1, 3, &[0, 1, 2]));
        let err = load(&other.0).unwrap_err();
        assert!(matches!(err, MeshError::AlreadyLoaded { .. }), "{err}");
    }
}
