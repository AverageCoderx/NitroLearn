//! Records a fingerprint of the Rust code the extension is built from (`nitro._core.SOURCE_HASH`), so the
//! Python side can tell when a `git pull` has changed that code since the last build
//! (`python/nitro/build_info.py` computes the same fingerprint; keep the two in step).

use std::path::{Path, PathBuf};
use std::{env, fs};

/// The crates the extension is built from, and the workspace files that pin their dependencies.
const DIRS: [&str; 2] = ["crates/nitro_core", "crates/nitro_py"];
const FILES: [&str; 2] = ["Cargo.toml", "Cargo.lock"];

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "target") {
                collect(&path, out);
            }
        } else if path
            .extension()
            .is_some_and(|ext| ext == "rs" || ext == "toml")
        {
            out.push(path);
        }
    }
}

/// FNV-1a, 64 bits.
fn feed(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("crates/nitro_py sits two folders below the workspace")
        .to_path_buf();
    let mut files = Vec::new();
    for dir in DIRS {
        // A fingerprint over missing code would never match the Python side's; better to stop here.
        assert!(
            root.join(dir).is_dir(),
            "{} is missing",
            root.join(dir).display()
        );
        println!("cargo:rerun-if-changed={}", root.join(dir).display());
        collect(&root.join(dir), &mut files);
    }
    for file in FILES {
        println!("cargo:rerun-if-changed={}", root.join(file).display());
        files.push(root.join(file));
    }
    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| {
            let relative = path.strip_prefix(&root).expect("under the workspace");
            let parts: Vec<String> = relative
                .components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect();
            (parts.join("/"), path)
        })
        .collect();
    named.sort();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for (name, path) in named {
        let text =
            fs::read(&path).unwrap_or_else(|err| panic!("reading {}: {err}", path.display()));
        let mut unix = Vec::with_capacity(text.len());
        let mut bytes = text.iter().peekable();
        while let Some(&byte) = bytes.next() {
            // Line endings as git stores them, whatever a Windows checkout turned them into.
            if byte == b'\r' && bytes.peek() == Some(&&b'\n') {
                continue;
            }
            unix.push(byte);
        }
        feed(&mut hash, name.as_bytes());
        feed(&mut hash, &[0]);
        feed(&mut hash, &unix);
        feed(&mut hash, &[0]);
    }
    println!("cargo:rustc-env=NITRO_SOURCE_HASH={hash:016x}");
}
