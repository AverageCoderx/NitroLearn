//! Compiles rewards of one's own into the core: the files listed in
//! `my-settings/rust-files.txt` (or the list `NITRO_RUST_FILES` names), each a module, every `Reward` in
//! them registered by name. `src/custom/example.rs` and every file in `src/custom/library/` (rewards that
//! come with NitroLearn) are always in.
//!
//! Cargo runs this again when the list changes, and when `rebuild.stamp` is newer than the last run:
//! `nitro rebuild` touches it, so a list made after the last build (or a listed file put back) is read.
//! A list that does not exist is not watched, since cargo would take a missing file as changed on every
//! build and compile the core again each time.

use std::path::{Path, PathBuf};
use std::{env, fs};

#[path = "src/custom/scan.rs"]
mod scan;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("crates/nitro_core sits two folders below the workspace");
    println!("cargo:rerun-if-env-changed=NITRO_RUST_FILES");
    println!("cargo:rerun-if-changed=src/custom/scan.rs");
    println!("cargo:rerun-if-changed=rebuild.stamp");
    let list = env::var_os("NITRO_RUST_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("my-settings").join("rust-files.txt"));
    if list.is_file() {
        println!("cargo:rerun-if-changed={}", list.display());
    }
    let mut paths = vec![manifest.join("src").join("custom").join("example.rs")];
    let library = manifest.join("src").join("custom").join("library");
    println!("cargo:rerun-if-changed={}", library.display());
    if let Ok(entries) = fs::read_dir(&library) {
        let mut found: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
            .collect();
        found.sort();
        paths.extend(found);
    }
    if let Ok(text) = fs::read_to_string(&list) {
        for line in text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let path = PathBuf::from(line);
            if path.is_absolute() && path.is_file() {
                paths.push(path);
            } else {
                // An update must still build when a listed file has moved: it is left out, said here.
                println!(
                    "cargo:warning={} lists {line}, which is not there (a full path is needed): left out",
                    list.display()
                );
            }
        }
    }
    let mut files = Vec::new();
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("reading {}: {err}", path.display()));
        files.push((path.to_string_lossy().into_owned(), source));
    }
    let code = scan::generate(&files).unwrap_or_else(|err| panic!("{err}"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("set by cargo")).join("custom_rewards.rs");
    fs::write(&out, code).unwrap_or_else(|err| panic!("writing {}: {err}", out.display()));
}
