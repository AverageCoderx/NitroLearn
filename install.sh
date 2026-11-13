#!/usr/bin/env bash
# Installs NitroLearn on Linux (x86-64) into this folder, and nowhere else:
#
#     bash install.sh [--torch cpu|cu126|cu128|cu130] [--cpu] [--rust] [--from-source] [--dev] [--recordings]
#                     [--bundles FOLDER]
#
# It downloads uv (a pinned version, checked by its SHA-256 hash) into .tools/uv, lets uv put its own
# Python 3.11 in .tools/python and make the private environment .venv, then tools/install_steps.py
# installs PyTorch for this computer's graphics card, NitroLearn (ready-made for this version when the
# release has it, else built from its code with Rust kept in .tools), the arena meshes, and runs
# `nitro doctor`. Nothing is installed system-wide and no shell profile, PATH or system Python is touched:
# to remove NitroLearn, delete this folder. Safe to run again. Afterwards: ./nitro.sh dashboard
#
# Before downloading, it shows a menu of the optional parts (graphics-card PyTorch, Rust for your own Rust
# rewards, developer tools, RocketSim's recordings): type a number to switch one on or off, Enter to install.
# Run again, it shows what is installed ticked and changes only what was switched. The options below set
# parts without the menu; without a terminal to answer in, nothing is asked and the parts stay as they are.
#
#   --torch        a PyTorch build other than the one picked for the card (remembered for later runs)
#   --cpu          PyTorch for the processor, not the graphics card (as --torch cpu)
#   --rust         also put Rust in this folder (needed only for your own Rust rewards)
#   --from-source  build NitroLearn from its code even when ready-made files exist (needs Rust and a C compiler)
#   --dev          developer install: tests and code checkers (builds from the code)
#   --recordings   download RocketSim's five game recordings now (119 MB), not when first needed
#   --no-launch    accepted as on Windows; nothing opens by itself here (./nitro.sh dashboard opens it)
#   --rlbot        accepted as on Windows; RLBot v5's installer is for Windows, so it says where to look
#   --bundles      a folder of the release's files (default: release-files in this folder, when there is one):
#                  install from it without downloading, except PyTorch and the arena meshes
#
# uv and Python come from NitroLearn's own release first (that folder, else the release on GitHub), and only
# then from PyPI and python-build-standalone; each is checked against its pinned hash wherever it comes from
# (the same pins as tools/bundles.py).
set -euo pipefail

UV_VERSION=0.11.32
UV_WHEEL=uv-0.11.32-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
UV_URL=https://files.pythonhosted.org/packages/90/b0/114463d056b6b328d45557001e848b8ab15539bd8f4fa7a457ccb83e2b5d/$UV_WHEEL
UV_SHA256=3da76cd4e2697de30928b8a8524bd39183ac1e08cb7e72833807c022b7cba6c4
PYTHON_VERSION=3.11
PYTHON_ASSET=python-3.11.15-linux-x64.tar.gz
PYTHON_SHA256=23ccae6f1ff73e8aa8378436f869da003b8eb7d6c95f2bc706f494115ba1447d
PYTHON_MIRROR_PATH=20260718/cpython-3.11.15+20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
RELEASES='https://github.com/AverageCoderx/NitroLearn/releases/download/v{version}/'
RELEASES=${NITRO_RELEASES:-$RELEASES}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
uv="$root/.tools/uv/uv"

say() { printf '\n==> %s\n' "$*"; }
stop() {
    printf '\nSETUP STOPPED: %s\n' "$*" >&2
    exit 1
}
# Said whenever a download fails or stalls (tools/install_steps.py says the same).
DOWNLOAD_HINT="Check the internet connection. Antivirus web shields (HTTPS scanning) and company proxies often hold back or block downloads like these: allow this folder in your antivirus (or pause its web shield for the install), then run the installer again. What was already downloaded is kept."

limited() { # limited <minutes> <what> <command...>: nothing to read on its input; stopped past the limit
    local minutes=$1 what=$2 code=0
    shift 2
    if command -v timeout > /dev/null; then
        timeout "$((minutes * 60))" "$@" < /dev/null || code=$?
    else
        "$@" < /dev/null || code=$?
    fi
    case "$code" in
        0) ;;
        124) stop "$what was stopped after $minutes minutes without finishing. $DOWNLOAD_HINT" ;;
        *) stop "$what failed (see above). $DOWNLOAD_HINT" ;;
    esac
}

download() { # download <url> <file>
    if command -v curl > /dev/null; then
        curl -fsSL --retry 4 --retry-delay 2 --connect-timeout 30 --max-time 600 -o "$2" "$1"
    elif command -v wget > /dev/null; then
        wget -q --tries=4 --timeout=60 -O "$2" "$1"
    else
        stop "curl or wget is needed to download uv: install one of them (e.g. sudo apt install curl)."
    fi
}

sha256_of() {
    if command -v sha256sum > /dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

unzip_one() { # unzip_one <zip> <member> <target>: one file out of a zip, with whatever this computer has
    if command -v unzip > /dev/null; then
        unzip -p "$1" "$2" > "$3"
    elif command -v python3 > /dev/null; then
        python3 -I -c 'import sys, zipfile; sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))' \
            "$1" "$2" > "$3"
    elif command -v bsdtar > /dev/null; then
        bsdtar -xOf "$1" "$2" > "$3"
    else
        stop "unzip is needed to unpack uv: install it (e.g. sudo apt install unzip)."
    fi
}

release_url() { # the release of this folder's version (Cargo.toml), where its files are
    local version
    version=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version *= *"\(.*\)"/\1/p' "$root/Cargo.toml" 2> /dev/null | head -n 1)
    printf '%s' "${RELEASES//\{version\}/$version}"
}

release_files="" # a folder of the release's files (--bundles, else release-files when there is one)
find_release_files() {
    local previous=""
    for arg in "$@"; do
        case "$previous" in --bundles) release_files=$arg ;; esac
        case "$arg" in --bundles=*) release_files=${arg#--bundles=} ;; esac
        previous=$arg
    done
    if [ -z "$release_files" ] && [ -d "$root/release-files" ]; then
        release_files="$root/release-files"
    fi
}

# get_pinned <name> <sha256> <file> [<original url>]: the release's copy of <name> (a folder of the release's
# files, else the release), else the original; kept only when it matches the hash. 0: got it; 1: nowhere;
# 2: what was found did not match.
get_pinned() {
    local name=$1 sha=$2 file=$3 original=${4:-} found=1 from
    if [ -n "$release_files" ] && [ -f "$release_files/$name" ]; then
        if [ "$(sha256_of "$release_files/$name")" = "$sha" ]; then
            cp "$release_files/$name" "$file"
            return 0
        fi
        found=2
    fi
    for from in "$(release_url)$name" $original; do
        download "$from" "$file" 2> /dev/null || continue
        [ "$(sha256_of "$file")" = "$sha" ] && return 0
        rm -f "$file"
        found=2
    done
    return $found
}

# uv in .tools/uv: NitroLearn's release's copy of its package, else the package on PyPI (never Astral's
# install script, which edits the PATH).
get_uv() {
    if [ -x "$uv" ] && [ "$("$uv" --version 2> /dev/null | cut -d' ' -f2)" = "$UV_VERSION" ]; then
        return 0
    fi
    if [ -n "${NITRO_UV:-}" ]; then # an offline copy of the same uv
        mkdir -p "$(dirname "$uv")"
        cp "$NITRO_UV" "$uv"
        chmod +x "$uv"
        return 0
    fi
    say "Getting uv $UV_VERSION (27 MB)"
    mkdir -p "$root/.tmp" "$(dirname "$uv")"
    local wheel="$root/.tmp/$UV_WHEEL" code=0
    get_pinned "$UV_WHEEL" "$UV_SHA256" "$wheel" "$UV_URL" || code=$?
    case "$code" in
        0) ;;
        2) stop "the uv download did not match its SHA-256 hash: nothing from it was used. Try again later." ;;
        *) stop "uv could not be downloaded from PyPI. $DOWNLOAD_HINT" ;;
    esac
    unzip_one "$wheel" "uv-$UV_VERSION.data/scripts/uv" "$uv.part"
    chmod +x "$uv.part"
    mv "$uv.part" "$uv"
    rm -f "$wheel"
}

# Python from NitroLearn's release's copy when uv has to download it: uv reads it from a folder laid out as
# python-build-standalone's releases, and checks it against the hash built into uv. Without a copy, uv
# downloads it from python-build-standalone itself.
python_mirror() {
    compgen -G "$UV_PYTHON_INSTALL_DIR/cpython-3.11*" > /dev/null && return 0
    local mirror="$root/.tmp/python-mirror"
    mkdir -p "$mirror/$(dirname "$PYTHON_MIRROR_PATH")"
    if get_pinned "$PYTHON_ASSET" "$PYTHON_SHA256" "$mirror/$PYTHON_MIRROR_PATH"; then
        export UV_PYTHON_INSTALL_MIRROR="file://${mirror// /%20}"
    else
        rm -rf "$mirror"
    fi
}

venv_is_ours() {
    local base
    base=$(sed -n 's/^home *= *//p' .venv/pyvenv.cfg 2> /dev/null | head -n 1)
    case "$base" in
        "$UV_PYTHON_INSTALL_DIR"/*) .venv/bin/python -c 'import sys' > /dev/null 2>&1 ;;
        *) return 1 ;;
    esac
}

main() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ;;
        *) stop "this installer is for Linux on x86-64 (64-bit Intel or AMD) computers; on Windows use Install NitroLearn.cmd." ;;
    esac
    cd "$root"
    # shellcheck source=tools/env.sh
    . "$root/tools/env.sh"
    find_release_files "$@"
    get_uv
    say "Python $PYTHON_VERSION (kept in .tools/python)"
    python_mirror
    limited 15 "Installing Python $PYTHON_VERSION" "$uv" python install "$PYTHON_VERSION"
    unset UV_PYTHON_INSTALL_MIRROR
    rm -rf "$root/.tmp/python-mirror"
    # A .venv that no longer runs (for example after the folder was moved), or that was made from a Python
    # outside this folder, is made again.
    if [ -e .venv ] && ! venv_is_ours; then
        say "The private environment .venv does not work here, or uses a Python outside this folder: making it again"
        rm -rf .venv
    fi
    if [ ! -e .venv ]; then
        say "Making the private environment .venv"
        limited 10 "Making .venv" "$uv" venv --seed --python "$PYTHON_VERSION" .venv
    fi
    exec .venv/bin/python -I tools/install_steps.py --uv "$uv" "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
