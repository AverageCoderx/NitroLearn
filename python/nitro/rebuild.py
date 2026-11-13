"""`nitro rebuild`: builds NitroLearn's compiled core again, with the Rust reward files listed in
`my-settings/rust-files.txt` (the dashboard's Resources page writes the list and
starts this). A few minutes; a mistake in a file of one's own is reported with its file and line."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from nitro import toolchain

LIST = Path("my-settings") / "rust-files.txt"
# Newer than the core's last build: the build reads the list again (crates/nitro_core/build.rs).
STAMP = Path("crates") / "nitro_core" / "rebuild.stamp"


def listed(root: Path) -> list[Path]:
    """The reward files the next build compiles in (besides the example, always built in)."""
    try:
        text = (root / LIST).read_text(encoding="utf-8")
    except OSError:
        return []
    return [Path(line.strip()) for line in text.splitlines() if line.strip() and not line.startswith("#")]


def write_list(root: Path, files: list[Path]) -> Path:
    """Lists `files` (full paths) for the next build."""
    target = root / LIST
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Rust reward files compiled into NitroLearn by `nitro rebuild` (the dashboard writes this)."]
    lines += [str(path.resolve()) for path in files]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def set_aside_loaded_core(folder: Path) -> list[tuple[Path, Path]]:
    """Windows will not overwrite a compiled core a running program has loaded (the dashboard, a training
    run), but lets it be renamed: the old file moves aside so the build can write the new one; the running
    program keeps the old one until it closes. Old set-aside files go when nothing uses them any more.
    Returns the moves, for `put_back` if the build fails."""
    if sys.platform != "win32":
        return []
    for old in folder.glob("_core*.pyd.old-*"):
        with contextlib.suppress(OSError):  # still loaded somewhere: next time
            old.unlink()
    moved = []
    for core in folder.glob("_core*.pyd"):
        aside = core.with_name(f"{core.name}.old-{time.strftime('%Y%m%d%H%M%S')}")
        with contextlib.suppress(OSError):  # the build says so if this was the problem
            core.rename(aside)
            moved.append((core, aside))
    return moved


def put_back(moved: list[tuple[Path, Path]]) -> None:
    """After a failed build: the set-aside core goes back where nothing new was written."""
    for core, aside in moved:
        if not core.exists():
            with contextlib.suppress(OSError):
                aside.rename(core)


def build_command(python: str = sys.executable) -> list[str]:
    """How the core is built again. In a venv with maturin (a dependency of NitroLearn): only the core,
    in place next to the Python code, installing nothing (reinstalling would replace nitro.exe, which Windows
    refuses while it runs this). Otherwise pip, or uv for a venv without pip, without downloading the build
    tool again when it is already installed (so it works offline)."""
    has_pip = subprocess.run([python, "-m", "pip", "--version"], capture_output=True).returncode == 0
    if sys.prefix != sys.base_prefix and importlib.util.find_spec("maturin") is not None:
        uv = [] if has_pip or not shutil.which("uv") else ["--uv"]  # maturin looks for pip even so
        return [python, "-m", "maturin", "develop", "--release", "--skip-install", *uv]
    if not has_pip and shutil.which("uv"):
        return ["uv", "pip", "install", "--python", python, "--editable", ".", "--no-build-isolation"]
    command = [python, "-m", "pip", "install", "--editable", "."]
    if importlib.util.find_spec("maturin") is not None:
        command.append("--no-build-isolation")
    return command


def build_environment() -> dict[str, str]:
    """This Python's programs folder first on the PATH (maturin and pip start programs of their own by name,
    and nitro.exe runs without the venv activated), and this venv named as the one built for."""
    env = toolchain.cargo_env()  # the folder's own Rust, as `nitro doctor` finds it
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    if sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
        env.pop("CONDA_PREFIX", None)  # maturin refuses to choose between the two
    return env


def compiler_found() -> bool:
    """A C/C++ compiler for the C code some Rust libraries build: Microsoft's build tools on Windows (found
    with Visual Studio's own vswhere), a `cc` elsewhere."""
    if sys.platform != "win32":
        return any(shutil.which(name) for name in ("cc", "gcc", "clang"))
    vswhere = (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.is_file():
        return False
    done = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0 and bool(str(done.stdout).strip())


def missing_tools(has_cargo: bool | None = None, has_compiler: bool | None = None) -> list[str]:
    """What to install for a build, in plain words: Rust (into NitroLearn's folder) and the C/C++ build
    tools (a system install the installer never makes). Looked for on this PC unless given."""
    from nitro.doctor import ADD_RUST, find_cargo

    if has_cargo is None:
        has_cargo = find_cargo() is not None
    if has_compiler is None:
        has_compiler = compiler_found()
    missing = []
    if not has_cargo:
        missing.append(f"Rust, which goes into NitroLearn's folder: {ADD_RUST}.")
    if not has_compiler:
        missing.append(
            "Microsoft's C++ build tools: install 'Build Tools for Visual Studio 2022' from "
            "https://visualstudio.microsoft.com/downloads/ and tick 'Desktop development with C++'. "
            "NitroLearn's installer never installs them, because they change Windows itself."
            if sys.platform == "win32"
            else "a C compiler: your system's build tools (on Ubuntu or Debian: "
            "sudo apt install build-essential)."
        )
    return missing


def rebuild(root: Path) -> int:
    files = listed(root)
    names = ", ".join(path.name for path in files) or "no reward files of your own (the example only)"
    print(f"Building NitroLearn with {names}. This takes a few minutes ...", flush=True)
    missing = [path for path in files if not path.is_file()]
    if missing:
        print(f"Not found: {', '.join(map(str, missing))}. Put them back or take them off the list.")
        return 1
    with contextlib.suppress(OSError):  # without it, a list made since the last build may be missed
        (root / STAMP).touch()
    moved = set_aside_loaded_core(root / "python" / "nitro")
    if subprocess.run(build_command(), cwd=root, env=build_environment(), check=False).returncode != 0:
        put_back(moved)
        needed = missing_tools()
        if needed:
            print("Building your own Rust rewards needs what this PC does not have yet:")
            for text in needed:
                print(f"- {text}")
            print("Then start the build again.")
            return 1
        print(
            "The build failed: the messages above say why. For a file of your own they name its file and "
            "line; the guide (docs/your_own_rewards.md) shows how a reward file is written."
        )
        return 1
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            # The library's rewards (custom/library) always come with it: only one's own files are named.
            "import nitro._core as c, re; print(', '.join(r[0] for r in c.custom_rewards() "
            "if not re.search(r'custom[/\\\\]library[/\\\\]', r[3]) and not r[3].endswith('example.rs')))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    built = check.stdout.strip() or "none (the rewards from your bots always come with it)"
    print(f"Built. Rewards from your own Rust files in NitroLearn now: {built}.")
    print("Close the dashboard and open it again to use them; bots training now carry on with the old build.")
    return 0
