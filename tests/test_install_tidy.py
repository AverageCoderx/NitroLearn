"""Tidying up after an older version (tools/install_tidy.py): an upgrade removes the packages, Pythons, caches
and build files the new version no longer uses, and nothing else. Every test builds its own fake folder: a
.venv's site-packages with packages (PyTorch and NVIDIA's libraries among them), .tools/python with an older
and a newer Python, and the user's folders, which must come through untouched."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_tidy", PROJECT / "tools" / "install_tidy.py")
assert spec is not None and spec.loader is not None
tidy = importlib.util.module_from_spec(spec)
sys.modules["install_tidy"] = tidy  # its dataclasses look their module up
spec.loader.exec_module(tidy)

LINUX = {
    "os_name": "posix",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "6.1",
    "platform_system": "Linux",
    "platform_version": "#1",
    "python_version": "3.11",
    "python_full_version": "3.11.15",
    "implementation_name": "cpython",
    "implementation_version": "3.11.15",
}
WINDOWS = LINUX | {
    "os_name": "nt",
    "sys_platform": "win32",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
}
USER_FOLDERS = ("runs", "bots", "my-settings", "saves", "replays", "packs", "ab", "wandb")
NVIDIA = 'platform_system == "Linux" and platform_machine == "x86_64"'


def dist(
    site: Path, name: str, version: str = "1.0", requires: Iterable[str] = (), installer: str | None = "uv"
) -> None:
    """A package as an installer leaves it in site-packages: its .dist-info with METADATA (and INSTALLER)."""
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    extras = dict.fromkeys(m for line in requires for m in re.findall(r'extra == "([^"]+)"', line))
    lines += [f"Provides-Extra: {extra}" for extra in extras]
    lines += [f"Requires-Dist: {line}" for line in requires]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if installer is not None:
        (info / "INSTALLER").write_text(installer + "\n", encoding="utf-8")


def nitrolearn_needs(torch: bool = True, old_helper: bool = False) -> list[str]:
    needs = ["numpy>=2", 'wandb>=0.19; extra == "wandb"', 'pywebview>=5.3; extra == "dashboard"']
    needs += ['pytest>=9; extra == "dev"', 'pyarrow>=15; extra == "replays"']
    if torch:
        needs.insert(0, "torch>=2.8")
    if old_helper:
        needs.append("old-helper>=0.3")
    return needs


def new_nitrolearn(site: Path, torch: bool = True, version: str = "1.0.3") -> None:
    """The newer version's install: NitroLearn of `version` replaces the one in .venv."""
    for info in site.glob("nitrolearn-*.dist-info"):
        shutil.rmtree(info)
    dist(site, "nitrolearn", version, nitrolearn_needs(torch))


def old_release(site: Path, torch_build: str = "cu126", nitrolearn_needs_torch: bool = True) -> None:
    """What an older NitroLearn's install leaves in .venv (Linux, graphics-card PyTorch)."""
    dist(site, "nitrolearn", "1.0.2", nitrolearn_needs(nitrolearn_needs_torch, old_helper=True))
    cuda = "cu12" if torch_build == "cu126" else "cu13"
    dist(
        site,
        "torch",
        f"2.14.1+{torch_build}",
        [
            "filelock",
            "sympy>=1.13.3",
            "typing-extensions>=4.10.0",
            f"nvidia-cublas-{cuda}==12.6.4.1; {NVIDIA}",
            f"nvidia-cudnn-{cuda}==9.5.1.17; {NVIDIA}",
            'triton==3.5.0; platform_system == "Linux"',
            'opt-einsum>=3.3; extra == "opt-einsum"',
        ],
    )
    dist(site, f"nvidia-cublas-{cuda}", "12.6.4.1")
    dist(site, f"nvidia-cudnn-{cuda}", "9.5.1.17", [f"nvidia-cublas-{cuda}"])
    dist(site, "triton", "3.5.0")
    dist(site, "sympy", "1.14.0", ["mpmath<1.4,>=1.1.0"])
    dist(site, "mpmath", "1.3.0")
    dist(site, "filelock", "3.20.0")
    dist(site, "typing-extensions", "4.15.0")
    dist(site, "numpy", "2.3.4")
    dist(site, "wandb", "0.22.0", ["requests<3,>=2.0.0", "typing-extensions>=4.4; python_version < '3.12'"])
    dist(site, "requests", "2.32.5", ["urllib3<3,>=1.21.1", 'PySocks!=1.5.7,>=1.5.6; extra == "socks"'])
    dist(site, "urllib3", "2.5.0")
    dist(site, "PySocks", "1.7.1")  # only for requests[socks], which nothing asks for: it goes
    dist(site, "pywebview", "6.1", ["bottle", "proxy-tools", 'QtPy; extra == "qt"'])
    dist(site, "bottle", "0.13.4")
    dist(site, "proxy_tools", "0.1.0")
    dist(site, "QtPy", "2.4.3", ["packaging"])
    dist(site, "pyarrow", "21.0.0")
    dist(site, "pytest", "9.1.0", ["pluggy<2,>=1.5", "iniconfig"])  # a developer install, switched off since
    dist(site, "pluggy", "1.6.0")
    dist(site, "iniconfig", "2.1.0")
    dist(site, "maturin", "1.15.0")
    for seed in ("pip", "setuptools"):
        dist(site, seed, "26.0")
    dist(site, "wheel", "0.48.0", ["packaging>=24.0"])
    dist(site, "packaging", "26.3")
    dist(site, "old-helper", "0.3")  # the older version's own package: the newer one no longer uses it
    dist(site, "cowsay", "6.1", ["six"], installer="pip")  # installed by hand: kept, with what it needs
    dist(site, "six", "1.17.0")


def first_record(site: Path, env: dict[str, str] = LINUX) -> set[str]:
    """The record the installer makes before it installs over a folder an older installer set up."""
    return set(tidy.first_record(tidy.installed([site]), env))


def roots(torch: bool = True, extras: str = "wandb,replays,dashboard") -> dict[str, list[str]]:
    """What install_steps.wanted_packages names on Linux (no release copy of the packages)."""
    wanted = {"nitrolearn": extras.split(","), "maturin": [], "pywebview": ["qt"]}
    return wanted | ({"torch": []} if torch else {})


def removed(
    site: Path, wanted: dict[str, list[str]], ours: set[str], env: dict[str, str] = LINUX
) -> set[str]:
    return set(tidy.unneeded(wanted, tidy.installed([site]), env, "nitrolearn", ours).remove)


# ---------- markers and requirements ----------


@pytest.mark.parametrize(
    ("marker", "holds"),
    [
        ('sys_platform == "linux"', True),
        ('sys_platform == "win32"', False),
        ("python_version < '3.12'", True),
        ("python_version >= '3.12'", False),
        ('python_version > "3.9"', True),  # compared as versions, not as text
        ('python_full_version >= "3.11.4"', True),
        ('python_version == "3.11.*"', True),
        ('python_version ~= "3.10"', True),
        ('python_version ~= "3.12"', False),
        (NVIDIA, True),
        ('platform_system == "Linux" and (platform_machine == "aarch64" or os_name == "nt")', False),
        ('platform_system == "Linux" and (platform_machine == "aarch64" or os_name == "posix")', True),
        ('"linux" in sys_platform', True),
        ('"win" not in sys_platform', True),
        ('implementation_name == "pypy"', False),
        ('extra == "dev"', False),
        ("platform_machine === 'x86_64'", True),
        ("no_such_name == 'x'", True),  # unreadable: counts as holding (keeping is safe)
        ("sys_platform ==", True),
        ("(sys_platform == 'win32'", True),
    ],
)
def test_markers_are_read_as_python_packaging_reads_them(marker: str, holds: bool) -> None:
    assert tidy.marker_holds(marker, LINUX | {"extra": ""}) is holds


def test_requirements_and_extras() -> None:
    assert tidy.requirement("Py_Socks[Extra_One, two] (>=1.0) ; extra == 'socks'") == (
        "py-socks",
        frozenset({"extra-one", "two"}),
        "extra == 'socks'",
    )
    assert tidy.requirement("nitrolearn @ file:///x.whl") == ("nitrolearn", frozenset(), "")
    assert tidy.applies('extra == "Dash_Board"', LINUX, ["dash-board"])  # names compared as packaging does
    assert not tidy.applies('extra == "dev"', LINUX, ["wandb"])
    assert tidy.applies("", LINUX, [])
    assert tidy.environment()["python_version"] == f"{sys.version_info[0]}.{sys.version_info[1]}"


# ---------- the packages ----------


def test_an_upgrade_removes_only_the_installers_packages_nothing_needs_any_more(tmp_path: Path) -> None:
    """An older installer's folder (no record): the record made before the install holds what that
    NitroLearn required, with each of its extras; afterwards only those the newer version does not need go."""
    site = tmp_path / "site-packages"
    old_release(site)
    ours = first_record(site)
    new_nitrolearn(site)
    gone = removed(site, roots(), ours)
    assert gone == {"pytest", "pluggy", "iniconfig", "old-helper"}
    # uv's seed packages and what wheel needs, maturin, the dashboard's Qt part, NVIDIA's libraries for the
    # PyTorch build in use, a package installed by hand (with what it needs), and PySocks, which nothing of
    # the installer's ever required, all stay.
    found = tidy.unneeded(roots(), tidy.installed([site]), LINUX, "nitrolearn", ours)
    assert found.by_hand == ("cowsay", "PySocks", "six")
    assert "old-helper" not in found.record and {"nitrolearn", "torch", "qtpy"} <= set(found.record)
    # A developer install keeps its tools.
    assert removed(site, roots(extras="dev"), ours) == {
        "old-helper",
        "wandb",
        "requests",
        "urllib3",
        "pyarrow",
    }


def test_packages_the_user_installed_with_uv_stay_with_what_they_need(tmp_path: Path) -> None:
    """uv marks a package the user installed (`uv pip install`) as it marks the installer's: the record
    tells them apart. Kept with it: what it needs, even a package the installer had put there."""
    site = tmp_path / "site-packages"
    old_release(site)
    dist(site, "pytest-xdist", "3.8", ["pytest>=7", "execnet>=2.1"])  # the user's, with uv
    dist(site, "execnet", "2.1.1")
    ours = first_record(site)
    assert not {"pytest-xdist", "execnet"} & ours
    new_nitrolearn(site)
    dist(site, "rich", "14.0", ["pysocks"])  # installed by the user after the newer version, with uv
    gone = removed(site, roots(), ours)
    assert gone == {"old-helper"}, "pytest stays for pytest-xdist, and pytest's own needs with it"


def test_a_version_without_pytorch_removes_it_and_nvidias_libraries(tmp_path: Path) -> None:
    """The version after PyTorch: NitroLearn no longer requires it and the installer no longer installs it,
    so PyTorch, NVIDIA's CUDA libraries and what only PyTorch needed go; typing-extensions stays (wandb).
    The same from an older installer's folder (no record) and from a record the installer kept."""
    site = tmp_path / "site-packages"
    old_release(site)
    seeded = first_record(site)
    new_nitrolearn(site, version="1.0.3")
    kept = set(tidy.unneeded(roots(), tidy.installed([site]), LINUX, "nitrolearn", seeded).record)
    new_nitrolearn(site, torch=False, version="1.1.0")
    torch_and_its = {
        "torch",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "triton",
        "sympy",
        "mpmath",
        "filelock",
    }
    for ours in (seeded, kept):
        gone = removed(site, roots(torch=False), ours)
        assert gone - {"pytest", "pluggy", "iniconfig", "old-helper"} == torch_and_its
        assert "typing-extensions" not in gone and "numpy" not in gone


def test_another_pytorch_build_drops_the_old_builds_libraries(tmp_path: Path) -> None:
    """cu126 to cu130 on Linux: the new build requires CUDA 13's libraries; CUDA 12's go."""
    site = tmp_path / "site-packages"
    old_release(site)
    ours = first_record(site)
    dist(site, "nvidia-cublas-cu13", "13.0")
    dist(site, "nvidia-cudnn-cu13", "9.13", ["nvidia-cublas-cu13"])
    for info in site.glob("torch-*.dist-info"):
        text = (info / "METADATA").read_text(encoding="utf-8").replace("-cu12", "-cu13")
        (info / "METADATA").write_text(text, encoding="utf-8")
    gone = removed(site, roots(), ours)
    assert {"nvidia-cublas-cu12", "nvidia-cudnn-cu12"} <= gone
    assert not {"nvidia-cublas-cu13", "nvidia-cudnn-cu13", "torch"} & gone
    # On Windows PyTorch's CUDA is inside its own package: the markers leave NVIDIA's packages unasked for.
    every = set(tidy.installed([site]))
    assert {"nvidia-cublas-cu13", "triton"} <= removed(site, roots(), every, WINDOWS)


def test_the_first_record_is_what_an_older_nitrolearn_and_its_installer_put_there(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    old_release(site)
    dist(site, "tqdm", "4.67")  # pinned by the release's copy of the packages
    ours = first_record(site)
    assert {
        "nitrolearn",
        "numpy",
        "torch",
        "nvidia-cudnn-cu12",
        "maturin",
        "pytest",
        "qtpy",
        "packaging",
    } <= ours
    assert not {"cowsay", "six", "pysocks", "pip", "setuptools", "wheel", "tqdm"} & ours
    assert "qtpy" not in first_record(site, WINDOWS), "the Qt window was installed on Linux only"
    assert "tqdm" in tidy.first_record(tidy.installed([site]), LINUX, ["tqdm"])
    # NitroLearn's first name, and no NitroLearn at all.
    for info in site.glob("nitrolearn-*.dist-info"):
        shutil.rmtree(info)
    assert tidy.first_record(tidy.installed([site]), LINUX) == set()
    dist(site, "gigalearn-nitro", "1.0.0", nitrolearn_needs(old_helper=True))
    assert {"gigalearn-nitro", "old-helper", "pytest", "torch"} <= first_record(site)


def test_the_record_is_made_once_before_the_install_and_kept_after_each_tidy(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    site = root / ".venv" / "lib" / "site-packages"
    old_release(site)
    assert not tidy.seed_record(root, [], LINUX), "not the folder's own .venv: nothing is written"
    assert tidy.seed_record(root, [site], LINUX)
    seeded = tidy.read_names(root / tidy.RECORD)
    assert seeded == first_record(site)
    new_nitrolearn(site)
    dist(site, "rich", "14.0")  # the user's
    assert not tidy.seed_record(root, [site], LINUX), "made once: never again over a record"
    assert tidy.read_names(root / tidy.RECORD) == seeded
    found = tidy.plan(root, [site], roots(), {}, prefix=root / ".venv", base=root, env=LINUX)
    assert set(found.packages.remove) == {"pytest", "pluggy", "iniconfig", "old-helper"}
    done = tidy.carry_out(found, root, FakeUv(site), [site])
    assert done.problems == []
    after = tidy.read_names(root / tidy.RECORD)
    assert after is not None and "rich" not in after and "old-helper" not in after and "nitrolearn" in after
    # A package uv could not remove stays in the record, so the next tidy tries again.
    dist(site, "old-helper", "0.3")
    tidy.write_names(root / tidy.RECORD, after | {"old-helper"})
    found = tidy.plan(root, [site], roots(), {}, prefix=root / ".venv", base=root, env=LINUX)
    assert found.packages.remove == ("old-helper",)
    tidy.carry_out(found, root, FakeUv(site, code=2), [site])
    assert "old-helper" in (tidy.read_names(root / tidy.RECORD) or set())
    assert tidy.read_names(root / "missing.json") is None


def test_nothing_is_removed_when_nitrolearn_itself_is_not_installed(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    dist(site, "torch", "2.14.1")
    dist(site, "requests", "2.0")
    found = tidy.unneeded(roots(), tidy.installed([site]), LINUX, "nitrolearn", {"torch", "requests"})
    assert found.remove == () and "cannot be told" in found.why_none


def test_packages_without_an_installer_mark_are_kept(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    old_release(site)
    dist(site, "unknown-origin", "1.0", installer=None)
    assert "unknown-origin" not in removed(site, roots(), first_record(site) | {"unknown-origin"})


# ---------- the Pythons in .tools/python ----------


def python_install(folder: Path, name: str, windows: bool = False) -> Path:
    place = folder / name
    if windows:
        place.mkdir(parents=True)
        (place / "python.exe").write_bytes(b"MZ")
    else:
        (place / "bin").mkdir(parents=True)
        (place / "bin" / "python3").write_bytes(b"")
    return place


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need extra rights on Windows")
def test_an_older_python_and_its_link_go_the_running_one_stays(tmp_path: Path) -> None:
    pythons = tmp_path / ".tools" / "python"
    old = python_install(pythons, "cpython-3.11.15-linux-x86_64-gnu")
    new = python_install(pythons, "cpython-3.12.13-linux-x86_64-gnu")
    (pythons / "cpython-3.11-linux-x86_64-gnu").symlink_to(old)
    (pythons / "cpython-3.12-linux-x86_64-gnu").symlink_to(new)
    (pythons / "cpython-3.10-linux-x86_64-gnu").symlink_to(pythons / "gone")  # a link left dangling
    for own in (".lock", ".gitignore"):
        (pythons / own).write_text("", encoding="utf-8")
    (pythons / ".temp").mkdir()
    (pythons / "notes").mkdir()  # not a name uv gives a Python
    (pythons / "cpython-3.9-without-a-python").mkdir()  # uv's link name, but no Python in it
    # .venv's pyvenv.cfg names uv's link; the running Python resolves to the install itself.
    running = [pythons / "cpython-3.12-linux-x86_64-gnu" / "bin", new]
    names = [path.name for path in tidy.old_pythons(pythons, running)]
    assert names == ["cpython-3.10-linux-x86_64-gnu", "cpython-3.11-linux-x86_64-gnu", old.name]
    # Only the link's name is enough to keep it and the install it leads to.
    assert [p.name for p in tidy.old_pythons(pythons, [pythons / "cpython-3.11-linux-x86_64-gnu"])] == [
        "cpython-3.10-linux-x86_64-gnu",
        "cpython-3.12-linux-x86_64-gnu",
        new.name,
    ]
    # A Python running from outside the folder: which one is in use cannot be told, so none goes.
    assert tidy.old_pythons(pythons, [tmp_path / "elsewhere" / "python"]) == []
    assert tidy.old_pythons(tmp_path / "missing", running) == []


def test_windows_pythons_are_recognised_by_their_python_exe(tmp_path: Path) -> None:
    pythons = tmp_path / ".tools" / "python"
    old = python_install(pythons, "cpython-3.11.15-windows-x86_64-none", windows=True)
    new = python_install(pythons, "cpython-3.12.13-windows-x86_64-none", windows=True)
    assert tidy.old_pythons(pythons, [new]) == [old]


def test_a_half_removed_python_is_known_by_uvs_name_for_it(tmp_path: Path) -> None:
    """An earlier tidy removed python.exe (or bin/) and then stopped: uv's full name still tells it."""
    pythons = tmp_path / ".tools" / "python"
    new = python_install(pythons, "cpython-3.12.13-linux-x86_64-gnu")
    for name in ("cpython-3.11.15-windows-x86_64-none", "cpython-3.13.1+freethreaded-linux-x86_64_v3-gnu"):
        (pythons / name / "Lib" / "site-packages").mkdir(parents=True)
    (pythons / "cpython-3.11.15-notes.txt").write_text("", encoding="utf-8")  # a file: never a Python
    names = [path.name for path in tidy.old_pythons(pythons, [new])]
    assert names == ["cpython-3.11.15-windows-x86_64-none", "cpython-3.13.1+freethreaded-linux-x86_64_v3-gnu"]


def test_the_venv_home_is_read_from_pyvenv_cfg(tmp_path: Path) -> None:
    (tmp_path / "pyvenv.cfg").write_text("home = C:\\N L\\.tools\\python\\cpython-3.11\nuv = 0.11\n", "utf-8")
    assert tidy.venv_home(tmp_path) == Path("C:\\N L\\.tools\\python\\cpython-3.11")
    assert tidy.venv_home(tmp_path / "none") is None


# ---------- paths: only the installer's own, never through a link ----------


def tree(root: Path) -> dict[str, bytes | str]:
    """Every file and link under `root` (links not followed): what a test compares before and after."""
    found: dict[str, bytes | str] = {}
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(folder) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                found[rel] = "-> " + os.readlink(path)
            elif path.is_file():
                found[rel] = path.read_bytes()
    return found


def user_folders(root: Path) -> None:
    for name in USER_FOLDERS:
        (root / name / "inside").mkdir(parents=True)
        (root / name / "inside" / "mine.txt").write_text(name, encoding="utf-8")
    replaced = root / ".tmp" / "update" / "replaced" / "1.0.2"
    replaced.mkdir(parents=True)
    (replaced / "changed.toml").write_text("the user's changed copy", encoding="utf-8")
    (root / "notes.txt").write_text("the user's own file", encoding="utf-8")


def test_only_the_installers_own_folders_can_be_reached(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    user_folders(root)
    before = tree(root)
    for name in (*USER_FOLDERS, ".tmp", ".venv", "python", "notes.txt"):
        assert tidy.remove_inside(root, root / name), name
        assert tidy.remove_inside(root, root / name / "inside"), name
    (root / ".tools").mkdir()
    (root / ".cache").mkdir()
    for name in (".tools", ".cache"):
        assert "not one of the installer's own" in tidy.remove_inside(root, root / name)
    assert tidy.remove_inside(root, root / ".tools" / ".." / "runs")
    assert "outside" in tidy.remove_inside(root, tmp_path / "elsewhere")
    assert tree(root) == before
    assert set(tidy.OWNED).isdisjoint(USER_FOLDERS)


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need extra rights on Windows")
def test_links_are_removed_themselves_never_followed(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (outside / "deep" / "precious.txt").write_text("keep", encoding="utf-8")
    (root / ".tools").mkdir(parents=True)
    (root / ".cache").symlink_to(outside)  # a whole owned folder that is a link: nothing inside is reached
    assert "is a link" in tidy.remove_inside(root, root / ".cache" / "deep")
    (root / ".tools" / "python").mkdir()
    link = root / ".tools" / "python" / "cpython-3.11-linux-x86_64-gnu"
    link.symlink_to(outside / "deep")
    assert tidy.remove_inside(root, link) == ""
    assert not link.is_symlink() and (outside / "deep" / "precious.txt").read_text(encoding="utf-8") == "keep"
    # A link inside a folder that goes is removed as a link: what it leads to stays.
    (root / ".tools" / "vendor.new").mkdir()
    (root / ".tools" / "vendor.new" / "to-outside").symlink_to(outside)
    assert tidy.remove_inside(root, root / ".tools" / "vendor.new") == ""
    assert (outside / "deep" / "precious.txt").is_file()


def rustc_info(target: Path, release: str, home: Path | None = None) -> None:
    """cargo's record of the Rust that last built in `target`: its version, and its home (sysroot), by
    default the folder's own Rust's (.tools/rustup beside `target`)."""
    target.mkdir(parents=True, exist_ok=True)
    home = home or target.parent / ".tools" / "rustup" / "toolchains" / f"{release}-x86_64-unknown-linux-gnu"
    version = f"rustc {release} (abc 2026-07-07)\nbinary: rustc\nrelease: {release}\nLLVM version: 22\n"
    paths = f"___\nlib___.rlib\n{home}\noff\npacked\nunix\n"
    outputs = {
        "1": {"success": True, "stdout": version, "stderr": ""},
        "2": {"success": True, "stdout": paths},
    }
    (target / ".rustc_info.json").write_text(
        json.dumps({"rustc_fingerprint": 1, "outputs": outputs}), "utf-8"
    )


def test_the_installers_leftovers(tmp_path: Path) -> None:
    root = tmp_path / "NitroLearn"
    cache = root / ".cache"
    for name in ("pip", "torch", "torchinductor", "triton", "cuda", "wandb", "uv"):
        (cache / name).mkdir(parents=True)
        (cache / name / "entry").write_bytes(b"x")
    (root / ".tools" / "vendor.new").mkdir(parents=True)
    (root / ".tools" / "vendor.new" / "half-unpacked.rs").write_bytes(b"x")
    (root / ".tools" / "torch-choice.txt").write_text("cu126\n", encoding="utf-8")
    (root / ".tools" / "install_choices.json").write_text("{}", encoding="utf-8")
    rustc_info(root / "target", "1.97.0")

    def names(**state: object) -> set[str]:
        found = tidy.stale_paths(
            root, **{"torch_gone": False, "torch_changed": False, "rust": "1.97.0"} | state
        )
        return {path.relative_to(root).as_posix() for path in found}

    assert names() == {".tools/vendor.new", ".cache/pip"}
    torch_caches = {".cache/torch", ".cache/torchinductor", ".cache/triton"}
    assert names(torch_changed=True) == {".tools/vendor.new", ".cache/pip"} | torch_caches
    assert (
        names(torch_gone=True)
        == {".tools/vendor.new", ".cache/pip", ".tools/torch-choice.txt"} | torch_caches
    )
    # target/: the folder's own Rust built it, and that Rust is another version now, or gone.
    assert ".cache/pip" in names() and "target" not in names()
    assert "target" in names(rust="1.98.0") and "target" in names(rust=None)
    # Built with the user's own Rust (outside the folder): never touched, whatever the folder's Rust.
    rustc_info(root / "target", "1.93.0", home=tmp_path / "home" / ".rustup" / "toolchains" / "stable")
    assert "target" not in names(rust=None) and "target" not in names(rust="1.97.0")
    rustc_info(root / "target", "1.93.0", home=Path("/usr"))
    assert "target" not in names(rust=None)
    (root / "target" / ".rustc_info.json").unlink()  # not cargo's build folder, as far as can be told
    assert "target" not in names(rust=None)
    # An empty cache folder is not worth a mention.
    for entry in (cache / "pip").iterdir():
        entry.unlink()
    assert ".cache/pip" not in names()


# ---------- the whole tidy ----------


class FakeUv:
    """Stands in for uv: `pip uninstall` removes the packages' .dist-info folders; every call is kept."""

    def __init__(self, site: Path, code: int = 0) -> None:
        self.site, self.code = site, code
        self.calls: list[list[str]] = []

    def __call__(self, arguments: list[str]) -> int:
        self.calls.append(arguments)
        if arguments[:2] == ["pip", "uninstall"] and self.code == 0:
            names = {tidy.normal(name) for name in arguments[4:]}
            for info in self.site.glob("*.dist-info"):
                if tidy.normal(info.name.split("-")[0]) in names:
                    for file in info.iterdir():
                        file.unlink()
                    info.rmdir()
        return self.code


def upgraded_folder(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An older version's folder after the install of a newer one without PyTorch (its record made before
    that install): (root, site-packages, the running Python)."""
    root = tmp_path / "NitroLearn"
    site = root / ".venv" / "lib" / "site-packages"
    old_release(site)
    assert tidy.seed_record(root, [site], LINUX)
    new_nitrolearn(site, torch=False, version="1.1.0")
    user_folders(root)
    pythons = root / ".tools" / "python"
    python_install(pythons, "cpython-3.11.15-linux-x86_64-gnu")
    new = python_install(pythons, "cpython-3.12.13-linux-x86_64-gnu")
    for name in ("pip", "torchinductor", "uv"):
        (root / ".cache" / name).mkdir(parents=True)
        (root / ".cache" / name / "entry").write_bytes(b"x")
    (root / ".tools" / "torch-choice.txt").write_text("cu126\n", encoding="utf-8")
    return root, site, new


def test_a_tidy_removes_what_the_plan_lists_and_the_plan_changes_nothing(tmp_path: Path) -> None:
    root, site, python = upgraded_folder(tmp_path)
    before = {name: dist.version for name, dist in tidy.installed([site]).items()}
    snapshot = tree(root)
    found = tidy.plan(root, [site], roots(torch=False), before, prefix=root / ".venv", base=python, env=LINUX)
    assert tree(root) == snapshot, "a plan only looks"
    assert "torch" in found.packages.remove and "cowsay" not in found.packages.remove
    assert {path.relative_to(root).as_posix() for path in found.paths} == {
        ".cache/pip",
        ".cache/torchinductor",
        ".tools/torch-choice.txt",
        ".tools/python/cpython-3.11.15-linux-x86_64-gnu",
    }
    assert found.whole_cache, "an older Python goes: uv's cache is for it"
    assert any(line == "package torch" for line in tidy.describe(found, root))

    uv = FakeUv(site)
    done = tidy.carry_out(found, root, uv, [site], python="py")
    assert uv.calls[0][:4] == ["pip", "uninstall", "--python", "py"]
    assert uv.calls[1:] == [["cache", "clean"], ["cache", "prune"]]
    assert done.problems == [] and set(done.packages) == set(found.packages.remove)
    left = tidy.installed([site])
    assert "torch" not in left and {"nitrolearn", "numpy", "cowsay", "six", "pip", "wheel"} <= set(left)
    after = tree(root)
    for rel, content in snapshot.items():
        if rel.split("/")[0] in (*USER_FOLDERS, ".tmp", "notes.txt"):
            assert after[rel] == content, f"{rel} is the user's"
    assert not (root / ".tools" / "python" / "cpython-3.11.15-linux-x86_64-gnu").exists()
    assert (root / ".tools" / "python" / "cpython-3.12.13-linux-x86_64-gnu").is_dir()
    assert (root / ".cache" / "uv" / "entry").is_file(), "uv's cache is uv's to tidy"
    line = tidy.summary(done, 3.04)
    assert line.startswith("Removed what this version no longer uses: ") and line.endswith(
        "about 3.0 GB freed."
    )
    assert "torch" in line or "more" in line

    # Run again: nothing is left to remove.
    again = tidy.plan(root, [site], roots(torch=False), {}, prefix=root / ".venv", base=python, env=LINUX)
    assert again.packages.remove == () and again.paths == () and again.cached == ()
    assert tidy.summary(tidy.carry_out(again, root, FakeUv(site), [site]), 0.0).startswith(
        "Nothing left over"
    )


def test_replaced_packages_leave_uvs_cache_and_the_same_ones_stay(tmp_path: Path) -> None:
    root, site, python = upgraded_folder(tmp_path)
    shutil.rmtree(root / ".tools" / "python" / "cpython-3.11.15-linux-x86_64-gnu")
    # The Python stays the same this time; numpy was replaced, torch removed, the rest unchanged.
    before = {name: dist.version for name, dist in tidy.installed([site]).items()} | {"numpy": "2.0.0"}
    found = tidy.plan(root, [site], roots(torch=False), before, prefix=root / ".venv", base=python, env=LINUX)
    assert not found.whole_cache
    assert "numpy" in found.cached and "torch" in found.cached and "six" not in found.cached
    uv = FakeUv(site)
    tidy.carry_out(found, root, uv, [site])
    assert (
        uv.calls[1][:2] == ["cache", "clean"] and "torch" in uv.calls[1] and uv.calls[2] == ["cache", "prune"]
    )


def test_what_cannot_be_removed_is_said_and_the_rest_goes_on(tmp_path: Path) -> None:
    root, site, python = upgraded_folder(tmp_path)
    before = {name: dist.version for name, dist in tidy.installed([site]).items()}
    found = tidy.plan(root, [site], roots(torch=False), before, prefix=root / ".venv", base=python, env=LINUX)
    done = tidy.carry_out(found, root, FakeUv(site, code=2), [site])
    assert done.packages == [] and done.problems[0].startswith("packages still in .venv (uv ended with 2): ")
    assert "uv's cache could not drop the old downloads" in done.problems
    assert not (root / ".cache" / "pip").exists(), "the folders still go"
    assert "torch-choice.txt" not in str(done.problems)


def test_what_a_tidy_could_not_remove_is_tried_again_by_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path in use (or too long, or held by antivirus) goes in the pending list; the next tidy tries it
    again even when nothing else would list it any more, unless it is in use by then."""
    root, site, python = upgraded_folder(tmp_path)
    before = {name: dist.version for name, dist in tidy.installed([site]).items()}
    old_python = root / ".tools" / "python" / "cpython-3.11.15-linux-x86_64-gnu"
    real_remove = tidy.remove_inside
    held = {root / ".cache" / "torchinductor", old_python}

    def locked(folder: Path, path: Path) -> str:
        return "in use" if path in held else real_remove(folder, path)

    monkeypatch.setattr(tidy, "remove_inside", locked)
    found = tidy.plan(root, [site], roots(torch=False), before, prefix=root / ".venv", base=python, env=LINUX)
    done = tidy.carry_out(found, root, FakeUv(site), [site])
    assert "in use" in done.problems
    assert tidy.read_names(root / tidy.PENDING) == {
        ".cache/torchinductor",
        old_python.relative_to(root).as_posix(),
    }

    # The next install: PyTorch went last time, so only the pending list names its cache now; the old
    # Python lost its bin/ meanwhile, and is known by its name.
    shutil.rmtree(old_python / "bin")
    (old_python / "lib").mkdir()
    monkeypatch.setattr(tidy, "remove_inside", real_remove)
    again = tidy.plan(root, [site], roots(torch=False), {}, prefix=root / ".venv", base=python, env=LINUX)
    assert {path.relative_to(root).as_posix() for path in again.paths} == {
        ".cache/torchinductor",
        old_python.relative_to(root).as_posix(),
    }
    done = tidy.carry_out(again, root, FakeUv(site), [site])
    assert (
        done.problems == [] and not old_python.exists() and not (root / ".cache" / "torchinductor").exists()
    )
    assert not (root / tidy.PENDING).exists(), "nothing left to try"


def test_a_pending_path_is_not_retried_once_it_is_in_use_or_not_the_installers(tmp_path: Path) -> None:
    root, site, python = upgraded_folder(tmp_path)
    shutil.rmtree(root / ".tools" / "python" / "cpython-3.11.15-linux-x86_64-gnu")
    (root / ".tools" / "torch-choice.txt").write_text("cu130\n", encoding="utf-8")
    rustc_info(root / "target", "1.97.0")
    (root / "runs" / "inside").mkdir(parents=True, exist_ok=True)
    pending = [
        ".tools/python/cpython-3.12.13-linux-x86_64-gnu",  # .venv runs on it now
        ".tools/torch-choice.txt",  # PyTorch is installed: the remembered build is in use
        "target",  # built since with the pinned Rust
        "runs/inside",  # never the installer's
        "../outside",
        ".cache/gone",  # not there any more
        ".cache/pip",
    ]
    tidy.write_names(root / tidy.PENDING, pending)
    found = tidy.plan(root, [site], roots(), {}, prefix=root / ".venv", base=python, rust="1.97.0", env=LINUX)
    assert [path.relative_to(root).as_posix() for path in found.paths] == [".cache/pip"]
    # The user builds with their own Rust: a target/ pending since an earlier tidy stays theirs.
    rustc_info(root / "target", "1.93.0", home=Path("/usr"))
    tidy.write_names(root / tidy.PENDING, ["target"])
    again = tidy.plan(root, [site], roots(), {}, prefix=root / ".venv", base=python, env=LINUX)
    assert [path.name for path in again.paths] == ["pip"]


def test_the_summary_names_a_few_and_counts_the_rest() -> None:
    done = tidy.Tidied([f"p{n}" for n in range(9)], [".tools/python/cpython-3.10"], [])
    line = tidy.summary(done, None)
    assert line == (
        "Removed what this version no longer uses: 9 packages (p0, p1, p2, p3 and 5 more), "
        ".tools/python/cpython-3.10."
    )
    assert tidy.summary(tidy.Tidied(), 0.01) == "Nothing left over from an older version to remove."
