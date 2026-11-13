"""The copies NitroLearn's release carries of what an install would otherwise fetch from other people's
servers, so a release keeps installing even when one of them moves or goes away. The release workflow
(.github/workflows/release.yml) builds them with this file and attaches them to the version's GitHub release
beside the ready-made core and programs; SHA256SUMS.txt lists every file's hash.

    uv-<uv version>-py3-none-<tag>.whl          uv itself, the very wheel PyPI has (MIT or Apache-2.0)
    python-<version>-<platform>.tar.gz          the Python uv installs (python-build-standalone; PSF licence
                                                and the licences of the libraries built into it)
    rocketsim-recordings.zip                     RocketSim's five game recordings and RocketSim's MIT licence
    nitrolearn-crates-<version>.tar.gz           every Rust crate a build from the code needs (`cargo vendor
                                                --locked`: the workspace, nitro-replicar and replicar-cli)
    nitrolearn-wheels-<version>-<platform>.zip   every package from PyPI an install takes, except PyTorch
    BUNDLED-NOTICES.md                           what each of these is, where it comes from, its licence

The installers take each file from a folder of the release's files when there is one (offline: `--bundles`,
or a `release-files` folder), else from the release, and only then from the original source. uv and Python
are pinned here and in install.ps1 / install.sh (a test keeps them equal), the recordings in
nitro/recordings.py; each is checked against its pinned hash wherever it comes from. Not carried: PyTorch
(its graphics-card builds are bigger than a release file may be, and the Linux ones need NVIDIA's own
libraries), the arena meshes (Rocket League's game data: only ever downloaded from RLGym's package), the
NVIDIA driver and RLBot (separate programs from their makers).

    python tools/bundles.py upstream OUT                  uv, Python and the recordings, hash-checked
    python tools/bundles.py crates OUT                    the crates archive (needs cargo and the network)
    python tools/bundles.py wheels OUT --platform P       the wheels archive for P (needs uv and pip)
    python tools/bundles.py notices OUT                   BUNDLED-NOTICES.md
    python tools/bundles.py check DIR                     all of them there, each under GitHub's size limit

Standard library only.
"""

from __future__ import annotations

import argparse
import email.parser
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RELEASES = "https://github.com/AverageCoderx/NitroLearn/releases/download/v{version}/"
SUMS = "SHA256SUMS.txt"
PLATFORMS = ("windows-x64", "linux-x64")
LIMIT = 2 * 1024**3  # GitHub's limit for one release file
NOTICES = "BUNDLED-NOTICES.md"
RECORDINGS = "rocketsim-recordings.zip"  # nitro/recordings.py RELEASE_ASSET


@dataclass(frozen=True)
class Upstream:
    """A file the release carries unchanged: its name on the release, where it comes from, its SHA-256."""

    asset: str
    url: str
    sha256: str


UV_VERSION = "0.11.32"
UV = {
    "windows-x64": Upstream(
        "uv-0.11.32-py3-none-win_amd64.whl",
        "https://files.pythonhosted.org/packages/57/60/a645cca710448004268b7731c4dc6fc7d084a7c620fc73ecc2127cb61661/"
        "uv-0.11.32-py3-none-win_amd64.whl",
        "855632ee1d2a8491986efa54c562b806cd32477761889a1e13624b519d98a45e",
    ),
    "linux-x64": Upstream(
        "uv-0.11.32-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "https://files.pythonhosted.org/packages/90/b0/114463d056b6b328d45557001e848b8ab15539bd8f4fa7a457ccb83e2b5d/"
        "uv-0.11.32-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "3da76cd4e2697de30928b8a8524bd39183ac1e08cb7e72833807c022b7cba6c4",
    ),
}
# The Python 3.11 the pinned uv installs (`uv python install 3.11`), as its own list of downloads names it.
# uv takes it from a folder laid out as python-build-standalone's releases (UV_PYTHON_INSTALL_MIRROR) and
# checks it against the hash built into uv.
PYTHON_VERSION = "3.11.15"
PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/"
PYTHON = {
    "windows-x64": Upstream(
        "python-3.11.15-windows-x64.tar.gz",
        PBS + "20260718/cpython-3.11.15%2B20260718-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        "a48c2dbe832319f61aa8557c9900caec70f7fed0cbee391a4c9ff9f98b50222d",
    ),
    "linux-x64": Upstream(
        "python-3.11.15-linux-x64.tar.gz",
        PBS + "20260718/cpython-3.11.15%2B20260718-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
        "23ccae6f1ff73e8aa8378436f869da003b8eb7d6c95f2bc706f494115ba1447d",
    ),
}
ROCKETSIM_LICENSE = Upstream(
    "LICENSE",
    "https://raw.githubusercontent.com/ZealanL/RocketSim/45722876705adf5dc673aa764e80382cfe8c3312/LICENSE",
    "a50ad111270f42d1c84186bcd0165d280055b11317d25e81b6141b7376d522f8",
)
# replicar-cli, which `nitro pretrain` builds with `cargo install` when the release has no ready-made one.
REPLICAR_CLI = Upstream(
    "replicar-cli-2.2.1.crate",
    "https://static.crates.io/crates/replicar-cli/replicar-cli-2.2.1.crate",
    "f69de584d7a699c55feaeaf95dacc27263cba680f668da6359741e27c1c0174c",
)
VENDOR = "vendor"  # the crates' folder in the archive (and in .tools once installed)
VENDOR_LIST = "VENDORED-CRATES.md"
WHEELS_LIST = "NOTICES.md"
REQUIREMENTS = "requirements.txt"
# Licences (SPDX names, or the words packages write instead) that allow passing the files on unchanged.
PERMISSIVE = frozenset(
    {
        "MIT",
        "MIT-0",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "ISC",
        "Zlib",
        "Unlicense",
        "CC0-1.0",
        "BSL-1.0",
        "PSF-2.0",
        "MPL-2.0",
        "Unicode-3.0",
        "Unicode-DFS-2016",
        "LLVM-exception",
    }
)
_WORDS = {
    "mit license": "MIT",
    "apache software license": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "bsd license": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "python software foundation license": "PSF-2.0",
}


# A wheel whose metadata names no licence: its licence file's text, recognised by a sentence of it.
_LICENCE_FILE = re.compile(r"\.dist-info/(?:licenses/)?(?:LICEN[CS]E|COPYING)[^/]*$", re.IGNORECASE)
_TEXTS = (
    ("Permission is hereby granted, free of charge", "MIT"),
    ("Apache License\n                           Version 2.0", "Apache-2.0"),
    ("Redistribution and use in source and binary forms", "BSD-3-Clause"),
)


class BundleError(RuntimeError):
    pass


def crates_asset(version: str) -> str:
    return f"nitrolearn-crates-{version}.tar.gz"


def wheels_asset(version: str, platform: str) -> str:
    return f"nitrolearn-wheels-{version}-{platform}.zip"


def bundled_assets(version: str) -> list[str]:
    """Every file of the release this module makes, in the order BUNDLED-NOTICES.md lists them."""
    return [
        *(UV[p].asset for p in PLATFORMS),
        *(PYTHON[p].asset for p in PLATFORMS),
        RECORDINGS,
        crates_asset(version),
        *(wheels_asset(version, p) for p in PLATFORMS),
        NOTICES,
    ]


def mirror_path(python: Upstream) -> str:
    """Where uv looks for `python` inside a UV_PYTHON_INSTALL_MIRROR folder: <release>/<file name>."""
    return urllib.parse.unquote(python.url.removeprefix(PBS))


def project_version(root: Path = ROOT) -> str:
    text = (root / "Cargo.toml").read_text(encoding="utf-8")
    found = re.search(
        r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', text, re.MULTILINE | re.DOTALL
    )
    if found is None:
        raise BundleError("Cargo.toml names no version")
    return found[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, sha256: str, attempts: int = 4) -> Path:
    """`url` into `target`, kept only when it matches `sha256`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as answer, part.open("wb") as out:
                shutil.copyfileobj(answer, out, 1 << 20)
            break
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    if sha256_file(part) != sha256:
        part.unlink()
        raise BundleError(f"{url} does not match its pinned SHA-256 hash")
    os.replace(part, target)
    return target


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BundleError(f"{path} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def recordings_module(root: Path = ROOT) -> ModuleType:
    return _module(root / "python" / "nitro" / "recordings.py", "nitro_recordings_for_bundles")


def _stable(info: zipfile.ZipInfo | tarfile.TarInfo) -> None:
    """The same archive from the same files, whenever and wherever it is made."""
    if isinstance(info, zipfile.ZipInfo):
        info.date_time = (2000, 1, 1, 0, 0, 0)
        info.external_attr = 0o644 << 16
    else:
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.isdir() else 0o644


def write_zip(target: Path, files: Iterable[tuple[str, Path]], compress: bool = False) -> Path:
    """A zip of (name inside, file) pairs in name order; stored, as the files are compressed already."""
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(target, "w", method) as out:
        for name, path in sorted(files):
            info = zipfile.ZipInfo(name)
            _stable(info)
            info.compress_type = method
            with path.open("rb") as source, out.open(info, "w") as dest:
                shutil.copyfileobj(source, dest, 1 << 20)
    return target


# ---------- uv, Python and the recordings ----------


def upstream(out: Path, root: Path = ROOT) -> list[Path]:
    """uv and Python for each platform, and the recordings archive, each checked against its pin."""
    made = []
    for platform in PLATFORMS:
        for item in (UV[platform], PYTHON[platform]):
            made.append(download(item.url, out / item.asset, item.sha256))
    made.append(recordings_archive(out / RECORDINGS, root))
    return made


def recordings_archive(target: Path, root: Path = ROOT) -> Path:
    """The five recordings from RocketSim's pinned commit, with that commit's LICENSE beside them."""
    recordings = recordings_module(root)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        files = [("LICENSE", download(ROCKETSIM_LICENSE.url, folder / "LICENSE", ROCKETSIM_LICENSE.sha256))]
        for name, (_, sha256) in recordings.RECORDINGS.items():
            files.append((name, download(recordings.URL + name, folder / name, sha256)))
        return write_zip(target, files)


# ---------- the Rust crates ----------

CARGO_CONFIG = """# Written by NitroLearn's installer: every Rust crate comes from the copy in NitroLearn's
# folder (.tools/{vendor}, from the release's nitrolearn-crates archive; cargo checks each crate against
# Cargo.lock), so builds need no crates.io. Delete this file to use crates.io again.
[source.crates-io]
replace-with = "nitrolearn-vendored"

[source.nitrolearn-vendored]
directory = "{vendor}"
"""


def add_crate(vendor: Path, crate: Path, sha256: str) -> Path:
    """A downloaded .crate (checked against `sha256`) unpacked into `vendor` as `cargo vendor` lays crates
    out, with the .cargo-checksum.json cargo reads."""
    if sha256_file(crate) != sha256:
        raise BundleError(f"{crate.name} does not match its pinned SHA-256 hash")
    with tempfile.TemporaryDirectory(dir=vendor) as temp:
        with tarfile.open(crate, "r:gz") as tar:
            tar.extractall(temp, filter="data")
        unpacked = Path(temp) / crate.name.removesuffix(".crate")
        package = tomllib.loads((unpacked / "Cargo.toml").read_text(encoding="utf-8"))["package"]
        target = vendor / f"{package['name']}-{package['version']}"
        if target.exists():
            raise BundleError(f"{package['name']} {package['version']} is vendored already")
        unpacked.rename(target)
    files = {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    checksum = {"files": files, "package": sha256}
    (target / ".cargo-checksum.json").write_text(json.dumps(checksum), encoding="utf-8")
    return target


def licence_terms(expression: str) -> set[str]:
    """The licence names in an SPDX expression (or the words a package wrote instead)."""
    text = expression.strip()
    if text.lower() in _WORDS:
        return {_WORDS[text.lower()]}
    words = re.split(r"\s+(?:AND|OR|WITH)\s+|[()/,]|\s+", text)
    return {_WORDS.get(word.lower(), word) for word in words if word}


def licence_allows(expression: str) -> bool:
    """Whether every licence the expression can be read under allows passing the files on unchanged: an
    OR needs one permissive choice, an AND all parts."""
    if not expression.strip():
        return False
    if expression.strip().lower() in _WORDS:
        return True
    for alternative in re.split(r"\s+OR\s+|\s*/\s*", expression.replace("(", " ").replace(")", " ")):
        if licence_terms(alternative) and licence_terms(alternative) <= PERMISSIVE:
            return True
    return False


def vendored_crates(vendor: Path) -> list[tuple[str, str, str, str]]:
    """(name, version, licence, repository) of every crate in `vendor`."""
    rows = []
    for folder in sorted(vendor.iterdir()):
        manifest = folder / "Cargo.toml"
        if not manifest.is_file():
            continue
        package = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]
        licence = str(package.get("license") or "")
        rows.append(
            (str(package["name"]), str(package["version"]), licence, str(package.get("repository", "")))
        )
    return rows


def crates_list(rows: Sequence[tuple[str, str, str, str]]) -> str:
    refused = [f"{name} {version} ({licence or 'no licence'})" for name, version, licence, _ in rows if not
               licence_allows(licence)]  # fmt: skip
    if refused:
        raise BundleError("crates whose licence does not allow passing them on: " + ", ".join(refused))
    lines = [
        "# The Rust crates in this archive",
        "",
        "Unchanged source code of every crate NitroLearn's Rust code is built with (`cargo vendor",
        "--locked`), each under its own licence, which its folder carries where the crate was published",
        "with one. Every licence here allows passing the code on.",
        "",
        "| Crate | Version | Licence | Source |",
        "|---|---|---|---|",
    ]
    lines += [f"| {name} | {version} | {licence} | {source} |" for name, version, licence, source in rows]
    return "\n".join(lines) + "\n"


def tar_folder(target: Path, folder: Path, extra: Sequence[tuple[str, bytes]] = ()) -> Path:
    """A .tar.gz of `folder` (as its own name) and `extra` files, in name order, with no dates or owners."""
    with (
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as packed,
        tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name, data in extra:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            _stable(info)
            tar.addfile(info, io.BytesIO(data))
        for path in sorted([folder, *folder.rglob("*")]):
            info = tar.gettarinfo(str(path), arcname=path.relative_to(folder.parent).as_posix())
            _stable(info)
            if path.is_file():
                with path.open("rb") as file:
                    tar.addfile(info, file)
            else:
                tar.addfile(info)
    return target


def crates(out: Path, root: Path = ROOT, cargo: str = "cargo") -> Path:
    """The archive of every crate the builds need: the workspace's Cargo.lock, nitro-replicar's and the
    one replicar-cli is published with, plus replicar-cli itself (for `cargo install`)."""
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        crate = download(REPLICAR_CLI.url, work / REPLICAR_CLI.asset, REPLICAR_CLI.sha256)
        with tarfile.open(crate, "r:gz") as tar:
            tar.extractall(work / "cli", filter="data")
        cli_manifest = work / "cli" / REPLICAR_CLI.asset.removesuffix(".crate") / "Cargo.toml"
        vendor = work / VENDOR
        command = [cargo, "vendor", "--locked", "--versioned-dirs"]
        command += ["-s", str(root / "crates" / "nitro_replicar" / "Cargo.toml"), "-s", str(cli_manifest)]
        subprocess.run([*command, str(vendor)], cwd=root, check=True, stdout=subprocess.DEVNULL)
        add_crate(vendor, crate, REPLICAR_CLI.sha256)
        listing = crates_list(vendored_crates(vendor)).encode("utf-8")
        config = CARGO_CONFIG.format(vendor=VENDOR).encode("utf-8")
        return tar_folder(out / crates_asset(project_version(root)), vendor, [
            (VENDOR_LIST, listing), ("cargo-config.toml", config)
        ])  # fmt: skip


# ---------- the Python packages ----------

# The Python platforms the release's core is built for, as uv and pip name them.
UV_PLATFORM = {"windows-x64": "x86_64-pc-windows-msvc", "linux-x64": "x86_64-manylinux_2_28"}
PIP_PLATFORMS = {
    "windows-x64": ["win_amd64"],
    "linux-x64": [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
    + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"],
}
LEFT_TO_PYPI = ("torch",)  # from PyTorch's own index, never from the release


def wheel_requirements(root: Path = ROOT) -> list[str]:
    """What an install takes from PyPI (pyproject.toml's dependencies and the extras the installer
    names, and the build backend `nitro rebuild` uses), without PyTorch."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    steps = _module(root / "tools" / "install_steps.py", "nitro_install_steps_for_bundles")
    wanted = list(project["project"]["dependencies"])
    for extra in str(steps.EXTRAS).split(","):
        wanted += project["project"]["optional-dependencies"][extra]
    wanted += project["build-system"]["requires"]
    kept = [r for r in wanted if re.split(r"[\s<>=!~\[;]", r, maxsplit=1)[0].lower() not in LEFT_TO_PYPI]
    return list(dict.fromkeys(kept))


def pinned(compiled: str) -> list[tuple[str, str, list[str]]]:
    """(name, version, hashes) for each requirement `uv pip compile --generate-hashes` wrote."""
    joined = re.sub(r"\\\s*\n", " ", compiled)
    found = []
    for line in joined.splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line)
        if match:
            found.append((match[1], match[2], re.findall(r"--hash=sha256:([0-9a-f]{64})", line)))
    return found


def pypi_source(name: str, version: str, hashes: Sequence[str], folder: Path) -> Path:
    """A package's source archive from PyPI, one of the hashes the compiled requirements pin."""
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=120) as answer:
        files = json.load(answer)["urls"]
    for file in files:
        if file["packagetype"] == "sdist" and file["digests"]["sha256"] in hashes:
            return download(file["url"], folder / file["filename"], file["digests"]["sha256"])
    raise BundleError(f"{name} {version} has neither a wheel for this platform nor a pinned source archive")


def wheel_licence(wheel: Path) -> str:
    """The licence a wheel's metadata names (License-Expression, else License, else its classifiers)."""
    with zipfile.ZipFile(wheel) as zipped:
        name = next(n for n in zipped.namelist() if n.endswith(".dist-info/METADATA"))
        meta = email.parser.HeaderParser().parsestr(zipped.read(name).decode("utf-8", "replace"))
    if meta.get("License-Expression"):
        return str(meta["License-Expression"]).strip()
    licence = str(meta.get("License") or "").strip()
    if licence and "\n" not in licence and len(licence) < 60:
        return licence
    classifiers = [
        str(c).split(" :: ")[-1] for c in meta.get_all("Classifier", []) if c.startswith("License")
    ]
    if classifiers:
        return " OR ".join(classifiers)
    if licence:
        return licence.splitlines()[0]
    with zipfile.ZipFile(wheel) as zipped:  # only a licence file: named from its text
        texts = [
            zipped.read(n).decode("utf-8", "replace") for n in zipped.namelist() if _LICENCE_FILE.search(n)
        ]
    return next((name for text in texts for words, name in _TEXTS if words in text), "")


def wheels_list(wheels: Sequence[Path]) -> str:
    rows = [(path.name.split("-")[0], path.name.split("-")[1], wheel_licence(path)) for path in wheels]
    refused = [f"{name} {version} ({licence or 'no licence'})" for name, version, licence in rows if not
               licence_allows(licence)]  # fmt: skip
    if refused:
        raise BundleError("packages whose licence does not allow passing them on: " + ", ".join(refused))
    lines = [
        "# The Python packages in this archive",
        "",
        "Unchanged wheels from PyPI (a package PyPI has only as source code is built into a wheel",
        "unchanged), each under its own licence, which the wheel carries in its .dist-info folder.",
        "",
        "| Package | Version | Licence |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {version} | {licence} |" for name, version, licence in sorted(rows)]
    return "\n".join(lines) + "\n"


def wheels(
    out: Path, platform: str, root: Path = ROOT, uv: str = "uv", requirements: list[str] | None = None
) -> Path:
    """The archive of the pinned wheels for `platform` and Python 3.11, with requirements.txt pinning each
    file's hash (the installer installs it with --no-index --require-hashes)."""
    wanted = requirements if requirements is not None else wheel_requirements(root)
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        (work / "wanted.in").write_text("\n".join(wanted) + "\n", encoding="utf-8")
        compile_command = [uv, "pip", "compile", str(work / "wanted.in"), "--python-version", "3.11"]
        compile_command += ["--python-platform", UV_PLATFORM[platform], "--generate-hashes", "--no-header"]
        compiled = subprocess.run(
            compile_command,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"UV_NO_CONFIG": "1"},
        ).stdout
        folder = work / "wheels"
        folder.mkdir()
        tags = [part for tag in PIP_PLATFORMS[platform] for part in ("--platform", tag)]
        for name, version, hashes in pinned(compiled):
            one = work / "one.txt"
            one.write_text(f"{name}=={version} " + " ".join(f"--hash=sha256:{h}" for h in hashes) + "\n")
            pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
            got = subprocess.run(
                [*pip, "download", "--no-deps", "--require-hashes", "--only-binary=:all:", *tags,
                 "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311", "--abi", "abi3",
                 "--abi", "none", "-d", str(folder), "-r", str(one)],
                check=False, stdout=subprocess.DEVNULL,
            )  # fmt: skip
            if got.returncode != 0:  # only source code on PyPI: built into a wheel (pure Python only)
                source = pypi_source(name, version, hashes, work / "sources")
                built = work / "built"
                subprocess.run([uv, "build", "--wheel", "--out-dir", str(built), str(source)], check=True)
                for wheel in built.glob("*.whl"):
                    if not wheel.name.endswith("-none-any.whl"):
                        raise BundleError(f"{wheel.name} is compiled: it cannot be built here for {platform}")
                    wheel.rename(folder / wheel.name)
        files = sorted(folder.glob("*.whl"))
        lines = [
            f"{p.name.split('-')[0]}=={p.name.split('-')[1]} --hash=sha256:{sha256_file(p)}" for p in files
        ]
        (work / REQUIREMENTS).write_text("\n".join(lines) + "\n", encoding="utf-8")
        (work / WHEELS_LIST).write_text(wheels_list(files), encoding="utf-8")
        entries = [(p.name, p) for p in files]
        entries += [(REQUIREMENTS, work / REQUIREMENTS), (WHEELS_LIST, work / WHEELS_LIST)]
        return write_zip(out / wheels_asset(project_version(root), platform), entries)


# ---------- the notices and the checks ----------


def notices(version: str) -> str:
    """BUNDLED-NOTICES.md: each bundled file, where it comes from and under which licence."""
    rows = [
        (
            ", ".join(UV[p].asset for p in PLATFORMS),
            f"uv {UV_VERSION} by Astral, the wheels on PyPI unchanged",
            "MIT or Apache-2.0 (both texts in each wheel's .dist-info/licenses; LICENSES/uv-MIT.txt)",
        ),
        (
            ", ".join(PYTHON[p].asset for p in PLATFORMS),
            f"Python {PYTHON_VERSION} as python-build-standalone builds it (Astral), unchanged: {PBS}",
            "Python Software Foundation License, and the licences of the libraries built in "
            "(LICENSES/Python-PSF.txt, LICENSES/python-build-standalone.md)",
        ),
        (
            RECORDINGS,
            "RocketSim's five game recordings (rocketsim_test/recordings) at commit 4572287, unchanged",
            "MIT, RocketSim's LICENSE inside the archive (LICENSES/RocketSim-MIT.txt)",
        ),
        (
            crates_asset(version),
            "the source code of every Rust crate the builds use, unchanged (cargo vendor --locked)",
            f"each crate's own licence, all permissive; the list is {VENDOR_LIST} inside",
        ),
        (
            ", ".join(wheels_asset(version, p) for p in PLATFORMS),
            "the Python packages an install takes from PyPI (not PyTorch), unchanged wheels",
            f"each package's own licence, all permissive; the list is {WHEELS_LIST} inside",
        ),
    ]
    lines = [
        f"# What NitroLearn {version}'s release carries from others",
        "",
        "Copies kept with the release so an install does not depend on other servers. The installer still",
        "falls back to the original source when a copy is missing. Every licence named allows passing the",
        "files on unchanged. Not carried: PyTorch (from PyTorch's own index: its graphics-card builds are",
        "bigger than a release file may be), the arena meshes (Rocket League game data, downloaded from",
        "RLGym's package on PyPI), the NVIDIA driver and RLBot v5 (programs from their makers).",
        "",
        "| Files | What and where from | Licence |",
        "|---|---|---|",
    ]
    lines += [f"| {files} | {what} | {licence} |" for files, what, licence in rows]
    return "\n".join(lines) + "\n"


def check(folder: Path, version: str) -> list[tuple[str, int]]:
    """Every bundled file there and each under GitHub's per-file limit; (name, size) of every file."""
    missing = [name for name in bundled_assets(version) if not (folder / name).is_file()]
    if missing:
        raise BundleError("missing from the release: " + ", ".join(missing))
    sizes = sorted((path.name, path.stat().st_size) for path in folder.iterdir() if path.is_file())
    too_big = [name for name, size in sizes if size >= LIMIT]
    if too_big:
        raise BundleError("over GitHub's 2 GiB per-file limit: " + ", ".join(too_big))
    return sizes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The release's own copies of what an install downloads")
    sub = parser.add_subparsers(dest="what", required=True)
    for name in ("upstream", "crates", "wheels", "notices", "check"):
        command = sub.add_parser(name)
        command.add_argument("folder", type=Path)
        if name == "wheels":
            command.add_argument("--platform", choices=PLATFORMS, required=True)
            command.add_argument("--uv", default="uv")
    args = parser.parse_args(argv)
    folder: Path = args.folder
    version = project_version()
    if args.what != "check":
        folder.mkdir(parents=True, exist_ok=True)
    try:
        if args.what == "upstream":
            made = upstream(folder)
        elif args.what == "crates":
            made = [crates(folder)]
        elif args.what == "wheels":
            made = [wheels(folder, args.platform, uv=args.uv)]
        elif args.what == "notices":
            made = [folder / NOTICES]
            made[0].write_text(notices(version), encoding="utf-8")
        else:
            sizes = check(folder, version)
            for name, size in sizes:
                print(f"{size / 1e6:10.1f} MB  {name}")
            print(f"{sum(size for _, size in sizes) / 1e6:10.1f} MB  in all")
            return 0
    except (BundleError, OSError, subprocess.CalledProcessError) as err:
        print(f"bundles: {err}", file=sys.stderr)
        return 1
    for path in made:
        print(f"{path.stat().st_size / 1e6:10.1f} MB  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
