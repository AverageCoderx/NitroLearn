"""Gets NitroLearn-Setup.exe's inputs ready in installer/build, for Inno Setup (installer/NitroLearn.iss):

    python tools/setup_wizard.py [--uv-wheel FILE]

    installer/build/app/          NitroLearn's files as the release ships them (`git archive HEAD`)
    installer/build/uv/           uv.exe and its two licences, from the pinned uv wheel (tools/bundles.py),
                                  checked against its hash: the setup carries uv, everything else downloads
    installer/build/license.rtf   the licence page: fair play, the licence and "your bots are yours" in short,
                                  then LICENSE and BOTS-LICENSE.md in full

The release workflow runs this on Windows, then
`ISCC /DAppVersion=<Cargo.toml's version> installer/NitroLearn.iss`.
Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "installer" / "build"
# The folders the user's own things live in: the uninstaller asks before removing them (NitroLearn.iss).
USER_FOLDERS = ("runs", "my-settings", "packs", "replays", "bots", "ab", "wandb", "saves")


def _bundles() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nitro_bundles_for_setup", ROOT / "tools" / "bundles.py")
    if spec is None or spec.loader is None:
        raise SystemExit("tools/bundles.py is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fair_play(readme: str) -> str:
    """The README's fair-play section, as plain text."""
    found = re.search(r"^## Fair play[^\n]*\n(.*?)(?=^## )", readme, re.MULTILINE | re.DOTALL)
    if found is None:
        raise SystemExit("README.md has no fair-play section")
    return plain(found[1])


def plain(markdown: str) -> str:
    """Markdown as plain text: links as their words, no bold marks, and the lines of each paragraph or list
    item joined (the licence page wraps them itself)."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: m[1], markdown)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text).replace("**", "")
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines: list[str] = []
        for line in block.splitlines():
            if lines and not re.match(r"\s*(?:[-*] |\d+\. |#|\|)", line):
                lines[-1] += " " + line.strip()
            else:
                lines.append(line.rstrip())
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


SUMMARY = (
    (
        "Licence: free for non-commercial use",
        "PolyForm Noncommercial 1.0.0. You may use, change and share NitroLearn for any non-commercial "
        "purpose. Selling NitroLearn, or a changed copy of it, is not allowed.",
    ),
    (
        "Your bots are yours",
        "Bots you train with NitroLearn are yours to use, copy, change, share and license, for any purpose "
        "except online matches against real players (BOTS-LICENSE.md).",
    ),
)


def _rtf_text(text: str) -> str:
    out = []
    for char in text:
        if char in "\\{}":
            out.append("\\" + char)
        elif char == "\n":
            out.append("\\line ")
        elif ord(char) > 127:
            out.append(f"\\u{ord(char) if ord(char) < 32768 else ord(char) - 65536}?")
        else:
            out.append(char)
    return "".join(out)


def licence_rtf(root: Path = ROOT) -> str:
    """The licence page: the three points in short, then the full texts NitroLearn is licensed under."""
    readme = (root / "README.md").read_text(encoding="utf-8")
    sections = [("Fair play: NitroLearn is not a cheating tool", fair_play(readme)), *SUMMARY]
    parts = [r"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fswiss Segoe UI;}}\f0\fs18 "]
    for heading, body in sections:
        parts.append(rf"{{\b\fs20 {_rtf_text(heading)}}}\par {_rtf_text(body)}\par\par ")
    parts.append(r"{\b\fs20 The full terms}\par The licence (LICENSE) and the permission for your bots ")
    parts.append(r"(BOTS-LICENSE.md) follow. They are the terms that count.\par\par ")
    for name in ("LICENSE", "BOTS-LICENSE.md"):
        text = plain((root / name).read_text(encoding="utf-8"))
        parts.append(rf"{{\b {name}}}\par {_rtf_text(text)}\par\par ")
    parts.append("}")
    return "".join(parts)


def app_tree(target: Path, root: Path = ROOT) -> None:
    """NitroLearn's files as shipped: what `git archive` gives for the commit checked out."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"], cwd=root, capture_output=True, check=True
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(target, filter="data")


def uv_files(target: Path, wheel: Path | None) -> None:
    """uv.exe and its licences from the pinned Windows uv wheel (downloaded unless given), hash-checked."""
    bundles = _bundles()
    pinned = bundles.UV["windows-x64"]
    with tempfile.TemporaryDirectory() as temp:
        if wheel is None:
            wheel = bundles.download(pinned.url, Path(temp) / pinned.asset, pinned.sha256)
        elif bundles.sha256_file(wheel) != pinned.sha256:
            raise SystemExit(f"{wheel} is not the pinned uv wheel")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel) as zipped:
            for name in zipped.namelist():
                base = name.rsplit("/", 1)[-1]
                if name.endswith("/scripts/uv.exe") or base in ("LICENSE-MIT", "LICENSE-APACHE"):
                    (target / base).write_bytes(zipped.read(name))
    if not (target / "uv.exe").is_file():
        raise SystemExit("the uv wheel holds no uv.exe")


def stage(build: Path = BUILD, root: Path = ROOT, wheel: Path | None = None) -> None:
    if build.exists():
        shutil.rmtree(build)
    app_tree(build / "app", root)
    uv_files(build / "uv", wheel)
    (build / "license.rtf").write_text(licence_rtf(root), encoding="ascii")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NitroLearn-Setup.exe's inputs, in installer/build")
    parser.add_argument("--uv-wheel", type=Path, help="the pinned uv wheel for Windows, already downloaded")
    args = parser.parse_args(argv)
    stage(wheel=args.uv_wheel)
    print(f"Ready in {BUILD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
