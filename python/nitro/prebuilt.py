"""Ready-made programs from a NitroLearn release: the bot program (nitro_bot) and the two replay converters
(replicar and nitro-replicar). The installer puts them in .tools/prebuilt with `prebuilt.json`, which records
what each was built from. A program is used only while that matches the code in this folder; otherwise (an
edited or newer folder) it is built with cargo, as without them.

    python -m nitro.prebuilt <NitroLearn folder> <folder of the programs>

writes that folder's prebuilt.json (the release build runs it).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(".tools") / "prebuilt"
LIST = "prebuilt.json"
# The bot program's code: its own crate and the core it shares with training.
BOT_DIRS = ("crates/nitro_bot", "crates/nitro_core")
NITRO_REPLICAR = Path("crates") / "nitro_replicar"


def file_name(program: str) -> str:
    return f"{program}.exe" if sys.platform == "win32" else program


def bot_source(root: Path = ROOT) -> str:
    """The fingerprint of the bot program's code in `root` (empty when it has none)."""
    from nitro.build_info import source_hash

    return source_hash(root, BOT_DIRS) or ""


def nitro_replicar_source(root: Path = ROOT) -> str:
    """nitro-replicar's fingerprint: what decides whether a built copy is still current. Line endings do
    not count (a Windows checkout has CRLF)."""
    source = root / NITRO_REPLICAR
    parts = ("Cargo.toml", "Cargo.lock", "src/main.rs")
    text = b"".join((source / part).read_bytes().replace(b"\r\n", b"\n") for part in parts)
    return hashlib.sha256(text).hexdigest()


def fingerprints(root: Path = ROOT) -> dict[str, str]:
    """What each program is built from in `root`, as prebuilt.json records it."""
    from nitro.replays import REPLICAR_VERSION

    return {
        "nitro_bot": bot_source(root),
        "nitro-replicar": nitro_replicar_source(root),
        "replicar": REPLICAR_VERSION,
    }


def program(name: str, wanted: str, root: Path = ROOT) -> Path | None:
    """The ready-made program `name` in `root`'s .tools/prebuilt when it was built from `wanted` (its
    fingerprint), else None."""
    folder = root / FOLDER
    try:
        listed = json.loads((folder / LIST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    built = listed.get("programs", {}) if isinstance(listed, dict) else {}
    path = folder / file_name(name)
    if not wanted or not isinstance(built, dict) or built.get(name) != wanted or not path.is_file():
        return None
    return path


def write_list(root: Path, folder: Path) -> Path:
    """Writes `folder`/prebuilt.json for programs built from `root`'s code."""
    target = folder / LIST
    target.write_text(json.dumps({"programs": fingerprints(root)}, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m nitro.prebuilt <NitroLearn folder> <folder of the programs>")
    print(write_list(Path(sys.argv[1]), Path(sys.argv[2])).read_text(encoding="utf-8"), end="")
