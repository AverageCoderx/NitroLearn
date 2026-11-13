"""The notices file `nitro export` writes into every bot folder (`NOTICES.md`), and the release's
`THIRD_PARTY_NOTICES.md` for its ready-made programs (`python python/nitro/notices.py programs`).

The bot program is built from NitroLearn's own crates and the third-party open-source crates they use
(RLBot's interface, RocketSim, serde, glam, zstd and the usual permissive libraries). The file carries what
travels with it: NitroLearn's Required Notice and licence, the permission BOTS-LICENSE.md gives for a bot
folder, and the copyright notices and licences of the third-party code compiled into the program.

The third-party list is the bot program's own build graph, read with `cargo tree -p nitro_bot -e
normal,no-proc-macro`, so it holds exactly the crates linked into the program (nothing from the rest of the
workspace, and no build-time helpers). Each crate's licence comes from its own metadata (`cargo metadata`)
and its copyright line from the crate's licence file in cargo's cache, never guessed; a crate published
without a licence file takes the line from its repository's licence file (`_UPSTREAM`), and any other
crate without a copyright line is credited by its repository. Where a crate offers a choice of licences,
the first of `_PREFERRED` it offers is the one used, and only the licence texts so used are printed (with
the NOTICE file of every crate used under Apache-2.0, as that licence asks). Both cargo commands run
offline.

The release's programs: nitro_bot and nitro-replicar (NitroLearn's own) and replicar (replicar-cli, built
unchanged from crates.io). The list for each is its own build graph for the target it was built for, read
in the job that built it, so the crates are in cargo's cache; unlike a bot folder's notices, any crate whose
licence cannot be read or has no text here stops the list rather than leave a gap.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

FILE = "NOTICES.md"
PROGRAMS_FILE = "THIRD_PARTY_NOTICES.md"  # beside the release's programs, and beside the compiled core
CORE_MANIFEST = Path("crates") / "nitro_py" / "Cargo.toml"  # the core's crate (nitro._core)
BOT_CRATE = "nitro_bot"
REPLICAR_CLI = "replicar-cli"
LICENCE_URL = "https://polyformproject.org/licenses/noncommercial/1.0.0"
REPOSITORY = "https://github.com/AverageCoderx/NitroLearn"
# NitroLearn's own crates, left out of the third-party list.
_OWN = {"nitro_bot", "nitro_core", "nitro_py", "nitro_replicar"}
# Licence files to read a copyright line from, in the order preferred (a crate's own MIT/BSD/Zlib text
# before the shared Apache text, which carries only the licence's own boilerplate).
_LICENSE_FILES = (
    "LICENSE-MIT",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYRIGHT",
    "NOTICE",
    "COPYING",
    "LICENSE-UNICODE",
    "LICENSE-BSD",
    "LICENSE.BSD-3-Clause",
    "LICENSE-APACHE",
    "UNLICENSE",
)
# A real copyright notice: the word Copyright followed by (c)/a year/©, not the Apache appendix template.
_COPYRIGHT = re.compile(r"^Copyright\s*(\(c\)|\(C\)|©|\d|\[)", re.IGNORECASE)
# Code a crate compiles in from another project, with that project's own licence file in the crate:
# (what it is, its licence file in the crate's folder, its SPDX licence).
_BUNDLED = {
    "zstd-sys": ("Zstandard, the C library it builds", "zstd/LICENSE", "BSD-3-Clause"),
    "libmimalloc-sys": ("mimalloc, the C library it builds", "c_src/mimalloc/v2/LICENSE", "MIT"),
}
# Crates published without a licence file: the copyright line of their repository's LICENSE.
_REPLICAR = "Copyright (c) 2026 Rolv Arild"
_UPSTREAM = {"replicar": _REPLICAR, "replicar-format": _REPLICAR, REPLICAR_CLI: _REPLICAR}
# Licences with a text here, in the order one is chosen where a crate offers several.
_PREFERRED = ("MIT", "BSD-3-CLAUSE", "ZLIB", "UNLICENSE", "APACHE-2.0")
# Apache-2.0's full text, in NitroLearn's folder.
APACHE_FILE = Path("LICENSES") / "Apache-2.0.txt"

_MIT = """\
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

_BSD_3 = """\
BSD 3-Clause License

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
   the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
   promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

_ZLIB = """\
zlib License

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be
held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and
to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original
   software. If you use this software in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the
   original software.
3. This notice may not be removed or altered from any source distribution.
"""

_UNLICENSE = """\
The Unlicense

This is free and unencumbered software released into the public domain. Anyone is free to copy, modify,
publish, use, compile, sell, or distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any means.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE. For more information, please refer to <https://unlicense.org/>.
"""

_APACHE = """\
Apache License, Version 2.0

Some components are offered under the Apache License, Version 2.0. The full text is at
<https://www.apache.org/licenses/LICENSE-2.0>. Software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. Where a component
is offered as "MIT OR Apache-2.0" (or the reverse), the MIT text above applies as the chosen license.
"""

# The licence family each SPDX expression draws on, so the right texts are printed for the crates present.
_TEXTS = (
    ("MIT", _MIT),
    ("BSD-3-CLAUSE", _BSD_3),
    ("ZLIB", _ZLIB),
    ("UNLICENSE", _UNLICENSE),
    ("APACHE-2.0", _APACHE),
)

_CHOICE = (
    "Where a component offers a choice of licences, it is used under the first of these it offers: MIT, "
    "BSD-3-Clause, zlib, the Unlicense, Apache-2.0."
)

# What NitroLearn's LICENSE says first, when its file cannot be read.
_REQUIRED_NOTICE = f"Required Notice: Copyright (c) 2026 AverageCoderx ({REPOSITORY})"


def _toolchain() -> Any:
    """nitro.toolchain, also when this file runs alone (the release builds the programs without NitroLearn
    installed)."""
    try:
        from nitro import toolchain
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "nitro_toolchain", Path(__file__).with_name("toolchain.py")
        )
        assert spec is not None and spec.loader is not None
        toolchain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(toolchain)
    return toolchain


def _run(project: Path, args: list[str]) -> str | None:
    """A cargo command in `project` (the folder's own Rust first, as for a build), its stdout, or None if
    cargo is missing or fails."""
    rust = _toolchain()
    cargo = rust.find_cargo(project)
    if cargo is None:
        return None
    try:
        done = subprocess.run(
            [str(cargo), *args],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=rust.cargo_env(project),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _graph_names(
    project: Path, select: Sequence[str] = ("-p", BOT_CRATE), target: str | None = None
) -> set[tuple[str, str]]:
    """The (name, version) of every crate linked into the program cargo's `select` arguments pick (a
    workspace package, or another manifest), NitroLearn's own left out: its normal-dependency graph without
    procedural macros (which run while building and are not linked), for `target` (default: this
    computer's)."""
    args = ["tree", *select, "-e", "normal,no-proc-macro", "--prefix", "none", "--format", "{p}"]
    out = _run(project, [*args, *(["--target", target] if target else []), "--offline", "--locked"])
    names: set[tuple[str, str]] = set()
    for line in (out or "").splitlines():
        match = re.match(r"^(\S+) v(\d[^\s]*)", line.strip())
        if match and match.group(1) not in _OWN:
            names.add((match.group(1), match.group(2)))
    return names


def _copyright_lines(path: Path) -> list[str]:
    """The real copyright notices in the licence file `path` (none when it cannot be read)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        template = "[yyyy]" in line or "name of copyright owner" in line
        if _COPYRIGHT.match(line) and not template and line not in found:
            found.append(line)
    return found


def _copyright_line(manifest_dir: Path) -> str | None:
    """Every copyright notice in the crate's first licence file that has one (the crate's own folder, or one
    level up for a crate whose licence sits in its workspace root), joined; or None. All of them, so a crate
    with two holders keeps both."""
    for base in (manifest_dir, manifest_dir.parent):
        try:
            present = {name.upper(): name for name in (p.name for p in base.iterdir())}
        except OSError:
            continue
        for want in _LICENSE_FILES:
            name = present.get(want.upper())
            if name is not None and (hits := _copyright_lines(base / name)):
                return "; ".join(hits)
    return None


class Component(NamedTuple):
    label: str  # name and version: "serde 1.0.228"
    licence: str  # its SPDX expression
    credit: str  # its copyright line, else its repository
    folder: Path | None  # its files in cargo's cache


def _packages(project: Path, select: Sequence[str] = ()) -> dict[tuple[str, str], dict[str, Any]]:
    """`cargo metadata`'s packages by (name, version), for the workspace or the manifest `select` names."""
    manifest = list(select) if select[:1] == ("--manifest-path",) else []
    meta_out = _run(project, ["metadata", "--format-version", "1", "--offline", "--locked", *manifest])
    meta: dict[str, Any] = json.loads(meta_out) if meta_out else {}
    return {(p["name"], p["version"]): p for p in meta.get("packages", [])}


def _components(
    project: Path, select: Sequence[str] = ("-p", BOT_CRATE), target: str | None = None, strict: bool = False
) -> list[Component]:
    """Each crate linked into the program (`_graph_names`), sorted; with a line for code a crate bundles
    from another project. `strict`: NoticesError when cargo cannot give the crates' details (licences)."""
    names = _graph_names(project, select, target)
    if not names:
        return []
    by_key = _packages(project, tuple(select))
    if strict and not by_key:
        raise NoticesError(
            "cargo could not read the crates' details offline: their sources (every platform's) are not all "
            "in cargo's cache. Download them first (--fetch, or cargo fetch --locked)."
        )
    rows: list[Component] = []
    for name, version in sorted(names):
        package = by_key.get((name, version), {})
        licence = package.get("license") or package.get("license_file") or "see the crate's own licence"
        manifest = package.get("manifest_path")
        folder = Path(manifest).parent if manifest else None
        found = (_copyright_line(folder) if folder else None) or _UPSTREAM.get(name)
        credit = found or package.get("repository") or ""
        rows.append(Component(f"{name} {version}", licence, credit, folder))
        if name in _BUNDLED and folder is not None:
            what, file, bundled_licence = _BUNDLED[name]
            lines = _copyright_lines(folder / file)
            if lines:
                rows.append(Component(f"{name} {version}: {what}", bundled_licence, "; ".join(lines), None))
    return rows


def _choices(expression: str) -> list[frozenset[str]]:
    """An SPDX licence expression as its alternatives, each the set of licences it asks for together:
    "(Apache-2.0 OR MIT) AND BSD-3-Clause" -> [{APACHE-2.0, BSD-3-CLAUSE}, {MIT, BSD-3-CLAUSE}]. The old
    "MIT/Apache-2.0" form reads as OR; an exception after WITH only adds permissions, so it is dropped."""
    tokens = re.findall(r"\(|\)|[^\s()/]+|/", expression.upper())
    tokens = ["OR" if token == "/" else token for token in tokens]
    at = 0

    def any_of() -> list[frozenset[str]]:
        nonlocal at
        found = all_of()
        while at < len(tokens) and tokens[at] == "OR":
            at += 1
            found += all_of()
        return found

    def all_of() -> list[frozenset[str]]:
        nonlocal at
        found = one()
        while at < len(tokens) and tokens[at] == "AND":
            at += 1
            right = one()
            found = [a | b for a in found for b in right]
        return found

    def one() -> list[frozenset[str]]:
        nonlocal at
        token = tokens[at] if at < len(tokens) else ""
        at += 1
        if token == "(":
            found = any_of()
            at += 1  # ")"
            return found
        if at < len(tokens) and tokens[at] == "WITH":
            at += 2
        return [frozenset({token})]

    return any_of() if tokens else []


def _chosen(expression: str) -> frozenset[str] | None:
    """The licences a crate is used under: of its alternatives with a text here, the one whose least
    preferred licence comes first in `_PREFERRED`; None when no alternative has every text."""
    known = [choice for choice in _choices(expression) if choice <= set(_PREFERRED)]
    if not known:
        return None
    return min(known, key=lambda choice: (max(_PREFERRED.index(t) for t in choice), len(choice)))


def _apache_text(project: Path | None) -> str:
    """Apache-2.0's full text from NitroLearn's folder; the short form when it cannot be read."""
    try:
        text = (project / APACHE_FILE).read_text(encoding="utf-8") if project else ""
    except OSError:
        text = ""
    return f"Apache License, Version 2.0\n\n```\n{text.strip()}\n```\n" if text else _APACHE


def _needed_texts(rows: Sequence[Component], project: Path | None = None) -> str:
    """The licence texts the components are used under (each once, in a fixed order); every common
    permissive text when none could be chosen."""
    used = set().union(*(_chosen(row.licence) or () for row in rows))
    blocks = [
        _apache_text(project) if token == "APACHE-2.0" else body for token, body in _TEXTS if token in used
    ]
    if not blocks:  # a list no licence could be matched in: the common permissive texts
        blocks = [_MIT, _APACHE, _BSD_3, _ZLIB]
    return "\n".join(f"### {body}" for body in blocks)


def _notice_files(rows: Sequence[Component]) -> list[tuple[str, str]]:
    """The NOTICE file of every component used under Apache-2.0 that has one, as (components, text), each
    text once."""
    found: dict[str, list[str]] = {}
    for row in rows:
        if row.folder is None or "APACHE-2.0" not in (_chosen(row.licence) or ()):
            continue
        for name in ("NOTICE", "NOTICE.txt", "NOTICE.md"):
            try:
                text = (row.folder / name).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            found.setdefault(text, []).append(row.label)
            break
    return [(", ".join(labels), text) for text, labels in found.items()]


def _notice_section(rows: Sequence[Component], level: str = "###") -> str:
    """The NOTICE files the components used under Apache-2.0 carry, as a section (empty without any)."""
    files = _notice_files(rows)
    if not files:
        return ""
    blocks = "\n".join(f"{level}# For {labels}\n\n```\n{text}\n```\n" for labels, text in files)
    why = "Apache-2.0 asks that these travel with the components it covers."
    return f"{level} NOTICE files\n\n{why}\n\n{blocks}\n"


@functools.cache
def third_party(project: Path) -> str:
    """The third-party part of the notices: the crates linked into the bot program built from `project`."""
    rows = _components(project)
    if rows:
        components = "\n".join(f"- {row.label}: {row.licence}" for row in rows)
        notices = "\n".join(f"- {row.label}: {row.credit}" for row in rows if row.credit)
        texts = _CHOICE + "\n\n" + _notice_section(rows) + _needed_texts(rows, project)
    else:
        # cargo could not describe the build graph (no Rust installed, or no crates downloaded yet): name the
        # licences that apply and point at the lockfile rather than list a wrong set.
        components = (
            "- The third-party Rust crates in NitroLearn's Cargo.lock that the bot program links, each under "
            "a permissive open-source licence (MIT, Apache-2.0, BSD-3-Clause, Zlib or the Unlicense)."
        )
        notices = "- See each crate's repository and licence file for its copyright notice."
        texts = f"### {_MIT}\n### {_APACHE}\n### {_BSD_3}\n### {_ZLIB}\n### {_UNLICENSE}"
    return f"""## Third-party software in the bot program

The bot program includes open-source software from the components below. Their copyright notices and
licences travel with the program, as those licences require. The list is the bot program's own build graph.

### Components
{components}

### Copyright notices
{notices}

### Licences
{texts}"""


def required_notices(project: Path) -> list[str]:
    """The `Required Notice:` lines of NitroLearn's LICENSE (the licence asks that they travel with it)."""
    try:
        text = (project / "LICENSE").read_text(encoding="utf-8")
    except OSError:
        return [_REQUIRED_NOTICE]
    return [line.strip() for line in text.splitlines() if line.startswith("Required Notice:")] or [
        _REQUIRED_NOTICE
    ]


def _bots_licence(project: Path) -> str:
    """BOTS-LICENSE.md as it reads in NitroLearn's folder, its headings two levels down and its links as plain
    words (the files they point at are not in a bot folder)."""
    try:
        text = (project / "BOTS-LICENSE.md").read_text(encoding="utf-8")
    except OSError:
        return f"NitroLearn's BOTS-LICENSE.md, in its repository: {REPOSITORY}/blob/main/BOTS-LICENSE.md"
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"^(#+) ", lambda m: "##" + m.group(1) + " ", text, flags=re.MULTILINE).strip()


def notices(project: Path) -> str:
    """The text of the bot folder's NOTICES.md, from NitroLearn's folder `project`."""
    required = "\n".join(required_notices(project))
    return f"""# Notices

This bot folder was made with NitroLearn, <{REPOSITORY}>. Keep this file with the bot program and
the other files the export put here.

## NitroLearn

{required}

NitroLearn is licensed under the PolyForm Noncommercial License 1.0.0,
<{LICENCE_URL}>. The bot folder itself is covered by the additional
permission below.

## Your bot: the additional permission (NitroLearn's BOTS-LICENSE.md)

{_bots_licence(project)}

## Fair play

Use this bot offline only: RLBot matches, private matches against your own bots, and bot tournaments. Never
use it in online matches against real players; that is cheating and breaks Rocket League's terms of
service.

{third_party(project)}"""


# ---------- the release's programs ----------


class NoticesError(RuntimeError):
    """The release's notices cannot be made complete; the message says what is missing."""


def registry_manifest(name: str, version: str) -> Path:
    """The manifest of crate `name` `version` in cargo's cache (put there by `cargo install`)."""
    home = Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo")
    found = sorted(home.glob(f"registry/src/*/{name}-{version}/Cargo.toml"))
    if not found:
        raise NoticesError(f"{name} {version} is not in cargo's cache ({home}): build it first")
    return found[-1]


def programs(project: Path, replicar_version: str, target: str | None = None) -> dict[str, list[str]]:
    """The release's programs, each with the cargo arguments that pick its crate."""
    return {
        "nitro_bot": ["-p", BOT_CRATE],
        "nitro-replicar": ["--manifest-path", str(project / "crates" / "nitro_replicar" / "Cargo.toml")],
        "replicar": ["--manifest-path", str(registry_manifest(REPLICAR_CLI, replicar_version))],
    }


def _document(project: Path, intro: str, chosen: dict[str, list[str]], target: str | None) -> str:
    """A THIRD_PARTY_NOTICES.md: `intro`, then every crate linked into each of `chosen` (name -> cargo's
    arguments that pick its crate) built for `target`, with the copyright notices, NOTICE files and licence
    texts they ask for. Raises NoticesError rather than leave any of that out."""
    lists: dict[str, list[Component]] = {}
    for name, select in chosen.items():
        rows = _components(project, select, target, strict=True)
        if not rows:
            raise NoticesError(f"cargo could not list the crates linked into {name} (is it built here?)")
        lists[name] = rows
    every = {row.label: row for rows in lists.values() for row in rows}
    unknown = sorted(f"{row.label} ({row.licence})" for row in every.values() if _chosen(row.licence) is None)
    if unknown:
        raise NoticesError("no licence text here for: " + ", ".join(unknown))
    rows = [every[label] for label in sorted(every)]
    used = set().union(*(_chosen(row.licence) or () for row in rows))
    if "APACHE-2.0" in used and not (project / APACHE_FILE).is_file():
        raise NoticesError(f"Apache-2.0's text is missing ({project / APACHE_FILE})")
    sections = "\n".join(
        f"## {name}\n\n" + "\n".join(f"- {row.label}: {row.licence}" for row in found) + "\n"
        for name, found in lists.items()
    )
    credits = "\n".join(f"- {row.label}: {row.credit}" for row in rows if row.credit)
    built_for = f"the {target} target" if target else "the computer it was built on"
    return f"""# Third-party notices

{intro}

Each includes the open-source components listed under its name; their copyright notices and licences
follow, as those licences require. Each list is its own build graph, for {built_for}.

{sections}
## Copyright notices

{credits}

{_notice_section(rows, "##")}## Licences

{_CHOICE}

{_needed_texts(rows, project)}"""


def _nitrolearn(project: Path, indent: str = "  ") -> str:
    """NitroLearn's Required Notice and licence, each line indented by `indent`."""
    lines = [
        *required_notices(project),
        "NitroLearn is licensed under the PolyForm Noncommercial License 1.0.0,",
    ]
    return "\n".join(f"{indent}{line}" for line in [*lines, f"<{LICENCE_URL}>."])


def program_notices(
    project: Path, chosen: dict[str, list[str]], replicar_version: str, target: str | None = None
) -> str:
    """THIRD_PARTY_NOTICES.md for the programs `chosen` (`programs`), built for `target`."""
    intro = f"""The programs next to this file, and what they are built from:

- nitro_bot and nitro-replicar are NitroLearn's own programs, <{REPOSITORY}>.
{_nitrolearn(project)}
- replicar is replicar-cli {replicar_version}, the replay converter by Rolv Arild
  (<https://github.com/Rolv-Arild/replicar>), built unchanged from crates.io under its MIT licence."""
    return _document(project, intro, chosen, target)


def core_notices(project: Path, target: str | None = None) -> str:
    """THIRD_PARTY_NOTICES.md for NitroLearn's compiled core (nitro._core, the nitro_py crate), built for
    `target`: beside the core in the release's wheel, and written by the installer after a build from the
    code."""
    intro = f"""NitroLearn's compiled core (nitro._core, next to this file) is NitroLearn's own code,
<{REPOSITORY}>, built with the open-source Rust crates below.

{_nitrolearn(project, "")}"""
    return _document(
        project, intro, {"nitro._core": ["--manifest-path", str(project / CORE_MANIFEST)]}, target
    )


def fetch(project: Path, chosen: dict[str, list[str]]) -> None:
    """Downloads the sources of every crate `chosen` may use, for every platform (`cargo fetch`, online), so
    the offline listing finds them all."""
    rust = _toolchain()
    cargo = rust.find_cargo(project)
    if cargo is None:
        raise NoticesError("cargo was not found")
    for name, select in chosen.items():
        manifest = list(select) if select[:1] == ["--manifest-path"] else []
        done = subprocess.run(
            [str(cargo), "fetch", "--locked", *manifest],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
            env=rust.cargo_env(project),
        )
        if done.returncode != 0:
            raise NoticesError(f"cargo fetch failed for {name}: {done.stderr.strip()[-500:]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The licence notices of NitroLearn's compiled parts")
    parser.add_argument("what", choices=["programs", "core"], help="the release's programs, or the core")
    parser.add_argument("--out", type=Path, required=True, help=f"the file to write ({PROGRAMS_FILE})")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--replicar-version", help="the replicar-cli version built (programs)")
    parser.add_argument("--target", help="the target they were built for (default: this computer)")
    parser.add_argument("--fetch", action="store_true", help="download every crate's sources first (online)")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if args.what == "programs" and not args.replicar_version:
        parser.error("programs needs --replicar-version")
    try:
        if args.what == "core":
            if args.fetch:
                fetch(project, {"nitro._core": ["--manifest-path", str(project / CORE_MANIFEST)]})
            text = core_notices(project, args.target)
        else:
            chosen = programs(project, args.replicar_version, args.target)
            if args.fetch:
                fetch(project, chosen)
            text = program_notices(project, chosen, args.replicar_version, args.target)
    except NoticesError as err:
        print(f"notices: {err}", file=sys.stderr)
        return 1
    args.out.write_text(text, encoding="utf-8")
    print(f"notices: {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
