"""The 3D match viewer: one page that draws a recorded or live game with three.js (vendored, MIT).

The page is built from the files in `web/` (the viewer's own script parts, joined in name order, its style,
its markup and the guard that speaks when the view cannot start) and `vendor/three.min.js`. Everything is
inside the page, so it opens from a file with no server and works offline. The arena is drawn from
RocketSim's collision meshes when they are given: inside the page (`arena`, for pages that stay on this
computer) or from a script file beside it (`arena_src`, written by `keep_arena_script`), so a page that is
passed on carries no mesh and falls back to a generated arena.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from pathlib import Path
from typing import Any

from nitro import brand, notices

WEB = Path(__file__).with_name("web")
VENDOR = Path(__file__).with_name("vendor")
_MARKS = ("/*GUARD*/", "/*STYLE*/", "/*MARKUP*/", "/*THREE*/", "/*VIEWER*/")
_DATA_MARK = "/*DATA*/null"
_ARENA_MARK = "<script>/*ARENA*/</script>"
_LOGO_MARK = "<!--LOGO-->"
_NOTICE_MARK = "<!--NOTICES-->"
# NitroLearn's folder, where its LICENSE is.
_PROJECT = Path(__file__).resolve().parents[3]
ARENA_FILE = "arena.js"
# In a run folder while its live window is open: that window's page, with the arena's meshes inside.
LIVE_PAGE = "live.html"


def script() -> str:
    """The viewer's own script: the parts in `web/`, joined in name order."""
    return "\n".join(part.read_text(encoding="utf-8") for part in sorted(WEB.glob("[0-9][0-9]_*.js")))


def guard() -> str:
    """The page's first script: it puts a plain message on the page when the 3D view cannot start."""
    return (WEB / "guard.js").read_text(encoding="utf-8")


def maker() -> str:
    """The viewer's script for a page that shows more than one game in its life (the dashboard): it defines
    the function `NV_MAKE`, which gives a viewer of its own (an `NV`) each time it is called. A viewer is
    started once and ended with its `stop()`; three.js is loaded once, before it."""
    return f'function NV_MAKE() {{\n"use strict";\n{script()}\nreturn NV;\n}}'


def three() -> str:
    return (VENDOR / "three.min.js").read_text(encoding="utf-8")


def css() -> str:
    """The viewer's style alone, for a page that already holds the brand fonts."""
    return (WEB / "viewer.css").read_text(encoding="utf-8")


def style() -> str:
    """The viewer's style, with the brand fonts inside it."""
    return brand.font_faces() + "\n" + css()


def markup() -> str:
    """The viewer's markup, with the logo where it is marked."""
    return (WEB / "markup.html").read_text(encoding="utf-8").replace(_LOGO_MARK, brand.logo())


def arena_blob(meshes: Path) -> str | None:
    """The soccar collision meshes in `meshes` (a folder with a `soccar` subfolder of .cmf files) as one
    base64 text: a count, then each file as its length and its bytes (little-endian u32). None if there
    are none."""
    files = sorted((meshes / "soccar").glob("*.cmf"), key=lambda file: (len(file.stem), file.stem))
    if not files:
        return None
    packed = struct.pack("<I", len(files))
    for file in files:
        content = file.read_bytes()
        packed += struct.pack("<I", len(content)) + content
    return base64.b64encode(packed).decode("ascii")


def arena_script(blob: str) -> str:
    """The script that hands the arena's meshes to the page (the content of a script file beside it)."""
    return f'window.NV_ARENA="{blob}";'


def keep_arena_script(folder: Path, meshes: Path) -> Path | None:
    """Makes sure `folder` holds the arena's script file (`arena.js`) for the meshes in `meshes`, writing
    it only when it is missing or holds something else. Returns the file, or None when there are no
    meshes."""
    blob = arena_blob(meshes)
    if blob is None:
        return None
    text = arena_script(blob)
    file = folder / ARENA_FILE
    try:
        same = file.read_text(encoding="utf-8") == text
    except (OSError, ValueError):
        same = False
    if not same:
        folder.mkdir(parents=True, exist_ok=True)
        partial = file.with_name(file.name + ".part")
        partial.write_text(text, encoding="utf-8")
        partial.replace(file)
    return file


def _comment(text: str) -> str:
    """`text` made safe inside an HTML comment: no `--` (which could end it) and no `-` at its end."""
    while "--" in text:
        text = text.replace("--", "- -")
    return text.rstrip("-").rstrip()


def _font_notices() -> str:
    """The brand fonts' copyright lines and their licence (the OFL asks that its text travel with every copy
    of the fonts, and the page holds them): read from the fonts' LICENSES.txt."""
    text = (brand.FONTS / "LICENSES.txt").read_text(encoding="utf-8")
    sections = re.split(r"^=====.*=====$", text, flags=re.MULTILINE)[1:]
    copyrights = re.findall(r"^Copyright \d.*$", text, flags=re.MULTILINE)
    name = re.search(r"licensed under the (.+?)\.?$", text, flags=re.MULTILINE)
    urls = dict.fromkeys(re.findall(r"^https?://\S+$", text, flags=re.MULTILINE))
    head = [*copyrights, ""]
    if name is not None:
        head.append(f"Licensed under the {name.group(1)}: {', '.join(urls)}")
    body = sections[0] if sections else text
    start = body.find("-" * 20)
    licence = body[start:].strip() if start >= 0 else body.strip()
    return "\n".join(head) + "\n\n" + licence


def notice_comment() -> str:
    """The comment at the top of a page that is passed on: NitroLearn's Required Notice and licence, and the
    notices and licences of three.js and the brand fonts that travel inside the page."""
    required = "\n".join(notices.required_notices(_PROJECT))
    three_licence = (VENDOR / "THREE-LICENSE.txt").read_text(encoding="utf-8").strip()
    text = (
        f"NitroLearn match viewer.\n{required}\nLicensed under the PolyForm Noncommercial License 1.0.0: "
        f"{notices.LICENCE_URL}\n\n"
        f"This page includes three.js (https://threejs.org):\n\n{three_licence}\n\n"
        f"This page includes the fonts Archivo and JetBrains Mono:\n\n{_font_notices()}"
    )
    return f"<!--\n{_comment(text)}\n-->"


def page(data: dict[str, Any], *, arena: str | None = None, arena_src: str | None = None) -> str:
    """The viewer page showing `data` (a recording, `nitro.watch`'s page data). `arena` is `arena_blob`'s
    text to put inside the page; `arena_src` the address of a script file holding `arena_script`'s text."""
    text = (WEB / "viewer.html").read_text(encoding="utf-8").replace(_NOTICE_MARK, notice_comment(), 1)
    for mark, part in zip(_MARKS, (guard(), style(), markup(), three(), script()), strict=True):
        text = text.replace(mark, part, 1)
    if arena is not None:
        arena_tag = f"<script>{arena_script(arena)}</script>"
    elif arena_src is not None:
        arena_tag = f'<script src="{_escape(arena_src)}"></script>'
    else:
        arena_tag = ""
    text = text.replace(_ARENA_MARK, arena_tag, 1)
    title = _escape(str(data.get("title", "")))
    text = text.replace("<title></title>", f"<title>{title}</title>", 1)
    return text.replace(
        _DATA_MARK, json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"), 1
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
