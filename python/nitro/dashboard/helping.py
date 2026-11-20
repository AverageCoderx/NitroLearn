"""The Help page's Python side: the handbook's pages (docs/guide), the training coach of a bot, and the
report for Discord: versions, the last Check this PC, the bot's settings and the end of its console, with
the user's name and folders taken out."""

from __future__ import annotations

import getpass
import importlib.metadata
import os
import platform
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nitro import __version__
from nitro.dashboard import coach
from nitro.dashboard.jobs import read_console
from nitro.runs import RunFolder

GUIDE = Path("docs") / "guide"
# A handbook page by its file name, or a page of the docs folder above it as "../name.md".
_PAGE = re.compile(r"^(\.\./)?[A-Za-z0-9][A-Za-z0-9_-]*\.md$")
CONSOLE_LINES = 40  # the end of the bot's console in a report
REPORT_LIMIT = 1900  # Discord's message limit is 2000 characters: a longer report goes as a file
NAME_MIN = 3  # a user name shorter than this is too common a word to take out everywhere


def page_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


class HelpApi:
    """Mixed into `Api`: every public method is callable from the page."""

    _project: Path
    _runs: Path
    _last_checks: dict[str, Any] | None = None

    if TYPE_CHECKING:

        def _folder(self, name: str) -> RunFolder: ...

    def guide_pages(self) -> dict[str, Any]:
        """The handbook's pages in the index's order (docs/guide/README.md's links), each with its title."""
        folder = self._project / GUIDE
        index = folder / "README.md"
        if not index.is_file():
            return {"error": f"the handbook is missing ({GUIDE / 'README.md'})"}
        text = index.read_text(encoding="utf-8")
        pages: list[dict[str, str]] = []
        for found in re.finditer(r"\]\(([a-z0-9-]+\.md)\)", text):
            page = found[1]
            if page not in [p["page"] for p in pages] and (folder / page).is_file():
                title = page_title((folder / page).read_text(encoding="utf-8"), page)
                pages.append({"page": page, "title": title})
        return {"pages": pages, "index": text}

    def guide_page(self, page: str) -> dict[str, Any]:
        """One handbook page's Markdown, by its file name (README.md is the index)."""
        if not _PAGE.match(page):
            return {"error": f"there is no handbook page called {page}"}
        file = self._project / GUIDE / page
        if not file.is_file():
            return {"error": f"there is no handbook page called {page}"}
        text = file.read_text(encoding="utf-8")
        return {"page": page, "title": page_title(text, page), "text": text}

    def coach(self, name: str) -> dict[str, Any]:
        """The training coach's plan and hints for the bot (suggestions only)."""
        try:
            return coach.advise(self._folder(name))
        except (OSError, ValueError) as err:
            return {"error": str(err) or type(err).__name__}

    def discord_report(self, name: str = "") -> dict[str, Any]:
        """The report to paste into a Discord message, scrubbed of the user's name and folders. `name`: the
        bot to describe, or none."""
        try:
            folder = self._folder(name) if name else None
        except ValueError as err:
            return {"error": str(err)}
        text = report(self._project, self._runs, folder, self._last_checks)
        return {"text": text, "long": len(text) > REPORT_LIMIT, "checked": self._last_checks is not None}


def versions() -> list[str]:
    lines = [f"NitroLearn {__version__}", f"Python {platform.python_version()}"]
    for package in ("torch", "pywebview"):
        try:
            lines.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{package}: not installed")
    lines.append(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    return lines


def report(project: Path, runs: Path, folder: RunFolder | None, checks: dict[str, Any] | None) -> str:
    parts = ["**NitroLearn report**", "```", *versions(), "```"]
    parts.append("**Check this PC**")
    if checks is None or "checks" not in checks:
        parts.append("(not run yet: press Check this PC, then copy the report again)")
    else:
        lines = [
            f"{c['state'].upper()}: {c['name']}: {c['message']}"
            for c in checks["checks"]
            if c["state"] != "ok"
        ]
        ok = sum(c["state"] == "ok" for c in checks["checks"])
        parts += ["```", f"{ok} checks OK", *lines, str(checks.get("verdict", "")), "```"]
    if folder is not None:
        parts.append(f"**Bot: {folder.root.name}**")
        try:
            settings = folder.settings.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeDecodeError) as err:
            settings = f"(its settings could not be read: {type(err).__name__})"
        parts += ["Settings:", "```toml", settings, "```"]
        text, _ = read_console(folder.root, 0, limit=20_000)
        tail = text.rstrip("\n").splitlines()[-CONSOLE_LINES:]
        parts += (
            [f"The console's last {len(tail)} lines:", "```", *tail, "```"] if tail else ["(no console yet)"]
        )
    return scrub("\n".join(parts), project, runs)


def scrub(text: str, project: Path, runs: Path, user: str | None = None, home: Path | None = None) -> str:
    """`text` with the user's folders and name replaced by placeholders: NitroLearn's folder, the runs
    folder, the home folder, any Windows or Linux user folder, and the user name wherever it appears."""
    user = user if user is not None else _user()
    home = home if home is not None else Path.home()
    for path, word in ((runs, "<runs>"), (project, "<NitroLearn>"), (home, "<home>")):
        for form in {str(path), str(path).replace("\\", "/"), str(path).replace("/", "\\")}:
            if len(form) > 3:
                text = re.compile(re.escape(form), re.IGNORECASE).sub(word, text)
    # A user folder's name may hold spaces ("Sam Player"): taken whole up to the next separator.
    text = re.sub(
        r"(?im)\b([A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+)"
        r"(?:[^\\/\s\"'<>|]+(?: [^\\/\s\"'<>|]+)*(?=[\\/\"'<>|]|$)|[^\\/\s\"'<>|]+)",
        r"\1<you>",
        text,
    )
    text = re.sub(
        r"(?m)(/home/|/Users/)(?:[^/\s\"'<>]+(?: [^/\s\"'<>]+)*(?=[/\"'<>]|$)|[^/\s\"'<>]+)", r"\1<you>", text
    )
    # The Weights & Biases account: its setting, its console greeting and its run links.
    text = re.sub(r'(?m)^(\s*entity\s*=\s*)(["\']).*?\2', r"\1\2<you>\2", text)
    text = re.sub(r"(?i)(logged in as:?\s+)\S+(\s+\([^)\s]+\))?", r"\1<you>", text)
    text = re.sub(r"(?i)(wandb\.ai/)[^/\s\"'<>]+", r"\1<you>", text)
    if user and len(user) >= NAME_MIN:
        text = re.sub(rf"(?i)(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", "<you>", text)
    return text


def _user() -> str:
    for key in ("USERNAME", "USER", "LOGNAME"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return ""
