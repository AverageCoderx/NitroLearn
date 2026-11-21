"""Checks the release's files for text that must not ship, before a release and in its CI.

    python tools/release_scrub.py [--base REF]

Every tracked text file is checked for internal decision numbers (a capital D and two or three digits, as a
word), "private:" markers and names of people and private bots. Sources that earlier releases credit but new
work must not draw on are checked in the lines added since REF (the previous release, or the commit a push
started from), or in every file without --base. The words are kept here as SHA-256 hashes, so this file does
not name them; a match prints the file, line and word, so it can be dropped. Exit 1 on any match.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

# The first 16 hex digits of each word's SHA-256 (the word lower-case). NEVER: names of people and of private
# bots, checked in every file. NEW: work credited in earlier releases that new work must not add to, checked
# in the lines added since --base.
NEVER = frozenset(
    {
        "04640e7dee5200e9",
        "8bd45e781a4ea998",
        "a6761ccff1191f3e",
        "53b2946160fcd21c",
        "0930417631d146a6",
        "e15d07b70392f79c",
    }
)
NEW = frozenset({"e0901b044970ff1b"})
PATTERNS = (
    ("a decision number", re.compile(r"\bD\d{2,3}\b")),
    ("a private marker", re.compile(r"\bprivate:", re.IGNORECASE)),
)
SKIPPED = ("tools/release_scrub.py", "tests/test_release_scrub.py")
_WORD = re.compile(r"[A-Za-z0-9]+")


def hashed(word: str) -> str:
    return hashlib.sha256(word.lower().encode()).hexdigest()[:16]


def private_words(line: str, words: frozenset[str]) -> list[str]:
    return [word for word in _WORD.findall(line) if hashed(word) in words]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def _text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None  # binary
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def tracked(root: Path) -> list[str]:
    return [name for name in _git(root, "ls-files", "-z").split("\0") if name and name not in SKIPPED]


def added_lines(root: Path, base: str) -> Iterator[tuple[str, int, str]]:
    """(file, line number, text) of every line added since `base`."""
    diff = _git(root, "diff", "--unified=0", "--no-color", "--no-ext-diff", base, "--", ".")
    file, number = "", 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            file = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            number = int(match.group(1)) if match else 0
        elif line.startswith("+") and file and file not in SKIPPED:
            yield file, number, line[1:]
            number += 1


def check(root: Path, base: str | None) -> list[str]:
    found: list[str] = []
    every: list[tuple[str, int, str]] = []
    for name in tracked(root):
        text = _text(root / name)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            every.append((name, number, line))
            for what, pattern in PATTERNS:
                if match := pattern.search(line):
                    found.append(f"{name}:{number}: {what} ({match.group(0)})")
            for word in private_words(line, NEVER):
                found.append(f"{name}:{number}: a name that must not ship ({word})")
    lines: Iterable[tuple[str, int, str]] = every if base is None else added_lines(root, base)
    for name, number, line in lines:
        for word in private_words(line, NEW):
            found.append(f"{name}:{number}: new work from a source that must not be added to ({word})")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="check names only in lines added since this commit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        found = check(args.root, args.base)
    except subprocess.CalledProcessError as err:
        print(f"Cannot read the files or the changes with git: {err.stderr.strip() or err}", file=sys.stderr)
        return 2
    for line in found:
        print(line)
    print(f"{len(found)} problem(s) found." if found else "Nothing found that must not ship.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
