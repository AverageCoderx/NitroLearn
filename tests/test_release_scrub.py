"""tools/release_scrub.py: what must not ship is found, in every file or in the lines added since a commit."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "tools" / "release_scrub.py"
_spec = importlib.util.spec_from_file_location("release_scrub", HERE)
assert _spec is not None and _spec.loader is not None
scrub = importlib.util.module_from_spec(_spec)
sys.modules["release_scrub"] = scrub
_spec.loader.exec_module(scrub)


def _repo(tmp: Path, files: dict[str, str]) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    for name, text in files.items():
        (tmp / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp / name).write_text(text, encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "files")
    return tmp


def test_words_are_matched_by_hash_only() -> None:
    assert scrub.hashed("Opti") in scrub.NEW
    assert scrub.private_words("ported from Opti's rewards", scrub.NEW) == ["Opti"]
    assert scrub.private_words("an optimizer, optional, adopting", scrub.NEW) == []


def test_every_file_is_checked_for_decision_numbers_and_markers(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        {
            "a.py": "# see D123 for why\nx = 1\n",
            "b.md": "Private: keep this out\nDevices D3 and D1234 are fine\n",
            "c.bin": "\0D123",
        },
    )
    found = scrub.check(root, None)
    assert found == ["a.py:1: a decision number (D123)", "b.md:1: a private marker (Private:)"]


def test_credited_sources_are_checked_only_in_new_lines(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"CREDITS.md": "Based on Opti's reward.\n"})
    assert scrub.check(root, "HEAD") == []
    assert scrub.check(root, None) == [
        "CREDITS.md:1: new work from a source that must not be added to (Opti)"
    ]
    (root / "new.rs").write_text("// fine\n// adapted from opti\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    assert scrub.check(root, "HEAD") == ["new.rs:2: new work from a source that must not be added to (opti)"]


def test_the_command_says_what_it_found(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.txt": "clean\n"})
    done = subprocess.run([sys.executable, str(HERE), "--root", str(root)], capture_output=True, text=True)
    assert done.returncode == 0 and "Nothing found" in done.stdout
    done = subprocess.run(
        [sys.executable, str(HERE), "--root", str(root), "--base", "no-such-commit"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2 and "Cannot read" in done.stderr


def test_the_scrub_tool_and_this_test_name_nothing_that_must_not_ship() -> None:
    """The check skips these two files, so they are held to it here: no name that must not ship, and no
    decision number in the tool (this test's own made-up ones are its examples)."""
    for path in (HERE, Path(__file__)):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert scrub.private_words(line, scrub.NEVER) == [], f"{path.name}:{number}"
            if path == HERE and "re.compile" not in line:
                assert not scrub.PATTERNS[0][1].search(line), f"{path.name}:{number}"
