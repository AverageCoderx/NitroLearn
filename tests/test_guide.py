"""The handbook (docs/guide) and the Help page's Python side: every link and anchor in the README and the docs
leads somewhere, every command the docs name answers --help, the dashboard lists every page and opens no
file outside the docs, and the report for Discord holds no user name or folder."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from nitro.cli import build_parser
from nitro.dashboard import helping
from nitro.dashboard.api import Api
from nitro.dashboard.jobs import CONSOLE
from test_dashboard import ROOT, make_bot

GUIDE = ROOT / "docs" / "guide"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted(GUIDE.glob("*.md"))]
LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces as hyphens."""
    words = re.sub(r"[`*]", "", heading).strip().lower()
    return re.sub(r"[^a-z0-9 _-]", "", words).replace(" ", "-")


def anchors(file: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", file.read_text(encoding="utf-8"), flags=re.DOTALL)
    return {slug(m[1]) for m in re.finditer(r"^#{1,6} (.+)$", text, re.MULTILINE)}


def links(file: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", file.read_text(encoding="utf-8"), flags=re.DOTALL)
    return [m[1] for m in LINK.finditer(re.sub(r"`[^`\n]+`", "", text))]


@pytest.mark.parametrize("file", DOCS, ids=lambda f: str(f.relative_to(ROOT)))
def test_every_link_and_anchor_leads_somewhere(file: Path) -> None:
    for link in links(file):
        if link.startswith(("https://", "mailto:")):
            continue
        assert not link.startswith("http://"), f"{file.name}: {link} (only https)"
        path, _, anchor = link.partition("#")
        target = (file.parent / path).resolve() if path else file
        assert target.exists(), f"{file.name}: {link} leads nowhere"
        assert ROOT in target.parents or target == ROOT, f"{file.name}: {link} leaves NitroLearn's folder"
        if anchor:
            assert target.suffix == ".md" and anchor in anchors(target), f"{file.name}: no heading for {link}"


def test_the_index_lists_every_page_and_the_dashboard_lists_them_in_its_order(tmp_path: Path) -> None:
    pages = {f.name for f in GUIDE.glob("*.md")} - {"README.md"}
    listed = [link for link in links(GUIDE / "README.md") if not link.startswith("https://")]
    assert pages == set(listed), "every page is in the index, and only pages"
    api = Api(ROOT, tmp_path / "runs")
    shown = api.guide_pages()["pages"]
    assert [p["page"] for p in shown] == list(dict.fromkeys(listed))
    assert all(p["title"] and not p["title"].endswith(".md") for p in shown), "each by its own title"


def test_the_readme_is_short_and_keeps_what_the_installer_and_the_release_read() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) < 200, "the details live in the handbook"
    for heading in (
        "## Fair play",
        "## Psyonix and Epic Games",
        "## License",
        "## What's new in ",
        "## Install",
    ):
        assert re.search(rf"^{re.escape(heading)}", readme, re.MULTILINE), heading
    assert "docs/guide/README.md" in readme


def _named_commands() -> set[str]:
    commands = set()
    for file in DOCS:
        text = file.read_text(encoding="utf-8")
        for m in re.finditer(r"(?:\.\\nitro|\./nitro\.sh|(?<![\w/.-])nitro) ([a-z][a-z-]+)", text):
            commands.add(m[1])
    return commands


def test_every_command_the_docs_name_is_one_and_answers_help() -> None:
    known = set(build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]
    named = _named_commands()
    assert {"train", "resume", "doctor", "export", "skills"} <= named, "the reading found the commands"
    unknown = named - known
    assert not unknown, f"the docs name commands that do not exist: {sorted(unknown)}"
    for command in sorted(named & known):
        done = subprocess.run(
            [sys.executable, "-m", "nitro", command, "--help"], capture_output=True, text=True, cwd=ROOT
        )
        assert done.returncode == 0 and "usage: nitro" in done.stdout, (command, done.stderr[-400:])


def test_the_docs_name_no_user_folder() -> None:
    for file in DOCS:
        text = file.read_text(encoding="utf-8")
        assert not re.search(r"(?i)[a-z]:\\users\\(?!<)", text), f"{file.name}: a user's folder"
        assert not re.search(r"/home/(?!<)\w", text), f"{file.name}: a user's folder"


def test_a_page_opens_by_its_name_and_nothing_outside_the_docs_does(tmp_path: Path) -> None:
    api = Api(ROOT, tmp_path / "runs")
    page = api.guide_page("coach.md")
    assert page["title"] == "The training coach" and "## The hints" in page["text"]
    assert api.guide_page("../rewards.md")["title"] == "The rewards that come with NitroLearn"
    for bad in (
        "../../LICENSE",
        "../../README.md",
        "a/../coach.md",
        "/etc/passwd",
        "coach",
        "..\\x.md",
        "nope.md",
    ):
        assert "error" in api.guide_page(bad), bad


def test_the_report_names_no_user_or_folder(tmp_path: Path) -> None:
    """The report: versions, the last Check this PC, the bot's settings and its console's end, with the user's
    name and every folder of theirs taken out, however a path is written."""
    project = tmp_path / "home" / "samplayer" / "NitroLearn"
    runs = project / "runs"
    project.mkdir(parents=True)
    folder = make_bot(runs, "my-bot")
    settings = folder.settings.read_text(encoding="utf-8")
    folder.settings.write_text(
        settings + f'\n# from {project}\n[pretrain]\nreplays = "C:/Users/SamPlayer/Documents/My Games/x"\n',
        encoding="utf-8",
    )
    console = (
        "Training my-bot\n"
        f"saved {runs / 'my-bot' / 'checkpoints'}\n"
        r"Traceback: File C:\Users\samplayer\AppData\Local\Programs\NitroLearn\python\nitro\trainer.py" + "\n"
        "warning from /home/samplayer/.cache/x and /Users/samplayer/x\n"
        "user samplayer said hi; SAMPLAYER too; samplayerx is another word\n"
    )
    (folder.root / CONSOLE).write_text("old line\n" * 100 + console, encoding="utf-8")
    checks = {
        "checks": [
            {"name": "PyTorch", "state": "ok", "message": "fine", "fix": ""},
            {"name": "Disk", "state": "warning", "message": f"little room on {project}", "fix": ""},
        ],
        "verdict": "1 warning",
    }
    text = helping.report(project, runs, folder, checks)
    home = tmp_path / "home" / "samplayer"
    scrubbed = helping.scrub(text, project, runs, user="samplayer", home=home)
    assert "NitroLearn " in scrubbed and "Python " in scrubbed, "the versions"
    assert "WARNING: Disk: little room on <NitroLearn>" in scrubbed and "1 checks OK" in scrubbed
    assert "[pretrain]" in scrubbed and "saved <runs>/my-bot/checkpoints" in scrubbed
    assert "C:/Users/<you>/Documents/My Games/x" in scrubbed
    assert r"C:\Users\<you>\AppData" in scrubbed and "/home/<you>/.cache" in scrubbed
    assert "/Users/<you>/x" in scrubbed and "user <you> said hi; <you> too" in scrubbed
    assert "samplayerx is another word" in scrubbed, "a longer word is left alone"
    assert "samplayer" not in scrubbed.replace("samplayerx", "").lower(), "no trace of the name"
    assert str(tmp_path) not in scrubbed
    assert scrubbed.count("old line") == helping.CONSOLE_LINES - 5, "only the console's last lines"


def test_the_window_copies_a_report_without_a_bot_and_before_any_check(tmp_path: Path) -> None:
    api = Api(ROOT, tmp_path / "runs")
    out = api.discord_report("")
    assert out["checked"] is False and "not run yet" in out["text"]
    assert str(tmp_path) not in out["text"] and str(ROOT) not in out["text"]
    assert "error" in api.discord_report("no-such-bot")
    make_bot(tmp_path / "runs", "b1")
    with_bot = api.discord_report("b1")
    assert "**Bot: b1**" in with_bot["text"] and "(no console yet)" in with_bot["text"]


def test_the_report_hides_a_user_folder_with_a_space_and_the_weights_and_biases_account() -> None:
    """A Windows user folder may hold a space ("Sam Player"), and settings files write backslashes twice; the
    W&B account shows in the settings (entity) and in W&B's own console lines."""
    text = (
        'replays = "C:\\\\Users\\\\Sam Player\\\\Documents\\\\x"\n'
        "C:/Users/Sam Player/Videos and C:\\Users\\Sam Player\n"
        "wandb: Currently logged in as: sammy99 (sammy-team). Use `wandb login --relogin` to force relogin\n"
        "wandb: View run at https://wandb.ai/sammy-team/nitrolearn/runs/abc\n"
        '[wandb]\nentity = "sammy-team"\nproject = "nitrolearn"\n'
    )
    scrubbed = helping.scrub(
        text, Path("/opt/NitroLearn"), Path("/opt/NitroLearn/runs"), user="sp", home=Path("/x")
    )
    assert "Player" not in scrubbed and "sammy" not in scrubbed, scrubbed
    assert 'C:\\\\Users\\\\<you>\\\\Documents\\\\x"' in scrubbed and "C:/Users/<you>/Videos and" in scrubbed
    assert "https://wandb.ai/<you>/nitrolearn/runs/abc" in scrubbed and 'entity = "<you>"' in scrubbed
    assert 'project = "nitrolearn"' in scrubbed, "other settings stay"
