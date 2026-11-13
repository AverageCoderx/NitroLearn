"""The double-click shortcuts (Shortcuts/), run through PowerShell with their questions
answered and nothing actually started: on Windows with its own PowerShell 5.1, elsewhere with `pwsh` when
there is one (NITRO_TEST_PWSH names it)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from nitro.config import load_config
from nitro.config.schema import PROJECT

ROOT = Path(__file__).parents[1]
SHELL = os.environ.get("NITRO_TEST_PWSH") or shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(SHELL is None, reason="no PowerShell on this machine")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project folder with the shortcuts and the presets."""
    shutil.copytree(ROOT / "Shortcuts", tmp_path / "Shortcuts")
    shutil.copytree(ROOT / "configs" / "presets", tmp_path / "configs" / "presets")
    (tmp_path / "collision_meshes").mkdir()
    return tmp_path


def shortcut(project: Path, action: str, *answers: str) -> tuple[int, str]:
    assert SHELL is not None
    done = subprocess.run(
        [
            SHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(project / "Shortcuts" / "shortcuts.ps1"),
            action,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "NITRO_SHORTCUTS_DRY_RUN": "1", "NITRO_SHORTCUTS_ANSWERS": ";".join(answers)},
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def run_folder(project: Path, name: str, age: float) -> None:
    folder = project / "runs" / name
    (folder / "checkpoints" / "000000050000").mkdir(parents=True)
    (folder / "run.json").write_text("{}", encoding="utf-8")
    stamp = time.time() - age
    os.utime(folder, (stamp, stamp))


def test_a_new_settings_file_is_a_preset_copy_that_still_finds_the_meshes(project: Path) -> None:
    """The copy in my-settings is the preset as it is, so it finds the meshes as the preset does: in a
    collision_meshes folder beside it when that holds them, else in NitroLearn's own folder (whether or not
    meshes are there yet)."""
    code, out = shortcut(project, "new-settings", "2", "my-team")
    assert code == 0, out
    copy = project / "my-settings" / "my-team.toml"
    assert "would edit:" in out and f"would run: nitro check-settings {copy}" in out
    preset = project / "configs" / "presets" / "2v2_starter.toml"
    assert copy.read_text(encoding="utf-8") == preset.read_text(encoding="utf-8")
    config = load_config(copy)
    assert config.env.team_size == 2 and config.paths.collision_meshes == PROJECT / "collision_meshes"
    (copy.parent / "collision_meshes" / "soccar").mkdir(parents=True)
    assert load_config(copy).paths.collision_meshes == (copy.parent / "collision_meshes").resolve()
    code, out = shortcut(project, "new-settings", "2", "my-team")
    assert code == 1 and "already exists" in out, "never overwrites your own file"


def test_train_offers_your_files_first_and_names_the_run(project: Path) -> None:
    shortcut(project, "new-settings", "1", "mine")
    # The list: your file, every preset, then "another settings file"; one past its end is refused.
    choices = 1 + len(list((project / "configs" / "presets").glob("*.toml"))) + 1
    code, out = shortcut(project, "train", str(choices + 1), "1", "has space", "")
    assert code == 0, out
    assert f"Please type a number from 1 to {choices}." in out and "no spaces" in out
    assert "1) my-settings\\mine.toml" in out
    line = next(line for line in out.splitlines() if line.startswith("would run: nitro train"))
    assert str(project / "my-settings" / "mine.toml") in line and " --name mine-20" in line
    assert "would open when written:" in out


def test_a_taken_run_name_offers_to_resume_it(project: Path) -> None:
    run_folder(project, "first", 0)
    code, out = shortcut(project, "train", "2", "first", "y")
    assert code == 0 and "would run: nitro resume first" in out, out


def test_runs_are_listed_newest_first(project: Path) -> None:
    run_folder(project, "older", 3600)
    run_folder(project, "newer", 0)
    code, out = shortcut(project, "watch", "2")
    assert code == 0 and "1) newer" in out and "would run: nitro watch older" in out, out
    code, out = shortcut(project, "export", "")
    assert code == 0 and "would run: nitro export newer" in out and "would open:" in out, out
    code, out = shortcut(project, "progress", "1")
    assert code == 1 and "no progress page yet" in out, out


def test_nothing_to_pick_is_said_plainly(project: Path) -> None:
    code, out = shortcut(project, "resume")
    assert code == 1 and "There are no runs yet" in out, out
    code, out = shortcut(project, "edit-settings")
    assert code == 1 and "New settings file" in out, out
    code, out = shortcut(project, "doctor")
    assert code == 0 and "would run: nitro doctor" in out, out


def test_every_double_click_file_runs_an_action_the_script_knows() -> None:
    script = (ROOT / "Shortcuts" / "shortcuts.ps1").read_text(encoding="utf-8")
    files = sorted((ROOT / "Shortcuts").glob("*.cmd"))
    assert len(files) == 9
    for file in files:
        action = file.read_text(encoding="ascii").split('shortcuts.ps1" ')[1].split()[0]
        assert f"'{action}' {{" in script, file.name
    assert script.isascii(), "Windows PowerShell 5.1 reads a file without a byte-order mark as ANSI"
