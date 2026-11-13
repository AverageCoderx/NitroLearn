"""Rewards of one's own, written in Rust and compiled in: the example that is always
built in, how settings files name them and their own settings, what reaches the core, `nitro rebuild`'s list
and command, and the dashboard's view of them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nitro import _core, rebuild
from nitro.config import rewards as reward_config
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.rewards import Custom, CustomKind
from nitro.config.schema import NitroConfig
from nitro.dashboard import rewards as dashboard_rewards
from nitro.dashboard.api import Api
from nitro.envs import core_settings

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "crates" / "nitro_core" / "src" / "custom" / "example.rs"


def parse(*blocks: str) -> NitroConfig:
    text = "".join(f"[[rewards]]\n{block}\n" for block in blocks)
    return parse_toml(text, NitroConfig, file=Path("own.toml"), base_dir=ROOT)


def issues(*blocks: str) -> dict[str, str]:
    with pytest.raises(ConfigError) as info:
        parse(*blocks)
    return {
        issue.path: issue.message + (f" ({issue.hint})" if issue.hint else "") for issue in info.value.issues
    }


def test_the_example_is_built_in_and_loads_like_a_built_in_reward() -> None:
    assert "example_ball_height" in reward_config.custom_kinds()
    kind = reward_config.custom_kinds()["example_ball_height"]
    assert kind.every_step and Path(kind.file).resolve() == EXAMPLE.resolve()
    config = parse('name = "example_ball_height"\nweight = 3\npower = 2')
    reward = config.rewards[0]
    assert isinstance(reward, Custom) and reward.settings == (("power", 2),)
    sent = json.loads(core_settings(config))["rewards"][0]
    assert sent["kind"] == {"name": "custom", "kind": "example_ball_height", "settings": {"power": 2}}
    assert _core.observation_layout(core_settings(config))[0] > 0, "the core made the reward from it"
    grouped = parse('name = "all_of"\nparts = [{ name = "example_ball_height" }, { name = "alignment" }]')
    assert (
        json.loads(core_settings(grouped))["rewards"][0]["kind"]["parts"][0]["kind"] == "example_ball_height"
    )


def test_its_own_settings_are_checked_by_name_and_type() -> None:
    assert issues('name = "example_ball_height"\npowr = 2') == {
        "rewards[0].powr": "'powr' is not a setting of example_ball_height (its own settings: power)"
    }
    assert issues('name = "example_ball_height"\npower = "high"') == {
        "rewards[0].power": "must be a number, but is 'high'"
    }
    assert "did you mean 'example_ball_height'" in issues('name = "example_ball_heigth"')["rewards[0].name"]


def test_required_settings_and_whole_numbers_follow_the_compiled_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = CustomKind(
        "fake",
        "",
        False,
        "/x.rs",
        (("scale", "f32", False, ""), ("count", "u32", True, ""), ("on", "bool", True, "")),
    )
    monkeypatch.setattr(reward_config, "custom_kinds", lambda: {"fake": fake})
    assert issues('name = "fake"') == {"rewards[0].scale": "this setting is required but missing"}
    assert issues('name = "fake"\nscale = 1\ncount = -1\non = "yes"') == {
        "rewards[0].count": "must be 0 or more",
        "rewards[0].on": "must be true or false, but is 'yes'",
    }
    config = parse('name = "fake"\nscale = 1.5\ncount = 2')
    with pytest.raises(ValueError, match="no reward called 'fake' in this build"):
        _core.observation_layout(core_settings(config))


def test_rebuild_keeps_the_list_and_builds_the_core_in_place_or_with_pip_or_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mine = tmp_path / "mine.rs"
    mine.write_text("// mine\n", encoding="utf-8")
    listed = rebuild.write_list(tmp_path, [mine])
    assert listed == tmp_path / "my-settings" / "rust-files.txt" and rebuild.listed(tmp_path) == [
        mine.resolve()
    ]
    mine.unlink()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no build with a missing file"))
    assert rebuild.rebuild(tmp_path) == 1

    def pip_missing(command: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", pip_missing)
    monkeypatch.setattr(rebuild.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    # In a venv with maturin: only the core, in place; nothing reinstalled (Windows refuses to replace the
    # nitro.exe that runs this).
    monkeypatch.setattr(rebuild.sys, "prefix", "/venv")
    monkeypatch.setattr(rebuild.sys, "base_prefix", "/usr")
    assert rebuild.build_command("py") == [
        "py",
        "-m",
        "maturin",
        "develop",
        "--release",
        "--skip-install",
        "--uv",
    ]
    monkeypatch.setattr(rebuild.sys, "prefix", "/usr")
    assert rebuild.build_command("py")[:3] == ["uv", "pip", "install"]
    monkeypatch.setattr(subprocess, "run", lambda c, **_: subprocess.CompletedProcess(c, 0))
    assert rebuild.build_command("py")[:5] == ["py", "-m", "pip", "install", "--editable"]
    # maturin and pip are found next to this Python even when its venv is not activated.
    monkeypatch.setattr(rebuild.sys, "prefix", "/venv")
    monkeypatch.setenv("PATH", "/elsewhere")
    env = rebuild.build_environment()
    assert (
        env["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent) and env["VIRTUAL_ENV"] == "/venv"
    )


def test_rebuild_touches_the_stamp_so_the_build_reads_a_new_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core's build watches the list only once it exists (a missing file would make cargo build the core
    again every time), so `nitro rebuild` marks the stamp the build also watches before it builds."""
    stamp = tmp_path / rebuild.STAMP
    stamp.parent.mkdir(parents=True)
    stamp.write_text("stamp\n", encoding="utf-8")
    os.utime(stamp, (1_000_000, 1_000_000))
    seen: list[float] = []

    def build(command: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(stamp.stat().st_mtime)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", build)
    assert rebuild.rebuild(tmp_path) == 1
    assert seen and seen[0] > 1_000_000, "touched before the build"
    assert (Path(__file__).parents[1] / rebuild.STAMP).is_file(), "the stamp the build watches is in the tree"


def test_a_failed_rebuild_puts_the_set_aside_core_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "python" / "nitro"
    folder.mkdir(parents=True)
    (folder / "_core.pyd").write_bytes(b"old core")
    monkeypatch.setattr(rebuild.sys, "platform", "win32")
    monkeypatch.setattr(rebuild, "build_command", lambda: ["build"])
    monkeypatch.setattr(subprocess, "run", lambda c, **_: subprocess.CompletedProcess(c, 1))
    assert rebuild.rebuild(tmp_path) == 1
    assert (folder / "_core.pyd").read_bytes() == b"old core" and not list(folder.glob("*.old-*"))


def test_the_dashboard_lists_and_builds_rewards_of_ones_own(tmp_path: Path) -> None:
    catalog = dashboard_rewards.catalog()
    own = [kind for kind in catalog if kind.get("own")]
    assert len(catalog) - len(own) == 13 and "example_ball_height" in {kind["name"] for kind in own}
    example = next(kind for kind in own if kind["name"] == "example_ball_height")
    assert (
        example["every_step"] and example["params"][0]["path"] == "power" and example["params"][0]["nullable"]
    )
    view = dashboard_rewards.view({"name": "example_ball_height", "weight": 2})
    assert view["known"] and view["params"] == {"power": None}
    assert dashboard_rewards.data_of(view) == {"name": "example_ball_height", "weight": 2}, (
        "left out: default"
    )
    project = tmp_path / "project"
    project.mkdir()
    api = Api(project, tmp_path / "runs")
    api.add_resource_paths([str(EXAMPLE)])
    entry = next(e for e in api.resources()["files"] if e["kind"] == "rust_reward")
    assert entry["built"] and "example_ball_height" in api.resources()["own_rewards"]
    started: list[list[str]] = []
    api._jobs.start = lambda bot, kind, args, folder: started.append(args)  # type: ignore[method-assign]
    out = api.build_rust()
    assert out == {"started": True, "files": ["example.rs"]} and started == [["rebuild"]]
    assert rebuild.listed(project) == [EXAMPLE.resolve()]
    assert api.build_state()["state"] == "none"


def test_every_library_file_opens_with_what_it_pays_and_its_credits() -> None:
    """Each reward file that comes with NitroLearn opens with a `//!` header: what it pays, then a credit line
    when it is based on someone else's reward. Files adapted from Necto say so with its licence; files
    adapted from rlgym-sim carry the Apache notice and say what was changed."""
    library = ROOT / "crates" / "nitro_core" / "src" / "custom" / "library"
    headers = {}
    for path in sorted(library.glob("*.rs")):
        lines = path.read_text(encoding="utf-8").splitlines()
        end = next(i for i, line in enumerate(lines) if not line.startswith("//!"))
        header = [line[3:].strip() for line in lines[:end]]
        assert header and header[0], f"{path.name} has no header"
        headers[path.name] = " ".join(header)
    necto = sorted(name for name, text in headers.items() if "Necto" in text)
    assert necto == ["behind_ball_alignment.rs", "boost_lose.rs", "goal_speed_bonus.rs"]
    for name in necto:
        text = headers[name]
        assert "Adapted from Necto" in text and "CC BY-NC-SA 4.0" in text, name
        assert "https://creativecommons.org/licenses/by-nc-sa/4.0/" in text, name
    apache = sorted(name for name, text in headers.items() if "rlgym-sim" in text)
    assert apache == ["align_ball_goal.rs", "ball_y_coordinate.rs"]
    for name in apache:
        assert "Apache License, Version 2.0" in headers[name] and "Modified:" in headers[name], name
    credited = {
        "apollo-bot": 9,
        "rishieissocool": 3,
        "Opti": 2,
        "RLGym-PPO guide": 4,
    }
    for credit, files in credited.items():
        assert sum(credit in text for text in headers.values()) == files, credit


def test_every_library_reward_loads_alone_and_every_reward_set_loads() -> None:
    """The rewards that come with NitroLearn (custom/library) can each be named with no settings
    and are made by the core, the dashboard knows their settings' defaults, and every reward set that comes
    with it (configs/reward-sets) is a list of rewards the core makes."""
    from nitro.config import load_config

    catalog = {kind["name"]: kind for kind in dashboard_rewards.catalog()}
    library = [name for name, kind in catalog.items() if kind.get("source") == "library"]
    assert len(library) >= 84 and catalog["example_ball_height"]["source"] == "example"
    for name in library:
        config = parse(f'name = "{name}"')
        assert _core.observation_layout(core_settings(config))[0] > 0, name
        params = catalog[name]["params"]
        assert all(p["nullable"] for p in params), f"{name}: every setting has a default"
    settings = [p for name in library for p in catalog[name]["params"]]
    assert sum(p["default"] is None for p in settings) <= 1, "the dashboard shows nearly every default"
    moves = sorted(name for name in library if catalog[name]["trick_move"])
    assert moves == [
        *("air_dribble", "ball_to_roof_shaping", "breezi_flick", "flip_reset", "goal_in_mouth"),
        *("mawkzy_flick", "musty_flick", "musty_pose_shaping", "raze_flick", "raze_flick_setup"),
        *("reset_musty", "reset_musty_pose_shaping", "speed_flip", "tornado_shaping", "wall_dash"),
        *("wavedash", "wizard_flick"),
    ], "the trick-move rewards are their own group (a `group: moves` line in their file's header)"
    sets = sorted((ROOT / "configs" / "reward-sets").glob("*.toml"))
    assert len(sets) >= 8
    for file in sets:
        config = load_config(file)  # a reward set is a settings file of rewards alone
        assert len(config.rewards) > 1 and _core.observation_layout(core_settings(config))[0] > 0, file.name


def test_wall_dash_loads_with_its_settings_and_a_file_naming_walldash_is_pointed_to_it() -> None:
    """The wall dash reward reaches the core with its own settings; the earlier, simpler `walldash` is not in
    NitroLearn, and a settings file that names it is refused with a pointer to `wall_dash`."""
    config = parse('name = "wall_dash"\nweight = 5\nchain_bonus = 0.5\ncurves = true')
    sent = json.loads(core_settings(config))["rewards"][0]["kind"]
    assert sent == {"name": "custom", "kind": "wall_dash", "settings": {"chain_bonus": 0.5, "curves": True}}
    assert _core.observation_layout(core_settings(config))[0] > 0
    assert issues('name = "walldash"') == {
        "rewards[0].name": "'walldash' is not a known name here (did you mean 'wall_dash'?)"
    }


def test_a_failed_rebuild_says_what_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rust (into NitroLearn's folder) and the C/C++ build tools (never installed by the installer): the
    build's console, which the dashboard shows, names what is missing and where to get it."""
    monkeypatch.setattr(rebuild, "build_command", lambda: ["build"])
    monkeypatch.setattr(subprocess, "run", lambda c, **_: subprocess.CompletedProcess(c, 1))
    monkeypatch.setattr(rebuild, "missing_tools", lambda: ["Rust (the installer's -Rust)"])
    assert rebuild.rebuild(tmp_path) == 1
    assert "- Rust (the installer's -Rust)" in capsys.readouterr().out
    monkeypatch.undo()
    rust, compiler = rebuild.missing_tools(has_cargo=False, has_compiler=False)
    assert "Tools to build your own Rust rewards" in rust and "install.sh --rust" in rust
    assert (
        "Build Tools for Visual Studio 2022" if sys.platform == "win32" else "build-essential"
    ) in compiler
    assert rebuild.missing_tools(has_cargo=True, has_compiler=True) == []
