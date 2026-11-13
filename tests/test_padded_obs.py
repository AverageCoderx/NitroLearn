"""The padded observation, `env.obs_team_size`: its checks, what reaches the core and the
exported bot, and resuming a padded run at another team size. The observation itself is tested in the core
(crates/nitro_core/src/obs.rs) and the bot (crates/nitro_bot/tests/parity.rs)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from nitro import _core
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig
from nitro.envs import core_settings
from nitro.export import write_policy
from nitro.model import ActorCritic
from nitro.runs import RunFolder, fixed_settings


def parse(env: str) -> NitroConfig:
    return parse_toml(f"[env]\n{env}\n", NitroConfig, file=Path("nitro.toml"), base_dir=Path.cwd())


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ("team_size = 3\nobs_team_size = 2", "must be 0 or at least team_size (3)"),
        ("obs_team_size = 5", "at most 4"),
    ],
)
def test_a_room_the_game_does_not_fit_is_refused(env: str, message: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse(env)
    issues = {issue.path: issue.message for issue in info.value.issues}
    assert message in issues["env.obs_team_size"], issues


def test_the_room_reaches_the_core_and_the_core_checks_it_too() -> None:
    settings = json.loads(core_settings(parse("team_size = 2\nobs_team_size = 3")))
    assert settings["obs_team_size"] == 3
    settings["obs_team_size"] = 1  # the core refuses it before it loads any meshes
    with pytest.raises(ValueError, match="obs_team_size must be 0 or from team_size to 4"):
        _core.EnvPool(json.dumps(settings))


def test_the_exported_policy_carries_the_room_for_the_bot(tmp_path: Path) -> None:
    config = dataclasses.replace(parse("obs_team_size = 3"), model=ModelConfig(policy=(8,), value=(8,)))
    path = tmp_path / "policy.nwt"
    write_policy(path, ActorCritic(20, 5, config.model), config, 20, 5, "run", 0)
    assert _core.Policy(path).header()["obs_team_size"] == 3


def test_a_padded_run_resumes_at_another_team_size_and_an_unpadded_one_does_not(tmp_path: Path) -> None:
    def folder_with(name: str, env: str) -> RunFolder:
        settings = tmp_path / f"{name}.toml"
        settings.write_text(f"[env]\n{env}\n", encoding="utf-8")
        return RunFolder.create(tmp_path / "runs", name, settings)

    padded = folder_with("padded", "obs_team_size = 3")
    state = {"settings": fixed_settings(padded.load_settings())}
    padded.settings.write_text("[env]\nobs_team_size = 3\nteam_size = 3\n", encoding="utf-8")
    assert padded.load_settings(state).env.team_size == 3
    padded.settings.write_text("[env]\nobs_team_size = 2\nteam_size = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"env\.obs_team_size: was 3 when the run was saved"):
        padded.load_settings(state)

    plain = folder_with("plain", "")
    state = {"settings": fixed_settings(plain.load_settings())}
    plain.settings.write_text("[env]\nteam_size = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"env\.team_size: was 1 when the run was saved"):
        plain.load_settings(state)
