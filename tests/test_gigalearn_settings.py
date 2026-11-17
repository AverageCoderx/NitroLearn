"""GigaLearn's observations and masks as settings: what each is refused with, what reaches the core and
the exported bot. The observations and masks themselves are tested in the core (crates/nitro_core/src/obs.rs,
actions.rs) and the bot (crates/nitro_bot/tests/parity.rs)."""

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
from nitro.model import ActorCritic, observation_size
from nitro.pretrain import previous_action

GIGALEARN = ("gigalearn_advanced", "gigalearn_default", "gigalearn_padder")


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path.cwd())


def issues_of(text: str) -> dict[str, str]:
    with pytest.raises(ConfigError) as info:
        parse(text)
    return {issue.path: issue.message for issue in info.value.issues}


@pytest.mark.parametrize(
    ("observation", "size_1v1", "size_2v2"),
    [("gigalearn_advanced", 109, 167), ("gigalearn_default", 89, 127), ("gigalearn_padder", 109, 173)],
)
def test_each_observation_has_its_size_and_reaches_the_core(
    observation: str, size_1v1: int, size_2v2: int
) -> None:
    for team_size, size in ((1, size_1v1), (2, size_2v2)):
        config = parse(f'[env]\nteam_size = {team_size}\nobservation = "{observation}"\n')
        assert observation_size(config) == size
        sent = json.loads(core_settings(config))
        assert (sent["observation"], sent["action_masks"]) == (observation, "standard")
        assert _core.observation_layout(core_settings(config))[0] == size
        assert previous_action(config) == slice(9, 17), "the ball first, then the previous action"


def test_the_padders_room_is_obs_team_size_and_the_other_copies_take_none() -> None:
    config = parse('[env]\nobs_team_size = 3\nobservation = "gigalearn_padder"\n')
    assert observation_size(config) == 237
    for observation in ("gigalearn_advanced", "gigalearn_default"):
        issues = issues_of(f'[env]\nobs_team_size = 2\nobservation = "{observation}"\n')
        assert "'native', 'nitro_v2' or 'gigalearn_padder'" in issues["env.obs_team_size"], issues
    with pytest.raises(ValueError, match="gigalearn_padder"):
        _core.observation_layout_of("gigalearn_default", 1, True, 2)


@pytest.mark.parametrize("observation", GIGALEARN)
def test_what_their_observations_lack_is_refused_plainly(observation: str) -> None:
    env = f'[env]\nobservation = "{observation}"\nobs_team_size = 2\nteam_size = 2\n'
    if observation != "gigalearn_padder":
        env = f'[env]\nobservation = "{observation}"\n'
    issues = issues_of(env + "ball_prediction = [0.5]\n")
    assert "'native' or 'nitro_v2'" in issues["env.ball_prediction"], issues
    issues = issues_of(f'[env]\nobservation = "{observation}"\n[model]\nencoder = "attention"\n')
    assert "native" in issues["model.encoder"], issues
    with pytest.raises(ValueError, match="ball_prediction"):
        _core.observation_layout_of(observation, 1, True, 0, 1)
    padded = issues_of('[env]\nobservation = "gigalearn_padder"\nobs_team_size = 2\nshuffle_slots = true\n')
    assert "'native' or 'nitro_v2'" in padded["env.shuffle_slots"], padded


def test_the_gigalearn_masks_reach_the_core_and_the_exported_bot(tmp_path: Path) -> None:
    config = parse('[env]\nobservation = "gigalearn_advanced"\naction_masks = "gigalearn"\n')
    assert json.loads(core_settings(config))["action_masks"] == "gigalearn"
    assert parse("").env.action_masks == "standard"
    issues = issues_of('[env]\naction_masks = "other"\n')
    assert "env.action_masks" in issues, issues
    config = dataclasses.replace(config, model=ModelConfig(policy=(8,), value=(8,)))
    path = tmp_path / "policy.nwt"
    write_policy(path, ActorCritic(109, 90, config.model), config, 109, 90, "run", 0)
    header = _core.Policy(path).header()
    assert (header["observation"], header["action_masks"]) == ("gigalearn_advanced", "gigalearn")
