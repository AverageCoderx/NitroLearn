"""Episode options: the ball-landing truncation chance."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from nitro.config import default_config
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import EnvConfig, NitroConfig
from nitro.envs import core_settings
from nitro.tasks import _LEARNING_ENV, _TASK_ENV, TASKS, practice_config

BASE = Path(Path.cwd().anchor) / "base"
OPTIONS = "[terminals]\ntruncate_on_ball_landing = 0.1\n"


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=BASE)


def test_the_options_reach_the_core_and_default_to_off() -> None:
    plain = json.loads(core_settings(default_config(BASE)))
    assert plain["terminals"]["truncate_on_ball_landing"] == 0.0
    settings = json.loads(core_settings(parse(OPTIONS)))
    assert settings["terminals"]["truncate_on_ball_landing"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "text",
    ["[terminals]\ntruncate_on_ball_landing = 1.5\n", "[terminals]\ntruncate_on_ball_landing = -0.1\n"],
)
def test_shares_outside_0_to_1_are_rejected(text: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse(text)
    paths = [issue.path for issue in info.value.issues]
    assert any("truncate_on_ball_landing" in path for path in paths), paths


def test_a_practice_task_keeps_its_own_endings() -> None:
    bot = parse(OPTIONS)
    aerial = practice_config(TASKS["aerial_touch"], bot, seed=1)
    assert aerial.terminals.truncate_on_ball_landing == 1.0, "the task keeps its own cut-off"
    assert practice_config(TASKS["ball_touch"], bot, seed=1).terminals.truncate_on_ball_landing == 0.0


def test_a_practice_task_takes_every_playing_setting_from_the_bot_but_its_own_players_and_arenas() -> None:
    """A new [env] setting reaches practice runs (nitro ab --task) unless the task sets it."""
    assert not set(_LEARNING_ENV) & set(_TASK_ENV)
    assert set(_LEARNING_ENV) | set(_TASK_ENV) == {field.name for field in dataclasses.fields(EnvConfig)}
    fast = parse("[env]\npipelined_inference = false\nconcurrent_passes = false\ncard_starts_games = false\n")
    env = practice_config(TASKS["touch_mix"], fast, seed=1).env
    assert (env.pipelined_inference, env.concurrent_passes, env.card_starts_games) == (False, False, False)
