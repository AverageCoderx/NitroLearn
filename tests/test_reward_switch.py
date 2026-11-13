"""Rewards switched off with `enabled = false` (the dashboard's on/off switch): kept in the
settings, but neither paid, measured nor compared."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nitro.ab import differences
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings

BASE = Path(Path.cwd().anchor) / "base"


def parse(*rewards: str) -> NitroConfig:
    text = "".join(f"[[rewards]]\n{reward}\n" for reward in rewards)
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=BASE)


OFF = 'name = "touch_ball"\nweight = 5\nenabled = false'


def test_a_switched_off_reward_is_kept_but_never_reaches_the_core() -> None:
    config = parse('name = "air"\nweight = 1', OFF, 'name = "goal"\nweight = 10')
    assert [r.name for r in config.rewards] == ["air", "touch_ball", "goal"]
    assert [r.name for r in config.active_rewards] == ["air", "goal"]
    sent = json.loads(core_settings(config))["rewards"]
    assert [r["kind"]["name"] for r in sent] == ["air", "goal"]
    assert all("enabled" not in r["kind"] for r in sent)  # the core refuses fields it does not know


def test_switching_every_reward_off_is_refused_and_a_part_cannot_be_switched_off() -> None:
    with pytest.raises(ConfigError, match="every reward is switched off"):
        parse(OFF, 'name = "air"\nenabled = false')
    with pytest.raises(ConfigError, match="zero_sum or enabled of its own"):
        parse('name = "all_of"\nparts = [{ name = "air", enabled = false }, { name = "alignment" }]')


def test_an_ab_comparison_sees_a_switched_off_reward_as_absent() -> None:
    with_off = parse('name = "air"\nweight = 1', OFF)
    without = parse('name = "air"\nweight = 1')
    assert differences(with_off, without) == []
    assert differences(without, parse('name = "air"\nweight = 2')) != []
