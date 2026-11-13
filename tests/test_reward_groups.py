"""The `alignment` reward and the `all_of` group in the settings, and reward-design
comparisons on a practice task (`--with-rewards`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nitro.ab import differences, reward_differences
from nitro.cli import main
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.rewards import Alignment, AllOf, VelocityPlayerToBall
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings
from nitro.tasks import TASKS, practice_config
from nitro.training import numbered

BASE = Path(Path.cwd().anchor) / "base"
GROUP = """
[[rewards]]
name = "all_of"
weight = 2
parts = [{ name = "velocity_player_to_ball" }, { name = "alignment" }]
[[rewards]]
name = "goal"
weight = 100
"""


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=BASE)


def issues(text: str) -> list[tuple[str, str]]:
    with pytest.raises(ConfigError) as info:
        parse(text)
    return [(issue.path, issue.message) for issue in info.value.issues]


def test_a_group_is_parsed_with_its_parts_and_reaches_the_core_tagged() -> None:
    config = parse(GROUP)
    assert config.rewards[0] == AllOf(weight=2.0, parts=(VelocityPlayerToBall(), Alignment()))
    settings = json.loads(core_settings(config))
    assert settings["rewards"][0] == {
        "kind": {"name": "all_of", "parts": [{"name": "velocity_player_to_ball"}, {"name": "alignment"}]},
        "weight": 2.0,
        "zero_sum": None,
    }
    assert settings["rewards"][1]["kind"] == {"name": "goal", "concede_scale": -1.0}


def test_a_group_needs_two_parts_that_carry_no_weight_of_their_own() -> None:
    found = issues('[[rewards]]\nname = "all_of"\nparts = [{ name = "alignment" }]\n')
    assert any(path == "rewards[0].parts" and "two parts" in message for path, message in found), found
    found = issues(
        '[[rewards]]\nname = "all_of"\nparts = [{ name = "alignment", weight = 3 }, { name = "air" }]\n'
    )
    assert any(path == "rewards[0].parts[0]" and "no weight" in message for path, message in found), found
    found = issues(
        '[[rewards]]\nname = "all_of"\n'
        'parts = [{ name = "alignment" }, '
        '{ name = "strong_touch", min_speed_kph = 50, max_speed_kph = 40 }]\n'
    )
    assert any("rewards[0].parts[1]" in path for path, _ in found), found
    found = issues(
        '[[rewards]]\nname = "all_of"\nparts = [{ name = "alignment" }, '
        '{ name = "all_of", parts = [{ name = "air" }, { name = "face_ball" }] }]\n'
    )
    assert found == [
        ("rewards[0].parts[1]", "a group cannot hold another group; list all the rewards in one")
    ]


def test_a_task_keeps_its_own_rewards_unless_the_comparison_is_about_rewards() -> None:
    bot = parse(GROUP)
    task = TASKS["kickoff"]
    assert practice_config(task, bot, seed=1).rewards == task.settings().rewards
    assert practice_config(task, bot, seed=1, with_rewards=True).rewards == bot.rewards


def test_repeated_reward_names_get_numbered_columns() -> None:
    assert numbered(["all_of", "touch_ball", "all_of", "all_of"]) == [
        "all_of",
        "touch_ball",
        "all_of_2",
        "all_of_3",
    ]
    assert numbered(["goal"]) == ["goal"]


def test_a_reward_comparison_lists_the_rewards_that_differ_in_plain_words() -> None:
    plain = parse('[[rewards]]\nname = "velocity_player_to_ball"\n[[rewards]]\nname = "alignment"\n')
    group = parse(GROUP)
    assert reward_differences(plain.rewards, group.rewards) == [
        "rewards[0]: velocity_player_to_ball weight 1 -> all_of(velocity_player_to_ball, alignment) weight 2",
        "rewards[1]: alignment weight 1 -> goal(concede_scale=-1) weight 100",
    ]
    shared = parse('[[rewards]]\nname = "touch_ball"\nzero_sum = { team_spirit = 0.5 }\n')
    assert reward_differences(shared.rewards, ()) == [
        "rewards[0]: touch_ball weight 1 zero-sum (team spirit 0.5, opponent scale 1) -> (none)"
    ]
    assert any(line.startswith("rewards[0]:") for line in differences(plain, group))
    assert not any(line.startswith("rewards:") for line in differences(plain, group)), "no raw dump"


def test_with_rewards_needs_a_task(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.toml", tmp_path / "b.toml"
    a.write_text("", encoding="utf-8")
    b.write_text(GROUP, encoding="utf-8")
    assert main(["ab", str(a), str(b), "--speed", "--with-rewards"]) == 1
    assert "add --task" in capsys.readouterr().err
