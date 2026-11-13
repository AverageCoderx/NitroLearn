from __future__ import annotations

import json
from pathlib import Path

import pytest

from nitro.config import default_config
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.rewards import DEFAULT_REWARDS, StrongTouch, VelocityBallToGoal, ZeroSum
from nitro.config.schema import NitroConfig, PathsConfig
from nitro.config.state_setters import Kickoff, RandomState
from nitro.envs import core_settings

# An absolute folder on this OS ("C:\base" on Windows, "/base" elsewhere), so the tests pass on both.
BASE = Path(Path.cwd().anchor) / "base"


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=BASE)


def issues_of(text: str) -> list[tuple[str, int | None, str, str | None]]:
    with pytest.raises(ConfigError) as info:
        parse(text)
    return [(issue.path, issue.line, issue.message, issue.hint) for issue in info.value.issues]


def test_defaults_match_the_common_example_setup() -> None:
    config = default_config(BASE)
    assert (config.env.team_size, config.env.tick_skip, config.env.action_delay) == (1, 4, 1), (
        "tick skip 4, action delay 1"
    )
    assert config.env.num_arenas == 1024
    assert (config.env.observation, config.model.activation, config.ppo.policy_lr) == (
        *("nitro_v2", "leaky_relu", 4.5e-4),
    ), "the measured best values"
    assert config.terminals.no_touch_seconds == 10.0
    assert config.rewards == DEFAULT_REWARDS
    assert config.state_setters == (Kickoff(),)


def test_rewards_and_setters_are_picked_by_name() -> None:
    config = parse(
        """
[[rewards]]
name = "strong_touch"
weight = 60
max_speed_kph = 120

[[rewards]]
name = "velocity_ball_to_goal"
weight = 2
zero_sum = { team_spirit = 1 }

[[state_setters]]
name = "random"
weight = 3
ground_chance = 0.25
"""
    )
    assert config.rewards == (
        StrongTouch(weight=60.0, max_speed_kph=120.0),
        VelocityBallToGoal(weight=2.0, zero_sum=ZeroSum(team_spirit=1.0, opponent_scale=1.0)),
    )
    assert config.state_setters == (RandomState(weight=3.0, ground_chance=0.25),)


def test_unknown_reward_name_suggests_the_closest() -> None:
    assert issues_of('[[rewards]]\nname = "strong_tuch"\n') == [
        ("rewards[0].name", 2, "'strong_tuch' is not a known name here", "did you mean 'strong_touch'?")
    ]


def test_reward_without_name_lists_the_choices() -> None:
    ((path, line, message, hint),) = issues_of("[[rewards]]\nweight = 3\n")
    assert (path, line, message) == ("rewards[0]", 1, "needs a name")
    assert hint is not None and "strong_touch" in hint and "goal" in hint


def test_parameters_of_another_reward_are_rejected() -> None:
    ((path, line, message, _),) = issues_of('[[rewards]]\nname = "air"\nexponent = 2\n')
    assert (path, line, message) == ("rewards[0].exponent", 3, "'exponent' is not a known setting")


def test_settings_that_depend_on_each_other_are_checked() -> None:
    assert issues_of("[env]\ntick_skip = 4\naction_delay = 4\n") == [
        ("env.action_delay", 3, "must be smaller than tick_skip (4)", None)
    ]
    assert issues_of('[[rewards]]\nname = "strong_touch"\nmin_speed_kph = 50\nmax_speed_kph = 40\n') == [
        ("rewards[0].max_speed_kph", 4, "must be larger than min_speed_kph", None)
    ]
    assert issues_of('[[state_setters]]\nname = "kickoff"\nweight = 0\n') == [
        ("state_setters", 1, "at least one state setter needs a weight above 0", None)
    ]


def test_core_settings_carry_every_value_in_the_cores_format() -> None:
    config = parse(
        """
[run]
seed = 5
[env]
num_arenas = 4
observation = "nitro_v2"
[[rewards]]
name = "goal"
weight = 150
concede_scale = -0.5
zero_sum = { team_spirit = 0.5, opponent_scale = 0.8 }
[[state_setters]]
name = "kickoff"
fuzz_uu = 50
"""
    )
    settings = json.loads(core_settings(config))
    assert settings["seed"] == 5 and settings["num_arenas"] == 4
    # Not written, and no meshes beside the file: NitroLearn's own folder.
    assert settings["collision_meshes"] == str(PathsConfig.default_path("collision_meshes", BASE / "x"))
    assert settings["observation"] == "nitro_v2" and settings["action_masks"] == "standard"
    assert settings["opponents"] is True
    assert settings["terminals"] == {
        "no_touch_seconds": 10.0,
        "max_episode_seconds": 0.0,
        "end_on_touch": False,
        "end_on_landing": False,
        "truncate_on_ball_landing": 0.0,
    }
    assert settings["rewards"] == [
        {
            "kind": {"name": "goal", "concede_scale": -0.5},
            "weight": 150.0,
            "zero_sum": {"team_spirit": 0.5, "opponent_scale": 0.8},
        }
    ]
    assert settings["state_setters"] == [{"kind": {"name": "kickoff", "fuzz_uu": 50.0}, "weight": 1.0}]
    for gone in ("infinite_boost_share", "setter_blocks", "physics", "action_delay_jitter", "lessons"):
        assert gone not in settings


@pytest.mark.parametrize(
    ("text", "path"),
    [
        ('[ppo]\nlr_schedule = "kl_adaptive"\n', "ppo.lr_schedule"),
        ("[env]\ngpu_physics = true\n", "env.gpu_physics"),
        ('[env]\naction_masks = "other"\n', "env.action_masks"),
        ("[terminals]\nair_sequence = true\n", "terminals.air_sequence"),
        ('[[rewards]]\nname = "touch_ball"\npaid_on_goal = 3\n', "rewards[0].paid_on_goal"),
        ('[[rewards]]\nname = "touch_ball"\nramp = { to = 2.0 }\n', "rewards[0].ramp"),
        ("[env.physics]\ngravity_z = -600\n", "env.physics"),
        ('[selector]\nspecialists = ["a"]\n', "selector"),
        ("[actions]\n", "actions"),
        ('[[lessons]]\npackage = "x"\n', "lessons"),
    ],
)
def test_settings_this_version_does_not_have_are_refused_plainly(text: str, path: str) -> None:
    (issue,) = issues_of(text)
    assert issue[0] == path and issue[2] == f"'{path.split('.')[-1]}' is not a known setting", issue


def test_settings_with_no_reward_switched_on_are_warned_about() -> None:
    """Allowed (a run can start), but the bot has nothing to learn from: check-settings and a starting run
    say so."""
    off = parse('[[rewards]]\nname = "goal"\nenabled = false\n[[rewards]]\nname = "air"\nweight = 0\n')
    assert any("the bot has nothing to learn from" in note for note in off.advice())
    assert not any("nothing to learn from" in note for note in parse("").advice()), "the default rewards"
    on = parse('[[rewards]]\nname = "goal"\nenabled = false\n[[rewards]]\nname = "air"\n')
    assert not any("nothing to learn from" in note for note in on.advice())
