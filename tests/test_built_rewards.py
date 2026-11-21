"""Rewards built from blocks without code: how a settings file writes them, what the settings
check refuses, what reaches the core, and that the Python list of blocks is the core's."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nitro import _core
from nitro.config import built as blocks
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.rewards import Built
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings

ROOT = Path(__file__).parents[1]

HIGH_TOUCH = """
[[rewards]]
name = "built"
title = "my_high_touch"
weight = 20
zero_sum = { team_spirit = 0.5 }

[[rewards.terms]]
when = ["in_air", "touching_ball", { ball_above = 300 }, { not = "kickoff" }]
measure = "ball_height"

[[rewards.terms]]
when = ["on_ground"]
measure = ["speed_toward_ball", "boost"]
weight = 0.05
"""


def parse(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("own.toml"), base_dir=ROOT)


def issues(text: str) -> dict[str, str]:
    with pytest.raises(ConfigError) as info:
        parse(text)
    return {issue.path: issue.message for issue in info.value.issues}


def test_a_built_reward_loads_and_reaches_the_core_as_written() -> None:
    config = parse(HIGH_TOUCH)
    (reward,) = config.rewards
    assert isinstance(reward, Built) and reward.title == "my_high_touch" and reward.weight == 20
    assert reward.zero_sum is not None and reward.zero_sum.team_spirit == 0.5
    assert len(reward.terms) == 2 and not reward.every_step
    sent = json.loads(core_settings(config))["rewards"][0]
    assert sent["kind"]["name"] == "built" and sent["kind"]["title"] == "my_high_touch"
    assert sent["kind"]["terms"][1]["measure"] == ["speed_toward_ball", "boost"]
    assert sent["zero_sum"]["team_spirit"] == 0.5
    assert _core.observation_layout(core_settings(config))[0] > 0, "the core made the reward from it"


def test_a_built_reward_is_paid_every_step_only_when_every_term_is() -> None:
    text = '[[rewards]]\nname = "built"\ntitle = "chase"\nterms = [{ measure = "alignment" }]\n'
    assert parse(text).rewards[0].every_step
    assert not parse(text.replace('"alignment"', '"touch"')).rewards[0].every_step


def test_mistakes_are_named_at_their_place() -> None:
    found = issues(
        '[[rewards]]\nname = "built"\ntitle = "Air Touch"\n'
        'terms = [{ measure = "speed" }, { when = ["flying", { facing_ball_within = 200 }], measure = "one", '
        "wieght = 2 }]\n"
    )
    assert found == {
        "rewards[0].title": "a built reward needs a title of small letters, digits and _",
        "rewards[0].terms[0].measure": "there is no measure called 'speed'",
        "rewards[0].terms[1].wieght": "'wieght' is not a setting of a term",
        "rewards[0].terms[1].when[0]": "there is no condition called 'flying'",
        "rewards[0].terms[1].when[1]": "facing_ball_within needs 0 to 180 degrees",
    }
    assert issues('[[rewards]]\nname = "built"\ntitle = "goal"\nterms = [{ measure = "goal" }]\n') == {
        "rewards[0].title": "'goal' is already a reward's name"
    }
    assert "rewards[0].terms" in issues('[[rewards]]\nname = "built"\ntitle = "x"\n')


def test_the_python_list_of_blocks_is_the_cores() -> None:
    flags, numbered, measures = _core.built_blocks()
    assert flags == list(blocks.FLAGS)
    assert numbered == list(blocks.NUMBERED)
    assert measures == [(name, every_step) for name, (_, every_step) in blocks.MEASURES.items()]


def test_a_built_reward_can_be_part_of_a_group_unless_it_remembers() -> None:
    part = '{ name = "built", title = "airy", terms = [{ when = ["in_air"], measure = "one"%s }] }'
    group = '[[rewards]]\nname = "all_of"\nparts = [%s, { name = "alignment" }]\n'
    config = parse(group % (part % ""))
    _core.observation_layout(core_settings(config))
    assert issues(group % (part % ", once_per_touch = true")) == {
        "rewards[0].parts[0]": "a built reward that remembers (once_per_touch, variables, since...) cannot "
        "be part of a group"
    }


ADVANCED = """
[[rewards]]
name = "built"
title = "my_air_chain"
weight = 3

[[rewards.vars]]
name = "chain"
start = 0
next = "if(me.on_ground, 0, if(me.touch, chain + 1, chain))"

[[rewards.terms]]
when = ["in_air", "me.touch and held(me.in_air) >= 1"]
value = "chain * decay(dist(me.pos, opp_goal), 3000)"
cap_episode = 5
"""


def test_an_advanced_built_reward_loads_and_reaches_the_core_with_its_variables() -> None:
    config = parse(ADVANCED)
    (reward,) = config.rewards
    assert isinstance(reward, Built) and reward.vars[0]["name"] == "chain" and reward.remembers
    assert not reward.every_step
    sent = json.loads(core_settings(config))["rewards"][0]["kind"]
    assert sent["vars"][0]["next"].startswith("if(")
    assert _core.observation_layout(core_settings(config))[0] > 0


def test_formula_mistakes_are_named_at_their_place_before_training() -> None:
    def problem(old: str, new: str) -> dict[str, str]:
        return issues(ADVANCED.replace(old, new))

    assert problem("chain * decay", "chain ** decay") == {"rewards[0].terms[0].value": "unexpected '*' at 8"}
    assert problem("me.touch and held", "me.tuch and held") == {
        "rewards[0].terms[0].when[1]": "'me' has no field 'tuch' (at 4)"
    }
    assert problem("chain + 1", "chian + 1") == {
        "rewards[0].vars[0]": "there is nothing called 'chian' (at 34)"
    }
    assert problem("cap_episode = 5", "cap_episode = 5\npotential = true") == {
        "rewards[0].terms[0].potential": "potential terms are not in this version of NitroLearn"
    }
    assert problem("cap_episode = 5", "cap_episode = true") == {
        "rewards[0].terms[0].cap_episode": "cap_episode must be a number, not True"
    }


def test_numbers_the_core_cannot_hold_are_refused_in_plain_words() -> None:
    # 1e300 is finite as written but infinite as the core's 32-bit number (it paid inf into training); inf
    # and nan reached the core as JSON it could not read ("expected value at line 1 column 67").
    for old, new, place in (
        ("cap_episode = 5", "cap_episode = 5\nweight = 1e300", "terms[0].weight"),
        ("cap_episode = 5", "cap_episode = 5\nweight = nan", "terms[0].weight"),
        ("cap_episode = 5", "cap_episode = 1e300", "terms[0].cap_episode"),
        ("cap_episode = 5", "cap_episode = 5\ncap_step = inf", "terms[0].cap_step"),
        ("start = 0", "start = -1e39", "vars[0].start"),
        ("start = 0", "start = nan", "vars[0].start"),
    ):
        found = issues(ADVANCED.replace(old, new))
        assert list(found) == [f"rewards[0].{place}"], found
        assert "must be a number" in found[f"rewards[0].{place}"]


def test_a_table_list_or_date_where_a_name_goes_is_named_not_a_crash() -> None:
    term = '[[rewards]]\nname = "built"\ntitle = "x"\nterms = [{ %s }]\n'
    assert list(issues(term % "measure = [{ a = 1 }]")) == ["rewards[0].terms[0].measure"]
    assert list(issues(term % 'measure = [["one"]]')) == ["rewards[0].terms[0].measure"]
    deep = "{ not = " * 40 + '"in_air"' + " }" * 40
    assert list(issues(term % f'when = [{deep}], measure = "one"')) == ["rewards[0].terms[0].when[0]"]
    named = '[[rewards]]\nname = "built"\ntitle = "x"\nvars = [{ name = %s, next = "1" }]\n'
    named += 'terms = [{ value = "1" }]\n'
    for name in ("1979-05-27", "{ a = 1 }", "5", '"Air"'):
        assert list(issues(named % name)) == ["rewards[0].vars[0].name"], name


def test_every_template_loads_as_a_settings_file_block() -> None:
    templates = json.loads((ROOT / "python" / "nitro" / "config" / "built_templates.json").read_text())
    assert len(templates) >= 10
    for template in templates:
        block = {"name": "built", "title": template["title"], "terms": template["terms"]}
        if "vars" in template:
            block["vars"] = template["vars"]
        config = parse_toml_from_dict(block)
        assert isinstance(config.rewards[0], Built), template["title"]


def parse_toml_from_dict(block: dict[str, object]) -> NitroConfig:
    import tomlkit

    doc = tomlkit.document()
    table = tomlkit.aot()
    table.append(tomlkit.item(block))
    doc["rewards"] = table
    return parse(tomlkit.dumps(doc))


def test_the_python_list_of_formula_words_is_the_cores() -> None:
    car, ball, names, functions = _core.built_words()
    assert car == list(blocks.CAR_FIELDS) and ball == list(blocks.BALL_FIELDS)
    assert set(names) | {"pi"} == set(blocks.NAMES) and set(functions) == set(blocks.FUNCTIONS)
