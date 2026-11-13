"""The practice tasks' measurements and settings files, checked without a simulator."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from nitro.config.state_setters import Save, Shot
from nitro.tasks import TASKS


def record(**fields: object) -> SimpleNamespace:
    base = {
        "env": 0,
        "car": 0,
        "seconds": 5.0,
        "terminal": False,
        "first_touch": math.nan,
        "landed": math.nan,
        "boost_collected": 0.0,
        "ball_y_after_touch": math.nan,
        "goal": 0,
        "setter": 0,
    }
    return SimpleNamespace(**(base | fields))


def test_touch_mix_counts_a_miss_as_ten_seconds_and_splits_the_times_by_kind() -> None:
    records = [
        record(setter=0, first_touch=2.0, terminal=True, seconds=2.0),
        record(setter=0, seconds=10.0),  # ground, missed: 10 s
        record(setter=1, first_touch=3.0, terminal=True, seconds=3.0),
        record(setter=1, seconds=4.0),  # aerial, ball came down: a miss counts the task's 10 s
    ]
    measured = TASKS["touch_mix"].measure(records)
    assert measured == {
        "time_to_touch": pytest.approx((2.0 + 10.0 + 3.0 + 10.0) / 4),
        "touch_rate": 0.5,
        "ground_time": 6.0,
        "aerial_time": 6.5,
    }
    task = TASKS["touch_mix"]
    assert task.metric == "time_to_touch" and not task.higher_is_better
    settings = task.settings()
    assert [setter.name for setter in settings.state_setters] == ["ball_touch", "aerial_ball"]
    assert settings.terminals.end_on_touch and settings.terminals.truncate_on_ball_landing == 1.0


def test_shot_counts_goals_and_the_time_to_score_with_misses_as_the_limit() -> None:
    records = [
        record(goal=1, terminal=True, seconds=2.0, first_touch=1.0),
        record(goal=1, terminal=True, seconds=4.0, first_touch=1.5),
        record(goal=0, seconds=8.0, first_touch=3.0),
        record(goal=0, seconds=8.0),
    ]
    measured = TASKS["shot"].measure(records)
    assert measured == {"goal_rate": 0.5, "time_to_goal": pytest.approx((2.0 + 4.0 + 8.0 + 8.0) / 4)}
    assert TASKS["shot"].metric == "goal_rate" and TASKS["shot"].higher_is_better


def test_save_counts_episodes_without_a_goal_conceded_and_touches() -> None:
    records = [
        record(goal=-1, terminal=True, seconds=2.5),
        record(goal=0, seconds=6.0, first_touch=1.2),
        record(goal=0, seconds=6.0),
        record(goal=-1, terminal=True, seconds=3.0, first_touch=2.0),
    ]
    measured = TASKS["save"].measure(records)
    assert measured == {"save_rate": 0.5, "touch_rate": 0.5}
    assert TASKS["save"].better(0.7, 0.5) == pytest.approx(0.2)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_every_task_file_loads_and_matches_its_task(name: str) -> None:
    task = TASKS[name]
    config = task.settings()
    assert config.problems() == []
    assert config.env.opponents == task.against_untrained
    assert len(task.metrics) >= 1 and task.smallest_difference > 0 and task.steps > 0
    setters = {type(setter) for setter in config.state_setters}
    if name == "shot":
        assert setters == {Shot}
        assert config.terminals.max_episode_seconds == 8
    if name == "save":
        assert setters == {Save}
        assert config.terminals.max_episode_seconds == 6


def test_shot_and_save_setters_check_their_ranges() -> None:
    assert Shot(min_distance_uu=3000.0, max_distance_uu=2000.0).problems() == [
        ("max_distance_uu", "must be at least min_distance_uu (3000.0)")
    ]
    assert Shot().problems() == [] and Save().problems() == []
    assert Save(min_speed=3000.0, max_speed=2000.0).problems() == [
        ("max_speed", "must be at least min_speed (3000.0)")
    ]


def test_aerial_touch_counts_a_miss_as_six_seconds() -> None:
    records = [
        record(first_touch=1.5, terminal=True, seconds=1.5),
        record(seconds=2.0),
    ]  # the second: ball landed
    assert TASKS["aerial_touch"].measure(records) == {"time_to_touch": pytest.approx(3.75), "touch_rate": 0.5}
    assert TASKS["aerial_touch"].settings().terminals.truncate_on_ball_landing == 1.0
    assert TASKS["ball_touch"].measure([record(seconds=10.0)])["time_to_touch"] == 10.0
