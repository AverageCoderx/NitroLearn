from __future__ import annotations

import dataclasses
import logging
import random
import statistics
from collections import deque

import pytest

from nitro.health import HealthMonitor, TrainingProblem, TrendWarnings, _is_goal, learning_problems
from nitro.learner import LearnStats

HEALTHY = LearnStats(
    policy_loss=0.1,
    value_loss=0.5,
    entropy=0.9,
    approx_kl=0.01,
    clip_fraction=0.1,
    explained_variance=0.3,
    grad_norms={"policy": 0.4, "value": 1.0},
    reward_scale=2.0,
    gradient_steps=8,
    skipped_updates=0,
    grad_scale=4096.0,
    param_change=1e-3,
    policy_change=2e-3,
    inference_mismatch=1e-9,
    params_finite=True,
)


def with_(**changes: object) -> LearnStats:
    return dataclasses.replace(HEALTHY, **changes)  # type: ignore[arg-type]


def test_a_nan_loss_stops_training_with_a_clear_message() -> None:
    with pytest.raises(TrainingProblem, match="NaN or infinite") as info:
        HealthMonitor("stop", fp16=True).check(with_(value_loss=float("nan")))
    assert 'on_training_problem = "warn"' in str(info.value)


def test_nan_weights_and_skipped_fp32_updates_are_problems() -> None:
    monitor = HealthMonitor("warn", fp16=False)
    assert monitor.check(with_(params_finite=False)) == ["the network weights contain NaN or infinity"]
    (problem,) = monitor.check(with_(skipped_updates=1))
    assert "1 of 8 learning updates had NaN or infinite gradients" in problem


def test_an_isolated_fp16_skip_is_normal_but_repeated_skips_stop_training() -> None:
    # With one update per iteration, FP16 loss scaling's designed overflow skips a whole iteration.
    monitor = HealthMonitor("stop", fp16=True)
    one_skipped = with_(gradient_steps=1, skipped_updates=1)
    assert monitor.check(one_skipped) == []
    assert monitor.check(with_(gradient_steps=1, skipped_updates=0)) == []
    monitor.check(one_skipped)
    monitor.check(one_skipped)
    with pytest.raises(TrainingProblem, match="100% of learning updates were skipped 3 iterations in a row"):
        monitor.check(one_skipped)
    # With eight updates per iteration, two fully skipped iterations already make 16 in a row.
    monitor = HealthMonitor("stop", fp16=True)
    monitor.check(with_(skipped_updates=8))
    with pytest.raises(TrainingProblem, match="the last 16 learning updates were all skipped"):
        monitor.check(with_(skipped_updates=8))


def test_a_collapsing_loss_scale_is_a_problem() -> None:
    (problem,) = HealthMonitor("warn", fp16=True).check(with_(grad_scale=0.25))
    assert "loss scale has fallen to 0.25" in problem


def test_repeated_partial_skipping_is_a_problem(caplog: pytest.LogCaptureFixture) -> None:
    monitor = HealthMonitor("warn", fp16=True)
    monitor.check(HEALTHY)
    with caplog.at_level(logging.WARNING, logger="nitro"):
        assert monitor.check(with_(skipped_updates=3)) == []
        assert monitor.check(with_(skipped_updates=3)) == []
        (problem,) = monitor.check(with_(skipped_updates=3))
    assert "38% of learning updates were skipped 3 iterations in a row" in problem
    assert "Training problem" in caplog.text


def test_learning_problems_spot_runs_whose_updates_did_not_work() -> None:
    assert learning_problems([HEALTHY, with_(entropy=0.8)]) == []
    frozen = learning_problems(
        [with_(param_change=0.0, policy_change=0.0), with_(param_change=0.0, policy_change=0.0)]
    )
    assert "the weights did not change in 2 of 2 iterations" in frozen
    assert any(p.startswith("the policy did not change") and "2 of 2" in p for p in frozen)
    assert any("entropy stayed" in p for p in frozen)
    assert learning_problems([with_(skipped_updates=8)]) == [
        "every learning update was skipped in 1 of 1 iterations"
    ]
    # One iteration in ten without an applied update (an FP16 overflow) is tolerated.
    one_skip = [with_(entropy=0.9 - i / 100) for i in range(9)] + [
        with_(skipped_updates=8, param_change=0.0, policy_change=0.0)
    ]
    assert learning_problems(one_skip) == []
    assert learning_problems([with_(value_loss=float("nan"), entropy=0.8), HEALTHY]) == [
        "NaN or infinity in losses or weights (iterations 1 to 1)"
    ]


SHARES = {"velocity_player_to_ball": 0.7, "face_ball": 0.1, "strong_touch": 0.2}


def test_a_steady_run_gets_no_trend_warnings() -> None:
    warnings = TrendWarnings()
    assert not any(warnings.check(HEALTHY, 300_000.0, SHARES) for _ in range(500))


@pytest.mark.parametrize(
    ("changes", "speed", "shares", "says"),
    [
        ({"entropy": 0.05}, 300_000.0, SHARES, "raise ppo.entropy"),
        ({"policy_change": 0.05}, 300_000.0, SHARES, "lower ppo.policy_lr"),
        ({"value_loss": 50.0}, 300_000.0, SHARES, "lower ppo.value_lr"),
        ({}, 100_000.0, SHARES, "another program"),
        ({}, 300_000.0, {"bump": 0.95, "goal": 0.05}, "'bump' reward makes up 95%"),
    ],
)
def test_a_trend_warning_says_what_to_change_and_is_not_repeated_every_iteration(
    changes: dict[str, float], speed: float, shares: dict[str, float], says: str
) -> None:
    warnings = TrendWarnings()
    for _ in range(30):
        assert warnings.check(HEALTHY, 300_000.0, SHARES) == []
    first = warnings.check(with_(**changes), speed, shares)
    assert len(first) == 1 and says in first[0]
    assert warnings.check(with_(**changes), speed, shares) == []


def test_a_single_paying_reward_is_a_choice_not_farming() -> None:
    warnings = TrendWarnings()
    only_one_pays = {"velocity_player_to_ball": 1.0, "goal": 0.0, "bump": 0.0}
    assert not any(warnings.check(HEALTHY, 300_000.0, only_one_pays) for _ in range(150))


def test_value_predictions_worse_than_the_average_warn_only_when_they_persist() -> None:
    warnings = TrendWarnings()
    bad = with_(explained_variance=-0.2)
    assert not any(warnings.check(bad, 300_000.0, SHARES) for _ in range(19))
    assert "predicted worse than simply guessing the average" in warnings.check(bad, 300_000.0, SHARES)[0]


def test_the_slow_first_iteration_after_the_games_are_rebuilt_is_not_a_speed_warning() -> None:
    warnings = TrendWarnings()
    for _ in range(30):
        assert warnings.check(HEALTHY, 300_000.0, SHARES) == []
    assert warnings.check(HEALTHY, None, SHARES) == [], "warming up after a save says nothing about the PC"
    assert "another program" in warnings.check(HEALTHY, 100_000.0, SHARES)[0]


def test_longer_games_with_the_goals_share_falling_warn_of_farming_and_name_the_rising_pay() -> None:
    """The farming pattern of long runs, warned once games and pay shares have moved."""
    trends = TrendWarnings()
    healthy = {"goal": 0.25, "pickup_boost": 0.10, "touch_ball": 0.65}
    farming = {"goal": 0.10, "pickup_boost": 0.35, "touch_ball": 0.55}
    found: list[str] = []
    for i in range(TrendWarnings.FARM_WINDOW):
        late = i >= TrendWarnings.FARM_WINDOW - TrendWarnings.FARM_RECENT
        found += trends.check(HEALTHY, 1000.0, farming if late else healthy, None, 120.0 if late else 60.0)
    assert len(found) == 1 and "farming" in found[0]
    assert "60 s to 120 s" in found[0] and "pickup_boost 10% -> 35%" in found[0]


def test_longer_games_alone_or_a_falling_goal_share_alone_are_not_farming() -> None:
    for late_shares, late_seconds in (
        ({"goal": 0.25, "touch_ball": 0.75}, 120.0),
        ({"goal": 0.1, "touch_ball": 0.9}, 60.0),
    ):
        trends = TrendWarnings()
        found: list[str] = []
        for i in range(TrendWarnings.FARM_WINDOW):
            late = i >= TrendWarnings.FARM_WINDOW - TrendWarnings.FARM_RECENT
            shares = late_shares if late else {"goal": 0.25, "touch_ball": 0.75}
            found += trends.check(HEALTHY, 1000.0, shares, None, late_seconds if late else 60.0)
        assert not [w for w in found if "farming" in w]


class _FarmingAsBefore(TrendWarnings):
    """TrendWarnings with `_farming` as it was in an earlier version, summing every kept row's goal share
    again each
    iteration: the reference the kept goal share must match."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: deque[tuple[dict[str, float], float]] = deque(maxlen=self.FARM_WINDOW)

    def _farming(self, shares: dict[str, float], episode_seconds: float | None) -> str | None:
        goal = sum(share for label, share in shares.items() if _is_goal(label))
        if episode_seconds is None or goal <= 0:
            return None
        self.rows.append((shares, episode_seconds))
        if len(self.rows) < self.FARM_WINDOW:
            return None
        rows = list(self.rows)
        old, new = rows[: self.FARM_RECENT], rows[-self.FARM_RECENT :]

        def goal_of(part: list[tuple[dict[str, float], float]]) -> float:
            return statistics.median(sum(v for k, v in s.items() if _is_goal(k)) for s, _ in part)

        def seconds_of(part: list[tuple[dict[str, float], float]]) -> float:
            return statistics.median(seconds for _, seconds in part)

        goal_old, goal_new = goal_of(old), goal_of(new)
        long_old, long_new = seconds_of(old), seconds_of(new)
        if goal_new >= self.FARM_GOAL_DROP * goal_old or long_new < self.FARM_LONGER * long_old:
            return None
        names = {name for s, _ in rows for name in s if not _is_goal(name)}

        def mean(part: list[tuple[dict[str, float], float]], name: str) -> float:
            return sum(s.get(name, 0.0) for s, _ in part) / len(part)

        rising = sorted(names, key=lambda name: mean(new, name) - mean(old, name), reverse=True)[:3]
        listed = ", ".join(f"{n} {mean(old, n):.0%} -> {mean(new, n):.0%}" for n in rising)
        return (
            f"The bot may be farming: games went from {long_old:.0f} s to {long_new:.0f} s and the goal's "
            f"share of the pay from {goal_old:.0%} to {goal_new:.0%}; rising most: {listed}. Consider going "
            "back to a save from before, lowering those weights or raising the goal's."
        )


def test_the_kept_goal_share_gives_the_farming_warnings_summing_it_again_gave() -> None:
    """The farming check keeps each iteration's goal share instead of summing 100 rows' again. Over 520
    noisy iterations (two goal columns, games that did not end, a goal paying nothing; the window full from
    about the 320th) the warnings are those of the check as it was, word for word, iteration by iteration."""
    rng = random.Random(11)
    kept, before = TrendWarnings(), _FarmingAsBefore()
    farming = 0
    for i in range(520):
        goal = max(0.0, 0.3 * (1 - i / 700) + rng.uniform(-0.03, 0.03)) if i % 37 else 0.0
        raw = {"goal": goal, "goal_2": goal / 3, "touch_ball": rng.uniform(0.2, 0.4)}
        raw |= {"pickup_boost": 0.1 + i / 1000 + rng.uniform(0, 0.05), "velocity": rng.uniform(0.1, 0.3)}
        total = sum(raw.values())
        shares = {label: value / total for label, value in raw.items()}
        seconds = None if i % 23 == 0 else 50 + 0.3 * i + rng.uniform(-5, 5)
        found = kept.check(HEALTHY, 1000.0, dict(shares), None, seconds)
        assert found == before.check(HEALTHY, 1000.0, dict(shares), None, seconds), f"iteration {i}"
        farming += sum("farming" in warning for warning in found)
    assert farming >= 2, "the series warns of farming, and again REPEAT iterations later"
