"""Gameplay statistics per iteration (the core's PlayStats, idea credited to RLMind): the columns
computed from the sums."""

from __future__ import annotations

import pytest

from nitro.training import PLAY_COLUMNS, play_columns


def test_the_columns_are_shares_rates_and_averages_of_the_counted_cars() -> None:
    play = {
        "car_steps": 900.0,  # 900 steps x 8 ticks = 1 minute of play for one car
        "in_air": 300.0,
        "touches": 6.0,
        "demolished": 9.0,
        "supersonic": 90.0,
        "speed": 900 * 1000.0,
        "toward_ball": 900 * 250.0,
        "boost": 900 * 33.0,
        "ball_on_car": 45.0,
        "aerial_touches": 2.0,
        "dribbles": 1.0,
        "flicks": 1.0,
        "double_taps": 0.0,
        "flip_resets": 2.0,
        "flip_reset_height": 2 * 1200.0,
        "flip_reset_goal_dist": 2 * 3000.0,
        "reset_musty_goals": 1.0,
        "reset_musty_goal_speed": 3000.0,
        "shots": 3.0,
        "saves": 2.0,
        "demos": 1.0,
    }
    row = play_columns(play, tick_skip=8)
    assert row["ball_on_car_share"] == pytest.approx(0.05) and row["shots_per_minute"] == pytest.approx(3.0)
    assert row["aerial_touches_per_minute"] == pytest.approx(2.0) and row[
        "saves_per_minute"
    ] == pytest.approx(2.0)
    assert set(row) == set(PLAY_COLUMNS)
    assert row["in_air_share"] == pytest.approx(1 / 3)
    assert row["touches_per_minute"] == pytest.approx(6.0)
    assert row["speed_kmh"] == pytest.approx(36.0)
    assert row["speed_to_ball_kmh"] == pytest.approx(9.0)
    assert row["boost_held"] == pytest.approx(33.0)
    assert row["supersonic_share"] == pytest.approx(0.1) and row["demolished_share"] == pytest.approx(0.01)
    assert row["flip_reset_height"] == pytest.approx(1200.0)
    assert row["flip_reset_goal_dist"] == pytest.approx(3000.0)
    assert row["reset_musty_goals_per_hour"] == pytest.approx(60.0)
    assert row["reset_musty_goal_speed_kmh"] == pytest.approx(108.0)
    assert set(play_columns({"car_steps": 0.0}, 8).values()) == {""}


def test_every_gameplay_column_has_a_chart_label_and_a_wandb_section() -> None:
    from nitro import metrics, tracking

    assert tracking._PLAY == PLAY_COLUMNS
    assert all(column in metrics.CHARTS for column in PLAY_COLUMNS)
