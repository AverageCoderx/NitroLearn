"""`nitro eval`: counting points, the interval and its reading on made-up games; the command's errors; and a
tiny real match when the collision meshes are present."""

from __future__ import annotations

import csv
import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro.cli import main
from nitro.config import default_config
from nitro.eval import (
    BothWays,
    EvalError,
    MatchResult,
    Side,
    load_side,
    play,
    play_both_ways,
    summarise,
    wilson,
)
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import RunFolder, fixed_settings

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


@dataclass(frozen=True)
class Record:
    env: int
    car: int
    goal: int
    seconds: float


def points(blue: int, orange: int, none: int) -> list[Record]:
    outcomes = [1] * blue + [-1] * orange + [0] * none
    records = []
    for env, goal in enumerate(outcomes):
        records.append(Record(env, 0, goal, 20.0 if goal else 60.0))
        records.append(Record(env, 1, -goal, 20.0 if goal else 60.0))  # Orange's view of the same point
    return records


def test_points_are_counted_once_and_read_with_an_interval() -> None:
    result = summarise(points(blue=301, orange=190, none=21), time_limit=60.0)
    assert (result.points, result.blue_goals, result.orange_goals, result.no_goal) == (512, 301, 190, 21)
    assert result.blue_share == pytest.approx(301 / 491)
    low, high = result.interval()
    assert 0.5 < low < result.blue_share < high < 0.7
    assert result.reading().startswith("Blue is clearly better")
    assert summarise(points(190, 301, 21), 60.0).reading().startswith("Orange is clearly better")
    assert "No clear difference" in summarise(points(26, 24, 0), 60.0).reading()
    assert "Nobody scored" in summarise(points(0, 0, 8), 60.0).reading()
    text = "\n".join(result.lines("new", "old"))
    assert "new (Blue) vs old (Orange): 512 points" in text and "61% (95% interval" in text
    assert "21 ended without a goal (the 60 s limit)." in text
    touchless = summarise(points(blue=301, orange=190, none=21), 20.0, no_touch=10.0)
    assert "21 ended without a goal (the 20 s limit or no touch for 10 s)." in "\n".join(
        touchless.lines("new", "old")
    ), "the no-touch timer ends points too"


def test_wilson_interval_matches_known_values() -> None:
    low, high = wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(bound) for bound in wilson(0, 0))


def save(folder: RunFolder, agent_steps: int, seed: int) -> None:
    config = folder.load_settings()
    torch.manual_seed(seed)
    model = ActorCritic(148, 90, config.model)
    state = {
        "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": agent_steps,
        "iterations": 1,
        "settings": fixed_settings(config),
    }
    folder.save_checkpoint(state, agent_steps, keep=5, milestone_steps=0)


def test_the_command_explains_what_it_cannot_do(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    assert main(["eval", "ghost", "--runs", str(runs)]) == 1
    assert "no run called 'ghost'" in capsys.readouterr().err
    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n', encoding="utf-8")
    folder = RunFolder.create(runs, "my-bot", settings)
    assert main(["eval", "my-bot", "--runs", str(runs)]) == 1
    assert "no save yet" in capsys.readouterr().err
    save(folder, 500, seed=1)
    assert main(["eval", "my-bot", "--runs", str(runs)]) == 1
    assert "only one save" in capsys.readouterr().err
    assert main(["eval", "my-bot", "--runs", str(runs), "--vs-at", "900"]) == 1
    assert "There is no save of my-bot at 900 agent steps. Its saves: 500." in capsys.readouterr().err
    lone = tmp_path / "lone.toml"
    lone.write_text(
        '[run]\ndevice = "cpu"\n[env]\nopponents = false\n[model]\npolicy = [8]\nvalue = [8]\n',
        encoding="utf-8",
    )
    alone = RunFolder.create(runs, "alone", lone)
    save(alone, 500, seed=1)
    save(alone, 1000, seed=2)
    with pytest.raises(EvalError, match="without opponents"):
        play(load_side(alone, None), load_side(alone, None, oldest=True), games=4, seed=0)
    # Two saves that saw different games (here: with and without opponents) are refused with the difference.
    with pytest.raises(EvalError, match=r"read the game differently.*whether it plays against opponents"):
        play(load_side(folder, None), load_side(alone, None), games=4, seed=0)


def tiny(folder: RunFolder, agent_steps: int) -> Side:
    side = load_side(folder, agent_steps)
    config = dataclasses.replace(
        side.config,
        paths=dataclasses.replace(side.config.paths, collision_meshes=MESHES.resolve()),
        env=dataclasses.replace(side.config.env, num_arenas=4, groups=2, threads=2),
        terminals=dataclasses.replace(side.config.terminals, max_episode_seconds=3.0),
    )
    return dataclasses.replace(side, config=config)


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_two_saves_play_a_short_match_and_every_point_is_counted(tmp_path: Path) -> None:
    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    save(folder, 500, seed=1)
    save(folder, 1000, seed=2)
    blue, orange = tiny(folder, 1000), tiny(folder, 500)
    result = play(blue, orange, games=6, seed=0)
    assert result.points == 8, "6 points round up to two rounds of 4 arenas"
    assert result.blue_goals + result.orange_goals + result.no_goal == result.points
    assert 0 < result.mean_seconds <= 3.0 + 0.1
    assert play(blue, orange, games=6, seed=0) == result, "the same seed plays the same games"
    assert isinstance(default_config(Path.cwd()), type(blue.config))


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_run_without_a_time_limit_plays_with_one_given_for_the_match(tmp_path: Path) -> None:
    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    save(folder, 500, seed=1)
    save(folder, 1000, seed=2)

    def endless(side: Side) -> Side:
        terminals = dataclasses.replace(side.config.terminals, max_episode_seconds=0.0)
        return dataclasses.replace(side, config=dataclasses.replace(side.config, terminals=terminals))

    blue, orange = endless(tiny(folder, 1000)), endless(tiny(folder, 500))
    given = play(blue, orange, games=4, seed=0)  # no limit asked for either: five minutes a point
    assert given.limit_added and given.time_limit == 300.0 and given.points == 4
    said = "\n".join(given.lines("a", "b"))
    assert "The bot's own games have no time limit, so each point here ended after 300 seconds" in said
    with pytest.raises(EvalError, match="above 0"):
        play(blue, orange, games=4, seed=0, max_seconds=0)
    result = play(blue, orange, games=4, seed=0, max_seconds=2.0)
    assert result.points == 4 and result.time_limit == 2.0 and not result.limit_added
    assert 0 < result.mean_seconds <= 2.0 + 0.1
    assert blue.config.terminals.max_episode_seconds == 0.0, "the run's own settings are not changed"


@pytest.mark.parametrize(
    "argv",
    [
        ["watch", "my-bot", "--vs-bot", "bot.toml"],
        ["eval", "my-bot", "--vs-bot", "bot.toml"],
    ],
)
def test_outside_bots_are_not_options(argv: list[str]) -> None:
    """Playing outside RLBot bots is not part of this version: the options are unknown."""
    with pytest.raises(SystemExit) as refused:
        main(argv)
    assert refused.value.code == 2


def at_tick_skip(side: Side, tick_skip: int, action_delay: int | None = None) -> Side:
    env = side.config.env
    env = dataclasses.replace(
        env, tick_skip=tick_skip, action_delay=env.action_delay if action_delay is None else action_delay
    )
    return dataclasses.replace(side, config=dataclasses.replace(side.config, env=env))


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_bots_at_different_tick_skips_play_one_game_each_on_its_own_rhythm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nitro.evaluate as evaluate_module

    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    save(folder, 500, seed=1)
    save(folder, 1000, seed=2)
    blue, orange = at_tick_skip(tiny(folder, 1000), 8, 0), at_tick_skip(tiny(folder, 500), 4, 0)
    seen: list[tuple[list[int], list[int]]] = []
    picked: list[list[int]] = []
    real_step, real_pick = evaluate_module.EnvPool.step, evaluate_module.apply_pick

    def recording_step(self: object, group: int) -> None:
        tensors = self.groups[group]  # type: ignore[attr-defined]
        seen.append((tensors.step.tolist(), tensors.actions.tolist()))
        real_step(self, group)  # type: ignore[arg-type]

    def recording_pick(*args: Any) -> torch.Tensor:
        chosen = real_pick(*args)
        picked.append(chosen.tolist())
        return chosen

    monkeypatch.setattr(evaluate_module.EnvPool, "step", recording_step)
    monkeypatch.setattr(evaluate_module, "apply_pick", recording_pick)
    # Sampled as in training, then both sides likeliest (the pick comes before the hold).
    for pick in ((1.0, 1.0), (0.0, 0.0)):
        seen.clear()
        picked.clear()
        result = play(blue, orange, games=4, seed=0, pick=pick)
        assert result.points == 4 and result.blue_goals + result.orange_goals + result.no_goal == 4
        # The match runs at tick skip 4: Blue (8) chooses on even steps of its episode and holds in between.
        previous: dict[tuple[int, int], int] = {}
        held = overruled = 0
        for group, (steps, actions) in enumerate(seen):
            for row, (step, action) in enumerate(zip(steps, actions, strict=True)):
                key = (group % 2, row)
                if row % 2 == 0 and step % 2 == 1 and key in previous:
                    assert action == previous[key], f"Blue holds its action for 8 ticks ({pick})"
                    held += 1
                    if picked and picked[group][row] != action:
                        overruled += 1
                previous[key] = action
        assert held > 0
        if pick == (0.0, 0.0):
            assert len(picked) == len(seen) and overruled > 0, "the hold overrules a new likeliest action"
        else:
            assert not picked, "sampled: no pick rule runs"
    # Swapped sides work the same way; tick skips that do not divide, or different delays, are refused.
    assert play(orange, blue, games=4, seed=0).points == 4
    with pytest.raises(EvalError, match="multiple of the other"):
        play(at_tick_skip(blue, 6), orange, games=4, seed=0)
    with pytest.raises(EvalError, match="the action delay: 1 against 0"):
        play(at_tick_skip(blue, 8, 1), orange, games=4, seed=0)


def test_both_ways_pools_the_two_legs_from_the_first_sides_view() -> None:
    """A's goals as Blue and as Orange against B's, one share with one Wilson interval."""
    as_blue = MatchResult(
        110, blue_goals=60, orange_goals=40, no_goal=10, mean_seconds=20.0, time_limit=300.0
    )
    as_orange = MatchResult(
        110, blue_goals=45, orange_goals=55, no_goal=10, mean_seconds=20.0, time_limit=300.0
    )
    both = BothWays(as_blue, as_orange)
    assert (both.goals_for, both.goals_against, both.decided, both.timeouts) == (115, 85, 200, 20)
    assert both.share == 115 / 200
    # By hand: p = 0.575, n = 200, z = 1.96: centre 0.5736, half-width 0.0679.
    low, high = both.interval()
    assert low == pytest.approx(0.50571, abs=1e-4) and high == pytest.approx(0.64146, abs=1e-4)
    assert both.reading().startswith("The first is clearly better")
    text = "\n".join(both.lines("new", "old"))
    assert "new vs old, both ways round from the same starts: 110 points as Blue, 110 as Orange." in text
    assert (
        "As Blue it scored 60 and let in 40; as Orange it scored 55 and let in 45; 20 ended without a goal "
        "(the 300 s limit)." in text
    )
    assert "as Blue 60%, as Orange 55%; both ways 57% (95% interval 51% to 64%)" in text
    assert BothWays(as_orange, as_blue).reading().startswith("The second is clearly better")
    even = BothWays(MatchResult(10, 3, 3, 4, 20.0, 300.0), MatchResult(10, 3, 3, 4, 20.0, 300.0))
    assert even.share == 0.5 and "No clear difference" in even.reading()
    none = BothWays(MatchResult(4, 0, 0, 4, 60.0, 60.0), MatchResult(4, 0, 0, 4, 60.0, 60.0))
    assert (
        math.isnan(none.share)
        and "Nobody scored" in none.reading()
        and "both ways -" in none.lines("a", "b")[2]
    )


def test_log_is_refused_without_both_ways(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    assert main(["eval", "my-bot", "--runs", str(runs), "--log"]) == 1
    assert "add --both-ways" in capsys.readouterr().err


def rolling_saves(tmp_path: Path) -> RunFolder:
    """A run whose every start rolls the ball at one goal: one colour scores far more from these starts."""
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[run]\ndevice = "cpu"\n[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[env]\n'
        "num_arenas = 8\ngroups = 2\nthreads = 2\n[terminals]\nmax_episode_seconds = 4.0\n[model]\n"
        'policy = [8]\nvalue = [8]\n[[state_setters]]\nname = "save"\n',
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    save(folder, 500, seed=1)
    return folder


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_save_against_itself_both_ways_is_even_whatever_the_starts_favour(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`nitro eval --both-ways --log`: both legs on one seed, so a save against itself plays the same games
    with the colours swapped; the starts' lean to one colour cancels and the pooled share is even."""
    import nitro.eval as eval_module

    folder = rolling_saves(tmp_path)
    seeds: list[int] = []
    real = eval_module.evaluate

    def recording(model: object, config: object, seed: int, *args: object, **kwargs: object) -> object:
        seeds.append(seed)
        return real(model, config, seed, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(eval_module, "evaluate", recording)
    runs = str(tmp_path / "runs")
    assert (
        main(["eval", "my-bot", "--runs", runs, "--vs-at", "500", "--games", "16", "--both-ways", "--log"])
        == 0
    )
    assert seeds == [0, 0], "both legs play from the same starts"
    out = capsys.readouterr().out
    assert out.startswith("Playing 16 points each way: my-bot at 500 agent steps as Blue, then as Orange")
    with (folder.root / "evals.csv").open(encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert list(row) == list(eval_module.EVAL_COLUMNS)
    as_blue = int(row["as_blue_for"]), int(row["as_blue_against"])
    as_orange = int(row["as_orange_for"]), int(row["as_orange_against"])
    assert sum(as_blue) + sum(as_orange) + int(row["timeouts"]) == 32, "every point of both legs counted once"
    assert as_blue == as_orange[::-1], "the same games with the colours swapped"
    assert as_blue[0] != as_blue[1], "these starts favour one colour"
    assert row["share"] == "0.5000" and float(row["low"]) < 0.5 < float(row["high"])
    assert (row["agent_steps"], row["opponent"], row["opponent_steps"], row["seed"]) == (
        "500",
        "my-bot",
        "500",
        "0",
    )
    # Without the new flags the command reports one way, as before, and writes no file.
    assert main(["eval", "my-bot", "--runs", runs, "--at", "500", "--vs-at", "500", "--games", "4"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "Playing 4 points: my-bot at 500 agent steps (Blue) vs my-bot at 500 agent steps (Orange)"
    )
    assert "  Blue's share of the goals:" in out
    with (folder.root / "evals.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 1


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_both_legs_play_the_first_sides_games(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`play(game=...)`: B as Blue still plays A's starts and endings; without `game`, Blue's as before."""
    import nitro.eval as eval_module

    folder = rolling_saves(tmp_path)
    save(folder, 1000, seed=2)
    a = tiny(folder, 1000)
    a = dataclasses.replace(
        a, config=dataclasses.replace(a.config, state_setters=load_side(folder, 500).config.state_setters)
    )
    b = tiny(folder, 500)
    b = dataclasses.replace(
        b, config=dataclasses.replace(b.config, state_setters=default_config(Path.cwd()).state_setters)
    )
    assert a.config.state_setters != b.config.state_setters
    seen: list[tuple[object, object]] = []
    real = eval_module.evaluate

    def recording(model: object, config: object, *args: object, **kwargs: object) -> object:
        seen.append((config.state_setters, config.terminals))  # type: ignore[attr-defined]
        return real(model, config, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(eval_module, "evaluate", recording)
    both = play_both_ways(a, b, games_each=4, seed=3, max_seconds=2.0)
    assert both.as_blue.points == both.as_orange.points == 4
    terminals = dataclasses.replace(a.config.terminals, max_episode_seconds=2.0)
    assert seen == [(a.config.state_setters, terminals)] * 2
    play(b, a, games=4, seed=3, max_seconds=2.0)
    assert seen[-1][0] == b.config.state_setters, "without game, Blue's settings as before"
