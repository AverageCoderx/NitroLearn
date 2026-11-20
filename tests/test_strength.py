"""The strength chart against pinned anchors : the scale recovers known strengths (synthetic and
scripted players), the schedule of anchors, the same seed the same ratings, a run folder's file."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import pytest
import torch

from nitro.cli import main
from nitro.config import load_config
from nitro.dashboard import charts
from nitro.gauntlet import play_scripts
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import RunFolder, fixed_settings
from nitro.strength import (
    ANCHOR,
    FILE,
    Match,
    chart,
    due_anchor,
    expected,
    fit,
    lines,
    rate,
    rate_save,
    read,
    write,
)

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")


def test_the_scale_is_elo_like() -> None:
    assert expected(0, 0) == pytest.approx(0.5)
    assert expected(400, 0) == pytest.approx(10 / 11)
    rating, low, high = fit([Match(0, 0.0, 909, 91)])  # ten to one
    assert rating == pytest.approx(400, abs=3) and low < rating < high


def test_a_clean_sweep_still_gets_a_finite_rating_and_more_points_narrow_it() -> None:
    rating, _, _ = fit([Match(0, 0.0, 20, 0)])
    assert math.isfinite(rating) and rating > 300
    wide = fit([Match(0, 0.0, 6, 4)])
    narrow = fit([Match(0, 0.0, 600, 400)])
    assert narrow[2] - narrow[1] < (wide[2] - wide[1]) / 5


def test_anchors_are_pinned_on_schedule() -> None:
    assert not due_anchor([], 0, 0, 100), "off"
    assert due_anchor([], 5, 3, 100), "the first save always"
    assert not due_anchor([5], 90, 3, 100)
    assert due_anchor([5], 100, 3, 100)
    assert not due_anchor([5, 100], 150, 3, 100)
    assert due_anchor([5, 100], 230, 3, 100)
    assert not due_anchor([5, 100, 230], 900, 3, 100), "at most `count`"


def _synthetic_run(seed: int, points: int = 400) -> dict[str, float]:
    """Saves of known strength (true Elo) rated one after another; the first, third and fifth are pinned."""
    truth = {0: 0.0, 1: 120.0, 2: 260.0, 3: 380.0, 4: 520.0, 5: 700.0}
    rng = random.Random(seed)
    data: dict = {"anchors": [], "ratings": []}

    for save, true in truth.items():

        def play(anchor: int, true: float = true) -> tuple[int, int, int]:
            p = expected(true, truth[anchor])
            won = sum(rng.random() < p for _ in range(points))
            return won, points - won, 0

        rate(data, save, play, pin=save in (0, 2, 4))
    rated = {r["agent_steps"]: r["rating"] for r in data["ratings"]}
    rated[0] = 0.0
    return rated


def test_players_of_known_strength_get_the_right_order_and_gaps() -> None:
    truth = [0.0, 120.0, 260.0, 380.0, 520.0, 700.0]
    for seed in range(5):
        rated = _synthetic_run(seed)
        order = sorted(rated, key=rated.get)
        assert order == list(range(6)), (seed, rated)
        for save, true in enumerate(truth):
            # 400 points a match: the anchors' own errors add up along the chain, still within ~45 Elo.
            assert abs(rated[save] - true) < 60, (seed, save, rated[save], true)
    assert _synthetic_run(3) == _synthetic_run(3), "the same seed gives the same ratings"


def test_an_anchors_rating_never_changes_once_pinned() -> None:
    data: dict = {"anchors": [], "ratings": []}
    rate(data, 0, lambda a: (0, 0, 0), pin=True)
    rate(data, 10, lambda a: (70, 30, 0), pin=True)
    pinned = [dict(a) for a in data["anchors"]]
    rate(data, 20, lambda a: (90, 10, 0), pin=False)
    assert data["anchors"] == pinned and pinned[0]["rating"] == 0.0
    assert [r["agent_steps"] for r in data["ratings"]] == [10, 20]


def test_the_chart_reads_the_file(tmp_path: Path) -> None:
    path = tmp_path / FILE
    assert chart(path) is None and read(path) == {"anchors": [], "ratings": []}
    assert lines(read(path))[0].startswith("No anchors yet")
    data: dict = {"anchors": [], "ratings": []}
    rate(data, 0, lambda a: (0, 0, 0), pin=True)
    rate(data, 10, lambda a: (60, 40, 2), pin=False)
    write(path, data)
    assert "10 agent steps: 70 (" in "\n".join(lines(read(path)))
    drawn = chart(path)
    assert drawn is not None and drawn["points"][0][0] == 10 and drawn["anchors"] == [[0, 0.0]]
    assert drawn["band"][0][1] < drawn["points"][0][1] < drawn["band"][0][2]


@needs_meshes
def test_scripted_players_rate_where_their_head_to_head_says(tmp_path: Path) -> None:
    """Anchors: the goalie (0) and the chaser (pinned from its goals against the goalie). A copy of the
    chaser must get the chaser's rating, the idle player the lowest, and the gap between the kickoff
    rusher and the chaser must predict their own head-to-head share."""
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nthreads = 2\nnum_arenas = 32\n",
        encoding="utf-8",
    )
    config = load_config(settings)
    points = 512
    names = {0: "goalie", 1: "chaser", 2: "chaser", 3: "kickoff", 4: "idle"}
    data: dict = {"anchors": [], "ratings": []}
    for steps, name in names.items():

        def play(anchor: int, name: str = name, steps: int = steps) -> tuple[int, int, int]:
            r = play_scripts(config, name, names[anchor], points, 11 + steps * 7 + anchor)
            return r.blue_goals, r.orange_goals, r.no_goal

        rate(data, steps, play, pin=steps in (0, 1))
    rated = {r["agent_steps"]: r for r in data["ratings"]}
    chaser, copy = rated[1], rated[2]
    assert copy["low"] < chaser["rating"] < copy["high"], (chaser, copy)
    assert rated[4]["rating"] < min(r["rating"] for s, r in rated.items() if s != 4)
    direct = play_scripts(config, "kickoff", "chaser", points, 99)
    low, high = direct.interval()
    predicted = expected(rated[3]["rating"], chaser["rating"])
    margin = 0.08  # the two ratings' own uncertainty
    assert low - margin < predicted < high + margin, (predicted, low, high)


def _save(folder: RunFolder, agent_steps: int, seed: int) -> Path:
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
    return folder.save_checkpoint(state, agent_steps, keep=1, milestone_steps=0)


def _rated_run(tmp_path: Path) -> tuple[RunFolder, dict]:
    tmp_path.mkdir()
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nthreads = 2\nnum_arenas = 16\n[model]\npolicy = [8]\nvalue = [8]\n"
        "[strength]\nanchors = 2\nanchor_every_steps = 1000\npoints = 32\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    config = folder.load_settings()
    for steps, seed in ((100, 1), (600, 2), (1200, 3)):
        path = _save(folder, steps, seed)
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        rate_save(folder, state, config, path)
    return folder, read(folder.root / FILE)


@needs_meshes
def test_a_run_pins_its_anchors_keeps_them_and_rates_each_save(tmp_path: Path) -> None:
    folder, data = _rated_run(tmp_path / "a")
    assert [a["agent_steps"] for a in data["anchors"]] == [100, 1200]
    assert [r["agent_steps"] for r in data["ratings"]] == [600, 1200]
    kept = [int(p.name) for p in folder.complete_checkpoints()]
    assert kept == [100, 1200], "keep=1 pruned the save at 600, never an anchor"
    assert all((folder.checkpoints / f"{s:012d}" / ANCHOR).is_file() for s in (100, 1200))
    folder.metrics.write_text("agent_steps,reward\n600,0.1\n1200,0.2\n", encoding="utf-8")
    shown = charts.view(folder, 0, None)
    assert "strength" in shown["default"]
    assert shown["series"]["strength"] == [[r["agent_steps"], r["rating"]] for r in data["ratings"]]
    assert main(["strength", "my-bot", "--runs", str(folder.root.parent)]) == 0
    _, again = _rated_run(tmp_path / "b")
    assert again == data, "the same seed plays the same matches"


def _sized_save(folder: RunFolder, agent_steps: int, seed: int) -> tuple[Path, dict]:
    """A save of `folder` as its settings now stand (networks sized for its game)."""
    from nitro.eval import _sizes
    from nitro.model import build_networks

    config = folder.load_settings()
    torch.manual_seed(seed)
    model = build_networks(config, *_sizes(config))
    state = {
        "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": agent_steps,
        "iterations": 1,
        "settings": fixed_settings(config),
    }
    path = folder.save_checkpoint(state, agent_steps, keep=5, milestone_steps=0)
    return path, torch.load(path / "state.pt", map_location="cpu", weights_only=True)


@needs_meshes
@pytest.mark.parametrize(
    "model",
    ["[model]\npolicy = [8]\nvalue = [8]\n", '[model]\nencoder = "attention"\npolicy = [8]\nvalue = [8]\n'],
)
def test_the_chart_carries_on_after_a_resume_at_another_team_size(tmp_path: Path, model: str) -> None:
    """A bot that can change its game size between stop and resume (a padded observation, or attention)
    keeps being rated against the anchors pinned at its old size."""
    settings = tmp_path / "bot.toml"
    base = (
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[strength]\nanchors = 3\nanchor_every_steps = 1000\npoints = 16\n" + model
    )
    settings.write_text(
        base + "[env]\nthreads = 2\nnum_arenas = 8\nteam_size = 1\nobs_team_size = 2\n", "utf-8"
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    path, state = _sized_save(folder, 100, 1)
    assert rate_save(folder, state, folder.load_settings(state), path) is None, "the first anchor"
    folder.settings.write_text(
        base + "[env]\nthreads = 2\nnum_arenas = 8\nteam_size = 2\nobs_team_size = 2\n", encoding="utf-8"
    )
    path, state = _sized_save(folder, 600, 2)
    entry = rate_save(folder, state, folder.load_settings(state), path)
    assert entry is not None and [m["anchor"] for m in entry["matches"]] == [100]
    assert sum(entry["matches"][0][k] for k in ("won", "lost", "draws")) >= 16
