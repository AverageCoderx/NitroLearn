"""Self-play against past versions: ratings, matchmaking, the pool, who drives which rows,
and that only the learner's rows are trained on."""

from __future__ import annotations

import csv
import dataclasses
import logging
import math
import os
import random
import shutil
import statistics
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest
import torch

from nitro.config import default_config, load_config
from nitro.config.loader import parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, SelfPlayConfig
from nitro.config.state_setters import StateSetter, _StateSetter
from nitro.envs import GroupTensors
from nitro.health import TrainingProblem
from nitro.inference import GroupInference
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ratings import (
    BETA,
    MIN_WEIGHT,
    MU,
    SIGMA,
    Rating,
    draw,
    matchmaking_weights,
    update,
    win_probability,
)
from nitro.rollout import Rollout
from nitro.runs import RunFolder
from nitro.selfplay import (
    _LOG_COLUMNS,
    GameStats,
    OpponentBlock,
    Pool,
    SelfPlay,
    arena_blocks,
    group_rows,
)
from nitro.training import (
    EVEN_START_COLUMNS,
    SELFPLAY_COLUMNS,
    TrainingRun,
    even_starts,
    rotation_rng,
    start_edge,
)

ROOT = Path(__file__).parents[1]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
OBS, ACTIONS = 16, 90
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")


def test_ratings_move_by_the_weng_lin_formula() -> None:
    # Two new players, one game: c = sqrt(2 sigma^2 + 2 beta^2) = 13.176, p = 0.5, omega = sigma^2 / c / 2.
    winner, loser = update(Rating(), Rating(), 1.0)
    assert abs(winner.mu - 27.635) < 0.01 and abs(loser.mu - 22.365) < 0.01
    assert abs(winner.sigma - 8.066) < 0.01 and winner.sigma == loser.sigma
    assert win_probability(Rating(), Rating()) == 0.5
    assert 0.8 < win_probability(Rating(30, 1), Rating(20, 1)) < 0.9  # c = 6.06, exp(-10/c) = 0.19
    # A draw between equals changes nothing but the uncertainty.
    a, b = update(Rating(), Rating(), 0.5)
    assert a.mu == b.mu == MU and a.sigma < SIGMA
    # The floor keeps a rating movable.
    floored, _ = update(Rating(25, 0.5), Rating(), 1.0, min_sigma=BETA)
    assert floored.sigma == BETA


def test_matchmaking_prefers_even_matches_and_draws_without_replacement() -> None:
    learner = Rating(30, 2)
    pool = [Rating(30, 2), Rating(10, 2), Rating(50, 2)]
    variance = matchmaking_weights(learner, pool, "variance")
    assert variance[0] > variance[1] and variance[0] > variance[2]
    hardest = matchmaking_weights(learner, pool, "hardest")
    assert hardest[2] > hardest[0] > hardest[1]
    assert matchmaking_weights(learner, pool, "uniform") == [1.0, 1.0, 1.0]
    chances = [win_probability(learner, opponent) for opponent in pool]
    squared = matchmaking_weights(learner, pool, "squared")
    capped = matchmaking_weights(learner, pool, "linear_capped")
    for p, s, c, h in zip(chances, squared, capped, hardest, strict=True):
        assert s == pytest.approx(max((1 - p) ** 2, MIN_WEIGHT), rel=1e-6) and s <= h + 1e-9
        assert c == pytest.approx(max(min(0.5, 1 - p), MIN_WEIGHT), rel=1e-6)
    # Everything the bot beats at most half the time gets the same capped weight.
    hard_pool = [Rating(60.0, 1.0), Rating(80.0, 1.0)]
    assert matchmaking_weights(learner, hard_pool, "linear_capped") == [0.5, 0.5]
    rng = random.Random(3)
    picked = draw([1.0, 1.0, 1.0, 1.0], 3, rng)
    assert len(picked) == len(set(picked)) == 3
    assert draw([1.0, 1.0], 5, rng) in ([0, 1], [1, 0]), "never more than there are"


def test_arena_blocks_are_contiguous_and_map_to_contiguous_rows_in_each_group() -> None:
    blocks = arena_blocks(num_arenas=10, past_share=0.4, slots=2)
    assert [(b.slot, b.start, b.stop) for b in blocks] == [(0, 6, 8), (1, 8, 10)]
    # Arena i sits in group i % 2 at position i // 2; two cars per arena.
    assert group_rows(blocks[0], groups=2, group=0, cars_per_env=2) == slice(6, 8)  # arena 6
    assert group_rows(blocks[0], groups=2, group=1, cars_per_env=2) == slice(6, 8)  # arena 7
    assert group_rows(blocks[1], groups=2, group=1, cars_per_env=2) == slice(8, 10)  # arena 9
    assert arena_blocks(8, 0.5, 8) == [dataclasses.replace(b, slot=b.slot) for b in arena_blocks(8, 0.5, 8)]
    assert len(arena_blocks(4, 0.25, 3)) == 1, "one past arena can fill only one slot"


def test_the_pool_keeps_a_spread_of_ratings_within_its_caps(tmp_path: Path) -> None:
    pool = Pool(tmp_path / "pool", size=3, disk_bytes=10**9)
    weights = {"policy.0.weight": torch.zeros(4)}
    ratings = {}
    for steps, mu in ((1, 10.0), (2, 11.0), (3, 20.0), (4, 30.0), (5, 40.0)):
        pool.save(steps, weights)
        ratings[steps] = Rating(mu, 1.0)
    removed = pool.prune(ratings, keep=set())
    assert removed == [1, 2] and pool.snapshots() == [3, 4, 5], (
        "the two closest to their neighbours went first"
    )
    # An active snapshot is never removed; the newest neither.
    assert pool.prune({**ratings, 6: Rating(41.0, 1.0)}, keep={3}) == []
    pool.save(6, weights)
    # Over the cap again: 5 (rated 40) sits closest to a neighbour (6 at 41) and goes; 6 is the newest.
    assert pool.prune({**ratings, 6: Rating(41.0, 1.0)}, keep={3}) == [5]
    tiny = Pool(tmp_path / "tiny", size=10, disk_bytes=1)
    tiny.save(1, weights)
    tiny.save(2, weights)
    assert tiny.prune({1: Rating(), 2: Rating()}, keep=set()) == [1], "over the byte cap: down to the newest"
    assert torch.equal(pool.load(6)["policy.0.weight"], torch.zeros(4))


def _config(**selfplay: object) -> NitroConfig:
    config = default_config(Path.cwd())
    return dataclasses.replace(
        config,
        env=dataclasses.replace(config.env, num_arenas=8, groups=2),
        run=dataclasses.replace(config.run, device="cpu"),
        model=ModelConfig(policy=(8,), value=(8,)),
        selfplay=SelfPlayConfig(
            enabled=True, past_share=0.5, active_opponents=2, snapshot_steps=1_000_000, **selfplay
        ),  # type: ignore[arg-type]
    )


def _selfplay(tmp_path: Path, model: ActorCritic, config: NitroConfig) -> SelfPlay:
    return SelfPlay(
        config, tmp_path / "pool", model, torch.device("cpu"), groups=2, cars_per_env=2, group_agents=[8, 8]
    )


def test_rotation_drives_orange_rows_of_the_past_arenas_and_excludes_them_from_learning(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    config = _config()
    model = ActorCritic(OBS, ACTIONS, config.model)
    selfplay = _selfplay(tmp_path, model, config)
    # Arenas 4-5 are slot 0 and 6-7 slot 1: group 0 holds arenas 0, 2, 4, 6 and group 1 arenas 1, 3, 5, 7.
    assert [(b.slot, b.start, b.stop) for b in selfplay.blocks] == [(0, 4, 6), (1, 6, 8)]
    selfplay.rotate(random.Random(0))
    assert selfplay.active == [None, None], "an empty pool: everyone is the learner"
    assert selfplay.learner_rows is not None and bool(selfplay.learner_rows.all())
    assert not any(block.select.any() for block in selfplay.group_opponents(0))

    assert selfplay.maybe_snapshot(999_999) is False
    assert selfplay.maybe_snapshot(1_000_000) and selfplay.maybe_snapshot(2_000_000)
    assert selfplay.pool.snapshots() == [1_000_000, 2_000_000] and selfplay.next_snapshot_steps == 3_000_000
    selfplay.rotate(random.Random(0))
    assert sorted(s for s in selfplay.active if s is not None) == [1_000_000, 2_000_000]
    rows = selfplay.learner_rows
    assert rows is not None
    # Group 0 rows 4-7 are arenas 4 (slot 0) and 6 (slot 1): their Orange cars (odd rows) are opponents.
    expected = torch.ones(16, dtype=torch.bool)
    expected[[5, 7, 8 + 5, 8 + 7]] = False
    assert torch.equal(rows, expected)
    for group in (0, 1):
        for slot, block in enumerate(selfplay.group_opponents(group)):
            assert block.rows == (slice(4, 6) if slot == 0 else slice(6, 8))
            assert block.select.tolist() == [False, True]
    assert (
        selfplay.slot_of_arena(4) == 0
        and selfplay.slot_of_arena(7) == 1
        and selfplay.slot_of_arena(2) is None
    )

    # A goal by Blue (the learner) in arena 5 counts as a win against slot 0's snapshot; car 1 is skipped.
    record = SimpleNamespace(env=5, car=0, goal=1, terminal=True)
    duplicate = SimpleNamespace(env=5, car=1, goal=-1, terminal=True)
    before = selfplay.learner.mu
    stats = selfplay.record(
        [record, duplicate, SimpleNamespace(env=1, car=0, goal=1, terminal=True)],  # type: ignore[arg-type]
        agent_steps=2_500_000,
    )
    assert (stats.games, stats.wins) == (1, 1) and selfplay.learner.mu > before
    opponent = selfplay.active[0]
    assert opponent is not None and selfplay.ratings[opponent].mu < MU
    assert (tmp_path / "pool" / "games.csv").read_text(encoding="utf-8").count("\n") == 2

    # The state survives a checkpoint round trip, active versions included, and a resume re-applies it.
    state = selfplay.state_dict()
    again = _selfplay(tmp_path, ActorCritic(OBS, ACTIONS, config.model), config)
    again.load_state_dict(state)
    assert again.active == selfplay.active and again.ratings == selfplay.ratings
    assert again.learner_rows is not None and torch.equal(again.learner_rows, rows)
    # Games recorded after a save are dropped on resume (the resumed run plays those iterations again).
    log = tmp_path / "pool" / "games.csv"
    selfplay.record([record], agent_steps=3_000_000)  # type: ignore[list-item]
    assert log.read_text(encoding="utf-8").count("\n") == 3
    again.truncate_log(2_500_000)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].startswith("agent_steps,") and lines[1].startswith("2500000,")


def test_a_resumed_run_draws_the_same_past_versions_as_the_straight_run(tmp_path: Path) -> None:
    """The straight run saves, then rotates for the next segment; the resumed run loads that save and must
    rotate to the same active versions (review finding 1)."""
    torch.manual_seed(0)
    config = _config()  # two active slots
    straight = _selfplay(tmp_path, ActorCritic(OBS, ACTIONS, config.model), config)
    for steps in range(1, 6):
        straight.maybe_snapshot(steps * 1_000_000)
        straight.ratings[steps * 1_000_000] = Rating(MU + 3.0 * steps, 2.0)
    differed = 0
    for checkpoint in range(1, 21):
        saved = straight.state_dict()  # saved before the straight run rotates, as TrainingRun does
        straight.rotate(random.Random(checkpoint))
        resumed = _selfplay(tmp_path, ActorCritic(OBS, ACTIONS, config.model), config)
        resumed.load_state_dict(saved)
        differed += resumed.active != straight.active  # what the old code played
        resumed.rotate(random.Random(checkpoint))  # what TrainingRun now does on resume
        assert resumed.active == straight.active, f"checkpoint {checkpoint}"
    assert differed > 0, "the test would pass without the fix otherwise"


def test_counted_cars_leave_out_the_frozen_opponents_of_occupied_slots(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _config()
    selfplay = _selfplay(tmp_path, ActorCritic(OBS, ACTIONS, config.model), config)
    selfplay.rotate(random.Random(0))
    # Empty slots: everyone is the learner, every car counts.
    assert selfplay.counted_cars() == {arena: [True, True] for arena in range(4, 8)}
    selfplay.maybe_snapshot(1_000_000)
    selfplay.rotate(random.Random(0))
    occupied = [slot for slot, steps in enumerate(selfplay.active) if steps is not None]
    assert len(occupied) == 1
    counted = selfplay.counted_cars()
    block = next(b for b in selfplay.blocks if b.slot == occupied[0])
    for arena in range(4, 8):
        expected = [True, False] if block.start <= arena < block.stop else [True, True]
        assert counted[arena] == expected, f"arena {arena}"


def test_only_the_learners_uncertainty_is_floored() -> None:
    learner, snapshot = update(Rating(MU, SIGMA), Rating(MU, 0.5), 1.0, min_sigma=BETA)
    assert learner.sigma >= BETA
    assert snapshot.sigma < 1.0, "a frozen snapshot's rating may settle"


def test_opponent_actions_replace_the_learners_on_selected_rows_only() -> None:
    torch.manual_seed(1)
    config = ModelConfig(policy=(8,), value=(8,))
    learner = ActorCritic(OBS, ACTIONS, config)
    opponent = ActorCritic(OBS, ACTIONS, config)
    with torch.no_grad():  # the learner always wants action 3, the opponent action 5
        learner.policy[-1].bias.fill_(0.0)
        learner.policy[-1].bias[3] = 50.0
        opponent.policy[-1].bias.fill_(0.0)
        opponent.policy[-1].bias[5] = 50.0
    tensors = GroupTensors.allocate(8, OBS, ACTIONS, pin_memory=False)
    tensors.obs.normal_()
    tensors.mask.fill_(True)
    block = OpponentBlock(slice(4, 8), torch.tensor([False, True, False, True]), opponent)
    engine = GroupInference(
        learner, tensors, torch.device("cpu"), fp16=False, cuda_graph=False, opponents=[block]
    )
    engine.run()
    assert tensors.actions.tolist() == [3, 3, 3, 3, 3, 5, 3, 5]
    block.select.fill_(False)  # an emptied slot: the learner drives every row again, no rebuild needed
    engine.run()
    assert tensors.actions.tolist() == [3] * 8


def test_the_learner_never_trains_on_opponent_rows() -> None:
    torch.manual_seed(2)
    config = default_config(Path.cwd())
    rows = torch.tensor([True, False, True, True, False, True])

    def learn_with(excluded_fill: float) -> list[torch.Tensor]:
        torch.manual_seed(5)
        model = ActorCritic(OBS, ACTIONS, ModelConfig(policy=(8,), value=(8,)))
        learner = Learner(
            model, dataclasses.replace(config.ppo, minibatches=2, epochs=1), 0.99, torch.device("cpu"), False
        )
        rollout = Rollout.allocate(4, 6, OBS, ACTIONS, torch.device("cpu"))
        torch.manual_seed(6)
        rollout.obs.normal_()
        rollout.mask.fill_(True)
        rollout.actions.random_(0, ACTIONS)
        rollout.log_probs.fill_(-4.5)
        rollout.rewards.normal_()
        # Whatever happened on the excluded rows must not matter.
        rollout.obs[:, ~rows] = excluded_fill
        rollout.rewards[:, ~rows] = excluded_fill
        rollout.terminal[:, ~rows] = excluded_fill > 0
        stats = learner.learn(rollout, rows)
        assert stats.losses_finite and stats.gradient_steps == 2
        return [p.detach().clone() for p in model.parameters()]

    first, second = learn_with(0.0), learn_with(100.0)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_trajectories_never_mix_with_the_pool_active_and_opponent_rows_stay_out_of_the_batch() -> None:
    """With the opponent pool: known environments (tests/test_trainer.py's counting ones) driven partly by
    a frozen opponent; every row keeps its own trajectory and the learner's update sees only its rows."""
    from test_trainer import CountingEnvs

    torch.manual_seed(4)
    envs = CountingEnvs((4, 4))  # two groups of four agents; obs = [step, row]
    learner_model = ActorCritic(2, 3, ModelConfig(policy=(8,), value=(8,)))
    opponent = ActorCritic(2, 3, ModelConfig(policy=(8,), value=(8,)))
    with torch.no_grad():  # the learner always takes action 0, the opponent action 1 (both always allowed)
        for model, action in ((learner_model, 0), (opponent, 1)):
            model.policy[-1].weight.zero_()
            model.policy[-1].bias.fill_(0.0)
            model.policy[-1].bias[action] = 50.0
    # Rows 2-3 of each group are the "past" arena; its second car (row 3) is the opponent's.
    selects = {g: torch.tensor([False, True]) for g in (0, 1)}
    learner_rows = torch.tensor([True, True, True, False, True, True, True, False])
    fake = SimpleNamespace(
        group_opponents=lambda g: [OpponentBlock(slice(2, 4), selects[g], opponent)],
        learner_rows=learner_rows,
    )
    config = default_config(Path.cwd()).ppo
    learner = Learner(
        learner_model, dataclasses.replace(config, epochs=1, minibatches=1), 0.9, torch.device("cpu"), False
    )
    from nitro.trainer import Trainer

    trainer = Trainer(
        envs, learner_model, learner, 8 * 3, torch.device("cpu"), fp16=False, cuda_graphs=False, selfplay=fake
    )  # type: ignore[arg-type]
    for _ in range(2):
        stats = trainer.iteration()
        assert stats.learn.losses_finite
    r = trainer.rollout
    # Continuity: steps 3, 4, 5 of every row, in order, after the first iteration took 0, 1, 2.
    torch.testing.assert_close(r.obs[:, :, 0], torch.arange(3.0, 6.0)[:, None].expand(3, 8))
    torch.testing.assert_close(r.obs[:, :, 1], torch.arange(8.0).expand(3, 8))
    # Who drove whom: the opponent's rows took action 1, every other row action 0, in the rollout and in
    # what the environments received.
    expected_actions = torch.where(learner_rows, 0, 1).expand(3, 8)
    assert torch.equal(r.actions, expected_actions)
    received = torch.cat([torch.stack(group) for group in envs.received], dim=1)
    assert torch.equal(received[3:], expected_actions.int())
    # The learner's log-probability of action 1 on the opponent's rows is tiny: those rows never train.
    assert (r.log_probs[:, ~learner_rows] < -40).all() and (r.log_probs[:, learner_rows] > -1e-3).all()


def test_self_play_settings_are_checked() -> None:
    config = default_config(Path.cwd())
    lone = dataclasses.replace(
        config, env=dataclasses.replace(config.env, opponents=False), selfplay=SelfPlayConfig(enabled=True)
    )
    assert any(path == "selfplay.enabled" for path, _ in lone.problems())
    few = dataclasses.replace(
        config,
        env=dataclasses.replace(config.env, num_arenas=64, groups=4),
        selfplay=SelfPlayConfig(enabled=True),
    )
    # 64 arenas x 0.2 = 13 past arenas, but 4 groups x 4 past versions need 16 (every version in every group).
    assert [path for path, _ in few.problems()] == ["selfplay.past_share"]
    enough = dataclasses.replace(few, selfplay=SelfPlayConfig(enabled=True, past_share=0.25))
    assert enough.problems() == []
    assert dataclasses.replace(config, selfplay=SelfPlayConfig(enabled=True)).problems() == []


class _AsBefore(SelfPlay):
    """`record` and `metrics` as they were in an earlier version (games.csv opened and closed for every
    iteration's
    games, the pool listed for every row): what the kept-open file and the counted pool must match."""

    def record(self, episodes: list[Any], agent_steps: int) -> GameStats:
        if not self.enabled:
            return GameStats(0, 0, math.nan)
        rows: list[tuple[int, int]] = []
        evenness = []
        for record in episodes:
            if record.car != 0 or record.goal == 0:
                continue
            slot = self.slot_of_arena(record.env)
            if slot is None or (steps := self.active[slot]) is None:
                continue
            score = 1.0 if record.goal > 0 else 0.0
            evenness.append(abs(win_probability(self.learner, self.ratings[steps]) - 0.5))
            self.learner, self.ratings[steps] = update(
                self.learner, self.ratings[steps], score, min_sigma=BETA
            )
            self.games[steps] = self.games.get(steps, 0) + 1
            rows.append((steps, int(score)))
        if rows and self._log is not None:
            new = not self._log.exists()
            with self._log.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if new:
                    writer.writerow(_LOG_COLUMNS)
                for steps, scored in rows:
                    learner_mu, snapshot_mu = round(self.learner.mu, 4), round(self.ratings[steps].mu, 4)
                    writer.writerow([agent_steps, steps, scored, learner_mu, snapshot_mu])
        return GameStats(
            len(rows), sum(s for _, s in rows), statistics.mean(evenness) if evenness else math.nan
        )

    def metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {
            "rating_mu": self.learner.mu,
            "rating_sigma": self.learner.sigma,
            "pool_size": float(len(self.pool.snapshots())),
        }


def test_games_csv_kept_open_and_the_counted_pool_match_what_was_written_before(tmp_path: Path) -> None:
    """games.csv stays open between iterations, flushed after each one's games, and the pool is
    counted again only after a snapshot: the file holds the same bytes after every iteration, through a resume
    dropping later games and a close, and every row the same pool size, as opening and listing each time."""
    torch.manual_seed(0)
    config = _config(pool_size=3)
    runs = [
        kind(
            config,
            tmp_path / name / "pool",
            ActorCritic(OBS, ACTIONS, config.model),
            torch.device("cpu"),
            groups=2,
            cars_per_env=2,
            group_agents=[8, 8],
        )
        for name, kind in (("kept", SelfPlay), ("before", _AsBefore))
    ]
    logs = [tmp_path / name / "pool" / "games.csv" for name in ("kept", "before")]
    listed = 0
    snapshots = runs[0].pool.snapshots

    def counted() -> list[int]:
        nonlocal listed
        listed += 1
        return snapshots()

    runs[0].pool.snapshots = counted  # type: ignore[method-assign]
    rng = random.Random(4)
    recorded = 0
    for iteration in range(1, 16):
        steps = iteration * 1_000_000
        episodes = [
            SimpleNamespace(env=rng.randrange(8), car=rng.randrange(2), goal=rng.choice((-1, 0, 1)))
            for _ in range(rng.randrange(4, 16))
        ]
        for selfplay in runs:
            selfplay.maybe_snapshot(steps)
            if iteration % 3 == 2:
                selfplay.rotate(random.Random(iteration))
        games = [repr(selfplay.record(episodes, steps)) for selfplay in runs]  # type: ignore[arg-type]
        assert games[0] == games[1]
        recorded += int(games[0].split("games=")[1].split(",")[0])
        before = listed
        assert runs[0].metrics() == runs[1].metrics() == runs[0].metrics()
        assert listed - before <= 1, "the pool is listed at most once between snapshots"
        if iteration == 7:  # a resume from a save at 5M steps drops the games after it
            for selfplay in runs:
                selfplay.truncate_log(5_000_000)
        if iteration == 11:
            runs[0].close()  # the next games open the file again, without a second header
        assert logs[0].exists() == logs[1].exists()
        if logs[1].exists():
            assert logs[0].read_bytes() == logs[1].read_bytes(), f"iteration {iteration}"
    runs[0].close()
    assert recorded >= 10 and logs[0].read_text(encoding="utf-8").count("agent_steps") == 1


# --- a pool that does not match the save -----------------------------------------------------------


def _load_as_before(self: SelfPlay, state: dict[str, Any]) -> None:
    """SelfPlay.load_state_dict in an earlier version, word for word: the guard that the notes change
    nothing."""
    self.learner = Rating(*state["learner"])
    self.ratings = {int(s): Rating(*r) for s, r in state["ratings"].items()}
    self.games = {int(s): n for s, n in state["games"].items()}
    self.next_snapshot_steps = state["next_snapshot_steps"]
    present = set(self.pool.snapshots()) if self.enabled else set()
    active = [None if s < 0 or s not in present else s for s in state["active"]]
    self.active = [active[i] if i < len(active) else None for i in range(len(self.active))]
    self._apply_active()


class _LoadAsBefore(SelfPlay):
    load_state_dict = _load_as_before


M = 1_000_000


def _saved(rated: dict[int, float], active: list[int], next_snapshot: int = 81 * M) -> dict[str, Any]:
    return {
        "learner": [31.0, 2.5],
        "ratings": {str(s): [mu, 2.0] for s, mu in rated.items()},
        "games": {str(s): 10 + k for k, s in enumerate(rated)},
        "active": active,
        "next_snapshot_steps": next_snapshot,
    }


def _pool_of(tmp_path: Path, model: ActorCritic, steps: list[int]) -> None:
    pool = Pool(tmp_path / "pool", size=64, disk_bytes=10**9)
    for s in steps:
        pool.save(s, model.policy_state_dict())


def test_past_versions_the_save_rates_missing_from_the_pool_are_said_and_draws_stay_as_before(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    config = _config()
    model = ActorCritic(OBS, ACTIONS, config.model)
    _pool_of(tmp_path, model, [20 * M, 40 * M])
    state = _saved({20 * M: 30.0, 40 * M: 32.0, 60 * M: 29.0, 80 * M: 33.0}, [60 * M, 20 * M])
    now, before = (
        kind(config, tmp_path / "pool", model, torch.device("cpu"), 2, 2, [8, 8])
        for kind in (SelfPlay, _LoadAsBefore)
    )
    now.load_state_dict(state)
    before.load_state_dict(state)
    assert len(now.resume_notes) == 1 and not now.resume_info
    note = now.resume_notes[0]
    assert "rates 2 past versions (0.06B-0.08B steps) that are not in test_" in note and "/pool/" in note
    assert "nitro pack" in note and "copied back" in note
    assert now.ratings == before.ratings and now.games == before.games
    assert now.active == before.active == [None, 20 * M] and now.learner == before.learner
    assert now.state_dict() == before.state_dict(), "the next save holds the same"
    for k in range(21):
        now.rotate(rotation_rng(7, k))
        before.rotate(rotation_rng(7, k))
        assert now.active == before.active, f"rotation {k}"
        assert now.learner_rows is not None and before.learner_rows is not None
        assert torch.equal(now.learner_rows, before.learner_rows)
    assert {s for s in now.active if s is not None} <= {20 * M, 40 * M}, "only versions on disk play"
    # Copied back, the missing versions can be drawn again: their ratings were kept.
    _pool_of(tmp_path, model, [60 * M, 80 * M])
    drawn = set()
    for k in range(40):
        now.rotate(rotation_rng(7, k))
        drawn |= {s for s in now.active if s is not None}
    assert {60 * M, 80 * M} & drawn


def test_a_past_version_that_does_not_read_is_said_once_and_never_played(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot cut short (empty after a power cut, or damaged): on resume and at a rotation it is said
    once by name, left where it is, and its slot plays the learner until the next draw; the others play."""
    torch.manual_seed(0)
    config = _config()
    model = ActorCritic(OBS, ACTIONS, config.model)
    _pool_of(tmp_path, model, [20 * M, 40 * M, 60 * M])
    pool = Pool(tmp_path / "pool", size=64, disk_bytes=10**9)
    pool.path(40 * M).write_bytes(b"")  # a power cut before the data reached the disk
    pool.path(60 * M).write_bytes(pool.path(20 * M).read_bytes()[:50])  # cut short
    selfplay = _selfplay(tmp_path, ActorCritic(OBS, ACTIONS, config.model), config)
    with caplog.at_level(logging.WARNING, logger="nitro"):
        selfplay.load_state_dict(_saved({20 * M: 30.0, 40 * M: 30.0, 60 * M: 30.0}, [40 * M, 20 * M]))
    assert selfplay.active == [None, 20 * M], "the unreadable one's slot plays the learner"
    assert selfplay.learner_rows is not None and int((~selfplay.learner_rows).sum()) == 2, "slot 1 only"
    assert [r.getMessage() for r in caplog.records] == [
        f"Self-play: the past version at 40,000,000 agent steps ({pool.path(40 * M)}) cannot be read: it is "
        "empty (cut short while it was written, as by a crash, a power cut or a full disk). It was left "
        "where it is and is not played against; the run goes on with the others."
    ]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nitro"):
        for seed in range(20):
            selfplay.rotate(random.Random(seed))
            assert 40 * M not in selfplay.active
    assert len(caplog.records) == 1 and "60,000,000 agent steps" in caplog.records[0].getMessage()
    assert selfplay.unreadable == {40 * M, 60 * M}
    assert pool.path(40 * M).is_file() and pool.path(60 * M).is_file(), "never deleted"


def test_a_snapshot_is_on_the_disk_before_it_is_renamed_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As a run's saves: written to a temporary file, fsync'd, then renamed; a failed write leaves nothing."""
    model = ActorCritic(OBS, ACTIONS, _config().model)
    pool = Pool(tmp_path / "pool", size=64, disk_bytes=10**9)
    synced: list[int] = []
    fsync = os.fsync

    def recording(fd: int) -> None:
        assert not pool.path(M).exists(), "synced before the rename"
        synced.append(fd)
        fsync(fd)

    monkeypatch.setattr(os, "fsync", recording)
    pool.save(M, model.policy_state_dict())
    assert synced and pool.snapshots() == [M] and pool.load(M).keys() == model.policy_state_dict().keys()

    def full(*args: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", full)
    with pytest.raises(OSError, match="No space left"):
        pool.save(2 * M, model.policy_state_dict())
    assert sorted(p.name for p in pool.folder.iterdir()) == [pool.path(M).name], "no partial file left"


def test_versions_the_save_never_rated_and_an_empty_pool_are_said_and_a_crash_is_no_news(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    config = _config()
    model = ActorCritic(OBS, ACTIONS, config.model)

    def loaded(folder: str, pool: list[int], rated: list[int], enabled: bool = True) -> SelfPlay:
        _pool_of(tmp_path / folder, model, pool)
        chosen = config if enabled else dataclasses.replace(config, selfplay=SelfPlayConfig(enabled=False))
        selfplay = SelfPlay(chosen, tmp_path / folder / "pool", model, torch.device("cpu"), 2, 2, [8, 8])
        selfplay.load_state_dict(_saved(dict.fromkeys(rated, 30.0), [-1, -1]))
        return selfplay

    # A version in the pool from before the save that the save never rated: from another copy of the run.
    other = loaded("other", [20 * M, 40 * M, 50 * M], [20 * M, 40 * M])
    assert len(other.resume_notes) == 1 and not other.resume_info
    assert "1 version (0.05B steps) this save never rated" in other.resume_notes[0]
    assert "never picked" in other.resume_notes[0]
    # An empty pool: the bot plays itself until its next snapshot.
    empty = loaded("empty", [], [20 * M, 40 * M])
    assert len(empty.resume_notes) == 1 and "holds no past versions" in empty.resume_notes[0]
    assert "2 past versions (0.02B-0.04B steps)" in empty.resume_notes[0]
    assert "0.08B steps" in empty.resume_notes[0], "when its next snapshot joins"
    # A run that went on past its save and stopped without saving: it took 90M and pruned 60M then.
    crashed = loaded("crashed", [20 * M, 40 * M, 90 * M], [20 * M, 40 * M, 60 * M])
    assert crashed.resume_notes == [] and len(crashed.resume_info) == 2
    assert "pruned it then" in crashed.resume_info[0] and "taken after this save" in crashed.resume_info[1]
    # Self-play off: nothing is looked at.
    off = loaded("off", [20 * M], [20 * M, 40 * M, 50 * M], enabled=False)
    assert off.resume_notes == [] and off.resume_info == []
    # A pool that matches: nothing to say.
    same = loaded("same", [20 * M, 40 * M], [20 * M, 40 * M])
    assert same.resume_notes == [] and same.resume_info == []


RUN_SETTINGS = """
[run]
seed = 7
device = "cpu"
deterministic = true
[env]
num_arenas = 8
threads = 4
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 320
minibatches = 2
[selfplay]
enabled = true
past_share = 0.5
active_opponents = 2
[[state_setters]]
name = "kickoff"
"""


def _run_config(tmp_path: Path, **run: Any) -> NitroConfig:
    parsed = parse_toml(RUN_SETTINGS, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    return dataclasses.replace(
        parsed,
        paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES),
        run=dataclasses.replace(parsed.run, **run),
        selfplay=dataclasses.replace(parsed.selfplay, snapshot_steps=640),  # a version every 2 iterations
    )


@NEEDS_MESHES
def test_a_resumed_run_missing_past_versions_says_so_and_learns_bit_for_bit_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bot resumed without some of the past versions its save rates is told so, at the start and in
    every report, and trains exactly as before the notes (the same draws, the same weights)."""
    config = _run_config(tmp_path)
    file = tmp_path / "s.toml"
    file.write_text(RUN_SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "moved", file)
    first = TrainingRun(config, folder)
    for k in range(6):
        first.iteration()
        if k % 2 == 1:
            first.checkpoint(continue_after=k < 5)
    first.metrics.close()
    first.selfplay.close()
    snapshots = first.selfplay.pool.snapshots()
    assert len(snapshots) == 3 and set(snapshots) <= set(first.selfplay.ratings)
    lost = first.selfplay.pool.path(snapshots[0])
    shutil.copy(lost, tmp_path / "lost.pt")
    lost.unlink()  # moved without it
    shutil.copytree(folder.root, tmp_path / "runs" / "before")

    def resumed(run_folder: RunFolder) -> tuple[TrainingRun, list[list[int | None]]]:
        run = TrainingRun(config, run_folder, run_folder.load_latest())
        drawn = [list(run.selfplay.active)]
        for k in range(5):
            run.iteration()
            if k in (1, 3):
                run.checkpoint()
                drawn.append(list(run.selfplay.active))
        return run, drawn

    with caplog.at_level(logging.WARNING, logger="nitro"):
        now, now_drawn = resumed(folder)
    told = [r.message for r in caplog.records if "past version" in r.message]
    assert len(told) == 1 and f"rates 1 past version ({snapshots[0] / 1e9:.2f}B steps) that is not" in told[0]
    assert "moved/pool/" in told[0]
    assert told[0] in now.recent_warnings and told[0] in now.console_warnings
    now.write_report()
    report = now.folder.report.read_text(encoding="utf-8")
    assert "When resumed: Self-play: this save rates 1 past version" in report
    now.metrics.close()
    now.selfplay.close()

    monkeypatch.setattr(SelfPlay, "load_state_dict", _load_as_before)
    before, before_drawn = resumed(RunFolder(tmp_path / "runs" / "before"))
    before.metrics.close()
    before.selfplay.close()
    assert before.missing_notes == []
    assert now_drawn == before_drawn and any(s is not None for slots in now_drawn for s in slots)
    for name, weights in before.model.state_dict().items():
        assert torch.equal(weights, now.model.state_dict()[name]), f"{name} differs"
    assert now.selfplay.state_dict() == before.selfplay.state_dict()
    monkeypatch.undo()

    # run.missing_data = 'stop': training refuses to start and says what is missing; copied back, it starts.
    stopping = _run_config(tmp_path, missing_data="stop")
    refused = TrainingRun(stopping, folder, folder.load_latest())  # what only plays is not refused
    with pytest.raises(TrainingProblem, match=r"training does not start(.|\n)*rates 1 past version"):
        refused.run(threading.Event())
    assert refused.iterations == folder.load_latest()["iterations"], "refused before any iteration"
    refused.metrics.close()
    refused.selfplay.close()
    shutil.copy(tmp_path / "lost.pt", lost)
    again = TrainingRun(stopping, folder, folder.load_latest())
    assert again.missing_notes == [] and lost.stat().st_size > 0
    stop = threading.Event()
    stop.set()
    again.run(stop)  # starts, then stops at once


# Self-play read apart from the scripted starts (one car placed, the rest parked).
UNEVEN = {"shot", "save", "aerial_ball"}


def test_every_start_says_whether_it_places_every_car_alike() -> None:
    assert not hasattr(_StateSetter, "even_start"), "no default: a new setter must choose"
    for kind in get_args(StateSetter):
        assert isinstance(vars(kind).get("even_start"), bool), kind.__name__
    assert {kind().name for kind in get_args(StateSetter) if not kind.even_start} == UNEVEN


def test_even_games_against_past_versions_are_counted_apart_and_nothing_else_changes(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _config()
    runs = [
        SelfPlay(
            config,
            tmp_path / name,
            ActorCritic(OBS, ACTIONS, config.model),
            torch.device("cpu"),
            2,
            2,
            [8, 8],
        )
        for name in ("with", "without")
    ]
    for selfplay in runs:
        selfplay.maybe_snapshot(1_000_000)
        selfplay.maybe_snapshot(2_000_000)
        selfplay.rotate(random.Random(0))
        assert None not in selfplay.active  # arenas 4-5 slot 0, 6-7 slot 1; 0-3 the learner against itself
    even = [True, False, True]  # setter 3 is past the end: uneven

    def game(env: int, car: int, goal: int, setter: int) -> SimpleNamespace:
        return SimpleNamespace(env=env, car=car, goal=goal, setter=setter)

    episodes = [
        game(4, 0, 1, 0),  # rated, even, won
        game(4, 1, -1, 0),  # the opponent's car: not a game
        game(5, 0, -1, 0),  # rated, even, lost
        game(6, 0, 1, 1),  # rated, uneven, won
        game(7, 0, 1, 2),  # rated, even, won
        game(7, 0, 0, 2),  # no goal
        game(1, 0, 1, 0),  # the learner against itself: not rated
        game(6, 0, -1, 3),  # rated, a setter past the list, lost
        game(5, 0, 1, 1),  # rated, uneven, won
    ]
    counted = runs[0].record(episodes, 3_000_000, even)  # type: ignore[arg-type]
    plain = runs[1].record(episodes, 3_000_000)  # type: ignore[arg-type]
    assert (counted.games, counted.wins, counted.even_games, counted.even_wins) == (6, 4, 3, 2)
    assert (plain.even_games, plain.even_wins) == (0, 0)
    assert (plain.games, plain.wins, plain.matchmaking) == (counted.games, counted.wins, counted.matchmaking)
    assert runs[0].learner == runs[1].learner and runs[0].ratings == runs[1].ratings
    assert runs[0].games == runs[1].games and runs[0].state_dict() == runs[1].state_dict()
    for selfplay in runs:
        selfplay.close()
    logs = [(tmp_path / name / "games.csv").read_bytes() for name in ("with", "without")]
    assert logs[0] == logs[1] and logs[0].count(b"\n") == 7


def test_start_edge_is_blues_share_from_uneven_starts_with_the_learner_on_both_sides() -> None:
    slots = {4: 0, 5: 0, 6: 1, 7: 1}
    active = [1_000_000, None]  # slot 1 empty: its arenas are the learner against itself
    even = [True, False, False]

    def game(env: int, car: int, goal: int, setter: int) -> SimpleNamespace:
        return SimpleNamespace(env=env, car=car, goal=goal, setter=setter)

    episodes = [
        game(0, 0, 1, 1),  # counted: Blue scored
        game(0, 1, -1, 1),  # the same episode's second car
        game(1, 0, -1, 2),  # counted: Blue conceded
        game(2, 0, 1, 0),  # an even start
        game(3, 0, 0, 1),  # no goal
        game(4, 0, 1, 1),  # against a past version
        game(6, 0, 1, 2),  # counted: an empty slot's arena
        game(7, 0, 1, 5),  # counted: a setter past the end is uneven
    ]
    assert start_edge(episodes, slots.get, active, even) == 0.75
    assert start_edge(episodes[3:6], slots.get, active, even) == ""


def test_the_columns_come_only_with_a_scripted_start() -> None:
    # Starts that place every car alike: metrics.csv without the columns.
    assert even_starts(load_config(GCO_STARTER).state_setters) is None
    text = (
        "[selfplay]\nenabled = true\n"
        '[[state_setters]]\nname = "kickoff"\n[[state_setters]]\nname = "shot"\n'
        '[[state_setters]]\nname = "random"\n[[state_setters]]\nname = "save"\n'
    )
    mixed = parse_toml(text, NitroConfig, file=ROOT / "mixed.toml", base_dir=ROOT)
    even = even_starts(mixed.state_setters)
    assert even is not None and any(even) and not all(even)
    assert even == [setter.name not in UNEVEN for setter in mixed.state_setters]
    assert len(EVEN_START_COLUMNS) == 3 and not set(EVEN_START_COLUMNS) & set(SELFPLAY_COLUMNS)
