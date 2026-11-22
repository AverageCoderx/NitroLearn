"""Multi-model bots: a selector over models that each keep their own view of the game, self-play, an attention
selector, minimum shares, models trained together with a pull back to their start, goal heads, standardised
values, `nitro eval` and `nitro watch`, and the exported bot deciding as training does through real games."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import _core
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig, SelectorConfig
from nitro.envs import EnvPool, GroupTensors, probe_config
from nitro.model import ActorCritic
from nitro.runs import FIXED_SETTINGS, RunFolder, saved_settings
from nitro.selector import (
    GoalHeads,
    SelectorEnvs,
    ShareBonus,
    ValueStats,
    extra_size,
    extras_at,
    from_model,
    game_size,
    load_specialists,
    network,
    problems_of,
    view_of,
    views,
)
from nitro.selector_learning import DROPPED, UNKNOWN, GoalSamples, goal_labels, model_rollout

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parent / "fixtures" / "saves"
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")


def settings(text: str = "") -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("s.toml"), base_dir=Path.cwd())


TWO = '[selector]\nspecialists = ["a", "b"]\n'


def test_the_settings_take_what_a_multi_model_bot_can_do_now_and_check_the_shares() -> None:
    for text in (
        TWO + "[selfplay]\nenabled = true\n",
        TWO + "[model]\nencoder = 'attention'\n[env]\nobs_team_size = 2\n",
        TWO + "min_share = [20, 30]\ntrain_models = true\ngoal_heads = true\nstandardise_values = true\n",
        TWO + "train_models = true\nmodel_lr = [1e-5, 3e-5]\ntrain_schedule = 'alternate'\n",
    ):
        settings(text)
    defaults = settings(TWO).selector
    assert (defaults.min_share, defaults.train_models, defaults.goal_heads, defaults.standardise_values) == (
        (),
        False,
        False,
        False,
    ), "every new part is off by default"
    for text, where in (
        (TWO + "min_share = [60, 40]\n", "add up to 100"),
        (TWO + "min_share = [10]\n", "one share per model"),
        (TWO + "min_share = [10, 100]\n", "below 100"),
        (TWO + "model_lr = [1e-5, 1e-5, 1e-5]\n", "one per model"),
        (TWO + "model_lr = [0.0]\n", "above 0"),
        (TWO + '[start_from]\nrun = "x"\n', "starting from another bot"),
        (TWO + "[pretrain]\nenabled = true\n", "copying replays first"),
        (TWO + "train_models = true\ntrain_schedule = 'alternate'\n[model]\nshared = [32]\n", "shared trunk"),
    ):
        with pytest.raises(ConfigError, match=where):
            settings(text)
    assert "selector.goal_heads" in FIXED_SETTINGS
    assert saved_settings({"settings": {}})["selector.goal_heads"] is False


def test_each_model_keeps_its_own_view_and_what_cannot_drive_is_named_in_plain_words() -> None:
    run = settings("[env]\nobservation = 'native'\n")
    torch.manual_seed(0)
    own = from_model("own", ActorCritic(10, 90, run.model), run)
    assert view_of(own, run) is None, "the run's own observation needs no view"
    v2 = from_model("v2", ActorCritic(10, 90, run.model), settings("[env]\nobservation = 'nitro_v2'\n"))
    alone = from_model("alone", ActorCritic(10, 90, run.model), settings("[env]\nopponents = false\n"))
    assert view_of(v2, run) == {
        "observation": "nitro_v2",
        "team_size": 1,
        "opponents": True,
        "obs_team_size": 0,
        "action_masks": "standard",
    }
    assert view_of(alone, run)["opponents"] is False  # type: ignore[index]
    assert views([own, v2, alone, v2], run) == [view_of(v2, run), view_of(alone, run)], "each view once"
    assert problems_of(v2, run) == [] and problems_of(alone, run) == []
    bigger = from_model("2v2", ActorCritic(10, 90, run.model), settings("[env]\nteam_size = 2\n"))
    (problem,) = problems_of(bigger, run)
    assert "2v2" in problem and "1v1" in problem
    slow = from_model("slow", ActorCritic(10, 90, run.model), settings("[env]\ntick_skip = 8\n"))
    assert "env.tick_skip must match" in problems_of(slow, run)[0]
    lone_run = settings("[env]\nopponents = false\n")
    assert "alone" in problems_of(own, lone_run)[0], "a model of 1v1 games cannot see a lone car's"


def test_the_attention_selector_reads_its_own_numbers_as_part_of_its_car() -> None:
    config = settings(TWO + "[env]\nobs_team_size = 2\n[model]\nencoder = 'attention'\n")
    plain = settings(TWO)
    assert extras_at(plain, 60) == 60, "after the game's numbers"
    game = game_size(config)
    assert extras_at(config, game) < game, "after the own part, before the other cars"
    torch.manual_seed(2)
    chooser = network(config, game, 2)
    assert chooser.obs_size == game + extra_size(2) and chooser.layout is not None
    assert chooser.layout[0] == extras_at(config, game) + extra_size(2)
    logits, value = chooser(torch.randn(3, chooser.obs_size))
    assert logits.shape == (3, 2) and value.shape == (3,)


def test_the_share_bonus_rises_while_a_model_is_below_its_share_and_fades_once_it_is_met() -> None:
    bonus = ShareBonus([20.0, 0.0], 0.1, CPU)
    for _ in range(5):
        bonus.update([5, 95], typical=1.0)
    rising = float(bonus.bonus[0])
    assert rising > 0.2 and float(bonus.bonus[1]) == 0.0, "only a model with a share to reach is paid"
    for _ in range(60):
        bonus.update([40, 60], typical=1.0)
    assert float(bonus.bonus[0]) == 0.0, "met: the bonus has faded back to 0"
    for _ in range(500):
        bonus.update([0, 100], typical=1.0)
    assert float(bonus.bonus[0]) == pytest.approx(ShareBonus.CAP), "never beyond its cap"
    again = ShareBonus([20.0, 0.0], 0.1, CPU)
    again.load_state_dict(bonus.state_dict())
    assert torch.equal(again.bonus, bonus.bonus)


def test_a_share_bonus_rate_outside_0_to_1_is_refused() -> None:
    for text in ("share_bonus_rate = -0.1", "share_bonus_rate = 1.5"):
        with pytest.raises(ConfigError, match="share_bonus_rate"):
            settings(f'[selector]\nspecialists = ["a", "b"]\nmin_share = [0, 40]\n{text}\n')


def share_toy() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).parents[1] / "benchmarks" / "micro" / "share_bonus_toy.py"
    spec = importlib.util.spec_from_file_location("share_bonus_toy", path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules["share_bonus_toy"] = script
    spec.loader.exec_module(script)
    return script


def test_the_share_bonus_brings_a_toy_selector_to_its_share_soon_and_holds_it_there() -> None:
    """benchmarks/micro/share_bonus_toy.py's toy (a bonus of about 2 typical step rewards needed): the first
    controller (an integral alone at rate 0.05) stayed near 30% for 70 collections here, as slow as on the
    PC check; the default now gets there within 40 and stays within 5 points."""
    toy = share_toy()
    shares, _ = toy.run(collections=70)
    mean5 = [sum(shares[i - 4 : i + 1]) / 5 for i in range(4, len(shares))]  # mean5[j]: collections j..j+4
    assert min(shares[5:15]) < 0.3, f"the selector alone falls toward its 20% first: {shares[:15]}"
    reached = next((j + 4 for j, m in enumerate(mean5) if j >= 10 and m >= 0.95 * toy.TARGET), None)
    assert reached is not None and reached <= 40, f"95% of the target by collection 40: {mean5}"
    assert all(abs(m - toy.TARGET) <= 0.05 for m in mean5[reached - 4 :]), f"held without swinging: {mean5}"


def test_value_inputs_are_standardised_per_model_over_this_runs_states() -> None:
    stats = ValueStats(2, CPU)
    values = torch.stack([torch.randn(1000) * 3 + 5, torch.randn(1000) * 0.1 - 1], dim=1)
    out = stats.apply(values)
    assert out.mean(0).abs().max() < 1e-4 and (out.std(0) - 1).abs().max() < 1e-2
    stats.add(values + 10)
    stats.update(rate=0.1)
    assert float(stats.mean[0]) == pytest.approx(5 + 1.0, abs=0.3), "a slow move toward new states"


class Always(torch.nn.Module):
    """A model that always plays `action` and values every state at `value`."""

    def __init__(self, action: int, value: float) -> None:
        super().__init__()
        self.action, self.value = action, value

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.full((obs.shape[0], 3), -30.0)
        logits[:, self.action] = 30.0
        return logits, self.normalised_value(obs)

    def normalised_value(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0],), self.value)


class Steps:
    """Fake games: 4 cars observe [s, row], reward 1 a step; step 5 ends every episode with car 0's team
    scoring (cars alternate Blue, Orange)."""

    def __init__(self) -> None:
        self.groups = (GroupTensors.allocate(4, 2, 3, pin_memory=False),)
        self.groups[0].obs[:, 1] = torch.arange(4, dtype=torch.float32)
        self.groups[0].mask.fill_(True)
        self.s = 0

    def step(self, group: int) -> None:
        tensors = self.groups[group]
        tensors.reward.fill_(1.0)
        tensors.terminal.fill_(self.s == 5)
        tensors.truncated.fill_(False)
        self.s += 1
        tensors.obs[:, 0] = float(self.s)

    def wait(self, group: int) -> None:
        pass

    def goals_into(self, group: int, out: torch.Tensor) -> None:
        out.copy_(
            torch.tensor([1, -1, 1, -1], dtype=torch.int8)
            if self.s == 6
            else torch.zeros(4, dtype=torch.int8)
        )


def test_the_selector_records_each_step_as_its_driver_saw_it_and_pays_the_share_bonus() -> None:
    games = Steps()
    config = SelectorConfig(specialists=("a", "b"), min_hold_seconds=0.0, min_share=(0.0, 30.0))
    heads = GoalHeads([2, 2])
    chooser = SelectorEnvs(
        games, [Always(0, 0.5), Always(2, -0.5)], config, 8, CPU, False, 0, pin_memory=False,  # type: ignore[arg-type]
        goal_heads=heads, record=True,
    )  # fmt: skip
    assert chooser.groups[0].obs.shape[1] == 2 + extra_size(2, goal_heads=True)
    chooser.learner_rows = torch.tensor([True, False, True, False])
    chooser.begin(8)
    assert chooser.bonus is not None
    chooser.bonus.bonus[1] = 0.25
    for step in range(8):
        chooser.groups[0].actions.copy_(torch.tensor([step % 2, 1, 0, 0], dtype=torch.int32))
        chooser.step(0)
        chooser.wait(0)
        paid = chooser.groups[0].reward.tolist()
        assert paid[1] == 1.25 and paid[2] == 1.0, "the bonus goes with the model below its share"
    recorded = chooser.recorded
    assert recorded is not None and recorded.steps == [8]
    assert recorded.driver[:, 1].tolist() == [1] * 8 and recorded.driver[:, 0].tolist() == [0, 1] * 4
    assert recorded.actions[:, 0].tolist() == [0, 2] * 4, "the driver's own action"
    assert recorded.rewards.eq(1.0).all(), "the game's reward, without the bonus"
    assert recorded.values[0, 0].tolist() == [0.5, -0.5]
    assert recorded.goals[5].tolist() == [1, -1, 1, -1] and recorded.goals[4].abs().sum() == 0
    stats = chooser.take_stats()
    assert stats.driven == [12, 4], "the learner's own cars only (cars 0 and 2)"


def test_goal_labels_say_who_scored_within_the_horizon_and_wait_for_what_is_not_seen_yet() -> None:
    goals = torch.zeros(6, 2, dtype=torch.int8)
    terminal = torch.zeros(6, 2, dtype=torch.bool)
    truncated = torch.zeros(6, 2, dtype=torch.bool)
    goals[3] = torch.tensor([1, -1], dtype=torch.int8)
    terminal[3] = True
    truncated[5, 1] = True
    found = goal_labels(goals, terminal, truncated, horizon=3)
    # Step 0's horizon (steps 0-2) ends just before the goal: no goal within it.
    assert found.labels[:, 0].tolist() == [0, 1, 1, 1, UNKNOWN, UNKNOWN]
    assert found.labels[:, 1].tolist() == [0, 2, 2, 2, DROPPED, DROPPED]
    assert found.first_end.tolist() == [3, 3]
    wide = goal_labels(goals, terminal, truncated, horizon=1)
    assert wide.labels[0, 0] == 0, "the whole horizon played without a goal"


def test_goal_states_left_unknown_are_settled_by_the_next_collection() -> None:
    from nitro.selector import Recorded

    def collection(goal_at: int | None, episode: int) -> Recorded:
        steps, agents = 4, 2
        z = torch.zeros
        goals = z(steps, agents, dtype=torch.int8)
        terminal = z(steps, agents, dtype=torch.bool)
        if goal_at is not None:
            goals[goal_at] = torch.tensor([1, -1], dtype=torch.int8)
            terminal[goal_at] = True
        return Recorded(
            obs=torch.randn(steps, agents, 3), mask=z(steps, agents, 3, dtype=torch.bool),
            actions=z(steps, agents, dtype=torch.int64), log_probs=z(steps, agents),
            driver=z(steps, agents, dtype=torch.int64), values=z(steps, agents, 2), rewards=z(steps, agents),
            terminal=terminal, truncated=z(steps, agents, dtype=torch.bool),
            truncated_values=z(steps, agents, 2), goals=goals, episode=torch.full((steps, agents), episode),
            episode_step=z(steps, agents, dtype=torch.int32), bootstrap_values=z(agents, 2),
        )  # fmt: skip

    samples = GoalSamples(horizon=6, per_collection=1000)
    assert samples.take(collection(None, 7), None) is None, "nothing settled: the horizon is longer"
    assert samples.pending is not None and samples.pending.row.numel() == 8
    taken = samples.take(collection(1, 7), None)
    assert taken is not None
    settled, labels = taken
    # The goal 2 steps into the second collection settles all 8 carried states and its own first 4.
    assert settled.row.numel() == 12 and sorted(labels.tolist()) == [1] * 6 + [2] * 6
    assert samples.pending is not None and samples.pending.row.numel() == 4, "its last 4 wait in turn"
    full = GoalSamples(horizon=6, per_collection=8, waiting=1)
    full.take(collection(None, 7), None)
    taken = full.take(collection(None, 7), None)
    assert taken is not None and taken[0].row.numel() == 6 and set(taken[1].tolist()) == {0}
    assert full.pending is not None and full.pending.row.numel() == 2, "no new states while full: all or none"
    rebuilt = GoalSamples(horizon=6, per_collection=1000)
    rebuilt.take(collection(None, 7), None)
    rebuilt.restart()  # games rebuilt at a save, episode numbers starting again
    after = rebuilt.take(collection(1, 7), None)
    assert after is not None and after[0].row.numel() == 4, "only the new games' states: the old dropped"


def test_a_models_stretch_ends_where_it_hands_the_car_over_valued_by_its_own_critic() -> None:
    from nitro.selector import Recorded

    steps, agents = 5, 1
    z = torch.zeros
    driver = torch.tensor([[0], [0], [1], [1], [0]])
    values = torch.arange(steps * 2, dtype=torch.float32).reshape(steps, 1, 2)
    recorded = Recorded(
        obs=torch.randn(steps, agents, 4), mask=z(steps, agents, 3, dtype=torch.bool),
        actions=z(steps, agents, dtype=torch.int64), log_probs=z(steps, agents), driver=driver, values=values,
        rewards=z(steps, agents), terminal=z(steps, agents, dtype=torch.bool),
        truncated=z(steps, agents, dtype=torch.bool), truncated_values=z(steps, agents, 2),
        goals=z(steps, agents, dtype=torch.int8), episode=z(steps, agents, dtype=torch.int64),
        episode_step=z(steps, agents, dtype=torch.int32), bootstrap_values=torch.tensor([[100.0, 200.0]]),
    )  # fmt: skip
    rollout, mine = model_rollout(recorded, 0, 3)
    assert mine[:, 0].tolist() == [True, True, False, False, True]
    assert rollout.truncated[:, 0].tolist() == [False, True, False, False, False], "handed over after step 1"
    assert float(rollout.truncated_values[1, 0]) == float(values[2, 0, 0]), "its own value of the next state"
    assert float(rollout.bootstrap_values[0]) == 100.0 and rollout.obs.shape[-1] == 3
    other, _ = model_rollout(recorded, 1, 4)
    assert other.truncated[:, 0].tolist() == [False, False, False, True, False]
    assert float(other.truncated_values[3, 0]) == float(values[4, 0, 1])


def multi_run(
    tmp: Path, extra: str = "", model: str = "", selfplay: bool = True, ppo: str = ""
) -> tuple[NitroConfig, RunFolder]:
    """A multi-model run over three v1 test saves: one sees the run's own observation, one sees nitro_v2,
    one is an attention network (its own room)."""
    runs = tmp / "runs"
    for name in ("v1-native", "v1-nitro_v2", "v1-attention"):
        shutil.copytree(FIXTURES / name, runs / name)
    text = (FIXTURES / "v1-native" / "settings.toml").read_text(encoding="utf-8")
    text = text.replace(
        'collision_meshes = "collision_meshes"', f"collision_meshes = {json.dumps(str(MESHES.resolve()))}"
    )
    text = text.replace("[run]\n", '[run]\ndevice = "cpu"\n').replace("[model]\n", "[model]\n" + model)
    text = text.replace("[ppo]\n", "[ppo]\n" + ppo)
    if selfplay:
        text = text.replace("[selfplay]\nenabled = false", "[selfplay]\nenabled = true")
    text += '[selector]\nspecialists = ["v1-native", "v1-nitro_v2", "v1-attention"]\nmin_hold_seconds = 0.2\n'
    text += extra
    (tmp / "mm.toml").write_text(text, encoding="utf-8")
    from nitro.config import load_config

    return load_config(tmp / "mm.toml"), RunFolder.create(runs, "mm", tmp / "mm.toml")


@NEEDS_MESHES
def test_models_with_their_own_views_and_an_attention_selector_train_with_self_play_and_resume(
    tmp_path: Path,
) -> None:
    from nitro.training import TrainingRun

    extra = "min_share = [10, 30, 0]\ntrain_models = true\nmodel_lr = [1e-3]\ngoal_heads = true\n"
    extra += "goal_head_seconds = 1.0\nstandardise_values = true\n"
    model = "encoder = 'attention'\n"
    config, folder = multi_run(tmp_path, extra, model)
    run = TrainingRun(config, folder)
    assert run.selfplay.enabled and run.selector is not None
    # The attention selector's observation has a room (1): every model here sees the game its own way.
    assert run.selector.views == [0, 1, 2], "each model through its own view"
    start = {k: v.clone() for k, v in run.models.models[1].state_dict().items()}  # type: ignore[union-attr]
    rows = [run.iteration() for _ in range(3)]
    last = rows[-1]
    shares = [last[f"selector_share_{label}"] for label in ("v1_native", "v1_nitro_v2", "v1_attention")]
    assert sum(shares) == pytest.approx(1.0)
    assert {f"selector_bonus_{label}" for label in ("v1_native", "v1_nitro_v2")} <= set(last)
    assert last["model_kl_to_start_v1_nitro_v2"] > 0, "the models learn"
    assert any(row.get("selector_goal_accuracy", "") != "" for row in rows), "the goal heads learn"
    moved = run.models.models[1].state_dict()  # type: ignore[union-attr]
    assert any(not torch.equal(start[k], moved[k]) for k in start)
    run.checkpoint(continue_after=False)
    run.metrics.close()
    for name in ("v1-native", "v1-nitro_v2", "v1-attention"):
        shutil.rmtree(folder.root.parent / name)
    state = folder.load_latest()
    assert state is not None
    kept = state["specialists"][1]
    assert any(not torch.equal(kept["weights"][k], kept["start"][k]) for k in kept["start"]), (
        "trained and start kept"
    )
    resumed = TrainingRun(folder.load_settings(state), folder, state)
    assert resumed.selector is not None and resumed.selector.bonus is not None
    assert torch.equal(resumed.selector.bonus.bonus, run.selector.bonus.bonus)  # type: ignore[union-attr]
    resumed_weights = resumed.models.models[1].state_dict()  # type: ignore[union-attr]
    assert all(torch.equal(resumed_weights[k], moved[k]) for k in moved), "the trained models resume"
    resumed.iteration()
    resumed.metrics.close()


@NEEDS_MESHES
def test_a_run_stopped_and_resumed_at_a_save_trains_on_as_one_never_stopped(tmp_path: Path) -> None:
    """Saves carry everything the next iteration needs: the goal heads and the value scale are in place before
    the resumed games' first decision, and each model's running returns end with the save's games."""
    from nitro.training import TrainingRun

    extra = "min_share = [10, 30, 0]\ntrain_models = true\nmodel_lr = [1e-3]\ngoal_heads = true\n"
    extra += "goal_head_seconds = 1.0\nstandardise_values = true\n"

    def first(tmp: Path, stop: bool) -> dict[str, Any]:
        torch.manual_seed(0)
        config, folder = multi_run(tmp, extra, selfplay=False)
        run = TrainingRun(config, folder)
        run.iteration()
        if stop:
            run.checkpoint(continue_after=False)
            run.metrics.close()
            state = folder.load_latest()
            assert state is not None
            run = TrainingRun(folder.load_settings(state), folder, state)
        else:
            run.checkpoint()
        run.iteration()
        run.metrics.close()
        return run.state()

    def same(a: Any, b: Any, where: str) -> None:
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), where
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), where
            for k in a:
                same(a[k], b[k], f"{where}.{k}")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), where
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                same(x, y, f"{where}[{i}]")
        else:
            assert a == b, where

    straight, resumed = first(tmp_path / "a", False), first(tmp_path / "b", True)
    for key in ("learner", "specialists", "selector", "rng_cpu"):
        same(straight[key], resumed[key], key)


def test_a_model_deciding_with_another_action_delay_is_named() -> None:
    run = settings()
    torch.manual_seed(0)
    late = from_model("late", ActorCritic(10, 90, run.model), settings("[env]\naction_delay = 3\n"))
    (problem,) = problems_of(late, run)
    assert "env.action_delay must match" in problem


def test_a_greedy_player_plays_each_models_likeliest_action_as_the_exported_bot_does() -> None:
    """`nitro eval` with pick 0 and the RLBot bot agree: every model's candidate is its likeliest allowed
    action (of equal ones the last, as nitro_bot's argmax), not a sampled one."""
    from nitro.ppo import likeliest

    class Fixed(torch.nn.Module):
        def __init__(self, row: list[float]) -> None:
            super().__init__()
            self.row = torch.tensor(row)

        def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.row.expand(obs.shape[0], 3).clone(), torch.zeros(obs.shape[0])

    games = Steps()
    models = [Fixed([0.0, 2.0, 1.0]), Fixed([1.0, 1.0, 0.5])]
    picker = SelectorConfig(specialists=("a", "b"))
    greedy = SelectorEnvs(games, models, picker, 8, CPU, False, 0, pin_memory=False, greedy=True)  # type: ignore[arg-type]
    obs, mask = games.groups[0].obs, games.groups[0].mask
    for index, model in enumerate(models):
        wanted = likeliest(model(obs)[0], mask)
        assert torch.equal(greedy.cars[0].candidates[index], wanted)
    assert greedy.cars[0].candidates[0].eq(1).all() and greedy.cars[0].candidates[1].eq(1).all(), (
        "equal logits: the last, as the bot"
    )


def test_the_models_wait_for_the_selectors_warm_up_then_ease_in() -> None:
    held = settings(TWO + "train_models = true\nmodels_start_after = 100\nmodels_ramp_steps = 50\n").selector
    assert [held.models_rate_at(s) for s in (0, 99, 100, 125, 150, 10**9)] == [0, 0, 0, 0.5, 1, 1]
    assert settings(TWO + "train_models = true\n").selector.models_rate_at(0) == 1, "no warm-up by default"
    assert settings(TWO + "models_start_after = 100\n").selector.models_rate_at(500) == 0, "train_models off"


@NEEDS_MESHES
def test_the_selector_learns_alone_first_then_the_models_are_eased_in(tmp_path: Path) -> None:
    from nitro.training import TrainingRun

    extra = "train_models = true\nmodel_lr = [1e-3]\nmodels_start_after = 1\nmodels_ramp_steps = 1000000000\n"
    extra += "goal_heads = true\n"
    config, folder = multi_run(tmp_path, extra, selfplay=False)
    run = TrainingRun(config, folder)
    assert run.models is not None and run.model_learning is not None
    start = {k: v.clone() for k, v in run.models.models[0].state_dict().items()}
    held = run.iteration()
    assert held["models_lr_share"] == 0 and held["model_policy_change_v1_native"] == ""
    assert held["policy_change"] > 0, "the selector learns in its warm-up"
    now = run.models.models[0].state_dict()
    assert all(torch.equal(start[k], now[k]) for k in start), "the models are held as they came"
    eased = run.iteration()
    share = eased["models_lr_share"]
    assert 0 < share < 0.01 and eased["model_policy_change_v1_native"] != ""
    assert run.model_learning.learners[0].learning_rates["policy"] == pytest.approx(1e-3 * share)
    assert run.model_learning.goal_samples.pending is not None
    run.checkpoint()
    assert run.model_learning.goal_samples.pending is None, "a save's new games drop the waiting states"
    run.metrics.close()


@NEEDS_MESHES
def test_alternate_turns_hold_the_selector_then_the_models(tmp_path: Path) -> None:
    from nitro.training import TrainingRun

    extra = "train_models = true\ntrain_schedule = 'alternate'\nalternate_iterations = 1\n"
    config, folder = multi_run(tmp_path, extra, selfplay=False)
    run = TrainingRun(config, folder)
    first, second = run.iteration(), run.iteration()
    assert first["model_policy_change_v1_native"] != "" and first["policy_change"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert second["model_policy_change_v1_native"] == "" and second["policy_change"] > 0
    run.metrics.close()


def selector_trace(
    bot: _core.Policy, config: NitroConfig, state: dict[str, Any], decisions: int = 12
) -> None:
    """The exported bot beside the save's selector and models in real games (with each model's view): the
    same logits from every model on its own view, the same selector scores and choices, the same actions."""
    from nitro.selector import Player, distinct_views, saved

    bundle = saved(state, config, _core.num_actions(), CPU)
    pool_views = distinct_views([bundle])
    small = probe_config(config, 3)
    envs = EnvPool(
        dataclasses.replace(small, env=dataclasses.replace(small.env, num_arenas=2)),
        pin_memory=False,
        views=pool_views,
    )
    player = Player(bundle, config, envs, pool_views, CPU, False, 0)
    games = player.games
    assert bot.views_count == len(pool_views)
    agents = envs.groups[0].obs.shape[0]
    states: list[tuple[int | None, int]] = [(None, 0)] * agents
    for _ in range(decisions):
        inner, outer = envs.groups[0], games.groups[0]
        own = [games._model_obs[0][k] for k in range(len(bundle.models))]
        masks = [games._model_mask[0][k] for k in range(len(bundle.models))]
        with torch.no_grad():
            expected = [model(obs)[0] for model, obs in zip(bundle.models, own, strict=True)]
            scores = bundle.selector.logits(outer.obs)
        view_rows = [(games._view_obs[0][v], games._view_mask[0][v]) for v in range(len(pool_views))]
        choices = []
        for car in range(agents):
            wanted = [
                int(e[car].masked_fill(~m[car], float("-inf")).argmax())
                for e, m in zip(expected, masks, strict=True)
            ]
            choice = int(scores[car].masked_fill(~outer.mask[car], float("-inf")).argmax())
            picks: list[tuple[list[float], list[bool]]] = []

            def pick(
                logits: list[float],
                allowed: list[bool],
                picks: Any = picks,
                wanted: Any = wanted,
                choice: int = choice,
            ) -> int:
                picks.append((logits, allowed))
                return wanted[len(picks) - 1] if len(picks) <= len(wanted) else choice

            active, held = states[car]
            action, active, held = bot.decide(
                inner.obs[car].tolist(), inner.mask[car].to(torch.uint8).tolist(), active, held, pick,
                [(o[car].tolist(), m[car].to(torch.uint8).tolist()) for o, m in view_rows],
            )  # fmt: skip
            for (logits, allowed), reference, mask in zip(picks, expected, masks, strict=False):
                torch.testing.assert_close(torch.tensor(logits), reference[car], rtol=1e-4, atol=1e-4)
                assert allowed == mask[car].tolist()
            torch.testing.assert_close(torch.tensor(picks[-1][0]), scores[car], rtol=1e-4, atol=1e-4)
            assert action == wanted[choice]
            states[car] = (active, held)
            choices.append(choice)
        # The games go on with training's sampled actions; the bot took each model's most likely one.
        inner.actions.copy_(games.act(0, torch.tensor(choices)).to(torch.int32))
        envs.step(0)
        envs.wait(0)
        player.after_step(0)
        for car in range(agents):
            if bool(inner.terminal[car] | inner.truncated[car]):
                states[car] = (None, 0)
            active = int(games.cars[0].active[car])
            assert states[car] == (None if active < 0 else active, int(games.cars[0].held[car]))


@NEEDS_MESHES
@pytest.mark.parametrize("selector_model", ["", "encoder = 'attention'\n"], ids=["plain", "attention"])
def test_a_multi_model_run_exports_as_one_bot_that_decides_as_training_does(
    tmp_path: Path, selector_model: str
) -> None:
    from nitro.export import export_run
    from nitro.training import TrainingRun

    extra = "train_models = true\nmodel_lr = [1e-3]\ngoal_heads = true\ngoal_head_seconds = 1.0\n"
    extra += "standardise_values = true\n"
    config, folder = multi_run(tmp_path, extra, selector_model, selfplay=False)
    run = TrainingRun(config, folder)
    run.iteration()
    run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    exported = export_run(folder, tmp_path / "bot", None, build=False)
    bot = _core.Policy(str(exported.folder / "policy.nwt"))
    header = bot.header()
    assert header["format"] == 10 and header["specialists"] == ["v1_native", "v1_nitro_v2", "v1_attention"]
    readme = (exported.folder / "README.md").read_text(encoding="utf-8")
    assert "multi-model bot" in readme
    state = folder.load_latest()
    assert state is not None
    selector_trace(bot, folder.load_settings(state), state)


@NEEDS_MESHES
def test_nitro_eval_and_nitro_watch_play_a_multi_model_bot(tmp_path: Path) -> None:
    from nitro.eval import load_side, play
    from nitro.training import TrainingRun
    from nitro.watch import record

    config, folder = multi_run(tmp_path, selfplay=False)
    run = TrainingRun(config, folder)
    run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    multi = load_side(folder, None)
    plain = load_side(RunFolder(folder.root.parent / "v1-native"), None)
    for blue, orange in ((multi, plain), (plain, multi), (multi, multi)):
        result = play(blue, orange, 4, 1, max_seconds=5)
        assert result.points == 4
    recording = record(multi.config, multi.state, 2.0, 1)
    assert recording.models == ["v1_native", "v1_nitro_v2", "v1_attention"]
    data = recording.data("t")
    assert data["models"] == recording.models
    for decision in recording.decisions:
        for figures in decision[1:]:
            assert len(figures) == 5 and 0 <= figures[4] < 3, "each car's figures name the model driving"


@NEEDS_MESHES
def test_load_specialists_takes_models_of_other_observations_now(tmp_path: Path) -> None:
    config, _ = multi_run(tmp_path, selfplay=False)
    found = load_specialists(config, tmp_path / "runs")
    assert [view_of(s, config) is None for s in found] == [True, False, False]
