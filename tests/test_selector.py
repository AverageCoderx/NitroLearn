"""The learned selector over frozen models (nitro.selector): what it sees, how a choice holds and hands over,
what a switch costs, episode ends and truncations, its saves, its settings and refusals; a selector learning
CartPole through the unchanged trainer; and, with the collision meshes, a tiny real multi-model run that saves
and resumes without its models' runs."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from nitro.cartpole import CartPoleEnvs
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig, SelectorConfig
from nitro.envs import GroupTensors
from nitro.health import learning_problems
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.runs import FIXED_SETTINGS, RunFolder, saved_settings
from nitro.selector import (
    SelectorEnvs,
    SelectorError,
    extra_size,
    from_model,
    from_state,
    hold_steps,
    labels,
    load_specialists,
    networks,
    not_yet,
    to_state,
    view_of,
)
from nitro.trainer import Trainer

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parent / "fixtures" / "saves"
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")


class Always(torch.nn.Module):
    """A specialist that always plays `action` and values every state at `value`."""

    def __init__(self, action: int, actions: int, value: float) -> None:
        super().__init__()
        self.action, self.actions, self.value = action, actions, value

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.full((obs.shape[0], self.actions), -30.0)
        logits[:, self.action] = 30.0
        return logits, self.normalised_value(obs)

    def normalised_value(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0],), self.value)


class Steps:
    """Fake games: cars observe [s, row], 3 actions, reward 1 a step; a step in `ends` ends every episode
    (terminal), one in `cuts` truncates it with final observation [1000 + s, row]. Keeps the actions."""

    def __init__(self, agents: int = 3, ends: tuple[int, ...] = (), cuts: tuple[int, ...] = ()) -> None:
        self.groups = (GroupTensors.allocate(agents, 2, 3, pin_memory=False),)
        self.groups[0].obs[:, 1] = torch.arange(agents, dtype=torch.float32)
        self.groups[0].mask.fill_(True)
        self.s, self.ends, self.cuts = 0, ends, cuts
        self.received: list[list[int]] = []

    @property
    def num_groups(self) -> int:
        return 1

    def step(self, group: int) -> None:
        tensors = self.groups[group]
        self.received.append(tensors.actions.tolist())
        tensors.reward.fill_(1.0)
        tensors.terminal.fill_(self.s in self.ends)
        tensors.truncated.fill_(self.s in self.cuts)
        tensors.final_obs[:, 0] = 1000.0 + self.s
        tensors.final_obs[:, 1] = tensors.obs[:, 1]
        self.s += 1
        tensors.obs[:, 0] = float(self.s)

    def wait(self, group: int) -> None:
        pass


def selector_over(games: Steps, hold: float = 0.2, cost: float = 0.0) -> SelectorEnvs:
    models = [Always(0, 3, 0.25), Always(2, 3, -0.5)]
    settings = SelectorConfig(specialists=("a", "b"), min_hold_seconds=hold, switch_cost=cost)
    return SelectorEnvs(games, models, settings, 8, CPU, False, 0, pin_memory=False)  # type: ignore[arg-type]


def act(chooser: SelectorEnvs, choice: list[int]) -> None:
    chooser.groups[0].actions.copy_(torch.tensor(choice, dtype=torch.int32))
    chooser.step(0)
    chooser.wait(0)


def test_the_selector_sees_the_game_who_drives_how_long_and_each_specialists_value() -> None:
    chooser = selector_over(Steps())
    tensors = chooser.groups[0]
    assert tensors.obs.shape == (3, 2 + extra_size(2)) and tensors.mask.shape == (3, 2)
    # [s, row | one-hot active (none yet) | time held | may choose | values]
    assert tensors.obs[1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, -0.5]
    assert tensors.mask.all(), "a new episode: every specialist may take the car"
    act(chooser, [1, 1, 0])
    assert tensors.obs[0, 2:6].tolist() == pytest.approx([0.0, 1.0, (8 / 120) / 5, 0.0])
    assert tensors.obs[2, 2:4].tolist() == [1.0, 0.0]


def test_a_choice_holds_then_the_selector_may_choose_again_and_the_chosen_specialist_drives() -> None:
    games = Steps()
    chooser = selector_over(games)  # 0.2 s at tick skip 8: 3 steps
    assert chooser.hold_steps == 3
    tensors = chooser.groups[0]
    act(chooser, [1, 0, 1])
    assert games.received[-1] == [2, 0, 2], "each car plays its chosen specialist's action"
    for _ in range(2):
        assert tensors.mask.tolist() == [[False, True], [True, False], [False, True]]
        act(chooser, [1, 0, 1])
    assert tensors.mask.all(), "after 3 steps the choice is free again"
    assert tensors.obs[:, 5].tolist() == [1.0, 1.0, 1.0]
    act(chooser, [0, 0, 1])
    assert games.received[-1] == [0, 0, 2]
    assert tensors.mask[0].tolist() == [True, False], "a switch starts a new hold"
    stats = chooser.take_stats()
    assert stats.driven == [5, 7] and stats.switches == 1
    assert chooser.take_stats().driven == [0, 0], "counted since the last call"


def test_holding_rounds_up_and_zero_lets_the_selector_choose_every_step() -> None:
    assert hold_steps(SelectorConfig(min_hold_seconds=0.5), 8) == 8  # 7.5 steps, rounded up
    assert hold_steps(SelectorConfig(min_hold_seconds=0.0), 8) == 1
    assert hold_steps(SelectorConfig(min_hold_seconds=0.2), 8) == 3  # exactly 3 despite float error


def test_a_switch_costs_its_reward_on_the_step_it_was_made_and_a_first_choice_costs_nothing() -> None:
    chooser = selector_over(Steps(), hold=0.0, cost=0.25)
    tensors = chooser.groups[0]
    act(chooser, [0, 1, 0])
    assert tensors.reward.tolist() == [1.0, 1.0, 1.0]
    act(chooser, [1, 1, 0])
    assert tensors.reward.tolist() == [0.75, 1.0, 1.0]
    act(chooser, [1, 1, 0])
    assert tensors.reward.tolist() == [1.0, 1.0, 1.0]


def test_an_episode_end_frees_the_car_and_a_truncation_keeps_who_drove_in_its_final_observation() -> None:
    chooser = selector_over(Steps(ends=(1,), cuts=(3,)))
    tensors = chooser.groups[0]
    act(chooser, [1, 0, 1])
    act(chooser, [1, 0, 1])  # step 1 ends the episode
    assert tensors.terminal.all() and tensors.mask.all()
    assert tensors.obs[0, 2:6].tolist() == [0.0, 0.0, 0.0, 1.0], "nobody drives the new episode yet"
    act(chooser, [0, 1, 1])
    act(chooser, [0, 1, 1])  # step 3 is cut short after 2 steps of the new choice
    assert tensors.truncated.all()
    held = (2 * 8 / 120) / 5
    assert tensors.final_obs[1].tolist() == pytest.approx([1003.0, 1.0, 0.0, 1.0, held, 0.0, 0.25, -0.5])
    assert tensors.mask.all(), "the next episode starts free"


def test_new_games_at_a_save_start_every_car_afresh() -> None:
    chooser = selector_over(Steps())
    act(chooser, [1, 1, 1])
    assert not chooser.groups[0].mask.all()
    chooser.release()
    chooser.attach(Steps(), 5)  # type: ignore[arg-type]
    assert chooser.groups[0].mask.all() and chooser.groups[0].obs[0, 2:4].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="same groups"):
        chooser.attach(Steps(agents=4), 5)  # type: ignore[arg-type]


def test_a_selector_learns_cartpole_from_a_push_left_and_a_push_right_bot() -> None:
    torch.manual_seed(0)
    games = CartPoleEnvs(num_envs=16, num_groups=2, seed=1, pin_memory=False)
    models = [Always(0, 2, 0.0), Always(1, 2, 0.0)]
    settings = SelectorConfig(specialists=("left", "right"), min_hold_seconds=0.0)
    chooser = SelectorEnvs(games, models, settings, 8, CPU, False, 0, pin_memory=False)  # type: ignore[arg-type]
    model = ActorCritic(chooser.groups[0].obs.shape[1], 2, ModelConfig(policy=(32,), value=(32,)))
    ppo = PPOConfig(epochs=4, minibatches=4, policy_lr=3e-3, value_lr=3e-3, entropy=0.0)
    trainer = Trainer(chooser, model, Learner(model, ppo, 0.98, CPU, False), 512, CPU, False, False)
    reports = [trainer.iteration() for _ in range(2)]
    first = games.take_episode_lengths()
    reports += [trainer.iteration() for _ in range(26)]
    games.take_episode_lengths()
    reports += [trainer.iteration() for _ in range(2)]
    last = games.take_episode_lengths() or [1000]  # no episode ended at all: balanced throughout
    assert learning_problems([r.learn for r in reports]) == []
    assert sum(last) / len(last) > 3 * sum(first) / len(first), (first, last)


def network(config: NitroConfig, obs_size: int = 10) -> ActorCritic:
    torch.manual_seed(3)
    return ActorCritic(obs_size, 90, config.model)


def settings(text: str = "") -> NitroConfig:
    return parse_toml(text, NitroConfig, file=Path("s.toml"), base_dir=Path.cwd())


def test_a_specialist_kept_in_a_save_rebuilds_the_same_network() -> None:
    config = settings("[model]\npolicy = [16]\nvalue = [8, 8]\nactivation = 'gelu'\n")
    model = network(config).eval()
    kept = to_state([from_model("touch", model, config)])
    buffer = io.BytesIO()
    torch.save(kept, buffer)
    buffer.seek(0)
    loaded = from_state(torch.load(buffer, weights_only=True))
    (rebuilt,) = networks(loaded, settings(), 10, 90, CPU)  # the run's own [model] does not matter
    obs = torch.randn(5, 10)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(model(obs), rebuilt(obs), strict=True))
    assert not any(p.requires_grad for p in rebuilt.parameters()), "a specialist is frozen"
    with pytest.raises(SelectorError, match="does not fit"):
        networks(loaded, settings(), 12, 90, CPU)


def test_a_specialist_kept_in_a_save_is_refused_when_the_game_changed_since() -> None:
    config = settings()
    (kept,) = from_state(to_state([from_model("touch", network(config), config)]))
    assert kept.game["env.tick_skip"] == 4
    edited = settings("[env]\ntick_skip = 12\naction_delay = 1\n")  # a resumed run's file, edited since
    with pytest.raises(SelectorError, match=r"decides every 4 ticks and this bot every 12"):
        networks([kept], edited, 10, 90, CPU)


def fixture_config(name: str = "v1-native") -> NitroConfig:
    folder = RunFolder(FIXTURES / name)
    state = folder.load_latest()
    assert state is not None
    return folder.load_settings(state)


def test_specialists_come_from_runs_by_name_and_must_play_the_same_game() -> None:
    config = fixture_config()
    two = dataclasses.replace(config, selector=SelectorConfig(specialists=("v1-native", "v1-native")))
    found = load_specialists(two, FIXTURES)
    assert [s.name for s in found] == ["v1_native", "v1_native"] and labels(found) == [
        "v1_native",
        "v1_native_2",
    ]
    assert found[0].label == "v1-native at 1,280 agent steps"
    state = RunFolder(FIXTURES / "v1-native").load_latest()
    assert state is not None
    obs_size = state["learner"]["model"]["policy.0.weight"].shape[1]
    built = networks(found, config, obs_size, 90, CPU)
    original = ActorCritic(obs_size, 90, config.model)
    original.load_state_dict(state["learner"]["model"])
    obs = torch.randn(3, obs_size)
    with torch.no_grad():
        assert torch.equal(built[1](obs)[0], original(obs)[0])
    # Another observation is not refused: that model gets a view of its own.
    other = dataclasses.replace(config, selector=SelectorConfig(specialists=("v1-native", "v1-nitro_v2")))
    native, v2 = load_specialists(other, FIXTURES)
    assert view_of(native, config) is None and view_of(v2, config) is not None
    missing = dataclasses.replace(config, selector=SelectorConfig(specialists=("v1-native", "nowhere")))
    with pytest.raises(SelectorError, match="no run or save called 'nowhere'"):
        load_specialists(missing, FIXTURES)


def test_the_settings_refuse_what_a_selector_cannot_do_yet() -> None:
    assert settings().selector.specialists == (), "off by default"
    two = '[selector]\nspecialists = ["a", "b"]\n'
    assert settings(two).selector.min_hold_seconds == 0.5
    for text, where in (
        ('[selector]\nspecialists = ["a"]\n', "at least 2"),
        ("[selector]\nspecialists = [" + ", ".join(f'"{i}"' for i in range(9)) + "]\n", "at most 8"),
        (two + '[start_from]\nrun = "x"\n', "starting from another bot"),
        (two + "[exploiters]\nenabled = true\n", "exploiters"),
        (two + "[env]\nteam_size = 2\nteam_sizes = [1, 2]\n", "several game sizes"),
    ):
        with pytest.raises(ConfigError, match=where):
            settings(text)
    assert "selector.specialists" in FIXED_SETTINGS
    assert saved_settings({"settings": {}})["selector.specialists"] == ()


def selector_run(tmp_path: Path) -> RunFolder:
    """A copy of a v1 save made to look like a selector run's (its settings and its save say so)."""
    root = tmp_path / "picker"
    shutil.copytree(FIXTURES / "v1-native", root)
    folder = RunFolder(root)
    with folder.settings.open("a", encoding="utf-8") as file:
        file.write('[selector]\nspecialists = ["a", "b"]\n')
    (save,) = folder.complete_checkpoints()
    state = torch.load(save / "state.pt", map_location="cpu", weights_only=True)
    state["settings"]["selector.specialists"] = ("a", "b")
    state["specialists"] = []
    torch.save(state, save / "state.pt")
    return folder


def test_a_selector_runs_save_is_refused_where_it_cannot_play_yet(tmp_path: Path) -> None:
    from nitro.eval import load_side
    from nitro.export import ExportError, export_run
    from nitro.surgery import SurgeryError, widen_run

    folder = selector_run(tmp_path)
    assert not_yet(folder.load_settings(), "widened") is not None and not_yet(settings(), "widened") is None
    # nitro eval and nitro watch play a multi-model bot now (tests/test_multimodel.py).
    assert load_side(folder, None).config.selector.specialists == ("a", "b")
    # Exporting works (tests/test_selector_export.py); this save keeps no specialists.
    with pytest.raises(ExportError, match=r"keeps 0 model\(s\)"):
        export_run(folder, tmp_path / "bot", None, build=False)
    with pytest.raises(SurgeryError, match="multi-model bot"):
        widen_run(tmp_path, "picker", 2)
    from nitro.transfer import TransferError, find_source

    with pytest.raises(TransferError, match="multi-model bot"):
        find_source(dataclasses.replace(settings().start_from, run="picker"), tmp_path)
    assert json.loads(folder.info.read_text())["settings_dir"] == "."


def test_a_selector_run_started_from_another_run_on_the_command_line_is_refused(tmp_path: Path) -> None:
    from nitro.training import TrainingRun

    settings_file = tmp_path / "picker.toml"
    settings_file.write_text('[selector]\nspecialists = ["a", "b"]\n', encoding="utf-8")
    config = settings(settings_file.read_text())
    start = dataclasses.replace(config.start_from, run="old")  # as nitro train --from sets it
    given = dataclasses.replace(config, start_from=start)
    folder = RunFolder.create(tmp_path / "runs", "picker", settings_file, {"run": "old", "at": None})
    with pytest.raises(SelectorError, match="cannot start from another bot"):
        TrainingRun(given, folder)


@NEEDS_MESHES
def test_a_tiny_selector_run_saves_its_specialists_and_resumes_without_their_runs(tmp_path: Path) -> None:
    from nitro.training import TrainingRun

    runs = tmp_path / "runs"
    for name in ("one", "two"):
        shutil.copytree(FIXTURES / "v1-native", runs / name)
    text = (FIXTURES / "v1-native" / "settings.toml").read_text(encoding="utf-8")
    text = text.replace(
        'collision_meshes = "collision_meshes"', f"collision_meshes = {json.dumps(str(MESHES.resolve()))}"
    )
    text += '[selector]\nspecialists = ["one", "two"]\nmin_hold_seconds = 0.2\nswitch_cost = 0.01\n'
    settings_file = tmp_path / "picker.toml"
    settings_file.write_text(text.replace("[run]\n", '[run]\ndevice = "cpu"\n'), encoding="utf-8")
    from nitro.config import load_config

    config = load_config(settings_file)
    folder = RunFolder.create(runs, "picker", settings_file)
    training = TrainingRun(config, folder)
    rows = [training.iteration() for _ in range(2)]
    assert 0.0 <= rows[-1]["selector_share_one"] <= 1.0
    assert rows[-1]["selector_share_one"] + rows[-1]["selector_share_two"] == pytest.approx(1.0)
    assert rows[-1]["selector_switches_per_minute"] >= 0.0
    training.checkpoint()
    training.iteration()
    training.checkpoint(continue_after=False)
    training.metrics.close()
    shutil.rmtree(runs / "one")
    shutil.rmtree(runs / "two")
    state = folder.load_latest()
    assert state is not None and [s["name"] for s in state["specialists"]] == ["one", "two"]
    resumed = TrainingRun(folder.load_settings(state), folder, state)
    resumed.iteration()
    resumed.metrics.close()
    # And it exports as one bot (the numbers are checked in tests/test_selector_export.py).
    from nitro import _core
    from nitro.export import export_run

    exported = export_run(folder, tmp_path / "bot", None, build=False)
    bot = _core.Policy(str(exported.folder / "policy.nwt"))
    assert bot.header()["specialists"] == ["one", "two"]
    obs = torch.randn(bot.obs_size).tolist()
    action, active, held = bot.decide(obs, [1] * 90, None, 0, lambda logits, allowed: allowed.index(True))
    assert (active, held) == (0, 1) and 0 <= action < 90
