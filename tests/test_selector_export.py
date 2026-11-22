"""A multi-model bot as one RLBot bot: `nitro export` writes the selector and its models into one bot file,
and the bot's decision (`nitro_core::policy::Policy::decide_seeing`) sees, allows, holds and hands over
exactly as training's `SelectorEnvs` does, with the same numbers as PyTorch. The models here are the v1 test
saves (a plain and an attention network)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import _core, selector
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig, SelectorConfig
from nitro.envs import GroupTensors, core_settings
from nitro.export import ExportError, export_run, write_selector
from nitro.model import ActorCritic, build_networks
from nitro.runs import RunFolder
from nitro.selector import (
    Saved,
    SelectorEnvs,
    extra_size,
    from_state,
    hold_steps,
    load_specialists,
    networks,
    to_state,
)

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parent / "fixtures" / "saves"
NAMES = ("v1-padded", "v1-attention")
HOLD = 0.2  # seconds: 6 decisions at tick skip 4


class Games:
    """Fake games with fresh random observations and masks every step; a step in `ends` ends car 0's
    episode."""

    def __init__(self, agents: int, obs_size: int, ends: tuple[int, ...] = ()) -> None:
        self.groups = (GroupTensors.allocate(agents, obs_size, 90, pin_memory=False),)
        self.generator = torch.Generator().manual_seed(5)
        self.s, self.ends = 0, ends
        self._fill()

    @property
    def num_groups(self) -> int:
        return 1

    def _fill(self) -> None:
        tensors = self.groups[0]
        tensors.obs.copy_(torch.randn(tensors.obs.shape, generator=self.generator))
        tensors.mask.copy_(torch.rand(tensors.mask.shape, generator=self.generator) > 0.3)
        tensors.mask[:, 0] = True

    def step(self, group: int) -> None:
        self.s += 1
        tensors = self.groups[group]
        tensors.reward.zero_()
        tensors.truncated.fill_(False)
        tensors.terminal.fill_(False)
        tensors.terminal[0] = self.s in self.ends
        self._fill()

    def wait(self, group: int) -> None:
        pass


def jitter(weights: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Every weight moved off its trained or initial value, so the order of each in the file is tested."""
    generator = torch.Generator().manual_seed(seed)
    return {k: v + torch.randn(v.shape, generator=generator) * 0.3 for k, v in weights.items()}


def selector_run(tmp_path: Path, specialists: int = 2) -> tuple[RunFolder, ActorCritic, list[ActorCritic]]:
    """A selector run's folder over the v1 test saves, with its save as training writes one; with its
    selector's network and its specialists' networks as training builds them."""
    root = tmp_path / "picker"
    shutil.copytree(FIXTURES / "v1-padded", root)
    folder = RunFolder(root)
    names = ", ".join(f'"{name}"' for name in NAMES)
    with folder.settings.open("a", encoding="utf-8") as file:
        file.write(f"[selector]\nspecialists = [{names}]\nmin_hold_seconds = {HOLD}\n")
    (save,) = folder.complete_checkpoints()
    state = torch.load(save / "state.pt", map_location="cpu", weights_only=True)
    state["settings"]["selector.specialists"] = NAMES
    config = folder.load_settings(state)
    found = load_specialists(config, FIXTURES)
    kept = to_state(found)
    for seed, entry in enumerate(kept):
        entry["weights"] = jitter(entry["weights"], seed)
    obs_size = int(state["learner"]["model"]["policy.0.weight"].shape[1])
    torch.manual_seed(1)
    chooser = build_networks(config, obs_size + extra_size(len(NAMES)), len(NAMES))
    chooser.load_state_dict(jitter(chooser.state_dict(), 9))
    state["learner"]["model"] = chooser.state_dict()
    state["specialists"] = kept[:specialists]
    state["all_settings"]["selector.specialists"] = list(NAMES)
    state["all_settings"]["selector.min_hold_seconds"] = HOLD
    torch.save(state, save / "state.pt")
    models = networks(from_state(kept), config, obs_size, 90, CPU)
    return folder, chooser.eval(), models


def masked_argmax(scores: torch.Tensor, mask: torch.Tensor) -> int:
    return int(scores.masked_fill(~mask, float("-inf")).argmax())


def trace_matches(bot: _core.Policy, chooser: ActorCritic, models: list[ActorCritic]) -> None:
    """16 decisions of 4 cars by the bot file `bot` beside training's `SelectorEnvs` over `models` with
    the selector `chooser`: the same numbers, masks, actions and selector state, through a car's
    episode end."""
    obs_size = models[0].obs_size
    agents = 4
    games = Games(agents, obs_size, ends=(5,))
    settings = SelectorConfig(specialists=NAMES, min_hold_seconds=HOLD)
    tick_skip = int(bot.header()["tick_skip"])
    envs = SelectorEnvs(games, models, settings, tick_skip, CPU, False, 0, pin_memory=False)  # type: ignore[arg-type]
    assert envs.hold_steps == hold_steps(settings, tick_skip)
    states: list[tuple[int | None, int]] = [(None, 0)] * agents
    switches = 0
    for step in range(16):
        inner, outer = games.groups[0], envs.groups[0]
        with torch.no_grad():
            expected = [model(inner.obs) for model in models]
            scores = chooser.logits(outer.obs)
        choices = []
        for car in range(agents):
            mask = inner.mask[car]
            wanted = [masked_argmax(logits[car], mask) for logits, _ in expected]
            choice = masked_argmax(scores[car], outer.mask[car])
            picks: list[tuple[list[float], list[bool]]] = []

            def pick(
                logits: list[float],
                allowed: list[bool],
                picks: list[Any] = picks,
                wanted: list[int] = wanted,
                choice: int = choice,
            ) -> int:
                picks.append((logits, allowed))
                # The reference's choice, so rounding never flips a near tie between the two.
                return wanted[len(picks) - 1] if len(picks) <= len(models) else choice

            active, held = states[car]
            allowed_actions = mask.tolist()
            action, active, held = bot.decide(
                inner.obs[car].tolist(), mask.to(torch.uint8).tolist(), active, held, pick
            )
            assert len(picks) == len(models) + 1, "each specialist, then the selector"
            for (logits, allowed), (reference, _) in zip(picks, expected, strict=False):
                torch.testing.assert_close(torch.tensor(logits), reference[car], rtol=1e-4, atol=1e-4)
                assert allowed == allowed_actions
            seen_scores, allowed = picks[-1]
            # The selector's scores carry what it sees: who drives, how long, whether free, each value.
            torch.testing.assert_close(torch.tensor(seen_scores), scores[car], rtol=1e-4, atol=1e-4)
            assert allowed == outer.mask[car].tolist(), f"step {step}, car {car}"
            assert action == wanted[choice]
            states[car] = (active, held)
            choices.append(choice)
        envs.groups[0].actions.copy_(torch.tensor(choices, dtype=torch.int32))
        envs.step(0)
        switches += int(envs.cars[0].switched.sum())
        envs.wait(0)
        for car in range(agents):
            if bool(games.groups[0].terminal[car]):
                states[car] = (None, 0)  # where the bot restarts too: outside live play
            active = int(envs.cars[0].active[car])
            assert states[car] == (None if active < 0 else active, int(envs.cars[0].held[car])), step
    assert switches > 0, "the trace should hand a car over at least once"


def weights_in(chooser: ActorCritic, models: list[ActorCritic]) -> int:
    """The numbers a selector bot file holds: the selector's network, then each specialist's policy and
    value, a shared trunk written with each."""
    count = sum(p.numel() for p in [*chooser.shared.parameters(), *chooser.policy.parameters()])
    for m in models:
        trunk = sum(p.numel() for p in m.shared.parameters())
        count += 2 * trunk + sum(p.numel() for p in [*m.policy.parameters(), *m.value.parameters()])
    return count


def test_a_selector_run_exports_as_one_bot_that_decides_as_training_does(tmp_path: Path) -> None:
    folder, chooser, models = selector_run(tmp_path)
    exported = export_run(folder, tmp_path / "bot", None, build=False)
    readme = (exported.folder / "README.md").read_text(encoding="utf-8")
    assert "multi-model bot" in readme and "`v1_padded`" in readme and "0.20 s" in readme
    bot = _core.Policy(str(exported.folder / "policy.nwt"))
    header = bot.header()
    assert header["format"] == 10 and header["specialists"] == ["v1_padded", "v1_attention"]
    obs_size = models[0].obs_size
    assert (bot.obs_size, bot.num_actions) == (obs_size, 90)
    with pytest.raises(ValueError, match="use decide"):
        bot.logits([0.0] * obs_size)

    trace_matches(bot, chooser, models)


def test_a_selector_save_missing_a_specialist_is_refused(tmp_path: Path) -> None:
    folder, _, _ = selector_run(tmp_path, specialists=1)
    with pytest.raises(ExportError, match=r"keeps 1 model\(s\)"):
        export_run(folder, tmp_path / "bot", None, build=False)


def test_the_bot_plays_with_the_value_scale_its_save_trained_with(tmp_path: Path) -> None:
    """A save trained with standardise_values keeps its value scale; the bot made from it uses that scale
    even after the setting is switched off in the settings file (the save's networks learned on it)."""
    folder, _, _ = selector_run(tmp_path)
    (save,) = folder.complete_checkpoints()
    state = torch.load(save / "state.pt", map_location="cpu", weights_only=True)
    stats = {"mean": torch.tensor([1.5, -2.0]), "std": torch.tensor([3.0, 0.5]), "started": True}
    state["selector"] = {"games": {"value_stats": stats}, "learning": {}}
    torch.save(state, save / "state.pt")
    config = folder.load_settings(state)
    assert not config.selector.standardise_values
    bundle = selector.saved(state, config, 90, CPU)
    assert (bundle.value_shift, bundle.value_scale) == ([1.5, -2.0], [3.0, 0.5])
    assert bundle.settings.standardise_values
    exported = export_run(folder, tmp_path / "bot", None, build=False)
    raw = (exported.folder / "policy.nwt").read_bytes()
    header = json.loads(raw[16 : 16 + int.from_bytes(raw[12:16], "little")])
    kept = header["selector"]["specialists"]
    assert [(m["value_shift"], m["value_scale"]) for m in kept] == [(1.5, 3.0), (-2.0, 0.5)]


def test_the_bot_file_holds_the_selector_then_each_specialists_policy_and_value(tmp_path: Path) -> None:
    folder, chooser, models = selector_run(tmp_path)
    exported = export_run(folder, tmp_path / "bot", None, build=False)
    raw = (exported.folder / "policy.nwt").read_bytes()
    length = int.from_bytes(raw[12:16], "little")
    header: dict[str, Any] = json.loads(raw[16 : 16 + length])
    kept = header["selector"]["specialists"]
    assert kept[1]["value"][0]["kind"] == "entity_encoder", "an attention specialist"
    assert kept[0]["value"][-1] == {"kind": "linear", "inputs": 32, "outputs": 1}
    assert header["selector"]["hold_steps"] == 6 and header["selector"]["held_cap_seconds"] == 5.0
    assert (len(raw) - 16 - length) // 4 == weights_in(chooser, models)


@pytest.mark.parametrize("lean", ["", "lean_attention = true\n"], ids=["", "lean"])
def test_shared_trunks_and_a_talking_specialist_are_written_and_read_in_order(
    tmp_path: Path, lean: str
) -> None:
    """The v1 test saves have neither: a selector and a specialist with a shared trunk, and a specialist
    whose cars talk first with the own part passed beside it; with model.lean_attention the same file and the
    same decisions."""
    padded = "[env]\nobs_team_size = 3\n[model]\npolicy = [16]\nvalue = [16]\n"
    trunk = padded + "shared = [24]\n"
    talk = padded + "encoder = 'attention_communication'\nattention_width = 16\nattention_heads = 2\n"
    talk += "attention_skip = true\n" + lean
    configs = [
        parse_toml(text, NitroConfig, file=Path("s.toml"), base_dir=Path.cwd()) for text in (trunk, talk)
    ]
    obs_size = _core.observation_layout(core_settings(configs[0]))[0]
    torch.manual_seed(2)
    models = []
    for seed, config in enumerate(configs):
        model = build_networks(config, obs_size, 90)
        model.load_state_dict(jitter(model.state_dict(), seed))
        models.append(model.eval())
    chooser = build_networks(configs[0], obs_size + extra_size(2), 2)
    chooser.load_state_dict(jitter(chooser.state_dict(), 7))
    chooser.eval()
    path = tmp_path / "policy.nwt"
    hold = hold_steps(SelectorConfig(specialists=NAMES, min_hold_seconds=HOLD), configs[0].env.tick_skip)
    bundle = Saved.of(chooser, [("trunk", models[0]), ("talk", models[1])])
    write_selector(path, bundle, hold, configs[0], "t", 0)
    raw = path.read_bytes()
    assert (len(raw) - 16 - int.from_bytes(raw[12:16], "little")) // 4 == weights_in(chooser, models)
    trace_matches(_core.Policy(str(path)), chooser, models)


@pytest.mark.parametrize("lean", ["", "lean_attention = true\n"], ids=["", "lean"])
def test_a_transformer_model_is_written_and_read_in_order(tmp_path: Path, lean: str) -> None:
    """A transformer model: the bot decides as training does, with model.lean_attention too."""
    padded = "[env]\nobs_team_size = 3\n[model]\npolicy = [16]\nvalue = [16]\n"
    transformer = padded + "encoder = 'transformer'\ntransformer_width = 16\ntransformer_heads = 2\n"
    transformer += "transformer_ff = 32\n" + lean
    configs = [
        parse_toml(text, NitroConfig, file=Path("s.toml"), base_dir=Path.cwd())
        for text in (padded, transformer)
    ]
    obs_size = _core.observation_layout(core_settings(configs[0]))[0]
    torch.manual_seed(3)
    models = []
    for seed, config in enumerate(configs):
        model = build_networks(config, obs_size, 90)
        model.load_state_dict(jitter(model.state_dict(), seed))
        models.append(model.eval())
    chooser = build_networks(configs[0], obs_size + extra_size(2), 2)
    chooser.load_state_dict(jitter(chooser.state_dict(), 9))
    chooser.eval()
    path = tmp_path / "policy.nwt"
    hold = hold_steps(SelectorConfig(specialists=NAMES, min_hold_seconds=HOLD), configs[0].env.tick_skip)
    specialists = [("plain", models[0]), ("transformer", models[1])]
    write_selector(path, Saved.of(chooser, specialists), hold, configs[0], "t", 0)
    raw = path.read_bytes()
    assert int.from_bytes(raw[8:12], "little") == 10, "a multi-model bot's format"
    trace_matches(_core.Policy(str(path)), chooser, models)
