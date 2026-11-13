"""Transfer learning: a new run starts from another run's trained networks instead of from
scratch (`[start_from]` in the settings, or `nitro train --from`).

`parts = "all"` takes the policy, the value network and the value and reward scaling (the same rewards,
something else changed: the game size through a padded observation, the start states, learning settings).
`parts = "policy"` takes the policy and any shared trunk and starts the value network afresh (new
rewards). `parts = "matching"`
takes the policy and shared trunk whole, every value tensor whose name and shape fit, and the scaling: the
same bot with another value side (a lighter critic, fewer heads). Adam's running averages, the self-play
pool and the step count always start afresh: it is a new run.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nitro.config.schema import NitroConfig, StartFromConfig
from nitro.learner import Learner
from nitro.ppo import discount_from_half_life
from nitro.runs import (
    RunFolder,
    all_settings,
    encoder_settings,
    fixed_settings,
    reads_any_cars,
    source_paths,
)

# Fixed settings the two runs must share for the networks to fit and mean the same thing.
_SAME = (
    "model.shared",
    "model.policy",
    "model.value",
    "model.layer_norm",
    "model.activation",
    "env.observation",
)
_SAME_GAME = ("env.team_size", "env.opponents")
# What the bot decides with: 'matching' takes it whole or not at all.
_POLICY_SIDE = ("shared.", "policy.")

log = logging.getLogger("nitro")


class TransferError(ValueError):
    """The run to start from cannot be found or does not fit this run; the message says why."""


@dataclass(frozen=True)
class Source:
    state: dict[str, Any]  # the checkpoint
    label: str  # "my-1v1-bot at 250,000,000 agent steps"
    run: str = ""  # the run's name: "my-1v1-bot"
    parts: str = (
        ""  # which parts it gives (start_from.parts when empty); a copy of replays has no value network
    )


def find_source(settings: StartFromConfig, runs: Path) -> Source | None:
    """The save `settings` names, looked for in the runs folder `runs`, then as a path; None when empty or
    switched off (start_from.enabled = false). A copy of replays whose value network was never trained
    gives its policy only (`untrained_value`), whatever start_from.parts says."""
    if not settings.source:
        return None
    source = find_save(settings.source, settings.at, runs, "start_from.run")
    if settings.parts != "policy" and untrained_value(source.state):
        return dataclasses.replace(source, parts="policy")
    return source


def untrained_value(state: dict[str, Any]) -> bool:
    """Whether the save `state` holds a value network that was never trained: a bot cloned from recordings
    (`nitro pretrain`), or a [pretrain] run's first save, the copy of its replays before any training."""
    copied = bool(state.get("started_from_replays")) and int(state.get("agent_steps", 0)) == 0
    return "pretrained" in state or copied


def policy_only_note(source: Source) -> str:
    """Why only the policy of `source` is taken (for a start whose settings ask for more)."""
    return (
        f"{source.label} is a copy of replays whose value network was never trained, so only its policy is "
        "taken; this run's value network starts fresh."
    )


def find_save(name: str, at: int, runs: Path, setting: str) -> Source:
    """The save of run (or save path) `name` at `at` agent steps (0: its newest), looked for in the runs
    folder `runs`, then as a path; `setting` names the setting in the errors."""
    for path in source_paths(name, runs):
        if (path / "state.pt").is_file():  # one save
            state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
            run = path.parent.parent.name
            return Source(state, f"{run} at {int(state['agent_steps']):,} agent steps", run)
        folder = RunFolder(path)
        if folder.info.is_file():
            state = folder.load_checkpoint(at) if at else folder.load_latest()
            if state is None:
                saves = ", ".join(f"{int(p.name):,}" for p in folder.complete_checkpoints()) or "none yet"
                which = f"no save at {at:,} agent steps" if at else "no save yet"
                raise TransferError(f"{setting}: '{path.name}' has {which} (saves: {saves})")
            return Source(state, f"{path.name} at {int(state['agent_steps']):,} agent steps", path.name)
    raise TransferError(f"{setting}: there is no run or save called '{name}' in {runs} or {Path.cwd()}")


def problems(source: Source, config: NitroConfig) -> list[str]:
    """Why the source's networks cannot start this run (empty when they can)."""
    saved, mine = source.state["settings"], fixed_settings(config)
    parts = source.parts or config.start_from.parts

    def differs(path: str) -> bool:
        return bool(_plain(saved[path]) != _plain(mine[path]))

    found = [
        f"{path} is {_text(saved[path])} there and {_text(mine[path])} here"
        for path in _SAME
        if differs(path) and not (parts == "matching" and path == "model.value")
    ]
    # The encoder and, for attention networks, its sizes: the heads only split the width, so other heads
    # would load without an error and compute something else.
    encoder = [path for path in ("model.encoder",) if saved[path] != mine[path]]
    if not encoder:
        encoder = [path for path in encoder_settings(saved, config) if saved[path] != mine[path]]
    found += [f"{path} is {_text(saved[path])} there and {_text(mine[path])} here" for path in encoder]
    # The residual trunk shapes the policy; the value heads matter only when the value side is taken.
    shaped = ["model.residual"]
    if parts == "all":
        shaped += ["model.value_heads", "model.value_head_layers"]
    found += [
        f"{path} is {_text(saved[path])} there and {_text(mine[path])} here"
        for path in shaped
        if differs(path)
    ]
    room = config.env.obs_team_size
    if reads_any_cars(saved, config):
        pass  # an attention encoder reads any number of cars: any game size and room fit
    elif saved["env.obs_team_size"] != room:
        found.append(f"env.obs_team_size is {saved['env.obs_team_size']} there and {room} here")
    elif room == 0:
        found += [
            f"{path} is {_text(saved[path])} there and {_text(mine[path])} here (only a padded observation, "
            "env.obs_team_size the same in both, lets the game size change)"
            for path in _SAME_GAME
            if differs(path)
        ]
    if parts in ("all", "matching") and differs("ppo.popart"):
        found.append(
            'ppo.popart differs, so the value network\'s scaling cannot be taken: use parts = "policy"'
        )
    return found


def rhythm_notes(source: Source, config: NitroConfig) -> list[str]:
    """What means other game time when this run's tick skip is not the source's: ppo.gae_lambda and
    the settings counted in agent steps are per decision, the discounts in seconds. Empty when the tick skips
    match or the source does not record its own."""
    saved = source.state.get("all_settings", {})
    if "env.tick_skip" not in saved or int(saved["env.tick_skip"]) == config.env.tick_skip:
        return []
    there, here = int(saved["env.tick_skip"]), config.env.tick_skip
    ratio, ppo = here / there, config.ppo
    lam = float(saved.get("ppo.gae_lambda", ppo.gae_lambda))
    start = f"Tick skip {there} there, {here} here:"

    def ahead(half_life: str) -> str:
        theirs = float(saved.get(f"ppo.{half_life}", getattr(ppo, half_life)))
        return (
            f"{_window(lam, theirs, there):.2f} s there ({lam:g}), "
            f"{_window(ppo.gae_lambda, getattr(ppo, half_life), here):.2f} s here ({ppo.gae_lambda:g}); "
            f"the same seconds is {lam**ratio:.4f}"
        )

    advantages = ahead("discount_half_life")
    notes = [f"{start} ppo.gae_lambda counts decisions, so the advantages look ahead {advantages}."]
    if "goal" in config.model.value_heads:
        notes.append(
            f"{start} the goal critic's advantages (ppo.gae_lambda, ppo.goal_critic_half_life) look ahead "
            f"{ahead('goal_critic_half_life')}."
        )
    notes += _counted_notes(saved, config, start, there)
    if ppo.entropy_target > 0:
        notes.append(
            f"{start} ppo.entropy_target {ppo.entropy_target:g} is per decision, at {1 / ratio:.3g} times "
            "the decisions a second."
        )
    seconds = "ppo.discount_half_life and ppo.goal_critic_half_life are in seconds and carry over unchanged."
    return [*notes, f"{start} {seconds}"]


# The settings both runs may have that count agent steps, with the setting that switches each on.
_COUNTED = {
    "ppo.steps_per_iteration": None,
    "selfplay.snapshot_steps": "selfplay.enabled",
    "selfplay.flat_rating_steps": "selfplay.enabled",
    "ppo.vdag_reinit_steps": "ppo.vdag_beta",
}


def _counted_notes(saved: dict[str, Any], config: NitroConfig, start: str, there: int) -> list[str]:
    """The settings counted in agent steps: each against the source's own number where both runs use it,
    those that start with this run as this run's own."""
    mine, ratio = all_settings(config), config.env.tick_skip / there
    kept, changed, own = [], [], []
    for name, switch in _COUNTED.items():
        value = float(mine[name])
        if value <= 0 or (switch and not mine[switch]):
            continue
        theirs = float(saved.get(name) or 0) if not switch or saved.get(switch) else 0.0
        if theirs <= 0:
            own.append(f"{name} {value:,.0f}")
        elif theirs == value:
            kept.append(f"{name} {value:,.0f}")
        else:
            times = value * config.env.tick_skip / (theirs * there)
            game = "the same game time" if math.isclose(times, 1.0) else f"{times:.3g} times its game time"
            changed.append(f"{name} {theirs:,.0f} there, {value:,.0f} here ({game})")
    counts = f"{start} these count agent steps (decisions)"
    lines = []
    if kept:
        lines.append(
            f"{counts} and keep the number they had there, so each now covers {ratio:.3g} times the game "
            f"time: {', '.join(kept)}."
        )
    if changed:
        lines.append(f"{counts} and were changed from there: {', '.join(changed)}.")
    if own:
        lines.append(
            f"{counts} from this run's start, so each covers {ratio:.3g} times the game time it would at "
            f"tick skip {there}: {', '.join(own)}."
        )
    return lines


def _window(lam: float, half_life: float, tick_skip: int) -> float:
    """Seconds GAE's advantages look ahead: one decision over 1 - gamma x lambda."""
    return (tick_skip / 120) / (1 - discount_from_half_life(half_life, tick_skip) * lam)


@dataclass(frozen=True)
class Matching:
    """What `parts = "matching"` takes from a save."""

    weights: dict[str, torch.Tensor]  # this model's state, every fitting saved tensor in its place
    taken: tuple[str, ...]
    numbers: int  # parameters taken
    fresh: tuple[str, ...]  # this model's tensors that start as built, by network
    left_out: tuple[str, ...]  # saved tensors of networks not started fresh, with no place here

    def summary(self, label: str) -> str:
        fresh = ", ".join(self.fresh) or "nothing"
        left = f"; the source's left out: {', '.join(self.left_out)}" if self.left_out else ""
        trunk = " and shared trunk" if any(k.startswith("shared.") for k in self.taken) else ""
        return (
            f"Took {len(self.taken)} tensors ({self.numbers:,} parameters) from {label}: the policy{trunk} "
            f"whole; starting fresh: {fresh}{left}."
        )


def matching(own: dict[str, torch.Tensor], saved: dict[str, torch.Tensor], label: str) -> Matching:
    """`own` (a model's state) with every tensor of `saved` that has the same name and shape; refused when
    any of the policy or shared trunk would not be taken (part of a policy decides like neither)."""
    taken = {k: v for k, v in saved.items() if k in own and own[k].shape == v.shape}
    for key in (*own, *saved):
        if key.startswith(_POLICY_SIDE) and key not in taken:
            there = tuple(saved[key].shape) if key in saved else "missing"
            here = tuple(own[key].shape) if key in own else "missing"
            raise TransferError(
                f"start_from: {label}'s policy does not fit this run whole ({key}: {there} there, {here} "
                'here); parts = "matching" never takes part of a policy: keep model.shared and '
                "model.policy as the source's"
            )
    fresh = [k for k in own if k not in taken]
    restarted = {_network(k) for k in fresh}
    return Matching(
        {**own, **taken},
        tuple(taken),
        sum(v.numel() for v in taken.values()),
        by_network(fresh, own),
        by_network([k for k in saved if k not in own and _network(k) not in restarted], saved),
    )


def by_network(keys: list[str], among: dict[str, torch.Tensor]) -> tuple[str, ...]:
    """`keys` grouped by network ("value_heads.main (all layers)", "value (4.weight, 4.bias)")."""
    out = []
    for name in dict.fromkeys(_network(k) for k in keys):
        mine = [k for k in keys if _network(k) == name]
        whole = len(mine) == sum(1 for k in among if _network(k) == name)
        rest = ", ".join(k[len(name) + 1 :] for k in mine)
        out.append(f"{name} (all layers)" if whole else f"{name} ({rest})")
    return tuple(out)


def _network(key: str) -> str:
    first, *rest = key.split(".")
    return f"{first}.{rest[0]}" if first == "value_heads" else first


def check(source: Source, config: NitroConfig) -> None:
    """Raises TransferError saying why the source's networks cannot start this run (`problems`)."""
    if found := problems(source, config):
        raise TransferError(
            f"start_from: {source.label} cannot start this run: " + "; ".join(found) + ". Change this run's "
            "settings to match, or start from scratch (`nitro surgery` widens a run's networks)."
        )


def apply(source: Source, learner: Learner, config: NitroConfig) -> None:
    """Puts the source's networks (and, for `parts = "all"` or "matching", its value and reward scaling)
    into `learner`. The two must have the same actions."""
    check(source, config)
    parts = source.parts or config.start_from.parts
    saved = source.state["learner"]
    weights: dict[str, torch.Tensor] = saved["model"]
    if parts == "policy":
        weights = {
            **learner.model.state_dict(),
            **{k: v for k, v in weights.items() if not k.startswith(("value.", "value_heads."))},
        }
    elif parts == "matching":
        taken = matching(learner.model.state_dict(), weights, source.label)
        weights = taken.weights
        log.info(taken.summary(source.label))
    try:
        learner.model.load_state_dict(weights)
    except RuntimeError as err:
        raise TransferError(
            f"start_from: {source.label}'s networks do not fit this run's observation or actions ({err})"
        ) from None
    if parts in ("all", "matching"):
        learner.value_norm.load_state_dict(saved["value_norm"])
        learner.reward_scaler.load_state_dict(saved["reward_scaler"])


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _text(value: Any) -> str:
    """A value as the settings file writes it."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_text(item) for item in value) + "]"
    return str(value).lower() if isinstance(value, bool) else str(value)
