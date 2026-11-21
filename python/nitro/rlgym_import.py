"""`nitro import-rlgym`: brings an RLGym-PPO or rlgym-learn bot into NitroLearn so it trains
on as the same network.

The save: RLGym-PPO's checkpoint folder (PPO_POLICY.pt, PPO_VALUE_NET.pt, their Adam states,
BOOK_KEEPING_VARS.json) or rlgym-learn's (ppo_learner/actor_critic.pt, optimizers.pt, ppo_agent.json,
trajectory_processor.json). Both networks are Linear and ReLU layers. The project's Python
(`rlgym_source`) says the observation builder, action parser, rewards and settings. NitroLearn builds the
observation itself with its exact copy of RLGym's layout, and a fixed input map (`model.input_map`) turns it
into the numbers the network was trained on: picked, scaled, and standardised and clipped as RLGym-PPO's
`standardize_obs` did. Before anything is written, NitroLearn's network is held to the original on real game
states: RLGym's own observation code (`nitro.rlgym_reference`) and the original networks against NitroLearn's
observation, map and networks, every action probability within 1e-5 and every value of the critic within
1e-4."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nitro.config.schema import NitroConfig
from nitro.gigalearn_import import (
    EVERY_STEP,
    MIN_PROB,
    GigaLearnImportError,
    _half_life,
    _toml_value,
    checked_settings,
    imported_state,
    nitro_probs,
    write_run,
)
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import discount_from_half_life
from nitro.rlgym_reference.advanced_obs import AdvancedObs
from nitro.rlgym_reference.advanced_padder import AdvancedObsPadder
from nitro.rlgym_reference.default_obs import DefaultObs
from nitro.rlgym_reference.game_state import GameState
from nitro.rlgym_reference.lookup_act import LookupAction
from nitro.rlgym_reference.physics_object import PhysicsObject
from nitro.rlgym_reference.player_data import PlayerData
from nitro.rlgym_source import PyCall, RlgymSource, SourceError, Unread
from nitro.rlgym_source import read as read_source

PROB_TOLERANCE = 1e-5
VALUE_TOLERANCE = 1e-4  # float32 sums over a few hundred numbers
NUM_ACTIONS = 90  # rlgym-tools' LookupAction with its default bins = NitroLearn's table, same order
OBS_CLIP = 5.0  # RLGym-PPO clips standardised observations to +-5 (batched_agent_manager.py)
REWARD_CLIP = 10.0  # and standardised rewards to +-10 (torch_functions.compute_gae)
RLGYM_V1 = ("rlgym_sim", "rlgym.utils", "rlgym_tools")  # the classic RLGym, the one RLGym-PPO uses
RLGYM_V2 = ("rlgym.rocket_league", "rlgym.api", "rlgym_tools.rocket_league")

# RLGym-PPO's Learner defaults (rlgym_ppo/learner.py).
LEARNER_DEFAULTS: dict[str, Any] = {
    "n_proc": 8,
    "ts_per_iteration": 50_000,
    "exp_buffer_size": 100_000,
    "standardize_returns": True,
    "standardize_obs": True,
    "ppo_epochs": 10,
    "ppo_batch_size": 50_000,
    "ppo_minibatch_size": None,
    "ppo_ent_coef": 0.005,
    "ppo_clip_range": 0.2,
    "gae_lambda": 0.95,
    "gae_gamma": 0.99,
    "policy_lr": 3e-4,
    "critic_lr": 3e-4,
    "random_seed": 123,
}
# rlgym_sim.make's defaults (make.py).
MAKE_DEFAULTS: dict[str, Any] = {"tick_skip": 8, "team_size": 1, "spawn_opponents": False}


class RlgymImportError(GigaLearnImportError):
    """The bot cannot be brought over; the message says why."""


# ---------------------------------------------------------------------------------------------------------
# The save


@dataclass
class RlgymSave:
    kind: str  # "RLGym-PPO" or "rlgym-learn"
    folder: Path
    policy: list[tuple[torch.Tensor, torch.Tensor]]  # (weight, bias) per linear layer, ReLU between
    critic: list[tuple[torch.Tensor, torch.Tensor]]
    policy_optim: dict[str, Any] | None
    critic_optim: dict[str, Any] | None
    steps: int
    iterations: int
    returns: tuple[float, float, int] | None  # mean, M2 (sum of squared differences), count
    obs_stats: tuple[np.ndarray, np.ndarray, int] | None  # mean, M2, count per observation number


def _newest(folder: Path, has: Callable[[Path], bool]) -> Path | None:
    """`folder` when it is a save, else the newest numbered save below it (three levels at most: the bot's
    folder, checkpoints, a run)."""
    if has(folder):
        return folder
    for pattern in ("*", "*/*", "*/*/*"):
        found = [p for p in folder.glob(pattern) if p.is_dir() and p.name.isdigit() and has(p)]
        if len({p.parent for p in found}) > 1:
            runs = ", ".join(sorted({p.parent.name for p in found}))
            raise RlgymImportError(f"{folder} holds several runs ({runs}); point at the one to import")
        if found:
            return max(found, key=lambda p: int(p.name))
    return None


def _rlgym_ppo(path: Path) -> bool:
    return (path / "PPO_POLICY.pt").is_file() and (path / "PPO_VALUE_NET.pt").is_file()


def _rlgym_learn(path: Path) -> bool:
    learner = path / "ppo_learner"
    return (learner / "actor_critic.pt").is_file() or (learner / "actor.pt").is_file()


def _layers(
    state: dict[str, torch.Tensor], prefix: str, file: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The linear layers of `prefix` (`model.` in RLGym-PPO), which RLGym-PPO builds as Linear, ReLU, ...,
    Linear: their indices step by 2."""
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise RlgymImportError(f"{file}: not a network's saved numbers")
    keys = sorted(
        {
            int(k[len(prefix) :].split(".")[0])
            for k in state
            if k.startswith(prefix) and k[len(prefix) :].split(".")[0].isdigit()
        }
    )
    if not keys or keys != list(range(0, 2 * len(keys), 2)):
        raise RlgymImportError(
            f"{file}: its layers are not RLGym-PPO's Linear-ReLU layout ({prefix}0, {prefix}2, ...); a "
            "network "
            "of another shape cannot be brought over yet"
        )
    out = []
    for index in keys:
        weight, bias = state.get(f"{prefix}{index}.weight"), state.get(f"{prefix}{index}.bias")
        if weight is None or bias is None or weight.dim() != 2:
            raise RlgymImportError(f"{file}: layer {index} is not a linear layer")
        out.append((weight.detach().float().clone(), bias.detach().float().clone()))
    for (a, _), (b, _) in itertools.pairwise(out):
        if a.shape[0] != b.shape[1]:
            raise RlgymImportError(f"{file}: the layers' sizes do not fit together")
    return out


def _load(file: Path) -> Any:
    """A PyTorch save of plain numbers; anything else (damaged, or holding code, which is never run) is
    refused."""
    try:
        return torch.load(file, map_location="cpu", weights_only=True)
    except Exception:  # torch raises many kinds; its message suggests loading unsafely, so it is not shown
        raise RlgymImportError(
            f"{file} could not be read as saved network numbers: it is damaged, or holds something other "
            "than plain numbers (NitroLearn never runs code from a save)"
        ) from None


def _json(file: Path) -> dict[str, Any]:
    if not file.is_file():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise RlgymImportError(f"{file} could not be read ({err})") from None
    if not isinstance(data, dict):
        raise RlgymImportError(f"{file} is not as RLGym writes it")
    return data


def _welford(stat: dict[str, Any] | None, width: int | None) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not stat:
        return None
    mean = np.asarray(stat["mean"], dtype=np.float64).reshape(-1)
    m2 = np.asarray(stat.get("var", stat.get("m2")), dtype=np.float64).reshape(-1)
    if width is not None and mean.shape[0] != width:
        raise RlgymImportError(
            f"the observation statistics have {mean.shape[0]} numbers, the network reads {width}"
        )
    if m2.shape != mean.shape:
        raise RlgymImportError("the save's running statistics have fewer spreads than means")
    return mean, m2, int(stat["count"])


def read_save(path: Path) -> RlgymSave:
    """The save at `path` (or the newest below it); files not as RLGym writes them are refused."""
    try:
        return _read_save(path)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as err:
        raise RlgymImportError(
            f"the save in {path} is not as RLGym-PPO or rlgym-learn write it ({type(err).__name__}: {err})"
        ) from None


def _read_save(path: Path) -> RlgymSave:
    folder = _newest(path, _rlgym_ppo)
    if folder is not None:
        policy = _layers(_load(folder / "PPO_POLICY.pt"), "model.", "PPO_POLICY.pt")
        critic = _layers(_load(folder / "PPO_VALUE_NET.pt"), "model.", "PPO_VALUE_NET.pt")
        book = _json(folder / "BOOK_KEEPING_VARS.json")
        files = [folder / "PPO_POLICY_OPTIMIZER.pt", folder / "PPO_VALUE_NET_OPTIMIZER.pt"]
        policy_optim, critic_optim = (_load(f) if f.is_file() else None for f in files)
        returns = _welford(book.get("reward_running_stats"), 1)
        return RlgymSave(
            "RLGym-PPO",
            folder,
            policy,
            critic,
            policy_optim,
            critic_optim,
            int(book.get("cumulative_timesteps", 0)),
            int(book.get("epoch", 0)),
            (float(returns[0][0]), float(returns[1][0]), returns[2]) if returns else None,
            _welford(book.get("obs_running_stats"), policy[0][0].shape[1]),
        )
    folder = _newest(path, _rlgym_learn)
    if folder is None:
        raise RlgymImportError(
            f"{path} holds no RLGym-PPO save (PPO_POLICY.pt and PPO_VALUE_NET.pt) and no rlgym-learn save "
            "(ppo_learner/actor_critic.pt). Point at the bot's checkpoints folder or at one save in it."
        )
    learner = folder / "ppo_learner"
    if (learner / "actor_critic.pt").is_file():
        both = _load(learner / "actor_critic.pt")
        policy = _layers(both, "actor.model.", "actor_critic.pt")
        critic = _layers(both, "critic.model.", "actor_critic.pt")
    else:
        policy = _layers(_load(learner / "actor.pt"), "model.", "actor.pt")
        critic = _layers(_load(learner / "critic.pt"), "model.", "critic.pt")
    saved_optims = _load(learner / "optimizers.pt") if (learner / "optimizers.pt").is_file() else None
    agent = _json(folder / "ppo_agent.json")
    processor = _json(folder / "trajectory_processor.json")
    stat = processor.get("return_running_stats")
    learn_returns = (float(stat["mean"]), float(stat["m2"]), int(stat["count"])) if stat else None
    pair = saved_optims if isinstance(saved_optims, list) and len(saved_optims) == 2 else [None, None]
    return RlgymSave(
        "rlgym-learn",
        folder,
        policy,
        critic,
        pair[0],
        pair[1],
        int(agent.get("cumulative_timesteps", 0)),
        int(agent.get("cur_iteration", 0)),
        learn_returns,
        None,
    )


# ---------------------------------------------------------------------------------------------------------
# The plan


@dataclass
class RlgymPlan:
    observation: str = ""
    team_size: int = 1
    room: int = 0  # env.obs_team_size
    opponents: bool = True
    tick_skip: int = 8
    # The input map before standardisation: per network input, the observation number it reads and its scale.
    index: list[int] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    rewards: list[dict[str, Any]] = field(default_factory=list)
    setters: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)  # the Learner's, defaults filled in
    toml: str = ""
    exact: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    to_port: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    builder: Any = None  # RLGym's own builder object, for the parity check

    def lines(self) -> list[str]:
        out: list[str] = []
        for title, items in (
            ("Carried over exactly", self.exact),
            ("Carried over, not exactly the same", self.changed),
            ("Not carried over", self.left_out),
            ("Your own code to port by hand (the bot was not changed for these)", self.to_port),
            ("Cannot import", self.refused),
        ):
            if items:
                out.append(f"{title}:")
                out += [f"  - {item}" for item in items]
        return out


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _from(call: Any, family: tuple[str, ...]) -> bool:
    return isinstance(call, PyCall) and call.module.startswith(family)


# The numbers before and per car of NitroLearn's copies (nitro_core::obs): gigalearn_default and the padders.
_DEFAULT_OWN, _DEFAULT_CAR = 51, 19
_PADDER_OWN, _PADDER_CAR, _PADDER_OTHER = 51, 26, 32


def _default_scales(call: PyCall) -> tuple[list[float], float, float] | None:
    """DefaultObs's (rlgym-sim/rlgym v1) coefficients: position (3), linear velocity, angular velocity."""
    pos = call.arg(0, "pos_coef", 1 / 2300)
    lin, ang = call.arg(2, "lin_vel_coef", 1 / 2300), call.arg(3, "ang_vel_coef", 1 / math.pi)
    pos = [pos] * 3 if _number(pos) is not None else pos
    if not (
        isinstance(pos, list)
        and len(pos) == 3
        and all(_number(p) for p in pos)
        and _number(lin)
        and _number(ang)
    ):
        return None
    return [float(p) for p in pos], float(lin), float(ang)


def plan_observation(save: RlgymSave, source: RlgymSource, plan: RlgymPlan) -> None:
    size = save.policy[0][0].shape[1]
    call = source.obs
    team = plan.team_size
    cars = team * 2 if plan.opponents else team
    if _from(call, RLGYM_V2):
        plan.refused.append(
            f"Observation {call.text} (line {call.line}) is RLGym 2's: it reads car states NitroLearn's game "
            "view (and RLBot's) does not have (whether a car is jumping or flipping, holds jump, its time "
            "since "
            "the jump, whether it boosts). Needed: those in NitroLearn's core and the RLBot bot."
        )
        return
    if not isinstance(call, PyCall) or not _from(call, RLGYM_V1):
        what = (
            call.text
            if isinstance(call, PyCall)
            else (call.text if isinstance(call, Unread) else "not found")
        )
        where = source.classes.get(call.name, "your project") if isinstance(call, PyCall) else "the project"
        plan.refused.append(
            f"Observation builder {what} ({where}) is not one of RLGym's built-in ones NitroLearn has copied "
            "(DefaultObs, AdvancedObs, rlgym-tools' AdvancedObsPadder). Needed: the same observation in "
            "NitroLearn's core, number for number."
        )
        return
    if call.name == "DefaultObs":
        scales = _default_scales(call)
        if scales is None:
            plan.refused.append(
                f"Observation {call.text} (line {call.line}): its coefficients could not be read"
            )
            return
        pos, lin, ang = scales
        plan.observation, expected = "gigalearn_default", _DEFAULT_OWN + _DEFAULT_CAR * cars
        core_pos, core_lin, core_ang = (1 / 4096, 1 / 5120, 1 / 2044), 1 / 2300, 1 / 5.5
        block = [*(p / c for p, c in zip(pos, core_pos, strict=True))]
        ball = [*block, *[lin / core_lin] * 3, *[ang / core_ang] * 3]
        car = [*block, 1, 1, 1, 1, 1, 1, *[lin / core_lin] * 3, *[ang / core_ang] * 3, 1, 1, 1, 1]
        plan.scale = [*ball, *[1.0] * (8 + 34), *car * cars]
        plan.index = list(range(expected))
        plan.builder = DefaultObs(np.asarray(pos), call.arg(1, "ang_coef", 1 / math.pi), lin, ang)
        note = "DefaultObs -> 'gigalearn_default' (the same layout) with its coefficients in the input map"
    elif call.name == "AdvancedObs" and not call.args and not call.kwargs:
        plan.observation = "gigalearn_padder"
        expected = _PADDER_OWN + (_PADDER_CAR - 1) + (_PADDER_OTHER - 1) * (cars - 1)
        # gigalearn_padder's cars have one more number (its 0) than AdvancedObs's: skipped by the map.
        own = list(range(_PADDER_OWN + _PADDER_CAR - 1))
        others = [
            _PADDER_OWN + _PADDER_CAR + k * _PADDER_OTHER + i
            for k in range(cars - 1)
            for i in range(_PADDER_OTHER)
            if i != 25
        ]
        plan.index, plan.scale = [*own, *others], [1.0] * expected
        plan.builder = AdvancedObs()
        note = "AdvancedObs -> 'gigalearn_padder' (the same numbers; its extra 0 per car skipped by the map)"
    elif call.name == "AdvancedObsPadder":
        places = call.arg(0, "team_size", 3)
        if not isinstance(places, int) or isinstance(places, bool) or not team <= places <= 4:
            plan.refused.append(f"Observation {call.text} (line {call.line}): team_size must be {team} to 4")
            return
        plan.observation, plan.room = "rlgym_padder", places if places != team else 0
        expected = _PADDER_OWN + _PADDER_CAR + _PADDER_OTHER * (2 * places - 1)
        plan.index, plan.scale = list(range(expected)), [1.0] * expected
        plan.builder = AdvancedObsPadder(team_size=places)
        note = f"rlgym-tools' AdvancedObsPadder({places}) -> 'rlgym_padder' (the same numbers)"
    else:
        plan.refused.append(
            f"Observation {call.text} (line {call.line}) has settings NitroLearn's copy does not take"
        )
        return
    if size != expected:
        plan.refused.append(
            f"The networks read {size} numbers, but {call.text} gives {expected} in {team}v{team}: the save "
            "and the "
            "project do not belong together (or the game size is not the project's)."
        )
        return
    plan.exact.append(f"Observation: {note}, {team}v{team}.")


def plan_actions(save: RlgymSave, source: RlgymSource, plan: RlgymPlan) -> None:
    outputs = save.policy[-1][0].shape[0]
    call = source.action
    if outputs in (16, 21) or (
        isinstance(call, PyCall) and call.name in ("ContinuousAction", "DiscreteAction", "DefaultAction")
    ):
        what = call.text if isinstance(call, PyCall) else f"a policy with {outputs} outputs"
        plan.refused.append(
            f"Action parser {what}: NitroLearn chooses from a table of whole actions; a continuous or "
            "per-control "
            "(multi-discrete) policy cannot become the same network. Needed: that kind of policy in "
            "NitroLearn."
        )
        return
    if not (isinstance(call, PyCall) and call.name == "LookupAction" and _from(call, RLGYM_V1)):
        what = call.text if isinstance(call, PyCall) else "not found"
        plan.refused.append(
            f"Action parser {what} is not rlgym-tools' LookupAction: its actions cannot be known from the "
            "network. Needed: its action list as a NitroLearn action table."
        )
        return
    bins = call.arg(0, "bins", None)
    if bins is not None or outputs != NUM_ACTIONS:
        plan.refused.append(
            f"Action parser {call.text} (line {call.line}) has its own bins ({outputs} actions); "
            "NitroLearn's table "
            "is LookupAction's default 90. Needed: that table in NitroLearn."
        )
        return
    table = LookupAction.make_lookup_table(LookupAction().bins)
    plan.builder_actions = table  # type: ignore[attr-defined]
    plan.exact.append(
        "Actions: rlgym-tools' LookupAction -> NitroLearn's 90-action table, same order; no masks, as "
        "RLGym-PPO."
    )


def _reward_blocks(call: Any, source: RlgymSource) -> tuple[list[tuple[dict[str, Any], float]], list[str]]:
    """RLGym's reward as NitroLearn rewards with a factor each (an EventReward gives several), and what is "
    "left."""
    if not isinstance(call, PyCall):
        return [], [f"a reward the import cannot read ({getattr(call, 'text', call)})"]
    if not _from(call, RLGYM_V1):
        where = source.classes.get(call.name, "not RLGym's")
        return [], [f"{call.text} (line {call.line}, {where}): port it as a NitroLearn Rust reward"]
    arg = call.arg
    if call.name == "VelocityPlayerToBallReward" and not arg(0, "use_scalar_projection", False):
        return [({"name": "velocity_player_to_ball"}, 1.0)], []
    if call.name == "VelocityBallToGoalReward" and not arg(1, "use_scalar_projection", False):
        own = bool(arg(0, "own_goal", False))
        block = (
            {"name": "gigalearn_velocity_ball_to_goal", "own_goal": True}
            if own
            else {"name": "velocity_ball_to_goal"}
        )
        return [(block, 1.0)], []
    if call.name == "FaceBallReward":
        return [({"name": "face_ball"}, 1.0)], []
    if call.name == "TouchBallReward" and (_number(arg(0, "aerial_weight", 0.0)) or 0.0) == 0.0:
        return [({"name": "touch_ball"}, 1.0)], []
    if call.name == "SaveBoostReward":
        return [({"name": "save_boost", "exponent": 0.5}, 1.0)], []
    if call.name == "LiuDistancePlayerToBallReward":
        return [({"name": "rlgym_liu_distance_player_to_ball"}, 1.0)], []
    if call.name == "LiuDistanceBallToGoalReward":
        return [
            ({"name": "rlgym_liu_distance_ball_to_goal", "own_goal": bool(arg(0, "own_goal", False))}, 1.0)
        ], []
    if call.name == "EventReward":
        names = ("goal", "team_goal", "concede", "touch", "shot", "save", "demo", "boost_pickup")
        values = {n: _number(arg(i, n, 0.0)) for i, n in enumerate(names)}
        if any(v is None for v in values.values()):
            return [], [f"{call.text} (line {call.line}): its weights could not be read"]
        blocks: list[tuple[dict[str, Any], float]] = []
        left = []
        team_goal, concede = values["team_goal"] or 0.0, values["concede"] or 0.0
        if team_goal or concede:
            blocks.append(({"name": "rlgym_goal", "team_goal": team_goal, "concede": concede}, 1.0))
        if values["touch"]:
            blocks.append(({"name": "rlgym_touch_start"}, values["touch"] or 0.0))
        if values["demo"]:
            blocks.append(({"name": "demo"}, values["demo"] or 0.0))
        if values["boost_pickup"]:
            blocks.append(({"name": "rlgym_boost_gain"}, values["boost_pickup"] or 0.0))
        for name in ("goal", "shot", "save"):
            if values[name]:
                left.append(
                    f"{call.text} (line {call.line}): its {name} part ({values[name]:g}) is paid by Rocket "
                    "League's "
                    "own goal, shot and save detection, which NitroLearn's simulator does not have"
                )
        return blocks, left
    return [], [
        f"{call.text} (line {call.line}) is an RLGym reward NitroLearn has not matched yet: port it by hand"
    ]


def plan_rewards(source: RlgymSource, plan: RlgymPlan) -> None:
    reward = source.reward
    pairs: list[tuple[Any, float | None]] = []
    if isinstance(reward, PyCall) and reward.name == "CombinedReward":
        functions = reward.arg(0, "reward_functions", None)
        weights = reward.arg(1, "reward_weights", None)
        if isinstance(functions, list):
            if weights is None:  # CombinedReward's default: 1 each
                weights = [1.0] * len(functions)
            if not isinstance(weights, list) or len(weights) != len(functions):
                plan.refused.append(
                    f"{reward.text} (line {reward.line}): its weights could not be read; write them as a "
                    "list of plain numbers, one per reward"
                )
                return
            pairs = list(zip(functions, [_number(w) for w in weights], strict=True))
        else:
            plan.refused.append(f"{reward.text} (line {reward.line}): its rewards could not be read")
            return
    elif reward is not None:
        pairs = [(reward, 1.0)]
    for call, weight in pairs:
        if weight is None:
            plan.to_port.append(f"{getattr(call, 'text', call)}: its weight could not be read")
            continue
        blocks, left = _reward_blocks(call, source)
        plan.to_port += left
        for block, factor in blocks:
            w = weight * factor
            scaled = (
                w * 8 / plan.tick_skip
                if block["name"]
                in EVERY_STEP | {"rlgym_liu_distance_player_to_ball", "rlgym_liu_distance_ball_to_goal"}
                else w
            )
            plan.rewards.append({**block, "weight": scaled})
            plan.exact.append(f"Reward: {call.text} x {weight:g} -> {block['name']} (weight {scaled:g}).")
    if not plan.rewards:
        plan.refused.append("No reward could be carried over; a bot cannot train on without rewards.")


def plan_game(source: RlgymSource, plan: RlgymPlan) -> None:
    make = source.make
    if make is None:
        plan.refused.append("The project's rlgym_sim.make(...) (or rlgym.make) was not found.")
        return
    values = {**MAKE_DEFAULTS, **{k: v for k, v in make.kwargs.items() if k in MAKE_DEFAULTS}}
    if not all(isinstance(values[k], (int, bool)) for k in MAKE_DEFAULTS):
        plan.refused.append(
            f"rlgym_sim.make(...) (line {make.line}): tick_skip, team_size or spawn_opponents not read"
        )
        return
    plan.tick_skip, plan.team_size, plan.opponents = (
        int(values["tick_skip"]),
        int(values["team_size"]),
        bool(values["spawn_opponents"]),
    )
    if plan.tick_skip < 1 or plan.team_size < 1:
        plan.refused.append(
            f"rlgym_sim.make(...) (line {make.line}): tick_skip and team_size must be 1 or more"
        )
        plan.tick_skip, plan.team_size = max(plan.tick_skip, 1), max(plan.team_size, 1)
        return
    if not plan.opponents:
        plan.refused.append("spawn_opponents = False: the import is for bots that play against opponents.")
    # rlgym_sim.make's game changes NitroLearn's games do not have (its default: 1, the normal game).
    for key, what in (("gravity", "gravity"), ("boost_consumption", "boost use")):
        value = make.kwargs.get(key, 1)
        if _number(value) is not None and value != 1:
            plan.changed.append(
                f"rlgym_sim.make {key} = {value:g}: the bot was trained with {value:g} times the normal "
                f"{what}; NitroLearn plays the normal game."
            )


def plan_ends_and_starts(source: RlgymSource, plan: RlgymPlan) -> tuple[float, float]:
    no_touch = timeout = 0.0
    seconds = plan.tick_skip / 120
    for call in source.terminals:
        if not isinstance(call, PyCall):
            plan.to_port.append(f"End condition {getattr(call, 'text', call)}: not read")
        elif call.name == "NoTouchTimeoutCondition" and _number(call.arg(0, "max_steps", None)):
            no_touch = float(call.arg(0, "max_steps", 0)) * seconds
            plan.changed.append(
                f"Ends: {call.text} -> no_touch_seconds = {no_touch:g}; RLGym-PPO ends the episode there as "
                "if done, "
                "NitroLearn as cut short (the value of the state is kept)."
            )
        elif call.name == "TimeoutCondition" and _number(call.arg(0, "max_steps", None)):
            timeout = float(call.arg(0, "max_steps", 0)) * seconds
            plan.changed.append(
                f"Ends: {call.text} -> max_episode_seconds = {timeout:g} (cut short, not done)."
            )
        elif call.name == "GoalScoredCondition":
            plan.exact.append("Ends: GoalScoredCondition -> a goal ends the episode.")
        else:
            plan.to_port.append(
                f"End condition {call.text} (line {call.line}): add a matching NitroLearn ending"
            )
    setter = source.setter
    if isinstance(setter, PyCall) and setter.name == "DefaultState":
        plan.setters = [{"name": "kickoff", "weight": 1.0}]
        plan.exact.append("Start: DefaultState -> kickoff.")
    elif isinstance(setter, PyCall) and setter.name == "RandomState":
        plan.setters = [{"name": "random", "weight": 1.0}]
        plan.changed.append(f"Start: {setter.text} -> NitroLearn's random start (drawn its own way).")
    else:
        plan.setters = [{"name": "kickoff", "weight": 1.0}]
        what = setter.text if isinstance(setter, PyCall) else "the default"
        plan.changed.append(f"Start: {what} -> kickoffs (add a matching NitroLearn start for your own).")
    return no_touch, timeout


# The Learner's settings the import uses: what each must be, as a test of its value.
_LEARNER_CHECKS: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "ts_per_iteration": ("a number above 0", lambda v: v > 0),
    "exp_buffer_size": ("a number above 0", lambda v: v > 0),
    "ppo_epochs": ("a number above 0", lambda v: v >= 1),
    "ppo_batch_size": ("a number above 0", lambda v: v >= 1),
    "ppo_minibatch_size": ("a number above 0", lambda v: v >= 1),
    "ppo_ent_coef": ("a number, 0 or more", lambda v: v >= 0),
    "ppo_clip_range": ("a number above 0", lambda v: v > 0),
    "gae_lambda": ("a number from 0 to 1", lambda v: 0 <= v <= 1),
    "gae_gamma": ("a number above 0 and below 1", lambda v: 0 < v < 1),
    "policy_lr": ("a number, 0 or more", lambda v: v >= 0),
    "critic_lr": ("a number, 0 or more", lambda v: v >= 0),
}


def _check_learner(source: RlgymSource, plan: RlgymPlan) -> None:
    """Refuses the Learner settings the import cannot use (not read, or not a sensible number)."""
    for key, (must, fits) in _LEARNER_CHECKS.items():
        value = plan.settings[key]
        if value is None and key == "ppo_minibatch_size":
            continue
        number = _number(value)
        if number is None or not math.isfinite(number) or not fits(number):
            plan.refused.append(
                f"Learner {key} = {getattr(value, 'text', repr(value))} (line {source.learner_line}): "
                f"must be {must}, written as a plain number"
            )
    for key in ("standardize_obs", "standardize_returns"):
        if not isinstance(plan.settings[key], bool):
            value = plan.settings[key]
            plan.refused.append(
                f"Learner {key} = {getattr(value, 'text', repr(value))} (line {source.learner_line}): must "
                "be True or False, written as such"
            )


def build_plan(save: RlgymSave, source: RlgymSource, keep_action_delay: bool = False) -> RlgymPlan:
    plan = RlgymPlan()
    plan.settings = {**LEARNER_DEFAULTS, **{k: v for k, v in source.learner.items() if not k.startswith("#")}}
    make = source.make
    if save.kind == "rlgym-learn" or (make is not None and (make.name == "RLGym" or _from(make, RLGYM_V2))):
        plan.refused.append(
            "This bot is an RLGym 2 bot (rlgym-learn runs RLGym 2 environments). NitroLearn imports "
            "RLGym-PPO "
            "bots on rlgym-sim (RLGym 1) for now: RLGym 2's observations read car states NitroLearn's game "
            "view "
            "and the RLBot bot do not have yet (whether a car is jumping or flipping, holds jump, its time "
            "since "
            "the jump, whether it boosts), and rlgym-learn's settings (its config models) are not read yet. "
            "The "
            "save itself was read (networks, Adam state, steps, return statistics)."
        )
        return plan
    _check_learner(source, plan)
    plan_game(source, plan)
    plan_observation(save, source, plan)
    plan_actions(save, source, plan)
    plan_rewards(source, plan)
    no_touch, timeout = plan_ends_and_starts(source, plan)
    for line in source.unread:
        plan.left_out.append(f"Not read from {source.file.name}, {line}")
    if not plan.refused:
        plan.toml = settings_toml(save, plan, no_touch, timeout, keep_action_delay)
    return plan


def settings_toml(save: RlgymSave, plan: RlgymPlan, no_touch: float, timeout: float, keep_delay: bool) -> str:
    s = plan.settings
    trained_delay = plan.tick_skip - 1
    delay = trained_delay if keep_delay else min(1, trained_delay)
    if not keep_delay and delay != trained_delay:
        plan.changed.append(
            "Action delay: RLGym-sim reads the game one tick into each step, which is NitroLearn's "
            "action_delay "
            f"{trained_delay}; set to {delay}, the lowest a real game allows (the rule for NitroLearn "
            "bots). The bot "
            "acts sooner after it sees than it was trained to; --keep-action-delay keeps its own."
        )
    elif keep_delay:
        plan.exact.append(f"Action delay: {delay} ticks (RLGym-sim's timing, kept with --keep-action-delay).")
    steps = int(s["ts_per_iteration"])
    batch = int(s["ppo_batch_size"])
    mini = int(s["ppo_minibatch_size"] or batch)
    ent = float(s["ppo_ent_coef"])
    gamma = float(s["gae_gamma"])
    lines = [
        "# Imported from an RLGym bot with `nitro import-rlgym`.",
        f"# From: {save.folder}",
        "",
        "[run]",
    ]
    if isinstance(s["random_seed"], int) and s["random_seed"] >= 0:
        lines.append(f"seed = {s['random_seed']}  # RLGym-PPO random_seed")
    lines += [
        "",
        "[env]",
        f"team_size = {plan.team_size}  # rlgym_sim.make team_size",
        f'observation = "{plan.observation}"  # NitroLearn\'s copy of the RLGym observation',
        *([f"obs_team_size = {plan.room}  # AdvancedObsPadder team_size"] if plan.room else []),
        'action_masks = "none"  # RLGym-PPO has no action masks',
        "num_arenas = 256",
        f"tick_skip = {plan.tick_skip}  # rlgym_sim.make tick_skip",
        f"action_delay = {delay}",
        "",
        "[terminals]",
        f"no_touch_seconds = {no_touch:g}",
        *([f"max_episode_seconds = {timeout:g}"] if timeout else []),
        "",
        "[model]",
        f"policy = {_toml_value([w.shape[0] for w, _ in save.policy[:-1]])}  # RLGym-PPO policy_layer_sizes",
        f"value = {_toml_value([w.shape[0] for w, _ in save.critic[:-1]])}  # RLGym-PPO critic_layer_sizes",
        "layer_norm = false",
        'activation = "relu"',
        'init = "pytorch"',
        f"input_map = {len(plan.index)}  # the numbers the network was trained on, made from the observation",
        "",
        "[ppo]",
        f"steps_per_iteration = {steps}  # RLGym-PPO ts_per_iteration",
        f"epochs = {int(s['ppo_epochs'])}  # RLGym-PPO ppo_epochs",
        f"minibatches = {max(1, round(steps / batch))}  # one learning step per ppo_batch_size ({batch})",
        *([f"microbatch = {mini}  # RLGym-PPO ppo_minibatch_size (memory only)"] if mini < batch else []),
        f"policy_lr = {float(s['policy_lr'])!r}",
        f"value_lr = {float(s['critic_lr'])!r}",
        f"discount_half_life = {_half_life(gamma, plan.tick_skip)!r}  # RLGym-PPO gae_gamma {gamma:g}",
        f"gae_lambda = {float(s['gae_lambda'])!r}",
        f"clip_range = {float(s['ppo_clip_range'])!r}",
        f"entropy = {ent * math.log(NUM_ACTIONS)!r}  # RLGym-PPO ppo_ent_coef {ent:g} x ln(90): "
        "NitroLearn's entropy is divided by ln(90)",
        'entropy_normalization = "all"',
        f"reward_clip = {REWARD_CLIP}",
        "popart = false",
        "normalize_advantages = false  # RLGym-PPO does not normalise advantages",
        "adam_eps = 1e-8",
        "max_grad_norm = 0.5",
        "graphed_steps = false",
    ]
    plan.exact.append(
        f"Networks: policy {[w.shape[0] for w, _ in save.policy]}, critic "
        f"{[w.shape[0] for w, _ in save.critic]}, ReLU."
    )
    plan.exact.append(
        f"Learning: {steps} steps an iteration, {int(s['ppo_epochs'])} epochs, a learning step per {batch}, "
        "rates "
        f"{float(s['policy_lr']):g}/{float(s['critic_lr']):g}, gamma {gamma:g}, lambda "
        f"{float(s['gae_lambda']):g}, "
        f"clip {float(s['ppo_clip_range']):g}, entropy {ent:g} (raw), rewards divided by the returns' "
        "spread and "
        "clipped at 10, no advantage normalisation."
    )
    if int(s["exp_buffer_size"]) > steps:
        plan.changed.append(
            f"exp_buffer_size {int(s['exp_buffer_size'])} > ts_per_iteration {steps}: RLGym-PPO also learns "
            "again from "
            "older iterations' games; NitroLearn learns from each iteration's own games only."
        )
    if not s["standardize_returns"]:
        plan.changed.append(
            "standardize_returns = False: NitroLearn always divides rewards by the returns' spread."
        )
    plan.changed.append(
        f"Games: {plan.team_size}v{plan.team_size} in 256 arenas (RLGym-PPO: n_proc {s['n_proc']} "
        "processes); only "
        "how fast games are played, not what is learned."
    )
    for block in plan.rewards:
        lines += ["", "[[rewards]]", *(f"{k} = {_toml_value(v)}" for k, v in block.items())]
    for block in plan.setters:
        lines += ["", "[[state_setters]]", *(f"{k} = {_toml_value(v)}" for k, v in block.items())]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------
# Weights, the input map and the parity check


def _once(items: list[str], line: str) -> None:
    if line not in items:
        items.append(line)


def standardisation(save: RlgymSave, plan: RlgymPlan) -> tuple[np.ndarray, np.ndarray, float]:
    """RLGym-PPO's observation standardisation as the save holds it (mean, std, clip), or none."""
    width = len(plan.index)
    if not plan.settings["standardize_obs"] or save.kind != "RLGym-PPO":
        return np.zeros(width), np.ones(width), 0.0
    if save.obs_stats is None:
        _once(
            plan.refused,
            "standardize_obs is on, but the save has no obs_running_stats in BOOK_KEEPING_VARS.json",
        )
        return np.zeros(width), np.ones(width), 0.0
    mean, m2, count = save.obs_stats
    if count < 2:
        mean, std = np.zeros(width), np.ones(width)
    else:
        var = m2 / (count - 1)
        std = np.sqrt(np.where(var == 0, 1.0, var))
    _once(
        plan.changed,
        "standardize_obs: the save's observation statistics (and the clip at +-5) are kept in the input "
        "map: the "
        "same numbers; from here on they stay fixed instead of following the games.",
    )
    return mean, std, OBS_CLIP


def build_model(config: NitroConfig, plan: RlgymPlan, save: RlgymSave, obs_size: int) -> ActorCritic:
    model = ActorCritic(obs_size, NUM_ACTIONS, config.model)
    mean, std, clip = standardisation(save, plan)
    with torch.no_grad():
        m = model.input_map
        m.index.copy_(torch.tensor(plan.index))
        m.scale.copy_(torch.tensor(plan.scale, dtype=torch.float32))
        m.offset.copy_(torch.tensor(mean, dtype=torch.float32))
        m.divisor.copy_(torch.tensor(std, dtype=torch.float32))
        m.clip.fill_(clip)
        for network, layers in ((model.policy, save.policy), (model.value, save.critic)):
            linears = [module for module in network if isinstance(module, nn.Linear)]
            for module, (weight, bias) in zip(linears, layers, strict=True):
                module.weight.copy_(weight)
                module.bias.copy_(bias)
    return model.eval()


def reference_networks(save: RlgymSave) -> tuple[nn.Sequential, nn.Sequential]:
    """The original networks, rebuilt as RLGym-PPO builds them (DiscreteFF without its softmax, "
    "ValueEstimator)."""

    def rebuild(layers: list[tuple[torch.Tensor, torch.Tensor]]) -> nn.Sequential:
        modules: list[nn.Module] = []
        for i, (weight, bias) in enumerate(layers):
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            modules += [linear] if i == len(layers) - 1 else [linear, nn.ReLU()]
        return nn.Sequential(*modules).eval()

    return rebuild(save.policy), rebuild(save.critic)


# The game-state layout NitroLearn's pool writes (nitro_core::env::Env::write_state): ball body (18), the
# team that scored, 34 pads (available, cooldown), then per car: team, body (18), boost, 7 flags, demo timer,
# flip time left, the last controls (8).
_HEAD, _CAR = 18 + 1 + 68, 1 + 18 + 10 + 8


def _physics(body: np.ndarray, invert: bool) -> PhysicsObject:
    flip = np.array([-1.0, -1.0, 1.0]) if invert else np.ones(3)
    obj = PhysicsObject(
        position=body[0:3] * flip, linear_velocity=body[12:15] * flip, angular_velocity=body[15:18] * flip
    )
    obj._rotation_mtx = np.stack([body[3:6] * flip, body[6:9] * flip, body[9:12] * flip], axis=1)
    obj._has_computed_rot_mtx = True
    return obj


def _gigalearn_slot(slot: int) -> int:
    """The game-state pad of RLGym's (RLBot's) pad slot `slot` (nitro_core::obs::gigalearn_pad)."""
    return {13: 14, 14: 13, 19: 20, 20: 19}.get(slot, slot)


def rlgym_state(row: np.ndarray, cars: int) -> tuple[GameState, list[np.ndarray]]:
    """RLGym's GameState and each car's previous action from one of NitroLearn's game states."""
    state = GameState()
    state.ball = _physics(row[0:18], False)
    state.inverted_ball = _physics(row[0:18], True)
    pads = row[19 : 19 + 68].reshape(34, 2)[:, 0]
    state.boost_pads = np.array([pads[_gigalearn_slot(i)] for i in range(34)], dtype=np.float32)
    state.inverted_boost_pads = state.boost_pads[::-1]
    previous = []
    for car in range(cars):
        at = _HEAD + car * _CAR
        player = PlayerData()
        player.car_id, player.team_num = car + 1, int(row[at])
        body = row[at + 1 : at + 19]
        player.car_data, player.inverted_car_data = _physics(body, False), _physics(body, True)
        boost, flags = row[at + 19], row[at + 20 : at + 27]
        player.boost_amount = float(boost) / 100
        player.on_ground, player.has_jump = bool(flags[0]), not bool(flags[1])
        player.has_flip, player.is_demoed = bool(flags[4]), bool(flags[5])
        state.players.append(player)
        previous.append(row[at + 29 : at + 37].astype(np.float64))
    return state, previous


@dataclass(frozen=True)
class Parity:
    states: int
    largest_obs_difference: float
    largest_prob_difference: float
    same_best_action: float
    largest_value_difference: float  # the critic's, relative to the value's size when above 1

    @property
    def passed(self) -> bool:
        return (
            self.largest_prob_difference <= PROB_TOLERANCE
            and self.same_best_action == 1.0
            and self.largest_value_difference <= VALUE_TOLERANCE
        )

    def line(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        return (
            f"Parity check {verdict} on {self.states} game states: RLGym's own observation code and the "
            "original "
            "networks against NitroLearn's; largest difference in the network's inputs "
            f"{self.largest_obs_difference:.2e}, "
            f"in any action's probability {self.largest_prob_difference:.2e} (allowed "
            f"{PROB_TOLERANCE:.0e}), same "
            f"most likely action in {self.same_best_action:.1%}; largest difference in the critic's value "
            f"{self.largest_value_difference:.2e} (allowed {VALUE_TOLERANCE:.0e})."
        )


def check(
    model: ActorCritic, save: RlgymSave, plan: RlgymPlan, obs: torch.Tensor, states: np.ndarray
) -> Parity:
    """NitroLearn's observation + input map + network against RLGym's observation code + standardisation + the
    original network, on the same game states (one row per car: `states` repeats each arena's state)."""
    mean_, std_, clip = standardisation(save, plan)
    mean, std = torch.as_tensor(mean_, dtype=torch.float64), torch.as_tensor(std_, dtype=torch.float64)
    reference_obs = []
    cars = plan.team_size * 2
    for row_index in range(0, len(states)):
        state, previous = rlgym_state(states[row_index], cars)
        me = row_index % cars
        if hasattr(plan.builder, "reset"):
            plan.builder.reset(state)
        reference_obs.append(
            np.asarray(plan.builder.build_obs(state.players[me], state, previous[me]), dtype=np.float64)
        )
    raw = torch.as_tensor(np.stack(reference_obs))
    standardised = (raw - mean) / std
    if clip:
        standardised = standardised.clamp(-clip, clip)
    policy, critic = reference_networks(save)
    with torch.no_grad():
        mapped = model.input_map(obs)
        ref_probs = torch.softmax(policy(standardised.float()), -1).clamp(MIN_PROB, 1)
        ref_values = critic(standardised.float()).flatten()
        mask = torch.ones(obs.shape[0], NUM_ACTIONS, dtype=torch.bool)
        probs, _ = nitro_probs(model, obs, mask)
        values = model(obs)[1].flatten()
    return Parity(
        int(obs.shape[0]),
        float((mapped.double() - standardised).abs().max()),
        float((probs.clamp(MIN_PROB, 1) - ref_probs).abs().max()),
        float((probs.argmax(-1) == ref_probs.argmax(-1)).float().mean()),
        float(((values - ref_values).abs() / ref_values.abs().clamp(min=1)).max()),
    )


def game_states(
    config: NitroConfig, model: ActorCritic, steps: int, seed: int = 0
) -> tuple[torch.Tensor, np.ndarray]:
    """Observations and, per car, its arena's whole game state, from real games played by the imported "
    "policy."""
    from nitro.envs import EnvPool

    small = dataclasses.replace(
        config,
        env=dataclasses.replace(config.env, num_arenas=4, groups=1, threads=1),
        run=dataclasses.replace(config.run, seed=seed),
    )
    pool = EnvPool(small, pin_memory=False)
    tensors = pool.groups[0]
    cars = config.env.team_size * 2
    generator = torch.Generator().manual_seed(seed)
    seen_obs, seen_states = [], []
    with torch.no_grad():
        for _ in range(steps):
            states = pool.game_states(0, small.env.num_arenas)
            seen_obs.append(tensors.obs.clone())
            seen_states.append(states.repeat_interleave(cars, dim=0).clone())
            probs, _ = nitro_probs(model, tensors.obs, tensors.mask)
            tensors.actions.copy_(torch.multinomial(probs, 1, generator=generator).flatten().int())
            pool.step(0)
            pool.wait(0)
    return torch.cat(seen_obs), torch.cat(seen_states).numpy()


def _adam_entry(entry: Any, param: torch.Tensor) -> bool:
    """`entry` is Adam's state for `param`: its step count and both moments, the parameter's shape."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("step"), (int, float, torch.Tensor))
        and all(
            isinstance(entry.get(k), torch.Tensor) and entry[k].shape == param.shape
            for k in ("exp_avg", "exp_avg_sq")
        )
    )


def learner_state(
    model: ActorCritic, config: NitroConfig, save: RlgymSave, plan: RlgymPlan
) -> dict[str, Any]:
    """The learner's state: the networks, Adam's moments and step counts as the save holds them, and the
    return spread the rewards were divided by."""
    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    learner = Learner(model, config.ppo, gamma, torch.device("cpu"), fp16=False)
    taken = 0
    for network, saved in ((model.policy, save.policy_optim), (model.value, save.critic_optim)):
        params = list(network.parameters())
        states = saved.get("state", {}) if isinstance(saved, dict) else {}
        if len(states) != len(params) or not all(_adam_entry(states.get(i), p) for i, p in enumerate(params)):
            continue
        for i, param in enumerate(params):
            entry = states[i]
            learner.optimizer.state[param] = {
                "step": torch.as_tensor(entry["step"], dtype=torch.float32).reshape(()),
                "exp_avg": entry["exp_avg"].float().clone(),
                "exp_avg_sq": entry["exp_avg_sq"].float().clone(),
            }
        taken += 1
    if taken == 2:
        plan.exact.append("Optimizer: the save's Adam running averages and step counts carried over.")
    else:
        plan.changed.append(
            "Optimizer: the save's Adam state was missing or did not fit; Adam starts afresh."
        )
    state: dict[str, Any] = learner.state_dict()
    if save.returns is not None and plan.settings["standardize_returns"] and save.returns[2] >= 2:
        mean, m2, count = save.returns
        variance = m2 / (count - 1)
        std = math.sqrt(variance) if variance > 0 else 1.0
        weight = max(save.steps, count)
        state["reward_scaler"] = {
            "count": torch.tensor(float(weight)),
            "mean": torch.tensor(mean),
            "m2": torch.tensor(std * std * weight),
        }
        plan.exact.append(f"Reward scaling: the returns' spread ({std:.4g}) carried over.")
    else:
        plan.changed.append(
            "Reward scaling: no return statistics in the save; NitroLearn measures them afresh."
        )
    return state


@dataclass
class Report:
    plan: RlgymPlan
    parity: Parity | None
    run: Path | None
    agent_steps: int

    def lines(self) -> list[str]:
        out = self.plan.lines()
        if self.parity is not None:
            out.append(self.parity.line())
        if self.run is not None:
            out.append(
                f"Done: '{self.run.name}' starts at {self.agent_steps:,} agent steps. Continue training "
                "with: "
                f"nitro resume {self.run.name}"
            )
        return out


def import_bot(
    checkpoint: Path,
    source_path: Path,
    name: str,
    runs: Path,
    *,
    meshes: Path | None = None,
    keep_action_delay: bool = False,
    check_steps: int = 200,
    say: Callable[[str], None] = print,
) -> Report:
    """Imports the RLGym save at `checkpoint` (with its project at `source_path`) as run `name` in `runs`;
    refuses before writing anything when a part cannot be brought over or the parity check fails."""
    if not name.strip() or name.startswith(".") or any(c in name for c in "/\\:"):
        raise RlgymImportError(
            f"'{name}' cannot be a bot's name: use a plain folder name, without / \\ : or a leading dot"
        )
    if (runs / name).exists():
        raise RlgymImportError(f"a run called '{name}' already exists in {runs}; choose another name")
    save = read_save(checkpoint)
    try:
        source = read_source(source_path)
    except SourceError as err:
        raise RlgymImportError(str(err)) from None
    plan = build_plan(save, source, keep_action_delay)
    if plan.refused:
        return Report(plan, None, None, 0)
    runs.mkdir(parents=True, exist_ok=True)
    text, config = checked_settings(plan.toml, runs, meshes)
    from nitro.envs import EnvPool

    obs_size = EnvPool(
        dataclasses.replace(config, env=dataclasses.replace(config.env, num_arenas=1, groups=1, threads=1)),
        pin_memory=False,
    )._core.obs_size
    model = build_model(config, plan, save, obs_size)
    if plan.refused:
        return Report(plan, None, None, 0)
    say(
        "Checking the imported bot against RLGym's observation code and the original networks on real game "
        "states..."
    )
    obs, states = game_states(config, model, check_steps)
    parity = check(model, save, plan, obs, states)
    if not parity.passed:
        plan.refused.append("The imported bot does not choose as the original does; nothing was written.")
        return Report(plan, parity, None, 0)
    learner = learner_state(model, config, save, plan)
    origin = {"from": str(save.folder), "framework": save.kind, "steps": save.steps}
    state = imported_state(learner, config, save.steps, save.iterations, origin)
    lines = Report(plan, parity, None, save.steps).lines()
    return Report(plan, parity, write_run(runs, name, text, config, state, lines), save.steps)
