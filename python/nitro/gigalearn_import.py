"""`nitro import-gigalearn`: brings a GigaLearn bot into NitroLearn so it trains on as the
same network, not a copy taught from it.

A GigaLearn save is a folder per step count holding each network as LibTorch's save of an `nn::Sequential`
(SHARED_HEAD.lt, POLICY.lt, CRITIC.lt: Linear, LayerNorm and activation layers; only the weights are stored),
each network's Adam state (*_OPTIM.lt) and RUNNING_STATS.json (step counts, return statistics). The layers'
sizes come from the weights; the activation, observation, actions, rewards and learning settings from the
project's C++ (`gigalearn_source`). What NitroLearn cannot do the same way is refused or listed, never
guessed. Before anything is written, the imported policy is checked against the GigaLearn networks on real
game states: the same action probabilities within float rounding (`check_parity`)."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nitro import __version__
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig
from nitro.gigalearn_source import Call, Source, Value
from nitro.gigalearn_source import read as read_source
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import discount_from_half_life, masked_log_probs
from nitro.runs import RunFolder, all_settings, fixed_settings

NUM_ACTIONS = 90  # GigaLearn's DefaultAction = NitroLearn's 90-action table, in the same order
# GigaLearn's masking and probability floor (PPOLearner::InferPolicyProbsFromModels).
MASKED_LOGIT = -1e10
MIN_PROB = 1e-11
# Largest difference in any action probability the parity check accepts (float32 rounding on wide networks).
PROB_TOLERANCE = 1e-4
# Largest difference in the value the check accepts, relative to the value where it is above 1.
VALUE_TOLERANCE = 1e-4

FILES = {  # role -> file names (GigaLearn's, then RLGymPPO_CPP's)
    "shared": ("SHARED_HEAD.lt",),
    "policy": ("POLICY.lt", "PPO_POLICY.lt"),
    "critic": ("CRITIC.lt", "PPO_CRITIC.lt"),
}
STATS_FILE = "RUNNING_STATS.json"

# GigaLearn's LearnerConfig / PPOLearnerConfig defaults (LearnerConfig.h, PPOLearnerConfig.h).
DEFAULTS: dict[str, Value] = {
    "tickSkip": 8,
    "actionDelay": 7,
    "numGames": 300,
    "randomSeed": -1,
    "standardizeObs": False,
    "minObsSTD": 0.1,
    "maxObsMeanRange": 3,
    "standardizeReturns": True,
    "trainAgainstOldVersions": False,
    "ppo.tsPerItr": 50_000,
    "ppo.batchSize": 50_000,
    "ppo.miniBatchSize": 0,
    "ppo.epochs": 2,
    "ppo.policyLR": 3e-4,
    "ppo.criticLR": 3e-4,
    "ppo.entropyScale": 0.018,
    "ppo.maskEntropy": False,
    "ppo.clipRange": 0.2,
    "ppo.policyTemperature": 1,
    "ppo.gaeLambda": 0.95,
    "ppo.gaeGamma": 0.99,
    "ppo.rewardClipRange": 10,
    "ppo.useHalfPrecision": False,
    "ppo.useGuidingPolicy": False,
    "ppo.deterministic": False,
    "ppo.maxEpisodeDuration": 120,
}
# Settings that only change how GigaLearn runs on the machine or logs: nothing to carry over.
NOT_LEARNING = (
    "deviceType",
    "renderMode",
    "renderTimeScale",
    "checkpointFolder",
    "tsPerSave",
    "checkpointsToKeep",
    "sendMetrics",
    "metricsProjectName",
    "metricsGroupName",
    "metricsRunName",
    "addRewardsToMetrics",
    "maxRewardSamples",
    "rewardSampleRandInterval",
    "maxReturnSamples",
    "maxObsSamples",
    "ppo.overbatching",
)
ACTIVATIONS = {"RELU": "relu", "LEAKY_RELU": "leaky_relu", "TANH": "tanh"}
# Rewards that GigaLearn's game event tracker pays (RocketSim's shot, save and goal detection), which
# NitroLearn's simulator does not have.
EVENT_TRACKER = ("PlayerGoalReward", "AssistReward", "ShotReward", "ShotPassReward", "SaveReward")


class GigaLearnImportError(ValueError):
    """The bot cannot be brought over; the message says why and what would be needed."""


# ---------------------------------------------------------------------------------------------------------
# The save


@dataclass(frozen=True)
class Network:
    """One GigaLearn network: its linear layers' (inputs, outputs) in order, whether each is followed by a
    LayerNorm, and the weights under GigaLearn's names ("0.weight", ...)."""

    role: str
    widths: tuple[tuple[int, int], ...]
    norm: bool
    weights: dict[str, torch.Tensor]
    linear_keys: tuple[str, ...]
    norm_keys: tuple[str, ...]  # per hidden layer: the LayerNorm's index, when there is one

    @property
    def inputs(self) -> int:
        return self.widths[0][0]

    @property
    def outputs(self) -> int:
        return self.widths[-1][1]


@dataclass
class Save:
    folder: Path
    shared: Network | None
    policy: Network
    critic: Network
    stats: dict[str, Any]
    optimizers: dict[str, Path]  # role -> its *_OPTIM.lt


def find_save(path: Path) -> Path:
    """The save folder: `path` itself when it holds POLICY.lt, else its (or its `checkpoints` folder's)
    numbered subfolder with the most steps."""
    for candidate in (path, path / "checkpoints"):
        if any((candidate / name).is_file() for name in FILES["policy"]):
            return candidate
        numbered = [p for p in candidate.glob("*") if p.is_dir() and p.name.isdigit()]
        numbered = [p for p in numbered if any((p / name).is_file() for name in FILES["policy"])]
        if numbered:
            return max(numbered, key=lambda p: int(p.name))
    raise GigaLearnImportError(
        f"{path} holds no GigaLearn save: expected POLICY.lt (or a checkpoints folder of numbered saves). "
        "Point at the folder your GigaLearn bot saves into, or at one numbered save in it."
    )


# LibTorch's saves are zip archives of a pickle (data.pkl) and the tensors' bytes (data/<key>). They are read
# here as data only: torch.jit.load would also compile and run the TorchScript code an archive carries, and a
# save from someone else may carry any.
_STORAGES = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}
_MAX_ARCHIVE_BYTES = 2**31  # far above any GigaLearn network or optimizer


class _Object(dict[str, Any]):
    """A TorchScript object in an archive: only its attributes, never its code."""

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("an object with its own way of loading")
        self.update(state)


@dataclass(frozen=True)
class _Storage:
    dtype: torch.dtype


def _rebuild_tensor(storage: torch.Tensor, offset: int, size: Any, stride: Any, *_: Any) -> torch.Tensor:
    shape, steps = tuple(size), tuple(stride)
    if not all(isinstance(n, int) and n >= 0 for n in (offset, *shape, *steps)) or len(shape) != len(steps):
        raise pickle.UnpicklingError("a tensor with a broken shape")
    if 0 not in shape and offset + sum((n - 1) * st for n, st in zip(shape, steps, strict=True)) >= len(
        storage
    ):
        raise pickle.UnpicklingError("a tensor larger than its data")
    return storage.as_strided(shape, steps, offset).clone()


class _ArchiveReader(pickle.Unpickler):
    def __init__(self, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(io.BytesIO(_member(archive, f"{prefix}data.pkl", 64 * 2**20)))
        self.archive, self.prefix, self.read_bytes = archive, prefix, 0

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name in _STORAGES:
            return _Storage(_STORAGES[name])
        if module == "__torch__" or module.startswith("__torch__."):
            return _Object
        raise pickle.UnpicklingError(f"{module}.{name}, which a GigaLearn save does not hold")

    def persistent_load(self, pid: Any) -> torch.Tensor:
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError("an unknown kind of stored data")
        _, kind, key, _, numel = pid
        if not isinstance(kind, _Storage) or not isinstance(key, str) or not key.isdigit():
            raise pickle.UnpicklingError("an unknown kind of stored data")
        if not isinstance(numel, int) or numel < 0:
            raise pickle.UnpicklingError("stored data of a broken size")
        size = numel * torch.empty((), dtype=kind.dtype).element_size()
        self.read_bytes += size
        if self.read_bytes > _MAX_ARCHIVE_BYTES:
            raise pickle.UnpicklingError("more data than any GigaLearn network")
        data = _member(self.archive, f"{self.prefix}data/{key}", size, exact=True)
        if not size:
            return torch.empty(0, dtype=kind.dtype)
        return torch.frombuffer(bytearray(data), dtype=kind.dtype)


def _member(archive: zipfile.ZipFile, name: str, limit: int, exact: bool = False) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise pickle.UnpicklingError(f"no {name}") from None
    if info.file_size > limit or (exact and info.file_size != limit):
        raise pickle.UnpicklingError(f"{name} is not the size it should be")
    return archive.read(info)


def _read_archive(file: Path) -> _Object:
    """A LibTorch save (torch::save, OutputArchive::save_to) as nested attributes, read without running it."""
    try:
        with zipfile.ZipFile(file) as archive:
            names = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith("/data.pkl")]
            if len(names) != 1:
                raise pickle.UnpicklingError("no data.pkl")
            prefix = names[0].removesuffix("data.pkl")
            order = f"{prefix}byteorder"
            if order in archive.namelist() and _member(archive, order, 16) != b"little":
                raise pickle.UnpicklingError("a byte order other than this PC's")
            root = _ArchiveReader(archive, prefix).load()
    except (OSError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError, ValueError) as err:
        raise GigaLearnImportError(f"{file.name} is not a save file LibTorch wrote ({err})") from None
    if not isinstance(root, _Object):
        raise GigaLearnImportError(f"{file.name} is not a GigaLearn save file LibTorch wrote")
    return root


def _tensors(node: _Object, prefix: str = "") -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in node.items():
        if isinstance(value, torch.Tensor):
            out[prefix + str(key)] = value
        elif isinstance(value, _Object):
            out.update(_tensors(value, f"{prefix}{key}."))
    return out


def _load_weights(file: Path) -> dict[str, torch.Tensor]:
    return {key: value.float() for key, value in _tensors(_read_archive(file)).items()}


def read_network(file: Path, role: str) -> Network:
    """A network's layers from its weights: a 2-D weight is a linear layer, a 1-D weight right after one is
    its LayerNorm; activations hold no weights."""
    weights = _load_weights(file)
    indices = sorted({int(key.split(".")[0]) for key in weights if key.split(".")[0].isdigit()})
    if not indices or any(not key.split(".")[0].isdigit() for key in weights):
        raise GigaLearnImportError(
            f"{file.name} is not a plain GigaLearn network (layers numbered 0, 1, ...)"
        )
    linears, norms, widths = [], [], []
    for index in indices:
        weight = weights.get(f"{index}.weight")
        if weight is None or f"{index}.bias" not in weights:
            raise GigaLearnImportError(
                f"{file.name}: layer {index} has no weight and bias; not a GigaLearn layer"
            )
        if weight.dim() == 2:
            linears.append(f"{index}")
            widths.append((int(weight.shape[1]), int(weight.shape[0])))
        elif weight.dim() == 1 and linears and int(linears[-1]) == index - 1:
            if weight.shape[0] != widths[-1][1]:
                raise GigaLearnImportError(f"{file.name}: LayerNorm {index} does not fit the layer before it")
            norms.append(f"{index}")
        else:
            raise GigaLearnImportError(
                f"{file.name}: layer {index} is neither a linear layer nor its LayerNorm"
            )
    hidden = len(linears) if role == "shared" else len(linears) - 1
    if norms and len(norms) != hidden:
        raise GigaLearnImportError(
            f"{file.name}: only some layers have a LayerNorm; NitroLearn normalises every hidden layer or "
            "none"
        )
    return Network(role, tuple(widths), bool(norms), weights, tuple(linears), tuple(norms))


def read_save(path: Path) -> Save:
    folder = find_save(path)

    def file(role: str) -> Path | None:
        return next((folder / name for name in FILES[role] if (folder / name).is_file()), None)

    policy_file, critic_file, shared_file = file("policy"), file("critic"), file("shared")
    if critic_file is None or policy_file is None:
        raise GigaLearnImportError(
            f"{folder} has no CRITIC.lt: GigaLearn saves the policy and critic together"
        )
    stats_path = folder / STATS_FILE
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    except (OSError, ValueError) as err:
        raise GigaLearnImportError(f"{STATS_FILE} in {folder} cannot be read: {err}") from None
    if not isinstance(stats, dict):
        raise GigaLearnImportError(f"{STATS_FILE} in {folder} is not GigaLearn's statistics")
    optimizers = {
        role: f.with_name(f.stem + "_OPTIM.lt")
        for role, f in (("shared", shared_file), ("policy", policy_file), ("critic", critic_file))
        if f is not None and f.with_name(f.stem + "_OPTIM.lt").is_file()
    }
    return Save(
        folder,
        read_network(shared_file, "shared") if shared_file else None,
        read_network(policy_file, "policy"),
        read_network(critic_file, "critic"),
        stats,
        optimizers,
    )


# ---------------------------------------------------------------------------------------------------------
# The plan: what each part becomes


@dataclass
class Plan:
    """Everything the import decided: the settings, how the weights change, and the report lines."""

    team_size: int
    observation: str
    activation: str
    settings: dict[str, Value]  # GigaLearn's settings, defaults filled in
    rewards: list[dict[str, Any]]  # NitroLearn [[rewards]] blocks
    setters: list[dict[str, Any]]
    # Per reward block, the GigaLearn reward it came from: class, arguments, weight, zero-sum (spirit, scale).
    reward_origins: list[dict[str, Any]] = field(default_factory=list)
    toml: str = ""
    temperature: float = 1.0
    steps: int = 0  # the agent steps GigaLearn had trained
    obs_room: int = 0  # env.obs_team_size: AdvancedObsPadderGGL's teamSize when above the game's
    obs_norm: tuple[torch.Tensor, torch.Tensor] | None = None  # (mean, std) folded into the first layer
    exact: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)  # carried over, but not exactly as GigaLearn did it
    left_out: list[str] = field(default_factory=list)  # not carried over at all
    to_port: list[str] = field(default_factory=list)  # the user's own C++ to port by hand
    refused: list[str] = field(default_factory=list)  # stops the import


# Observation sizes: own part + per other car, GigaLearn's built-in builders.
OBSERVATIONS = {
    "AdvancedObs": ("gigalearn_advanced", 51 + 29, 29),
    "DefaultObs": ("gigalearn_default", 51 + 19, 19),
}


def _team_size(obs_size: int, builder: str) -> int | None:
    _, own, other = OBSERVATIONS[builder]
    cars, rest = divmod(obs_size - own, other)
    return (cars + 1) // 2 if rest == 0 and cars % 2 == 1 and 1 <= (cars + 1) // 2 <= 3 else None


def plan_observation(save: Save, source: Source | None, plan: Plan) -> None:
    size = (save.shared or save.policy).inputs
    if source is not None and source.obs is not None:
        call = source.obs
        name = call.name
        if name == "DefaultObsPadded":
            players = call.args[0] if call.args else None
            if isinstance(players, int) and _team_size(size, "DefaultObs") == players and not call.args[1:]:
                plan.changed.append(
                    f"Observation: DefaultObsPadded({players}) with every place filled is DefaultObs with "
                    "the other players in a fixed order instead of shuffled each step: 'gigalearn_default'."
                )
                name = "DefaultObs"
            else:
                plan.refused.append(
                    f"Observation DefaultObsPadded({', '.join(map(str, call.args))}) (line {call.line}) pads "
                    "empty player places, which NitroLearn's copy of GigaLearn's observations does not do "
                    "yet. Needed: a padded 'gigalearn_default' in NitroLearn's core."
                )
                return
        if name == "AdvancedObsPadderGGL":
            _plan_padder(call, size, source, plan)
            return
        if name not in OBSERVATIONS:
            where = source.classes.get(name, "your project")
            plan.refused.append(
                f"Observation builder {name} (line {call.line}, {where}) is your own C++; it cannot be "
                "converted automatically. Needed: the same observation written into NitroLearn's core "
                "(crates/nitro_core/src/obs.rs) and the RLBot bot, number for number, then this import again."
            )
            return
        if call.args:
            plan.refused.append(
                f"Observation {name}({', '.join(map(str, call.args))}) (line {call.line}) has its own "
                "scales; NitroLearn's copy has GigaLearn's default ones only."
            )
            return
        builder = name
    else:
        found = [b for b in OBSERVATIONS if _team_size(size, b)]
        padded = [n for n in range(1, 5) if padder_size(n) == size]
        if padded:
            plan.refused.append(
                f"The networks read {size} numbers, which is also AdvancedObsPadderGGL({padded[0]}); give "
                "--source so the import knows which observation the bot was trained with."
            )
            return
        if not found:
            plan.refused.append(
                f"The networks read {size} numbers, which fits neither GigaLearn's AdvancedObs nor "
                "DefaultObs in 1v1, 2v2 or 3v3: a custom observation. Give --source so the import can say "
                "what it is."
            )
            return
        builder = found[0]
        plan.changed.append(
            f"Observation: no --source, so taken from the networks' input size ({size} = GigaLearn's "
            f"{builder})."
        )
    team = _team_size(size, builder)
    if team is None:
        plan.refused.append(f"The networks read {size} numbers, which no game size gives for {builder}")
        return
    if source is not None and source.players_per_team not in (None, team):
        plan.refused.append(
            f"The project sets playersPerTeam = {source.players_per_team}, but the networks' {size} inputs "
            f"are {builder} for {team} a side"
        )
        return
    plan.team_size = team
    plan.observation = OBSERVATIONS[builder][0]
    plan.exact.append(
        f"Observation: GigaLearn's {builder} -> '{plan.observation}' (the same numbers), {team}v{team}."
    )


def padder_size(places: int) -> int:
    """AdvancedObsPadderGGL's size: 51 shared numbers, 26 for the own car, 32 for each of the other places."""
    return 51 + 26 + 32 * (2 * places - 1)


def _plan_padder(call: Call, size: int, source: Source, plan: Plan) -> None:
    places = call.args[0] if call.args else 3  # the C++ default teamSize
    if not isinstance(places, int) or isinstance(places, bool) or not 1 <= places <= 4 or len(call.args) > 2:
        plan.refused.append(
            f"Observation {call.text} (line {call.line}): its teamSize must be a whole number from 1 to 4."
        )
        return
    if size != padder_size(places):
        plan.refused.append(
            f"Observation {call.text} (line {call.line}) gives {padder_size(places)} numbers, but the "
            "networks "
            f"read {size}: the save and the project do not belong together."
        )
        return
    team = source.players_per_team or places
    if source.players_per_team is None:
        plan.changed.append(
            f"Game size: the project's playersPerTeam was not found; {team}v{team}, the observation's "
            "teamSize."
        )
    if team > places:
        plan.refused.append(
            f"playersPerTeam = {team} is more than {call.text}'s teamSize {places}: it would leave players "
            "out."
        )
        return
    plan.team_size, plan.observation = team, "gigalearn_padder"
    plan.obs_room = places if places != team else 0
    room = f", room for {places} a side (env.obs_team_size)" if plan.obs_room else ""
    plan.exact.append(
        f"Observation: {call.text} -> 'gigalearn_padder' (the same numbers: AdvancedObsPadderGGL ported "
        f"number for number), {team}v{team}{room}."
    )


def plan_actions(source: Source | None, plan: Plan, save: Save) -> None:
    if save.policy.outputs != NUM_ACTIONS:
        plan.refused.append(
            f"The policy has {save.policy.outputs} outputs; GigaLearn's DefaultAction has 90. A bot with its "
            "own action parser cannot be brought over: NitroLearn's bots use DefaultAction's 90 actions."
        )
        return
    action = source.action if source is not None else None
    if action is not None and action.name != "DefaultAction":
        plan.refused.append(
            f"Action parser {action.name} (line {action.line}) is your own C++: a bot with its own action "
            "parser cannot be brought over (its 90 outputs alone do not say what they do)."
        )
        return
    plan.exact.append(
        "Actions: GigaLearn's DefaultAction -> NitroLearn's 90-action table, same order, masks 'gigalearn'."
    )
    plan.changed.append(
        "Action masks: GigaLearn also allows jumping whenever the car's body rests on the floor (contact "
        "data the "
        "RLBot game does not send); NitroLearn's 'gigalearn' rule counts only an upside-down car near the "
        "floor."
    )


def _activation(source: Source | None, given: str | None, plan: Plan, shared: bool) -> None:
    if source is None:
        if given is None:
            plan.refused.append(
                "Without --source the networks' activation cannot be known (GigaLearn saves only weights). "
                "Give --source <your GigaLearn project>, or --activation relu|leaky_relu|tanh if you are "
                "sure."
            )
            return
        plan.activation = given
        plan.changed.append(f"Activation: '{given}' as given with --activation (not read from the project).")
        return
    nets = ("policy", "critic", "sharedHead") if shared else ("policy", "critic")  # only those in the save
    kinds = {source.settings.get(f"ppo.{n}.activationType", "RELU") for n in nets}
    if len(kinds) > 1:
        plan.refused.append(
            f"The networks use different activations ({', '.join(sorted(kinds))}); NitroLearn has one for all"
        )
        return
    kind = kinds.pop()
    if kind not in ACTIVATIONS:
        plan.refused.append(
            f"Activation {kind}: NitroLearn's networks have relu, leaky_relu, gelu and tanh. Needed: "
            f"'{kind.lower()}' "
            "added to model.activation."
        )
        return
    plan.activation = ACTIVATIONS[kind]
    plan.exact.append(f"Activation: {kind} -> '{plan.activation}'.")


def plan_networks(save: Save, source: Source | None, activation: str | None, plan: Plan) -> None:
    _activation(source, activation, plan, save.shared is not None)
    nets = [n for n in (save.shared, save.policy, save.critic) if n is not None]
    if len({n.norm for n in nets}) > 1:
        plan.refused.append(
            "Some networks have LayerNorm and some not; NitroLearn's layer_norm is one setting."
        )
    if save.critic.outputs != 1:
        plan.refused.append(f"The critic has {save.critic.outputs} outputs; a GigaLearn critic has 1.")
    features = save.shared.outputs if save.shared else save.policy.inputs
    if save.policy.inputs != features or save.critic.inputs != features:
        plan.refused.append("The networks' sizes do not fit together (shared head, policy and critic).")
    if source is not None:
        for role in ("policy", "critic", "sharedHead"):
            kind = source.settings.get(f"ppo.{role}.optimType")
            if kind not in (None, "ADAM"):
                plan.changed.append(
                    f"Optimizer: {role} used {kind}; NitroLearn trains with Adam, its running averages "
                    "afresh."
                )


def _number(entry: Any) -> float | None:
    return float(entry) if isinstance(entry, (int, float)) and not isinstance(entry, bool) else None


def _reward_block(call: Call, source: Source) -> tuple[dict[str, Any] | None, str]:
    """A built-in GigaLearn reward as NitroLearn's reward (name and settings), or None and why not."""
    args = call.args

    def arg(i: int, default: Value) -> Value:
        return args[i] if len(args) > i else default

    simple = {
        "AirReward": "air",
        "FaceBallReward": "face_ball",
        "VelocityPlayerToBallReward": "velocity_player_to_ball",
        "TouchBallReward": "touch_ball",
        "PickupBoostReward": "pickup_boost",
        "BumpReward": "bump",
        "DemoReward": "demo",
        "TouchAccelReward": "gigalearn_touch_accel",
        "WavedashReward": "gigalearn_wavedash",
        "BumpedPenalty": "gigalearn_bumped_penalty",
        "DemoedPenalty": "gigalearn_demoed_penalty",
    }
    if call.name in simple and not args:
        return {"name": simple[call.name]}, ""
    if call.name == "StrongTouchReward" and len(args) <= 2:
        # GigaLearn's pays nothing below the minimum; the built-in strong_touch ramps up from it instead.
        return {
            "name": "gigalearn_strong_touch",
            "min_speed_kph": float(arg(0, 20)),
            "max_speed_kph": float(arg(1, 130)),
        }, ""
    if call.name == "SaveBoostReward" and len(args) <= 1:
        return {"name": "save_boost", "exponent": float(arg(0, 0.5))}, ""
    if call.name == "GoalReward" and len(args) <= 1:
        return {"name": "goal", "concede_scale": float(arg(0, -1))}, ""
    if call.name == "VelocityBallToGoalReward" and len(args) <= 1:
        if arg(0, False):
            return {"name": "gigalearn_velocity_ball_to_goal", "own_goal": True}, ""
        return {"name": "velocity_ball_to_goal"}, ""
    if call.name == "VelocityReward" and len(args) <= 1:
        return {"name": "gigalearn_velocity", "negative": bool(arg(0, False))}, ""
    if call.name == "SpeedReward" and not args:
        return {"name": "gigalearn_velocity", "negative": False}, ""
    if call.name in EVENT_TRACKER:
        return None, (
            f"{call.name} (line {call.line}) is paid by RocketSim's game event tracker (shots, saves, goals "
            "by "
            "player), which NitroLearn's simulator does not have. Needed: that tracker in NitroLearn's core."
        )
    where = source.classes.get(call.name.split("<")[0])
    known = KNOWN_OWN.get(_fingerprint(source.class_code.get(call.name, "")))
    if known is not None and known[0] == call.name:
        return known[1](call), ""
    if where:
        return (
            None,
            f"{call.text} (line {call.line}) is your own reward ({where}): port it as a NitroLearn Rust "
            "reward.",
        )
    return None, f"{call.text} (line {call.line}) is not a GigaLearn built-in reward NitroLearn knows."


def _fingerprint(code: str) -> str:
    """A class's code (comments out, whitespace collapsed, as `gigalearn_source` keeps it) as a hash."""
    return hashlib.sha256(code.encode()).hexdigest()


def _scale(call: Call) -> dict[str, Any]:
    scale = call.args[0] if call.args else 1.0
    return {"name": "speed_towards_opponent", "scale": float(scale)}


# Community rewards that NitroLearn has exactly, recognised by their code (not their name alone): the same
# formulas as the library rewards custom/library/kickoff_proximity.rs and speed_towards_opponent.rs. A changed
# version hashes differently and is listed to port by hand.
KNOWN_OWN: dict[str, tuple[str, Callable[[Call], dict[str, Any]]]] = {
    "1f130c2ff77f4aaaf6bd1e9a2cf9db36255f64219efdf8faaf8ce2a896be53dd": (
        "KickoffProximityReward",
        lambda call: {"name": "kickoff_proximity", "zero_within_200": False},
    ),
    "7e15cfde454fa41e6cad908aa4e3968de820f0649176da30152151f392f4fca4": (
        "SpeedTowardsOpponentReward",
        _scale,
    ),
}

# Rewards paid every step: NitroLearn scales them to the tick skip (per 8 ticks), GigaLearn does not.
EVERY_STEP = {
    "kickoff_proximity",
    "speed_towards_opponent",
    "air",
    "face_ball",
    "velocity_player_to_ball",
    "velocity_ball_to_goal",
    "save_boost",
    "gigalearn_velocity",
    "gigalearn_velocity_ball_to_goal",
}


def plan_rewards(source: Source, plan: Plan) -> None:
    tick_skip = int(plan.settings["tickSkip"])
    for entry in source.rewards:
        call, zero_sum = entry.reward, None
        if call.name == "ZeroSumReward":
            if not call.args or not isinstance(call.args[0], Call) or len(call.args) > 3:
                plan.to_port.append(f"{call.text} (line {call.line}): a ZeroSumReward the import cannot read")
                continue
            spirit, scale = (
                _number(call.args[1]) if len(call.args) > 1 else None,
                (_number(call.args[2]) if len(call.args) > 2 else 1.0),
            )
            if spirit is None or scale is None:
                plan.to_port.append(
                    f"{call.text} (line {call.line}): its team spirit or opponent scale is not a number"
                )
                continue
            zero_sum = {"team_spirit": spirit, "opponent_scale": scale}
            call = call.args[0]
        block, why = _reward_block(call, source)
        weight = _number(entry.weight)
        if block is None:
            plan.to_port.append(f"{why} Weight {entry.weight}.")
            continue
        if weight is None:
            plan.to_port.append(f"{call.text} (line {entry.line}): its weight could not be read")
            continue
        # NitroLearn scales every-step rewards to the tick skip (defined per 8 ticks); GigaLearn does not.
        scaled = weight * 8 / tick_skip if block["name"] in EVERY_STEP else weight
        plan.rewards.append({**block, "weight": scaled, **({"zero_sum": zero_sum} if zero_sum else {})})
        plan.reward_origins.append(
            {
                "class": call.name,
                "args": list(call.args),
                "weight": weight,
                "zero_sum": [zero_sum["team_spirit"], zero_sum["opponent_scale"]] if zero_sum else None,
            }
        )
        note = f"{entry.reward.text} x {weight:g} -> {block['name']}"
        if zero_sum:
            spirit, scale = zero_sum["team_spirit"], zero_sum["opponent_scale"]
            note += f" (zero-sum, team spirit {spirit:g}, opponents x{scale:g})"
        if scaled != weight:
            note += f" (weight {scaled:g}: NitroLearn scales every-step rewards to tick skip {tick_skip})"
        plan.exact.append(f"Reward: {note}.")
    if not plan.rewards:
        plan.refused.append("No reward could be carried over; a bot cannot train on without rewards.")


def plan_setters(source: Source | None, plan: Plan) -> None:
    call = source.setter if source is not None else None
    if call is None:
        plan.setters = [{"name": "kickoff", "weight": 1.0}]
        if source is not None:
            plan.left_out.append("State setter: not found in the project; kickoffs used.")
        return
    parts: list[tuple[Call, float]] = []
    if call.name == "CombinedState" and call.args and isinstance(call.args[0], list):
        for pair in call.args[0]:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and isinstance(pair[0], Call)
                and _number(pair[1]) is not None
            ):
                parts.append((pair[0], float(pair[1])))
    else:
        parts.append((call, 1.0))
    for setter, weight in parts:
        if setter.name == "KickoffState":
            plan.setters.append({"name": "kickoff", "weight": weight})
            plan.exact.append(f"Start: KickoffState -> kickoff (weight {weight:g}).")
        elif setter.name == "FuzzedKickoffState":
            plan.setters.append({"name": "kickoff", "weight": weight, "fuzz_uu": 0.1})
            plan.changed.append(
                f"Start: FuzzedKickoffState -> kickoff with fuzz_uu = 0.1 (weight {weight:g}): GigaLearn "
                "moves "
                "each car up to 0.1 uu per axis, NitroLearn up to 0.1 uu in one random direction."
            )
        elif setter.name == "RandomState":
            plan.setters.append({"name": "random", "weight": weight})
            plan.changed.append(
                f"Start: {setter.text} -> NitroLearn's 'random' start (weight {weight:g}): random ball and "
                "cars, "
                "drawn NitroLearn's way, not GigaLearn's."
            )
        else:
            where = source.classes.get(setter.name, "not a GigaLearn built-in") if source else ""
            plan.to_port.append(
                f"Start {setter.text} (line {setter.line}, {where}): add a matching NitroLearn state setter "
                "by "
                f"hand (weight {weight:g})."
            )
    if not plan.setters:
        plan.setters = [{"name": "kickoff", "weight": 1.0}]
        plan.changed.append("Start: none of the project's starts could be carried over; kickoffs used.")


def _terminals(source: Source | None, plan: Plan) -> float:
    """No-touch seconds (GigaLearn's ExampleMain's 10 without a source)."""
    no_touch = 10.0
    if source is None:
        plan.changed.append("Ends: no --source; a game restarts after 10 s without a touch and on a goal.")
        return no_touch
    seen_goal = False
    no_touch = 0.0
    for call in source.terminals:
        if call.name == "NoTouchCondition" and call.args and _number(call.args[0]) is not None:
            no_touch = float(call.args[0])
            plan.exact.append(f"Ends: NoTouchCondition({no_touch:g}) -> no_touch_seconds = {no_touch:g}.")
        elif call.name == "GoalScoreCondition":
            seen_goal = True
            plan.exact.append(
                "Ends: GoalScoreCondition -> a goal ends the episode (always so in NitroLearn)."
            )
        else:
            plan.to_port.append(
                f"End condition {call.text} (line {call.line}): add a matching NitroLearn ending by hand."
            )
    if not seen_goal:
        plan.changed.append(
            "Ends: the project has no GoalScoreCondition; NitroLearn always ends an episode on a goal."
        )
    return no_touch


def build_plan(
    save: Save, source: Source | None, activation: str | None = None, keep_action_delay: bool = False
) -> Plan:
    settings = {**DEFAULTS, **(source.settings if source else {})}
    plan = Plan(team_size=1, observation="", activation="relu", settings=settings, rewards=[], setters=[])
    plan.steps = int(save.stats.get("total_timesteps", 0))
    plan_observation(save, source, plan)
    plan_actions(source, plan, save)
    plan_networks(save, source, activation, plan)
    if source is None:
        plan.refused.append(
            "Without --source the rewards cannot be read, and a bot cannot train on without them. Give "
            "--source "
            "<your GigaLearn project folder> (the one with your main .cpp)."
        )
    else:
        plan_rewards(source, plan)
        for line in source.unread:
            if not any(f".{key} =" in line for key in NOT_LEARNING):  # logging names and the like
                plan.left_out.append(f"Not read from {source.file.name}, {line}")
    plan_setters(source, plan)
    no_touch = _terminals(source, plan)
    plan.toml = settings_toml(save, plan, no_touch, keep_action_delay)
    return plan


# ---------------------------------------------------------------------------------------------------------
# Settings


def _half_life(gamma: float, tick_skip: int) -> float:
    return tick_skip * math.log(0.5) / (120.0 * math.log(gamma))


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def settings_toml(save: Save, plan: Plan, no_touch: float, keep_action_delay: bool = False) -> str:
    """The new bot's settings file, each line saying where it came from; records what changed in `plan`."""
    s = plan.settings
    tick_skip, delay = int(s["tickSkip"]), int(s["actionDelay"])
    lines = [
        "# Imported from a GigaLearn bot with `nitro import-gigalearn`.",
        f"# From: {save.folder}",
        "# Settings marked 'GigaLearn' keep the original bot's values; change them like any NitroLearn "
        "setting.",
        "",
        "[run]",
    ]
    if int(s["randomSeed"]) >= 0:
        lines.append(f"seed = {int(s['randomSeed'])}  # GigaLearn randomSeed")
    lines += [
        "",
        "[env]",
        f"team_size = {plan.team_size}",
        *(
            [f"obs_team_size = {plan.obs_room}  # GigaLearn AdvancedObsPadderGGL teamSize"]
            if plan.obs_room
            else []
        ),
        f'observation = "{plan.observation}"  # GigaLearn exact copy',
    ]
    lines += [
        'action_masks = "gigalearn"  # GigaLearn DefaultAction masks',
        f"num_arenas = {int(s['numGames'])}  # GigaLearn numGames",
    ]
    lines.append(f"tick_skip = {tick_skip}  # GigaLearn tickSkip")
    if delay >= tick_skip:
        plan.changed.append(
            f"actionDelay {delay} is not below tickSkip {tick_skip}; NitroLearn needs it below: "
            f"{tick_skip - 1}."
        )
        delay = tick_skip - 1
    lowest = min(1, tick_skip - 1)  # the lowest delay a bot keeps up with in a real game
    if not keep_action_delay and delay != lowest:
        plan.changed.append(
            f"Action delay: GigaLearn's actionDelay {delay} set to {lowest}, the lowest a real game allows: "
            "the bot acts sooner after it sees than it was trained to. --keep-action-delay keeps its own."
        )
        lines.append(f"action_delay = {lowest}  # the lowest; GigaLearn actionDelay was {delay}")
    else:
        lines.append(f"action_delay = {delay}  # GigaLearn actionDelay")
    lines += ["", "[terminals]", f"no_touch_seconds = {no_touch:g}  # GigaLearn NoTouchCondition"]

    model = [save.shared.widths if save.shared else (), save.policy.widths[:-1], save.critic.widths[:-1]]
    sizes = [[out for _, out in widths] for widths in model]
    lines += ["", "[model]"]
    if sizes[0]:
        lines.append(f"shared = {_toml_value(sizes[0])}  # GigaLearn sharedHead")
    lines += [
        f"policy = {_toml_value(sizes[1])}  # GigaLearn policy",
        f"value = {_toml_value(sizes[2])}  # GigaLearn critic",
    ]
    lines.append(f"layer_norm = {_toml_value(save.policy.norm)}  # GigaLearn addLayerNorm")
    lines.append(f'activation = "{plan.activation}"  # GigaLearn activationType')
    lines.append('init = "pytorch"  # unused: the weights come from the GigaLearn bot')
    plan.exact.append(
        f"Networks: shared {sizes[0] or 'none'}, policy {sizes[1]}, critic {sizes[2]}, LayerNorm "
        f"{save.policy.norm}."
    )

    steps = int(s["ppo.tsPerItr"])
    batch = int(s["ppo.batchSize"]) or steps
    mini = int(s["ppo.miniBatchSize"]) or batch
    learning_steps = max(1, round(steps / batch))
    gamma = float(s["ppo.gaeGamma"])
    lines += [
        "",
        "[ppo]",
        f"steps_per_iteration = {steps}  # GigaLearn tsPerItr",
        f"epochs = {int(s['ppo.epochs'])}  # GigaLearn epochs",
    ]
    lines.append(f"minibatches = {learning_steps}  # GigaLearn: one learning step per batchSize ({batch})")
    if mini < batch:
        lines.append(f"microbatch = {mini}  # GigaLearn miniBatchSize (memory only, the same step)")
    if steps % batch:
        plan.changed.append(
            f"tsPerItr {steps} is not a whole number of batches of {batch}: {learning_steps} learning steps "
            "a pass."
        )
    lines += [
        f"policy_lr = {float(s['ppo.policyLR'])!r}  # GigaLearn policyLR",
        f"value_lr = {float(s['ppo.criticLR'])!r}  # GigaLearn criticLR",
        f"discount_half_life = {_half_life(gamma, tick_skip)!r}  # GigaLearn gaeGamma {gamma:g} at tick "
        f"skip {tick_skip}",
        f"gae_lambda = {float(s['ppo.gaeLambda'])!r}  # GigaLearn gaeLambda",
        f"clip_range = {float(s['ppo.clipRange'])!r}  # GigaLearn clipRange",
        f"entropy = {float(s['ppo.entropyScale'])!r}  # GigaLearn entropyScale",
        f'entropy_normalization = "{"allowed" if s["ppo.maskEntropy"] else "all"}"  # GigaLearn maskEntropy',
        f"reward_clip = {float(s['ppo.rewardClipRange'])!r}  # GigaLearn rewardClipRange",
        "popart = false  # GigaLearn learns the scaled returns directly",
        "normalize_advantages = false  # GigaLearn does not normalise advantages",
        "adam_eps = 1e-8  # GigaLearn uses PyTorch's default",
        "max_grad_norm = 0.5  # GigaLearn clips each network at 0.5",
        "graphed_steps = false  # the plain learning step, as GigaLearn's",
    ]
    plan.exact.append(
        f"Learning: {steps} steps an iteration, {int(s['ppo.epochs'])} epochs, {learning_steps} learning "
        "steps a pass, "
        f"rates {float(s['ppo.policyLR']):g}/{float(s['ppo.criticLR']):g}, gamma {gamma:g}, lambda "
        f"{float(s['ppo.gaeLambda']):g}, clip {float(s['ppo.clipRange']):g}, entropy "
        f"{float(s['ppo.entropyScale']):g}, "
        "no advantage normalisation, rewards divided by the returns' spread and clipped as GigaLearn does."
    )
    if save.shared is not None and float(s["ppo.policyLR"]) != float(s["ppo.criticLR"]):
        plan.changed.append(
            "Shared head's learning rate: GigaLearn uses the smaller of policyLR and criticLR, NitroLearn "
            "the policy's."
        )
    _carry_over_notes(plan)

    for block in plan.rewards:
        lines += ["", "[[rewards]]"] + [f"{k} = {_toml_value(v)}" for k, v in block.items()]
    for block in plan.setters:
        lines += ["", "[[state_setters]]"] + [f"{k} = {_toml_value(v)}" for k, v in block.items()]
    return "\n".join(lines) + "\n"


def _carry_over_notes(plan: Plan) -> None:
    s = plan.settings
    temperature = float(s["ppo.policyTemperature"])
    if temperature != 1:
        plan.temperature = temperature
        plan.changed.append(
            f"policyTemperature {temperature:g}: folded into the policy's last layer (the same choices); "
            "from here "
            "on it learns as a temperature of 1."
        )
    if not s["standardizeReturns"]:
        plan.changed.append(
            "standardizeReturns = false: NitroLearn always divides rewards by the returns' spread, so the "
            "critic's "
            "values change scale as it learns on."
        )
    if s["ppo.useHalfPrecision"]:
        plan.left_out.append(
            "useHalfPrecision (GigaLearn: inference only): set run.precision yourself if wanted."
        )
    if s["ppo.useGuidingPolicy"]:
        plan.left_out.append(
            "useGuidingPolicy: NitroLearn has no guiding policy; start_from.keep_close is the nearest."
        )
    if s["trainAgainstOldVersions"]:
        plan.left_out.append(
            "trainAgainstOldVersions: GigaLearn's past versions are not imported; turn on NitroLearn's "
            "[selfplay] "
            "to train against its own past versions from here."
        )
    plan.left_out.append(
        f"maxEpisodeDuration ({float(s['ppo.maxEpisodeDuration']):g} s): GigaLearn only splits long "
        "episodes for "
        "learning; NitroLearn has no such split."
    )
    for key in NOT_LEARNING:
        if key in s and key not in DEFAULTS:
            plan.left_out.append(f"{key}: how GigaLearn runs or logs, nothing to carry over.")


# ---------------------------------------------------------------------------------------------------------
# Weights


def _fold_obs_norm(
    linear_w: torch.Tensor, linear_b: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """W((x - mean) / std) + b as W'x + b'."""
    w = linear_w / std
    return w, linear_b - w @ mean


def obs_normalisation(save: Save, plan: Plan) -> tuple[torch.Tensor, torch.Tensor] | None:
    """GigaLearn's observation standardisation as it stands in the save (standardizeObs), or None."""
    if not plan.settings["standardizeObs"]:
        return None
    stat = save.stats.get("obs_stat")
    if not stat:
        plan.refused.append("standardizeObs is on but the save has no obs_stat in RUNNING_STATS.json")
        return None
    # GigaLearn writes "means"/"vars" (BatchedWelfordStat::ToJSON); its reader looks for "mean"/"var".
    means = torch.tensor(stat.get("means", stat.get("mean")), dtype=torch.float64)
    count = int(stat["count"])
    variances = torch.tensor(stat.get("vars", stat.get("var")), dtype=torch.float64)
    if count < 2:
        std = torch.ones_like(means)
    else:
        var = variances / (count - 1)
        std = torch.where(var == 0, torch.ones_like(var), var).sqrt()
    limit = float(plan.settings["maxObsMeanRange"])
    mean = means.clamp(-limit, limit)
    std = std.clamp_min(float(plan.settings["minObsSTD"]))
    plan.changed.append(
        "standardizeObs: the save's observation statistics are folded into the first layer (the same "
        "outputs); "
        "from here on they stay fixed instead of following the games."
    )
    return mean.float(), std.float()


def nitro_weights(save: Save, plan: Plan) -> dict[str, torch.Tensor]:
    """The GigaLearn weights under NitroLearn's names: each hidden layer is Linear, (LayerNorm), activation in
    both, so GigaLearn's layer i lands in the same place; obs standardisation and temperature folded in."""
    out: dict[str, torch.Tensor] = {}
    stride = 3 if save.policy.norm else 2
    for prefix, net in (("shared", save.shared), ("policy", save.policy), ("value", save.critic)):
        if net is None:
            continue
        for layer, key in enumerate(net.linear_keys):
            w, b = net.weights[f"{key}.weight"], net.weights[f"{key}.bias"]
            if layer == 0 and plan.obs_norm is not None and (prefix == "shared" or save.shared is None):
                w, b = _fold_obs_norm(w, b, *plan.obs_norm)
            last = prefix != "shared" and layer == len(net.linear_keys) - 1
            if last and prefix == "policy" and plan.temperature != 1:
                w, b = w / plan.temperature, b / plan.temperature
            out[f"{prefix}.{layer * stride}.weight"], out[f"{prefix}.{layer * stride}.bias"] = w, b
        for layer, key in enumerate(net.norm_keys):
            out[f"{prefix}.{layer * stride + 1}.weight"] = net.weights[f"{key}.weight"]
            out[f"{prefix}.{layer * stride + 1}.bias"] = net.weights[f"{key}.bias"]
    return out


_ACT: dict[str, Callable[[], nn.Module]] = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


def gigalearn_networks(save: Save, activation: str) -> dict[str, nn.Sequential]:
    """GigaLearn's networks rebuilt layer for layer as GigaLearn builds them (Models.cpp), with its weights:
    the reference the parity check holds NitroLearn's networks to."""
    nets = {}
    for net in (save.shared, save.policy, save.critic):
        if net is None:
            continue
        layers: list[nn.Module] = []
        hidden = len(net.widths) if net.role == "shared" else len(net.widths) - 1
        for i, (inputs, outputs) in enumerate(net.widths):
            layers.append(nn.Linear(inputs, outputs))
            if i < hidden:
                if net.norm:
                    layers.append(nn.LayerNorm(outputs))
                layers.append(_ACT[activation]())
        seq = nn.Sequential(*layers)
        seq.load_state_dict(net.weights, strict=True)
        nets[net.role] = seq.eval()
    return nets


@torch.no_grad()
def gigalearn_probs(
    nets: dict[str, nn.Sequential],
    obs: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    obs_norm: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GigaLearn's action probabilities and values for raw observations (PPOLearner's
    InferPolicyProbsFromModels and InferCritic, with Learner's observation standardisation)."""
    if obs_norm is not None:
        obs = (obs - obs_norm[0]) / obs_norm[1]
    features = nets["shared"](obs) if "shared" in nets else obs
    logits = nets["policy"](features) / temperature
    probs = torch.softmax(logits + MASKED_LOGIT * (~mask).float(), -1).clamp(MIN_PROB, 1)
    return probs, nets["critic"](features).flatten()


@torch.no_grad()
def nitro_probs(
    model: ActorCritic, obs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NitroLearn's action probabilities (the sampler's masked softmax) and values."""
    logits, values = model(obs)
    return masked_log_probs(logits, mask).exp(), values


def build_model(config: NitroConfig, plan: Plan, save: Save) -> ActorCritic:
    model = ActorCritic((save.shared or save.policy).inputs, NUM_ACTIONS, config.model)
    try:
        model.load_state_dict(nitro_weights(save, plan), strict=True)
    except RuntimeError as err:
        raise GigaLearnImportError(f"the GigaLearn weights do not fit NitroLearn's networks: {err}") from None
    return model.eval()


@dataclass(frozen=True)
class Parity:
    states: int
    largest_prob_difference: float
    largest_value_difference: float  # relative to the value where it is above 1
    same_best_action: float  # share of states where both pick the same most likely action

    @property
    def passed(self) -> bool:
        return (
            self.largest_prob_difference <= PROB_TOLERANCE
            and self.largest_value_difference <= VALUE_TOLERANCE
            and self.same_best_action == 1.0
        )

    def line(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        return (
            f"Parity check {verdict} on {self.states} game states: largest difference in any action's "
            "probability "
            f"{self.largest_prob_difference:.2e} (allowed {PROB_TOLERANCE:.0e}), same most likely action in "
            f"{self.same_best_action:.1%}, largest value difference {self.largest_value_difference:.2e} "
            f"(allowed {VALUE_TOLERANCE:.0e})."
        )


def compare(model: ActorCritic, save: Save, plan: Plan, obs: torch.Tensor, mask: torch.Tensor) -> Parity:
    nets = gigalearn_networks(save, plan.activation)
    ref_probs, ref_values = gigalearn_probs(nets, obs, mask, plan.temperature, plan.obs_norm)
    probs, values = nitro_probs(model, obs, mask)
    probs = torch.where(mask, probs, torch.full_like(probs, MIN_PROB))  # GigaLearn's floor on masked actions
    return Parity(
        int(obs.shape[0]),
        float((probs - ref_probs).abs().max()),
        float(((values - ref_values).abs() / ref_values.abs().clamp_min(1)).max()),
        float((probs.argmax(-1) == ref_probs.argmax(-1)).float().mean()),
    )


def game_states(
    config: NitroConfig, model: ActorCritic, steps: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Observations and masks from real games played by the imported policy (a few arenas, `steps` steps)."""
    from nitro.envs import EnvPool

    small = dataclasses.replace(
        config,
        env=dataclasses.replace(config.env, num_arenas=4, groups=1, threads=1),
        run=dataclasses.replace(config.run, seed=seed),
    )
    pool = EnvPool(small, pin_memory=False)
    tensors = pool.groups[0]
    generator = torch.Generator().manual_seed(seed)
    seen_obs, seen_mask = [], []
    with torch.no_grad():
        for _ in range(steps):
            seen_obs.append(tensors.obs.clone())
            seen_mask.append(tensors.mask.clone())
            probs, _ = nitro_probs(model, tensors.obs, tensors.mask)
            tensors.actions.copy_(torch.multinomial(probs, 1, generator=generator).flatten().int())
            pool.step(0)
            pool.wait(0)
    return torch.cat(seen_obs), torch.cat(seen_mask)


# ---------------------------------------------------------------------------------------------------------
# The optimizer and the run


def _count(value: Any) -> int | None:
    """A count as LibTorch's OutputArchive keeps it (a one-number integer tensor), or None."""
    if isinstance(value, torch.Tensor) and value.numel() == 1 and not value.is_floating_point():
        return int(value.item())
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _adam_state(file: Path) -> list[dict[str, torch.Tensor]] | None:
    """Per parameter in order, Adam's step count and moments from a LibTorch optimizer save (None if not "
    "Adam)."""
    archive = _read_archive(file)
    groups, states = archive.get("param_groups"), archive.get("state")
    if not isinstance(groups, _Object) or not isinstance(states, _Object):
        return None
    group = groups.get("param_groups/0")
    if _count(groups.get("param_groups/size")) != 1 or not isinstance(group, _Object):
        return None
    size = _count(group.get("params/size"))
    out = []
    for i in range(size or 0):
        state = states.get(str(group.get(f"params/{i}")))
        if not isinstance(state, _Object):
            return None  # a parameter without state: never stepped
        step, avg, avg_sq = (state.get(name) for name in ("step", "exp_avg", "exp_avg_sq"))
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            step = step.item()
        if not isinstance(step, (int, float)) or not isinstance(avg, torch.Tensor):
            return None
        if not isinstance(avg_sq, torch.Tensor):
            return None
        out.append({"step": torch.tensor(float(step)), "exp_avg": avg.float(), "exp_avg_sq": avg_sq.float()})
    return out or None


def learner_state(model: ActorCritic, config: NitroConfig, save: Save, plan: Plan) -> dict[str, Any]:
    """NitroLearn's learner state: the imported networks, Adam's moments where GigaLearn's Adam state fits
    unchanged (not where a weight was folded), and the return spread GigaLearn's rewards were divided by."""
    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    learner = Learner(model, config.ppo, gamma, torch.device("cpu"), fp16=False)
    networks = {"shared": model.shared, "policy": model.policy, "critic": model.value}
    folded = plan.obs_norm is not None or plan.temperature != 1
    taken = 0
    for role, file in save.optimizers.items():
        moments = _adam_state(file)
        params = list(networks[role].parameters())
        if moments is None or len(moments) != len(params) or folded:
            continue
        if any(m["exp_avg"].shape != p.shape for m, p in zip(moments, params, strict=True)):
            continue
        for param, moment in zip(params, moments, strict=True):
            learner.optimizer.state[param] = moment
        taken += 1
    if folded:
        plan.changed.append("Adam's running averages start afresh (folded weights no longer match them).")
    elif taken == len(networks) - (save.shared is None):
        plan.exact.append("Optimizer: GigaLearn's Adam running averages and step counts carried over.")
    else:
        plan.changed.append(
            "Optimizer: GigaLearn's Adam state was missing or not Adam for some networks; those start afresh."
        )
    state: dict[str, Any] = learner.state_dict()
    stat = save.stats.get("return_stat")
    if plan.settings["standardizeReturns"] and stat and int(stat["count"]) >= 2:
        count = int(stat["count"])
        variance = float(stat["var"]) / (count - 1)
        std = math.sqrt(variance) if variance > 0 else 1.0
        # NitroLearn's spread is m2 / count over every agent step's return; given the weight of the steps
        # already trained so the scale holds as the run goes on.
        weight = max(int(save.stats.get("total_timesteps", 0)), count)
        state["reward_scaler"] = {
            "count": torch.tensor(float(weight)),
            "mean": torch.tensor(float(stat["mean"])),
            "m2": torch.tensor(std * std * weight),
        }
        plan.exact.append(f"Reward scaling: GigaLearn's returns spread ({std:.4g}) carried over.")
    else:
        plan.changed.append(
            "Reward scaling: no return statistics in the save; NitroLearn measures them afresh."
        )
    return state


@dataclass
class Report:
    plan: Plan
    parity: Parity | None
    run: Path | None
    agent_steps: int

    def lines(self) -> list[str]:
        plan = self.plan
        out = []
        sections = (
            ("Carried over exactly", plan.exact),
            ("Carried over, not exactly the same", plan.changed),
            ("Not carried over", plan.left_out),
            ("Your own C++ to port by hand (the bot was not changed for these)", plan.to_port),
            ("Cannot import", plan.refused),
        )
        for title, items in sections:
            if items:
                out.append(f"{title}:")
                out += [f"  - {item}" for item in items]
        if self.parity is not None:
            out.append(self.parity.line())
        if self.run is not None:
            out.append(
                f"Done: '{self.run.name}' starts at {self.agent_steps:,} agent steps. Continue training "
                "with: "
                f"nitro resume {self.run.name}"
            )
        return out


def _read_and_plan(
    checkpoint: Path, source_path: Path | None, activation: str | None, keep_action_delay: bool = False
) -> tuple[Save, Plan]:
    save = read_save(checkpoint)
    try:
        source = read_source(source_path) if source_path is not None else None
    except ValueError as err:
        raise GigaLearnImportError(f"the project's C++ cannot be read: {err}") from None
    plan = build_plan(save, source, activation, keep_action_delay)
    plan.obs_norm = obs_normalisation(save, plan)
    return save, plan


def plan_import(checkpoint: Path, source_path: Path | None, activation: str | None = None) -> Plan:
    """What `import_bot` would bring over, read and decided but not checked on game states or written."""
    return _read_and_plan(checkpoint, source_path, activation)[1]


def checked_name(name: str, runs: Path) -> None:
    """Refuses a name that is not one folder name, or a run that exists already."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise GigaLearnImportError(f"'{name}' cannot name a bot: use one folder name, such as my-bot")
    if (runs / name).exists():
        raise GigaLearnImportError(f"a run called '{name}' already exists in {runs}; choose another name")


def checked_settings(toml: str, runs: Path, meshes: Path | None) -> tuple[str, NitroConfig]:
    """The new bot's whole settings file and its settings, checked. Without `meshes` the bot uses
    NitroLearn's own arena meshes, as every bot does (nothing written)."""
    paths = f"[paths]\ncollision_meshes = {json.dumps(meshes.resolve().as_posix())}\n\n" if meshes else ""
    text = paths + toml
    try:
        return text, parse_toml(text, NitroConfig, file=Path("settings.toml"), base_dir=runs.resolve())
    except ConfigError as err:
        raise GigaLearnImportError(
            f"the settings made for this bot do not check out (please report this): {err}"
        ) from None


def imported_state(
    learner: dict[str, Any], config: NitroConfig, steps: int, iterations: int, origin: dict[str, Any]
) -> dict[str, Any]:
    """A NitroLearn save of an imported bot at `steps` agent steps, as `nitro resume` reads it."""
    return {
        "learner": learner,
        "agent_steps": steps,
        "iterations": iterations,
        "checkpoint": 0,
        "device": "cpu",
        "fp16": False,
        "rng_cpu": torch.get_rng_state(),
        "settings": fixed_settings(config),
        "all_settings": all_settings(config),
        "imported": origin,
        "nitro_version": __version__,
    }


def write_run(
    runs: Path, name: str, text: str, config: NitroConfig, state: dict[str, Any], report: list[str]
) -> Path:
    """Writes the imported bot as run `name`: its settings, its one save and imported.txt (the report); built
    under a temporary name and renamed last, so a failure leaves nothing behind."""
    final = runs / name
    partial = runs / f".{name}.import-partial"
    settings_file = runs / f".{name}.import.toml"
    shutil.rmtree(partial, ignore_errors=True)
    try:
        settings_file.write_text(text, encoding="utf-8")
        folder = RunFolder.create(runs, partial.name, settings_file)
        settings_file.unlink()
        steps = int(state["agent_steps"])
        folder.save_checkpoint(state, steps, config.run.keep_checkpoints, config.run.milestone_steps)
        (folder.root / "imported.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        folder.load_settings(state)  # the settings and the save agree
        os.replace(partial, final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        settings_file.unlink(missing_ok=True)
        raise
    return final


def import_bot(
    checkpoint: Path,
    source_path: Path | None,
    name: str,
    runs: Path,
    *,
    activation: str | None = None,
    meshes: Path | None = None,
    keep_action_delay: bool = False,
    check_steps: int = 300,
    say: Callable[[str], None] = print,
) -> Report:
    """Imports the GigaLearn save at `checkpoint` as run `name` in `runs`; refuses (GigaLearnImportError)
    before writing anything when a part cannot be brought over or the parity check fails."""
    checked_name(name, runs)
    save, plan = _read_and_plan(checkpoint, source_path, activation, keep_action_delay)
    if plan.refused:
        return Report(plan, None, None, 0)
    runs.mkdir(parents=True, exist_ok=True)
    text, config = checked_settings(plan.toml, runs, meshes)
    model = build_model(config, plan, save)
    say("Checking the imported bot against the GigaLearn networks on real game states...")
    obs, mask = game_states(config, model, check_steps)
    parity = compare(model, save, plan, obs, mask)
    if not parity.passed:
        plan.refused.append(
            "The imported policy does not choose as the GigaLearn one does; nothing was written."
        )
        return Report(plan, parity, None, 0)
    steps = int(save.stats.get("total_timesteps", 0))
    origin = {"from": str(save.folder), "gigalearn_steps": steps}
    learner = learner_state(model, config, save, plan)
    state = imported_state(learner, config, steps, int(save.stats.get("total_iterations", 0)), origin)
    report = Report(plan, parity, None, steps).lines()
    return Report(plan, parity, write_run(runs, name, text, config, state, report), steps)
