"""`nitro surgery`: make a run's networks wider without changing what the bot does.

Widening by a whole factor K copies every hidden unit K times. The weights that read a copied unit are
shared out unequally among its copies (a quarter and three quarters of the original for K = 2, see
`copy_shares`), adding up to the original weight, so every layer computes what it did before, LayerNorm
included (the copies have the same mean and spread as the original). The widened bot therefore plays as the
old one did, to within the computer's rounding (checked before anything is written). The unequal shares
make the copies learn apart within the first few updates: equal shares would give every copy the same
gradient, and the copies would stay identical for good, K² times the work for nothing new. Adam's running
averages start afresh; the learner's other state (value normalisation, reward scale, loss scale, learning
rates) carries over.

The result is a new run folder (the old one is left as it was): the settings with the new widths, the
widened save, the metrics so far, and the self-play pool with every past version widened the same way, so
`nitro resume <new run>` continues. Only widening is offered: adding inputs or actions, or deeper networks,
cannot be done with identical outputs under LayerNorm and ReLU.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nitro import __version__
from nitro.config import NitroConfig
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import discount_from_half_life
from nitro.runs import UNREADABLE, RunFolder, all_settings, fixed_settings, resume_command

# Random observations the old and the new network are compared on after a widening, in FP32 on the CPU.
# The largest difference allowed is this share of the largest output (logit or value), or this much when
# every output is below 1. FP32 rounding gives about 1e-6 of it; an FP16 run differs by more (about 1e-3),
# which is FP16's own rounding and makes no difference to play.
CHECK_OBSERVATIONS = 4096
CHECK_TOLERANCE = 1e-4
# How unequal the copies' shares of a unit's outgoing weights are: (1 + SPREAD * d) / K, d from -1 to 1.
SPREAD = 0.5


class SurgeryError(Exception):
    """A plain-English reason the surgery cannot be done."""


def widened_config(model: ModelConfig, factor: int) -> ModelConfig:
    def scale(widths: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(width * factor for width in widths)

    return dataclasses.replace(
        model, shared=scale(model.shared), policy=scale(model.policy), value=scale(model.value)
    )


def copy_shares(factor: int) -> torch.Tensor:
    """Each copy's share of the weights that read a copied unit: unequal, adding up to 1 ([0.25, 0.75] for
    2). With equal shares every copy gets the same gradient and the copies never part; unequal ones, which
    Adam's (and Muon's) steps do not keep in proportion, set them learning apart within a few updates."""
    return (1 + SPREAD * torch.linspace(-1.0, 1.0, factor, dtype=torch.float64)) / factor


def _linears_and_norms(head: nn.Sequential) -> list[nn.Module]:
    return [m for m in head if isinstance(m, nn.Linear | nn.LayerNorm)]


@torch.no_grad()
def widen_model(old: ActorCritic, factor: int) -> ActorCritic:
    """A network `factor` times as wide in every hidden layer that computes the same function as `old`, its
    copies of a unit reading on with unequal shares (`copy_shares`) so they learn apart."""
    if factor < 2:
        raise SurgeryError("the widening factor must be a whole number of 2 or more")
    if old.config.uses_attention:
        raise SurgeryError("widening works on plain networks; this run's networks have an attention encoder")
    obs_size = next(m for m in [*old.shared, *old.policy] if isinstance(m, nn.Linear)).in_features
    output = old.policy[-1]
    assert isinstance(output, nn.Linear)
    num_actions = output.out_features
    new = ActorCritic(obs_size, num_actions, widened_config(old.config, factor)).to(
        next(old.parameters()).device
    )
    trunk_out_duplicated = len(old.shared) > 0
    heads = [(old.shared, new.shared, False), (old.policy, new.policy, trunk_out_duplicated)]
    heads.append((old.value, new.value, trunk_out_duplicated))
    for old_head, new_head, input_duplicated in heads:
        old_layers, new_layers = _linears_and_norms(old_head), _linears_and_norms(new_head)
        duplicated_in = input_duplicated
        for i, (o, n) in enumerate(zip(old_layers, new_layers, strict=True)):
            last = i == len(old_layers) - 1 and isinstance(o, nn.Linear)
            if isinstance(o, nn.Linear):
                assert isinstance(n, nn.Linear)
                if duplicated_in:  # column j * factor + c reads copy c of unit j: its share of the weight
                    shares = copy_shares(factor).repeat(o.in_features).to(o.weight.dtype)
                    weight = o.weight.repeat_interleave(factor, dim=1) * shares.to(o.weight.device)
                else:
                    weight = o.weight
                if last and old_head is not old.shared:
                    n.weight.copy_(weight)  # the output layer keeps its outputs
                    n.bias.copy_(o.bias)
                else:
                    n.weight.copy_(weight.repeat_interleave(factor, dim=0))
                    n.bias.copy_(o.bias.repeat_interleave(factor, dim=0))
                    duplicated_in = True
            else:
                assert isinstance(o, nn.LayerNorm) and isinstance(n, nn.LayerNorm)
                n.weight.copy_(o.weight.repeat_interleave(factor, dim=0))
                n.bias.copy_(o.bias.repeat_interleave(factor, dim=0))
    return new


@torch.no_grad()
def compare(a: ActorCritic, b: ActorCritic, obs_size: int, seed: int = 0) -> tuple[float, float]:
    """The largest difference in logits or values between `a` and `b` on random observations, and the
    largest output of `a` (the scale the difference is judged against)."""
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(CHECK_OBSERVATIONS, obs_size, generator=generator).to(next(a.parameters()).device)
    logits_a, values_a = a(obs)
    logits_b, values_b = b(obs)
    difference = max((logits_a - logits_b).abs().max().item(), (values_a - values_b).abs().max().item())
    scale = max(logits_a.abs().max().item(), values_a.abs().max().item())
    return float(difference), float(scale)


def allowed(scale: float) -> float:
    """The largest difference the check accepts for outputs as large as `scale`."""
    return CHECK_TOLERANCE * max(1.0, scale)


def widen_settings_text(text: str, model: ModelConfig, factor: int) -> str:
    """The settings file's text with the `[model]` widths multiplied by `factor` (comments kept; missing
    keys or a missing section written out, so the new widths are stated even where the old relied on
    defaults)."""
    wide = widened_config(model, factor)
    values = {"shared": wide.shared, "policy": wide.policy, "value": wide.value}
    lines = text.splitlines()
    header = next(
        (i for i, line in enumerate(lines) if re.match(r"^\s*\[\s*model\s*\]\s*(#.*)?$", line)), None
    )
    if header is None:
        lines += ["", "[model]"]
        header = len(lines) - 1
    end = next((i for i in range(header + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines))
    written = set()
    i = header + 1
    while i < end:
        match = re.match(r"^(\s*)([\"']?)(shared|policy|value)\2(\s*=\s*)\[(.*)$", lines[i])
        if match:
            indent, quote, key, equals, after = match.groups()
            # The list may run over several lines; it ends at the first "]" (a list of numbers has no other).
            last, tail = i, after
            while "]" not in tail and last + 1 < end:
                last += 1
                tail = lines[last]
            rest = tail.split("]", 1)[1] if "]" in tail else ""
            lines[i : last + 1] = [f"{indent}{quote}{key}{quote}{equals}{_toml_list(values[key])}{rest}"]
            end -= last - i
            written.add(key)
        i += 1
    for key in ("shared", "policy", "value"):
        if key not in written:
            lines.insert(header + 1, f"{key} = {_toml_list(values[key])}  # set by nitro surgery")
            end += 1
    return "\n".join(lines) + "\n"


def _toml_list(widths: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(w) for w in widths) + "]"


def model_shapes(model_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Observation size and action count from a saved network's weights."""
    first = model_state.get("shared.0.weight", model_state["policy.0.weight"])
    output = max(
        (
            k
            for k in model_state
            if k.startswith("policy.") and k.endswith(".weight") and model_state[k].dim() == 2
        ),
        key=lambda k: int(k.split(".")[1]),
    )
    return int(first.shape[1]), int(model_state[output].shape[0])


@dataclasses.dataclass(frozen=True)
class SurgeryReport:
    source: str
    target: str
    agent_steps: int
    factor: int
    old_widths: ModelConfig
    new_widths: ModelConfig
    largest_difference: float
    pool_versions: int
    resume: str = ""  # the command that continues the new run

    def lines(self) -> list[str]:
        def widths(m: ModelConfig) -> str:
            parts = [f"policy {list(m.policy)}", f"value {list(m.value)}"]
            if m.shared:
                parts.insert(0, f"shared {list(m.shared)}")
            return ", ".join(parts)

        return [
            f"Made '{self.target}' from '{self.source}' at {self.agent_steps:,} agent steps: networks "
            f"{self.factor}x wider ({widths(self.old_widths)} -> {widths(self.new_widths)}).",
            f"  Same behaviour to within rounding: the old and the new network differ by at most "
            f"{self.largest_difference:.2g} on {CHECK_OBSERVATIONS:,} random observations.",
            "  Adam's running averages start afresh; everything else carries over"
            + (
                f", with {self.pool_versions} past versions in the self-play pool widened too."
                if self.pool_versions
                else "."
            ),
            f"  Continue with: {self.resume or f'nitro resume {self.target}'}",
        ]


def widen_run(
    runs: Path, source: str, factor: int, target: str | None = None, at: int | None = None
) -> SurgeryReport:
    """Creates run `target` from `source`'s save at `at` agent steps (newest by default) with every hidden
    layer `factor` times as wide; see the module docstring."""
    if factor < 2:
        raise SurgeryError("--widen must be a whole number of 2 or more (2 doubles every hidden layer)")
    folder = RunFolder(runs / source)
    if not folder.info.is_file():
        raise SurgeryError(f"There is no run called '{source}' in {runs}.")
    state = folder.load_checkpoint(at) if at is not None else folder.load_latest()
    if state is None:
        saves = ", ".join(f"{int(p.name):,}" for p in folder.complete_checkpoints()) or "none"
        where = f"at {at:,} agent steps" if at is not None else "yet"
        raise SurgeryError(f"'{source}' has no save {where}. Saves: {saves}.")
    config = folder.load_settings(state)
    target = target or f"{source}-x{factor}"
    final = runs / target
    if final.exists():
        raise SurgeryError(
            f"a run called '{target}' already exists in {runs}; choose another name with --as."
        )
    if widened_config(config.model, factor) == config.model:
        raise SurgeryError(
            f"'{source}' has no hidden layers to widen (model.shared, model.policy and model.value are all "
            "empty)."
        )

    if config.model.uses_attention:
        raise SurgeryError(
            f"'{source}' has attention networks (model.encoder); widening works on plain networks only."
        )
    if config.model.residual or config.model.value_heads_apart:
        raise SurgeryError(
            f"'{source}' has residual blocks or value heads (model.residual, model.value_heads); "
            "widening works on plain networks only."
        )
    model_state = state["learner"]["model"]
    obs_size, num_actions = model_shapes(model_state)
    old = ActorCritic(obs_size, num_actions, config.model)
    old.load_state_dict(model_state)
    new = widen_model(old, factor)
    difference, scale = compare(old, new, obs_size)
    if difference > allowed(scale):
        raise SurgeryError(
            f"the widened network does not match the old one (largest difference {difference:.3g} on outputs "
            f"up to {scale:.3g}); nothing was written. This should not happen: please report it with the "
            "run's settings."
        )
    # The new settings file, checked before anything is written: it must read back with the new widths.
    new_text = widen_settings_text(folder.settings.read_text(encoding="utf-8-sig"), config.model, factor)
    base_dir = Path(json.loads(folder.info.read_text(encoding="utf-8"))["settings_dir"])
    reread: ModelConfig | None
    try:
        reread = parse_toml(new_text, NitroConfig, file=folder.settings, base_dir=base_dir).model
        reason = f"it would read back as {reread}"
    except ConfigError as err:
        reread, reason = None, str(err)
    if reread != new.config:
        raise SurgeryError(
            "the new widths could not be written into the settings file automatically "
            f"({reason}). Put shared, policy and value in the [model] section of {folder.settings}, each on "
            "one line (for example: policy = [256, 256]), and run the surgery again. Nothing was written."
        )
    new_config = dataclasses.replace(config, model=new.config)
    gamma = discount_from_half_life(config.ppo.discount_half_life, config.env.tick_skip)
    learner = Learner(new, config.ppo, gamma, torch.device("cpu"), fp16=False)
    old_learner = state["learner"]
    new_learner = learner.state_dict()  # fresh Adam state for the new shapes
    for key in ("scaler", "value_norm", "reward_scaler", "learning_rates", "return_range"):
        if key in old_learner:
            new_learner[key] = old_learner[key]
    new_state = {
        **state,
        "learner": new_learner,
        "settings": fixed_settings(new_config),
        "all_settings": all_settings(new_config, final),
        "nitro_version": __version__,  # this release made the widened networks
    }

    # The new run folder, built under a temporary name and renamed last, so a failure leaves nothing that
    # blocks a second try: settings with the new widths, the save, the metrics so far, the pool widened.
    partial = runs / f".{target}.surgery-partial"
    shutil.rmtree(partial, ignore_errors=True)
    try:
        pool_versions = _write_run(
            RunFolder(partial), folder, new_text, new_state, obs_size, num_actions, config, factor
        )
        partial.rename(final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return SurgeryReport(
        source,
        target,
        int(state["agent_steps"]),
        factor,
        config.model,
        new.config,
        difference,
        pool_versions,
        resume_command(RunFolder(final)),
    )


def _write_run(
    new_folder: RunFolder,
    folder: RunFolder,
    settings_text: str,
    new_state: dict[str, Any],
    obs_size: int,
    num_actions: int,
    config: NitroConfig,
    factor: int,
) -> int:
    """Writes the widened run into `new_folder`; returns how many past versions of the pool it widened."""
    new_folder.checkpoints.mkdir(parents=True)
    new_folder.settings.write_text(settings_text, encoding="utf-8")
    shutil.copyfile(folder.info, new_folder.info)
    if folder.metrics.is_file():
        shutil.copyfile(folder.metrics, new_folder.metrics)
    pool_versions = 0
    pool, new_pool = folder.root / "pool", new_folder.root / "pool"
    if pool.is_dir():
        new_pool.mkdir()
        for file in pool.iterdir():
            if file.suffix == ".pt" and file.stem.isdigit():
                try:
                    weights = torch.load(file, map_location="cpu", weights_only=True)
                except UNREADABLE:  # cut short: copied as it is; the new run's self-play passes it over
                    shutil.copyfile(file, new_pool / file.name)
                    continue
                carrier = ActorCritic(obs_size, num_actions, config.model)
                carrier.load_policy_state_dict(weights)
                torch.save(widen_model(carrier, factor).policy_state_dict(), new_pool / file.name)
                pool_versions += 1
            elif file.is_file():
                shutil.copyfile(file, new_pool / file.name)
    run = config.run
    new_folder.save_checkpoint(
        new_state, int(new_state["agent_steps"]), run.keep_checkpoints, run.milestone_steps
    )
    new_folder.load_settings(new_state)  # the new settings and the new save agree
    return pool_versions
