"""Would sums kept in FP16 inside the card's matrix products make a bot faster, and how far would its answers
move? Measures only; changes nothing.

GeForce cards multiply FP16 numbers at half their rate when each product's running sum is kept in FP32, which
PyTorch does by default, and at the full rate when it is kept in FP16
(`torch.backends.cuda.matmul.allow_fp16_accumulation`). This trains a scratch copy of the bot (its settings,
latest save and pool; the bot itself is not touched) for `--warmup` iterations on one card for real
observations, then compares three ways of computing, on the same rows:
- fp32: no FP16 at all (the yardstick of how far FP16 already moves the answers);
- fp16: what training does now (autocast, sums in FP32);
- fp16-sums: autocast with sums in FP16 (for learning also fp16-sums-forward: the backward pass's in FP32),
  as `run.fp16_sums` does.
For the playing pass (the FP16 playing copy, replayed as a CUDA graph as collection does) on `--rows` rows:
milliseconds, and against fp32 (and FP16 sums against fp16, how the bot plays now) the action chances' KL, the
largest change of one chance, the share of sampled actions that differ under the same noise and the value's
change, with how far PPO's ratio starts from 1 when the playing pass keeps FP16 sums. For a learning pass
(forward and backward under autocast, a policy and value loss) on `--learn-rows` rows: milliseconds, and
against fp32 the gradients' cosine and the size of their difference. Then the largest matrix products alone.

    python -I benchmarks/profile/fp16_accumulation.py --bot my-bot \
        --set replay_mix.enabled=false --out my-bot_fp16_accumulation.txt

`--set section.name=value` (repeatable) changes a setting of the scratch copy, as in cards_speed.py.

Needs a CUDA card and PyTorch 2.7 or newer, and a quiet machine for the milliseconds.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import shutil
import statistics
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch

from nitro.config.schema import NitroConfig
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.runs import RunFolder
from nitro.training import TrainingRun

ROOT = Path(__file__).resolve().parents[2]
WAYS = ("fp32", "fp16", "fp16-sums")
LEARN_WAYS = ("fp32", "fp16", "fp16-sums-forward", "fp16-sums")  # forward: the backward's sums in FP32


def overridden(config: NitroConfig, changes: Sequence[str]) -> NitroConfig:
    """`config` with each `section.name=value` (the value as TOML writes it) applied."""
    for change in changes:
        path, _, text = change.partition("=")
        section, _, name = path.partition(".")
        part = getattr(config, section, None)
        if part is None or not hasattr(part, name):
            raise SystemExit(f"there is no setting {path}")
        value = tomllib.loads(f"v = {text}")["v"]
        config = dataclasses.replace(config, **{section: dataclasses.replace(part, **{name: value})})
    return config


@contextlib.contextmanager
def sums_in_fp16(on: bool) -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    before = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = on
    try:
        yield
    finally:
        matmul.allow_fp16_accumulation = before


def autocast(way: str) -> contextlib.AbstractContextManager[None]:
    return torch.autocast("cuda", dtype=torch.float16, enabled=way != "fp32", cache_enabled=False)


def milliseconds(work: Callable[[], object], repeats: int) -> float:
    """The median of `repeats` timed runs of `work` on the card, after three untimed ones."""
    for _ in range(3):
        work()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        work()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def playing(
    model: ActorCritic, obs: torch.Tensor, way: str
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor]:
    """A CUDA graph of the playing pass on `obs` computed `way`, and its logits and values (a replay fills
    them)."""
    stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    stream.wait_stream(torch.cuda.current_stream())
    with sums_in_fp16(way == "fp16-sums"), torch.no_grad():
        with torch.cuda.stream(stream):
            for _ in range(3):
                with autocast(way):
                    model(obs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), autocast(way):
            logits, values = model(obs)
    graph.replay()
    return graph.replay, logits, values


def learning(
    model: ActorCritic, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor, way: str
) -> Callable[[], torch.Tensor]:
    """A forward and backward pass computed `way`; returns the gradients as one vector."""

    def run() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with sums_in_fp16(way.startswith("fp16-sums")), autocast(way):
            logits, values = model(obs)
        log_probs = masked_log_probs(logits, mask)
        loss = -log_probs.gather(1, actions[:, None]).mean() + 0.5 * values.float().square().mean()
        with sums_in_fp16(way == "fp16-sums"):
            loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    return run


def compare_playing(
    reference: tuple[torch.Tensor, torch.Tensor], other: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor
) -> str:
    (ref_logits, ref_values), (logits, values) = reference, other
    ref, test = masked_log_probs(ref_logits.float(), mask), masked_log_probs(logits.float(), mask)
    allowed_ref = torch.where(mask, ref, torch.zeros_like(ref))
    allowed_test = torch.where(mask, test, torch.zeros_like(test))
    kl = (allowed_ref.exp() * (allowed_ref - allowed_test)).sum(-1)
    largest = (allowed_ref.exp() - allowed_test.exp()).abs().max().item()
    generator = torch.Generator(device=mask.device).manual_seed(0)
    uniform = torch.rand(ref.shape, device=mask.device, generator=generator).clamp_(1e-20, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(uniform))
    differ = ((ref + gumbel).argmax(-1) != (test + gumbel).argmax(-1)).float().mean().item()
    spread = ref_values.float().std().item() or 1.0
    value = (values.float() - ref_values.float()).abs().mean().item() / spread
    return (
        f"KL mean {kl.mean().item():.2e} (largest {kl.max().item():.2e}); a chance moved at most "
        f"{largest:.2e}; sampled actions differ {differ:.4%}; value moved {value:.2e} of its spread"
    )


def ratios(now: torch.Tensor, sums: torch.Tensor, mask: torch.Tensor, clip: float = 0.2) -> str:
    """Learning's view: actions sampled by the playing pass with FP16 sums (`sums` logits), and the chance
    learning gives them (`now`: the learner's pass is the playing copy's to the bit). PPO's ratio
    starts at their quotient instead of exactly 1; how far, and how often beyond its clip range."""
    taken, learner = masked_log_probs(sums.float(), mask), masked_log_probs(now.float(), mask)
    generator = torch.Generator(device=mask.device).manual_seed(2)
    uniform = torch.rand(taken.shape, device=mask.device, generator=generator).clamp_(1e-20, 1.0 - 1e-7)
    actions = (taken - torch.log(-torch.log(uniform))).argmax(-1, keepdim=True)
    ratio = (learner.gather(1, actions) - taken.gather(1, actions)).exp()
    away = (ratio - 1).abs()
    return (
        f"PPO's starting ratio off 1 by {away.mean().item():.2e} on average (largest "
        f"{away.max().item():.2e}), beyond the {clip} clip "
        f"{(away > clip).float().mean().item():.4%} of the rows"
    )


def compare_gradients(reference: torch.Tensor, other: torch.Tensor) -> str:
    cosine = torch.nn.functional.cosine_similarity(reference, other, dim=0).item()
    size = ((other - reference).norm() / reference.norm()).item()
    return f"gradient cosine {cosine:.6f}, difference {size:.2e} of the gradient's size"


def products(model: ActorCritic, rows: tuple[int, ...], repeats: int) -> list[str]:
    """The model's largest linear layers as (rows x inputs) @ (inputs x outputs) in FP16: TFLOP/s each way."""
    shapes = sorted(
        {(m.in_features, m.out_features) for m in model.modules() if isinstance(m, torch.nn.Linear)},
        key=lambda shape: -shape[0] * shape[1],
    )[:4]
    lines = []
    for count in rows:
        for inputs, outputs in shapes:
            a = torch.randn(count, inputs, device="cuda", dtype=torch.float16)
            b = torch.randn(inputs, outputs, device="cuda", dtype=torch.float16)
            rates = []
            for way in ("fp16", "fp16-sums"):
                with sums_in_fp16(way == "fp16-sums"):
                    took = milliseconds(functools.partial(torch.mm, a, b), repeats)
                rates.append(2 * count * inputs * outputs / took / 1e9)
            lines.append(
                f"    {count:>6} x {inputs:>5} @ {inputs:>5} x {outputs:>5}: {rates[0]:6.1f} TFLOP/s, "
                f"sums in FP16 {rates[1]:6.1f} ({rates[1] / rates[0]:.2f}x)"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bot", required=True, help="a bot's name in runs/")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--warmup", type=int, default=2, help="training iterations for real observations")
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 2048], help="playing pass rows")
    parser.add_argument("--learn-rows", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--set", action="append", default=[], help="section.name=value (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        parser.error("needs a CUDA card and PyTorch 2.7 or newer")
    bot = RunFolder(args.runs / args.bot)
    state = bot.load_latest()
    config = overridden(bot.load_settings(state), args.set)
    config = dataclasses.replace(
        config, run=dataclasses.replace(config.run, cards=1, checkpoint_minutes=1e6, checkpoint_steps=0)
    )
    lines = [f"== {bot.root.name} on {torch.cuda.get_device_name()} (PyTorch {torch.__version__})"]
    with tempfile.TemporaryDirectory(prefix="nitro-fp16-sums-", ignore_cleanup_errors=True) as scratch:
        folder = RunFolder.create(Path(scratch), f"fp16-sums-{time.monotonic_ns()}", bot.settings)
        (folder.root / "run.json").write_text(
            (bot.root / "run.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        if (bot.root / "pool").is_dir():
            shutil.copytree(bot.root / "pool", folder.root / "pool")
        run = TrainingRun(config, folder, state)
        try:
            for _ in range(args.warmup):
                run.iteration()
            trainer = run.trainer
            rollout = trainer.rollout
            obs = rollout.obs.flatten(0, 1)
            mask = rollout.mask.flatten(0, 1)
            actions = rollout.actions.flatten(0, 1)
            generator = torch.Generator(device=obs.device).manual_seed(1)
            playing_copy = trainer.inference_model
            lines.append(
                f"Playing pass ({'the FP16 playing copy' if playing_copy is not run.model else 'the model'}):"
            )
            for count in args.rows:
                rows = torch.randperm(obs.shape[0], device=obs.device, generator=generator)[:count]
                passes = {
                    way: playing(playing_copy if way != "fp32" else run.model, obs[rows], way) for way in WAYS
                }
                outputs = {
                    way: (logits.clone(), values.clone()) for way, (_, logits, values) in passes.items()
                }
                took = {way: milliseconds(replay, args.repeats) for way, (replay, _, _) in passes.items()}
                lines.append(
                    f"  {count} rows: "
                    + ", ".join(f"{way} {took[way]:.3f} ms" for way in WAYS)
                    + f" (sums in FP16 {took['fp16'] / took['fp16-sums']:.2f}x as fast as now)"
                )
                for way in WAYS[1:]:
                    lines.append(
                        f"    {way} against fp32: "
                        + compare_playing(outputs["fp32"], outputs[way], mask[rows])
                    )
                lines.append(
                    "    fp16-sums against fp16 (how the bot plays now): "
                    + compare_playing(outputs["fp16"], outputs["fp16-sums"], mask[rows])
                )
                lines.append("    " + ratios(outputs["fp16"][0], outputs["fp16-sums"][0], mask[rows]))
                del passes, outputs
            count = min(args.learn_rows, obs.shape[0])
            rows = torch.randperm(obs.shape[0], device=obs.device, generator=generator)[:count]
            model = run.model
            gradients, took = {}, {}
            for way in LEARN_WAYS:
                step = learning(model, obs[rows], mask[rows], actions[rows], way)
                gradients[way] = step().clone()
                took[way] = milliseconds(step, max(5, args.repeats // 5))
            model.zero_grad(set_to_none=True)
            lines.append(
                f"Learning pass, forward and backward, {count} rows: "
                + ", ".join(f"{way} {took[way]:.2f} ms" for way in LEARN_WAYS)
                + f" (sums in FP16 {took['fp16'] / took['fp16-sums']:.2f}x as fast as now)"
            )
            for way in LEARN_WAYS[1:]:
                lines.append(
                    f"    {way} against fp32: {compare_gradients(gradients['fp32'], gradients[way])}"
                )
            lines.append("The largest matrix products alone:")
            lines += products(model, (512, 1024, count), args.repeats)
        finally:
            run.metrics.close()
            shutil.rmtree(folder.root, ignore_errors=True)
    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
