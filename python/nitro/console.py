"""The console report (`run.console = "report"`): every `run.console_seconds` a training run
prints a block of what it is doing: speed and where the time goes, the rewards
and each reward's part, how the policy and value learn, how the bots play, self-play, the bot it started
from and the next save; the replay copy prints one each round. Plain ASCII, so a Windows
console or a job's captured output never fails on a character. Training is the same either way."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

WIDTH = 100
_MECHANICS = (
    ("aerial_touches_per_minute", "aerial touches"),
    ("dribbles_per_minute", "dribbles"),
    ("flicks_per_minute", "flicks"),
    ("double_taps_per_minute", "double taps"),
    ("flip_resets_per_minute", "flip resets"),
    ("shots_per_minute", "shots"),
    ("saves_per_minute", "saves"),
    ("demos_per_minute", "demolitions"),
)
# Means over a count of their own: averaged weighted by it, so an iteration with more kickoffs counts more.
_KICKOFFS = (("diagonal", "diagonal"), ("offcenter", "off-centre"), ("back", "back"))
_WEIGHTED = {f"kickoff_touch_s_{name}": f"kickoff_count_{name}" for name, _ in _KICKOFFS}


@dataclass
class Facts:
    """What a block says beyond metrics.csv's row."""

    name: str
    running_seconds: float
    steps_per_second: float  # over the time since the last block
    collect_seconds: float = 0.0  # the newest iteration's
    learn_seconds: float = 0.0
    reward_labels: Sequence[str] = ()
    started_from: str = ""
    transfer: bool = False  # started from another run, not from a copy of replays
    next_save: str = ""
    warnings: Sequence[str] = field(default_factory=list)  # new since the last block


def averaged(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The rows since the last block as one: each number's mean over the rows that have it, the newest
    row's value for the counters."""
    if not rows:
        return {}
    out: dict[str, Any] = dict(rows[-1])
    for key in out:
        if key in ("agent_steps", "iteration", "pool_size", "grad_scale"):
            continue
        if key in _WEIGHTED:
            pairs = [(_number(row.get(key)), _number(row.get(_WEIGHTED[key])) or 0.0) for row in rows]
            weighed = [(v, n) for v, n in pairs if v is not None and n > 0]
            total = sum(n for _, n in weighed)
            out[key] = sum(v * n for v, n in weighed) / total if total else ""
            continue
        values = [_number(row.get(key)) for row in rows]
        kept = [v for v in values if v is not None]
        if kept:
            out[key] = sum(kept) / len(kept)
    return out


def iteration_report(row: Mapping[str, Any], facts: Facts) -> list[str]:
    """The block for a training run, from `averaged` rows."""
    steps = int(_number(row.get("agent_steps")) or 0)
    lines = [
        _rule(
            f"{facts.name} | iteration {int(_number(row.get('iteration')) or 0):,} | "
            f"{_steps(steps)} agent steps | {_duration(facts.running_seconds)} this session"
        )
    ]
    speed = f"{facts.steps_per_second:,.0f} steps/s"
    if facts.collect_seconds or facts.learn_seconds:
        speed += (
            f" | an iteration: playing {facts.collect_seconds:.2f} s, learning {facts.learn_seconds:.2f} s"
        )
    lines.append(_item("Speed", speed))
    rewards = [f"{_fixed(row.get('reward_per_step'), 4)} a step"]
    if (episode := _number(row.get("episode_seconds"))) is not None:
        rewards.append(f"episodes {episode:.1f} s on average")
    lines.append(_item("Reward", " | ".join(rewards)))
    parts = [
        f"{label} {_fixed(row.get(f'raw_{label}'), 4)}"
        for label in facts.reward_labels
        if _number(row.get(f"raw_{label}")) is not None
    ]
    lines += _wrapped("", parts, "(each reward's own value a step, before its weight)")
    policy = [
        f"entropy {_fixed(row.get('entropy'), 3)}",
        f"change {_fixed(row.get('policy_change'), 4)}",
        f"clipped {_percent(row.get('clip_fraction'))}",
    ]
    if (rate := _number(row.get("policy_lr"))) is not None:
        policy.append(f"learning rate {rate:.1e}")
    lines.append(_item("Policy", " | ".join(policy)))
    lines.append(
        _item(
            "Value",
            f"prediction {_fixed(row.get('explained_variance'), 2)} (1 = perfect) | "
            f"loss {_fixed(row.get('value_loss'), 4)}",
        )
    )
    play = [
        (row.get("touches_per_minute"), "{:.1f} touches a minute"),
        (row.get("in_air_share"), "{:.0%} in the air"),
        (row.get("speed_kmh"), "{:.0f} km/h"),
        (row.get("boost_held"), "boost {:.0f}"),
        (row.get("supersonic_share"), "{:.0%} supersonic"),
        (row.get("demolished_share"), "{:.0%} demolished"),
    ]
    shown = [form.format(v) for value, form in play if (v := _number(value)) is not None]
    if shown:
        lines.append(_item("Play", " | ".join(shown)))
    kickoffs = [
        (label, v, _number(row.get(f"kickoff_count_{name}")) or 0.0)
        for name, label in _KICKOFFS
        if (v := _number(row.get(f"kickoff_touch_s_{name}"))) is not None
    ]
    if kickoffs:
        times = " | ".join(f"{label} {v:.2f} s" for label, v, _ in kickoffs)
        counts = " / ".join(f"{n:.1f}" for _, _, n in kickoffs)
        lines.append(_item("Kickoffs", f"first touch {times} (kickoffs an iteration: {counts})"))
    tricks = [
        f"{label} {v:.2f}" for key, label in _MECHANICS if (v := _number(row.get(key))) is not None and v > 0
    ]
    if tricks:
        lines += _wrapped("Mechanics", tricks, "(per car and minute)")
    if (mu := _number(row.get("rating_mu"))) is not None:
        selfplay = [f"rating {mu:.1f}"]
        if (share := _number(row.get("pool_goal_share"))) is not None:
            selfplay.append(f"{share:.0%} of the goals against past versions")
        if (pool := _number(row.get("pool_size"))) is not None:
            selfplay.append(f"{pool:.0f} past versions")
        lines.append(_item("Self-play", " | ".join(selfplay)))
    if facts.started_from:
        start = facts.started_from + (" (transfer learning)" if facts.transfer else "")
        lines.append(_item("Started", f"from {start}"))
    if (copy_loss := _number(row.get("replay_mix_loss"))) is not None:  # [replay_mix]
        copied = (
            f"{_number(row.get('replay_mix_steps')) or 0:.1f} steps an update | loss {copy_loss:.3f} | "
            f"first choice as the players' {_percent(row.get('replay_mix_agreement'))}"
        )
        if (wait := _number(row.get("replay_mix_wait_seconds"))) is not None and wait >= 0.05:
            copied += f" | waited {wait:.1f} s for replays"
        kept, left = _number(row.get("replay_mix_rows_kept")), _number(row.get("replay_mix_rows_left_out"))
        if kept is not None and left is not None and kept + left > 0:  # kickoff_only_seconds
            copied += f" | kickoffs only: {kept / (kept + left):.0%} of the new replays' decisions kept"
        lines.append(_item("Replays", copied))
    fp16 = []
    if (scale := _number(row.get("grad_scale"))) is not None and scale > 1:  # 1: FP32, nothing scaled
        fp16.append(f"loss scale {scale:,.0f}")
    if (skipped := _number(row.get("skipped_updates"))) is not None and skipped > 0:
        fp16.append(f"{skipped:.1f} updates skipped")
    if fp16:
        lines.append(_item("FP16", " | ".join(fp16)))
    if facts.next_save:
        lines.append(_item("Next save", facts.next_save))
    lines += [_item("Warning", warning) for warning in facts.warnings[-3:]]
    return lines


@dataclass
class Round:
    """One round of the large replay copy, for its block."""

    name: str
    number: int
    used: int
    limit: int
    decisions: int
    train_loss: float
    held_loss: float | None  # None: a round between measurements (pretrain.measure_every)
    agreement: float | None
    held_decisions: int
    held_replays: int
    rate: float  # replays made ready a minute
    learned_seconds: float
    left: str = ""  # "about 3 h 39 min" or ""
    used_before: int | None = None  # new_only
    before: Mapping[str, float] | None = None  # continue_from


def round_report(r: Round) -> list[str]:
    """The block for one round of the replay copy."""
    rounds = math.ceil(r.limit / max(r.used / max(r.number, 1), 1))
    lines = [_rule(f"{r.name} | copying replays | round {r.number:,} of about {rounds:,}")]
    replays = f"{r.used:,} of {r.limit:,} learned ({r.decisions:,} decisions)"
    if r.used_before is not None:
        replays += f" | {r.used_before:,} left out as used before"
    lines.append(_item("Replays", replays))
    held = "not measured this round" if r.held_loss is None else f"{r.held_loss:.3f} (lower is closer)"
    loss = f"this round {r.train_loss:.3f} | held back {held}"
    if r.before:
        loss += f" | before these replays {r.before['held_loss']:.3f}"
    lines.append(_item("Loss", loss))
    if r.agreement is not None:
        agree = (
            f"{r.agreement:.1%} of the players' first choices, on {r.held_decisions:,} decisions from "
            f"{r.held_replays} held-back replays"
        )
        if r.before:
            agree += f" (before {r.before['held_agreement']:.1%})"
        lines.append(_item("Agreement", agree))
    speed = f"{r.rate:.1f} replays made ready a minute | round learned in {r.learned_seconds:.0f} s"
    lines.append(_item("Speed", speed))
    if r.left:
        lines.append(_item("Time left", r.left))
    return lines


def _number(value: Any) -> float | None:
    if value in ("", None) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fixed(value: Any, digits: int) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{0.0 if round(number, digits) == 0 else number:.{digits}f}"  # no "-0.0000"


def _percent(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.1%}"


def steps_text(steps: int) -> str:
    """1.23B, 12.3M or 5,000."""
    return _steps(steps)


def _steps(steps: int) -> str:
    if steps >= 1e9:
        return f"{steps / 1e9:,.2f}B"
    return f"{steps / 1e6:,.1f}M" if steps >= 1e6 else f"{steps:,}"


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes >= 60:
        return f"{minutes // 60} h {minutes % 60} min"
    return f"{minutes} min {int(seconds % 60)} s"


def _rule(title: str) -> str:
    head = f"---- {title} "
    return head + "-" * max(4, WIDTH - len(head))


def _item(label: str, text: str) -> str:
    return f" {label:<10} {text}"


def _wrapped(label: str, parts: Sequence[str], note: str) -> list[str]:
    """`parts` over as many lines as they need, `note` after them."""
    lines: list[str] = []
    line = ""
    for part in [*parts, note] if parts else []:
        joined = f"{line} | {part}" if line and part != note else (f"{line}   {part}" if line else part)
        if line and len(joined) > WIDTH - 12:
            lines.append(line)
            line = part
        else:
            line = joined
    if line:
        lines.append(line)
    return [_item(label if i == 0 else "", text) for i, text in enumerate(lines)]
