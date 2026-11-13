"""Training metrics: one CSV row per iteration and a self-contained HTML report.

The report is a single file that opens in any browser, with no server and nothing to install. Each
metric keeps a fixed-size, evenly thinned sample in memory, so writing the report costs the same at
iteration ten and at iteration a million.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import string
import time
from collections.abc import Iterable
from pathlib import Path

_TEMPLATE = string.Template(Path(__file__).with_name("report.html").read_text(encoding="utf-8"))
# Points kept per metric for the report; when full, every other point is dropped.
_SAMPLE_POINTS = 512

# Column -> (title, what it means), in report order. Columns not listed are charted after these.
CHARTS: dict[str, tuple[str, str]] = {
    "steps_per_second": ("Training speed", "agent steps per second"),
    "wall_steps_per_second": (
        "Training speed, wall clock",
        "agent steps per second with the time between iterations (saves, the run's own bookkeeping)",
    ),
    "reward_per_step": ("Reward per agent step", "what the bots are paid per step, a stand-in for skill"),
    "entropy": (
        "Randomness of the policy",
        "entropy: 1 = picking at random among allowed actions, 0 = certain",
    ),
    "policy_change": ("How far each update moved the policy", "KL divergence before vs after the update"),
    "explained_variance": (
        "Value predictions",
        "explained variance: 1 = perfect, 0 = no better than the average, below 0 = worse",
    ),
    "value_loss": ("Value loss", "how wrong the value predictions are (lower is better)"),
    "start_value": (
        "Value of a fresh start",
        "the critic's mean value of the fresh starts learned from, in raw reward units; in a shaped "
        "self-play run this is about what a goal cuts off when goals end the game",
    ),
    "start_value_share": (
        "Value of a fresh start, per goal",
        "start_value as a share of the goal reward's weight",
    ),
    "clip_fraction": ("Clipped updates", "share of samples where PPO limited the update"),
    "skipped_updates": ("Skipped learning updates", "updates dropped because a number overflowed"),
    "grad_scale": ("Loss scale", "FP16 only: how far gradients are scaled up to stay representable"),
    "policy_lr": ("Policy learning rate", "the policy's learning rate (ppo.policy_lr)"),
    "update_size": ("Size of each update", "how far an iteration moved the networks' weights, relative"),
    "playing_seconds": ("Time playing", "seconds an iteration spends playing the games"),
    "learning_seconds": ("Time learning", "seconds an iteration spends learning from them"),
    "entropy_coef": ("Entropy bonus", "the bonus ppo.entropy_target moves to hold the randomness near it"),
    "goal_explained_variance": (
        "Goal critic's predictions",
        "explained variance of the long-horizon goal critic: 1 = perfect, 0 = no better than the average",
    ),
    "goal_loss": ("Goal critic's loss", "how wrong the goal critic's predictions are (lower is better)"),
    "vdag_loss": ("Twin critics' loss", "the optimistic twin critics' expectile loss, per twin"),
    "vdag_dose": (
        "Twin critics' push",
        "spread of what the twin critics add to the advantages, as a share of theirs (about vdag_beta)",
    ),
    "vdag_headroom": (
        "Twin critics' headroom",
        "how much more the twin critics think the bot could get than the value expects (mean, at least 0)",
    ),
    "vdag_clamped": (
        "Twin critics' clamped share",
        "share of samples whose push was cut at 3 times its spread",
    ),
    "head_loss_weight": (
        "Extra critics' loss weight",
        "ppo.head_loss_scale = 'popart': the goal and twin critics' losses are multiplied by this (1 / "
        "PopArt's variance)",
    ),
    "advantage_kept": ("Samples the policy learns from", "share kept by ppo.advantage_keep"),
    "advantage_kept_size": (
        "Advantage the policy learns from",
        "share of the advantages' total size in the samples ppo.advantage_keep kept",
    ),
    "advantage_rows": (
        "Samples the learning steps run on",
        "share of all samples: the kept ones plus ppo.advantage_keep_rest's random share of the others",
    ),
    "replay_mix_loss": ("Copying replays: loss", "the copy loss on converted replays (lower is closer)"),
    "replay_mix_agreement": (
        "Copying replays: agreement",
        "share of the replays' decisions where the bot's first choice is the player's",
    ),
    "touches_per_minute": ("Ball touches", "per car and minute of play"),
    "in_air_share": ("Time in the air", "share of the time the bots' cars are off the ground"),
    "speed_kmh": ("Speed", "the bots' average speed, km/h"),
    "speed_to_ball_kmh": ("Speed toward the ball", "km/h; high when the bots chase the ball"),
    "boost_held": ("Boost held", "average boost in the tank, 0 to 100"),
    "supersonic_share": ("Supersonic", "share of the time at full speed"),
    "demolished_share": ("Demolished", "share of the time the bots' cars are waiting to respawn"),
    "ball_on_car_share": ("Ball on the car", "share of the time the ball rests on a bot's car (dribbling)"),
    "aerial_touches_per_minute": ("Aerial touches", "per car and minute: touches in the air above 300 uu"),
    "dribbles_per_minute": ("Dribbles", "per car and minute: the ball carried on the car for 0.75 s or more"),
    "flicks_per_minute": ("Flicks", "per car and minute: a carried ball fired off by the car's own dodge"),
    "double_taps_per_minute": ("Double taps", "per car and minute: aerial, backboard, the same car again"),
    "flip_resets_per_minute": ("Flip resets", "per car and minute: the flip regained by touching the ball"),
    "flip_reset_height": (
        "Flip reset height",
        "the ball's average height at the flip resets (uu; the ceiling is 2044)",
    ),
    "reset_musty_goals_per_hour": (
        "Reset musty goals",
        "per car-hour: goals within 6 s of the car's flip reset with its back flip between (a counter)",
    ),
    "reset_musty_goal_speed_kmh": (
        "Reset musty goal speed",
        "the ball's average speed into the net at the reset musty goals, km/h",
    ),
    "flip_reset_goal_dist": (
        "Flip reset distance from the net",
        "the ball's average distance from the opponents' goal line at the flip resets (uu; midfield is 5120)",
    ),
    "shots_per_minute": ("Shots", "per car and minute: touches that send the ball into the goal's path"),
    "saves_per_minute": ("Saves", "per car and minute: touches that turn the ball away from the own goal"),
    "demos_per_minute": ("Demolitions", "per car and minute"),
    # The kickoff setter's episodes by spot.
    "kickoff_touch_s_diagonal": (
        "Kickoff first touch, diagonal",
        "seconds until either car first touches the ball, from the two diagonal spots",
    ),
    "kickoff_touch_s_offcenter": (
        "Kickoff first touch, off-centre",
        "seconds until either car first touches the ball, from the two spots back and to the side",
    ),
    "kickoff_touch_s_back": (
        "Kickoff first touch, straight back",
        "seconds until either car first touches the ball, from the spot straight back",
    ),
    "kickoff_count_diagonal": ("Diagonal kickoffs", "touched diagonal kickoffs this iteration"),
    "kickoff_count_offcenter": ("Off-centre kickoffs", "touched off-centre kickoffs this iteration"),
    "kickoff_count_back": ("Straight-back kickoffs", "touched straight-back kickoffs this iteration"),
    "rating_mu": ("Rating against past versions", "skill rating from games against saved past versions"),
    "pool_goal_share": ("Goals against past versions", "share of goals scored by the bot in those games"),
    # With a scripted (uneven) start.
    "pool_goal_share_even": (
        "Goals against past versions, even starts",
        "the same from starts that place every car alike (kickoff, random...), free of the practice starts' "
        "head start for the bot",
    ),
    "pool_games_even": ("Games against past versions, even starts", "how many this iteration"),
    "start_edge": (
        "Blue's goals from practice starts",
        "Blue's share of goals from the scripted starts (one car placed, the rest parked), the bot against "
        "itself. Most favour Blue, save favours Orange, so a mix can read 0.5 with each start one-sided",
    ),
    "matchmaking": ("Match evenness", "how far the predicted win chance was from 50% (0 = even)"),
}


class _Sample:
    """Every `stride`-th point of a series, at most `_SAMPLE_POINTS` of them, plus the latest point."""

    def __init__(self) -> None:
        self.kept: list[tuple[float, float]] = []
        self.latest: tuple[float, float] | None = None
        self.stride = 1
        self.seen = 0

    def add(self, x: float, y: float) -> None:
        if self.seen % self.stride == 0:
            self.kept.append((x, y))
            if len(self.kept) > _SAMPLE_POINTS:
                self.kept = self.kept[::2]
                self.stride *= 2
        self.latest = (x, y)
        self.seen += 1

    @property
    def points(self) -> list[tuple[float, float]]:
        """The thinned series ending with the latest point, so "now" is never stale."""
        if self.latest is None or (self.kept and self.kept[-1] == self.latest):
            return self.kept
        return [*self.kept, self.latest]


class MetricsLog:
    """Appends iterations to `metrics.csv` and keeps the samples the report is drawn from.

    Rows after agent step `until` (iterations past the checkpoint a run resumes from) are dropped,
    since the resumed run plays them again, and so is a row cut short when a run was killed. The file is
    rewritten row by row, so a resume costs the same memory after a hundred hours as after one.
    """

    def __init__(self, path: Path, columns: list[str], until: int = 0) -> None:
        self.path = path
        if path.exists():
            # Columns the run had before (a reward removed since) keep their history.
            with path.open(encoding="utf-8") as file:
                before = next(csv.reader(file), [])
            columns = columns + [column for column in before if column not in columns]
        self.columns = columns
        self.samples = {column: _Sample() for column in columns if column not in ("agent_steps", "iteration")}
        partial = path.with_name(path.name + ".partial")
        with partial.open("w", newline="", encoding="utf-8") as out:
            # The columns follow the rewards, which may have been changed before resuming.
            writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            if path.exists():
                with path.open(encoding="utf-8") as file:
                    for row in csv.DictReader(file):
                        if None in row.values() or float(row["agent_steps"]) > until:
                            continue
                        writer.writerow(row)
                        self._sample(row)
        for attempt in range(10):  # Windows refuses the swap while another program has it open for a moment
            try:
                os.replace(partial, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1)
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=columns)

    def _sample(self, row: dict[str, str] | dict[str, float]) -> None:
        x = float(row["agent_steps"])
        for column, sample in self.samples.items():
            value = row.get(column)
            if value not in (None, ""):
                sample.add(x, float(value))

    def write(self, row: dict[str, float]) -> None:
        self._writer.writerow(row)
        self._file.flush()
        self._sample(row)

    def close(self) -> None:
        self._file.close()

    def write_report(self, path: Path, title: str, notes: Iterable[str], warnings: Iterable[str]) -> None:
        """Rewrites the report atomically, so a browser refresh never sees half a file."""
        partial = path.with_name(path.name + ".partial")
        partial.write_text(_report(title, list(notes), list(warnings), self.samples), encoding="utf-8")
        os.replace(partial, path)


def _nice(value: float) -> str:
    """A short label: 12,345 -> 12.3K, 0.000123 -> 1.23e-4."""
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value / 1e6:.3g}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.3g}K"
    if magnitude != 0 and magnitude < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def _chart(column: str, sample: _Sample) -> str:
    title, meaning = CHARTS.get(column, (column.replace("_", " ").capitalize(), ""))
    points = [(x, y) for x, y in sample.points if math.isfinite(y)]
    if not points:
        return ""
    width, height, left, right, top, bottom = 320.0, 120.0, 4.0, 52.0, 8.0, 18.0
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = xs[0], max(xs[-1], xs[0] + 1)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-12:
        y0, y1 = y0 - 1, y1 + 1

    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (width - left - right)

    def py(y: float) -> float:
        return top + (1 - (y - y0) / (y1 - y0)) * (height - top - bottom)

    line = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
    area = f"{px(xs[0]):.1f},{py(y0):.1f} {line} {px(xs[-1]):.1f},{py(y0):.1f}"
    last_x, last_y = px(xs[-1]), py(ys[-1])
    data = json.dumps([[round(x), y] for x, y in points])
    return f"""
<figure class="chart">
  <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(meaning)}</span></figcaption>
  <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{html.escape(title)}"
       data-points='{data}' data-x0="{x0}" data-x1="{x1}"
       data-left="{left}" data-plot="{width - left - right}">
    <line class="grid" x1="{left}" x2="{width - right}" y1="{py(y1):.1f}" y2="{py(y1):.1f}"/>
    <line class="grid" x1="{left}" x2="{width - right}" y1="{py(y0):.1f}" y2="{py(y0):.1f}"/>
    <text class="tick" x="{width - right + 4}" y="{py(y1) + 4:.1f}">{_nice(y1)}</text>
    <text class="tick" x="{width - right + 4}" y="{py(y0) + 4:.1f}">{_nice(y0)}</text>
    <polygon class="wash" points="{area}"/>
    <polyline class="series" points="{line}"/>
    <circle class="end" cx="{last_x:.1f}" cy="{last_y:.1f}" r="4"/>
    <text class="tick" x="{left}" y="{height - 4}">{_nice(x0)} steps</text>
    <text class="tick end-x" x="{width - right}" y="{height - 4}">{_nice(xs[-1])}</text>
    <line class="cross" y1="{top}" y2="{height - bottom}" x1="-10" x2="-10"/>
  </svg>
  <div class="now">now <strong>{_nice(ys[-1])}</strong></div>
</figure>"""


def _report(title: str, notes: list[str], warnings: list[str], samples: dict[str, _Sample]) -> str:
    # Each reward's paid size is for the dashboard's shares; its signed pay is charted.
    charted = [c for c in samples if not c.startswith("paid_abs_")]
    order = [c for c in CHARTS if c in samples] + sorted(c for c in charted if c not in CHARTS)
    return _TEMPLATE.substitute(
        title=html.escape(title),
        notes="".join(f"<li>{html.escape(n)}</li>" for n in notes),
        warnings="".join(f'<li><span class="badge">Warning</span> {html.escape(w)}</li>' for w in warnings),
        charts="".join(_chart(column, samples[column]) for column in order),
    )
