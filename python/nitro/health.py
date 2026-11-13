"""Training-health checks: training that goes wrong must fail loudly, not quietly learn nothing.

`HealthMonitor` runs after every learning update and, by default, stops training with a plain-English
error. `learning_problems` is the check every benchmark applies to its runs: a run whose learning
updates did not work is invalid, however fast it was.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Sequence
from typing import Literal

from nitro.learner import LearnStats

log = logging.getLogger("nitro")

# FP16 loss scaling drops an update now and then by design (on an overflow it halves its scale and
# skips that step). Trouble is many updates in a row skipped, or a high share for several iterations.
MAX_SKIP_SHARE = 0.25
SKIP_PATIENCE = 3
MAX_CONSECUTIVE_SKIPS = 16
# In a benchmark run, an iteration without an applied update or a moved policy is tolerated in up to this
# share of the measured iterations (an isolated FP16 skip with one update per iteration).
MAX_STILL_SHARE = 0.1

_HINT = (
    "\nCommon causes: a learning rate that is too high, rewards that are far too large, or a bug. "
    'To keep training anyway, set on_training_problem = "warn" in the [run] section.'
)


class TrainingProblem(RuntimeError):
    """Training stopped because the learning update went wrong."""


class HealthMonitor:
    def __init__(self, on_problem: Literal["stop", "warn"], fp16: bool) -> None:
        self.on_problem = on_problem
        self.fp16 = fp16
        self.skip_streak = 0
        self.consecutive_skips = 0

    def check(self, stats: LearnStats) -> list[str]:
        """Problems with this iteration's update; raises `TrainingProblem` for any if set to stop."""
        problems = []
        if not stats.losses_finite:
            problems.append(
                "a loss, the entropy, the KL divergence or the explained variance is NaN or infinite"
            )
        if not stats.params_finite:
            problems.append("the network weights contain NaN or infinity")
        if stats.skipped_updates and not self.fp16:
            problems.append(
                f"{stats.skipped_updates} of {stats.gradient_steps} learning updates had NaN or infinite "
                "gradients and were skipped"
            )
        share = stats.skipped_updates / stats.gradient_steps
        self.skip_streak = self.skip_streak + 1 if share > MAX_SKIP_SHARE else 0
        # Only an iteration that skipped everything extends a run of skipped updates.
        all_skipped = stats.skipped_updates >= stats.gradient_steps
        self.consecutive_skips = self.consecutive_skips + stats.skipped_updates if all_skipped else 0
        if self.consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
            problems.append(f"the last {self.consecutive_skips} learning updates were all skipped")
        elif self.skip_streak >= SKIP_PATIENCE:
            problems.append(
                f"{share:.0%} of learning updates were skipped {self.skip_streak} iterations in a row"
            )
        if stats.grad_scale < 1.0:
            problems.append(
                f"the FP16 loss scale has fallen to {stats.grad_scale:g} (overflowing again and again)"
            )
        if problems:
            message = "Training problem: " + "; ".join(problems) + "."
            if self.on_problem == "stop":
                raise TrainingProblem(message + _HINT)
            log.warning(message)
        return problems


def learning_problems(stats: Sequence[LearnStats]) -> list[str]:
    """Why a run's learning updates did not work (empty if they did).

    Checks that the updates ran: losses and weights stay finite, updates are applied, and they move the
    weights and the policy (a non-zero KL between the policy before and after an iteration's update),
    with the entropy not frozen across iterations. An iteration without an applied update or a moved
    policy is tolerated in up to `MAX_STILL_SHARE` of the iterations (an isolated FP16 overflow). This
    does not show that the policy learns the task: that is what the narrow tasks measure.
    """
    problems = []
    broken = [n for n, s in enumerate(stats, start=1) if not (s.losses_finite and s.params_finite)]
    if broken:
        problems.append(f"NaN or infinity in losses or weights (iterations {broken[0]} to {broken[-1]})")
    for label, still in (
        ("every learning update was skipped", [s.skipped_updates >= s.gradient_steps for s in stats]),
        ("the weights did not change", [not s.param_change > 0 for s in stats]),
        (
            "the policy did not change (KL before vs after the update was 0)",
            [not s.policy_change > 0 for s in stats],
        ),
    ):
        if sum(still) > MAX_STILL_SHARE * len(stats):
            problems.append(f"{label} in {sum(still)} of {len(stats)} iterations")
    entropies = [s.entropy for s in stats]
    if len(entropies) > 1 and max(entropies) - min(entropies) <= 1e-7:
        problems.append(f"the entropy stayed at {entropies[0]} for every iteration")
    return problems


class TrendWarnings:
    """Plain-English warnings about trends that suggest changing a setting.

    They never stop training. Each kind of warning is repeated at most every `REPEAT` iterations.
    """

    HISTORY = 50
    REPEAT = 100
    # Farming: over FARM_WINDOW iterations, the goal's share of the pay in the newest FARM_RECENT
    # against the oldest FARM_RECENT, and the game length the same way.
    FARM_WINDOW = 300
    FARM_RECENT = 50
    FARM_GOAL_DROP = 0.6  # the goal's share fell below this fraction of what it was
    FARM_LONGER = 1.4  # and games became at least this much longer
    # A rating rise smaller than this is noise (about 3 percentage points of win chance).
    RATING_RISE = 1.0

    def __init__(self, flat_rating_steps: int = 0, entropy_floor: float = 0.1) -> None:
        """`entropy_floor`: the entropy below which the bot is called almost certain of its choices."""
        self.policy_change: deque[float] = deque(maxlen=self.HISTORY)
        self.value_loss: deque[float] = deque(maxlen=self.HISTORY)
        self.speed: deque[float] = deque(maxlen=self.HISTORY)
        self.negative_ev = 0
        self.iteration = 0
        self.last_warned: dict[str, int] = {}
        self.flat_rating_steps = flat_rating_steps
        self.entropy_floor = entropy_floor
        self.best_rating: float | None = None
        self.best_rating_steps = 0
        # Each iteration's reward shares, game length and the goal's share of them (kept, not summed again).
        self.farm: deque[tuple[dict[str, float], float, float]] = deque(maxlen=self.FARM_WINDOW)

    def check(
        self,
        stats: LearnStats,
        steps_per_second: float | None,
        reward_shares: dict[str, float],
        rating: tuple[float, int] | None = None,
        episode_seconds: float | None = None,
    ) -> list[str]:
        """Warnings for this iteration; `reward_shares` is each reward's share of all |reward| paid;
        `rating` is (the bot's rating against its past versions, agent steps so far) when self-play is on;
        `episode_seconds` the mean length of the games that ended this iteration, if any did.

        `steps_per_second` is None for an iteration whose speed says nothing about the PC: the first after
        the games were (re)built, which runs about 2.4x slower while everything warms up.
        """
        self.iteration += 1
        found: dict[str, str] = {}
        if rating is not None and self.flat_rating_steps > 0:
            mu, agent_steps = rating
            if self.best_rating is None or mu >= self.best_rating + self.RATING_RISE:
                self.best_rating, self.best_rating_steps = mu, agent_steps
            elif agent_steps - self.best_rating_steps >= self.flat_rating_steps:
                found["rating"] = (
                    f"The bot's rating against its past versions has not risen for "
                    f"{(agent_steps - self.best_rating_steps) / 1e6:,.0f}M agent steps "
                    f"(best {self.best_rating:.1f}). Learning may have stalled; look at the replay "
                    '(nitro watch), try selfplay.matchmaking = "hardest", or lower the learning rates.'
                )
        if stats.entropy < self.entropy_floor:
            found["entropy"] = (
                f"The bot has become almost certain of its choices (entropy {stats.entropy:.2f}), so it "
                "barely explores. If learning stalls, raise ppo.entropy a little."
            )
        if len(self.policy_change) >= 20 and stats.policy_change > 10 * statistics.median(self.policy_change):
            found["kl"] = (
                f"An update moved the policy much more than usual (KL {stats.policy_change:.2g}, typically "
                f"{statistics.median(self.policy_change):.2g}). If this keeps happening, lower ppo.policy_lr."
            )
        if len(self.value_loss) >= 20 and stats.value_loss > 10 * statistics.median(self.value_loss):
            found["value_loss"] = (
                f"The value loss jumped to {stats.value_loss:.3g} (typically "
                f"{statistics.median(self.value_loss):.3g}). Check for a reward that became much larger; "
                "if it keeps happening, lower ppo.value_lr."
            )
        self.negative_ev = self.negative_ev + 1 if stats.explained_variance < 0 else 0
        if self.negative_ev >= 20:
            found["explained_variance"] = (
                f"For {self.negative_ev} iterations the value network has predicted worse than simply "
                "guessing the average. Rewards may be too random to predict, or ppo.value_lr may be off."
            )
        paying = {name: share for name, share in reward_shares.items() if share > 0}
        if paying:
            name, share = max(paying.items(), key=lambda item: item[1])
            # The default mix pays about 70% through velocity_player_to_ball. A single
            # paying reward (the others at weight 0) is a choice, not farming.
            if share > 0.9 and len(paying) > 1:
                found["dominance"] = (
                    f"The '{name}' reward makes up {share:.0%} of all reward paid; the bot may be farming "
                    "it instead of playing. Consider lowering its weight."
                )
        if farming := self._farming(reward_shares, episode_seconds):
            found["farming"] = farming
        if (
            steps_per_second is not None
            and len(self.speed) >= 20
            and steps_per_second < 0.5 * statistics.median(self.speed)
        ):
            found["speed"] = (
                f"Training slowed to {steps_per_second:,.0f} steps/s from about "
                f"{statistics.median(self.speed):,.0f}. Is another program using the PC?"
            )
        self.policy_change.append(stats.policy_change)
        self.value_loss.append(stats.value_loss)
        if steps_per_second is not None:
            self.speed.append(steps_per_second)
        warnings = []
        for kind, text in found.items():
            if self.iteration - self.last_warned.get(kind, -self.REPEAT) >= self.REPEAT:
                self.last_warned[kind] = self.iteration
                warnings.append(text)
        return warnings

    def _farming(self, shares: dict[str, float], episode_seconds: float | None) -> str | None:
        """A farming pattern long runs can fall into: games grow longer and the
        goal's share of the pay falls while other pay rises, both sides playing for it instead of scoring."""
        goal = sum(share for label, share in shares.items() if _is_goal(label))
        if episode_seconds is None or goal <= 0:
            return None
        self.farm.append((shares, episode_seconds, goal))
        if len(self.farm) < self.FARM_WINDOW:
            return None
        rows = list(self.farm)
        old, new = rows[: self.FARM_RECENT], rows[-self.FARM_RECENT :]

        def goal_of(part: list[tuple[dict[str, float], float, float]]) -> float:
            return statistics.median(goal for _, _, goal in part)

        def seconds_of(part: list[tuple[dict[str, float], float, float]]) -> float:
            return statistics.median(seconds for _, seconds, _ in part)

        goal_old, goal_new = goal_of(old), goal_of(new)
        long_old, long_new = seconds_of(old), seconds_of(new)
        if goal_new >= self.FARM_GOAL_DROP * goal_old or long_new < self.FARM_LONGER * long_old:
            return None
        names = {name for s, _, _ in rows for name in s if not _is_goal(name)}

        def mean(part: list[tuple[dict[str, float], float, float]], name: str) -> float:
            return sum(s.get(name, 0.0) for s, _, _ in part) / len(part)

        rising = sorted(names, key=lambda name: mean(new, name) - mean(old, name), reverse=True)[:3]
        listed = ", ".join(f"{n} {mean(old, n):.0%} -> {mean(new, n):.0%}" for n in rising)
        return (
            f"The bot may be farming: games went from {long_old:.0f} s to {long_new:.0f} s and the goal's "
            f"share of the pay from {goal_old:.0%} to {goal_new:.0%}; rising most: {listed}. Consider going "
            "back to a save from before, lowering those weights or raising the goal's."
        )


def _is_goal(label: str) -> bool:
    """A reward column of the goal reward ("goal", or "goal_2" when listed twice)."""
    return label == "goal" or (label.startswith("goal_") and label[5:].isdigit())
