"""Skill ratings with uncertainty for self-play.

The Weng-Lin Bayesian approximation (Weng & Lin 2011, the "OpenSkill" model) in its Bradley-Terry form
for a game between two players, written from the paper. Each player has a mean skill `mu` and an
uncertainty `sigma`; a result moves the mean toward what happened by an amount that shrinks as the player
becomes known. Everything is in plain floats, unit-tested against hand-computed values. Used only for
matchmaking and for reading progress, never as evidence for a setting.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# OpenSkill's defaults: a new player at 25 ± 25/3; beta is the per-game performance spread.
MU = 25.0
SIGMA = MU / 3
BETA = SIGMA / 2
KAPPA = 0.0001  # keeps a variance from collapsing to 0
MIN_WEIGHT = 1e-3  # every past version keeps a small chance of being picked

Matchmaking = Literal["variance", "uniform", "hardest", "squared", "linear_capped"]


@dataclass(frozen=True)
class Rating:
    mu: float = MU
    sigma: float = SIGMA

    @property
    def ordinal(self) -> float:
        """A single conservative number: the mean less three standard deviations (OpenSkill's `ordinal`)."""
        return self.mu - 3 * self.sigma


def win_probability(a: Rating, b: Rating) -> float:
    """How likely `a` is to beat `b` (Bradley-Terry with both uncertainties)."""
    c = math.sqrt(a.sigma**2 + b.sigma**2 + 2 * BETA**2)
    return 1.0 / (1.0 + math.exp((b.mu - a.mu) / c))


def update(a: Rating, b: Rating, score: float, min_sigma: float = 0.0) -> tuple[Rating, Rating]:
    """New ratings after one game: `score` is 1 if `a` won, 0 if `b` won, 0.5 for a draw.

    `min_sigma` floors `a`'s uncertainty only: the learner (always `a`) keeps changing, so its rating must
    stay movable, while a frozen snapshot's rating may settle as its games accumulate.
    """
    c = math.sqrt(a.sigma**2 + b.sigma**2 + 2 * BETA**2)
    p = win_probability(a, b)

    def one(me: Rating, mine: float, expected: float, floor: float) -> Rating:
        # Weng & Lin 2011, Bradley-Terry full pairing: Omega moves the mean, Delta shrinks the variance,
        # with OpenSkill's gamma = sigma / c.
        omega = me.sigma**2 / c * (mine - expected)
        gamma = me.sigma / c
        delta = gamma * me.sigma**2 / c**2 * expected * (1 - expected)
        sigma = math.sqrt(me.sigma**2 * max(1 - delta, KAPPA))
        return Rating(me.mu + omega, max(sigma, floor))

    return one(a, score, p, min_sigma), one(b, 1 - score, 1 - p, 0.0)


def matchmaking_weights(learner: Rating, pool: Sequence[Rating], rule: Matchmaking) -> list[float]:
    """How much each past version deserves to be picked (unnormalised, all above 0), from the learner's
    chance `p` of beating it: 'variance' p(1-p) (even matches, AlphaStar's rule), 'uniform' 1, 'hardest' 1-p,
    'squared' (1-p)^2 (the hardest ones much more, AlphaStar's "hard" rule), 'linear_capped' min(0.5, 1-p)
    (the hardest ones, but every version the bot beats at most half the time alike)."""
    weights = []
    for opponent in pool:
        p = win_probability(learner, opponent)
        weight = {
            "variance": p * (1 - p),
            "uniform": 1.0,
            "hardest": 1 - p,
            "squared": (1 - p) ** 2,
            "linear_capped": min(0.5, 1 - p),
        }[rule]
        weights.append(max(weight, MIN_WEIGHT))
    return weights


def draw(weights: Sequence[float], count: int, rng: random.Random) -> list[int]:
    """`count` distinct indices drawn without replacement, in proportion to `weights`."""
    remaining = list(range(len(weights)))
    chosen: list[int] = []
    while remaining and len(chosen) < count:
        picked = rng.choices(remaining, weights=[weights[i] for i in remaining], k=1)[0]
        chosen.append(picked)
        remaining.remove(picked)
    return chosen
