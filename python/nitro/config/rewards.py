"""The rewards you can use in a settings file.

Each `[[rewards]]` block in your settings file picks one reward by its `name` and gives it a `weight`
(how much it counts). Positive weights encourage the behaviour, negative weights discourage it. Example:

    [[rewards]]
    name = "strong_touch"
    weight = 60
    min_speed_kph = 20
    max_speed_kph = 100

Any reward can be made "zero-sum" by adding a `zero_sum` line. The reward is then shared with your
teammates and the opponents' reward is subtracted, so the bot can't farm it when it helps nobody:

    zero_sum = { team_spirit = 0.5, opponent_scale = 1.0 }

Rewards that are paid every step (marked "every step" below) are automatically scaled when you change
`tick_skip`, so the same weights mean the same thing at any tick skip.

Several rewards can be tied together with `all_of`, which pays only when every part is positive (an
"AND": here, moving toward the ball counts only while lined up to send it goalward):

    [[rewards]]
    name = "all_of"
    weight = 2
    parts = [{ name = "velocity_player_to_ball" }, { name = "alignment" }]

A reward can be switched off without deleting it: `enabled = false` keeps its block and settings for later,
but the run neither pays nor measures it (the dashboard's on/off switch).

If you list no `[[rewards]]` at all, a starting mix is used (see `DEFAULT_REWARDS` at the bottom of this
file).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from nitro.config.loader import setting


@dataclass(frozen=True)
class ZeroSum:
    """Shares a reward with teammates and subtracts what the opponents get."""

    team_spirit: float = setting(
        0.0,
        "0 = each player keeps their own reward, 1 = everyone on the team gets the team's average.",
        minimum=0.0,
        maximum=1.0,
    )
    opponent_scale: float = setting(
        1.0,
        "How much of the opponents' average reward is subtracted (1 = all of it).",
        minimum=0.0,
    )


# The built-in rewards that depend on the state alone (paid every step).
STATE_REWARDS = frozenset(
    (
        "air",
        "face_ball",
        "velocity_player_to_ball",
        "velocity_ball_to_goal",
        "save_boost",
        "alignment",
    )
)


@dataclass(frozen=True)
class _Reward:
    weight: float = setting(1.0, "How much this reward counts. Negative values turn it into a penalty.")
    zero_sum: ZeroSum | None = None
    enabled: bool = setting(
        True,
        "false switches the reward off: it stays written with its settings but is neither paid nor "
        "measured, as if its block were not there.",
    )

    def problems(self) -> list[tuple[str, str]]:
        return []


@dataclass(frozen=True)
class Air(_Reward):
    """Every step: 1 while the car is off the ground. Small weights help bots learn to leave the floor."""

    name: Literal["air"] = "air"


@dataclass(frozen=True)
class FaceBall(_Reward):
    """Every step: how directly the car's nose points at the ball (1 = straight at it, -1 = away)."""

    name: Literal["face_ball"] = "face_ball"


@dataclass(frozen=True)
class VelocityPlayerToBall(_Reward):
    """Every step: how fast the car moves toward the ball, as a fraction of the car's top speed."""

    name: Literal["velocity_player_to_ball"] = "velocity_player_to_ball"


@dataclass(frozen=True)
class VelocityBallToGoal(_Reward):
    """Every step: how fast the ball moves toward the opponent's goal, as a fraction of its top speed."""

    name: Literal["velocity_ball_to_goal"] = "velocity_ball_to_goal"


@dataclass(frozen=True)
class TouchBall(_Reward):
    """1 on each step where the car touches the ball."""

    name: Literal["touch_ball"] = "touch_ball"


@dataclass(frozen=True)
class StrongTouch(_Reward):
    """On a touch: rewards hitting the ball hard, by how much the touch changed the ball's speed: 0 up to
    `min_speed_kph`, rising in a straight line to 1 at `max_speed_kph`."""

    name: Literal["strong_touch"] = "strong_touch"
    min_speed_kph: float = setting(20.0, "Changes in ball speed up to this (km/h) earn nothing.", minimum=0.0)
    max_speed_kph: float = setting(100.0, "A change in ball speed of this much (km/h) earns the full reward.")

    def problems(self) -> list[tuple[str, str]]:
        found = super().problems()
        if self.max_speed_kph <= self.min_speed_kph:
            found.append(("max_speed_kph", "must be larger than min_speed_kph"))
        return found


@dataclass(frozen=True)
class PickupBoost(_Reward):
    """When boost goes up: the gain in the square root of boost, so pickups count more on an empty tank."""

    name: Literal["pickup_boost"] = "pickup_boost"


@dataclass(frozen=True)
class SaveBoost(_Reward):
    """Every step: (boost / 100) to the power `exponent`. Below 1, the first bits of boost matter most."""

    name: Literal["save_boost"] = "save_boost"
    exponent: float = setting(0.5, "Shape of the curve; 0.5 = square root, 1 = straight line.", minimum=0.01)


@dataclass(frozen=True)
class Bump(_Reward):
    """1 when the car bumps an opponent."""

    name: Literal["bump"] = "bump"


@dataclass(frozen=True)
class Demo(_Reward):
    """1 when the car demolishes an opponent."""

    name: Literal["demo"] = "demo"


@dataclass(frozen=True)
class Goal(_Reward):
    """1 for every player of the team that scores; `concede_scale` for every player of the other team."""

    name: Literal["goal"] = "goal"
    concede_scale: float = setting(-1.0, "What conceding is worth compared with scoring (-1 = equally bad).")


@dataclass(frozen=True)
class Alignment(_Reward):
    """Every step: how well the car, the ball and the opponent's goal line up. 1 = the ball is exactly between
    the car and the goal (a touch would send it goalward), -1 = the car is between the ball and the goal."""

    name: Literal["alignment"] = "alignment"


@dataclass(frozen=True)
class AllOf(_Reward):
    """Several rewards tied together as an "AND" (the KRC of the Lucy-SKG paper): positive only when every
    part is positive, and then their geometric mean; 0 when any part is 0; negative (minus the geometric mean
    of their sizes) when any part is negative. Paid every step if every part is; the group carries the weight
    and the zero-sum setting, its parts do not."""

    name: Literal["all_of"] = "all_of"
    parts: tuple[Reward, ...] = setting(
        (),
        'The rewards to tie together, at least two, each as { name = "..." } with its own parameters; e.g. '
        '[{ name = "velocity_player_to_ball" }, { name = "alignment" }].',
    )

    def problems(self) -> list[tuple[str, str]]:
        found = super().problems()
        if len(self.parts) < 2:
            found.append(("parts", "needs at least two parts"))
        for i, part in enumerate(self.parts):
            if isinstance(part, AllOf):
                found.append(
                    (f"parts[{i}]", "a group cannot hold another group; list all the rewards in one")
                )
                continue
            own = part.zero_sum is not None or not part.enabled
            if part.weight != 1.0 or own:
                found.append(
                    (f"parts[{i}]", "a part has no weight, zero_sum or enabled of its own; the group has it")
                )
            found.extend((f"parts[{i}].{path}", message) for path, message in _part_problems(part))
        return found


@dataclass(frozen=True)
class CustomKind:
    """A reward of one's own as the compiled core lists it."""

    name: str
    about: str
    every_step: bool
    file: str
    # (setting, Rust type, optional, description, its value when left out as JSON or "" when not known)
    params: tuple[tuple[str, str, bool, str, str], ...]


@functools.cache
def custom_kinds() -> dict[str, CustomKind]:
    """The rewards of one's own compiled into the core; none without a core."""
    try:
        from nitro import _core

        listed = _core.custom_rewards()
    except (ImportError, AttributeError):
        return {}
    return {
        name: CustomKind(name, about, every_step, file, tuple((*p, "")[:5] for p in params))
        for name, about, every_step, file, params in listed
    }


_INTEGERS = {"i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize"}


@dataclass(frozen=True)
class Custom(_Reward):
    """A reward of one's own, written in Rust and compiled into NitroLearn. Its block names it and holds its
    own settings beside the weight and the rest:

        [[rewards]]
        name = "flip_reset_touch"
        weight = 20
        scale = 1.5
    """

    name: str = ""  # the reward's own name; no fixed name, so the settings loader asks `accepts`
    settings: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def accepts(cls, name: str) -> bool:
        return name in custom_kinds()

    @classmethod
    def names(cls) -> list[str]:
        return list(custom_kinds())

    @classmethod
    def build(
        cls,
        raw: dict[str, Any],
        path: str,
        build: Callable[[type[Any], dict[str, Any], str], Any],
        report: Callable[[str, str, str | None], None],
    ) -> Custom:
        """The reward a `[[rewards]]` block names, its own settings checked against the compiled reward."""
        kind = custom_kinds()[raw["name"]]
        shared = {f.name for f in dataclasses.fields(cls)} - {"settings"}
        made: Custom = build(cls, {k: v for k, v in raw.items() if k in shared}, path)
        own = {k: v for k, v in raw.items() if k not in shared}
        params = {name: rust_type for name, rust_type, *_ in kind.params}
        for key, value in own.items():
            where = f"{path}.{key}"
            if key not in params:
                listed = ", ".join(params) or "none"
                report(where, f"'{key}' is not a setting of {kind.name}", f"its own settings: {listed}")
            elif problem := _rust_type_problem(params[key], value):
                report(where, problem, None)
        for name, _, optional, *_ in kind.params:
            if not optional and name not in own:
                report(f"{path}.{name}", "this setting is required but missing", None)
        return dataclasses.replace(made, settings=tuple(sorted(own.items())))


def _rust_type_problem(rust_type: str, value: Any) -> str | None:
    if rust_type in ("f32", "f64"):
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        return None if ok else f"must be a number, but is {value!r}"
    if rust_type in _INTEGERS:
        if not isinstance(value, int) or isinstance(value, bool):
            return f"must be a whole number, but is {value!r}"
        return "must be 0 or more" if rust_type.startswith("u") and value < 0 else None
    if rust_type == "bool":
        return None if isinstance(value, bool) else f"must be true or false, but is {value!r}"
    if rust_type == "String":
        return None if isinstance(value, str) else f"must be text in quotes, but is {value!r}"
    return None  # another type: the core checks it when the run starts


def _part_problems(part: Reward) -> list[tuple[str, str]]:
    problems = getattr(part, "problems", None)
    return list(problems()) if problems else []


Reward = (
    Air
    | FaceBall
    | VelocityPlayerToBall
    | VelocityBallToGoal
    | TouchBall
    | StrongTouch
    | PickupBoost
    | SaveBoost
    | Bump
    | Demo
    | Goal
    | Alignment
    | AllOf
    | Custom
)

# The starting mix a run uses when its settings list no rewards.
DEFAULT_REWARDS: tuple[Reward, ...] = (
    Air(weight=0.25),
    FaceBall(weight=0.25),
    VelocityPlayerToBall(weight=4.0),
    StrongTouch(weight=60.0, min_speed_kph=20.0, max_speed_kph=100.0),
    VelocityBallToGoal(weight=2.0, zero_sum=ZeroSum(team_spirit=1.0)),
    PickupBoost(weight=10.0),
    SaveBoost(weight=0.2),
    Bump(weight=20.0, zero_sum=ZeroSum(team_spirit=0.5)),
    Demo(weight=80.0, zero_sum=ZeroSum(team_spirit=0.5)),
    Goal(weight=150.0),
)
