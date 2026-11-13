"""The ways an episode can start ("state setters").

Each `[[state_setters]]` block picks one by `name` and gives it a `weight`. When an episode starts, one
setter is picked at random in proportion to the weights. Example: kickoffs half the time, random
situations the other half:

    [[state_setters]]
    name = "kickoff"
    weight = 1

    [[state_setters]]
    name = "random"
    weight = 1

If you list no `[[state_setters]]`, every episode starts with a normal kickoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from nitro.config.loader import setting


@dataclass(frozen=True)
class _StateSetter:
    # True: every car is drawn the same way. False: the first or a Blue car gets a scripted start and the rest
    # wait parked, so one side is favoured: Blue (the learner in self-play) mostly, Orange in save.
    # No default: each setter says.
    even_start: ClassVar[bool]
    weight: float = setting(1.0, "How often this setter is picked, compared with the others.", minimum=0.0)


@dataclass(frozen=True)
class Kickoff(_StateSetter):
    """A normal kickoff from one of the real kickoff spots."""

    even_start: ClassVar[bool] = True
    name: Literal["kickoff"] = "kickoff"
    fuzz_uu: float = setting(
        0.0,
        "Moves each car by up to this many units in a random direction, so kickoffs vary a little "
        "(a car is about 120 units long). 0 = exact kickoff spots.",
        minimum=0.0,
        maximum=300.0,
    )


@dataclass(frozen=True)
class RandomState(_StateSetter):
    """Ball and cars anywhere on the field with random speeds, directions and boost."""

    even_start: ClassVar[bool] = True
    name: Literal["random"] = "random"
    max_ball_speed: float = setting(
        2000.0,
        "Fastest the ball may start, in units per second (its top speed is 6000).",
        minimum=0.0,
        maximum=6000.0,
    )
    ground_chance: float = setting(
        0.5,
        "Chance that each car starts on its wheels on the floor instead of in the air.",
        minimum=0.0,
        maximum=1.0,
    )


@dataclass(frozen=True)
class BallTouch(_StateSetter):
    """Practice: the ball at rest on the floor and the car at rest on its wheels some distance away,
    facing a random direction. Used by the ball-touch task."""

    even_start: ClassVar[bool] = True
    name: Literal["ball_touch"] = "ball_touch"
    min_distance_uu: float = setting(
        1000.0, "Closest the car may start to the ball, in units.", minimum=250.0, maximum=4000.0
    )
    max_distance_uu: float = setting(
        3000.0,
        "Farthest the car may start from the ball, in units (at most 4000, so there is always room).",
        minimum=250.0,
        maximum=4000.0,
    )

    def problems(self) -> list[tuple[str, str]]:
        if self.max_distance_uu < self.min_distance_uu:
            return [("max_distance_uu", f"must be at least min_distance_uu ({self.min_distance_uu})")]
        return []


@dataclass(frozen=True)
class Airborne(_StateSetter):
    """Practice: the car in the air, turned and spinning at random, with the ball at rest on the floor,
    anywhere on the field or, with a ball distance, that far from the ball. Used by the airborne-recovery
    task."""

    even_start: ClassVar[bool] = True
    name: Literal["airborne"] = "airborne"
    min_height_uu: float = setting(
        300.0, "Lowest the car may start above the floor, in units.", minimum=100.0, maximum=1800.0
    )
    max_height_uu: float = setting(
        1200.0, "Highest the car may start above the floor, in units.", minimum=100.0, maximum=1800.0
    )
    min_ball_distance_uu: float = setting(
        0.0,
        "With max_ball_distance_uu: the closest the car may start to the ball along the floor, in units.",
        minimum=0.0,
        maximum=4000.0,
    )
    max_ball_distance_uu: float = setting(
        0.0,
        "The farthest the car may start from the ball along the floor, in units (at most 4000); 0 = "
        "anywhere on the field.",
        minimum=0.0,
        maximum=4000.0,
    )

    def problems(self) -> list[tuple[str, str]]:
        found = []
        if self.max_height_uu < self.min_height_uu:
            found.append(("max_height_uu", f"must be at least min_height_uu ({self.min_height_uu})"))
        if self.max_ball_distance_uu > 0 and self.min_ball_distance_uu < 250:
            found.append(("min_ball_distance_uu", "must be at least 250 with a max_ball_distance_uu"))
        if 0 < self.max_ball_distance_uu < self.min_ball_distance_uu:
            found.append(
                (
                    "max_ball_distance_uu",
                    f"must be at least min_ball_distance_uu ({self.min_ball_distance_uu})",
                )
            )
        if self.max_ball_distance_uu == 0 and self.min_ball_distance_uu > 0:
            found.append(("max_ball_distance_uu", "must be set (above 0) with a min_ball_distance_uu"))
        return found


@dataclass(frozen=True)
class BoostPickup(_StateSetter):
    """Practice: the car at rest on its wheels somewhere on the field with no boost, every boost pad
    available. Used by the boost-pickup task."""

    even_start: ClassVar[bool] = True
    name: Literal["boost_pickup"] = "boost_pickup"


@dataclass(frozen=True)
class Shot(_StateSetter):
    """Practice: an empty-net shot. The ball at rest in front of the goal the car attacks, the car a short
    way behind it facing the ball. Used by the shot task."""

    even_start: ClassVar[bool] = False
    name: Literal["shot"] = "shot"
    min_distance_uu: float = setting(
        1500.0, "Closest the ball may start to the goal line, in units.", minimum=500.0, maximum=4500.0
    )
    max_distance_uu: float = setting(
        3500.0, "Farthest the ball may start from the goal line, in units.", minimum=500.0, maximum=4500.0
    )
    car_distance_uu: float = setting(
        1000.0, "How far behind the ball the car starts, in units.", minimum=250.0, maximum=2500.0
    )

    def problems(self) -> list[tuple[str, str]]:
        if self.max_distance_uu < self.min_distance_uu:
            return [("max_distance_uu", f"must be at least min_distance_uu ({self.min_distance_uu})")]
        return []


@dataclass(frozen=True)
class Save(_StateSetter):
    """Practice: a save. The ball rolls from the car's own half toward its goal; the car waits in the goal
    facing the field. Used by the save task."""

    even_start: ClassVar[bool] = False
    name: Literal["save"] = "save"
    min_speed: float = setting(
        1200.0,
        "Slowest the ball may roll toward the goal, in units per second.",
        minimum=200.0,
        maximum=4000.0,
    )
    max_speed: float = setting(
        2200.0,
        "Fastest the ball may roll toward the goal, in units per second.",
        minimum=200.0,
        maximum=4000.0,
    )

    def problems(self) -> list[tuple[str, str]]:
        if self.max_speed < self.min_speed:
            return [("max_speed", f"must be at least min_speed ({self.min_speed})")]
        return []


@dataclass(frozen=True)
class AerialBall(_StateSetter):
    """Practice: an aerial. The ball hangs or rises in the air some way off; the car waits on its wheels
    with a full tank, facing it. Used by the aerial-touch task."""

    even_start: ClassVar[bool] = False
    name: Literal["aerial_ball"] = "aerial_ball"
    min_height_uu: float = setting(
        400.0, "Lowest the ball may start above the floor, in units.", minimum=300.0, maximum=1800.0
    )
    max_height_uu: float = setting(
        1000.0, "Highest the ball may start above the floor, in units.", minimum=300.0, maximum=1800.0
    )
    max_ball_speed: float = setting(
        500.0,
        "Fastest the ball may be thrown, in units per second (mostly upward, so it hangs a moment).",
        minimum=0.0,
        maximum=2000.0,
    )

    def problems(self) -> list[tuple[str, str]]:
        if self.max_height_uu < self.min_height_uu:
            return [("max_height_uu", f"must be at least min_height_uu ({self.min_height_uu})")]
        return []


@dataclass(frozen=True)
class Replay(_StateSetter):
    """A real situation: a moment from a recording of a real match (the five recordings RocketSim ships, 25
    minutes of 1v1, 2v2 and 3v3 play by top players and bots, or recordings made from your own replays).
    From each team the cars nearest the ball stand in for the game's teams; half the moments
    are turned around so both ends of the field are used alike. Jump and flip state is not carried over."""

    even_start: ClassVar[bool] = True
    name: Literal["replay"] = "replay"
    files: tuple[Path, ...] = setting(
        (),
        "Recording files (.rlpr or .rlpr.zst), with * and ? wildcards, relative to this settings file; e.g. "
        '["recordings/*.rlpr"].',
    )
    bundled: bool = setting(
        False,
        "Also use the five recordings RocketSim ships (119 MB, downloaded once the first time a run needs "
        "them, or ahead with `nitro fetch-recordings`). A recording with too few cars per team for your game "
        "is left out (the 1v1 one in a 2v2 run).",
    )
    air_bias: float = setting(
        0.0,
        "Prefer moments with the ball or a car high up: a moment's chance is 1 + air_bias x (its height / "
        "the ceiling height). 0 = every moment alike; 10 = a ceiling-high moment is drawn 11 times as often.",
        minimum=0.0,
        maximum=10.0,
    )

    def problems(self) -> list[tuple[str, str]]:
        from nitro.recordings import missing_replay_files

        missing = missing_replay_files(self)
        return [("files", missing)] if missing else []


def metrics_label(setter: StateSetter) -> str:
    """The setter's name in the per-start columns of metrics.csv."""
    return setter.name


StateSetter = Kickoff | RandomState | BallTouch | Airborne | BoostPickup | Shot | Save | AerialBall | Replay

DEFAULT_STATE_SETTERS: tuple[StateSetter, ...] = (Kickoff(),)
