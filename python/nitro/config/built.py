"""Rewards built without code: the building blocks and formulas.

A built reward is a sum of *terms*. Each term pays a *measure* (or several multiplied together) times its
weight, on the steps where all of its *conditions* hold. Nothing is compiled: NitroLearn reads the blocks
and adds them up as the game runs. Example, a reward for touching the ball high in the air and for chasing
the ball while on the ground:

    [[rewards]]
    name = "built"
    title = "touch_up_high"
    weight = 20

    [[rewards.terms]]
    when = ["in_air", "touching_ball", { ball_above = 300 }]
    measure = "ball_height"

    [[rewards.terms]]
    when = ["on_ground"]
    measure = "speed_toward_ball"
    weight = 0.05

A condition is a name in quotes ("in_air") or a name and a number ({ ball_above = 300 }); { not = "in_air" }
turns one around. Distances and heights are in game units (the ball is about 92 units across; the field is
10240 long and the ceiling is 2044 high), speeds in km/h, angles in degrees, boost 0 to 100.

The reward's own weight and zero_sum (team spirit, opponents' share) work as for any other reward. A built
reward of measures of the state alone ("every step" below) is scaled to the tick skip like the built-in
ones.

Advanced rewards use formulas: a term can pay `value = "formula"` instead of a measure, a condition
can be a formula ("me.z > 300 and ball.speed > 1000"), and a reward can keep variables from step to step:

    [[rewards.vars]]
    name = "air"                                  # seconds in the air
    start = 0
    next = "if(me.on_ground, 0, air + dt)"

    [[rewards.terms]]
    when = "me.touch"
    value = "min(min(prev(air), 1.75) / 1.75, ball.z / 2044)"

Every word a formula knows is listed below (CAR_FIELDS, BALL_FIELDS, NAMES, FUNCTIONS) and in
docs/reward_builder.md. NitroLearn checks every formula before training starts.
"""

from __future__ import annotations

import math

# Conditions that are a plain name: what each means.
FLAGS: dict[str, str] = {
    "in_air": "The car is off the ground (no wheels on anything).",
    "on_ground": "The car is on its wheels, on the floor, a wall or the ceiling.",
    "on_floor": "The car is on its wheels on the floor (not a wall).",
    "on_wall": "The car is on its wheels on a wall or the ceiling.",
    "touching_ball": "The car touched the ball during this step.",
    "has_flip": "The car can still jump or flip (always true on the ground).",
    "supersonic": "The car is going supersonic (about 79 km/h and up).",
    "boosting": "The car is using boost this step.",
    "own_half": "The car is in its own half of the field.",
    "opponent_half": "The car is in the other team's half.",
    "ball_own_half": "The ball is in the car's own half.",
    "ball_opponent_half": "The ball is in the other team's half.",
    "kickoff": "It is a kickoff: the ball sits still on the centre spot.",
    "closest_to_ball": "The car is its team's nearest to the ball.",
    "last_to_touch": "The car was the last to touch the ball.",
}

# Conditions with a number: what each means and what the number is.
NUMBERED: dict[str, str] = {
    "ball_above": "The ball is higher than this (game units; resting on the floor it is at 93).",
    "ball_below": "The ball is lower than this (game units).",
    "car_above": "The car is higher than this (game units; on its wheels on the floor it is at 17).",
    "car_below": "The car is lower than this (game units).",
    "ball_closer_than": "The ball is closer to the car than this (game units).",
    "ball_farther_than": "The ball is farther from the car than this (game units).",
    "facing_ball_within": "The car's nose points at the ball within this many degrees (0 to 180).",
    "facing_goal_within": "The car's nose points at the other goal within this many degrees (0 to 180).",
    "boost_above": "The car has more boost than this (0 to 100).",
    "boost_below": "The car has less boost than this (0 to 100).",
    "speed_above": "The car is faster than this (km/h).",
    "speed_below": "The car is slower than this (km/h).",
    "ball_speed_above": "The ball is faster than this (km/h).",
    "ball_speed_below": "The ball is slower than this (km/h).",
}

# Measures: what each pays, and whether it is of the state alone (paid every step, scaled to the tick skip).
MEASURES: dict[str, tuple[str, bool]] = {
    "one": ("Every step: 1 (so the term pays its weight while its conditions hold).", True),
    "speed_toward_ball": (
        "Every step: how fast the car moves toward the ball, as a share of its top speed.",
        True,
    ),
    "ball_speed_toward_goal": (
        "Every step: how fast the ball moves toward the other team's goal, as a share of its top speed.",
        True,
    ),
    "car_speed": ("Every step: the car's speed, as a share of its top speed.", True),
    "ball_speed": ("Every step: the ball's speed, as a share of its top speed.", True),
    "car_height": ("Every step: the car's height, 0 on the floor to 1 at the ceiling.", True),
    "ball_height": ("Every step: the ball's height, 0 on the floor to 1 at the ceiling.", True),
    "boost": ("Every step: the car's boost, 0 empty to 1 full.", True),
    "facing_ball": (
        "Every step: how directly the car's nose points at the ball (1 = straight at it, -1 = away).",
        True,
    ),
    "alignment": (
        "Every step: how well the car is lined up behind the ball toward the other team's goal (1 = right "
        "behind it, -1 = between the ball and the goal).",
        True,
    ),
    "distance_to_ball": (
        "Every step: the car's distance to the ball, as a share of the field's length.",
        True,
    ),
    "closeness_to_ball": ("Every step: 1 at the ball, falling to 0 a field's length away.", True),
    "air_time": (
        "Every step: how long the car has been in the air, as a share of 1.75 seconds (at most 1).",
        True,
    ),
    "boost_used": ("The boost the car used this step (0 to 1 of a full tank).", False),
    "boost_gained": ("The boost the car picked up this step (0 to 1 of a full tank).", False),
    "ball_speed_change": (
        "How much the ball's speed changed this step, as a share of its top speed (hit strength).",
        False,
    ),
    "touch": ("1 when the car touches the ball.", False),
    "goal": ("1 when the car's team scores.", False),
    "concede": ("1 when the other team scores (give it a negative weight).", False),
    "shot": ("1 when the car's touch sends the ball heading into the other team's goal.", False),
    "demo": ("1 when the car demolishes an opponent.", False),
    "demoed": ("1 when the car is demolished (give it a negative weight).", False),
    "bump": ("1 when the car bumps an opponent.", False),
    "jump": ("1 when the car jumps.", False),
    "flip": ("1 when the car flips.", False),
}

# Conditions about what happened during the step: a term with one is not of the state alone.
EVENT_FLAGS = frozenset({"touching_ball"})

TERM_KEYS: dict[str, str] = {
    "when": "Conditions that must all hold for the term to pay (none: it always pays): blocks or formulas.",
    "measure": "What the term pays: one measure, or a list of them multiplied together.",
    "value": 'What the term pays, as a formula (instead of a measure), e.g. "me.z / 2044".',
    "combine": '"product" (the default) multiplies the measures; "min" pays the smallest.',
    "weight": "How much the term counts inside the reward (default 1).",
    "once_per_touch": "true: the term pays at most once between two of the car's ball touches.",
    "once_per_episode": "true: the term pays on the first step it pays in an episode only.",
    "cap_step": "The most the term pays (or charges) in one step, after its weight.",
    "cap_episode": "The most the term pays in one episode, after its weight (charges are not capped).",
    "every_step": "true: pay this term as an every-step reward (scaled to the tick skip) even though it "
    "reads an event.",
}

_NAME_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")

VAR_KEYS: dict[str, str] = {
    "name": "The variable's name in formulas (small letters, digits and _).",
    "start": "Its value when an episode starts (default 0).",
    "next": "A formula for its value each step; inside it, the variable's own name is its value before.",
}

# The fields formulas read from a car (`me.`, `mate.` nearest teammate, `opp.` nearest opponent, `it.` each
# car inside sum_mates(...) and the like). Directions and positions are marked (vector).
CAR_FIELDS: dict[str, str] = {
    "pos": "Position (vector, game units).",
    "vel": "Velocity (vector, units per second).",
    "angvel": "Spin (vector, radians per second).",
    "forward": "Where the nose points (vector of length 1).",
    "up": "Where the roof points (vector of length 1).",
    "right": "Where the right side points (vector of length 1).",
    "x": "Position across the field.",
    "y": "Position along the field (Blue defends -y).",
    "z": "Height (17 on the floor on its wheels).",
    "speed": "Speed (units per second; supersonic from 2200).",
    "boost": "Boost, 0 to 100.",
    "on_ground": "1 on its wheels on anything, else 0.",
    "on_floor": "1 on its wheels on the floor.",
    "on_wall": "1 on its wheels on a wall (not the ceiling).",
    "on_ceiling": "1 on its wheels on the ceiling.",
    "in_air": "1 with no wheels on anything.",
    "has_flip": "1 while it can still jump or flip.",
    "jumped": "1 once it has jumped (until it lands).",
    "double_jumped": "1 once it has double jumped.",
    "flipped": "1 once it has flipped (until it lands).",
    "flipping": "1 during a flip.",
    "boosting": "1 while using boost.",
    "supersonic": "1 while supersonic.",
    "demoed": "1 while demolished.",
    "air_time": "Seconds in the air.",
    "team": "+1 for Blue, -1 for Orange (multiply a y by it to see the field as the car's team does).",
    "wall_dist": "Distance to the nearest side or back wall.",
    "ceiling_dist": "Distance to the ceiling.",
    "closest": "1 when it is its team's nearest to the ball.",
    "touch": "1 when it touched the ball this step.",
    "touch_air": "1 when it touched the ball this step without touching anything else.",
    "touch_strength": "How much its touch changed the ball's velocity this step (units/s; 0 without).",
    "shot": "1 when its touch sends the ball heading into the other goal.",
    "bump": "1 when it bumps an opponent.",
    "demo": "1 when it demolishes an opponent.",
    "got_demoed": "1 on the step it is demolished.",
    "jump": "1 on the step it jumps.",
    "flip": "1 on the step it flips.",
    "regained_flip": "1 on the step it gets its jump and flip back (landing, or a flip reset).",
    "flip_reset": "1 on the step it gets its flip back while still in the air (a flip reset).",
    "last_touch": "1 while it is the last to have touched the ball.",
}

BALL_FIELDS: dict[str, str] = {
    "pos": "Position (vector).",
    "vel": "Velocity (vector).",
    "angvel": "Spin (vector).",
    "x": "Position across the field.",
    "y": "Position along the field.",
    "z": "Height (93 resting on the floor).",
    "speed": "Speed (units per second).",
    "wall_dist": "Distance to the nearest side or back wall.",
    "ceiling_dist": "Distance to the ceiling.",
    "touched": "1 when any car touched it this step.",
}

NAMES: dict[str, str] = {
    "dt": "The step's length in seconds (8 ticks: 0.067).",
    "goal": "1 when the car's team scores.",
    "concede": "1 when the other team scores.",
    "kickoff": "1 while the ball sits still on the centre spot.",
    "has_mate": "1 when the car has a teammate.",
    "has_opp": "1 when the car has an opponent.",
    "own_goal": "The centre of the car's own goal mouth (vector).",
    "opp_goal": "The centre of the other goal mouth (vector).",
    "own_net": "The back of the car's own net (vector).",
    "opp_net": "The back of the other net (vector).",
    "pi": "3.14159...",
}

FUNCTIONS: dict[str, str] = {
    "if": "if(condition, a, b): a when the condition holds, else b.",
    "min": "min(a, b, ...): the smallest.",
    "max": "max(a, b, ...): the largest.",
    "clamp": "clamp(x, low, high): x kept between low and high.",
    "abs": "abs(x): x without its sign.",
    "sign": "sign(x): 1, 0 or -1.",
    "sqrt": "sqrt(x): square root.",
    "pow": "pow(x, p): x to the power p.",
    "exp": "exp(x): e to the power x.",
    "log": "log(x): natural logarithm.",
    "decay": "decay(x, half): 1 at 0, halving every `half` (e.g. decay(dist(me.pos, ball.pos), 1000)).",
    "dot": "dot(a, b): dot product of two vectors.",
    "cross": "cross(a, b): cross product (a vector).",
    "dist": "dist(a, b): distance between two positions.",
    "angle": "angle(a, b): angle between two directions, in degrees (0 to 180).",
    "length": "length(v): a vector's length.",
    "norm": "norm(v): the vector made length 1.",
    "flat": "flat(v): the vector with its height part removed (seen from above).",
    "local": "local(v): v in the car's own frame: .x forward, .y right, .z up.",
    "vec": "vec(x, y, z): a vector.",
    "nearest_pad": "nearest_pad(p): distance (seen from above) to the nearest boost pad that is up.",
    "nearest_big_pad": "nearest_big_pad(p): the same, big pads only.",
    "prev": "prev(x): x at the step before (not for things that happen during a step, such as touches).",
    "delta": "delta(x): x now minus x at the step before.",
    "since": "since(c): seconds since c was last true this episode (1e9 before it ever was).",
    "held": "held(c): seconds c has been true without a break (0 while false).",
    "count": "count(c): how many steps c was true this episode.",
    "once": "once(c): 1 on the first step c is true this episode, else 0.",
    "sum_mates": "sum_mates(x): x added up over the teammates (`it` is each).",
    "sum_opps": "sum_opps(x): x added up over the opponents.",
    "min_mates": "min_mates(x): the smallest x over the teammates (a huge number with none).",
    "min_opps": "min_opps(x): the smallest x over the opponents.",
    "max_mates": "max_mates(x): the largest x over the teammates (a huge negative number with none).",
    "max_opps": "max_opps(x): the largest x over the opponents.",
    "count_mates": "count_mates(c): how many teammates c holds for.",
    "count_opps": "count_opps(c): how many opponents c holds for.",
}


# The largest number the core holds (a 32-bit float); past it a weight would pay infinity.
_LARGEST = 3.4028234663852886e38
# How many { not = ... } may be wrapped around one another (the core reads settings only so deep).
MAX_NOTS = 8


def is_number(value: object) -> bool:
    """A number the core can hold: not true/false, inf, nan or past a 32-bit float."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and abs(value) <= _LARGEST
    )


def condition_problem(written: object, known: frozenset[str] = frozenset(), nots: int = 0) -> str | None:
    """What is wrong with one written condition, if anything (`known`: the reward's variable names)."""
    if isinstance(written, str):
        plain = all(c.isalnum() or c == "_" for c in written)
        # A name that is not a block is a mistake; anything else is a formula, which the core checks.
        named = written in FLAGS or written in NAMES or written in known
        return f"there is no condition called '{written}'" if plain and not named else None
    if isinstance(written, dict) and len(written) == 1:
        ((key, value),) = written.items()
        if key == "not":
            if nots >= MAX_NOTS:
                return f"{{ not = ... }} can be wrapped around a condition at most {MAX_NOTS} times"
            return condition_problem(value, known, nots + 1)
        if key not in NUMBERED:
            return f"there is no condition called '{key}'"
        if not is_number(value):
            return f"{key} needs a number, not {value!r}"
        if key.startswith("facing_") and not 0 <= value <= 180:
            return f"{key} needs 0 to 180 degrees"
        return None
    return f"a condition is a name or a formula in quotes, or {{ name = number }}, not {written!r}"


def is_event_condition(written: object) -> bool:
    if isinstance(written, dict) and "not" in written:
        return is_event_condition(written["not"])
    return isinstance(written, str) and written in EVENT_FLAGS


def as_list(value: object) -> list[object]:
    """A setting that may be one item or a list of them, as a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def term_every_step(term: dict[str, object]) -> bool:
    """A term of blocks alone that is of the state alone (every step): its measures are, it has no event
    condition and pays as is. Terms with formulas: the core decides (`core_check`)."""
    measures = as_list(term.get("measure"))
    paid_as_is = not any(
        term.get(k) for k in ("once_per_touch", "once_per_episode", "cap_step", "cap_episode")
    )
    return (
        paid_as_is
        and all(MEASURES.get(str(m), ("", False))[1] for m in measures)
        and not any(is_event_condition(c) for c in as_list(term.get("when")))
    )


def term_problems(term: object, known: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """What is wrong with one written term that can be told without the core: (setting, message) pairs.
    Formulas are checked by the core (`nitro._core.built_check`)."""
    if not isinstance(term, dict):
        return [("", 'a term is a table: { measure = ... or value = "formula", when = [...], weight = ... }')]
    found = [
        (
            key,
            "potential terms are not in this version of NitroLearn"
            if key == "potential"
            else f"'{key}' is not a setting of a term",
        )
        for key in term
        if key not in TERM_KEYS
    ]
    for i, condition in enumerate(as_list(term.get("when"))):
        if problem := condition_problem(condition, known):
            found.append((f"when[{i}]", problem))
    measures = as_list(term.get("measure"))
    value = term.get("value")
    if value is not None and measures:
        found.append(("value", "a term has a measure or a value, not both"))
    elif value is None and not measures:
        found.append(
            ("measure", 'a term needs a measure ("one" pays 1 while its conditions hold) or a value')
        )
    elif value is not None and (not isinstance(value, str) or not value.strip()):
        found.append(("value", "value is a formula in quotes"))
    for measure in measures:
        if not isinstance(measure, str) or measure not in MEASURES:
            found.append(("measure", f"there is no measure called '{measure}'"))
    if term.get("combine", "product") not in ("product", "min"):
        found.append(("combine", 'combine is "product" or "min"'))
    for key in ("weight", "cap_step", "cap_episode"):
        number = term.get(key, 1.0)
        if not is_number(number):
            found.append((key, f"{key} must be a number, not {number!r}"))
    for key in ("once_per_touch", "once_per_episode", "every_step"):
        if not isinstance(term.get(key, False), bool):
            found.append((key, f"{key} is true or false"))
    return found


def var_problems(var: object) -> list[tuple[str, str]]:
    """What is wrong with one written variable that can be told without the core."""
    if not isinstance(var, dict):
        return [("", 'a variable is a table: { name = "x", start = 0, next = "formula" }')]
    found = [(key, f"'{key}' is not a setting of a variable") for key in var if key not in VAR_KEYS]
    name = var.get("name")
    if not isinstance(name, str) or not name[:1].islower() or not all(c in _NAME_LETTERS for c in name):
        found.append(("name", f"a variable's name is small letters, digits and _, not {name!r}"))
    if not isinstance(var.get("next"), str):
        found.append(("next", 'next = "formula" is needed'))
    start = var.get("start", 0)
    if not is_number(start):
        found.append(("start", f"start must be a number, not {start!r}"))
    return found


def core_check(title: str, vars: list[object], terms: list[object]) -> tuple[str, str, bool, int]:
    """The core's own check of a whole built reward: (where, problem, every step, numbers remembered),
    `where` being "terms[0].value", "vars[1]" and the like ("" for the reward), problem "" when it is fine."""
    import json

    from nitro import _core

    problem, every_step, memory = _core.built_check(
        json.dumps({"title": title, "vars": vars, "terms": terms})
    )
    return (
        (_place(problem, vars), _plain(problem), every_step, memory)
        if problem
        else ("", "", every_step, memory)
    )


def _place(problem: str, vars: list[object]) -> str:
    """Where in the block the core's message points ("term 2: when[0]: ..." -> "terms[1].when[0]")."""
    head, _, rest = problem.partition(": ")
    if head.startswith("term ") and head[5:].isdigit():
        inner, _, _ = rest.partition(": ")
        part = inner if inner.startswith(("when[", "value")) else ""
        return f"terms[{int(head[5:]) - 1}]" + (f".{part}" if part else "")
    if head.startswith("variable "):
        name = head[9:].strip("'")
        if name.isdigit():
            return f"vars[{int(name) - 1}]"
        names = [v.get("name") if isinstance(v, dict) else None for v in vars]
        return f"vars[{names.index(name)}]" if name in names else "vars"
    return ""


def _plain(problem: str) -> str:
    """The core's message without the place it names (the settings check says the place itself)."""
    head, _, rest = problem.partition(": ")
    if head.startswith("term ") and head[5:].isdigit():
        inner, _, after = rest.partition(": ")
        return after if inner.startswith(("when[", "value")) and after else rest
    return rest if head.startswith("variable ") and rest else problem
