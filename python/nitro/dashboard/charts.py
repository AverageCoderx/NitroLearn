"""The Overview's charts: every column a bot's metrics.csv records, sorted into folders with a plain title,
what it means and its unit, and each chart's points over all of the training or its newest part, thinned on
this side so the page draws a few hundred points whatever the length of the run.

The folders are worked out from the file's own columns: a column this catalog does not know (one a newer
version adds) lands in "Other" under its name in words, so nothing recorded is ever left out."""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any

from nitro.config.rewards import custom_kinds
from nitro.dashboard import bots, describe
from nitro.dashboard.rewards import KINDS, decisions_a_minute
from nitro.runs import RunFolder

# The folders in the order the page shows them: (id, title, colour, what it holds). Folders of one kind share
# a colour (the page's categorical colours 1 to 7, in this order; "other" is neutral).
FOLDERS = (
    (
        "rewards",
        "Rewards",
        1,
        "Each reward before its weight, per step: rising means the bot does more of it.",
    ),
    ("shares", "Reward shares", 1, "Each reward's part of all the reward paid."),
    ("fires", "How often rewards pay", 1, "How many times a minute each reward pays one car."),
    ("reward_details", "Reward details", 1, "Each reward after its weight, before and after zero-sum."),
    ("play", "Game play", 2, "How the bots play: touches, speed, boost, time in the air."),
    ("mechanics", "Mechanics", 2, "Shots, saves, aerials, dribbles, flicks, flip resets, demos."),
    ("starts", "Kickoffs and starts", 2, "How the games each way of starting them go."),
    ("sizes", "Game sizes", 2, "Each game size of a bot that plays several (1v1, 2v2, 3v3)."),
    ("learning", "Learning", 3, "How the policy learns: its randomness and the size of each update."),
    ("network", "Network health", 3, "The size of the networks' weights and of each change to them."),
    ("value", "Value and critic", 4, "How well the bot predicts what a situation is worth."),
    ("speed", "Speed and hardware", 5, "How fast it trains, and where each round's time goes."),
    ("selfplay", "Self-play", 6, "Games against saved past versions of the bot, and its skill rating."),
    ("replays", "Learning from replays", 7, "Copying replays while it trains (Replays in training)."),
    ("other", "Other", 0, "Numbers this list has no folder for, under their names in metrics.csv."),
)
FOLDER_IDS = tuple(folder[0] for folder in FOLDERS)
# The charts of "Your charts" until it is changed: the ones the bot records, in this order.
DEFAULT_PINNED = (
    "reward_per_step",
    "steps_per_second",
    "touches_per_minute",
    "fired_goal",
    "episode_seconds",
    "rating_mu",
    "entropy",
    "explained_variance",
)
# Columns that are the charts' own axis or count, never charts.
_NOT_CHARTS = {"agent_steps", "iteration"}
_PER_MIN = "a minute, per car"


@dataclass(frozen=True)
class Chart:
    """One chart: its id (its column, or `share_<label>` for a reward's share), folder, plain title, what it
    means, the unit after its numbers, and the factor its column's numbers are shown multiplied by."""

    id: str
    folder: str
    title: str
    about: str
    unit: str = ""
    scale: float = 1.0
    nerd: str = ""

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "folder": self.folder,
            "title": self.title,
            "about": self.about,
            "unit": self.unit,
            "nerd": self.nerd,
        }


# Column -> (folder, title, what it means, unit, factor shown, nerd explanation).
_FIXED: dict[str, tuple[str, str, str, str, float, str]] = {
    "reward_per_step": (
        "rewards",
        "All rewards together",
        "What the bot is paid per step, all rewards added up: going up is good, but it also moves when "
        "rewards change.",
        "",
        1,
        "reward_per_step: the weighted reward per agent step after zero-sum, summed over every reward.",
    ),
    "episode_seconds": (
        "play",
        "Game length",
        "Seconds until a goal or a reset. Neither way is better on its own.",
        "s",
        1,
        "episode_seconds: mean length of the episodes that ended this iteration.",
    ),
    "touches_per_minute": (
        "play",
        "Ball touches",
        "Touches a minute per car. Going up is good early on.",
        "/min",
        1,
        "",
    ),
    "in_air_share": (
        "play",
        "Time in the air",
        "Share of the time the cars are off the ground.",
        "%",
        100,
        "",
    ),
    "speed_kmh": (
        "play",
        "Car speed",
        "The cars' average speed.",
        "km/h",
        1,
        "",
    ),
    "speed_to_ball_kmh": (
        "play",
        "Speed toward the ball",
        "High when the cars chase the ball.",
        "km/h",
        1,
        "",
    ),
    "boost_held": (
        "play",
        "Boost in the tank",
        "Average boost held, 0 to 100.",
        "",
        1,
        "",
    ),
    "supersonic_share": (
        "play",
        "Supersonic",
        "Share of the time at full speed.",
        "%",
        100,
        "",
    ),
    "demolished_share": (
        "play",
        "Time demolished",
        "Share of the time cars wait to respawn.",
        "%",
        100,
        "",
    ),
    "ball_on_car_share": (
        "play",
        "Ball on the car",
        "Share of the time the ball rests on a car (dribbling).",
        "%",
        100,
        "",
    ),
    "shots_per_minute": (
        "mechanics",
        "Shots",
        "Touches that send the ball toward the goal, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "saves_per_minute": (
        "mechanics",
        "Saves",
        "Touches that turn the ball away from its own goal, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "aerial_touches_per_minute": (
        "mechanics",
        "Aerial touches",
        "Touches in the air, a minute, per car.",
        "/min",
        1,
        "Touches with the car above 300 units.",
    ),
    "dribbles_per_minute": (
        "mechanics",
        "Dribbles",
        "The ball carried on the car for 0.75 s or more, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "flicks_per_minute": (
        "mechanics",
        "Flicks",
        "A carried ball fired off by the car's own flip, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "double_taps_per_minute": (
        "mechanics",
        "Double taps",
        "Aerial, off the backboard, the same car again, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "flip_resets_per_minute": (
        "mechanics",
        "Flip resets",
        "The flip won back by touching the ball with the wheels, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "flip_reset_height": (
        "mechanics",
        "Flip reset height",
        "How high the ball is at the flip resets (the ceiling is 2044).",
        "uu",
        1,
        "Mean ball height in Unreal units.",
    ),
    "flip_reset_goal_dist": (
        "mechanics",
        "Flip reset distance from the net",
        "How far the ball is from the opponents' goal line at the flip resets (midfield is 5120).",
        "uu",
        1,
        "",
    ),
    "reset_musty_goals_per_hour": (
        "mechanics",
        "Reset musty goals",
        "Goals within 6 s of a flip reset with a back flip between, per car and hour.",
        "/h",
        1,
        "",
    ),
    "reset_musty_goal_speed_kmh": (
        "mechanics",
        "Reset musty goal speed",
        "How fast the ball goes into the net at those goals.",
        "km/h",
        1,
        "",
    ),
    "demos_per_minute": (
        "mechanics",
        "Demolitions",
        "Cars demolished, a minute, per car.",
        "/min",
        1,
        "",
    ),
    "kickoff_touch_s_diagonal": (
        "starts",
        "Kickoff first touch, diagonal",
        "Seconds until a car first touches the ball from the diagonal spots. Lower is faster.",
        "s",
        1,
        "",
    ),
    "kickoff_touch_s_offcenter": (
        "starts",
        "Kickoff first touch, off-centre",
        "Seconds until a car first touches the ball from the spots back and to the side.",
        "s",
        1,
        "",
    ),
    "kickoff_touch_s_back": (
        "starts",
        "Kickoff first touch, straight back",
        "Seconds until a car first touches the ball from the spot straight back.",
        "s",
        1,
        "",
    ),
    "kickoff_count_diagonal": (
        "starts",
        "Diagonal kickoffs",
        "Diagonal kickoffs touched each round.",
        "",
        1,
        "",
    ),
    "kickoff_count_offcenter": (
        "starts",
        "Off-centre kickoffs",
        "Off-centre kickoffs touched each round.",
        "",
        1,
        "",
    ),
    "kickoff_count_back": (
        "starts",
        "Straight-back kickoffs",
        "Straight-back kickoffs touched each round.",
        "",
        1,
        "",
    ),
    "entropy": (
        "learning",
        "Randomness of its choices",
        "Falls slowly as it gets sure of itself. 1 is picking at random.",
        "",
        1,
        "entropy: the policy's mean entropy, scaled to 1 for uniform.",
    ),
    "policy_change": (
        "learning",
        "How far each update moved it",
        "How much the choices changed in one update. Small and steady is healthy.",
        "",
        1,
        "policy_change: exact KL divergence of the policy before and after the iteration's update, on "
        "sampled states.",
    ),
    "approx_kl": (
        "learning",
        "Change per learning step",
        "How much each learning step changed the choices.",
        "",
        1,
        "approx_kl: PPO's approximate KL divergence per minibatch.",
    ),
    "clip_fraction": (
        "learning",
        "Clipped updates",
        "Share of samples where the learning step was held back for being too large.",
        "%",
        100,
        "clip_fraction: share of samples PPO's ratio clip cut.",
    ),
    "policy_loss": (
        "learning",
        "Policy loss",
        "The policy's training loss. It moves around zero; its trend says little on its own.",
        "",
        1,
        "policy_loss: PPO's clipped surrogate loss.",
    ),
    "policy_lr": (
        "learning",
        "Policy learning rate",
        "How big the policy's learning steps are (ppo.policy_lr).",
        "",
        1,
        "",
    ),
    "update_size": (
        "learning",
        "Size of each update",
        "How far an update moved the networks' weights, relative to their size.",
        "",
        1,
        "update_size: |θ after - θ before| / |θ before|.",
    ),
    "skipped_updates": (
        "learning",
        "Skipped updates",
        "Updates dropped because a number overflowed. Zero is normal.",
        "",
        1,
        "",
    ),
    "grad_scale": (
        "learning",
        "Loss scale",
        "With FP16 precision: how far the numbers are scaled up to stay precise. It is 1 with FP32.",
        "",
        1,
        "grad_scale: the mixed-precision loss scale.",
    ),
    "reward_scale": (
        "learning",
        "Reward scale",
        "What the rewards are divided by before learning, so their size stays steady.",
        "",
        1,
        "reward_scale: the running return scale of the reward normaliser.",
    ),
    "entropy_coef": (
        "learning",
        "Exploration bonus",
        "The bonus moved to hold the randomness near its target (ppo.entropy_target).",
        "",
        1,
        "",
    ),
    "advantage_kept": (
        "learning",
        "Samples it learns from",
        "Share of the samples the policy learns from (ppo.advantage_keep).",
        "%",
        100,
        "",
    ),
    "advantage_kept_size": (
        "learning",
        "Advantage it learns from",
        "Share of the advantages' total size in the samples kept.",
        "%",
        100,
        "",
    ),
    "advantage_rows": (
        "learning",
        "Samples the learning steps run on",
        "Kept samples plus a random share of the others (ppo.advantage_keep_rest).",
        "%",
        100,
        "",
    ),
    "explained_variance": (
        "value",
        "Value prediction",
        "1 is perfect; near 0 it is still guessing.",
        "",
        1,
        "explained_variance of the value network's predictions of the returns.",
    ),
    "value_loss": (
        "value",
        "Value loss",
        "How wrong the value predictions are. Lower is better.",
        "",
        1,
        "",
    ),
    "start_value": (
        "value",
        "Value of a fresh start",
        "What the bot thinks a new game is worth, in reward.",
        "",
        1,
        "start_value: the critic's mean value of the fresh starts learned from, raw reward units.",
    ),
    "start_value_share": (
        "value",
        "Value of a fresh start, per goal",
        "The same as a share of the goal reward's weight.",
        "",
        1,
        "",
    ),
    "goal_explained_variance": (
        "value",
        "Goal critic's prediction",
        "How well the long-range goal critic predicts. 1 is perfect.",
        "",
        1,
        "",
    ),
    "goal_loss": (
        "value",
        "Goal critic's loss",
        "How wrong the goal critic is. Lower is better.",
        "",
        1,
        "",
    ),
    "vdag_loss": (
        "value",
        "Twin critics' loss",
        "The optimistic twin critics' loss.",
        "",
        1,
        "vdag_loss: the twin critics' expectile loss, per twin.",
    ),
    "vdag_dose": (
        "value",
        "Twin critics' push",
        "How much the twin critics add to what the policy learns.",
        "",
        1,
        "Spread of the twin critics' addition to the advantages, as a share of theirs.",
    ),
    "vdag_headroom": (
        "value",
        "Twin critics' headroom",
        "How much more the twin critics think the bot could get than the value expects.",
        "",
        1,
        "",
    ),
    "vdag_clamped": (
        "value",
        "Twin critics' clamped share",
        "Share of samples whose push was cut.",
        "%",
        100,
        "",
    ),
    "head_loss_weight": (
        "value",
        "Extra critics' loss weight",
        "What the goal and twin critics' losses are multiplied by.",
        "",
        1,
        "1 / PopArt's variance (ppo.head_loss_scale = 'popart').",
    ),
    "steps_per_second": (
        "speed",
        "Training speed",
        "Steps a second while it trains. Higher is better.",
        "steps/s",
        1,
        "",
    ),
    "wall_steps_per_second": (
        "speed",
        "Speed, saves included",
        "Steps a second by the clock, with saves and bookkeeping counted.",
        "steps/s",
        1,
        "",
    ),
    "playing_steps_per_second": (
        "speed",
        "Playing speed",
        "Steps a second while the games play.",
        "steps/s",
        1,
        "",
    ),
    "playing_seconds": (
        "speed",
        "Time playing",
        "Seconds each round spends playing the games.",
        "s",
        1,
        "",
    ),
    "learning_seconds": (
        "speed",
        "Time learning",
        "Seconds each round spends learning from them.",
        "s",
        1,
        "",
    ),
    "env_wait_seconds": (
        "speed",
        "Waiting for the games",
        "Seconds each round waits for the game physics.",
        "s",
        1,
        "",
    ),
    "inference_seconds": (
        "speed",
        "Choosing actions",
        "Seconds each round the network spends choosing the cars' actions.",
        "s",
        1,
        "",
    ),
    "launch_seconds": (
        "speed",
        "Handing work to the card",
        "Seconds each round spends handing the network's work to the graphics card.",
        "s",
        1,
        "",
    ),
    "submit_seconds": (
        "speed",
        "Handing steps to the games",
        "Seconds each round spends handing the actions to the game workers.",
        "s",
        1,
        "",
    ),
    "other_seconds": (
        "speed",
        "Other playing time",
        "The rest of the playing time.",
        "s",
        1,
        "",
    ),
    "rating_mu": (
        "selfplay",
        "Skill rating",
        "Its rating from games against its past versions. Higher is better.",
        "",
        1,
        "",
    ),
    "rating_sigma": (
        "selfplay",
        "Rating uncertainty",
        "How unsure the rating is. It shrinks as games are played.",
        "",
        1,
        "",
    ),
    "pool_size": (
        "selfplay",
        "Past versions kept",
        "How many saved past versions it can play against.",
        "",
        1,
        "",
    ),
    "pool_goal_share": (
        "selfplay",
        "Goals against past versions",
        "Its share of the goals in those games. Over 50% means it beats them.",
        "%",
        100,
        "",
    ),
    "matchmaking": (
        "selfplay",
        "Match evenness",
        "How far the predicted win chance was from 50%. 0 is even.",
        "",
        1,
        "",
    ),
    "pool_goal_share_even": (
        "selfplay",
        "Goals against past versions, even starts",
        "The same, from starts that place every car alike.",
        "%",
        100,
        "",
    ),
    "pool_games_even": (
        "selfplay",
        "Games against past versions, even starts",
        "How many each round.",
        "",
        1,
        "",
    ),
    "start_edge": (
        "selfplay",
        "Blue's goals from practice starts",
        "Blue's share of the goals from the practice starts, the bot against itself.",
        "%",
        100,
        "",
    ),
    "replay_mix_steps": (
        "replays",
        "Copying steps",
        "Learning steps on replays each round.",
        "",
        1,
        "",
    ),
    "replay_mix_loss": (
        "replays",
        "Copying loss",
        "How far its choices are from the players'. Lower is closer.",
        "",
        1,
        "",
    ),
    "replay_mix_agreement": (
        "replays",
        "Chooses as the players did",
        "Share of the replays' decisions where its first choice is the player's.",
        "%",
        100,
        "",
    ),
    "replay_mix_wait_seconds": (
        "replays",
        "Waiting for replays",
        "Seconds each round waits for replays to be made ready.",
        "s",
        1,
        "",
    ),
    "replay_mix_rows_kept": (
        "replays",
        "Replay moments kept",
        "Moments of the replays used each round (kickoffs only).",
        "",
        1,
        "",
    ),
    "replay_mix_rows_left_out": (
        "replays",
        "Replay moments left out",
        "Moments left out each round (kickoffs only).",
        "",
        1,
        "",
    ),
}

# Per network (policy, value, shared trunk): (column prefix, title, what it means).
_NETWORK = (
    ("grad_norm_", "Gradient size", "How strongly each learning step pulls on the {net}'s weights."),
    ("update_norm_", "Update size", "How far each round moved the {net}'s weights."),
    ("weight_norm_", "Weight size", "The overall size of the {net}'s weights."),
    ("turn_", "Weight turn", "How far the {net}'s weights turned in a round, over their size."),
    ("growth_", "Weight growth", "The size of the {net}'s weights over their size at the start."),
)
_NETWORK_NAMES = {"policy": "policy", "value": "value network", "shared": "shared trunk"}
# Per way of starting games (several state setters): (column prefix, title, what it means, unit, factor).
_SETTER = (
    ("episodes_", "games", "Games it started each round.", "", 1.0),
    ("goal_share_", "goals", "Share of the goals the team of its first car scored.", "%", 100.0),
    ("episode_seconds_", "game length", "Seconds until a goal or a reset.", "s", 1.0),
    ("touch_share_", "touched", "Share of its games where the first car touched the ball.", "%", 100.0),
)
# Per game size of a bot that plays several (`<n>v<n>_<name>`).
_SIZES = {
    "car_step_share": ("share of play", "Its share of all the cars' steps.", "%", 100.0),
    "episode_seconds": ("game length", "Seconds until a goal or a reset.", "s", 1.0),
    "goals_per_minute": ("goals", "Goals a minute of its games.", "/min", 1.0),
    "touches_per_minute": ("ball touches", "Touches a minute per car.", "/min", 1.0),
    "aerial_touches_per_minute": ("aerial touches", "Touches in the air a minute per car.", "/min", 1.0),
}
_SIZE = re.compile(r"^(\d+)v\1_(.+)$")
# Per reward (`<prefix><label>`): (prefix, folder, title after the reward's name, what it means, unit).
_REWARD = (
    (
        "raw_",
        "rewards",
        "",
        "What it gives per step before its weight: going up means the bot does more of it (less, for a "
        "penalty).",
        "",
        "raw_{label}: the reward's mean value per agent step, before weight and zero-sum.",
    ),
    (
        "paid_abs_",
        "reward_details",
        "size paid",
        "Its size per step after its weight, a penalty counted as positive.",
        "",
        "paid_abs_{label}: mean |weighted value after zero-sum| per agent step.",
    ),
    (
        "paid_",
        "reward_details",
        "paid",
        "What it pays per step after its weight, opponents' share taken off (zero-sum).",
        "",
        "paid_{label}: mean weighted value after zero-sum per agent step.",
    ),
    (
        "own_abs_",
        "reward_details",
        "earned itself",
        "Its size per step after its weight, before the teammates' and opponents' share is mixed in.",
        "",
        "own_abs_{label}: mean |weighted value before zero-sum| per agent step.",
    ),
    (
        "fired_",
        "fires",
        "times a minute",
        "How many times a minute it pays one car.",
        "/min",
        "fired_{label}: share of agent steps on which it was not zero, times one car's decisions a minute.",
    ),
)


def reward_labels(header: list[str]) -> list[str]:
    """The rewards a metrics header records, by their label (`raw_<label>`), in its order."""
    return [column[4:] for column in header if column.startswith("raw_")]


def catalog(header: list[str], per_minute: float) -> list[Chart]:
    """A chart for every column in `header` (and a share chart per reward), in the folders' order. A
    reward's `fired_` count is shown a minute of one car's play: `per_minute` decisions."""
    labels = set(reward_labels(header)) | {c[9:] for c in header if c.startswith("paid_abs_")}
    found: list[Chart] = []
    for column in header:
        if column in _NOT_CHARTS:
            continue
        found.append(_chart(column, labels, per_minute))
        if column.startswith("paid_abs_"):
            label = column[9:]
            found.append(
                Chart(
                    f"share_{label}",
                    "shares",
                    _reward_title(label) + ": share",
                    "Its part of all the reward paid, a penalty counted by its size.",
                    "%",
                    100,
                    f"paid_abs_{label} over the sum of every reward's paid_abs_.",
                )
            )
    order = {folder: i for i, folder in enumerate(FOLDER_IDS)}
    return sorted(found, key=lambda chart: order[chart.folder])  # stable: each folder keeps the file's order


def _reward_title(label: str) -> str:
    return describe.label(label)


def _reward_about(label: str) -> str:
    """The first sentence of what the reward `label` (a second copy's label ends in its number) pays for."""
    own = custom_kinds()
    name = next((n for n in (label, re.sub(r"_\d+$", "", label)) if n in KINDS or n in own), "")
    text = inspect.cleandoc(KINDS[name].__doc__ or "") if name in KINDS else own[name].about if name else ""
    first = re.split(r"(?<=[.!?])\s", " ".join(text.split()), maxsplit=1)[0]
    return first.replace("`", "")


def _chart(column: str, labels: set[str], per_minute: float) -> Chart:
    fixed = _FIXED.get(column)
    if fixed is not None:
        folder, title, about, unit, scale, nerd = fixed
        return Chart(column, folder, title, about, unit, scale, nerd or column)
    for prefix, folder, words, about, unit, nerd in _REWARD:
        label = column[len(prefix) :]
        if column.startswith(prefix) and label in labels:
            title = _reward_title(label) + (f": {words}" if words else "")
            scale = per_minute if prefix == "fired_" else 1.0
            nerd = nerd.format(label=label)
            if prefix == "raw_" and (own := _reward_about(label)):  # what it pays for, in its own words
                about, nerd = own, f"{about} {nerd}"
            return Chart(column, folder, title, about, unit, scale, nerd)
    for prefix, words, about in _NETWORK:
        net = column[len(prefix) :]
        if column.startswith(prefix) and net:
            name = _NETWORK_NAMES.get(net, describe.label(net).lower())
            return Chart(column, "network", f"{words}: {name}", about.format(net=name), "", 1.0, column)
    size = _SIZE.match(column)
    if size is not None and size.group(2) in _SIZES:
        words, about, unit, scale = _SIZES[size.group(2)]
        n = size.group(1)
        return Chart(column, "sizes", f"{n}v{n} {words}", about, unit, scale, column)
    for prefix, words, about, unit, scale in _SETTER:
        setter = column[len(prefix) :]
        if column.startswith(prefix) and setter:
            return Chart(
                column, "starts", f"{describe.label(setter)} starts: {words}", about, unit, scale, column
            )
    return Chart(column, "other", describe.label(column), "Recorded in metrics.csv.", "", 1.0, column)


def folders() -> list[dict[str, Any]]:
    return [
        {"id": i, "title": title, "colour": colour, "about": about} for i, title, colour, about in FOLDERS
    ]


# ---------- the points ----------
_SERIES: dict[Any, tuple[tuple[Any, ...], dict[str, list[list[float]]]]] = {}


def view(
    folder: RunFolder, first: int, ids: list[str] | None, points: int = 240, m: bots.Metrics | None = None
) -> dict[str, Any]:
    """The bot's charts (folders, every chart, the default "Your charts") and the points of the charts `ids`
    (every chart when None) over the metrics rows from `first` on (of `m`, the bot's metrics when None), at
    most `points` (2 or more) each: the rows' means in equal stretches of agent steps, so a spike in a long
    run still moves its stretch, and the newest row as it is."""
    m = m or bots.metrics(folder)
    per_minute = decisions_a_minute(folder)
    charts = catalog(m.header, per_minute) if m.header else []
    by_id = {chart.id: chart for chart in charts}
    wanted = list(by_id) if ids is None else [i for i in ids if i in by_id]
    # The file as read (a resume writes it anew, maybe with as many rows), and what the points depend on.
    key = (m.identity, m.count, first, points, per_minute)
    cached = _SERIES.get(folder.metrics)
    done = cached[1] if cached is not None and cached[0] == key else {}
    missing = [i for i in wanted if i not in done]
    if missing:
        done = {**done, **_points(m, first, [by_id[i] for i in missing], points)}
        _SERIES[folder.metrics] = (key, done)
    return {
        "folders": folders(),
        "charts": [chart.view() for chart in charts],
        "default": [i for i in DEFAULT_PINNED if i in by_id],
        "series": {i: done[i] for i in wanted},
        "rows": m.count,
    }


def _points(m: bots.Metrics, first: int, charts: list[Chart], points: int) -> dict[str, list[list[float]]]:
    rows = m.rows_from(first)
    steps = [m.value("agent_steps", row) for row in rows]
    kept = [(row, x) for row, x in zip(rows, steps, strict=True) if x is not None]
    if not kept:
        return {chart.id: [] for chart in charts}
    # Equal stretches of steps, but the newest row alone in the last: its number is the chart's "now".
    x0, x1 = kept[0][1], kept[-1][1]
    width = (x1 - x0) / (points - 1) if x1 > x0 else 1.0
    buckets = [min(points - 2, int((x - x0) / width)) for _, x in kept[:-1]] + [points - 1]
    shares = (
        _share_sums(m, [row for row, _ in kept]) if any(c.id.startswith("share_") for c in charts) else []
    )
    out = {}
    for chart in charts:
        share = chart.id.startswith("share_")
        column = "paid_abs_" + chart.id[6:] if share else chart.id
        found: list[tuple[int, float, float]] = []  # (stretch, steps, number) of the rows that hold one
        for k, ((row, x), bucket) in enumerate(zip(kept, buckets, strict=True)):
            value = m.value(column, row)
            if value is None or (share and not shares[k]):
                continue
            found.append((bucket, x, value / shares[k] if share else value))
        if found:  # the newest row holding a number alone at the end, as it is (a column left blank at times)
            found[-1] = (points - 1, *found[-1][1:])
        sums: dict[int, list[float]] = {}
        for bucket, x, value in found:
            got = sums.setdefault(bucket, [0.0, 0.0, 0.0])
            got[0] += x
            got[1] += value
            got[2] += 1
        out[chart.id] = [
            [round(sx / n), _round(sy / n * chart.scale)] for _, (sx, sy, n) in sorted(sums.items())
        ]
    return out


def _share_sums(m: bots.Metrics, rows: list[int]) -> list[float]:
    """Each row's sum of every reward's `paid_abs_`: what its shares are of."""
    columns = [c for c in m.header if c.startswith("paid_abs_")]
    return [sum(m.value(c, row) or 0.0 for c in columns) for row in rows]


def _round(value: float) -> float:
    """Six significant figures: enough for any chart, a third of the text."""
    if value == 0 or not math.isfinite(value):
        return value
    return round(value, 5 - math.floor(math.log10(abs(value))))
