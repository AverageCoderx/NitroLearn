"""Weights & Biases tracking: with `[wandb] enabled = true`, every row a training run
adds to metrics.csv also goes to wandb.ai, charted against agent steps, with the run's settings attached, and
every round of a large copy of replays, charted against the replays learned.

Nothing here can stop training once it has started: a failure while sending is logged once and tracking
stops for the rest of the session; metrics.csv and report.html carry on as always. The W&B run's id is
kept in the run folder (`wandb_run.json`), so `nitro resume` continues the same W&B run; iterations a
resumed run plays again (those after the save it resumed from) are not sent twice.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import netrc
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from nitro.config.schema import WandbConfig

log = logging.getLogger("nitro")

STATE_FILE = "wandb_run.json"
# What a round of a copy of replays and a pass of a copy learned all at once send (pretrain.chart_rounds).
COPY_KEYS = (
    "round", "replays", "held_loss", "held_agreement", "train_loss", "decisions", "replays_per_minute",
    "seconds_learning",
)  # fmt: skip
CLONE_KEYS = ("train_loss", "held_loss", "held_agreement")
_SELFPLAY = (
    "rating_mu", "rating_sigma", "pool_size", "pool_goal_share", "matchmaking", "pool_goal_share_even",
    "pool_games_even", "start_edge",
)  # fmt: skip
# The same as nitro.training.PLAY_COLUMNS (a test keeps them equal; training imports this module).
_PLAY = (
    "in_air_share", "touches_per_minute", "speed_kmh", "speed_to_ball_kmh", "boost_held", "supersonic_share",
    "demolished_share", "ball_on_car_share", "aerial_touches_per_minute", "dribbles_per_minute",
    "flicks_per_minute", "double_taps_per_minute", "flip_resets_per_minute", "shots_per_minute",
    "saves_per_minute", "demos_per_minute", "flip_reset_height", "flip_reset_goal_dist",
    "reset_musty_goals_per_hour", "reset_musty_goal_speed_kmh",
)  # fmt: skip


class TrackingError(RuntimeError):
    """W&B tracking is on but cannot start; the message says what to do."""


class Tracker:
    """Tracking switched off: does nothing."""

    def log(self, row: dict[str, Any]) -> None:
        """Sends one metrics.csv row."""

    def finish(self) -> None:
        """Flushes and closes the connection; the run can be resumed later."""


def chart_name(column: str, setter_labels: list[str]) -> str:
    """The W&B chart name of a metrics.csv column, grouped into sections W&B shows as panels."""
    timing = column.endswith("_seconds") and column != "episode_seconds"
    if timing or column in ("steps_per_second", "playing_steps_per_second", "wall_steps_per_second"):
        return f"speed/{column}"
    if column == "reward_per_step" or column.startswith(("raw_", "paid_")):
        return f"rewards/{column}"
    if column in _SELFPLAY:
        return f"selfplay/{column}"
    if column in _PLAY:
        return f"play/{column}"
    if column.startswith("kickoff_"):
        return f"kickoffs/{column}"
    if any(column.endswith(f"_{label}") for label in setter_labels) and column != "episode_seconds":
        return f"start_states/{column}"
    return f"learning/{column}"


class WandbTracker(Tracker):
    def __init__(self, run: Any, state_file: Path, sent_until: int, setter_labels: list[str]) -> None:
        self.run = run
        self.state_file = state_file
        self.sent_until = sent_until
        self.setter_labels = setter_labels
        self.charts: dict[str, str] = {}  # each column's chart name, worked out once
        self.broken = False
        self.state_warned = False

    def log(self, row: dict[str, Any]) -> None:
        steps = int(row["agent_steps"])
        if self.broken or steps <= self.sent_until:
            return
        values: dict[str, float | None] = {"agent_steps": steps}
        charts = self.charts
        for column, value in row.items():
            if column == "agent_steps" or value in ("", None):
                continue
            number = float(value)
            chart = charts.get(column)
            if chart is None:
                chart = charts[column] = chart_name(column, self.setter_labels)
            values[chart] = number if math.isfinite(number) else None
        try:
            self.run.log({key: value for key, value in values.items() if value is not None})
        except Exception as err:  # never stop training for a chart
            self.broken = True
            log.warning("W&B stopped working (%s); training goes on, metrics.csv has every number.", err)
            return
        self.sent_until = steps
        try:
            _note(self.state_file, id=self.run.id, sent_until=steps)
        except OSError as err:  # a virus scanner holding the file, a full disk: the next row tries again
            if not self.state_warned:
                self.state_warned = True
                log.warning(
                    "Could not note W&B's progress in %s (%s); training goes on.", self.state_file, err
                )

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as err:
            log.warning("W&B did not close cleanly (%s); the numbers sent so far are kept.", err)


def _begin(settings: WandbConfig, root: Path, config: dict[str, Any]) -> tuple[Any, dict[str, Any], str]:
    """The W&B run of the run in `root` (the same one again on a resume) and the note kept about it. Online
    W&B that cannot start (no internet, not logged in) falls back to offline, W&B's files kept in the run
    folder for `wandb sync` (training must work offline); only both failing stops."""
    try:
        import wandb
    except ImportError:
        raise TrackingError(
            "W&B tracking is on ([wandb] enabled = true) but the wandb package is not installed. Install it "
            "with: .venv\\Scripts\\python -m pip install wandb   (or set enabled = false under [wandb])"
        ) from None
    state_file = root / STATE_FILE
    try:
        saved = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    except (OSError, ValueError) as err:
        log.warning("W&B's note %s could not be read (%s); a new W&B run starts.", state_file, err)
        saved = {}

    def init(mode: Literal["online", "offline"]) -> Any:
        return wandb.init(  # init_timeout: offline but logged in, W&B gives up after this many seconds
            project=settings.project,
            entity=settings.entity or None,
            name=settings.name or root.name,
            group=settings.group or None,
            tags=list(settings.tags) or None,
            mode=mode,
            id=saved.get("id"),
            resume="allow" if "id" in saved else None,
            dir=str(root),
            config=config,
            # The console stays ours (W&B would otherwise copy it), and no code is uploaded.
            settings=wandb.Settings(console="off", save_code=False, init_timeout=45),
        )

    mode = settings.mode
    failed: object = None
    run = None
    if mode == "online" and not logged_in():  # online, W&B would stop at its question in the console
        failed = "not logged in on this PC: `wandb login`"
    else:
        try:
            run = init(mode)
        except Exception as err:
            failed = err
    if run is None and mode == "online":
        # The failed online start still holds the run's id; without this the offline one is refused.
        with contextlib.suppress(Exception):
            wandb.teardown()
        try:
            run, mode = init("offline"), "offline"
        except Exception:
            run = None
        if run is not None:
            log.warning(
                "W&B could not go online (%s): the run goes on and W&B's files are kept in %s; once online "
                "(and logged in), `nitro wandb %s` uploads them.",
                failed,
                Path(run.dir).parent,
                root.name,
            )
    if run is None:
        raise TrackingError(
            f"W&B could not start ({failed}). Log in once with `wandb login` (your key is then kept by W&B "
            'on this PC, never in the settings), or set mode = "offline" or enabled = false under [wandb].'
        ) from (failed if isinstance(failed, BaseException) else None)
    # The copy of replays is charted against the replays learned, a copy learned all at once against
    # its passes; each by name, as W&B picks between matching patterns in no fixed order.
    run.define_metric("copy/replays_in_all")
    for key in COPY_KEYS:
        run.define_metric(f"copy/{key}", step_metric="copy/replays_in_all")
    run.define_metric("clone/pass")
    for key in CLONE_KEYS:
        run.define_metric(f"clone/{key}", step_metric="clone/pass")
    return run, saved, mode


def logged_in() -> bool:
    """Whether W&B has a key on this PC: WANDB_API_KEY, or the user's netrc file `wandb login` writes."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = urlparse(os.environ.get("WANDB_BASE_URL") or "https://api.wandb.ai").hostname or "api.wandb.ai"
    names = [os.environ["NETRC"]] if os.environ.get("NETRC") else []
    for name in [*names, str(Path.home() / ".netrc"), str(Path.home() / "_netrc")]:
        try:
            if netrc.netrc(name).authenticators(host):
                return True
        except (OSError, netrc.NetrcParseError):
            continue
    return False


def start(settings: WandbConfig, root: Path, config: dict[str, Any], setter_labels: list[str]) -> Tracker:
    """Starts tracking for the run in `root`, with `config` (every setting) attached; a no-op when off."""
    if not settings.enabled:
        return Tracker()
    run, saved, mode = _begin(settings, root, config)
    run.define_metric("agent_steps")
    run.define_metric("*", step_metric="agent_steps")
    state_file = root / STATE_FILE
    try:
        _note(state_file, id=run.id, sent_until=saved.get("sent_until", 0), **_url(run, mode))
    except OSError as err:
        log.warning(
            "Could not note W&B's run in %s (%s); a resume will start a new W&B run.", state_file, err
        )
    if mode == "offline":
        log.info("W&B offline: upload later with `wandb sync %s`", Path(run.dir).parent)
    else:
        log.info("W&B: %s", getattr(run, "url", None) or run.id)
    return WandbTracker(run, state_file, int(saved.get("sent_until", 0)), setter_labels)


class CopyTracker:
    """Charts of the copy of replays switched off: does nothing."""

    def rounds(self, entries: list[dict[str, Any]]) -> int:
        """Sends the copy's rounds not sent yet (see `pretrain.chart_rounds`); how many were sent."""
        return 0

    def restart(self) -> None:
        """The copy is made again from its first round: its rounds are sent again."""

    def finish(self) -> None:
        """Flushes and closes the connection."""


class WandbCopyTracker(CopyTracker):
    def __init__(self, run: Any, state_file: Path, saved: dict[str, Any]) -> None:
        self.run, self.state_file, self.saved = run, state_file, saved
        self.sent = int(saved.get("copy_rounds", -1))  # round 0 is the copy carried on, before learning
        self.broken = False

    def rounds(self, entries: list[dict[str, Any]]) -> int:
        count = 0
        for entry in entries:
            if self.broken or entry["round"] <= self.sent:
                continue
            values = {f"copy/{k}": float(v) for k, v in entry.items() if v is not None and k != "round"}
            values["copy/round"] = float(entry["round"])
            try:
                self.run.log(values)
            except Exception as err:  # never stop the copy for a chart
                self.broken = True
                log.warning("W&B stopped working (%s); the copy goes on, pretrain.json has every round.", err)
                return count
            self.sent, count = int(entry["round"]), count + 1
            with contextlib.suppress(OSError):  # the next round tries again
                _note(self.state_file, id=self.run.id, copy_rounds=self.sent)
        return count

    def restart(self) -> None:
        self.sent = -1
        with contextlib.suppress(OSError):
            _note(self.state_file, id=self.run.id, copy_rounds=-1)

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as err:
            log.warning("W&B did not close cleanly (%s); the rounds sent so far are kept.", err)


def chart_passes(
    settings: WandbConfig, root: Path, config: dict[str, Any], epochs: list[dict[str, Any]]
) -> None:
    """Sends the passes of a copy learned all at once (`nitro pretrain`, [pretrain] without batch_replays)
    once it is made and saved: losses and agreement against the pass, all of them (a copy made again
    starts its passes over). A failure is only said, as the copy is already saved."""
    if not settings.enabled or not epochs:
        return
    try:
        run, saved, _ = _begin(settings, root, config)
    except TrackingError as err:
        log.warning("The copy's charts were not sent: %s", err)
        return
    try:
        _send_passes(run, root / STATE_FILE, {**saved, "clone_passes": 0}, epochs)
        run.finish()
    except Exception as err:  # never fail a finished copy for a chart
        log.warning("W&B stopped working (%s); pretrain.json has every pass.", err)


def _send_passes(run: Any, state_file: Path, saved: dict[str, Any], epochs: list[dict[str, Any]]) -> int:
    """Sends the passes after the note's; how many were sent."""
    sent, count = int(saved.get("clone_passes", 0)), 0
    for epoch in epochs:
        if int(epoch["epoch"]) > sent:
            values = {f"clone/{k}": float(v) for k, v in epoch.items() if k in CLONE_KEYS and v is not None}
            run.log(values | {"clone/pass": float(epoch["epoch"])})
            sent, count = int(epoch["epoch"]), count + 1
    with contextlib.suppress(OSError):
        _note(state_file, id=run.id, clone_passes=sent)
    return count


def send_all(
    settings: WandbConfig,
    root: Path,
    config: dict[str, Any],
    setter_labels: list[str],
    rounds: list[dict[str, Any]],
    passes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[str, int, int, int]:
    """Everything of the run in `root` W&B does not have yet, in one W&B session (`nitro wandb`): the
    copy's rounds or passes, then the training's rows. Returns the mode it ran in ("offline" when it could
    not go online) and how many rounds, passes and rows were sent."""
    run, saved, mode = _begin(settings, root, config)
    run.define_metric("agent_steps")
    # The training's charts by name: no pattern that could also catch the copy's.
    for column in dict.fromkeys(column for row in rows[:1] for column in row):
        if column != "agent_steps":
            run.define_metric(chart_name(column, setter_labels), step_metric="agent_steps")
    state_file = root / STATE_FILE
    sent_rounds = WandbCopyTracker(run, state_file, saved).rounds(rounds)
    sent_passes = _send_passes(run, state_file, saved, passes)
    training = WandbTracker(run, state_file, int(saved.get("sent_until", 0)), setter_labels)
    sent_rows = 0
    for row in rows:
        if int(row["agent_steps"]) > training.sent_until and not training.broken:
            training.log(row)
            sent_rows += not training.broken
    run.finish()
    return mode, sent_rounds, sent_passes, sent_rows


def offline_runs(root: Path) -> list[Path]:
    """W&B's files of the run in `root` written offline and not uploaded yet (`wandb sync` marks those it
    uploaded)."""
    folder = root / "wandb"
    found = sorted(folder.glob("offline-run-*")) if folder.is_dir() else []
    return [path for path in found if not any(path.glob("*.synced"))]


def start_copy(settings: WandbConfig, root: Path, config: dict[str, Any]) -> CopyTracker:
    """Charts for the large copy of replays in the run folder `root`: every round's losses, agreement
    and speed against the replays learned in all (with those of a copy carried on), in the same W&B run as
    the run's training later; a no-op when [wandb] is off."""
    if not settings.enabled:
        return CopyTracker()
    run, saved, mode = _begin(settings, root, config)
    if url := _url(run, mode):
        with contextlib.suppress(OSError):  # only the dashboard's link to the charts would be missing
            _note(root / STATE_FILE, **url)
    return WandbCopyTracker(run, root / STATE_FILE, saved)


def _url(run: Any, mode: str) -> dict[str, str]:
    """The run's W&B page for its note (the dashboard links to it), when online."""
    url = getattr(run, "url", None)
    return {"url": url} if mode != "offline" and isinstance(url, str) and url.startswith("https://") else {}


def _note(path: Path, **changes: Any) -> None:
    """Updates the run's W&B note with `changes`, keeping the rest (the training's and the copy's notes)."""
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        state = {}
    _write_state(path, {**state, **changes})


def _write_state(path: Path, state: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(state), encoding="utf-8")
    os.replace(partial, path)
