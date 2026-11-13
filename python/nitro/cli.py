"""The `nitro` command.

PyTorch is imported only by the commands that train, so `nitro doctor` still runs when PyTorch is broken.
"""

from __future__ import annotations

import _thread
import argparse
import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from nitro import __version__
from nitro.config import ConfigError, NitroConfig, default_config, load_config
from nitro.config.loader import ConfigIssue
from nitro.doctor import Status, as_data, format_report, run_checks

if TYPE_CHECKING:
    from nitro.eval import Side
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun
    from nitro.watch import Live


# How often `nitro watch --follow` looks for a newer save.
_FOLLOW_SECONDS = 20.0


def _settings(file: Path | None) -> NitroConfig:
    return load_config(file) if file is not None else default_config(Path.cwd())


def _window_hint() -> str:
    """How to get a window when the system has no toolkit pywebview can use (Linux without Qt or GTK)."""
    if sys.platform.startswith("linux"):
        return (
            "On Linux the window needs Qt or GTK for Python. The simplest is Qt, inside NitroLearn's own "
            'Python: .venv/bin/python -m pip install "pywebview[qt]" - then start it again.'
        )
    return (
        "Run the installer again (Install NitroLearn.cmd) so its window parts are complete, then start it "
        "again."
    )


def _dashboard(args: argparse.Namespace) -> int:
    from nitro.dashboard import app

    project = Path(__file__).resolve().parents[2]  # the project folder (python/nitro/cli.py)
    if not (project / "configs" / "presets").is_dir():
        project = Path.cwd()
    try:
        from webview.errors import WebViewException
    except ImportError:
        return app.run(project, args.runs.resolve(), debug=args.debug)  # it says how to install pywebview
    # pywebview logs a traceback for each window toolkit it cannot load before it gives up; the message
    # below says the same in plain words.
    logging.getLogger("pywebview").addFilter(lambda record: "cannot be loaded" not in record.getMessage())
    try:
        return app.run(project, args.runs.resolve(), debug=args.debug)
    except WebViewException as err:
        print(f"The dashboard could not open its window: {err}", file=sys.stderr)
        print(_window_hint(), file=sys.stderr)
        return 1


def _rebuild(args: argparse.Namespace) -> int:
    from nitro import rebuild

    root = Path(__file__).resolve().parents[2]
    if args.files is not None:
        rebuild.write_list(root, args.files)
    return rebuild.rebuild(root)


def _update(args: argparse.Namespace) -> int:
    from nitro import update

    root = Path(__file__).resolve().parents[2]
    return update.command(
        root,
        args.runs.resolve(),
        check_only=args.check,
        yes=args.yes,
        installer=args.installer,
        result=args.result,
    )


def _doctor(args: argparse.Namespace) -> int:
    config = _settings(args.settings)
    results = run_checks(config.paths.collision_meshes, __version__)
    print(json.dumps(as_data(results, __version__)) if args.json else format_report(results, __version__))
    return 1 if any(result.status is Status.FAIL for result in results) else 0


def _fetch_meshes(args: argparse.Namespace) -> int:
    from nitro.fetch_meshes import MeshFetchError, fetch_meshes

    try:
        fetch_meshes(args.dest, args.archive)
    except MeshFetchError as err:
        print(f"Fetching the meshes failed: {err}", file=sys.stderr)
        return 1
    return 0


def _fetch_recordings(args: argparse.Namespace) -> int:
    from nitro.recordings import RecordingFetchError, fetch_recordings

    try:
        fetch_recordings(args.dest)
    except RecordingFetchError as err:
        print(f"Fetching RocketSim's recordings failed: {err}", file=sys.stderr)
        return 1
    return 0


def _install_rlbot(args: argparse.Namespace) -> int:
    from nitro import rlbot_install

    if args.status:
        return rlbot_install.status(args.runs)
    return rlbot_install.install(args.runs)


def _check_settings(args: argparse.Namespace) -> int:
    from nitro.config.loader import KeyLocator
    from nitro.doctor import meshes_result

    config = load_config(args.settings)
    # The arena meshes the settings point at, checked as `nitro doctor` checks them: a run cannot start
    # without them.
    meshes = meshes_result(config.paths.collision_meshes)
    if meshes.status is Status.FAIL:
        where = "paths.collision_meshes"
        line = KeyLocator(args.settings.read_text(encoding="utf-8-sig")).line_of(where)
        problem = ConfigIssue(where, meshes.detail, line, meshes.fix)
        print(ConfigError(args.settings, [problem]), file=sys.stderr)
    else:
        print(f"{args.settings}: no problems found.")
    print(_network_size(config))
    for note in config.advice():
        print(f"Note: {note}")
    return 1 if meshes.status is Status.FAIL else 0


def _network_size(config: NitroConfig) -> str:
    """The networks' parameters in one line."""
    from nitro.model import parameter_counts

    n = parameter_counts(config)
    parts = [f"shared trunk {n['shared']:,}"] if "shared" in n else []
    parts += [f"policy {n['policy']:,}", f"value {n['value']:,}"]
    heads = [f"{key[5:]} {value:,}" for key, value in n.items() if key.startswith("head_")]
    if heads:
        parts[-1] += f" (critic trunk {n['critic_trunk']:,}; heads: {', '.join(heads)})"
    runs = "the shared trunk and the policy" if "shared" in n else "the policy"
    return f"Networks: {n['total']:,} parameters: {'; '.join(parts)}. The bot runs {n['playing']:,} ({runs})."


class _Unbroken:
    """A text stream whose writes stop quietly once the other end has gone (a closed pipe)."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._gone = False

    def write(self, text: str) -> int:
        if not self._gone:
            try:
                return int(self._stream.write(text))
            except OSError:  # BrokenPipeError among them
                self._gone = True
        return len(text)

    def flush(self) -> None:
        if not self._gone:
            try:
                self._stream.flush()
            except OSError:
                self._gone = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _train(args: argparse.Namespace) -> int:
    from nitro.runs import RunFolder

    config = load_config(args.settings)
    given = {"run": args.start_from, "at": args.start_at} if args.start_from is not None else None
    if given and config.pretrain.enabled:
        print(
            "--from and [pretrain] enabled = true both choose what the run starts from: use one.",
            file=sys.stderr,
        )
        return 1
    if given and not config.start_from.enabled:
        print(
            "--from asks for transfer learning, but the settings file switches it off ([start_from] "
            "enabled = false): set enabled = true there, or leave out --from.",
            file=sys.stderr,
        )
        return 1
    activation = None
    if given:  # transfer learning, as [start_from] run = ... would
        config = dataclasses.replace(config, start_from=dataclasses.replace(config.start_from, **given))
        from nitro.transfer import TransferError

        # The source is checked before the run's folder is made, so a refused start leaves nothing behind.
        try:
            activation = _check_source(config, args.runs)
        except (TransferError, OSError) as err:
            print(f"Cannot start: {err}", file=sys.stderr)
            return 1
    try:
        folder = RunFolder.create(args.runs, args.name, args.settings, given)
    except FileExistsError as err:
        print(err, file=sys.stderr)
        return 1
    if activation is not None:
        from nitro.dashboard import edits

        config = _with_activation(config, activation)
        text = folder.settings.read_text(encoding="utf-8")
        edits.save(folder.settings, edits.written(text, {"model.activation": activation}))
        print(f'model.activation set to "{activation}", the networks\' own: the run starts from them.')
    print(
        f"The run keeps its own copy of the settings: {folder.settings}. To change a setting later, stop "
        "the run, edit that copy, and resume."
    )
    return _run(config, folder, None, args.steps, args.then_train)


def _check_source(config: NitroConfig, runs: Path) -> str | None:
    """Checks that the save a transfer starts from (`config.start_from`, looked for in `runs`) can start
    this run, raising TransferError when not. Returns the activation of its networks when the settings name
    another one: they are that bot's networks, so they keep it; None when they agree."""
    from nitro import transfer
    from nitro.runs import actions_problem

    source = transfer.find_source(config.start_from, runs)
    if source is None:
        return None
    if problem := actions_problem(source.state["learner"]["model"], f"start_from: {source.label}"):
        raise transfer.TransferError(problem)
    theirs = str(source.state["settings"]["model.activation"])
    activation = theirs if theirs != config.model.activation else None
    transfer.check(source, _with_activation(config, activation) if activation else config)
    return activation


def _with_activation(config: NitroConfig, activation: str) -> NitroConfig:
    model = dataclasses.replace(config.model, activation=activation)  # type: ignore[arg-type]
    return dataclasses.replace(config, model=model)


def _new(args: argparse.Namespace) -> int:
    from nitro.runs import RunFolder

    load_config(args.settings)  # a settings file with mistakes makes no bot
    try:
        folder = RunFolder.create(args.runs, args.name, args.settings)
    except FileExistsError as err:
        print(err, file=sys.stderr)
        return 1
    from nitro.runs import resume_command

    name = folder.root.name
    print(
        f"Made the bot '{name}' (not started; settings: {folder.settings}). Start it here with "
        f"`{resume_command(folder)}`, or carry it to another computer with `nitro pack {name}`."
    )
    return 0


def _convert_replays(args: argparse.Namespace) -> int:
    from nitro import replays

    sources = args.folders or replays.game_folders(replays.documents())
    missing = [str(f) for f in sources if not f.is_dir()]
    if missing:
        print(f"Not a folder: {', '.join(missing)}", file=sys.stderr)
        return 1
    replays.lower_priority()
    try:
        done = replays.convert_to_store(sources, args.meshes, args.limit, args.workers)
    except replays.ReplayError as err:
        print(f"Cannot convert: {err}", file=sys.stderr)
        return 1
    summary = replays.store_summary()
    kinds = ", ".join(f"{k} {v:,}" for k, v in summary["counts"].items()) or "none"
    print(
        f"Done: {done.done:,} converted, {done.kept:,} kept, {sum(done.failed.values()):,} could not be. "
        f"The store holds {kinds} ({summary['bytes'] / 2**30:.1f} GB)."
    )
    return 0


def _pretrain(args: argparse.Namespace) -> int:
    from nitro import pretrain, tracking
    from nitro.recordings import RecordingFetchError, needed_recordings
    from nitro.runs import RunFolder, all_settings, portable
    from nitro.training import resolve_device

    config = load_config(args.settings)
    notes: list[str] = []
    try:
        files = pretrain.recordings(args.recordings) if args.recordings else []
        if args.rocketsim:
            try:
                files += needed_recordings()
            except RecordingFetchError as err:
                raise pretrain.PretrainError(
                    f"RocketSim's recordings are not on this computer: {err}"
                ) from None
        if not files:
            raise pretrain.PretrainError("name some recordings, or add --rocketsim for RocketSim's own")
        print(f"Reading {len(files)} recording(s)...")
        samples = pretrain.load(files, config, args.softness, notes, args.rows)
        train_set, held = pretrain.split(samples, args.held_out, config.run.seed)
    except pretrain.PretrainError as err:
        print(f"Cannot pretrain: {err}", file=sys.stderr)
        return 1
    for note in notes:
        print(note)
    replicar, recorded = samples.read_as["replicar"], samples.read_as["recorded"]
    if replicar or recorded:  # what each recording's rows were taken to hold
        how = "by folder" if args.rows == "auto" else "as --rows says"
        print(
            f"Read {replicar} recording(s) as made from replays with replicar and {recorded} as recorded by "
            f"the bot or RocketSim ({how})."
        )
    print(
        f"{len(samples):,} samples ({samples.skipped:,} decision steps left out at gaps); learning from "
        f"{len(train_set):,}, measuring on {len(held):,} from recordings it does not learn from."
    )
    try:
        folder = RunFolder.create(args.runs, args.name, args.settings)
    except FileExistsError as err:
        print(err, file=sys.stderr)
        return 1

    def report(epoch: pretrain.Epoch) -> None:
        print(
            f"epoch {epoch.epoch}: loss {epoch.train_loss:.3f} | held back: loss {epoch.held_loss:.3f}, "
            f"first choice agrees {epoch.held_agreement:.1%}",
            flush=True,
        )

    model, history = pretrain.train(
        config,
        train_set,
        held,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=resolve_device(config.run),
        seed=config.run.seed,
        report=report,
    )
    details = {
        "recordings": [portable(f) for f in samples.files],
        "left_out": notes,
        "rows": args.rows,
        "read_as": samples.read_as,  # how many recordings were read each way
        "samples": len(samples),
        "held_out_samples": len(held),
        "softness": args.softness,
        "prev_action_dropout": config.pretrain.prev_action_dropout,  # from the settings file
        "epochs": [dataclasses.asdict(e) for e in history],
    }
    pretrain.save(folder, model, config, details)
    tracking.chart_passes(config.wandb, folder.root, all_settings(config, folder.root), details["epochs"])
    print(
        f"Saved the cloned bot as '{folder.root.name}'. Watch it with: nitro watch {folder.root.name}\n"
        f"Train on from it with: nitro train <settings> --from {folder.root.name} (it takes the clone's "
        "policy only, since cloning trains no value network: the new run's starts fresh). "
        "For your own replays, [pretrain] in the settings does all of this when a run starts."
    )
    return 0


def _resume(args: argparse.Namespace) -> int:
    from nitro.runs import RunFolder

    applied = _wait_for_previous_trainer()
    folder = RunFolder(args.runs / args.name)
    if not folder.info.is_file():
        print(f"There is no run called '{args.name}' in {args.runs}.", file=sys.stderr)
        return 1
    so_far = folder.copy_so_far()
    state = None if so_far is not None else folder.load_latest()  # a copy so far: the copy carries on
    if state is not None and "pretrained" in state:
        print(
            f"'{args.name}' holds a bot cloned from recordings (nitro pretrain), not a training run. Start a "
            f"run from it with: nitro train <settings> --from {args.name}",
            file=sys.stderr,
        )
        return 1
    if state is not None and args.steps is not None and int(state["agent_steps"]) >= args.steps:
        print(
            f"{args.name} already has {int(state['agent_steps']):,} agent steps, past the {args.steps:,} "
            "asked for: nothing was done."
        )
        return 0
    if state is None and so_far is None and (note := folder.refresh_settings()):
        print(note)  # a run that stopped before its first save takes its fixed settings file
    config = folder.load_settings(state)
    if state is None and config.pretrain.enabled:
        then = "then trains" if args.then_train else "stops once the copy is saved (start it again to train)"
        made = "goes on with" if (folder.root / "pretrained").is_dir() else "makes"
        has = (
            f"a save of its copy so far only ({so_far.get('learned', 0):,} replays)"
            if so_far is not None
            else "no save yet"
        )
        print(f"'{args.name}' has {has}: it {made} its copy of the replays ([pretrain]), {then}.")
    elif state is None:
        print(f"'{args.name}' has no save yet, so it starts again from the beginning with {folder.settings}.")
    return _run(config, folder, state, args.steps, args.then_train, applied)


def _log_to(file: Path) -> None:
    """Everything goes to the run's train.log; warnings and errors also to the console."""
    logger = logging.getLogger("nitro")
    logger.setLevel(logging.INFO)
    # A line that cannot be written (a full disk) is dropped, not answered with logging's own traceback:
    # the run stops with a plain message when its saves cannot be written.
    logging.raiseExceptions = False
    to_file = logging.FileHandler(file, encoding="utf-8")
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    to_console = logging.StreamHandler()
    to_console.setLevel(logging.WARNING)
    to_console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(to_file)
    logger.addHandler(to_console)


def _run(
    config: NitroConfig,
    folder: RunFolder,
    state: dict[str, Any] | None,
    steps: int | None,
    then_train: bool = False,
    applied: bool = False,
) -> int:
    """Trains `folder`, noting in running.json that it runs and taking a stop.request file (the dashboard's
    Stop and Apply) as Ctrl+C once: save, then stop. A run that copies replays first ([pretrain]) stops
    once the copy is saved, unless `then_train`. `applied`: started again after the dashboard's Apply."""
    from nitro.dashboard.bots import APPLY_RESTART, RUNNING, STOP_REQUEST, claim
    from nitro.dashboard.jobs import AFTER_PID, JOB_TOKEN, spawn

    request = folder.root / STOP_REQUEST
    marker = folder.root / RUNNING
    # One trainer a bot: two would delete each other's saves (a double click on Start).
    info = {
        "pid": os.getpid(),
        "started": time.time(),
        "command": sys.argv[1:2],
        "job": os.environ.get(JOB_TOKEN, ""),
    }
    if not claim(folder, info):
        print(
            f"'{folder.root.name}' is already training in another process: one trainer a bot. Stop that one "
            "first (the dashboard's Stop, or Ctrl+C in its window).",
            file=sys.stderr,
        )
        return 1
    request.unlink(missing_ok=True)
    finished = threading.Event()
    then = {"after": "stop"}

    def watch_requests() -> None:
        asked = False
        while not finished.wait(1.0):
            if not request.exists():
                continue
            try:
                what = request.read_text(encoding="utf-8").strip()
            except OSError:
                what = "stop"
            request.unlink(missing_ok=True)
            if not asked:  # once only: a second Ctrl+C would quit without saving
                asked = True
                then["after"] = "apply" if what == "apply" else "stop"
                print(
                    "Saving, then starting again with the changes made in the dashboard."
                    if what == "apply"
                    else "Stop asked for from the dashboard.",
                    flush=True,
                )
                _thread.interrupt_main(signal.SIGINT)  # as Ctrl+C: the handler below saves first
            elif what == "stop" and then["after"] == "apply":
                then["after"] = "stop"
                print("Stop asked for: it saves and stops, without starting again.", flush=True)

    threading.Thread(target=watch_requests, name="nitro-stop-request", daemon=True).start()
    try:
        code = _run_until_stopped(config, folder, state, steps, then_train, applied)
    finally:
        finished.set()
        marker.unlink(missing_ok=True)
        request.unlink(missing_ok=True)
    if then["after"] == "apply":
        # Here, not in the dashboard: closing its window after Apply must not leave the bot stopped. The new
        # trainer waits for this process to end first, so the graphics card's memory is free.
        outcome: dict[str, Any] = {"at": time.time(), "restarted": code == 0}
        if code == 0:
            spawn(
                ["resume", folder.root.name, "--runs", str(folder.root.parent)],
                folder.root,
                Path.cwd(),
                env={AFTER_PID: str(os.getpid())},
            )
            print("Starting again with the changes.", flush=True)
        else:
            outcome["why"] = (
                "it stopped with a problem before it could start again (its Console tab says what); "
                "fix that and press Start"
            )
            print(f"Not starting again: {outcome['why']}.", flush=True)
        (folder.root / APPLY_RESTART).write_text(json.dumps(outcome), encoding="utf-8")
    return code


def _wait_for_previous_trainer(limit: float = 300.0) -> bool:
    """A trainer started by the one before it (after an Apply) waits until that process has ended, so its
    graphics card memory is free: starting while the old trainer still held it could run out of memory and
    leave the bot stopped. True for such a trainer."""
    from nitro.dashboard.bots import pid_alive
    from nitro.dashboard.jobs import AFTER_PID

    pid = int(os.environ.pop(AFTER_PID, "0") or 0)
    began = time.monotonic()
    while pid and pid_alive(pid) and time.monotonic() - began < limit:
        time.sleep(0.5)
    return pid > 0


def _from_dashboard() -> bool:
    """Started by the dashboard (its Start, or a trainer starting itself again after Apply): the output goes
    to its Console tab, where Ctrl+C means nothing; its Stop button stops the run."""
    from nitro.dashboard.jobs import JOB_TOKEN

    return JOB_TOKEN in os.environ


def _run_until_stopped(
    config: NitroConfig,
    folder: RunFolder,
    state: dict[str, Any] | None,
    steps: int | None,
    then_train: bool,
    applied: bool = False,
) -> int:
    from nitro import _core, pretrain
    from nitro.health import TrainingProblem
    from nitro.runs import resume_command
    from nitro.tracking import TrackingError
    from nitro.training import TrainingRun
    from nitro.transfer import TransferError

    _log_to(folder.log)
    copy_only = state is None and config.pretrain.enabled and not then_train
    if state is None and config.pretrain.enabled:
        # Here, so the console shows the progress; the run then finds the copy ready.
        log = logging.getLogger("nitro")

        def say(line: str) -> None:
            print(line, flush=True)
            log.info(line)  # train.log too

        try:
            pretrain.start_source(
                config, folder, say, save_so_far=lambda learned: _save_so_far(folder, learned, say)
            )
        except pretrain.PretrainError as err:
            print(f"\nCannot start: {err}.", file=sys.stderr)
            print(_before_first_save(folder), file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print(f"\nStopped. To start again: {resume_command(folder)}")
            return 1
    stop = threading.Event()
    # Output piped to a program Ctrl+C also stops (`| tee train.log`) must not stop training mid-iteration: a
    # failed print would, and with several cards the others then wait for it. train.log keeps it all.
    sys.stdout = _Unbroken(sys.stdout)

    dashboard = _from_dashboard()

    def on_ctrl_c(signum: int, frame: FrameType | None) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        again = "" if dashboard else " Press Ctrl+C again to quit at once without saving."
        print(f"Stopping after this iteration and saving.{again}")

    signal.signal(signal.SIGINT, on_ctrl_c)
    if not copy_only:
        print(
            f"Training '{folder.root.name}'."
            + ("" if dashboard else f" Progress: {folder.report} (open it in a browser). Ctrl+C stops.")
        )
    try:
        training = TrainingRun(config, folder, state, applied=applied)
        if copy_only:
            return _save_copy(training, folder)
        print(f"Networks run on {training.device_note}.")
        training.run(stop, until_steps=steps)
    except TrackingError as err:
        print(f"\nCannot start: {err}", file=sys.stderr)
        return 1
    except (TransferError, pretrain.PretrainError) as err:
        print(f"\nCannot start: {err}", file=sys.stderr)
        if not folder.complete_checkpoints():
            print(_before_first_save(folder), file=sys.stderr)
        return 1
    except (TrainingProblem, _core.EnvError) as err:  # EnvError: the games stopped
        logging.getLogger("nitro").info("Training stopped: %s", err)  # train.log too
        print(f"\nTraining stopped: {err}", file=sys.stderr)
        if folder.complete_checkpoints():
            print(f"To continue from the last save: {resume_command(folder)}", file=sys.stderr)
        else:  # e.g. at the first start: nothing saved yet
            print(_before_first_save(folder), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\nStopped without saving. To continue from the last save: {resume_command(folder)}")
        return 1
    except OSError as err:  # a save, the metrics or the report could not be written: a full disk, mostly
        print(f"\n{_write_failed(err, folder)}", file=sys.stderr)
        return 1
    print(f"Saved. To continue: {resume_command(folder)}")
    return 0


def _write_failed(err: OSError, folder: RunFolder) -> str:
    """What to say when training stops because a file could not be written, with where the last good save
    is (a save cut short is removed, so the newest finished one is good)."""
    import errno

    from nitro.runs import resume_command

    log = logging.getLogger("nitro")
    log.info("Training stopped: %s", err, exc_info=True)  # train.log, when there is room for it
    saves = folder.complete_checkpoints()
    if err.errno == errno.ENOSPC:
        what = f"Training stopped: the disk is full (the disk that holds {folder.root})."
        fix = "free some space on that disk"
    else:
        where = f" ({err.filename})" if err.filename else ""
        what = f"Training stopped: a file could not be written{where}: {err.strerror or err}."
        fix = "fix that"
    last = (
        f"The last good save is at {int(saves[-1].name):,} agent steps ({saves[-1]})"
        if saves
        else "Nothing was saved yet"
    )
    return f"{what}\n{last}; {fix}, then start again with: {resume_command(folder)}"


def _before_first_save(folder: RunFolder) -> str:
    """What to do when a run stops before its first save: fix the settings, then start it again."""
    from nitro.runs import resume_command

    original = folder.started_with()
    where = (
        f"in the settings file it was started from ({original}) or in the run's own copy ({folder.settings})"
        if original is not None
        else f"in the run's settings ({folder.settings})"
    )
    again = (
        " It has no save yet, so it reads the file it was started from again if that changed."
        if original is not None
        else ""
    )
    return f"No save yet. Fix it {where}, then start it again with: {resume_command(folder)}.{again}"


def _save_copy(training: TrainingRun, folder: RunFolder) -> int:
    """The copy of the replays kept as the run's first save, at 0 agent steps, without training: it can
    be watched, exported and have rewards tried on it (nitro reward-check) before the next start trains it."""
    from nitro.runs import resume_command

    try:
        training.checkpoint(continue_after=False)
        training.write_report()
    finally:
        training.metrics.close()
    name = folder.root.name
    print(
        f"The copy of the replays is ready and saved as '{name}''s first save (0 agent steps). Training has "
        f"not started. Try rewards on it first with: nitro reward-check {name} (the dashboard: Rewards, Try "
        f"these rewards). Start training with: {resume_command(folder)} (the dashboard: Start training)."
    )
    return 0


def _save_so_far(folder: RunFolder, learned: int, say: Callable[[str], None]) -> None:
    """The copy of replays so far saved as the bot's save, by `nitro copy-save` in a process of its own, so
    the copy's memory and random numbers stay as they were; the copy goes on either way."""
    began = time.perf_counter()
    # No console window of its own on Windows (a copy started by the dashboard has none to share).
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined, unused-ignore]
    try:
        done = subprocess.run(
            [sys.executable, "-m", "nitro", "copy-save", folder.root.name, "--runs", str(folder.root.parent)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        say(f"  the copy so far was not saved ({type(err).__name__}); the copy goes on")
        return
    if done.returncode != 0:
        lines = [line for line in (done.stderr or done.stdout).splitlines() if line.strip()]
        why = lines[-1][:300] if lines else "no reason given"
        say(f"  the copy so far was not saved ({why}); the copy goes on")
        return
    took = time.perf_counter() - began
    say(
        f"  saved the copy so far ({learned:,} replays) as the bot's save in {took:.0f} s: it can be watched,"
        " made into an RLBot bot and have rewards tried on it now"
    )


def _copy_save(args: argparse.Namespace) -> int:
    """Saves a bot's unfinished copy of replays as its save; the copy runs this itself."""
    from nitro import pretrain
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    folder = RunFolder(args.runs / args.name)
    if not folder.info.is_file():
        print(f"There is no run called '{args.name}' in {args.runs}.", file=sys.stderr)
        return 1
    if folder.complete_checkpoints() and folder.copy_so_far() is None:
        print(f"'{args.name}' has a save of its own already: nothing to do.", file=sys.stderr)
        return 1
    try:
        config = folder.load_settings()
        source, learned = pretrain.copy_so_far(config, folder)
        training = TrainingRun(config, folder, None, start=source)
    except (ConfigError, pretrain.PretrainError) as err:
        print(f"Cannot save the copy so far: {err}", file=sys.stderr)
        return 1
    try:
        training.written = config
        training.copy_so_far = learned
        training.checkpoint(continue_after=False)
        training.write_report()
    finally:
        training.metrics.close()
    print(
        f"Saved the copy so far ({learned['learned']:,} of {learned['limit']:,} replays) as the bot's save."
    )
    return 0


def _reward_check(args: argparse.Namespace) -> int:
    from nitro import reward_check
    from nitro.runs import RunFolder

    folder = RunFolder(args.runs / args.name)
    if not folder.info.is_file():
        print(f"There is no run called '{args.name}' in {args.runs}.", file=sys.stderr)
        return 1
    try:
        text = reward_check.trial_settings(folder, args.rewards)
        result = reward_check.check(folder, text, args.steps, out=args.out)
    except (reward_check.CheckError, OSError) as err:
        print(f"Cannot check the rewards: {err}.", file=sys.stderr)
        return 1
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped: nothing kept.")
        return 1
    print("\n".join(reward_check.table(result)))
    return 0


def _wandb(args: argparse.Namespace) -> int:
    """Sends what a run has not sent to W&B yet: its copy of replays and its training."""
    import csv
    import dataclasses

    from nitro import pretrain, tracking
    from nitro.runs import RunFolder, all_settings
    from nitro.training import setter_labels

    folder = RunFolder(args.runs / args.name)
    if not folder.root.is_dir():
        print(f"No run named '{args.name}' in {args.runs}.", file=sys.stderr)
        return 1
    if not tracking.logged_in():
        print("W&B has no login on this PC yet: run `.venv\\Scripts\\wandb login` once, then this again.")
        return 1
    config = folder.load_settings()
    if not config.wandb.enabled:
        print("W&B is off in this run's settings: sending anyway (turned on, it sends while training).")
    settings = dataclasses.replace(config.wandb, enabled=True, mode="online")  # asked to send: online
    everything = all_settings(config, folder.root)
    if waiting := tracking.offline_runs(folder.root):  # sessions W&B kept offline: uploaded first, in order
        print(f"Uploading {len(waiting)} part(s) W&B kept on this PC while offline ...", flush=True)
        try:
            synced = subprocess.run(
                [sys.executable, "-m", "wandb", "sync", *map(str, waiting)], check=False, timeout=600
            ).returncode
        except subprocess.TimeoutExpired:  # no connection: W&B waits for ever
            synced = 1
        if synced != 0:
            print("The upload did not work (no internet?); they stay on this PC for the next try.")
    rounds, passes = pretrain.copy_for_charts(folder.root)
    rows: list[dict[str, str]] = []
    if folder.metrics.is_file():
        with folder.metrics.open(encoding="utf-8") as metrics:
            rows = list(csv.DictReader(metrics))
    try:
        labels = setter_labels(config)
        mode, sent_rounds, sent_passes, sent_rows = tracking.send_all(
            settings, folder.root, everything, labels, rounds, passes, rows
        )
    except tracking.TrackingError as err:
        print(f"Cannot send: {err}", file=sys.stderr)
        return 1
    print(
        f"'{args.name}' to W&B (project '{settings.project}'): {sent_rounds} rounds of its copy of replays, "
        f"{sent_passes} passes and {sent_rows:,} training iterations sent (the rest were there already)"
        + (". Kept on this PC (offline): run this again when online." if mode == "offline" else ".")
    )
    return 0


def _watch(args: argparse.Namespace) -> int:
    from nitro.runs import RunFolder

    folder = RunFolder(args.runs / args.name)
    if not folder.info.is_file():
        print(f"There is no run called '{args.name}' in {args.runs}.", file=sys.stderr)
        return 1
    state = folder.load_checkpoint(args.at) if args.at is not None else folder.load_latest()
    if state is None and args.at is not None:
        saves = ", ".join(f"{int(p.name):,}" for p in folder.complete_checkpoints()) or "none"
        print(f"'{args.name}' has no save at {args.at:,} agent steps. Saves: {saves}.", file=sys.stderr)
        return 1
    if state is None:
        print(
            f"'{args.name}' has no save yet; the first comes after run.checkpoint_minutes (or "
            "run.checkpoint_steps).",
            file=sys.stderr,
        )
        return 1
    from nitro.runs import actions_problem

    if problem := actions_problem(state["learner"]["model"], f"'{args.name}'"):
        print(f"{problem}.", file=sys.stderr)
        return 1
    follow = args.follow and args.at is None
    from nitro.watch import WatchError

    try:
        if args.feed:
            return _watch_feed(folder, state, args, follow)
        if args.live:
            return _watch_live(folder, state, args, follow)
        out = _write_watch(folder, state, args, follow)
    except WatchError as err:
        print(f"'{args.name}' cannot be watched: {err}", file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    if not follow:
        return 0
    print("Watching for new saves: the page shows the newest one from its next game on. Ctrl+C stops.")
    shown = int(state["agent_steps"])
    try:
        while True:
            time.sleep(_FOLLOW_SECONDS)
            complete = folder.complete_checkpoints()
            if complete and int(complete[-1].name) > shown:
                newest = folder.load_latest()
                if newest is not None:
                    _write_watch(folder, newest, args, follow)
                    shown = int(newest["agent_steps"])
                    print(f"Updated to the save at {shown:,} agent steps.")
    except KeyboardInterrupt:
        return 0


def _live_game(
    folder: RunFolder, state: dict[str, Any], config: NitroConfig, args: argparse.Namespace, follow: bool
) -> tuple[Live, Callable[[], tuple[NitroConfig, dict[str, Any], str] | None] | None]:
    """The live game of a run's save, played with `config`, and with `follow` where it finds a newer save
    to carry on with."""
    from nitro.watch import Live

    shown = int(state["agent_steps"])

    def title(steps: int) -> str:
        return f"{folder.root.name} at {steps:,} agent steps"

    def newer() -> tuple[NitroConfig, dict[str, Any], str] | None:
        nonlocal shown
        complete = folder.complete_checkpoints()
        if not complete or int(complete[-1].name) <= shown:
            return None
        newest = folder.load_latest()
        if newest is None:
            return None
        shown = int(newest["agent_steps"])
        return _watch_settings(folder, newest, args), newest, title(shown)

    size = config.env.team_size
    subtitle = (
        f"{size}v{size}, played live on this computer (seed {args.seed}); the saved bot drives every car, "
        "choosing as it does in training."
        + (" Each new save takes over from its next game." if follow else "")
    )
    return Live(config, state, title(shown), subtitle, args.seed, args.fps), newer if follow else None


def _watch_live(folder: RunFolder, state: dict[str, Any], args: argparse.Namespace, follow: bool) -> int:
    """The live game in its own window, played on a thread of this process until the window is closed."""
    from nitro import viewer
    from nitro.viewer import feed, window

    if not window.available():
        print(
            "The live view opens its own window and needs pywebview for it. Install it with:  "
            ".venv\\Scripts\\python -m pip install pywebview",
            file=sys.stderr,
        )
        return 1
    config = _watch_settings(folder, state, args)
    live, newer = _live_game(folder, state, config, args, follow)
    chunks = feed.Chunks()
    chunks.add(live.header)
    header = chunks.header or {}
    # The window takes a page of this size only from a file. The page stays on this computer and goes when
    # the window closes, so the arena's meshes are inside it.
    page = folder.root / viewer.LIVE_PAGE
    _replace_text(page, viewer.page(header, arena=viewer.arena_blob(config.paths.collision_meshes)))
    stop = threading.Event()

    def play() -> None:
        try:
            live.play(chunks.add, stop, newer, _FOLLOW_SECONDS)
        finally:
            chunks.close()

    player = threading.Thread(target=play, name="nitro-live-game", daemon=True)
    player.start()
    print("Playing live in its own window (nothing is served on any port). Close the window to stop.")
    try:
        window.show(page, str(header.get("title", folder.root.name)), chunks, stop.set)
    except KeyboardInterrupt:
        pass
    except window.NoWindow as err:
        print(f"The live view could not open its window: {err}", file=sys.stderr)
        print(_window_hint(), file=sys.stderr)
        return 1
    finally:
        stop.set()
        player.join(timeout=30)
        _remove(page)
    print("Stopped the live view.")
    return 0


def _watch_feed(folder: RunFolder, state: dict[str, Any], args: argparse.Namespace, follow: bool) -> int:
    """The live game as the feed's lines on standard output, until the input closes or the command is
    ended. Standard output carries the lines and nothing else."""
    lines = sys.stdout
    sys.stdout = sys.stderr  # whatever else prints goes beside the feed, not into it
    stop = threading.Event()

    def write(line: str) -> None:
        try:
            lines.write(line + "\n")
            lines.flush()
        except (OSError, ValueError):
            stop.set()  # the reader is gone

    def until_input_closes() -> None:
        try:
            # Read below Python's own buffer, which a waiting thread would hold locked at exit.
            descriptor = sys.stdin.fileno()
            while os.read(descriptor, 4096):
                pass
        except (AttributeError, OSError, ValueError):
            pass
        stop.set()

    try:
        live, newer = _live_game(folder, state, _watch_settings(folder, state, args), args, follow)
        threading.Thread(target=until_input_closes, name="nitro-feed-input", daemon=True).start()
        write(live.header)
        live.play(write, stop, newer, _FOLLOW_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout = lines
    return 0


def _watch_settings(folder: RunFolder, state: dict[str, Any], args: argparse.Namespace) -> NitroConfig:
    """The run's settings for a game to watch, at `--team-size` when the bot can play that size."""
    from nitro.dashboard.bots import watch_sizes
    from nitro.model import play_settings

    config = folder.load_settings(state)
    size = getattr(args, "team_size", None)
    if size is None or size == config.env.team_size:
        return config
    if size not in watch_sizes(config):
        raise ConfigError(
            folder.settings,
            [
                ConfigIssue(
                    "env.team_size",
                    f"this bot reads {config.env.team_size}v{config.env.team_size} games only, so it cannot "
                    f"play {size}v{size} (a padded observation, env.obs_team_size, lets a bot play smaller "
                    "games; an attention network, model.encoder, every size)",
                )
            ],
        )
    return play_settings(config, size)


def _write_watch(folder: RunFolder, state: dict[str, Any], args: argparse.Namespace, follow: bool) -> Path:
    """Records a game and writes its page (`watch.html`) and the recording itself (`watch.nlrec`) into the
    run folder. The arena's meshes stay out of the page, which may be passed on: they go into one script
    file for all the runs' pages, and a page without that file draws a generated arena."""
    from nitro import viewer
    from nitro.viewer import recording as recordings
    from nitro.watch import page, record

    config = _watch_settings(folder, state, args)
    recording = record(config, state, args.seconds, args.seed, args.fps)
    title = f"{folder.root.name} at {state['agent_steps']:,} agent steps"
    size = config.env.team_size
    subtitle = (
        f"{size}v{size}, {args.seconds:g} seconds of play (seed {args.seed}); the saved bot drives every "
        "car, choosing as it does in training."
        + (" The page loads the newest save at the end of each game." if follow else "")
    )
    shared = folder.root.parent / ".viewer"
    arena = viewer.keep_arena_script(shared, config.paths.collision_meshes)
    arena_src = f"../{shared.name}/{arena.name}" if arena is not None else None
    out = folder.root / "watch.html"
    _replace_text(out, page(recording, title, subtitle, follow=follow, arena_src=arena_src))
    recordings.save(folder.root / f"watch{recordings.SUFFIX}", recording.data(title, subtitle))
    return out


def _remove(file: Path, seconds: float = 10.0) -> None:
    """Removes `file`, trying again for up to `seconds` while something still holds it open (on Windows the
    closed window's web engine can keep the page open a moment longer); a file that stays held is left."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            file.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                print(f"Could not remove {file}: it is still in use.", file=sys.stderr)
                return
            time.sleep(0.2)


def _replace_text(file: Path, text: str) -> None:
    """Writes `file` whole: a page that is open never reads half of a new one."""
    partial = file.with_name(file.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, file)


def _eval(args: argparse.Namespace) -> int:
    from nitro.eval import EvalError, load_side, pick_line, play, randomness_line
    from nitro.runs import RunFolder

    if args.as_bot and (args.blue_temperature is not None or args.orange_temperature is not None):
        print(
            "--as-bot sets both sides to their likeliest action: leave out the temperatures.", file=sys.stderr
        )
        return 1
    blue_t = 1.0 if args.blue_temperature is None else args.blue_temperature
    orange_t = 1.0 if args.orange_temperature is None else args.orange_temperature
    pick = (0.0, 0.0) if args.as_bot else (blue_t, orange_t)
    entropy: list[float] = []
    if args.both_ways and (
        args.as_bot or args.blue_temperature is not None or args.orange_temperature is not None
    ):
        print(
            "--both-ways plays both sides sampled as in training; leave out --as-bot and the temperatures.",
            file=sys.stderr,
        )
        return 1
    if args.log and not args.both_ways:
        print("--log records both-ways matches only: add --both-ways.", file=sys.stderr)
        return 1
    try:
        blue = load_side(RunFolder(args.runs / args.name), args.at)
        if args.vs is not None:
            orange = load_side(RunFolder(args.runs / args.vs), args.vs_at)
        elif args.vs_at is not None:
            orange = load_side(RunFolder(args.runs / args.name), args.vs_at)
        else:
            orange = load_side(RunFolder(args.runs / args.name), None, oldest=True)
            if orange.agent_steps == blue.agent_steps:
                print(
                    f"'{args.name}' has only one save, so there is nothing older to play against. Train "
                    "more, or name an opponent with --vs or --vs-at.",
                    file=sys.stderr,
                )
                return 1
        if args.both_ways:
            return _eval_both_ways(args, blue, orange)
        print(f"Playing {args.games} points: {blue.label} (Blue) vs {orange.label} (Orange)...")
        print(pick_line(pick))
        result = play(
            blue, orange, args.games, args.seed, args.team_size, args.max_seconds, pick=pick, entropy=entropy
        )
    except EvalError as err:
        print(err, file=sys.stderr)
        return 1
    print("\n".join(result.lines(blue.label, orange.label)))
    print(randomness_line(entropy))
    return 0


def _eval_both_ways(args: argparse.Namespace, a: Side, b: Side) -> int:
    """`nitro eval --both-ways`: `a` as Blue, then as Orange, from the same starts; with --log, a row
    in runs/<name>/evals.csv."""
    from nitro.eval import EVAL_COLUMNS, append_row, eval_row, play_both_ways

    print(
        f"Playing {args.games} points each way: {a.label} as Blue, then as Orange, against {b.label}, "
        "from the same starts..."
    )
    result = play_both_ways(a, b, args.games, args.seed, args.team_size, args.max_seconds)
    print("\n".join(result.lines(a.label, b.label)))
    if args.log:
        path = args.runs / args.name / "evals.csv"
        append_row(path, EVAL_COLUMNS, eval_row(a, b, result, args.seed))
        print(f"Added to {path}.")
    return 0


def _skills(args: argparse.Namespace) -> int:
    from nitro.eval import EvalError, load_side
    from nitro.evaluate import check_pick, pick_rule
    from nitro.runs import RunFolder
    from nitro.skills import DEFAULT_TASKS, append_csv, run_battery, skills_file

    folder = RunFolder(args.runs / args.name)
    try:
        check_pick((args.temperature, 1.0))
    except ValueError:
        print(
            f"--temperature must be 0 (likeliest) or a positive number (got {args.temperature:g}).",
            file=sys.stderr,
        )
        return 1
    try:
        side = load_side(folder, args.at)
    except EvalError as err:
        print(err, file=sys.stderr)
        return 1
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    print(f"Scoring {side.label} on {len(tasks)} skills, {args.games} situations each...")
    if args.temperature != 1:
        print(f"Picking actions: {pick_rule(args.temperature)}.")
    results = run_battery(side, tasks, args.games, args.seed, args.temperature)
    for result in results:
        print(result.line())
    saved = folder.root / skills_file(args.temperature)
    append_csv(saved, side.agent_steps, results, args.seed)
    print(
        f"Saved to {saved}. The bar is what a fresh bot trained on that one skill alone "
        "reaches; it says whether the skill is there, not how good the bot is."
    )
    return 0


def _surgery(args: argparse.Namespace) -> int:
    from nitro.surgery import SurgeryError, widen_run

    try:
        report = widen_run(args.runs, args.name, args.widen, args.as_name, args.at)
    except SurgeryError as err:
        print(err, file=sys.stderr)
        return 1
    print("\n".join(report.lines()))
    return 0


def _pack(args: argparse.Namespace) -> int:
    from nitro import pack

    try:
        out, notes = pack.pack(
            args.runs / args.name, args.to, args.all_saves, args.include_meshes and not args.no_meshes
        )
    except pack.PackError as err:
        print(f"Cannot pack {args.name}: {err}", file=sys.stderr)
        return 1
    print(f"Packed {args.name} into {out} ({out.stat().st_size / 2**20:,.1f} MB).")
    for note in notes:
        print(f"Note: {note}")
    print(
        f"On the other computer, in NitroLearn's folder: nitro unpack {out.name}, then "
        f"nitro resume {args.name}"
    )
    return 0


def _unpack(args: argparse.Namespace) -> int:
    from nitro import pack
    from nitro.runs import RunFolder, resume_command

    try:
        done = pack.unpack(args.file, args.runs, args.name, args.replace)
    except (pack.PackError, OSError, KeyError, ValueError) as err:
        print(f"Cannot unpack {args.file}: {err}", file=sys.stderr)
        return 1
    name = done.folder.name
    print(f"Unpacked {name} into {done.folder}.")
    for note in done.notes:
        print(f"Note: {note}")
    if done.rust:
        print("It uses Rust rewards of its own, now added to NitroLearn: build them in with nitro rebuild.")
    print(f"Carry on training it with: {resume_command(RunFolder(done.folder))}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from nitro.export import ExportError, export_run
    from nitro.runs import RunFolder

    folder = RunFolder(args.runs / args.name)
    if not folder.info.is_file():
        print(f"There is no run called '{args.name}' in {args.runs}.", file=sys.stderr)
        return 1
    out = args.out or Path("bots") / args.name
    try:
        exported = export_run(folder, out, args.bot_name, build=not args.no_build)
    except ExportError as err:
        print(err, file=sys.stderr)
        return 1
    print(f"Exported to {exported.folder}:")
    for file in exported.files:
        print(f"  {file.name}")
    if exported.built:
        print("Add this folder to RLBot v5 as a bot folder and the bot is ready to play (README.md there).")
        return 0
    if args.no_build:
        print("Building the bot program was skipped (--no-build): export again without it to add it.")
        return 0
    print(f"The bot program is missing: {exported.build_message}", file=sys.stderr)
    print("The folder's README.md says how to build it.", file=sys.stderr)
    return 1


def _ab(args: argparse.Namespace) -> int:
    from nitro.ab import Experiment, compare_on_task, compare_speed, curve_marks, differences, not_measured
    from nitro.tasks import TASKS, practice_config

    settings = {"A": load_config(args.a), "B": load_config(args.b)}
    task = TASKS[args.task] if args.task else None
    if args.with_rewards and task is None:
        print(
            "--with-rewards compares two reward designs on a practice task: add --task TASK.",
            file=sys.stderr,
        )
        return 1
    if (args.also_likeliest or args.likeliest_decides) and task is None:
        print(
            "--also-likeliest and --likeliest-decides score a practice task: add --task TASK.",
            file=sys.stderr,
        )
        return 1
    if task is None and (args.checkpoints or args.decide != "final" or args.auc_smallest is not None):
        print(
            "--checkpoints, --decide and --auc-smallest score a practice task's learning curve: add "
            "--task TASK.",
            file=sys.stderr,
        )
        return 1
    if task is not None:
        try:
            curve_marks(
                args.steps or task.steps,
                args.checkpoints,
                args.decide,
                args.auc_smallest,
                likeliest_decides=args.likeliest_decides,
            )
        except ValueError as err:
            print(err, file=sys.stderr)
            return 1
    if task is None:
        # Left out of the speed runs (nitro.ab.not_measured), so no difference of its own to measure.
        changed = [c for c in differences(settings["A"], settings["B"]) if not c.startswith("replay_mix.")]
    else:
        changed = differences(*(practice_config(task, settings[s], 0, args.with_rewards) for s in ("A", "B")))
    unused = not_measured(settings.values(), task is not None)
    if not changed:
        for line in unused:
            print(line, file=sys.stderr)
        print(
            "A and B would train exactly the same way here. (On a practice task only the settings for how "
            "the bot learns count: [model], [ppo], [run] precision and deterministic, the observation and "
            "actions, and how the training data is arranged; add --with-rewards to compare the files' "
            "[[rewards]] as well.)",
            file=sys.stderr,
        )
        return 1
    try:
        name = args.name or f"{args.a.stem}-vs-{args.b.stem}-{args.task or 'speed'}"
        experiment = Experiment(args.folder / name, {"A": args.a, "B": args.b})
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    header = [f"A: {args.a}", f"B: {args.b}", "What differs:", *(f"  {line}" for line in changed)]
    header += unused
    print("\n".join(header))
    print(f"Results go to {experiment.folder}. If this stops, the same command carries on where it was.")
    try:
        if task is None:
            lines = compare_speed(experiment, settings, args.repeats)
        else:
            lines = compare_on_task(
                experiment,
                task,
                settings,
                list(range(args.seeds)),
                args.steps or task.steps,
                with_rewards=args.with_rewards,
                also_likeliest=args.also_likeliest,
                likeliest_decides=args.likeliest_decides,
                checkpoints=args.checkpoints,
                decide=args.decide,
                auc_smallest=args.auc_smallest,
            )
    except ValueError as err:  # the folder holds a comparison made with other settings
        print(err, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to carry on.")
        return 1
    experiment.finish([*header, *lines])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nitro", description="NitroLearn - train Rocket League bots.")
    parser.add_argument("--version", action="version", version=f"NitroLearn {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    dashboard = commands.add_parser(
        "dashboard", help="open the dashboard: make, train, change, watch and export bots in a window"
    )
    dashboard.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the bots (default: runs)"
    )
    dashboard.add_argument("--debug", action="store_true", help="open the window's developer tools")
    dashboard.set_defaults(run=_dashboard)

    rebuild_command = commands.add_parser(
        "rebuild", help="build NitroLearn again, with your own Rust rewards (my-settings/rust-files.txt)"
    )
    rebuild_command.add_argument(
        "--files",
        type=Path,
        nargs="*",
        help="list these reward files (full paths) first, in place of the list",
    )
    rebuild_command.set_defaults(run=_rebuild)

    update_command = commands.add_parser(
        "update",
        help="update NitroLearn to its newest release (your bots, settings, replays and saves are kept)",
    )
    update_command.add_argument(
        "--check", action="store_true", help="only say whether a newer version is out; change nothing"
    )
    update_command.add_argument("--yes", action="store_true", help="update without asking first")
    update_command.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the bots (default: runs)"
    )
    # For the dashboard: when the installer runs ("now", "window" or "later") and where the result goes.
    update_command.add_argument(
        "--installer", choices=("now", "window", "later"), default="", help=argparse.SUPPRESS
    )
    update_command.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    update_command.set_defaults(run=_update)

    doctor = commands.add_parser("doctor", help="check that this PC is ready to train")
    doctor.add_argument("settings", nargs="?", type=Path, help="settings file to check paths from (optional)")
    doctor.add_argument("--json", action="store_true", help="print the checks as JSON, for a program to read")
    doctor.set_defaults(run=_doctor)

    fetch = commands.add_parser(
        "fetch-meshes", help="download the arena's collision meshes from RLGym's package (checked by hash)"
    )
    fetch.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "collision_meshes",
        help="folder to put them in, as a 'soccar' subfolder (default: NitroLearn's collision_meshes)",
    )
    fetch.add_argument(
        "--archive",
        type=Path,
        help="a downloaded copy of rlgym-rocket-league-2.0.1.tar.gz to take them from (no internet needed)",
    )
    fetch.set_defaults(run=_fetch_meshes)

    recordings = commands.add_parser(
        "fetch-recordings",
        help="download the five real-game recordings RocketSim ships (119 MB, checked by hash; runs that "
        "use them download them by themselves)",
    )
    recordings.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "rocketsim_recordings",
        help="folder to put them in (default: NitroLearn's rocketsim_recordings, where runs look for them)",
    )
    recordings.set_defaults(run=_fetch_recordings)

    rlbot = commands.add_parser(
        "install-rlbot",
        help="Windows: download RLBot v5's official installer and open it (RLBot is a separate program)",
    )
    rlbot.add_argument(
        "--runs", type=Path, default=Path("runs"), help="the dashboard's bots folder (default: runs)"
    )
    rlbot.add_argument("--status", action="store_true", help="only say whether RLBot is installed, and where")
    rlbot.set_defaults(run=_install_rlbot)

    check = commands.add_parser("check-settings", help="check a settings file for mistakes")
    check.add_argument("settings", type=Path, help="the settings file (.toml)")
    check.set_defaults(run=_check_settings)

    runs = argparse.ArgumentParser(add_help=False)
    runs.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    runs.add_argument(
        "--steps",
        type=int,
        help="stop and save once the run has this many agent steps in total"
        " (one car's one decision is one step)",
    )

    train = commands.add_parser("train", parents=[runs], help="start a new training run")
    train.add_argument("settings", type=Path, help="the settings file (.toml)")
    train.add_argument("--name", help="name of the run's folder (default: run-<date>-<time>)")
    train.add_argument(
        "--from",
        dest="start_from",
        metavar="RUN",
        help="start from another run's trained networks instead of from scratch (transfer learning; see "
        "[start_from] in the settings for which parts)",
    )
    train.add_argument(
        "--from-at", dest="start_at", type=int, default=0, metavar="STEPS", help="that run's save at STEPS"
    )
    train.set_defaults(run=_train)

    new_command = commands.add_parser(
        "new", parents=[runs], help="make a bot from a settings file without starting it (to pack it)"
    )
    new_command.add_argument("settings", type=Path, help="the settings file (.toml)")
    new_command.add_argument("--name", help="name of the bot's folder (default: run-<date>-<time>)")
    new_command.set_defaults(run=_new)

    resume = commands.add_parser("resume", parents=[runs], help="continue a run from its last save")
    resume.add_argument("name", help="the run's name (its folder in --runs)")
    resume.set_defaults(run=_resume)
    for command in (train, resume):
        command.add_argument(
            "--then-train",
            action="store_true",
            help="with [pretrain]: train straight after the copy of the replays instead of stopping once it "
            "is saved",
        )

    copy_save = commands.add_parser(
        "copy-save",
        parents=[runs],
        help="save a bot's unfinished copy of replays as its save (the copy does this itself)",
    )
    copy_save.add_argument("name", help="the run's name (its folder in --runs)")
    copy_save.set_defaults(run=_copy_save)

    trial = commands.add_parser(
        "reward-check",
        help="play a run's newest save, learning nothing, and say what each reward would pay it",
    )
    trial.add_argument("name", help="the run's name (its folder in --runs)")
    trial.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    trial.add_argument(
        "--rewards",
        type=Path,
        metavar="FILE",
        help="try the [[rewards]] of this reward set or settings file (default: the run's own rewards)",
    )
    trial.add_argument(
        "--steps", type=int, default=1_000_000, help="agent steps to play (default: 1,000,000)"
    )
    trial.add_argument(
        "--out", type=Path, help="where to write the result (default: in the run's reward_checks folder)"
    )
    trial.set_defaults(run=_reward_check)

    convert_command = commands.add_parser(
        "convert-replays",
        help="convert replays for later replay copies, learning nothing (every thread, low priority)",
    )
    convert_command.add_argument(
        "folders", nargs="*", type=Path, help="folders of .replay files (default: the game's own)"
    )
    convert_command.add_argument(
        "--limit", type=int, default=0, help="at most this many, newest first (0 = all)"
    )
    convert_command.add_argument(
        "--workers", type=int, default=0, help="replays converted at once (default: every thread of this PC)"
    )
    convert_command.add_argument(
        "--meshes", type=Path, default=Path("collision_meshes"), help="the arena meshes folder"
    )
    convert_command.set_defaults(run=_convert_replays)

    pre = commands.add_parser(
        "pretrain", parents=[runs], help="teach a bot to copy recorded games (behaviour cloning)"
    )
    pre.add_argument("settings", type=Path, help="the settings file (.toml) of the runs that will train on")
    pre.add_argument(
        "recordings",
        nargs="*",
        type=Path,
        help=".rlpr / .rlpr.zst recordings, or folders of them",
    )
    pre.add_argument(
        "--rocketsim",
        action="store_true",
        help="also the five real-game recordings RocketSim ships (MIT; downloaded once, 119 MB)",
    )
    pre.add_argument("--name", help="name of the new run folder (default: run-<date>-<time>)")
    pre.add_argument("--epochs", type=int, default=10, help="passes over the samples (default: 10)")
    pre.add_argument("--batch", type=int, default=4096, help="samples per learning step (default: 4096)")
    pre.add_argument("--lr", type=float, default=3e-4, help="learning rate (default: 3e-4)")
    pre.add_argument(
        "--softness",
        type=float,
        default=0.1,
        help="how evenly a mix of controls is shared between the actions near it (default: 0.1)",
    )
    pre.add_argument(
        "--rows",
        choices=("auto", "replicar", "recorded"),
        default="auto",
        help="what a recording's ticks hold: recordings made from replays with replicar hold the "
        "controls applied next (replicar); the bot's own and RocketSim's recordings hold the controls that "
        "produced each tick (recorded); auto decides by where the file is: replicar inside the project's "
        "replays folder, where the converters write unless told otherwise, recorded elsewhere; it says how "
        "many it read each way (default: auto)",
    )
    pre.add_argument(
        "--held-out",
        type=float,
        default=0.1,
        help="share of recordings kept back for measuring (default: 0.1)",
    )
    pre.set_defaults(run=_pretrain)

    charts = commands.add_parser(
        "wandb",
        help="send a run's charts to Weights & Biases: its copy of replays and its training",
        description="For a run made before [wandb] was on, or trained offline: sends its copy of replays "
        "(round by round, or pass by pass) and its training (metrics.csv) to W&B, skipping what was sent "
        "before. Log in once with `wandb login` first.",
    )
    charts.add_argument("name", help="the run's name (its folder in --runs)")
    charts.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    charts.set_defaults(run=_wandb)

    watch = commands.add_parser(
        "watch", help="watch a game played by a run's newest saved bot, in the 3D match viewer"
    )
    watch.add_argument("name", help="the run's name (its folder in --runs)")
    watch.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    watch.add_argument(
        "--at",
        type=int,
        help="play the save made at this many agent steps instead of the newest (the step counts are the "
        "folder names in runs/<name>/checkpoints)",
    )
    watch.add_argument("--seconds", type=float, default=60.0, help="length of the game (default: 60)")
    watch.add_argument("--seed", type=int, default=0, help="a different seed plays a different game")
    watch.add_argument(
        "--fps",
        type=int,
        default=60,
        help="pictures per second of game time, up to 120 (default: 60); rounded so every decision falls on "
        "a picture",
    )
    watch.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    watch.add_argument(
        "--follow",
        action="store_true",
        help="keep going: after every new save, record a new game; the open page shows it from its next "
        "game on",
    )
    watch.add_argument(
        "--team-size",
        type=int,
        choices=(1, 2, 3, 4),
        help="play this game size instead of the run's own (bots with a padded observation can play smaller "
        "games than their room; bots with an attention network every size, beyond what they trained at "
        "unmeasured)",
    )
    watch.add_argument(
        "--live",
        action="store_true",
        help="play on in real time and show it as it is played, in a window of its own (nothing is served on "
        "any port); closing the window or Ctrl+C stops it; --seconds does not apply; with --follow each new "
        "save takes over",
    )
    # For programs that show the live game themselves: the feed's lines on standard output, no window.
    watch.add_argument("--feed", action="store_true", help=argparse.SUPPRESS)
    watch.set_defaults(run=_watch)

    evaluate = commands.add_parser(
        "eval",
        help="play a run's newest save against an older save or another run",
        description="Plays a run's save (Blue) against another save (Orange) under the run's own settings, "
        "many games at once, and reports Blue's share of the goals with a 95% interval. Without --vs or "
        "--vs-at the opponent is the run's oldest save: how far has it come?",
    )
    evaluate.add_argument("name", help="the run's name (its folder in --runs)")
    evaluate.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    evaluate.add_argument(
        "--at", type=int, help="play this run's save at this many agent steps (default: newest)"
    )
    evaluate.add_argument("--vs", help="the opponent run's name (default: this run)")
    evaluate.add_argument(
        "--vs-at",
        type=int,
        help="the opponent's save at this many agent steps (default: the other run's newest, or this "
        "run's oldest)",
    )
    evaluate.add_argument("--games", type=int, default=512, help="points to play (default: 512)")
    evaluate.add_argument(
        "--team-size",
        type=int,
        choices=(1, 2, 3, 4),
        help="play this game size instead of the run's own, when both bots can (a padded observation lets a "
        "bot play smaller games than its room, an attention network every size)",
    )
    evaluate.add_argument("--seed", type=int, default=0, help="a different seed plays different games")
    evaluate.add_argument(
        "--max-seconds",
        type=float,
        help="end a point with no goal after this many seconds of game time, for this match only (needed "
        "for a run whose games have no time limit, terminals.max_episode_seconds = 0)",
    )
    evaluate.add_argument(
        "--blue-temperature",
        type=float,
        metavar="T",
        help="how Blue picks its actions: 1 = sampled as in training (default), 0 = its likeliest action as "
        "the exported RLBot bot plays, between = less random than training",
    )
    evaluate.add_argument(
        "--orange-temperature", type=float, metavar="T", help="the same for Orange (default: 1)"
    )
    evaluate.add_argument(
        "--as-bot",
        action="store_true",
        help="both sides pick their likeliest action, as the exported RLBot bots play (both temperatures 0)",
    )
    evaluate.add_argument(
        "--both-ways",
        action="store_true",
        help="play --games points with this run's save as Blue and as many as Orange, from the same starts, "
        "and report its share of the goals over both (the starts can favour one colour)",
    )
    evaluate.add_argument(
        "--log",
        action="store_true",
        help="with --both-ways: add the result as a row to runs/<name>/evals.csv",
    )
    evaluate.set_defaults(run=_eval)

    skills = commands.add_parser(
        "skills",
        help="score a run's save on the practice skills (ball touch, recovery, boost, shot, save, aerial)",
        description="Plays a run's save through each practice task's held-out situations with an idle "
        "opponent and scores it by the task's own measure against the task's bar. Appends the rows to "
        "runs/<name>/skills.csv.",
    )
    skills.add_argument("name", help="the run's name (its folder in --runs)")
    skills.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    skills.add_argument("--at", type=int, help="score the save at this many agent steps (default: newest)")
    skill_tasks = sorted(
        file.stem for file in (Path(__file__).parent / "tasks").glob("*.toml") if file.stem != "kickoff"
    )
    skills.add_argument("--tasks", nargs="+", choices=skill_tasks, help="which skills (default: all)")
    skills.add_argument("--games", type=int, default=512, help="situations per skill (default: 512)")
    skills.add_argument(
        "--seed", type=int, default=1_000_001, help="which held-out situations (default: 1000001)"
    )
    skills.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="how the bot picks its actions: 1 = sampled as in training (default), 0 = its likeliest "
        "action as the exported RLBot bot plays; other than 1, the rows go to a file of their own "
        "(skills_likeliest.csv for 0)",
    )
    skills.set_defaults(run=_skills)

    surgery = commands.add_parser(
        "surgery",
        help="make a run's networks wider mid-run, keeping how it plays",
        description="Makes a new run from a run's save with every hidden layer K times as wide, computing "
        "what the old networks did to within the computer's rounding (checked on random observations "
        "before anything is written), so training can go on with bigger networks instead of starting over. "
        "The old run is left as it is; continue with `nitro resume <new run>`. Adam's running averages "
        "start afresh.",
    )
    surgery.add_argument("name", help="the run to widen (its folder in --runs)")
    surgery.add_argument(
        "--widen", type=int, required=True, help="how many times wider every hidden layer gets (2 = double)"
    )
    surgery.add_argument("--as", dest="as_name", help="name of the new run (default: <name>-x<K>)")
    surgery.add_argument(
        "--at", type=int, help="widen the save made at this many agent steps instead of the newest"
    )
    surgery.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    surgery.set_defaults(run=_surgery)

    export = commands.add_parser("export", help="turn a run's newest save into an RLBot v5 bot folder")
    export.add_argument("name", help="the run's name (its folder in --runs)")
    export.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    export.add_argument("--out", type=Path, help="the bot folder to write (default: bots/<name>)")
    export.add_argument("--bot-name", help="the name RLBot shows for the bot (default: the run's name)")
    export.add_argument(
        "--no-build", action="store_true", help="write the files but don't build the bot program"
    )
    export.set_defaults(run=_export)

    pack_command = commands.add_parser(
        "pack", help="put a bot into one file, to carry on training it on another computer"
    )
    pack_command.add_argument("name", help="the bot's name (its folder in --runs)")
    pack_command.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    pack_command.add_argument(
        "--to",
        type=Path,
        help="the file to write (default: <name>-<date>-<time>.nitropack.zip in this folder; an existing "
        "file is never replaced)",
    )
    pack_command.add_argument(
        "--all-saves", action="store_true", help="every save, not only the newest (a bigger file)"
    )
    pack_command.add_argument(
        "--include-meshes",
        action="store_true",
        help="also pack the arena meshes (Rocket League's own files: for your own computers only, never "
        "to share)",
    )
    pack_command.add_argument(
        "--no-meshes",
        action="store_true",
        help="leave the arena meshes out (the default; kept for older scripts)",
    )
    pack_command.set_defaults(run=_pack)

    unpack_command = commands.add_parser("unpack", help="add a bot packed with `nitro pack`")
    unpack_command.add_argument("file", type=Path, help="the .nitropack.zip file")
    unpack_command.add_argument(
        "--runs", type=Path, default=Path("runs"), help="folder that holds the runs (default: runs)"
    )
    unpack_command.add_argument("--name", default="", help="the bot's name here (default: its own)")
    unpack_command.add_argument(
        "--replace",
        action="store_true",
        help="replace a bot of that name here (it is kept in the runs folder's .replaced)",
    )
    unpack_command.set_defaults(run=_unpack)

    ab = commands.add_parser(
        "ab",
        help="compare two settings files on the same seeds",
        description="Compare two settings files on the same seeds, with a plain-English verdict that "
        "allows for the noise between seeds. On a practice task, a fresh bot is trained per seed and setting "
        "and judged by how well it does the task; with --speed, by training speed.",
    )
    ab.add_argument("a", type=Path, help="settings file A (.toml)")
    ab.add_argument("b", type=Path, help="settings file B (.toml)")
    kind = ab.add_mutually_exclusive_group(required=True)
    tasks = sorted(file.stem for file in (Path(__file__).parent / "tasks").glob("*.toml"))
    kind.add_argument("--task", choices=tasks, help="the practice task to compare on")
    kind.add_argument("--speed", action="store_true", help="compare training speed instead")
    ab.add_argument("--seeds", type=int, default=10, help="seeds per setting on a task (default: 10)")
    ab.add_argument(
        "--with-rewards",
        action="store_true",
        help="on a task: the two files' [[rewards]] replace the task's, to compare two reward designs "
        "(still judged by the task's measure)",
    )
    ab.add_argument(
        "--also-likeliest",
        action="store_true",
        help="on a task: also score each trained bot picking its likeliest action, as the RLBot bot plays; "
        "the sampled score still decides",
    )
    ab.add_argument(
        "--likeliest-decides",
        action="store_true",
        help="on a task: as --also-likeliest, but the likeliest-action score decides; written into the "
        "comparison's plan before the first run (not with --decide auc)",
    )
    ab.add_argument("--steps", type=int, help="agent steps of training per run (default: the task's)")
    ab.add_argument(
        "--checkpoints",
        type=int,
        default=0,
        metavar="K",
        help="on a task: evaluate at K evenly spaced points incl. the end (2 to 10; default 0, off); shown, "
        "decides only with --decide auc",
    )
    ab.add_argument(
        "--decide",
        choices=("final", "auc"),
        default="final",
        help="on a task: what gives the verdict, the final point (default) or the area under the curve of "
        "the --checkpoints points; auc only under a plan written before the runs",
    )
    ab.add_argument(
        "--auc-smallest",
        type=float,
        help="with --decide auc (needed there, refused otherwise): the smallest difference in the area under "
        "the curve that matters, fixed in that plan",
    )
    ab.add_argument("--repeats", type=int, default=6, help="runs per setting with --speed (default: 6)")
    ab.add_argument("--folder", type=Path, default=Path("ab"), help="folder for the results (default: ab)")
    ab.add_argument("--name", help="the comparison's name (default: <A>-vs-<B>-<task>)")
    ab.set_defaults(run=_ab)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from nitro.build_info import CoreOutOfDate, check_core
    from nitro.recordings import RecordingFetchError

    args = build_parser().parse_args(argv)
    try:
        # These never use the core (rebuild makes it; the dashboard only lists and starts commands).
        # An update must also work on an old core.
        never_core = (
            _doctor,
            _fetch_meshes,
            _fetch_recordings,
            _install_rlbot,
            _check_settings,
            _surgery,
            _export,
            _rebuild,
            _dashboard,
            _update,
        )
        if args.run not in never_core:
            check_core()  # after a `git pull`, before anything runs on an old core
        code: int = args.run(args)
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 1
    except CoreOutOfDate as err:
        print(err, file=sys.stderr)
        return 1
    except RecordingFetchError as err:
        print(f"RocketSim's recordings (bundled = true) are not on this computer: {err}.", file=sys.stderr)
        return 1
    except Exception as err:
        runs = sys.modules.get("nitro.runs")  # loaded by every command that reads a save
        if runs is not None and isinstance(err, runs.UnreadableSave):
            print(err, file=sys.stderr)
            return 1
        raise
    return code
