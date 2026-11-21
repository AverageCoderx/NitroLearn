"""run.auto_restart: the keeper starts a crashed trainer again, stops at its limit, never undoes a stop
the user asked for, and keeps a start-up entry only while a run needs it."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from nitro import keeper
from nitro.dashboard import winapp
from nitro.dashboard.bots import RUNNING, STOP_REQUEST
from nitro.runs import RunFolder

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

# A stand-in trainer: each start appends a line to starts.txt, then does what plan.json says for that start.
CHILD = """
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
starts = root / "starts.txt"
n = len(starts.read_text().splitlines()) if starts.exists() else 0
with starts.open("a") as f:
    f.write("start\\n")
plan = json.loads((root / "plan.json").read_text())
step = plan[min(n, len(plan) - 1)]
if step.get("save"):
    (root / "checkpoints" / f"{step['save']:012d}").mkdir(parents=True)
    (root / "checkpoints" / f"{step['save']:012d}" / "state.pt").write_text("x")
if step.get("user_stop"):
    (root / "user_stop").write_text("stop")
if step.get("stop_request"):  # the dashboard's Stop as this start ends
    (root / "stop.request").write_text("stop")
if step.get("sleep"):
    time.sleep(step["sleep"])
print(step.get("say", "child ending"), flush=True)
sys.exit(step["code"])
"""


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunFolder, Path]:
    runs = tmp_path / "runs"
    folder = RunFolder(runs / "bot")
    folder.root.mkdir(parents=True)
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    monkeypatch.setattr(keeper, "_child_command", lambda f, s, t: [sys.executable, str(script), str(f.root)])
    monkeypatch.setenv(keeper.DELAY_ENV, "0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(keeper, "project_root", lambda: tmp_path / "nitrolearn")
    return folder, tmp_path


def plan(folder: RunFolder, steps: list[dict[str, object]]) -> None:
    (folder.root / "plan.json").write_text(json.dumps(steps), encoding="utf-8")


def starts(folder: RunFolder) -> int:
    return len((folder.root / "starts.txt").read_text().splitlines())


def test_a_crash_loop_stops_after_the_limit_and_says_why(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    plan(folder, [{"code": 3, "say": "CUDA error: something broke"}])
    assert keeper.keep(folder, None, False, limit=2) == 3
    assert starts(folder) == 3  # the first start and 2 restarts
    state = keeper.read_state(folder)
    assert not state["armed"]
    assert [e["kind"] for e in state["events"]] == ["crash", "crash", "gave_up"]
    shown = keeper.banner(folder)
    assert shown is not None and shown["bad"]
    assert "3 crashes in a row without a new save" in shown["text"]
    assert "exit code 3" in shown["text"]
    assert not (folder.root / keeper.KEEPER).exists()


def test_a_new_save_resets_the_count(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    plan(folder, [{"code": 1, "save": 100}, {"code": 1, "save": 200}, {"code": 1, "save": 300}, {"code": 1}])
    assert keeper.keep(folder, None, False, limit=1) == 1
    assert starts(folder) == 5  # three crashes after a new save each, then two without one


def test_a_stop_the_user_asked_for_is_never_undone(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    plan(folder, [{"code": 1, "user_stop": True}])  # stopped by the user, then crashed while saving
    assert keeper.keep(folder, None, False, limit=3) == 1
    assert starts(folder) == 1
    assert keeper.banner(folder) is None
    assert not keeper.read_state(folder)["armed"]


def test_a_finished_run_ends_the_keeper(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    plan(folder, [{"code": 0}])
    assert keeper.keep(folder, None, False, limit=3) == 0
    assert starts(folder) == 1


def test_an_apply_starts_it_again_without_counting_a_crash(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    plan(folder, [{"code": keeper.RESTART_CODE}, {"code": keeper.RESTART_CODE}, {"code": 0}])
    assert keeper.keep(folder, None, False, limit=1) == 0
    assert starts(folder) == 3
    assert keeper.banner(folder) is None


def test_stop_while_it_waits_to_start_again(
    setup: tuple[RunFolder, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, _ = setup
    monkeypatch.setenv(keeper.DELAY_ENV, "30")
    (folder.root / STOP_REQUEST).write_text("stop\n")  # older than the keeper: not for it
    plan(folder, [{"code": 1, "stop_request": True}, {"code": 1}])  # then the dashboard's Stop
    began = time.monotonic()
    assert keeper.keep(folder, None, False, limit=3) == 0
    assert time.monotonic() - began < 10
    assert starts(folder) == 1


def test_ctrl_c_in_the_keepers_window_is_a_users_stop(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    plan(folder, [{"code": 1, "sleep": 1.5}])
    # raise_signal, not os.kill: on Windows os.kill with SIGINT ends the process outright.
    timer = __import__("threading").Timer(0.5, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    assert keeper.keep(folder, None, False, limit=3) == 1
    assert starts(folder) == 1


def test_the_start_up_entry_lives_while_a_run_is_armed(setup: tuple[RunFolder, Path]) -> None:
    folder, tmp = setup
    entry = keeper.startup_entry(folder.root.parent)
    assert entry is not None and entry.parent == tmp / "config" / "autostart"
    seen: dict[str, object] = {}
    plan(folder, [{"code": 0, "sleep": 1.0}])

    def look() -> None:
        seen["made"] = entry.is_file()
        seen["record"] = json.loads((tmp / "nitrolearn" / winapp.SHORTCUTS).read_text())["shortcuts"]
        seen["text"] = entry.read_text()

    timer = __import__("threading").Timer(0.5, look)
    timer.start()
    keeper.keep(folder, None, False, limit=3)
    timer.join()
    assert seen["made"]
    assert seen["record"] == [str(entry)]
    assert "carry-on --runs" in str(seen["text"])
    assert not entry.exists()
    assert not (tmp / "nitrolearn" / winapp.SHORTCUTS).exists()


def test_uninstalling_removes_the_start_up_entry(tmp_path: Path) -> None:
    entry = tmp_path / "autostart" / "NitroLearn carry on training 1234.desktop"
    entry.parent.mkdir()
    entry.write_text("[Desktop Entry]\n")
    winapp.record_shortcuts(tmp_path, [entry])
    removed, kept, _ = winapp.remove_shortcuts(tmp_path)
    assert removed == [str(entry)] and not kept and not entry.exists()


def test_carry_on_starts_armed_runs_only(
    setup: tuple[RunFolder, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, _ = setup
    other = RunFolder(folder.root.parent / "other")
    other.root.mkdir()
    (folder.root / keeper.STATE).write_text(json.dumps({"armed": True}))
    (other.root / keeper.STATE).write_text(json.dumps({"armed": False}))
    spawned: list[tuple[list[str], dict[str, str] | None]] = []
    monkeypatch.setattr(
        "nitro.dashboard.jobs.spawn", lambda args, f, cwd, token="", env=None: spawned.append((args, env))
    )
    assert keeper.carry_on(folder.root.parent) == 0
    assert spawned == [
        (["resume", "bot", "--runs", str(folder.root.parent.resolve())], {keeper.WHY_ENV: "reboot"})
    ]


def test_after_a_computer_restart_the_banner_says_so(
    setup: tuple[RunFolder, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, _ = setup
    plan(folder, [{"code": 0}])
    monkeypatch.setenv(keeper.WHY_ENV, "reboot")
    keeper.keep(folder, None, False, limit=3)
    shown = keeper.banner(folder)
    assert shown is not None and not shown["bad"]
    assert "after the computer restarted" in shown["text"]


def test_a_second_keeper_of_the_same_run_is_refused(setup: tuple[RunFolder, Path]) -> None:
    folder, _ = setup
    (folder.root / keeper.KEEPER).write_text(json.dumps({"pid": os.getpid()}))
    assert keeper.keep(folder, None, False, limit=3) == 1
    (folder.root / keeper.KEEPER).write_text(json.dumps({"pid": 2**22 + 12345}))  # a keeper long gone
    plan(folder, [{"code": 0}])
    assert keeper.keep(folder, None, False, limit=3) == 0


# ---------- the real trainer ----------

SETTINGS = """
[run]
seed = 7
deterministic = true
checkpoint_steps = 640
auto_restart = {auto}
[env]
num_arenas = 8
threads = 4
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 320
minibatches = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""


def _nitro(tmp: Path, *args: str) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        keeper.DELAY_ENV: "1",
        "XDG_CONFIG_HOME": str(tmp / "config"),
    }
    return subprocess.Popen(
        [sys.executable, "-m", "nitro", *args],
        cwd=tmp,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _settings(tmp: Path, name: str, auto: bool) -> Path:
    file = tmp / f"{name}.toml"
    text = SETTINGS.format(auto="true" if auto else "false")
    file.write_text(text + f'[paths]\ncollision_meshes = "{MESHES.as_posix()}"\n', encoding="utf-8")
    return file


def _tensors(value: object, prefix: str = "") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        return {k: t for key, item in value.items() for k, t in _tensors(item, f"{prefix}.{key}").items()}
    if isinstance(value, (list, tuple)):
        return {k: t for i, item in enumerate(value) for k, t in _tensors(item, f"{prefix}.{i}").items()}
    return {}


def _weights(folder: RunFolder) -> dict[str, torch.Tensor]:
    state = folder.load_latest()
    assert state is not None
    return _tensors(state["learner"])


@NEEDS_MESHES
def test_a_trainer_killed_mid_iteration_starts_again_and_ends_as_if_never_stopped(tmp_path: Path) -> None:
    steps = "3200"  # ten iterations, a save every two
    straight = _nitro(
        tmp_path, "train", str(_settings(tmp_path, "a", False)), "--name", "straight", "--steps", steps
    )
    out, _ = straight.communicate(timeout=900)
    assert straight.returncode == 0, out.decode()

    kept = _nitro(tmp_path, "train", str(_settings(tmp_path, "b", True)), "--name", "kept", "--steps", steps)
    folder = RunFolder(tmp_path / "runs" / "kept")
    deadline = time.monotonic() + 600
    victim = 0
    while time.monotonic() < deadline and kept.poll() is None:
        if folder.complete_checkpoints() and (folder.root / RUNNING).is_file():
            info = json.loads((folder.root / RUNNING).read_text())
            if info["pid"] != kept.pid:
                victim = info["pid"]
                time.sleep(0.3)  # into the next iteration
                os.kill(victim, getattr(signal, "SIGKILL", signal.SIGTERM))  # Windows: TerminateProcess
                break
        time.sleep(0.05)
    assert victim, "the kept trainer never reached its first save"
    out, _ = kept.communicate(timeout=900)
    text = out.decode()
    assert kept.returncode == 0, text
    assert "Auto restart: training stopped unexpectedly (exit code" in text
    state = keeper.read_state(folder)
    assert not state["armed"]
    assert [e["kind"] for e in state["events"]] == ["crash"]
    assert int(folder.complete_checkpoints()[-1].name) == int(steps)

    a, b = _weights(RunFolder(tmp_path / "runs" / "straight")), _weights(folder)
    for name, weights in a.items():
        assert torch.equal(weights, b[name]), f"{name} differs from the run that never stopped"
    assert not (tmp_path / "config" / "autostart").exists() or not any(
        (tmp_path / "config" / "autostart").iterdir()
    )


@NEEDS_MESHES
def test_the_dashboards_stop_on_a_kept_trainer_is_not_undone(tmp_path: Path) -> None:
    kept = _nitro(
        tmp_path, "train", str(_settings(tmp_path, "b", True)), "--name", "kept", "--steps", "100000"
    )
    folder = RunFolder(tmp_path / "runs" / "kept")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and kept.poll() is None:
        if (
            (folder.root / RUNNING).is_file()
            and folder.metrics.is_file()
            and folder.metrics.stat().st_size > 400
        ):
            (folder.root / STOP_REQUEST).write_text("stop\n")
            break
        time.sleep(0.05)
    out, _ = kept.communicate(timeout=600)
    text = out.decode()
    assert kept.returncode == 0, text
    assert "stopped unexpectedly" not in text
    assert text.count("Training 'kept'") == 1
    assert not keeper.read_state(folder)["armed"]


def test_uninstalling_removes_the_windows_start_up_entry(tmp_path: Path) -> None:
    """The uninstaller reads what each recorded shortcut opens (tools/app_shortcuts.ps1 -Remove): the
    keeper's start-up entry opens nitro.cmd carry-on, not the dashboard, and is still NitroLearn's to delete
    (also when the folder was moved since and its nitro.cmd is not there any more)."""
    root = tmp_path / "NitroLearn"
    root.mkdir()
    (root / "nitro.cmd").write_text("@echo off\n")
    startup = tmp_path / "Startup"
    startup.mkdir()
    here = startup / "NitroLearn carry on training 1234abcd.lnk"
    moved = startup / "NitroLearn carry on training 5678ef90.lnk"
    for link in (here, moved):
        link.write_bytes(b"lnk")
    winapp.record_shortcuts(root, [here, moved])
    found = {
        str(here): {"target": str(root / "nitro.cmd"), "arguments": f'carry-on --runs "{root / "runs"}"'},
        str(moved): {
            "target": str(tmp_path / "Old" / "nitro.cmd"),
            "arguments": 'carry-on --runs "C:/Old/runs"',
        },
    }
    removed, kept, left = winapp.remove_shortcuts(root, winapp.found_shortcuts(json.dumps(found)))
    assert sorted(removed) == sorted([str(here), str(moved)]) and not kept and not left
    assert not here.exists() and not moved.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a program that is not Python: sleep")
def test_a_keeper_number_another_program_took_after_a_restart_is_not_a_keeper(
    setup: tuple[RunFolder, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """keeper.json is left by a power cut; after the restart another program can have its process number.
    That is not the keeper: the run is started again, and a start by hand is not refused."""
    folder, _ = setup
    other = subprocess.Popen(["sleep", "30"])  # not Python
    try:
        (folder.root / keeper.KEEPER).write_text(json.dumps({"pid": other.pid}))
        (folder.root / keeper.STATE).write_text(json.dumps({"armed": True}))
        assert not keeper.alive(folder)
        spawned: list[list[str]] = []
        monkeypatch.setattr(
            "nitro.dashboard.jobs.spawn", lambda args, f, cwd, token="", env=None: spawned.append(args)
        )
        keeper.carry_on(folder.root.parent)
        assert len(spawned) == 1
        plan(folder, [{"code": 0}])
        assert keeper.keep(folder, None, False, limit=3) == 0
    finally:
        other.kill()
        other.wait()


def test_restarts_of_the_computer_without_a_new_save_count_as_crashes(
    setup: tuple[RunFolder, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training that brings the whole computer down (a driver crash, overheating) must not start again at
    every log-in for ever: restarts in a row without a new save count towards the limit."""
    folder, _ = setup
    plan(folder, [{"code": 0}])

    def restarted(crashes: int) -> int:
        (folder.root / keeper.STATE).write_text(
            json.dumps({"armed": True, "crashes": crashes, "last_save": -1})
        )
        monkeypatch.setenv(keeper.WHY_ENV, "reboot")
        return keeper.keep(folder, None, False, limit=1)

    assert restarted(0) == 0 and starts(folder) == 1  # the first restart without a save: started again
    assert restarted(1) == 1 and starts(folder) == 1  # one more: not started again, and says why
    assert not keeper.read_state(folder)["armed"]
    shown = keeper.banner(folder)
    assert shown is not None and shown["bad"] and "without a new save" in shown["text"]
    (folder.root / "checkpoints" / f"{50:012d}").mkdir(parents=True)
    (folder.root / "checkpoints" / f"{50:012d}" / "state.pt").write_text("x")
    assert restarted(5) == 0 and starts(folder) == 2  # it saved since: a restart is no crash


@NEEDS_MESHES
def test_the_dashboards_stop_while_a_kept_trainer_starts_is_not_lost(tmp_path: Path) -> None:
    """The dashboard's Stop between the keeper starting the trainer and the trainer training (it loads for
    a few seconds) stops it: the trainer used to delete the request as a stale one and train on."""
    kept = _nitro(
        tmp_path, "train", str(_settings(tmp_path, "b", True)), "--name", "kept", "--steps", "100000000"
    )
    folder = RunFolder(tmp_path / "runs" / "kept")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and kept.poll() is None and not (folder.root / keeper.KEEPER).is_file():
        time.sleep(0.02)
    assert not (folder.root / RUNNING).is_file(), "the trainer was quicker than this test expects"
    (folder.root / STOP_REQUEST).write_text("stop\n")  # what the dashboard's Stop writes then
    try:
        out, _ = kept.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        kept.kill()
        with __import__("contextlib").suppress(OSError, ValueError):
            os.kill(int(json.loads((folder.root / RUNNING).read_text())["pid"]), signal.SIGTERM)
        out, _ = kept.communicate(timeout=60)
        pytest.fail("the Stop was lost: it kept training\n" + out.decode()[-2000:])
    text = out.decode()
    assert kept.returncode == 0, text
    assert "stopped unexpectedly" not in text
    assert not keeper.read_state(folder)["armed"]


def test_carry_on_leaves_a_run_whose_auto_restart_was_turned_off(
    setup: tuple[RunFolder, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Armed is left on when the keeper itself is ended at once (its window closed, a power cut). Turned off
    since in the bot's settings, the run is not started again at the next log-in, nor at every one after."""
    folder, _ = setup
    folder.settings.write_text("[run]\nauto_restart = false\n", encoding="utf-8")
    (folder.root / keeper.STATE).write_text(json.dumps({"armed": True}))
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        "nitro.dashboard.jobs.spawn", lambda args, f, cwd, token="", env=None: spawned.append(args)
    )
    keeper.carry_on(folder.root.parent)
    assert spawned == []
    assert not keeper.read_state(folder)["armed"]
    folder.settings.write_text("[run]\nauto_restart = true\n", encoding="utf-8")
    (folder.root / keeper.STATE).write_text(json.dumps({"armed": True}))
    keeper.carry_on(folder.root.parent)
    assert len(spawned) == 1
