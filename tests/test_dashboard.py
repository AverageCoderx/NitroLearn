"""The dashboard: every setting described from its own definition, changes held until
Apply and written with the file's comments kept, bots started, stopped and restarted by Apply, and the page
itself."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import api as dashboard_api
from nitro.dashboard import bots, charts, describe, edits, layout
from nitro.dashboard import rewards as dashboard_rewards
from nitro.dashboard.api import Api
from nitro.dashboard.app import page_source
from nitro.dashboard.jobs import CONSOLE, read_console
from nitro.runs import FIXED_SETTINGS, RunFolder, fixed_settings

ROOT = Path(__file__).parents[1]
MESHES = Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

TINY = f"""# a tiny bot for the dashboard's tests
[paths]
collision_meshes = "{MESHES.as_posix()}"

[run]
device = "cpu"
seed = 3
checkpoint_minutes = 60   # saves come from Stop, not the clock

[env]
num_arenas = 2
threads = 2

[terminals]
no_touch_seconds = 2

[model]
policy = [16]
value = [16]

[ppo]
steps_per_iteration = 160
minibatches = 1
policy_lr = 3e-4  # the learning rate

[[state_setters]]
name = "kickoff"
"""


def make_bot(runs: Path, name: str, text: str = TINY, saves: int = 0) -> RunFolder:
    settings = runs.parent / f"{name}.toml"
    settings.write_text(text, encoding="utf-8")
    folder = RunFolder.create(runs, name, settings)
    for k in range(saves):
        save = folder.checkpoints / f"{(k + 1) * 1000:012d}"
        save.mkdir()
        (save / "state.pt").write_bytes(b"")
    return folder


def items(sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section in sections:
        out.update({item["path"]: item for item in section["items"]})
        out.update(items(section["sections"]))
    return out


def test_every_setting_is_described_with_its_help_and_the_networks_settings_lock_once_saved() -> None:
    config = parse_toml(TINY, NitroConfig, file=Path("tiny.toml"), base_dir=ROOT)
    found = items(describe.describe(config, TINY, locked=False))
    expected = set()
    for section in dataclasses.fields(NitroConfig):
        value = getattr(config, section.name)
        if dataclasses.is_dataclass(value):
            for f in dataclasses.fields(value):
                child = getattr(value, f.name)
                if not dataclasses.is_dataclass(child):
                    expected.add(f"{section.name}.{f.name}")
    expected -= set(layout.NOT_SHOWN)
    assert expected <= set(found), sorted(expected - set(found))
    assert not set(layout.NOT_SHOWN) & set(found), "a setting left out of the editor is not described"
    lr = found["ppo.policy_lr"]
    assert lr["label"] == "Policy learning rate" and lr["kind"] == "float" and lr["value"] == 3e-4
    assert lr["help"] and lr["written"] and not lr["fixed"]
    assert found["model.encoder"]["kind"] == "choice" and "attention" in found["model.encoder"]["options"]
    assert found["model.policy"]["kind"] == "numbers" and found["model.policy"]["value"] == [16]
    assert found["run.deterministic"]["kind"] == "bool" and not found["run.deterministic"]["written"]
    assert not any(item["fixed"] for item in found.values()), "a bot without saves can still change anything"
    locked = items(describe.describe(config, TINY, locked=True))
    assert {p for p, item in locked.items() if item["fixed"]} == set(FIXED_SETTINGS) & set(locked)
    json.dumps(describe.describe(config, TINY, locked=True), allow_nan=False)  # the window can carry it


def test_values_from_the_window_are_checked_and_written_with_the_files_comments_kept(tmp_path: Path) -> None:
    config = parse_toml(TINY, NitroConfig, file=tmp_path / "t.toml", base_dir=ROOT)
    found = items(describe.describe(config, TINY, locked=False))
    assert edits.convert(found["ppo.policy_lr"], "2e-4") == 2e-4
    assert edits.convert(found["env.tick_skip"], "4") == 4
    assert edits.convert(found["model.policy"], "512, 512") == [512, 512]
    assert edits.convert(found["model.encoder"], "attention") == "attention"
    assert edits.convert(found["run.deterministic"], True) is True
    for item, bad in (("env.tick_skip", "abc"), ("env.tick_skip", "4.5"), ("model.encoder", "big")):
        with pytest.raises(edits.EditError):
            edits.convert(found[item], bad)
    with pytest.raises(edits.EditError, match="smallest"):
        edits.convert(found["env.num_arenas"], "0")
    text = edits.written(TINY, {"ppo.policy_lr": 2e-4, "run.deterministic": True, "env.threads": None})
    assert "# a tiny bot for the dashboard's tests" in text and "# saves come from Stop" in text
    assert "policy_lr = 0.0002" in text and "threads" not in text
    written = parse_toml(text, NitroConfig, file=tmp_path / "t.toml", base_dir=ROOT)
    assert written.ppo.policy_lr == 2e-4 and written.run.deterministic is True
    assert written.env.threads == NitroConfig().env.threads, "a removed line goes back to the default"
    assert edits.written(TINY, {"selfplay.enabled": True}).rstrip().endswith("[selfplay]\nenabled = true")
    assert edits.problems(
        edits.written(TINY, {"env.tick_skip": 1}), tmp_path / "t.toml", ROOT
    )  # delay 1 >= 1


def test_changes_wait_for_apply_and_refuse_what_the_bot_cannot_take(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "saved", saves=2)
    api = Api(ROOT, runs)
    before = folder.settings.read_text(encoding="utf-8")
    assert api.stage("saved", "ppo.policy_lr", "0.0002") == {"pending": {"ppo.policy_lr": 0.0002}}
    refused = api.stage("saved", "model.policy", "64, 64")
    assert "networks were made with this" in refused["error"], "fixed once the bot has saves"
    assert "whole number" in api.stage("saved", "env.tick_skip", "abc")["error"]
    assert "smaller than tick_skip" in api.stage("saved", "env.tick_skip", "1")["error"], (
        "the whole file is checked (the delay of 1 must be below the tick skip)"
    )
    assert api.stage("saved", "no.such", 1)["error"]
    assert folder.settings.read_text(encoding="utf-8") == before, "nothing written before Apply"
    assert api.home()["bots"][0]["pending_count"] == 1
    assert api.unstage("saved", "ppo.policy_lr") == {"pending": {}}
    api.stage("saved", "ppo.policy_lr", "0.0002")
    api.stage("saved", "run.deterministic", True)
    assert api.apply("saved") == {"applied": 2, "restarting": False}
    after = folder.settings.read_text(encoding="utf-8")
    assert "policy_lr = 0.0002  # the learning rate" in after, "the line's own comment stays"
    assert "# saves come from Stop, not the clock" in after
    assert folder.load_settings().run.deterministic is True and api.bot("saved")["pending"] == {}
    api.stage("saved", "ppo.policy_lr", "0.0005")
    assert api.discard("saved") == {"pending": {}} and api.apply("saved") == {"applied": 0}


def test_notes_preferences_new_bots_and_copies(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_bot(runs, "first", saves=1)
    api = Api(ROOT, runs)
    assert api.set_note("first", "setting:ppo.policy_lr", "lowered at 600M")["notes"] == {
        "setting:ppo.policy_lr": "lowered at 600M"
    }
    assert api.bot("first")["notes"]["setting:ppo.policy_lr"] == "lowered at 600M"
    assert api.set_note("first", "setting:ppo.policy_lr", "  ")["notes"] == {}, "an empty note is removed"
    assert api.home()["prefs"] == {"theme": "dark"}, "dark by default"
    api.set_pref("theme", "light")
    assert Api(ROOT, runs).home()["prefs"]["theme"] == "light"
    assert [b["name"] for b in api.home()["bots"]] == ["first"], "the preferences folder is not a bot"
    assert api.home()["rlbot_ready"] is False, "no RLBot chosen yet, none installed"
    older = tmp_path / "RLBotGUIX" / "RLBot.exe"
    v5 = tmp_path / "RLBot5" / "bin" / "launcher.exe"
    for program in (older, v5):
        program.parent.mkdir(parents=True)
        program.touch()
    api.set_pref("rlbot_program", str(older))
    assert api.home()["rlbot_ready"] is False, "an older RLBot does not count as chosen"
    api.set_pref("rlbot_program", str(v5))
    assert api.home()["rlbot_ready"] is True
    v5.unlink()
    assert api.home()["rlbot_ready"] is False, "a program that is gone is asked for again"
    api.set_pref("rlbot_program", "")
    assert api.copy_bot("first", "second", True) == {"made": "second", "notes": []}
    second = RunFolder(runs / "second")
    info = json.loads(second.info.read_text(encoding="utf-8"))
    assert "start_from" not in info, "the transfer is in its settings file, not hidden in run.json"
    assert info["settings_dir"] == json.loads((runs / "first" / "run.json").read_text())["settings_dir"]
    assert second.load_settings().start_from.source == "first" and not second.complete_checkpoints()
    assert api.copy_bot("first", "plain", False) == {"made": "plain", "notes": []}
    assert "start_from" not in json.loads((runs / "plain" / "run.json").read_text())
    assert RunFolder(runs / "plain").load_settings().start_from.source == "", "from scratch"
    for bad in ("second", "../escape", "", "has space"):
        assert api.copy_bot("first", bad, True)["error"]
    assert api.bot("../first")["error"] and api.bot("nobody")["error"]
    assert api.new_bot("third", "no_such_preset")["error"]


def test_a_bot_with_broken_settings_is_listed_with_its_problem_not_a_crash(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "broken")
    folder.settings.write_text(TINY.replace("threads = 2", "threads = -5"), encoding="utf-8")
    api = Api(ROOT, runs)
    listed = api.home()["bots"][0]
    assert listed["status"] == "problem" and "threads" in listed["problem"]
    shown = api.bot("broken")
    assert shown["settings"] == [] and "threads" in shown["problem"]


def test_running_bots_are_found_by_their_marker_and_a_live_process(tmp_path: Path) -> None:
    folder = make_bot(tmp_path / "runs", "a")
    assert bots.running(folder) is None
    (folder.root / bots.RUNNING).write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    assert bots.running(folder) is not None and bots.summary(folder)["status"] == "training"
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (folder.root / bots.RUNNING).write_text(json.dumps({"pid": gone.pid, "started": time.time()}))
    assert bots.running(folder) is None, "a marker a crash left behind"
    assert not bots.pid_alive(0) and bots.pid_alive(os.getpid())


def test_charts_and_console_are_read_back(tmp_path: Path) -> None:
    folder = make_bot(tmp_path / "runs", "a")
    rows = "\n".join(f"{i * 100},{i},{1000 + i},{0.01 * i}" for i in range(1, 1001))
    folder.metrics.write_text("agent_steps,iteration,steps_per_second,reward_per_step\n" + rows + "\n")
    shown = charts.view(folder, 0, None, points=100)
    assert set(shown["series"]) == {"steps_per_second", "reward_per_step"}
    points = shown["series"]["reward_per_step"]
    assert len(points) == 100 and points[-1] == [100_000, 10.0], "at most 100 points, the newest as it is"
    assert points[0] == [600, pytest.approx(0.06)], "the first stretch (rows 1 to 11): the mean of its rows"
    assert bots.newest(folder)["agent_steps"] == 100_000
    # Bytes, so the offsets are the same on Windows (whose text files end lines with \r\n).
    (folder.root / CONSOLE).write_bytes(b"one\ntwo\n")
    text, end = read_console(folder.root, 0)
    assert text == "one\ntwo\n" and end == 8
    with (folder.root / CONSOLE).open("ab") as f:
        f.write(b"three\r\n")
    assert read_console(folder.root, end) == ("three\n", 15), "Windows' line ends read as plain ones"
    (folder.root / CONSOLE).write_bytes(b"new\n")
    assert read_console(folder.root, 15) == ("new\n", 4), "a file made anew is read from its start"


def test_watch_plays_only_the_game_sizes_the_bot_can_read(tmp_path: Path) -> None:
    from argparse import Namespace

    from nitro.cli import _watch_settings
    from nitro.config import ConfigError

    folder = make_bot(tmp_path / "runs", "plain", TINY.replace("[env]\n", "[env]\nteam_size = 2\n"))
    assert _watch_settings(folder, saved(folder), Namespace(team_size=2)).env.team_size == 2
    with pytest.raises(ConfigError, match="reads 2v2 games only"):
        _watch_settings(folder, saved(folder), Namespace(team_size=1))
    padded = make_bot(
        tmp_path / "runs", "padded", TINY.replace("[env]\n", "[env]\nteam_size = 2\nobs_team_size = 3\n")
    )
    assert _watch_settings(padded, saved(padded), Namespace(team_size=3)).env.team_size == 3
    assert bots.summary(padded)["sizes"] == [1, 2, 3] and bots.summary(folder)["sizes"] == [2]


def saved(folder: RunFolder) -> dict[str, Any]:
    """What a save of `folder` remembers of its settings (enough for `load_settings` to check them)."""
    return {"settings": fixed_settings(folder.load_settings())}


def wait_for(check: Any, seconds: float, what: str) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if check():
            return
        time.sleep(0.3)
    raise AssertionError(f"waited {seconds:.0f} s for {what}")


@NEEDS_MESHES
def test_a_bot_is_started_stopped_with_a_save_and_restarted_by_apply(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "live")
    api = Api(ROOT, runs)
    try:
        assert api.start("live") == {"started": True}
        wait_for(lambda: bots.running(folder) is not None, 120, "training to start")
        wait_for(
            lambda: folder.metrics.is_file() and bots.newest(folder).get("iteration", 0) >= 2,
            180,
            "iterations",
        )
        assert api.home()["bots"][0]["status"] == "training"
        running = json.loads((folder.root / bots.RUNNING).read_text())
        assert running["job"] == api._jobs.get("live", "train").token, "its start is known by its token"
        assert api.start("live")["error"], "never two trainings of one bot"
        assert api.stop("live") == {"stopping": True}
        wait_for(lambda: bots.running(folder) is None, 180, "the save and stop")
        assert folder.complete_checkpoints(), "Stop saves first, as Ctrl+C once does"
        assert "Stop asked for from the dashboard" in read_console(folder.root, 0)[0]
        saved_steps = int(folder.complete_checkpoints()[-1].name)

        assert api.start("live") == {"started": True}, "the stopped trainer's process may still be ending"
        wait_for(lambda: bots.running(folder) is not None, 120, "the second start")
        wait_for(lambda: bots.newest(folder).get("agent_steps", 0) > saved_steps, 180, "training on")
        assert api.stage("live", "ppo.policy_lr", "0.0002") == {"pending": {"ppo.policy_lr": 0.0002}}
        assert api.apply("live") == {"applied": 1, "restarting": True}
        first_pid = json.loads((folder.root / bots.RUNNING).read_text())["pid"]
        wait_for(
            lambda: (info := bots.running(folder)) is not None and info["pid"] != first_pid,
            300,
            "the restart with the change",
        )
        assert folder.load_settings().ppo.policy_lr == 0.0002
        assert len(folder.complete_checkpoints()) >= 2, "it saved before the restart"
        assert "Starting again with the changes" in read_console(folder.root, 0)[0], "the trainer did it"
        assert json.loads((folder.root / bots.APPLY_RESTART).read_text())["restarted"], "and said so"

        # Stop right after Apply: it saves and stops, and does not start again.
        wait_for(lambda: bots.newest(folder).get("iteration", 0) >= 1, 180, "training after the restart")
        starts = read_console(folder.root, 0)[0].count("$ nitro resume")
        assert api.stage("live", "ppo.policy_lr", "0.0003")["pending"]
        assert api.apply("live") == {"applied": 1, "restarting": True}
        assert api.stop("live") == {"stopping": True}
        wait_for(lambda: bots.running(folder) is None, 180, "the stop")
        time.sleep(6)
        assert bots.running(folder) is None
        assert read_console(folder.root, 0)[0].count("$ nitro resume") == starts, "not started again"
        assert api.home()["bots"][0]["status"] == "stopped" and "applying" not in api.home()["bots"][0]
    finally:
        if bots.running(folder) is not None:
            api.stop("live")
            wait_for(lambda: bots.running(folder) is None, 180, "the last stop")


def test_the_page_runs_on_the_python_side_with_no_errors(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    make_bot(runs, "shown", saves=1)
    api = Api(ROOT, runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page()
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click(".botcard")
            tab.click("[data-tab=settings]")
            tab.fill("#f-ppo-policy_lr", "0.0002")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("#pending:not([hidden])")
            assert api.bot("shown")["pending"] == {"ppo.policy_lr": 0.0002}, (
                "the page staged it, nothing more"
            )
            tab.fill("#f-env-tick_skip", "abc")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("text=is not a whole number")
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
            assert RunFolder(runs / "shown").load_settings().ppo.policy_lr == 0.0002
        finally:
            browser.close()
    assert errors == []


def test_no_window_call_has_a_name_the_window_keeps_for_itself() -> None:
    """pywebview's WebView2 window takes a call named `console` as its own console forwarding (and its
    `_pywebview...` names): a method of that name never got its answer on Windows."""
    public = [name for name in dir(Api) if not name.startswith("_") and callable(getattr(Api, name))]
    assert "console_text" in public and "console" not in public
    page_calls = page_source()
    assert 'call("console"' not in page_calls and "pywebview" not in " ".join(public).lower()


def test_popups_stay_open_on_clicks_inside_and_closed_once_closed(tmp_path: Path) -> None:
    """Clicking anything in a popup once closed it (the backdrop took every click inside). And the new-bot
    form, closed before its settings list came back, opened again. Real clicks here, not fill(), which sends
    none. (What is typed in the new-bot wizard outliving its redraws: tests/test_dashboard_home.py.)"""
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    make_bot(runs, "shown", saves=1)
    api = Api(ROOT, runs)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1300, "height": 850})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            # With __slow set, the settings list answers a second late, as the bridge can on Windows.
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => m === 'settings_sources' && "
                "window.__slow ? new Promise(r => setTimeout(r, 1000)).then(() => window.__py(m, a)) : "
                "window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click("#band [data-go=rlbot]")
            tab.wait_for_selector(".modal")
            for target in (".modal h2", ".modal .banner", ".modal b >> nth=0"):
                tab.click(target)
                assert tab.is_visible(".modal"), f"a click on {target} closed the popup"
            box = tab.locator(".modal .banner").bounding_box()
            assert box is not None
            tab.mouse.move(box["x"] + 5, box["y"] + 5)
            tab.mouse.down()
            tab.mouse.move(5, 5)
            tab.mouse.up()
            assert tab.is_visible(".modal"), "a drag from inside the popup to the backdrop is no click on it"
            tab.mouse.click(5, 5)
            tab.wait_for_selector(".modal", state="detached")

            tab.evaluate("window.__slow = true")
            tab.click("[data-act=newbot] >> nth=0")
            tab.click("[data-act=wiz-cancel]")
            tab.wait_for_selector(".botcard")
            tab.wait_for_timeout(1600)
            assert tab.is_visible(".botcard") and not tab.is_visible("#wizard"), (
                "a new-bot form left before its list came back stays closed"
            )
        finally:
            browser.close()
    assert errors == []


def test_one_trainer_a_bot_and_a_start_under_way_refuses_another(tmp_path: Path) -> None:
    """A double click on Start ran two trainers of one bot; their saves deleted each other's."""
    folder = make_bot(tmp_path / "runs", "solo")
    me = {"pid": os.getpid(), "started": time.time()}
    assert bots.claim(folder, me) and not bots.claim(folder, me), "the second is refused"
    (folder.root / bots.RUNNING).write_text(json.dumps({"pid": 2**22 + 7, "started": 0}), encoding="utf-8")
    old = time.time() - 60
    os.utime(folder.root / bots.RUNNING, (old, old))
    assert bots.claim(folder, me), "a file left by a trainer that ended is taken over"
    (folder.root / bots.RUNNING).write_text(json.dumps({"pid": 2**22 + 7, "started": 0}), encoding="utf-8")
    assert bots.claim(folder, me), "also at once (a trainer killed in its first seconds)"
    (folder.root / bots.RUNNING).write_text("", encoding="utf-8")
    assert not bots.claim(folder, me), "a file being written by a trainer starting this moment"
    (folder.root / bots.RUNNING).unlink()

    api = Api(ROOT, tmp_path / "runs")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        from nitro.dashboard.jobs import Job

        api._jobs._jobs[("solo", "train")] = Job(
            "solo", "train", [], folder.root / CONSOLE, sleeper, token="tok"
        )
        assert "is starting already" in api.start("solo")["error"], "a second click while it loads"
        assert api.home()["bots"][0]["status"] == "starting"
        # On Windows the trainer is a child of the started process (the venv's python.exe starts the real
        # one): its running.json has another pid, and the start's token tells it is this job's.
        marker = folder.root / bots.RUNNING
        marker.write_text(
            json.dumps({"pid": os.getpid(), "started": time.time(), "job": "tok"}), encoding="utf-8"
        )
        assert "already training" in api.start("solo")["error"]
        marker.unlink()  # stopped: saved and gone, its process only ending
        assert api._busy("solo") == "", "a Start right after a Stop is not taken for one under way"
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_live_game_stopped_from_the_window_is_not_reported_as_failed(tmp_path: Path) -> None:
    """Stop the live game ended its process with a non-zero code, so the page said it did not
    work. One stopped from the window is done; one that ends by itself with an error still failed."""
    from nitro.dashboard.jobs import Job

    folder = make_bot(tmp_path / "runs", "lively")
    api = Api(ROOT, tmp_path / "runs")
    for stopped_here in (True, False):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        api._jobs._jobs[("lively", "live")] = Job("lively", "live", [], folder.root / CONSOLE, sleeper)
        if stopped_here:
            assert api.stop_job("lively", "live") == {"stopped": True}
        else:
            sleeper.terminate()
        sleeper.wait()
        assert api.job("lively", "live")["state"] == ("done" if stopped_here else "failed")


def test_the_dashboard_does_not_load_pytorch() -> None:
    """The dashboard's process held 1.4 GB, most of it PyTorch's (with CUDA), which it never
    needs to show bots; a checkpoint is read or written by the trainer."""
    code = "import sys, nitro.dashboard.api, nitro.dashboard.app; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_waiting_changes_outlive_the_window(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_bot(runs, "kept")
    api = Api(ROOT, runs)
    assert api.stage("kept", "ppo.policy_lr", "0.0002") == {"pending": {"ppo.policy_lr": 0.0002}}
    again = Api(ROOT, runs)  # the window closed and opened again
    assert again.bot("kept")["pending"] == {"ppo.policy_lr": 0.0002}
    assert again.home()["bots"][0]["pending_count"] == 1
    again.discard("kept")
    assert Api(ROOT, runs).bot("kept")["pending"] == {}


def test_values_the_window_cannot_take_are_refused_plainly(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_bot(runs, "strict")
    api = Api(ROOT, runs)
    assert "it needs a value" in api.stage("strict", "model.policy", "")["error"], "no line silently dropped"
    assert "without commas" in api.stage("strict", "model.policy", "1,024, 1,024")["error"]
    assert api.stage("strict", "model.policy", "64,64")["pending"]["model.policy"] == [64, 64]
    for bad in ("inf", "1e999", "nan"):
        assert "finite" in api.stage("strict", "ppo.epochs", bad)["error"], bad
    item = {"kind": "path", "nullable": True}
    assert edits.convert(item, '"D:\\\\My replays"') == "D:\\\\My replays", "Copy as path's quotes go"
    from nitro.dashboard.api import _finite

    assert _finite({"a": [float("inf"), 1.5], "b": float("nan")}) == {"a": [None, 1.5], "b": None}


def test_metrics_are_read_from_where_the_last_read_stopped(tmp_path: Path) -> None:
    """Every refresh read the whole metrics.csv (hundreds of MB after a day: seconds and GB)."""
    folder = make_bot(tmp_path / "runs", "long")
    header = "agent_steps,iteration,steps_per_second,reward_per_step,paid_abs_touch,paid_touch,raw_touch,x\n"
    rows = "100,1,10,0.5,0.1,0.1,1,x\n200,2,10,0.6,0.1,0.1,1,y\n"
    folder.metrics.write_text(header + rows, encoding="utf-8")
    first = bots.metrics(folder)
    assert first.count == 2 and "x" not in first.columns and bots.newest(folder)["reward_per_step"] == 0.6
    read = first.offset
    with folder.metrics.open("a", encoding="utf-8") as out:
        out.write("300,3,10,0.7,0.2,-0.2,1,z\n400,4,1")  # the last line still being written
    grown = bots.metrics(folder)
    assert grown is first and grown.count == 3 and grown.offset > read, "only the new whole line was read"
    assert [p[1] for p in charts.view(folder, 0, ["reward_per_step"])["series"]["reward_per_step"]] == [
        0.5,
        0.6,
        0.7,
    ]
    shares = dashboard_rewards.shares(folder, 15.0)
    assert shares["rewards"][0]["label"] == "touch" and shares["agent_steps"] == 200
    folder.metrics.unlink()
    folder.metrics.write_text(header + "100,1,10,0.9,0.1,0.1,1,x\n", encoding="utf-8")  # anew on a resume
    assert bots.metrics(folder).count == 1 and bots.newest(folder)["reward_per_step"] == 0.9


def test_the_page_keeps_what_is_typed_and_says_what_is_wrong(tmp_path: Path) -> None:
    """A note being written was erased by the 5-second refresh; a bot whose settings file has a
    mistake showed an empty Settings tab; waiting changes on another bot were invisible."""
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    make_bot(runs, "shown", saves=1)
    make_bot(runs, "broken", TINY.replace("seed = 3", 'seed = "three"'))
    api = Api(ROOT, runs)
    api.stage("shown", "ppo.policy_lr", "0.0002")
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1300, "height": 850})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.wait_for_selector("#main [data-bot] >> text=1 waiting")
            assert tab.evaluate("pct(0.997)") == "99.7%" and tab.evaluate("pct(0.5)") == "50%"
            open_bot(tab, "shown")
            tab.wait_for_selector("body.has-pending"), "the toasts move up off the Apply bar"
            tab.click("[data-tab=notes]")
            tab.click("#botnote")
            tab.keyboard.type("half a thought")
            tab.click("h1")  # the box loses the focus, as when the window refreshes
            tab.evaluate("refreshBot()")
            tab.click("[data-tab=overview]")
            tab.click("[data-tab=notes]")
            assert tab.input_value("#botnote") == "half a thought", "a note being written outlives redraws"
            open_bot(tab, "broken")
            tab.click("[data-tab=settings]")
            tab.wait_for_selector("text=The settings file has a mistake")
            assert tab.is_visible("#tab [data-act=open-settings]")
        finally:
            browser.close()
    assert errors == []


def test_a_start_again_after_apply_that_fails_is_shown_and_does_not_lock_start(tmp_path: Path) -> None:
    """'restarting' could stay forever with Start greyed out when the start-again did not happen."""
    runs = tmp_path / "runs"
    folder = make_bot(runs, "b")
    api = Api(ROOT, runs)
    api._applying["b"] = {"text": "Saving, then starting again with the changes", "at": time.time() - 5}
    assert api.home()["bots"][0]["applying"] and api._busy("b"), "while it saves"
    outcome = {"at": time.time(), "restarted": False, "why": "it stopped with a problem"}
    (folder.root / bots.APPLY_RESTART).write_text(json.dumps(outcome))
    bot = api.home()["bots"][0]
    assert "did not start again: it stopped with a problem" in bot["restart_problem"]
    assert "applying" not in bot and api._busy("b") == "", "Start can be pressed again"
    # It started again, but that start ended before training: told once the grace time is over.
    api._applying["b"] = {"text": "Saving, then starting again with the changes", "at": time.time() - 5}
    outcome = {"at": time.time() - dashboard_api.RESTART_GRACE - 1, "restarted": True}
    api._applying["b"]["at"] = outcome["at"] - 5
    (folder.root / bots.APPLY_RESTART).write_text(json.dumps(outcome))
    assert "that start ended before training" in api.home()["bots"][0]["restart_problem"]


def test_a_trainer_started_by_the_one_before_waits_for_it_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new trainer must not load the graphics card while the old one still holds its memory."""
    from nitro import cli
    from nitro.dashboard.jobs import AFTER_PID

    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    monkeypatch.setenv(AFTER_PID, str(old.pid))
    began = time.monotonic()
    cli._wait_for_previous_trainer()
    assert time.monotonic() - began >= 1.0 and old.poll() is not None
    assert AFTER_PID not in os.environ, "only the first start waits"


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc (Linux)")
def test_an_ended_process_its_parent_has_not_collected_is_not_running() -> None:
    """On Linux an ended child stays a zombie until collected; it must not count as running."""
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 10
    while time.time() < deadline and Path(f"/proc/{ended.pid}/stat").read_text().split()[2] != "Z":
        time.sleep(0.05)
    assert not bots.pid_alive(ended.pid), "ended, not yet collected"
    ended.wait()
