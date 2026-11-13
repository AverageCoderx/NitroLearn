"""Watching a bot play inside the dashboard's window: the viewer's files through the bridge and never in the
page, the recorded games, the live game's own process, and the page itself (the dialog, the viewer over the
window, closing it), with nothing listening on any port."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from browser import playwright_or_skip
from nitro import viewer
from nitro.dashboard import app, watching
from nitro.dashboard.api import Api
from nitro.dashboard.jobs import CONSOLE, Job
from nitro.runs import RunFolder
from nitro.viewer import recording as layouts
from test_dashboard import MESHES, NEEDS_MESHES, ROOT, TINY, make_bot, wait_for
from test_watch import chromium, fake, save

PADDED = TINY.replace("[env]\n", "[env]\nteam_size = 2\nobs_team_size = 3\n")
STARTED = "window.NV && NV.frames > 2"


def whole(api: Api, method: str, *args: Any) -> str:
    """A text the page asks for piece by piece, put together."""
    first = getattr(api, method)(*args, 0)
    assert "error" not in first, first
    rest = [getattr(api, method)(*args, piece) for piece in range(1, first["pieces"])]
    assert all(0 < len(piece["text"]) <= watching.PIECE for piece in [first, *rest][: first["pieces"]])
    return "".join(piece["text"] for piece in [first, *rest])


def playing_bot(tmp_path: Path, *steps: int, env: str = "") -> RunFolder:
    """A bot with a small network of random weights saved at each of `steps`: it really plays."""
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        f"[env]\nnum_arenas = 2\nthreads = 2\n{env}[model]\npolicy = [16]\nvalue = [16]\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    for agent_steps in steps:
        save(folder, agent_steps)
    return folder


def sockets_of(pid: int) -> list[str]:
    """The sockets a process holds open (Linux)."""
    found = []
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        with contextlib.suppress(OSError):
            target = os.readlink(descriptor)
            if target.startswith("socket:"):
                found.append(target)
    return found


def files_open_under(folder: Path, pid: int | None = None) -> list[str] | None:
    """The files under `folder` a process (this one by default) holds open, or None where that cannot be
    seen (psutil missing and no /proc)."""
    root = folder.resolve()
    try:
        import psutil

        paths = [Path(item.path) for item in psutil.Process(pid).open_files()]
    except ImportError:
        descriptors = Path(f"/proc/{pid or os.getpid()}/fd")
        if not descriptors.is_dir():
            return None
        paths = []
        for descriptor in descriptors.iterdir():
            with contextlib.suppress(OSError):
                paths.append(Path(os.readlink(descriptor)))
    return [str(path) for path in paths if path.is_relative_to(root)]


def ended_and_let_go(live: watching.Live, folder: Path) -> None:
    """The live game's process has ended, every pipe to it is closed, its threads are done, and the dashboard
    holds no file in the bot's folder (on Windows such a file would keep the folder from being removed)."""
    assert live.process.poll() is not None, "stopping waits for the process to end"
    assert all(pipe is None or pipe.closed for pipe in (live.process.stdin, live.process.stdout))
    assert live.process.stderr is None or live.process.stderr.closed
    assert live.reader.wait(0) and live.reader.closed, "the thread reading its lines is done"
    assert live.notices is None or not live.notices.is_alive(), "so is the one copying its notices"
    assert files_open_under(folder) in ([], None), "nothing in the bot's folder is held open"


def no_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """From here on, a socket of this process that starts to listen fails the test."""

    def listen(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("something started to listen on a port")

    monkeypatch.setattr(socket.socket, "listen", listen)


class Bridge:
    """The page's calls, passed to the real Api and noted."""

    def __init__(self, api: Api) -> None:
        self.api = api
        self.calls: list[str] = []

    def __call__(self, method: str, args: list[Any]) -> Any:
        self.calls.append(method)
        return getattr(self.api, method)(*args)


def opened(browser: Any, bridge: Bridge, problems: list[str], text: str | None = None) -> Any:
    """The dashboard's page in a new tab, talking to the bridge's Api."""
    tab = browser.new_page(viewport={"width": 900, "height": 640})
    tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
    tab.on("console", lambda said: problems.append(said.text) if said.type == "error" else None)
    tab.expose_function("__py", bridge)
    tab.add_init_script("window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});")
    tab.goto("about:blank")
    tab.set_content(app.page(bridge.api) if text is None else text)
    return tab


def pumped(tab: Any, check: Any, seconds: float, what: str) -> None:
    """Waits for `check` while the page's calls are answered (a plain sleep would leave them waiting)."""
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, f"waited {seconds:.0f} s for {what}"
        tab.wait_for_timeout(100)


def freed(tab: Any, name: str) -> bool:
    """Whether what the page's weak reference `window[name]` pointed at has gone from memory. The browser's
    own tools collect the garbage: a collection asked for by the page's script keeps what that script's stack
    still seems to hold."""
    tools = tab.context.new_cdp_session(tab)
    try:
        for _ in range(20):
            tools.send("HeapProfiler.collectGarbage")
            if tab.evaluate(f"window.{name}.deref() === undefined"):
                return True
            tab.wait_for_timeout(250)
        return False
    finally:
        tools.detach()


def test_the_viewers_files_cross_the_bridge_in_pieces_and_are_never_in_the_page(tmp_path: Path) -> None:
    (tmp_path / "collision_meshes" / "soccar").mkdir(parents=True)
    for k in range(3):
        (tmp_path / "collision_meshes" / "soccar" / f"mesh_{k}.cmf").write_bytes(bytes([k]) * 200_000)
    nowhere = TINY.replace(MESHES.as_posix(), (tmp_path / "no-meshes").as_posix())
    make_bot(tmp_path / "runs", "shown", nowhere, saves=1)
    (tmp_path / "own" / "soccar").mkdir(parents=True)
    (tmp_path / "own" / "soccar" / "mesh_0.cmf").write_bytes(b"its own")
    make_bot(tmp_path / "runs", "elsewhere", TINY.replace(MESHES.as_posix(), (tmp_path / "own").as_posix()))
    watching._arenas.clear()
    api = Api(tmp_path, tmp_path / "runs")
    page = app.page(api)
    # The window (WebView2) takes a page of at most 2 MB, and holds it at two bytes a character.
    assert len(page) < 1_000_000 and len(page.encode()) < 1_200_000
    assert "var THREE=" not in page and "three.js 0." not in page, "three.js is not in the page"
    assert "const NV = {" not in page and "NV_MAKE() {" not in page, "nor is the viewer's script"
    assert "nv-scoreboard" not in page and "data-nv=" not in page, "nor its markup and style"

    assert whole(api, "viewer_assets", "three") == viewer.three()
    assert api.viewer_assets("three")["pieces"] >= 3, "three.js alone takes several calls"
    script = whole(api, "viewer_assets", "script")
    assert script == viewer.maker() and script.startswith('function NV_MAKE() {\n"use strict";\n')
    assert script.endswith("\nreturn NV;\n}") and viewer.script() in script
    style = whole(api, "viewer_assets", "style")
    assert style == viewer.css() and "@font-face" not in style, "the page holds the fonts already"
    assert "@font-face" in page and viewer.style().endswith(style)
    assert whole(api, "viewer_assets", "markup") == viewer.markup()
    assert "no part called" in api.viewer_assets("three.min.js")["error"]
    assert api.viewer_assets("three", 99) == {"text": "", "pieces": api.viewer_assets("three")["pieces"]}

    blob = viewer.arena_blob(tmp_path / "collision_meshes")
    assert blob is not None and len(blob) > 2 * watching.PIECE
    assert whole(api, "viewer_arena", "") == blob, "the project's meshes, in pieces"
    assert whole(api, "viewer_arena", "shown") == blob, "a bot whose own meshes folder holds none"
    assert whole(api, "viewer_arena", "elsewhere") == viewer.arena_blob(tmp_path / "own") != blob
    assert whole(api, "viewer_arena", "no-such-bot") == blob
    without = Api(tmp_path / "elsewhere", tmp_path / "runs")
    watching._arenas.clear()
    assert without.viewer_arena("") == {"text": "", "pieces": 0}, "no meshes: a generated arena"

    assert api.viewer_prefs() == {"prefs": {}}
    api.set_pref("theme", "light")
    assert api.set_viewer_prefs({"camera": "ball", "speed": 2}) == {"saved": True}
    assert Api(tmp_path, tmp_path / "runs").viewer_prefs() == {"prefs": {"camera": "ball", "speed": 2}}
    assert api.home()["prefs"]["theme"] == "light", "kept with the dashboard's own, which stay"
    assert "error" in api.set_viewer_prefs("fast") and "error" in api.set_viewer_prefs({"x": "y" * 20_000})


def test_recorded_games_are_listed_kept_before_a_new_one_and_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two = TINY.replace("[env]\n", "[env]\nteam_size = 2\n")
    folder = make_bot(tmp_path / "runs", "shown", two, saves=3).root
    api = Api(ROOT, tmp_path / "runs")
    started: list[list[str]] = []
    monkeypatch.setattr(api._jobs, "start", lambda bot, kind, args, where: started.append(args))
    assert api.recordings("shown") == {"recordings": [], "page": ""}
    assert api.saves("shown")["saves"] == [
        {"steps": steps, "time": pytest.approx(time.time(), abs=60)} for steps in (3000, 2000, 1000)
    ]
    assert "not there any more" in api.recording("shown")["error"]

    last = folder / "watch.nlrec"
    for game in range(1, 8):
        data = fake(frames=20 * game, cars=4).data(f"shown at {game * 1000:,} agent steps", "made up")
        layouts.save(last, data)
        os.utime(last, (1_700_000_000 + game * 60, 1_700_000_000 + game * 60))
        listed = api.recordings("shown")["recordings"]
        assert listed[0] == {
            "id": "last",
            "title": f"shown at {game * 1000:,} agent steps",
            "steps": game * 1000,
            "size": 2,
            "seconds": 0,
            "time": 1_700_000_000 + game * 60,
            "bytes": last.stat().st_size,
        }
        assert [note["steps"] for note in listed] == [k * 1000 for k in range(game, max(0, game - 5), -1)]
        assert json.loads(whole(api, "recording", "shown", "last")) == data
        assert api.watch_record("shown", {"seconds": 30}) == {"started": True}
        assert api.watch_record("shown") == {"started": True}, "asked twice: the last one is kept once"
        kept = sorted(file.name for file in (folder / "recordings").iterdir())
        assert len(kept) == min(game, watching.KEEP_RECORDINGS - 1), "the newest and four before it"
        assert kept[-1].startswith(f"{game * 1000:012d}-2023") and kept[-1].endswith(".nlrec")
    older = api.recordings("shown")["recordings"][1]
    assert older["steps"] == 6000 and older["id"] == kept[-2].removesuffix(".nlrec")
    assert json.loads(whole(api, "recording", "shown", older["id"]))["title"] == "shown at 6,000 agent steps"
    for ident in ("../watch", "nothing", "", "a/b", "..", 7):
        assert "error" in api.recording("shown", ident), ident

    layouts.save(last, fake(frames=300, cars=4).data("shown at 8,000 agent steps"))
    monkeypatch.setattr(watching, "KEEP_BYTES", last.stat().st_size + 10)
    api.watch_record("shown")
    assert len(list((folder / "recordings").iterdir())) == 1, "no more than the space they may take"
    (folder / "recordings" / "notes.txt").write_text("not a recording", encoding="utf-8")
    (folder / "recordings" / "broken.nlrec").write_bytes(b"not a recording")
    assert [note["steps"] for note in api.recordings("shown")["recordings"]] == [8000], "what can be read"
    assert api.recordings("shown")["page"] == ""
    (folder / "watch.html").write_text("<html></html>", encoding="utf-8")
    assert api.recordings("shown")["page"] == str(folder / "watch.html")
    assert "error" in api.recordings("no-such-bot")

    runs = str(tmp_path / "runs")
    assert started[0][:7] == ["watch", "shown", "--runs", runs, "--no-open", "--seconds", "30"]
    assert started[0][7] == "--seed" and len(started[0]) == 9, "the newest save, the bot's own size"
    assert api.watch_record("shown", {"seconds": 120, "size": 2, "at": 2000}) == {"started": True}
    assert started[-1][5:7] == ["--seconds", "120"] and started[-1][9:] == [
        "--team-size",
        "2",
        "--at",
        "2000",
    ]
    count = len(started)
    assert "reads 2v2 games only, so it cannot play 1v1" in api.watch_record("shown", {"size": 1})["error"]
    assert "no save at 2,500 agent steps" in api.watch_record("shown", {"at": 2500})["error"]
    assert "5 to 600 seconds" in api.watch_record("shown", {"seconds": 3})["error"]
    assert "5 to 600 seconds" in api.watch_record("shown", {"seconds": 6000})["error"]
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        api._jobs._jobs[("shown", "watch")] = Job("shown", "watch", [], folder / CONSOLE, sleeper)
        assert "being recorded already" in api.watch_record("shown")["error"]
    finally:
        sleeper.kill()
        sleeper.wait()
    assert len(started) == count, "nothing was started for what was refused"


def test_the_dashboard_shows_games_without_loading_pytorch(tmp_path: Path) -> None:
    make_bot(tmp_path / "runs", "shown", saves=1)
    layouts.save(tmp_path / "runs" / "shown" / "watch.nlrec", fake().data("shown at 1,000 agent steps"))
    code = (
        "import sys; from pathlib import Path; from nitro.dashboard.api import Api\n"
        f"api = Api(Path({str(ROOT)!r}), Path({str(tmp_path / 'runs')!r}))\n"
        "for part in ('three', 'script', 'style', 'markup'): assert api.viewer_assets(part)['text']\n"
        "api.viewer_arena('shown'); api.saves('shown'); api.viewer_prefs(); api.watch_live_state()\n"
        "assert api.recordings('shown')['recordings'] and api.recording('shown')['text']\n"
        "assert api.viewer_poll(0)['closed']\n"
        "print('torch' in sys.modules, 'numpy' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"], "games are played and read by processes of their own"


class Handlers:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def __iadd__(self, handler: Any) -> Handlers:
        self.added.append(handler)
        return self


class Events:
    def __init__(self) -> None:
        self.closed = Handlers()


class Window:
    """Stands in for pywebview's window: handlers are added to its events with `+=`."""

    def __init__(self) -> None:
        self.events = Events()


@NEEDS_MESHES
def test_a_live_game_is_played_by_a_process_of_its_own_that_ends_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = playing_bot(tmp_path, 500, env="team_size = 2\nobs_team_size = 2\n")
    api = Api(ROOT, tmp_path / "runs")
    api._window = Window()
    no_listening(monkeypatch)
    assert api.watch_live_state() == {"state": "none"}
    assert api.viewer_poll(7) == {"chunks": [], "next": 7, "restart": False, "closed": True}
    assert api.watch_live_stop() == {"stopped": False}
    assert "cannot play 3v3" in api.watch_live_start("my-bot", {"size": 3})["error"]
    assert "no bot called" in api.watch_live_start("nobody")["error"] and api._live is None
    try:
        assert api.watch_live_start("my-bot", {"size": 1}) == {"started": True}
        first = api._live
        assert first is not None and first.process.poll() is None
        asked = [str(part) for part in first.process.args]  # type: ignore[union-attr]
        assert asked[3:6] == ["nitro", "watch", "my-bot"] and asked[8:10] == ["--feed", "--seed"]
        assert asked[11:] == ["--team-size", "1", "--follow"], "each new save takes over unless switched off"
        assert api.watch_live_state() == {"state": "starting", "bot": "my-bot"}
        wait_for(lambda: api.watch_live_state()["state"] != "starting", 120, "the game's first line")
        state = api.watch_live_state()
        assert state["state"] == "playing" and state["bot"] == "my-bot"
        data = state["data"]
        assert data["live"] is True and data["cars"] == 2 and data["title"] == "my-bot at 500 agent steps"
        assert "played live on this computer" in data["subtitle"] and data["frames"] == ""
        wait_for(lambda: api.viewer_poll(0)["next"] >= 20, 60, "the game to be played")
        played = api.viewer_poll(0)
        assert not played["closed"] and not played["restart"] and len(played["chunks"]) == played["next"]
        assert set(played["chunks"][-1]) == {"f", "d"} and played["chunks"][0]["d"] == []
        later = api.viewer_poll(played["next"] - 2.0)
        assert later["chunks"][:2] == played["chunks"][-2:], "the page's cursor is a JavaScript number"
        monkeypatch.setattr(watching, "POLL_CHUNKS", 5)
        capped = api.viewer_poll(3)
        assert capped["chunks"] == played["chunks"][3:8] and capped["next"] == 8 and not capped["closed"]
        monkeypatch.undo()
        no_listening(monkeypatch)
        if Path("/proc").is_dir():
            assert sockets_of(first.process.pid) == [], "the game's process has no socket at all"
        for pid in (None, first.process.pid):
            assert files_open_under(folder.root, pid) in ([], None), "no file in the bot's folder stays open"
        assert "$ nitro watch my-bot" in (folder.root / CONSOLE).read_text(encoding="utf-8")

        assert api.watch_live_start("my-bot", {"follow": False}) == {"started": True}, "a second game"
        second = api._live
        assert second is not None and second is not first
        asked = [str(part) for part in second.process.args]  # type: ignore[union-attr]
        assert asked[8:10] == ["--feed", "--seed"] and len(asked) == 11, "its own size, no following"
        ended_and_let_go(first, folder.root)
        assert first.process.returncode == 0, "it took its closed input as the sign to stop"
        wait_for(lambda: api.watch_live_state()["state"] == "playing", 120, "the second game")
        assert api.watch_live_state()["data"]["cars"] == 4, "the bot's own size when none is asked for"
        assert api.watch_live_stop("another-bot") == {"stopped": False} and api._live is second
        assert api.watch_live_stop("my-bot") == {"stopped": True} and api._live is None
        ended_and_let_go(second, folder.root)
        assert api.viewer_poll(3)["closed"] and api.watch_live_state() == {"state": "none"}

        assert api.watch_live_start("my-bot") == {"started": True}
        third = api._live
        assert third is not None and len(api._window.events.closed.added) == 1, "hooked to the window once"
        api._window.events.closed.added[0]()
        assert api._live is None, "closing the dashboard's window ends the game"
        ended_and_let_go(third, folder.root)

        assert api.watch_live_start("my-bot", {"size": 1}) == {"started": True}
        fourth = api._live
        assert fourth is not None
        wait_for(lambda: api.watch_live_state()["state"] == "playing", 120, "the fourth game")
        moved = tmp_path / "gone"
        shutil.move(str(folder.root), moved)  # the bot is deleted while its game plays
        wait_for(lambda: api.viewer_poll(0)["closed"] and api._live is None, 30, "the game to end")
        ended_and_let_go(fourth, moved)
    finally:
        api._end_live()  # waits for the game's process, so the temporary folder can be removed


def test_a_live_game_that_cannot_start_says_why(tmp_path: Path) -> None:
    folder = make_bot(tmp_path / "runs", "broken", saves=1)  # its save is an empty file
    api = Api(ROOT, tmp_path / "runs")
    assert "no save yet" in api.watch_live_start(make_bot(tmp_path / "runs", "unsaved").root.name)["error"]
    try:
        assert api.watch_live_start("broken") == {"started": True}
        wait_for(lambda: api.watch_live_state()["state"] != "starting", 120, "the game's process to give up")
        state = api.watch_live_state()
        assert state["state"] == "ended" and state["bot"] == "broken"
        said = (folder.root / CONSOLE).read_text(encoding="utf-8")
        assert state["why"].strip() and all(line in said for line in state["why"].splitlines())
        assert api.viewer_poll(0)["closed"]
        assert api.watch_live_state()["state"] == "ended", "it is still said when asked again"
        assert api.watch_live_stop() == {"stopped": True} and api.watch_live_state() == {"state": "none"}
    finally:
        api._end_live()


def test_the_dialog_offers_what_the_bot_can_play_and_follows_a_recording(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "padded", PADDED, saves=3)
    make_bot(runs, "plain", TINY.replace("[env]\n", "[env]\nteam_size = 2\n"), saves=1)
    api = Api(ROOT, runs)
    asked: list[Any] = []
    job = {"state": "running"}
    api.watch_record = lambda name, options=None: asked.append((name, options)) or {"started": True}  # type: ignore[method-assign, func-returns-value]
    api.job = lambda name, kind: dict(job)  # type: ignore[method-assign]
    api.stop_job = lambda name, kind: asked.append(("stop", name, kind)) or {"stopped": True}  # type: ignore[method-assign, func-returns-value]
    api.open_watch_page = lambda name: asked.append(("page", name)) or {"opened": name}  # type: ignore[method-assign, func-returns-value]
    api.show_watch_file = lambda name: asked.append(("file", name)) or {"shown": name}  # type: ignore[method-assign, func-returns-value]
    problems: list[str] = []
    page = app.page(api)
    assert page.count('"use strict";') == 1
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        try:
            tab = opened(browser, Bridge(api), problems)
            tab.click("#main [data-bot=padded]")
            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-act=watch-live]")
            assert tab.text_content(".modal h2") == "Watch padded play"
            sizes = tab.locator("[data-seg=watch-size] button")
            assert sizes.all_text_contents() == ["1v1", "2v2", "3v3"], (
                "the sizes its observation has room for"
            )
            assert tab.get_attribute("[data-seg=watch-size] [aria-pressed=true]", "data-v") == "2"
            assert "Its padded observation lets it play these sizes." in tab.text_content(".modal")
            said = tab.text_content(".modal")
            assert "on the processor, not the graphics card" in said and "may slow training slightly" in said
            assert (
                "each new save takes over" in said and tab.locator("[data-act=watch-recorded]").count() == 0
            )
            saves = tab.locator("#watch-at option")
            assert [option.get_attribute("value") for option in saves.all()] == ["", "2000", "1000"]
            assert saves.all_text_contents()[0].startswith("Newest: 3.0K steps")
            assert saves.all_text_contents()[2].startswith("1.0K steps")
            tab.click("[data-seg=watch-size] [data-v='3']")
            tab.click("[data-seg=watch-length] [data-v='120']")
            tab.select_option("#watch-at", "1000")
            tab.click("[data-act=watch-record]")
            tab.wait_for_selector("#watch-took")
            assert asked == [("padded", {"size": 3, "seconds": 120, "at": 1000})], "an older save was chosen"
            assert "A 3v3 game of 2 min is being played and recorded." in tab.text_content(".modal")
            tab.click(".modal [data-close]")  # hidden: the recording goes on, and its box comes back
            tab.click("[data-act=watch]")
            tab.wait_for_selector("#watch-took")
            job.update(state="failed", why="the game could not be played")
            tab.wait_for_selector("#bothead .banner.bad")
            assert tab.locator(".modal").count() == 0, "the box closes"
            banner = tab.text_content("#bothead .banner.bad")
            assert "Recording a game did not work." in banner and "the game could not be played" in banner

            job.update(state="running")
            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-act=watch-record]")
            tab.click("[data-act=watch-record]")
            tab.wait_for_selector("[data-act=watch-cancel]")
            assert asked[-1] == ("padded", {"size": 2, "seconds": 60, "at": None}), "the newest save"
            assert tab.locator("#bothead .banner.bad").count() == 0, "a new try clears the old failure"
            tab.click("[data-act=watch-cancel]")
            tab.wait_for_selector(".modal", state="detached")
            assert asked[-1] == ("stop", "padded", "watch")
            job.update(state="done")
            tab.wait_for_timeout(1200)
            assert tab.evaluate("window.NV === null && $('#watching') === null"), (
                "a stopped one opens nothing"
            )

            folder = runs / "padded"
            layouts.save(folder / "watch.nlrec", fake().data("padded at 3,000 agent steps", "made up"))
            (folder / "watch.html").write_text("<html></html>", encoding="utf-8")
            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-act=watch-recorded]")
            said = tab.text_content(".modal")
            assert (
                "The last recording" in said and "Save at 3.0K steps · 1v1 · 0 s · recorded just now" in said
            )
            assert "can be passed on to someone else" in said and "it draws a generated arena" in said
            tab.click("[data-act=watch-page]")
            tab.click("[data-act=watch-file]")
            tab.wait_for_function("document.querySelector('[data-act=watch-file]:not([disabled])')")
            assert asked[-2:] == [("page", "padded"), ("file", "padded")]
            tab.click(".modal [data-close]")

            tab.click(".nav [data-go=home]")
            tab.click("#main [data-bot=plain]")
            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-act=watch-live]")
            assert tab.locator("[data-seg=watch-size] button").all_text_contents() == ["2v2"]
            assert "It plays the size it was made for." in tab.text_content(".modal")
            assert tab.locator("#watch-at option").count() == 1

            # The search box lists watching live and the last recording (Watch it play is its own entry).
            listed = opened(browser, Bridge(api), problems, page)
            listed.wait_for_selector(".botcard")
            assert listed.evaluate(
                "['watch', 'watch-live', 'watch-last'].every(id => COMMANDS.some(c => c.id === id))"
            )
            assert not listed.evaluate("COMMANDS.find(c => c.id === 'watch-live').when()"), "no bot is open"
            listed.click("#main [data-bot=plain]")
            listed.wait_for_selector("[data-act=watch]")
            assert listed.evaluate("COMMANDS.find(c => c.id === 'watch-live').when()")
            listed.evaluate("COMMANDS.find(c => c.id === 'watch').run()")
            listed.wait_for_selector("[data-act=watch-live]")
            listed.close()
        finally:
            browser.close()
    assert problems == []


def test_the_viewer_opens_over_the_window_loads_once_and_goes_from_memory_when_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "shown", saves=1).root
    layouts.save(folder / "watch.nlrec", fake(frames=240).data("shown at 1,000 agent steps", "made up"))
    (folder / "recordings").mkdir()
    earlier = fake(frames=120, cars=4).data("shown at 500 agent steps", "made up")
    layouts.save(folder / "recordings" / "000000000500-20260101-120000.nlrec", earlier)
    api = Api(ROOT, runs)
    api.set_viewer_prefs({"camera": "top", "helpSeen": True, "chase": {"distance": 350, "ballLock": False}})
    bridge = Bridge(api)
    problems: list[str] = []
    files = sum(api.viewer_assets(part)["pieces"] for part in ("three", "script", "style", "markup"))
    resting = ("home", "bot", "console_text", "build_state", "converting")
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        no_listening(monkeypatch)
        try:
            tab = opened(browser, bridge, problems)
            tab.click("#main [data-bot=shown]")
            tab.click("[data-tab=console]")
            assert tab.evaluate("typeof THREE + typeof NV_MAKE") == "undefinedundefined", (
                "not yet in the page"
            )
            tab.click("[data-act=watch]")
            tab.click("[data-act=watch-recorded] >> nth=0")
            tab.wait_for_function(STARTED, timeout=120_000)
            assert tab.locator(".watch-wait").count() == 0 and tab.locator(".modal").count() == 0
            assert bridge.calls.count("viewer_assets") == files and bridge.calls.count("viewer_prefs") == 1
            inside = "(() => { const box = $('#watching'), host = $('#watching > .nv-root');"
            inside += " const at = box.getBoundingClientRect();"
            inside += " return [!box.hidden, host === NV.host, host.contains(NV.renderer.domElement),"
            inside += " at.width === innerWidth && at.height === innerHeight, NV.options.embedded,"
            inside += " $('[data-nv=close]').hidden, $('[data-nv=hint]').hidden]; })()"
            assert tab.evaluate(inside) == [True, True, True, True, True, False, True]
            assert tab.evaluate("[NV.rec.title, NV.rec.cars, NV.rec.count, NV.rec.live]") == [
                "shown at 1,000 agent steps",
                2,
                240,
                False,
            ]
            assert tab.evaluate("NV.cams.mode") == "top", "the choices kept with the dashboard's own"
            assert tab.evaluate(
                "[NV.cams.settings.distance, NV.cams.settings.ballLock, NV.cams.settings.fov]"
            ) == [
                350,
                False,
                110,
            ], "the camera settings too; one not kept takes its starting value"
            drawn = tab.evaluate("NV.frames")
            tab.wait_for_function(f"NV.frames > {drawn + 5}", timeout=60_000)
            assert tab.evaluate("[S.resting, $('#main').inert, $('#band').inert, $('#toasts').inert]") == [
                True,
                True,
                True,
                False,
            ]
            assert tab.evaluate("document.documentElement.hasAttribute('data-watching')")

            # The viewer's keys are its own: the page underneath never sees them.
            tab.evaluate(
                "window.__seen = 0; document.addEventListener('keydown', () => { window.__seen += 1; });"
                " window.__first = new WeakRef(NV); window.__three = THREE;"
                " window.__scripts = document.scripts.length;"
            )
            tab.keyboard.press("h")
            tab.wait_for_function("NV.host.classList.contains('nv-hidden')")
            tab.keyboard.press("h")
            tab.keyboard.press("2")
            tab.wait_for_function("NV.cams.mode === 'ball' && !NV.host.classList.contains('nv-hidden')")
            pumped(tab, lambda: api.viewer_prefs()["prefs"].get("camera") == "ball", 10, "the kept choice")
            tab.evaluate("NV.cams.setSetting('height', 150)")
            pumped(
                tab,
                lambda: api.viewer_prefs()["prefs"].get("chase", {}).get("height") == 150,
                10,
                "a camera setting kept with the dashboard's own",
            )
            assert api.viewer_prefs()["prefs"]["helpSeen"] is True and api.home()["prefs"]["theme"] == "dark"
            mark = len(bridge.calls)
            tab.wait_for_timeout(5600)
            assert not [call for call in bridge.calls[mark:] if call in resting], "the page rests underneath"
            assert tab.evaluate("window.__seen") == 0

            tab.keyboard.press("Escape")
            tab.wait_for_function("window.NV === null")
            assert tab.evaluate("[$('#watching').hidden, $('#watching').childElementCount, S.resting]") == [
                True,
                0,
                False,
            ]
            assert tab.evaluate(
                "[$('#main').inert, document.querySelectorAll('canvas.nv-canvas').length]"
            ) == [
                False,
                0,
            ]
            assert not tab.evaluate("document.documentElement.hasAttribute('data-watching')")
            assert freed(tab, "__first"), "the closed viewer, its scene and its recording left the memory"
            pumped(tab, lambda: "console_text" in bridge.calls[mark:], 10, "the page to wake up")
            tab.click("[data-tab=settings]")
            tab.wait_for_selector("#f-ppo-policy_lr")
            tab.fill("#f-ppo-policy_lr", "0.0002")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("#pending:not([hidden])")
            assert api.bot("shown")["pending"] == {"ppo.policy_lr": 0.0002}, "the page works as before"

            # A second game, with other cars: nothing is loaded or run again.
            tab.click("[data-act=watch]")
            tab.wait_for_selector(".watch-rec")
            assert "Save at 500 steps · 2v2" in tab.text_content(".watch-rec")
            tab.click(".watch-rec [data-act=watch-recorded]")
            tab.wait_for_function(STARTED, timeout=120_000)
            assert tab.evaluate("[NV.rec.title, NV.rec.cars, NV.cams.mode]") == [
                "shown at 500 agent steps",
                4,
                "ball",
            ]
            assert tab.evaluate(
                "[window.__three === THREE, document.scripts.length === window.__scripts]"
            ) == [
                True,
                True,
            ]
            assert bridge.calls.count("viewer_assets") == files and bridge.calls.count("viewer_arena") == 1
            assert bridge.calls.count("viewer_prefs") == 1
            drawn = tab.evaluate("NV.frames")
            tab.wait_for_function(f"NV.frames > {drawn + 5}", timeout=60_000)
            # The browser may keep an element of a closed viewer (the last one clicked or under the pointer):
            # holding one here, the viewer must still go.
            tab.evaluate(
                "window.__second = new WeakRef(NV); window.__seen = 0;"
                " window.__kept = document.querySelector('[data-nv=close]');"
            )
            tab.keyboard.press("3")
            tab.wait_for_function("NV.cams.mode === 'car'")
            tab.mouse.move(400, 300)
            tab.mouse.move(430, 330)
            tab.click("[data-nv=close]")
            tab.wait_for_function("window.NV === null")
            assert freed(tab, "__second"), "a closed viewer leaves no listener on its elements to keep it"
            assert tab.evaluate("!window.__kept.isConnected")
            tab.evaluate("window.__kept = null")
            assert tab.evaluate("window.__seen") == 0

            # Closed while it is still opening: nothing starts afterwards.
            # The waiting box is checked in the same call: with every file at hand the game can open before a
            # later look.
            assert tab.evaluate("void watchShow('shown', { id: 'last' }); !!$('.watch-wait')")
            tab.keyboard.press("Escape")
            tab.wait_for_timeout(1500)
            assert tab.evaluate("[window.NV, $('#watching').hidden, S.resting, WATCH.open]") == [
                None,
                True,
                False,
                None,
            ]
            tab.evaluate("void watchShow('shown', { id: 'no-such-game' })")
            tab.wait_for_selector(".toast.bad")
            assert "not there any more" in tab.text_content(".toast.bad")
            assert tab.evaluate("[$('#watching').hidden, S.resting]") == [True, False]

            # A live game that cannot start (this bot's save is an empty file) says so, and why on its page.
            tab.click("[data-act=watch]")
            tab.click("[data-act=watch-live]")
            tab.wait_for_selector(".watch-wait")
            failed = tab.locator(".toast.bad", has_text="The live game of shown did not start")
            failed.wait_for(timeout=120_000)
            assert "the reason is on its page" in failed.text_content()
            assert "The live game did not work." in tab.text_content("#bothead .banner.bad")
            assert tab.evaluate("[window.NV, $('#watching').hidden, S.resting]") == [None, True, False]
            pumped(
                tab, lambda: api.watch_live_state() == {"state": "none"}, 30, "the failed game to be let go"
            )
        finally:
            browser.close()
    assert problems == []


@NEEDS_MESHES
def test_a_recording_of_a_newer_version_is_refused_with_a_message(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "shown", saves=1).root
    newer = {**fake().data("shown at 1,000 agent steps", "made up"), "format": layouts.FORMAT + 1}
    layouts.save(folder / "watch.nlrec", newer)
    bridge = Bridge(Api(ROOT, runs))
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        try:
            tab = opened(browser, bridge, problems)
            tab.click("#main [data-bot=shown]")
            tab.evaluate("void watchShow('shown', { id: 'last' })")
            tab.wait_for_selector(".toast.bad", timeout=120_000)
            said = f"a newer NitroLearn (format {layouts.FORMAT + 1}). Update NitroLearn to watch it."
            assert said in tab.text_content(".toast.bad")
            assert tab.evaluate("[window.NV, $('#watching').hidden, S.resting]") == [None, True, False]
        finally:
            browser.close()
    assert problems == []


def test_a_game_recorded_from_the_dialog_opens_by_itself_and_a_live_game_plays_until_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_api = playwright_or_skip()
    folder = playing_bot(tmp_path, 500, 900, env="team_size = 2\nobs_team_size = 2\n")
    api = Api(ROOT, tmp_path / "runs")
    bridge = Bridge(api)
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        no_listening(monkeypatch)
        try:
            tab = opened(browser, bridge, problems)
            tab.click("#main [data-bot=my-bot]")
            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-act=watch-record]")
            assert tab.locator("[data-seg=watch-size] button").all_text_contents() == ["1v1", "2v2"]
            tab.click("[data-seg=watch-size] [data-v='1']")
            tab.select_option("#watch-at", "500")
            tab.evaluate("WATCH.pick.seconds = 5")  # a short game, for the test
            tab.click("[data-act=watch-record]")
            tab.wait_for_selector("#watch-took")
            tab.wait_for_function(STARTED, timeout=180_000)  # the real `nitro watch`, then the viewer
            assert tab.evaluate("[NV.rec.title, NV.rec.cars, NV.rec.live, !!NV.options.arena]") == [
                "my-bot at 500 agent steps",
                2,
                False,
                True,
            ]
            assert (
                tab.evaluate("Math.round(NV.rec.duration)") == 5
                and api.job("my-bot", "watch")["state"] == "done"
            )
            assert (folder.root / "watch.html").is_file() and (folder.root / "watch.nlrec").is_file()
            drawn = tab.evaluate("NV.frames")
            tab.wait_for_function(f"NV.frames > {drawn + 5}", timeout=60_000)
            tab.keyboard.press("Escape")
            tab.wait_for_function("window.NV === null")
            assert api._live is None, "a recorded game needs no process once it is recorded"

            tab.click("[data-act=watch]")
            tab.wait_for_selector("[data-act=watch-recorded]")
            assert "Save at 500 steps · 1v1 · 5 s · recorded just now" in tab.text_content(".modal")
            tab.click("[data-act=watch-live]")
            tab.wait_for_selector(".watch-wait")
            assert "Starting the live game" in tab.text_content(".watch-wait")
            tab.wait_for_function(STARTED, timeout=180_000)
            live = api._live
            assert live is not None and live.process.poll() is None and live.bot == "my-bot"
            assert tab.evaluate("[NV.rec.title, NV.rec.cars, NV.rec.live, NV.player.atLive]") == [
                "my-bot at 900 agent steps",
                4,
                True,
                True,
            ]
            assert not tab.evaluate("$('[data-nv=liveBadge]').hidden")
            played = tab.evaluate("NV.rec.count")
            tab.wait_for_function(f"NV.rec.count > {played + 60}", timeout=60_000)
            assert "viewer_poll" in bridge.calls and tab.evaluate("S.resting")
            if Path("/proc").is_dir():
                assert sockets_of(live.process.pid) == []
            tab.keyboard.press("Escape")
            tab.wait_for_function("window.NV === null")
            pumped(tab, lambda: live.process.poll() is not None, 30, "the live game's process to end")
            assert api._live is None and live.process.returncode == 0

            # A live game whose process goes by itself is said, and the viewer keeps what was played.
            tab.click("[data-act=watch]")
            tab.click("[data-act=watch-live]")
            tab.wait_for_function(STARTED, timeout=180_000)
            again = api._live
            assert again is not None and again is not live
            again.process.kill()
            tab.wait_for_selector(".toast.bad")
            said = "The live game of my-bot stopped by itself: the reason is on its page."
            assert said in tab.text_content(".toast.bad")
            tab.wait_for_function("NV.live.ended && !NV.rec.live")
            assert tab.evaluate("$('[data-nv=liveBadge]').textContent") == "Ended"
            tab.keyboard.press("Escape")
            tab.wait_for_function("window.NV === null")
            pumped(tab, lambda: api._live is None, 30, "the ended game to be let go")
            assert "The live game did not work." in tab.text_content("#bothead .banner.bad")

            # Cancelled while its process is still loading the bot: the process ends, nothing opens.
            tab.click("[data-act=watch]")
            tab.click("[data-act=watch-live]")
            tab.wait_for_selector(".watch-wait")
            assert tab.locator("#bothead .banner.bad").count() == 0, "a new try clears the old failure"
            pumped(tab, lambda: api._live is not None, 30, "the game's process to be started")
            cancelled = api._live
            assert cancelled is not None
            tab.click(".watch-wait [data-watch-close]")
            tab.wait_for_selector("#watching", state="hidden")
            pumped(tab, lambda: cancelled.process.poll() is not None, 30, "the cancelled game to end")
            tab.wait_for_timeout(1000)
            assert api._live is None and tab.evaluate("[window.NV, S.resting]") == [None, False]
        finally:
            browser.close()
            api._end_live()
    assert problems == []
