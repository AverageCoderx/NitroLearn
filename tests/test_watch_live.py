"""`nitro watch --live`: a game played in real time into the viewer's feed and shown in a window of its own,
with nothing listening on any port; and `--feed`, the same game as lines on standard output."""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nitro import viewer
from nitro.cli import main
from nitro.config import NitroConfig
from nitro.runs import RunFolder
from nitro.viewer import feed, window
from nitro.viewer import recording as layouts
from nitro.watch import CAR, HEAD, LAYOUT, Live
from test_eval import save as plain_save
from test_watch import NEEDS_MESHES, chromium, fake, open_tab, save, saved_run, tiny_config

WIDTH = HEAD + 2 * CAR


def frames_of(chunk: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(chunk["f"]), dtype="<i2").reshape(-1, WIDTH)


def wait_for(done: Any, seconds: float = 120.0) -> None:
    deadline = time.monotonic() + seconds
    while not done():
        assert time.monotonic() < deadline, "waited too long"
        time.sleep(0.02)


def state_of(config: NitroConfig, seed: int) -> dict[str, Any]:
    import torch

    from nitro.training import build

    torch.manual_seed(seed)
    return {"learner": {"model": build(config).model.state_dict()}}


@NEEDS_MESHES
def test_a_live_game_fills_the_store_in_real_time_and_a_newer_save_takes_over() -> None:
    config = tiny_config()
    live = Live(config, state_of(config, 1), "my run", "a test", seed=3, fps=60)
    chunks = feed.Chunks()
    chunks.add(live.header)
    header = chunks.header
    assert header is not None and header["live"] is True and header["frames"] == ""
    assert (header["format"], header["head"], header["car"], header["cars"]) == (LAYOUT, HEAD, CAR, 2)
    assert (header["title"], header["fps"], header["decisionsPerSecond"]) == ("my run", 60.0, 15.0)
    assert len(header["pads"]) == 34 and len(header["scales"]) == WIDTH
    offers = [(config, state_of(config, 2), "a newer save")]
    asked: list[float] = []

    def newer() -> tuple[NitroConfig, dict[str, Any], str] | None:
        asked.append(time.monotonic())
        return offers.pop() if offers and len(asked) > 1 else None

    stop = threading.Event()
    player = threading.Thread(target=live.play, args=(chunks.add, stop, newer), kwargs={"check_seconds": 0.5})
    began = time.monotonic()
    player.start()
    try:
        wait_for(lambda: any("title" in chunk for chunk in chunks.after(0)["chunks"]))
        taken = len(chunks.after(0)["chunks"])
        wait_for(lambda: chunks.after(0)["next"] >= taken + 5)
        seconds = time.monotonic() - began
    finally:
        stop.set()
        player.join(timeout=60)
    assert not player.is_alive(), "stopping ends the game"
    played = chunks.after(0)
    assert (
        played["restart"] is False and played["closed"] is False and played["next"] == len(played["chunks"])
    )
    first, *rest = played["chunks"]
    assert frames_of(first).shape == (1, WIDTH) and first["d"] == [], "the start: one frame in the layout"
    assert frames_of(first)[0, 0] == 1, "marked as the start of an episode"
    seen = 1
    decisions = 0
    for chunk in rest:
        frames = frames_of(chunk)
        if "title" in chunk:
            assert chunk["title"] == "a newer save" and chunk["d"] == [] and frames.shape == (1, WIDTH)
            assert frames[0, 0] == 1, "the newer save starts with a new game"
        else:
            assert frames.shape == (4, WIDTH), "a decision's step: every 2 of its 8 ticks"
            [decision] = chunk["d"]
            assert decision[0] == seen - 1, "the decision looked at the newest frame before its step"
            assert len(decision) == 3 and all(len(car) == 4 for car in decision[1:]), "both cars' figures"
            decisions += 1
        seen += len(frames)
    assert sum("title" in chunk for chunk in rest) == 1 and len(asked) >= 2
    assert decisions / 15 <= seconds + 0.5, "no faster than the game is played"
    assert chunks.after(played["next"] - 2)["chunks"] == played["chunks"][-2:]


def test_the_bridge_offers_the_page_one_call_and_nothing_else() -> None:
    chunks = feed.Chunks()
    chunks.add(feed.header(fake().data("live")))
    chunks.add(json.dumps({"f": "", "d": []}))
    bridge = window.Bridge(chunks)
    assert [name for name in dir(bridge) if not name.startswith("_")] == ["viewer_poll"]
    assert bridge.viewer_poll(0) == {
        "chunks": [{"f": "", "d": []}],
        "next": 1,
        "restart": False,
        "closed": False,
    }
    assert bridge.viewer_poll(1.0)["chunks"] == [], "the page's cursor is a JavaScript number"
    chunks.close()
    assert bridge.viewer_poll(1)["closed"] is True


class FakeGui:
    """Stands in for the system's window: takes the window pywebview hands over, as if it was shown and
    closed at once."""

    renderer = "test"

    def __init__(self) -> None:
        self.shown: list[Any] = []

    def create_window(self, shown: Any) -> None:
        self.shown.append(shown)
        shown.events.closed.set()


@pytest.fixture
def no_servers(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeGui, list[str]]:
    """pywebview with no system window behind it, noting every web server it starts and every socket
    that is bound or listens."""
    webview = pytest.importorskip("webview")
    gui = FakeGui()
    started: list[str] = []
    monkeypatch.setattr(webview, "windows", [])
    monkeypatch.setattr(webview, "guilib", None)
    monkeypatch.setattr(webview, "renderer", getattr(webview, "renderer", None))
    monkeypatch.setattr(webview, "initialize", lambda forced=None: gui)

    def note(name: str) -> Any:
        def noted(*args: Any, **kwargs: Any) -> Any:
            started.append(name)
            return "http://127.0.0.1:1/", None, None

        return noted

    monkeypatch.setattr(webview.http, "start_server", note("start_server"))
    monkeypatch.setattr(webview.http, "start_global_server", note("start_global_server"))
    monkeypatch.setattr(webview.http.BottleServer, "start_server", note("BottleServer.start_server"))
    monkeypatch.setattr(socket.socket, "listen", note("socket.listen"))
    monkeypatch.setattr(socket.socket, "bind", note("socket.bind"))
    return gui, started


def test_the_window_opens_the_page_as_a_file_and_starts_no_server_and_no_socket(
    tmp_path: Path, no_servers: tuple[FakeGui, list[str]]
) -> None:
    gui, started = no_servers
    chunks = feed.Chunks()
    chunks.add(feed.header(fake().data("live")))
    page = tmp_path / "runs" / "my bot" / "live.html"
    page.parent.mkdir(parents=True)
    page.write_text(viewer.page(chunks.header or {}), encoding="utf-8")
    closed = threading.Event()
    window.show(page, "my bot, live", chunks, closed.set)
    [shown] = gui.shown
    assert shown.real_url == page.resolve().as_uri() and shown.real_url.startswith("file:///")
    assert shown.html is None, "the page is too big to be handed over as text"
    assert shown.title == "my bot, live" and shown._js_api.viewer_poll(0)["next"] == 0
    assert closed.wait(10), "closing the window is passed on"
    assert started == [], "no web server was started, and no socket bound or listening"


def test_pywebview_would_start_its_web_server_for_a_plain_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_servers: tuple[FakeGui, list[str]]
) -> None:
    """Why the window is given a file address: the same call with a plain path starts pywebview's local
    web server. This also shows the test above would notice one."""
    import webview

    _, started = no_servers
    (tmp_path / "live.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    webview.create_window("a plain path", url="live.html")
    webview.start(http_server=False)
    assert "start_global_server" in started


@NEEDS_MESHES
def test_the_live_command_plays_into_its_window_until_it_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = saved_run(tmp_path, 500)
    runs = folder.root.parent
    seen: dict[str, Any] = {}

    def show(page: Path, title: str, chunks: feed.Chunks, closed: Any) -> None:
        """A window that is closed once it has asked a few times and a newer save has taken over."""
        bridge = window.Bridge(chunks)
        seen.update(page=page, title=title, text=page.read_text(encoding="utf-8"), chunks=[])
        cursor = 0
        save(folder, 900)
        deadline = time.monotonic() + 120
        while not any("title" in chunk for chunk in seen["chunks"]) and time.monotonic() < deadline:
            answer = bridge.viewer_poll(cursor)
            seen["chunks"] += answer["chunks"]
            cursor = answer["next"]
            time.sleep(0.05)
        closed()

    monkeypatch.setattr(window, "show", show)
    monkeypatch.setattr("nitro.cli._FOLLOW_SECONDS", 0.5)
    assert main(["watch", "my-bot", "--runs", str(runs), "--live", "--follow"]) == 0
    said = capsys.readouterr().out
    assert "in its own window" in said and "nothing is served on any port" in said and "Stopped" in said
    assert seen["page"] == folder.root / "live.html" and seen["title"] == "my-bot at 500 agent steps"
    assert "NV_ARENA=" in seen["text"], "the page stays on this computer: its arena is inside"
    data = layouts.from_page(seen["text"])
    assert data["live"] is True and data["title"] == "my-bot at 500 agent steps" and data["cars"] == 2
    assert "played live" in data["subtitle"] and "Each new save takes over" in data["subtitle"]
    assert frames_of(seen["chunks"][0]).shape == (1, WIDTH)
    assert [chunk["title"] for chunk in seen["chunks"] if "title" in chunk] == ["my-bot at 900 agent steps"]
    assert not [thread for thread in threading.enumerate() if thread.name == "nitro-live-game"], (
        "closing the window ends the game"
    )
    assert not (folder.root / "watch.html").exists(), "a live view records nothing"
    assert not (folder.root / "live.html").exists(), "and its page goes with its window"


def test_a_system_that_cannot_make_a_window_is_reported_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_servers: tuple[FakeGui, list[str]]
) -> None:
    import webview
    from webview.errors import WebViewException

    def no_gui(forced: Any = None) -> Any:
        raise WebViewException("no web engine here")

    monkeypatch.setattr(webview, "initialize", no_gui)
    (tmp_path / "live.html").write_text("<html></html>", encoding="utf-8")
    with pytest.raises(window.NoWindow, match="no web engine here"):
        window.show(tmp_path / "live.html", "live", feed.Chunks())
    assert no_servers[1] == []


@NEEDS_MESHES
def test_the_live_command_ends_its_game_when_the_window_cannot_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = saved_run(tmp_path, 500)

    def show(page: Path, title: str, chunks: feed.Chunks, closed: Any) -> None:
        raise window.NoWindow("no web engine here")

    monkeypatch.setattr(window, "show", show)
    assert main(["watch", "my-bot", "--runs", str(folder.root.parent), "--live"]) == 1
    assert "could not open its window: no web engine here" in capsys.readouterr().err
    assert not [thread for thread in threading.enumerate() if thread.name == "nitro-live-game"]


def test_the_live_command_says_what_it_needs_when_no_window_can_be_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n', encoding="utf-8")
    plain_save(RunFolder.create(tmp_path / "runs", "my-bot", settings), 500, seed=1)
    monkeypatch.setattr(window, "available", lambda: False)
    assert main(["watch", "my-bot", "--runs", str(tmp_path / "runs"), "--live"]) == 1
    assert "needs pywebview" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "my-bot" / "live.html").exists()


@NEEDS_MESHES
def test_the_feed_writes_a_header_then_chunks_and_nothing_else_until_its_input_closes(tmp_path: Path) -> None:
    folder = saved_run(tmp_path, 500)
    command = [sys.executable, "-m", "nitro", "watch", "my-bot", "--runs", str(folder.root.parent), "--feed"]
    collected: list[str] = []
    with (tmp_path / "notices.txt").open("w", encoding="utf-8") as notices:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=notices,
            text=True,
            encoding="utf-8",
        )
        assert process.stdin is not None and process.stdout is not None
        reader = threading.Thread(target=lambda: collected.extend(process.stdout or []), daemon=True)
        reader.start()
        try:
            wait_for(lambda: len(collected) >= 12 or process.poll() is not None)
            assert process.poll() is None, (tmp_path / "notices.txt").read_text(encoding="utf-8")
            process.stdin.close()  # whoever was reading has gone: the game ends
            assert process.wait(timeout=60) == 0
        finally:
            process.kill()
        reader.join(timeout=30)
    assert all(line.endswith("\n") for line in collected)
    header, *chunks = (json.loads(line) for line in collected)
    assert (
        header["live"] is True and header["frames"] == "" and header["title"] == "my-bot at 500 agent steps"
    )
    assert (header["format"], header["head"], header["car"], header["cars"]) == (LAYOUT, HEAD, CAR, 2)
    assert len(chunks) >= 11 and all(set(chunk) == {"f", "d"} for chunk in chunks)
    assert frames_of(chunks[0]).shape == (1, WIDTH) and chunks[0]["d"] == []
    assert all(frames_of(chunk).shape == (4, WIDTH) and len(chunk["d"]) == 1 for chunk in chunks[1:])
    store = feed.Chunks()
    for line in collected:
        store.add(line)
    assert store.header == header and store.after(0)["chunks"] == chunks, "what a reader keeps of it"


def test_the_live_page_plays_what_its_window_hands_it_and_says_when_the_game_has_ended(
    tmp_path: Path,
) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    game = fake(frames=125)
    chunks = feed.Chunks()
    chunks.add(feed.header(game.data("a live game", "made up")))
    chunks.add(feed.chunk(game.frames[:1], [], 2))
    for at in range(1, 121, 4):
        decision = [at - 1, [0.5, 0.9, 0.2, 3], [-0.25, 0.4, 0.7, 11]]
        chunks.add(feed.chunk(game.frames[at : at + 4], [decision], 2))
    file = tmp_path / "live.html"
    file.write_text(viewer.page(chunks.header or {}), encoding="utf-8")
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        try:
            tab = open_tab(browser, problems)
            tab.expose_function("windowPoll", lambda cursor: window.Bridge(chunks).viewer_poll(cursor))
            # The bridge comes a little after the page, as a window's does.
            tab.add_init_script(
                "setTimeout(() => {"
                " window.pywebview = { api: { viewer_poll: (cursor) => window.windowPoll(cursor) } };"
                " }, 400);"
            )
            tab.goto(file.as_uri())
            tab.wait_for_function("typeof NV !== 'undefined' && NV.frames > 2", timeout=120_000)
            tab.wait_for_function("NV.rec.count === 121 && NV.rec.decisions.length === 30", timeout=60_000)
            assert tab.evaluate("[NV.rec.live, NV.player.atLive, NV.live.ended]") == [True, True, False]
            chunks.add(feed.chunk(game.frames[121:125], [], 2, "a newer save"))
            tab.wait_for_function("NV.rec.count === 125 && NV.rec.title === 'a newer save'", timeout=60_000)
            chunks.close()
            tab.wait_for_function("NV.live.ended && !NV.rec.live", timeout=60_000)
        finally:
            browser.close()
    assert problems == []


def test_a_live_page_opened_without_its_window_says_so(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    file = tmp_path / "live.html"
    file.write_text(viewer.page(json.loads(feed.header(fake().data("a live game")))), encoding="utf-8")
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        try:
            tab = open_tab(browser, problems)
            tab.goto(file.as_uri())
            notice = tab.wait_for_selector(".nv-notice", timeout=60_000)
            assert "nitro watch NAME --live" in notice.inner_text()
            assert tab.evaluate("NV.frames === undefined"), "the viewer was not started"
        finally:
            browser.close()
    assert problems == []
