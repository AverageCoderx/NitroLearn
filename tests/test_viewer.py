"""The match viewer: how recordings are packed and read back, and what the page is made of."""

from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nitro import viewer
from nitro.viewer import recording


def frames_of(cars: int, count: int = 40) -> np.ndarray:
    """Made-up frames: every column moving within what its scale can hold."""
    random = np.random.default_rng(cars)
    scales = np.asarray(recording.scales(cars), dtype=np.float64)
    whole = random.integers(-3000, 3000, size=(count, recording.width(cars)))
    return (whole / scales).astype(np.float32)


def game(cars: int = 2) -> dict[str, Any]:
    return recording.data(
        frames_of(cars),
        fps=60,
        pads=[[0.0, -4240.0, 70.0, 0.0], [3584.0, 0.0, 73.0, 1.0]],
        decisions=[[0, [0.1, 0.5, 0.9, 3], [0.2, 0.4, 0.8, 7]]],
        decisions_per_second=15,
        title="a test game",
    )


@pytest.mark.parametrize("cars", [2, 6])
def test_packed_frames_come_back_as_they_were_stored(cars: int) -> None:
    frames = frames_of(cars)
    stored = recording.quantize(frames, cars)
    assert np.array_equal(recording.unpack(recording.pack(stored), recording.width(cars)), stored)
    scales = np.asarray(recording.scales(cars))
    assert np.abs(stored / scales - frames).max() <= 0.5 / scales.min() + 1e-6, "to within the rounding"


def test_a_value_beyond_16_bits_is_held_at_the_limit_and_a_wide_jump_survives_packing() -> None:
    frames = np.zeros((3, recording.width(1)), dtype=np.float32)
    frames[1, 4] = 90_000.0  # a velocity no game reaches
    frames[2, 4] = -90_000.0
    stored = recording.quantize(frames, 1)
    assert stored[1, 4] == 32767 and stored[2, 4] == -32768
    assert np.array_equal(recording.unpack(recording.pack(stored), recording.width(1)), stored)


def test_frames_that_do_not_fit_the_layout_are_refused() -> None:
    with pytest.raises(ValueError, match="do not fit format 2"):
        recording.data(
            np.zeros((2, 60), dtype=np.float32),
            fps=60,
            pads=[],
            decisions=[],
            decisions_per_second=15,
        )


def test_a_recording_is_saved_and_loaded_and_read_out_of_a_page(tmp_path: Path) -> None:
    data = game()
    file = tmp_path / f"game{recording.SUFFIX}"
    recording.save(file, data)
    assert recording.load(file) == data
    assert not list(tmp_path.glob("*.part")), "the unfinished file is gone"
    page = tmp_path / "game.html"
    page.write_text(viewer.page(data), encoding="utf-8")
    assert recording.load(page) == data
    (tmp_path / "other.html").write_text("<html>no game here</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no recording"):
        recording.load(tmp_path / "other.html")


def test_a_recording_in_a_format_this_version_does_not_read_is_refused(tmp_path: Path) -> None:
    newer = {**game(), "format": recording.FORMAT + 1}
    said = rf"made by a newer NitroLearn \(format {recording.FORMAT + 1}\); update NitroLearn to watch it"
    file = tmp_path / f"newer{recording.SUFFIX}"
    recording.save(file, newer)
    with pytest.raises(ValueError, match=said):
        recording.load(file)
    page = tmp_path / "newer.html"
    page.write_text(viewer.page(newer), encoding="utf-8")
    with pytest.raises(ValueError, match=said):
        recording.load(page)
    with pytest.raises(ValueError, match=said):
        recording.from_page(viewer.page(newer))
    unnamed = {key: value for key, value in game().items() if key != "format"}
    with pytest.raises(ValueError, match="a format this NitroLearn does not read"):
        recording.from_page(viewer.page(unnamed))


def test_the_page_holds_everything_and_loads_nothing_from_outside() -> None:
    page = viewer.page(game())
    assert "three.js 0.186.1 | MIT License" in page, "the library's notice travels with it"
    assert (viewer.VENDOR / "THREE-LICENSE.txt").read_text(encoding="utf-8").startswith("The MIT License")
    for mark in ("/*STYLE*/", "/*MARKUP*/", "/*THREE*/", "/*VIEWER*/", "/*DATA*/", "/*ARENA*/"):
        assert mark not in page, f"{mark} was filled in"
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", page), (
        "no script, style or image from the net"
    )
    own = viewer.script()
    for call in ("fetch(", "XMLHttpRequest", "WebSocket(", "EventSource(", "import("):
        assert call not in own, f"the viewer's own script never calls {call}"


def test_a_page_to_pass_on_carries_the_notices_of_what_is_inside_it() -> None:
    page = viewer.page(game())
    start = page.index("<!--")
    end = page.index("-->", start)
    assert start < page.index("<script>"), "the notices are at the top of the page"
    notice = page[start + 4 : end]
    assert "--" not in notice, "nothing in the notices can end the comment early"
    licence = (Path(viewer.__file__).resolve().parents[3] / "LICENSE").read_text(encoding="utf-8")
    required = [line for line in licence.splitlines() if line.startswith("Required Notice:")]
    assert required and all(line in notice for line in required)
    assert "https://polyformproject.org/licenses/noncommercial/1.0.0" in notice
    three = (viewer.VENDOR / "THREE-LICENSE.txt").read_text(encoding="utf-8").strip()
    assert three in notice, "three.js's licence text, word for word"
    fonts = (viewer.brand.FONTS / "LICENSES.txt").read_text(encoding="utf-8")
    for line in re.findall(r"^Copyright \d.*$", fonts, flags=re.MULTILINE):
        assert line in notice
    assert "SIL Open Font License, Version 1.1" in notice
    assert "PERMISSION & CONDITIONS" in notice, "the OFL's text travels with the fonts inside the page"
    assert "Required Notice:" not in viewer.maker(), "the dashboard's viewer carries no page notice"
    assert recording.from_page(page)["title"] == game()["title"]


def test_nothing_in_a_recording_can_end_the_pages_script() -> None:
    data = game()
    data["title"] = "</script><script>alert(1)</script>"
    page = viewer.page(data)
    assert "</script><script>alert(1)" not in page
    assert "<title>&lt;/script&gt;" in page
    assert recording.from_page(page)["title"] == data["title"]


def test_the_arena_goes_in_the_page_or_beside_it_or_not_at_all(tmp_path: Path) -> None:
    soccar = tmp_path / "soccar"
    soccar.mkdir()
    for number in (10, 2):
        (soccar / f"mesh_{number}.cmf").write_bytes(bytes([number]) * 5)
    blob = viewer.arena_blob(tmp_path)
    assert blob is not None
    packed = base64.b64decode(blob)
    assert packed[:4] == (2).to_bytes(4, "little")
    assert packed[4:13] == (5).to_bytes(4, "little") + bytes([2]) * 5, "in the files' number order"
    assert viewer.arena_blob(tmp_path / "nowhere") is None
    data = game()
    assert "NV_ARENA=" in viewer.page(data, arena=blob)
    beside = viewer.page(data, arena_src="arena.js")
    assert '<script src="arena.js"></script>' in beside and "NV_ARENA=" not in beside
    assert "NV_ARENA=" not in viewer.page(data), "a page to pass on carries no mesh"
    assert viewer.arena_script(blob) == f'window.NV_ARENA="{blob}";'


BALL_HEIGHT = 3  # its column in a frame


class ScriptedGame:
    """A game of one car whose frames hold nothing but the ball's height, which tells them apart: each
    decision adds two frames and looks at its newest own frame."""

    cars = 1
    width = recording.width(cars)

    def __init__(self, first: float) -> None:
        self.start = self.frame(first)
        self.frames = 1

    @classmethod
    def frame(cls, height: float) -> list[float]:
        numbers = [0.0] * cls.width
        numbers[BALL_HEIGHT] = height
        return numbers

    def decide(self) -> tuple[list[Any], list[float]]:
        decision = [self.frames - 1, [0.5]]
        self.frames += 2
        return decision, self.frame(1.0) + self.frame(2.0)


def test_a_live_game_is_written_chunk_by_chunk_and_a_newer_save_takes_over() -> None:
    import threading

    from nitro.viewer import feed

    lines: list[dict[str, Any]] = []
    stop = threading.Event()
    now = [0.0]
    offers = iter([None, (ScriptedGame(9.0), "a newer save")])

    def write(line: str) -> None:
        lines.append(json.loads(line))
        if len(lines) == 7:
            stop.set()

    def newer() -> tuple[feed.Playing, str] | None:
        return next(offers, None)

    def clock() -> float:
        now[0] += 0.5  # always behind: never waits
        return now[0]

    feed.play(ScriptedGame(0.0), write, cars=ScriptedGame.cars, seconds_per_decision=0.1, stop=stop,
              newer=newer, check_seconds=1.0, clock=clock)  # fmt: skip
    scale = recording.scales(ScriptedGame.cars)[BALL_HEIGHT]
    frames = [
        np.frombuffer(base64.b64decode(line["f"]), dtype="<i2").reshape(-1, ScriptedGame.width)
        for line in lines
    ]
    heights = [(chunk[:, BALL_HEIGHT] / scale).tolist() for chunk in frames]
    assert heights[0] == [0.0] and lines[0]["d"] == []
    assert heights[1] == [1.0, 2.0], "a decision's frames, in the recording's layout"
    assert [line["d"][0][0] for line in lines[1:3]] == [0, 2], "each decision names the frame it looked at"
    taken = next(i for i, line in enumerate(lines) if line.get("title") == "a newer save")
    before = sum(len(f) for f in frames[:taken])
    assert heights[taken] == [9.0], "the newer save starts with its own first frame"
    assert lines[taken + 1]["d"][0][0] == before, "and its decisions count on from there"


def test_the_reader_hands_out_what_is_new_and_says_when_a_page_fell_behind() -> None:
    import io

    from nitro.viewer import feed

    text = feed.header({"title": "live", "frames": "abc", "decisions": [[0]]}) + "\n"
    text += "".join(json.dumps({"f": str(i), "d": []}) + "\n" for i in range(5))
    reader = feed.Reader(io.StringIO(text), keep=3)
    head = reader.wait_header(5)
    assert head is not None and head["live"] is True and head["frames"] == "" and head["decisions"] == []
    reader._thread.join(5)
    late = reader.after(0)
    assert late["restart"] and [c["f"] for c in late["chunks"]] == ["2", "3", "4"] and late["next"] == 5
    assert late["closed"], "the process ended"
    assert reader.after(4) == {"chunks": [{"f": "4", "d": []}], "next": 5, "restart": False, "closed": True}
    assert reader.after(5)["chunks"] == []


def played_game(seconds: float = 3.0, fps: int = 60) -> dict[str, Any]:
    """A short made-up 1v1 that moves like a game: two cars circling, one boosting and going supersonic, a
    bouncing spinning ball, touches, a demolition and a goal for each team."""
    cars = 2
    count = int(seconds * fps)
    frames = np.zeros((count, recording.width(cars)), dtype=np.float32)
    time = np.arange(count) / fps
    frames[:, 1] = 800 * np.sin(time)  # ball x, y, z
    frames[:, 2] = 2500 * np.sin(time * 0.7)
    frames[:, 3] = 93 + 600 * np.abs(np.sin(time * 2))
    frames[:, 4] = 800 * np.cos(time)
    frames[:, 5] = 1750 * np.cos(time * 0.7)
    frames[:, 7] = np.cos(time * 3)  # ball forward and up: turning about y
    frames[:, 9] = np.sin(time * 3)
    frames[:, 10] = -np.sin(time * 3)
    frames[:, 12] = np.cos(time * 3)
    frames[:, 14] = 3.0
    frames[count // 3, 0] = 2.0  # a goal by Blue, then a new kick-off, then a goal by Orange
    frames[count // 3 + 1, 0] = 1.0
    frames[2 * count // 3, 0] = 3.0
    frames[count // 3 :, 16] = 1
    frames[2 * count // 3 :, 17] = 1
    frames[:, 18 : 18 + recording.PADS] = np.where(np.arange(recording.PADS) % 3 == 0, 2.5, 0.0)
    for car in range(cars):
        at = recording.HEAD + car * recording.CAR
        turn = time * (1.2 if car == 0 else -0.9) + car * 3
        radius = 1500 + 800 * car
        speed = 2250 if car == 0 else 900
        frames[:, at] = car  # team
        frames[:, at + 1] = radius * np.cos(turn)
        frames[:, at + 2] = radius * np.sin(turn)
        frames[:, at + 3] = 17
        frames[:, at + 4] = -speed * np.sin(turn)
        frames[:, at + 5] = speed * np.cos(turn)
        frames[:, at + 7] = -np.sin(turn)  # forward
        frames[:, at + 8] = np.cos(turn)
        frames[:, at + 12] = 1  # up
        frames[:, at + 16] = 100 - 20 * time  # boost
        flags = np.full(count, 1 + (4 if car == 0 else 0))  # on the ground; car 0 supersonic
        flags[:: fps // 2] += 64  # a touch twice a second
        state = np.full(count, 15 + (16 if car == 0 else 0))  # four wheels down; car 0 boosting
        if car == 1:
            gone = slice(count // 2, count // 2 + fps // 2)
            flags[gone] = 2
            state[gone] = 0
            frames[gone, at + 29] = 1.5
        frames[:, at + 17] = flags
        frames[:, at + 18] = 1  # throttle
        frames[:, at + 19] = 0.4  # steer
        frames[:, at + 24] = car == 0  # boost held
        frames[:, at + 26] = state
    decisions = [[k * 8, [0.1, 0.5, 0.9, 3], [-0.2, 0.4, 0.8, 7]] for k in range(count // 8)]
    return recording.data(
        frames,
        fps=fps,
        pads=PADS,
        decisions=decisions,
        decisions_per_second=7.5,
        title="a made-up game",
    )


PADS = [
    [float((i % 6 - 2.5) * 1200), float((i // 6 - 2.5) * 1600), 70.0, float(i % 9 == 0)] for i in range(34)
]
MESHES = Path(__file__).resolve().parents[1] / "collision_meshes"
GRAPHICS = [
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--no-proxy-server",
]


@pytest.mark.parametrize("arena", ["meshes", "generated"])
def test_the_page_draws_every_camera_at_every_quality_with_no_errors(tmp_path: Path, arena: str) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    blob = viewer.arena_blob(MESHES) if arena == "meshes" else None
    if arena == "meshes" and blob is None:
        pytest.skip(f"no collision meshes in {MESHES}")
    page = tmp_path / "game.html"
    page.write_text(viewer.page(played_game(), arena=blob), encoding="utf-8")
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        try:
            browser = play.chromium.launch(args=GRAPHICS)
        except sync_api.Error as error:
            pytest.skip(f"no Chromium to open the page in: {error}")
        try:
            tab = browser.new_page(viewport={"width": 640, "height": 400})
            tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
            tab.on("console", lambda said: problems.append(said.text) if said.type == "error" else None)
            tab.goto(page.as_uri())
            tab.wait_for_function("typeof NV !== 'undefined' && NV.frames > 2", timeout=120_000)
            assert tab.evaluate("[NV.rec.format, NV.rec.cars, NV.rec.goals.length, NV.rec.demos.length]") == [
                2,
                2,
                2,
                1,
            ]
            assert tab.evaluate("NV.world.pads.filter((pad) => pad.wait > 0).length") == 12, (
                "pad waits are read"
            )
            for quality in ("low", "medium", "high", "ultra"):
                tab.evaluate("(q) => NV.post.setQuality(q)", quality)
                for camera in tab.evaluate("NV.cams.MODES"):
                    # Played through from the start: every effect (boost, touches, demolition, goals) fires.
                    tab.evaluate(
                        "(mode) => { NV.player.seek(0); NV.player.setSpeed(4); NV.player.playing = true;"
                        " NV.cams.set(mode, 1); }",
                        camera,
                    )
                    drawn = tab.evaluate("NV.frames")
                    tab.wait_for_function("(n) => NV.frames >= n", arg=drawn + 6, timeout=120_000)
                    assert tab.evaluate("NV.cams.mode") == camera
                assert tab.evaluate("NV.q.tier") == quality
            assert tab.evaluate("NV.world.score.join('-')") in {"0-0", "1-0", "1-1"}
        finally:
            browser.close()
    assert problems == []


def circling_game(seconds: float = 40.0, fps: int = 30) -> dict[str, Any]:
    """A quiet made-up 1v1 for the chase cameras: Blue drives a steady circle round the centre (1.2 radians a
    second) with the ball rolling along its track a little behind it; Orange waits by its goal. No goals,
    touches or demolitions, so nothing shakes the camera."""
    cars = 2
    count = int(seconds * fps)
    frames = np.zeros((count, recording.width(cars)), dtype=np.float32)
    turn = np.arange(count) / fps * 1.2
    radius, speed = 1500, 1800
    behind = turn - 0.5
    frames[:, 1] = radius * np.cos(behind)  # ball x, y, z
    frames[:, 2] = radius * np.sin(behind)
    frames[:, 3] = 93
    frames[:, 4] = -speed * np.sin(behind)
    frames[:, 5] = speed * np.cos(behind)
    frames[:, 7] = 1  # ball forward and up
    frames[:, 12] = 1
    blue, orange = recording.HEAD, recording.HEAD + recording.CAR
    frames[:, blue + 1] = radius * np.cos(turn)
    frames[:, blue + 2] = radius * np.sin(turn)
    frames[:, blue + 4] = -speed * np.sin(turn)
    frames[:, blue + 5] = speed * np.cos(turn)
    frames[:, blue + 7] = -np.sin(turn)  # forward
    frames[:, blue + 8] = np.cos(turn)
    frames[:, orange] = 1
    frames[:, orange + 2] = 4800
    frames[:, orange + 8] = -1
    for at in (blue, orange):
        frames[:, at + 3] = 17
        frames[:, at + 12] = 1  # up
        frames[:, at + 16] = 50  # boost
        frames[:, at + 17] = 1  # on the ground
        frames[:, at + 26] = 15  # four wheels down
    return recording.data(frames, fps=fps, pads=PADS, decisions=[], decisions_per_second=15, title="circles")


# Where the camera is and how it looks, against the followed car: metres behind it along its nose, to its
# side and above its origin; the camera's pitch (degrees); how much its view points along the car's nose
# (cosine, flat); its vertical field of view and the picture's aspect.
CHASE = """() => {
  const cam = NV.camera, car = NV.world.cars[0];
  const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(car.quat).setY(0).normalize();
  const off = cam.position.clone().sub(car.pos), view = new THREE.Vector3();
  cam.getWorldDirection(view);
  return { back: -(off.x * nose.x + off.z * nose.z), side: off.x * nose.z - off.z * nose.x, up: off.y,
    pitch: THREE.MathUtils.radToDeg(Math.asin(view.y)),
    along: (view.x * nose.x + view.z * nose.z) / Math.hypot(view.x, view.z),
    fov: cam.fov, aspect: cam.aspect };
}"""
STARTING = {
    "fov": 110,
    "distance": 270,
    "height": 100,
    "angle": -3,
    "stiffness": 0.45,
    "swivel": 5,
    "transition": 1.2,
    "ballLock": True,
}


def vertical_fov(horizontal: float, aspect: float) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(horizontal) / 2) / aspect))


def test_the_chase_cameras_follow_the_camera_settings_and_keep_them(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    page = tmp_path / "game.html"
    page.write_text(viewer.page(circling_game()), encoding="utf-8")
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        try:
            browser = play.chromium.launch(args=GRAPHICS)
        except sync_api.Error as error:
            pytest.skip(f"no Chromium to open the page in: {error}")
        try:
            tab = browser.new_page(viewport={"width": 900, "height": 640})
            tab.set_default_timeout(120_000)
            tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
            tab.on("console", lambda said: problems.append(said.text) if said.type == "error" else None)

            def start() -> None:
                tab.goto(page.as_uri())
                tab.wait_for_function("typeof NV !== 'undefined' && NV.frames > 2")
                tab.evaluate("NV.player.playing = false")  # the controls stay up

            def frames(count: int) -> None:
                tab.wait_for_function("(n) => NV.frames >= n", arg=tab.evaluate("NV.frames") + count)

            def still(mode: str) -> dict[str, float]:
                """The camera on the paused game at 5 seconds, its mode's pose taken at once."""
                tab.evaluate(
                    "(mode) => { NV.player.playing = false; NV.cams.set(mode, 0); NV.player.seek(5);"
                    " NV.cams.cut(); }",
                    mode,
                )
                frames(3)
                return tab.evaluate(CHASE)

            start()
            assert tab.evaluate("NV.cams.settings") == STARTING
            tab.evaluate("NV.ui.dismissHint(); NV.ui.setQuality('low')")  # quicker pictures from here on

            # The panel: from the gear and the View menu, by mouse and keyboard; Escape closes it.
            panel = tab.locator("[data-nv=camPanel]")
            tab.click("[data-nv=camGear]")
            assert panel.is_visible() and tab.get_attribute("[data-nv=camGear]", "aria-expanded") == "true"
            tab.keyboard.press("Escape")
            assert panel.is_hidden() and tab.get_attribute("[data-nv=camGear]", "aria-expanded") == "false"
            tab.click("[data-nv=menuButton]")
            tab.click("[data-nv=camMenu]")
            assert panel.is_visible() and tab.locator("[data-nv=menu]").is_hidden()
            assert tab.locator(".nv-slider span").all_inner_texts() == [
                "Field of View",
                "Distance",
                "Height",
                "Angle",
                "Stiffness",
                "Swivel Speed",
                "Transition Speed",
            ]
            assert tab.locator(".nv-slider output").all_inner_texts() == [
                "110",
                "270",
                "100",
                "-3",
                "0.45",
                "5.0",
                "1.2",
            ]
            ranges = "[...document.querySelectorAll('.nv-slider input')]"
            ranges += ".map((i) => [i.min, i.max, i.step].join(' '))"
            assert tab.evaluate(ranges) == [
                "60 110 1",
                "100 400 10",
                "40 200 10",
                "-15 0 1",
                "0 1 0.05",
                "1 10 0.1",
                "1 2 0.1",
            ]
            tips = tab.evaluate(
                "[...document.querySelectorAll('.nv-camset [data-tip]')].map((n) => n.dataset.tip)"
            )
            assert len(tips) == 10 and all(tip.strip() for tip in tips), "a line for every setting"
            # At 900 by 640 everything in it shows, above the bar and with no scrolling.
            fits = tab.evaluate(
                "() => { const box = document.querySelector('.nv-camset'), at = box.getBoundingClientRect();"
                " const bar = document.querySelector('.nv-bar').getBoundingClientRect();"
                " return [at.top >= 0, at.right <= innerWidth, at.bottom <= bar.top + 1,"
                " box.scrollHeight <= box.clientHeight + 1]; }"
            )
            assert fits == [True, True, True, True]
            tab.focus("input[data-cam=distance]")
            tab.keyboard.press("Escape")
            assert panel.is_hidden(), "Escape closes it from a slider too"
            tab.focus("[data-nv=camGear]")
            tab.keyboard.press("Enter")
            assert tab.evaluate("document.activeElement.dataset.nv") == "camClose"
            tab.keyboard.press("Tab")
            assert tab.evaluate("document.activeElement.dataset.cam") == "fov"
            tab.keyboard.press("ArrowLeft")
            assert tab.evaluate("NV.cams.settings.fov") == 109
            assert tab.inner_text("[data-cam-value=fov]") == "109"
            tab.click("[data-nv=camReset]")
            assert (
                tab.evaluate("NV.cams.settings.fov") == 110
                and tab.input_value("input[data-cam=fov]") == "110"
            )

            # The starting values: 2.7 m behind, 1 m above the car, tilted down 3 degrees, 110 degrees across.
            at = still("car")
            assert at["back"] == pytest.approx(2.7, abs=0.02) and at["up"] == pytest.approx(1.0, abs=0.02)
            assert abs(at["side"]) < 0.02 and at["pitch"] == pytest.approx(-3, abs=0.2)
            assert at["aspect"] == pytest.approx(900 / 640, abs=0.01)
            assert at["fov"] == pytest.approx(vertical_fov(110, at["aspect"]), abs=0.05)
            # Each slider changes the picture at once, paused as it is.
            tab.fill("input[data-cam=distance]", "400")
            tab.fill("input[data-cam=height]", "200")
            tab.fill("input[data-cam=fov]", "80")
            tab.fill("input[data-cam=angle]", "-12")
            frames(3)
            at = tab.evaluate(CHASE)
            assert at["back"] == pytest.approx(4.0, abs=0.02) and at["up"] == pytest.approx(2.0, abs=0.02)
            assert at["fov"] == pytest.approx(vertical_fov(80, at["aspect"]), abs=0.05)
            assert at["pitch"] == pytest.approx(-12, abs=0.2)
            tab.fill("input[data-cam=distance]", "100")
            tab.fill("input[data-cam=height]", "40")
            frames(3)
            at = tab.evaluate(CHASE)
            assert at["back"] == pytest.approx(1.0, abs=0.02) and at["up"] == pytest.approx(0.4, abs=0.02)

            # Stiffness 1 keeps the camera right behind the turning car; 0 lets it trail and swing out.
            def turning(stiffness: float) -> dict[str, float]:
                tab.evaluate(
                    "(s) => { NV.cams.setSetting('stiffness', s); NV.player.setSpeed(1);"
                    " NV.player.playing = true; }",
                    stiffness,
                )
                tab.wait_for_function("NV.player.t > 7")
                frames(4)
                found = tab.evaluate(CHASE)
                tab.evaluate("NV.player.playing = false")
                return found

            tab.click("[data-nv=camReset]")
            at = turning(1)
            assert abs(math.degrees(math.atan2(at["side"], at["back"]))) < 0.5 and at["along"] > 0.9995
            assert at["back"] == pytest.approx(2.7, abs=0.02)
            still("car")
            at = turning(0)
            assert abs(math.degrees(math.atan2(at["side"], at["back"]))) > 4, "the camera trails the turn"
            assert at["along"] < 0.99

            # Ball cam faces the ball even behind the car; with the switch off it faces where the car drives.
            tab.click("[data-nv=camReset]")
            at = still("ball")
            assert at["along"] < -0.9 and at["back"] < 0, "in front of the car, looking back at the ball"
            tab.click("[data-cam-switch=ballLock]")
            assert tab.get_attribute("[data-cam-switch=ballLock]", "aria-checked") == "false"
            frames(15)  # the blend to the new pose
            at = tab.evaluate(CHASE)
            assert at["along"] > 0.99 and at["back"] == pytest.approx(2.7, abs=0.05)

            # Kept: the next page opens with them; the reset is kept as well.
            tab.fill("input[data-cam=distance]", "350")
            tab.fill("input[data-cam=stiffness]", "0.8")
            tab.fill("input[data-cam=swivel]", "7.3")
            tab.fill("input[data-cam=transition]", "1.9")
            assert tab.evaluate("JSON.parse(localStorage.getItem('nitro-viewer')).chase.transition") == 1.9
            start()
            kept = tab.evaluate("NV.cams.settings")
            assert kept == {**STARTING, "distance": 350, "stiffness": 0.8, "swivel": 7.3, "transition": 1.9,
                            "ballLock": False}  # fmt: skip
            assert (
                tab.input_value("input[data-cam=distance]") == "350"
                and tab.evaluate("JSON.parse(localStorage.getItem('nitro-viewer')).chase.distance") == 350
            )
            tab.click("[data-nv=camGear]")
            tab.click("[data-nv=camReset]")
            start()
            assert tab.evaluate("NV.cams.settings") == STARTING
        finally:
            browser.close()
    assert problems == []


def test_a_page_whose_recording_is_in_an_unknown_format_says_so_and_does_not_start(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    newer = {**game(), "format": recording.FORMAT + 1}
    unnamed = {key: value for key, value in game().items() if key != "format"}
    said = {
        "newer": f"a newer NitroLearn (format {recording.FORMAT + 1}). Update NitroLearn to watch it.",
        "unnamed": "in a format this NitroLearn does not read",
    }
    for name, data in (("newer", newer), ("unnamed", unnamed)):
        (tmp_path / f"{name}.html").write_text(viewer.page(data), encoding="utf-8")
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        try:
            browser = play.chromium.launch(args=GRAPHICS)
        except sync_api.Error as error:
            pytest.skip(f"no Chromium to open the page in: {error}")
        try:
            tab = browser.new_page(viewport={"width": 640, "height": 400})
            tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
            tab.on(
                "console", lambda message: problems.append(message.text) if message.type == "error" else None
            )
            for name, words in said.items():
                tab.goto((tmp_path / f"{name}.html").as_uri())
                notice = tab.wait_for_selector(".nv-notice", timeout=60_000)
                assert words in notice.inner_text(), name
                assert tab.evaluate("NV.rec === null && NV.frames === undefined"), (
                    "the viewer was not started"
                )
            tab.wait_for_timeout(9_000)
            assert tab.query_selector(".nv-fail") is None, "no second message over the page's own"
        finally:
            browser.close()
    assert problems == []


def test_the_page_is_dark_and_names_its_encoding_before_anything_else() -> None:
    page = viewer.page(game())
    assert page.index('<meta charset="utf-8">') < 1024, "where a browser looks for it"
    head = page[: page.index("<script>")]
    assert '<meta name="color-scheme" content="dark">' in head
    assert "<style>html, body { background: #141413;" in head, "dark from the first paint"
    assert page.index("window.NV_GUARD") < page.index("var THREE="), "the guard runs before any other script"


def open_page(sync_api: Any, play: Any, args: list[str], **context: Any) -> tuple[Any, Any]:
    """A browser launched with `args` and a tab in it, 1280 by 720."""
    try:
        browser = play.chromium.launch(args=args)
    except sync_api.Error as error:
        pytest.skip(f"no Chromium to open the page in: {error}")
    return browser, browser.new_context(viewport={"width": 1280, "height": 720}, **context).new_page()


ASPHALT = "rgb(20, 20, 19)"


def test_a_page_whose_scripts_never_run_is_dark_and_says_why(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    page = tmp_path / "game.html"
    page.write_text(viewer.page(game()), encoding="utf-8")
    with sync_api.sync_playwright() as play:
        browser, tab = open_page(sync_api, play, GRAPHICS, java_script_enabled=False)
        try:
            tab.goto(page.as_uri())
            assert tab.evaluate("getComputedStyle(document.body).backgroundColor") == ASPHALT
            assert "JavaScript is turned off" in tab.inner_text(".nv-notice")
        finally:
            browser.close()


@pytest.mark.parametrize(
    ("case", "args", "words"),
    [
        ("no graphics", ["--disable-gpu", "--disable-software-rasterizer"], "cannot draw 3D graphics"),
        ("WebGL turned off", ["--disable-webgl"], "cannot draw 3D graphics"),
        ("an error at start", GRAPHICS, "stopped with an error"),
        ("a script that does not parse", GRAPHICS, "stopped with an error"),
        ("no first picture", GRAPHICS, "first picture was not drawn within 8 seconds"),
    ],
)
def test_a_view_that_cannot_start_says_so_on_a_dark_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, args: list[str], words: str
) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    own = viewer.script()
    if case == "an error at start":
        monkeypatch.setattr(viewer, "script", lambda: 'throw new Error("a test failure");\n' + own)
    elif case == "a script that does not parse":
        monkeypatch.setattr(viewer, "script", lambda: own + "\n}}")
    page = tmp_path / "game.html"
    page.write_text(viewer.page(played_game()), encoding="utf-8")
    with sync_api.sync_playwright() as play:
        browser, tab = open_page(sync_api, play, args)
        try:
            if case == "no first picture":
                tab.add_init_script("window.requestAnimationFrame = () => 0;")
            tab.goto(page.as_uri())
            message = tab.wait_for_selector(".nv-fail", timeout=60_000)
            said = message.inner_text()
            assert "The 3D view could not start" in said and words in said, said
            assert "turn on hardware acceleration" in said
            if case == "an error at start":
                assert "a test failure" in said, "the technical reason, in small type"
            if case == "a script that does not parse":
                assert "SyntaxError" in said
            look = message.evaluate(
                "(box) => { const style = getComputedStyle(box), at = box.getBoundingClientRect();"
                " const top = document.elementFromPoint(640, 360);"
                " return [style.backgroundColor, style.color, at.width, at.height, box.contains(top)]; }"
            )
            assert look[:2] == [ASPHALT, "rgb(243, 241, 236)"], "light words on the dark page"
            assert look[2:] == [1280, 720, True], "over the whole page, on top"
        finally:
            browser.close()


def test_a_broken_script_beside_the_page_does_not_stop_it(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    (tmp_path / "arena.js").write_text("window.NV_ARENA = ;", encoding="utf-8")
    page = tmp_path / "game.html"
    data = {**played_game(), "subtitle": "1v1, played here"}
    page.write_text(viewer.page(data, arena_src="arena.js"), encoding="utf-8")
    with sync_api.sync_playwright() as play:
        browser, tab = open_page(sync_api, play, GRAPHICS)
        try:
            # A browser that keeps no site data refuses the page its storage.
            tab.add_init_script(
                "Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('no', "
                "'SecurityError'); } });"
            )
            tab.goto(page.as_uri())
            tab.wait_for_function("typeof NV !== 'undefined' && NV.frames > 2", timeout=120_000)
            assert tab.query_selector(".nv-fail") is None, "the generated arena, and no message"
            # The title and the line under it on a flat dark panel of their own, square like the others.
            backing = tab.evaluate(
                "() => { const row = getComputedStyle(document.querySelector('.nv-title'));"
                " return [row.backgroundColor, row.borderRadius, row.display]; }"
            )
            assert backing == ["rgba(20, 20, 19, 0.86)", "0px", "flex"]
            assert tab.inner_text(".nv-subtitle") == "1v1, played here"
        finally:
            browser.close()


def test_a_page_opened_behind_other_tabs_waits_without_a_message(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    page = tmp_path / "game.html"
    page.write_text(viewer.page(played_game()), encoding="utf-8")
    with sync_api.sync_playwright() as play:
        browser, tab = open_page(sync_api, play, GRAPHICS)
        try:
            # Out of view the browser draws nothing: the time the first picture may take has not begun.
            tab.add_init_script(
                "Object.defineProperty(document, 'visibilityState', { get: () => 'hidden' });"
                " window.requestAnimationFrame = () => 0;"
            )
            tab.goto(page.as_uri())
            tab.wait_for_timeout(9_000)
            assert tab.query_selector(".nv-fail") is None
        finally:
            browser.close()
