"""`nitro watch`: a recorded game captured every few ticks with the bot's own figures, the match viewer's
page it becomes, and what the command writes."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nitro import viewer
from nitro.cli import main
from nitro.config import NitroConfig, default_config
from nitro.envs import EnvPool, probe_config
from nitro.learner import Learner
from nitro.model import build_networks
from nitro.runs import RunFolder, fixed_settings
from nitro.training import build
from nitro.viewer import recording as layouts
from nitro.watch import CAR, HEAD, LAYOUT, Recording, attention_targets, page, record
from test_viewer import GRAPHICS

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
# Where a car's columns sit in its block (nitro_core::capture).
POS, VEL, FORWARD, UP, SPIN, BOOST, FLAGS, CONTROLS, STATE, HANDBRAKE, BODY, RESPAWN = (
    slice(1, 4),
    slice(4, 7),
    slice(7, 10),
    slice(10, 13),
    slice(13, 16),
    16,
    17,
    slice(18, 26),
    26,
    27,
    28,
    29,
)


def fake(frames: int = 5, cars: int = 2) -> Recording:
    """A made-up game in the core's layout: every column within what a game holds."""
    rng = np.random.default_rng(1)
    data = np.zeros((frames, HEAD + cars * CAR), dtype=np.float32)
    data[:, 1:4] = rng.uniform(-3000, 3000, (frames, 3))  # the ball's place and velocity
    data[:, 4:7] = rng.uniform(-3000, 3000, (frames, 3))
    data[:, 7], data[:, 12] = 1.0, 1.0  # its forward and up
    data[:, 18:HEAD] = rng.uniform(0, 10, (frames, HEAD - 18))  # pad waits
    for c in range(cars):
        car = data[:, HEAD + c * CAR : HEAD + (c + 1) * CAR]
        car[:, 0] = c % 2
        car[:, POS] = rng.uniform(-4000, 4000, (frames, 3))
        car[:, VEL] = rng.uniform(-2300, 2300, (frames, 3))
        car[:, FORWARD] = rng.uniform(-1, 1, (frames, 3))
        car[:, UP] = rng.uniform(-1, 1, (frames, 3))
        car[:, SPIN] = rng.uniform(-5.5, 5.5, (frames, 3))
        car[:, BOOST] = rng.uniform(0, 100, frames)
        car[:, FLAGS] = 65
        car[:, CONTROLS] = rng.uniform(-1, 1, (frames, 8))
        car[:, STATE] = 15
    data[0, 0] = 1.0
    pads = [[0.0, -4240.0, 70.0, 0.0]] * 34
    decisions = [[0, [0.5, 0.9, 0.2, 3], [-0.25, 0.4, 0.7, 11]]]
    return Recording(data, pads, every=2, tick_skip=8, decisions=decisions)


def shown(text: str) -> tuple[dict[str, Any], np.ndarray]:
    """The data inside a page and its frames as real numbers again."""
    data = layouts.from_page(text)
    stored = layouts.unpack(data["frames"], data["head"] + data["cars"] * data["car"])
    return data, stored / np.asarray(data["scales"])


def tiny_config(**env: Any) -> NitroConfig:
    base = default_config(Path.cwd())
    return dataclasses.replace(
        base,
        paths=dataclasses.replace(base.paths, collision_meshes=MESHES),
        run=dataclasses.replace(base.run, device="cpu"),
        env=dataclasses.replace(base.env, num_arenas=2, threads=2, **({"tick_skip": 8} | env)),
    )


def saved_run(tmp_path: Path, *steps: int) -> RunFolder:
    """A run with a small network of random weights saved at each of `steps` agent steps."""
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nnum_arenas = 2\nthreads = 2\ntick_skip = 8\n[model]\npolicy = [16]\nvalue = [16]\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    for agent_steps in steps:
        save(folder, agent_steps)
    return folder


def save(folder: RunFolder, agent_steps: int) -> None:
    config = folder.load_settings()
    probe = EnvPool(probe_config(config, 1), pin_memory=False).groups[0]
    torch.manual_seed(agent_steps)
    model = build_networks(config, probe.obs.shape[1], probe.mask.shape[1])
    state = {
        "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": agent_steps,
        "iterations": 1,
        "settings": fixed_settings(config),
    }
    folder.save_checkpoint(state, agent_steps, keep=5, milestone_steps=0)


def test_the_page_carries_the_game_to_within_its_rounding() -> None:
    recording = fake()
    text = page(recording, "my <run>", "2v2 & more")
    assert "<title>my &lt;run&gt;</title>" in text and "/*DATA*/" not in text
    data, decoded = shown(text)
    assert (data["format"], data["cars"], data["head"], data["car"], data["fps"]) == (
        LAYOUT,
        2,
        HEAD,
        CAR,
        60.0,
    )
    assert data["subtitle"] == "2v2 & more" and "follow" not in data and data["decisionsPerSecond"] == 15.0
    assert decoded.shape == recording.frames.shape
    scales = np.asarray(data["scales"])
    assert np.all(np.abs(decoded - recording.frames) <= 0.5 / scales + 1e-6), "every column to its rounding"
    places = [1, 2, 3] + [HEAD + c * CAR + k for c in range(2) for k in (1, 2, 3)]
    np.testing.assert_allclose(decoded[:, places], recording.frames[:, places], atol=0.126)  # a quarter unit
    directions = [HEAD + c * CAR + k for c in range(2) for k in range(7, 13)]
    np.testing.assert_allclose(decoded[:, directions], recording.frames[:, directions], atol=1e-4)
    assert data["decisions"] == recording.decisions and len(data["pads"]) == 34


def test_nothing_in_the_data_can_end_the_script_early() -> None:
    plain = page(fake(), "a title")
    text = page(fake(), "</script><script>alert(1)</script>")
    assert "</script><script>alert(1)" not in text
    assert text.count("</script>") == plain.count("</script>"), "only the page's own closing tags"
    assert layouts.from_page(text)["title"] == "</script><script>alert(1)</script>"


def test_values_beyond_sixteen_bits_are_held_at_the_limit_not_wrapped() -> None:
    recording = fake(frames=3, cars=1)
    recording.frames[1, 1] = 50_000.0  # a place no arena holds, at a quarter unit a number
    recording.frames[2, 1] = -50_000.0
    data = layouts.from_page(page(recording, "far"))
    stored = layouts.unpack(data["frames"], HEAD + CAR)
    assert stored[1, 1] == 32767 and stored[2, 1] == -32768


def test_the_mechanics_marks_reach_the_page_with_the_other_flags() -> None:
    recording = fake(cars=1)
    every_mark = 64 | 0xFF << 7  # touched, and all eight mechanics (nitro_core::capture::FLAG_MARKS_SHIFT)
    recording.frames[2, HEAD + FLAGS] = every_mark
    data = layouts.from_page(page(recording, "marks"))
    stored = layouts.unpack(data["frames"], HEAD + CAR)
    assert stored[2, HEAD + FLAGS] == every_mark and stored[1, HEAD + FLAGS] == 65
    assert "marksShift: 7" in viewer.script(), "the page reads the marks from the bit the core writes them at"


def test_a_page_that_follows_a_run_says_so_and_loads_itself_again_at_the_end() -> None:
    assert layouts.from_page(page(fake(), "newest", follow=True))["follow"] is True
    assert "data.follow" in viewer.script() and "location.reload()" in viewer.script()


def test_the_arena_stays_out_of_a_page_that_names_its_script(tmp_path: Path) -> None:
    text = page(fake(), "beside", arena_src="../.viewer/arena.js")
    assert '<script src="../.viewer/arena.js"></script>' in text and "NV_ARENA=" not in text
    assert "NV_ARENA=" in page(fake(), "inside", arena="AAAA")
    assert "<script src=" not in page(fake(), "none")
    assert viewer.keep_arena_script(tmp_path / "shared", tmp_path / "no-meshes") is None
    assert not (tmp_path / "shared").exists(), "nothing is written without meshes"


def test_the_arena_script_is_written_once_and_again_only_when_it_differs(tmp_path: Path) -> None:
    soccar = tmp_path / "meshes" / "soccar"
    soccar.mkdir(parents=True)
    (soccar / "mesh_0.cmf").write_bytes(b"\x01\x02\x03")
    file = viewer.keep_arena_script(tmp_path / "shared", tmp_path / "meshes")
    assert file == tmp_path / "shared" / "arena.js"
    blob = viewer.arena_blob(tmp_path / "meshes")
    assert blob is not None and file.read_text(encoding="utf-8") == viewer.arena_script(blob)
    os.utime(file, ns=(1_000_000_000, 1_000_000_000))
    assert viewer.keep_arena_script(tmp_path / "shared", tmp_path / "meshes") == file
    assert file.stat().st_mtime_ns == 1_000_000_000, "the same meshes: the file is left alone"
    (soccar / "mesh_0.cmf").write_bytes(b"\x09")
    viewer.keep_arena_script(tmp_path / "shared", tmp_path / "meshes")
    other = viewer.arena_blob(tmp_path / "meshes")
    assert other is not None and other != blob, "other meshes"
    assert file.read_text(encoding="utf-8") == viewer.arena_script(other), "the file follows them"
    assert not list((tmp_path / "shared").glob("*.part"))


@NEEDS_MESHES
def test_a_recorded_game_is_captured_every_few_ticks_with_the_bots_figures() -> None:
    config = tiny_config()
    state = {"learner": {"model": build(config).model.state_dict()}}
    recording = record(config, state, seconds=2.0, seed=3, fps=60)
    frames = recording.frames
    assert recording.cars == 2 and len(recording.pads) == 34
    assert frames.shape == (1 + 2 * 60, HEAD + 2 * CAR), "the start, then every 2 ticks for 2 s"
    assert frames[0, 0] == 1.0, "the first frame starts an episode"
    assert [frames[0, HEAD + c * CAR] for c in range(2)] == [0.0, 1.0], "Blue, then Orange"
    assert np.allclose(frames[0, 1:3], 0.0), "a kickoff: the ball at centre"
    assert np.allclose(np.linalg.norm(frames[:, 7:10], axis=1), 1, atol=1e-3), "the ball's forward"
    assert np.all(frames[:, 16:18] >= 0) and np.all(frames[:, 18:HEAD] >= 0), "score and pad waits"
    for c in range(2):
        car = frames[:, HEAD + c * CAR : HEAD + (c + 1) * CAR]
        assert np.allclose(np.linalg.norm(car[:, FORWARD], axis=1), 1, atol=1e-3)
        assert np.allclose(np.linalg.norm(car[:, UP], axis=1), 1, atol=1e-3)
        assert np.all(np.abs(car[:, POS]) < 6000) and np.all((car[:, BOOST] >= 0) & (car[:, BOOST] <= 100))
        assert np.all(np.abs(car[:, CONTROLS]) <= 1) and np.all(car[:, BODY] == 0)
        assert np.all(
            (car[:, STATE] >= 0) & (car[:, STATE] < 128) & (car[:, STATE] == np.rint(car[:, STATE]))
        )
        assert np.all((car[:, HANDBRAKE] >= 0) & (car[:, HANDBRAKE] <= 1)) and np.all(car[:, RESPAWN] >= 0)
    assert len(recording.decisions) == 30 and recording.decisions[0][0] == 0
    assert [d[0] for d in recording.decisions[:3]] == [0, 4, 8], (
        "each decision looks at its step's last frame"
    )
    assert all(0 <= car[1] <= 1 and 0 <= car[2] <= 1.01 for d in recording.decisions for car in d[1:])
    again = record(config, state, seconds=2.0, seed=3, fps=60)
    np.testing.assert_array_equal(again.frames, frames)
    assert again.decisions == recording.decisions, "the same seed plays the same game"
    data, decoded = shown(page(recording, "a real game"))
    assert data["format"] == LAYOUT and decoded.shape == frames.shape


def test_an_attention_networks_shares_reach_the_page() -> None:
    recording = fake()
    recording.decisions = [[0, [0.5, 0.9, 0.2, 3, [0.3, 0.7]], [-0.25, 0.4, 0.7, 11, [1.0, 0.0]]]]
    recording.attention = {
        "labels": ["nothing", "opponent"],
        "talk": False,
        "targets": [["nothing", 1], ["nothing", 0]],
    }
    data = layouts.from_page(page(recording, "attention"))
    assert data["attention"] == recording.attention
    assert data["decisions"][0][1][4] == [0.3, 0.7]
    assert layouts.from_page(page(fake(), "plain"))["attention"] is None


@NEEDS_MESHES
def test_a_recorded_game_with_an_attention_network_keeps_what_each_car_looked_at() -> None:
    base = tiny_config(team_size=2, obs_team_size=2)
    config = dataclasses.replace(
        base,
        model=dataclasses.replace(
            base.model,
            encoder="attention",
            attention_width=32,
            policy=(32,),
            value=(32,),
            attention_fast=True,
        ),
    )
    state = {"learner": {"model": build(config).model.state_dict()}}
    recording = record(config, state, seconds=1.0, seed=3, fps=30)
    assert recording.attention is not None
    assert recording.attention["labels"] == ["nothing", "teammate", "opponent", "opponent"]
    assert recording.attention["talk"] is False and len(recording.attention["targets"]) == 4
    for decision in recording.decisions:
        for car in decision[1:]:
            shares = "shares of the null car and 3 others"
            assert len(car[4]) == 4 and abs(sum(car[4]) - 1) <= 0.005 * 4 + 1e-9, shares


@NEEDS_MESHES
def test_a_talking_networks_game_keeps_who_listened_to_whom() -> None:
    base = tiny_config(team_size=2, obs_team_size=2)
    config = dataclasses.replace(
        base,
        model=dataclasses.replace(
            base.model, encoder="attention_communication", attention_width=32, policy=(32,), value=(32,)
        ),
    )
    state = {"learner": {"model": build(config).model.state_dict()}}
    recording = record(config, state, seconds=1.0, seed=3, fps=30)
    labels = ["nothing", "ball", "itself", "teammate", "opponent", "opponent"]
    assert recording.attention is not None
    assert recording.attention["labels"] == labels and recording.attention["talk"] is True
    for decision in recording.decisions:
        for car in decision[1:]:
            assert len(car[4]) == 6 and abs(sum(car[4]) - 1) <= 0.005 * 6 + 1e-9
            assert len(car[5]) == 5, "every token but the null one names the one it heard most"
            for k, (token, share) in enumerate(car[5], start=1):
                assert token not in (0, k) and 0 <= share <= 1
    data = layouts.from_page(page(recording, "talk"))
    assert data["attention"]["talk"] is True and data["decisions"][0][1][5] == recording.decisions[0][1][5]


def test_the_attention_places_name_the_cars_in_the_cores_order() -> None:
    labels = ["nothing", "ball", "itself", "teammate", "teammate", "opponent", "opponent", "opponent"]
    interleaved = [0, 1, 0, 1, 0, 1]  # RocketSim's 3v3: Blue, Orange, Blue, ...
    assert attention_targets(labels, interleaved, 0) == ["nothing", "ball", "itself", 2, 4, 1, 3, 5]
    assert attention_targets(labels, interleaved, 3) == ["nothing", "ball", "itself", 1, 5, 0, 2, 4]
    two_a_side = [0, 1, 0, 1]
    assert attention_targets(labels, two_a_side, 1) == ["nothing", "ball", "itself", 3, None, 0, 2, None]


def test_the_command_explains_what_it_cannot_watch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert main(["watch", "ghost", "--runs", str(runs)]) == 1
    assert "no run called 'ghost'" in capsys.readouterr().err
    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n', encoding="utf-8")
    RunFolder.create(runs, "my-bot", settings)
    for mode in ([], ["--live"], ["--feed"]):
        assert main(["watch", "my-bot", "--runs", str(runs), *mode]) == 1
        assert "no save yet" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["watch", "my-bot", "--runs", str(runs), "--vs-bot", "somewhere"])
    assert "unrecognized arguments: --vs-bot" in capsys.readouterr().err


def test_the_help_tells_what_live_does_and_keeps_the_feed_to_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["watch", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--live" in text and "a window of its own" in text and "nothing is served on any port" in text
    assert "--feed" not in text and "127.0.0.1" not in text and "--vs-bot" not in text


@NEEDS_MESHES
def test_the_command_writes_the_page_the_recording_and_the_arena_beside_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = saved_run(tmp_path, 500)
    runs = folder.root.parent
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    assert main(["watch", "my-bot", "--runs", str(runs), "--seconds", "2", "--fps", "30"]) == 0
    out = folder.root / "watch.html"
    assert f"Wrote {out}" in capsys.readouterr().out and opened == [out.resolve().as_uri()]
    text = out.read_text(encoding="utf-8")
    blob = viewer.arena_blob(MESHES)
    assert blob is not None
    assert '<script src="../.viewer/arena.js"></script>' in text, "the page names the arena's script"
    assert "NV_ARENA=" not in text and blob[:200] not in text, "and holds no mesh"
    arena = runs / ".viewer" / "arena.js"
    assert arena.read_text(encoding="utf-8") == viewer.arena_script(blob)
    data = layouts.from_page(text)
    assert data["title"] == "my-bot at 500 agent steps" and data["subtitle"].startswith("1v1, 2 seconds")
    assert (data["format"], data["cars"], data["fps"]) == (LAYOUT, 2, 30.0) and "follow" not in data
    frames = layouts.unpack(data["frames"], HEAD + 2 * CAR)
    assert len(frames) == 1 + 2 * 30 and len(data["decisions"]) == 30
    assert layouts.load(folder.root / "watch.nlrec") == data, "the recording itself, for other tools"
    assert not list(folder.root.glob("*.part*")), "no unfinished file is left"

    os.utime(arena, ns=(1_000_000_000, 1_000_000_000))
    assert main(["watch", "my-bot", "--runs", str(runs), "--seconds", "1", "--seed", "4", "--no-open"]) == 0
    assert len(opened) == 1, "--no-open"
    assert arena.stat().st_mtime_ns == 1_000_000_000, "the arena's script is written once"
    other = layouts.load(out)
    assert other["frames"] != data["frames"] and "seed 4" in other["subtitle"]

    save(folder, 900)
    assert main(["watch", "my-bot", "--runs", str(runs), "--seconds", "1", "--no-open", "--at", "500"]) == 0
    assert layouts.load(out)["title"] == "my-bot at 500 agent steps"
    assert (
        main(["watch", "my-bot", "--runs", str(runs), "--seconds", "1", "--no-open", "--team-size", "2"]) == 1
    )
    assert "cannot play 2v2" in capsys.readouterr().err


@NEEDS_MESHES
def test_following_a_run_writes_a_page_that_follows(tmp_path: Path) -> None:
    from argparse import Namespace

    from nitro.cli import _write_watch

    folder = saved_run(tmp_path, 500)
    state = folder.load_latest()
    assert state is not None
    args = Namespace(seconds=1.0, seed=0, fps=30, team_size=None)
    data = layouts.load(_write_watch(folder, state, args, follow=True))
    assert data["follow"] is True and "newest save at the end of each game" in data["subtitle"]
    assert "follow" not in layouts.load(folder.root / "watch.nlrec"), "a property of the page, not the game"


def open_tab(browser: Any, problems: list[str]) -> Any:
    tab = browser.new_page(viewport={"width": 640, "height": 400})
    tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
    tab.on("console", lambda said: problems.append(said.text) if said.type == "error" else None)
    return tab


def chromium(sync_api: Any, play: Any) -> Any:
    try:
        return play.chromium.launch(args=GRAPHICS)
    except sync_api.Error as error:
        pytest.skip(f"no Chromium to open the page in: {error}")


@NEEDS_MESHES
def test_the_page_of_a_real_game_is_drawn_with_its_arena_and_without(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    folder = saved_run(tmp_path, 500)
    runs = folder.root.parent
    assert main(["watch", "my-bot", "--runs", str(runs), "--seconds", "3", "--no-open"]) == 0
    out = folder.root / "watch.html"
    started = "typeof NV !== 'undefined' && NV.frames > 2"
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        try:
            tab = open_tab(browser, problems)
            tab.goto(out.as_uri())
            tab.wait_for_function(started, timeout=120_000)
            assert tab.evaluate("!!window.NV_ARENA"), "the arena comes from the script beside the runs"
            assert tab.evaluate("[NV.rec.format, NV.rec.cars, NV.rec.count, NV.rec.decisions.length]") == [
                LAYOUT,
                2,
                181,
                45,
            ]
            assert tab.evaluate("NV.rec.title") == "my-bot at 500 agent steps"
            kickoff = tab.evaluate("NV.rec.sample(0, NV.data.newWorld(NV.rec)).cars.map((car) => car.pos.z)")
            assert kickoff[0] < 0 < kickoff[1], "Blue starts on its own side, Orange on the other"
            tab.evaluate("NV.player.setSpeed(4)")
            drawn = tab.evaluate("NV.frames")
            tab.wait_for_function("(n) => NV.frames >= n", arg=drawn + 6, timeout=120_000)
            assert problems == []

            # The page passed on, or the shared file gone: the generated arena, and the game all the same.
            (runs / ".viewer" / "arena.js").unlink()
            tab.reload()
            tab.wait_for_function(started, timeout=120_000)
            assert not tab.evaluate("!!window.NV_ARENA") and tab.evaluate("NV.rec.count") == 181
            drawn = tab.evaluate("NV.frames")
            tab.wait_for_function("(n) => NV.frames >= n", arg=drawn + 6, timeout=120_000)
        finally:
            browser.close()
    assert not [problem for problem in problems if "page error" in problem]
    assert all("ERR_FILE_NOT_FOUND" in problem for problem in problems), "only the missing file is reported"


def test_a_page_that_follows_loads_itself_again_when_its_game_ends(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    file = tmp_path / "watch.html"
    file.write_text(page(fake(frames=20), "first", follow=True), encoding="utf-8")
    started = "typeof NV !== 'undefined' && NV.frames > 2"
    problems: list[str] = []
    with sync_api.sync_playwright() as play:
        browser = chromium(sync_api, play)
        try:
            tab = open_tab(browser, problems)
            tab.goto(file.as_uri())
            tab.wait_for_function(f"{started} && NV.rec.title === 'first'", timeout=120_000)
            file.write_text(page(fake(frames=20), "second", follow=True), encoding="utf-8")
            tab.wait_for_function(f"{started} && NV.rec.title === 'second'", timeout=120_000)
        finally:
            browser.close()
    assert problems == []
