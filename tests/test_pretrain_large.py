"""The large replay mode (`pretrain.batch_replays`): replays become compact samples, several
at once, batch by batch, the copy learns from each batch with the ones kept before it, and a stopped copy
carries on from its last round. Built with replicar-style files and a stand-in for replicar's tool, as
tests/test_replays.py does."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import struct
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("pyarrow")

from nitro import _core, pretrain, replays
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig, PretrainConfig, WandbConfig
from nitro.runs import RunFolder
from test_replays import random_arrays, replay_folder, replicar_file, replicar_match, stand_in_tool
from test_tracking import FakeRun

SHELL = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")


@pytest.fixture
def charts(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A stand-in for the wandb package that records what is sent (as tests/test_tracking.py's)."""
    monkeypatch.setenv("WANDB_API_KEY", "a-key")  # logged in
    calls: dict[str, Any] = {"init": [], "define": [], "log": [], "finished": 0}
    module = types.ModuleType("wandb")
    module.init = lambda **kwargs: FakeRun(calls, **kwargs)  # type: ignore[attr-defined]
    module.Settings = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    return calls


SETTINGS = '[run]\ndevice = "cpu"\n[model]\npolicy = [32]\nvalue = [32]\n'


def numpy_compact(part: dict[str, Any], actions: int) -> tuple[Any, Any, Any, Any]:
    """The compact form as the first version made it with NumPy, which the core's must equal bit for bit."""
    count, obs_size = int(part["count"]), int(part["obs_size"])
    obs = np.frombuffer(part["obs"], dtype="<f4").reshape(count, obs_size).astype(np.float16)
    mask = np.frombuffer(part["mask"], dtype=np.uint8).reshape(count, actions) != 0
    packed = np.packbits(mask, axis=1, bitorder="little")
    label = np.frombuffer(part["label"], dtype="<f4").reshape(count, actions)
    index = np.argsort(-label, axis=1, kind="stable")[:, : replays.LABEL_ENTRIES]
    share = np.take_along_axis(label, index, axis=1).astype(np.float64)
    share /= np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
    return obs, packed, index.astype(np.uint8), share.astype(np.float16)


def settings_of(team_size: int, replicar_rows: bool = True) -> str:
    """The copy's core settings for a game size: of replicar's rows, as the large mode's jobs make them, or
    of rows that hold the controls that produced them."""
    text = SETTINGS + f"[env]\nteam_size = {team_size}\n"
    config = parse_toml(text, NitroConfig, file=Path("s.toml"), base_dir=Path())
    return pretrain.copy_settings(config, replicar_rows=replicar_rows)


def same_bits(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return all(a.shape == b.shape and a.tobytes() == b.tobytes() for a, b in zip(left, right, strict=True))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_the_core_makes_the_samples_of_the_recording_path_bit_for_bit(seed: int) -> None:
    per_side, optional = 1 + seed % 3, seed < 3
    arrays = random_arrays(seed, rows=600, per_side=per_side, optional=optional, gaps=0.1)
    players = [{"index": k, "team": (k + seed) % 2} for k in range(2 * per_side)]
    settings = settings_of(per_side)
    made = replays.recording(arrays, players)
    part = _core.demonstrations_bytes(made.data, "x", settings, 0.1)
    fast = _core.replay_samples(*replays.core_columns(arrays, players), "x", settings, 0.1)
    assert part["count"] > 20, "samples to compare"
    assert (fast["count"], fast["skipped"], fast["kickoffs"]) == (
        part["count"],
        part["skipped"],
        made.kickoffs,
    )
    assert (fast["ticks"], fast["rows_left_out"]) == (made.ticks, made.rows_left_out)
    assert same_bits(replays.compact_arrays(fast), numpy_compact(part, _core.num_actions()))


def test_replicars_match_gives_the_same_corrected_samples_by_both_paths() -> None:
    """Replicar's rows as a recording and straight from its columns, with the jump, the dodge and the
    row written twice of tests/test_replays.py's match."""
    arrays, players = replicar_match()
    settings = settings_of(1)
    part = _core.demonstrations_bytes(replays.recording(arrays, players).data, "x", settings, 0.1)
    fast = _core.replay_samples(*replays.core_columns(arrays, players), "x", settings, 0.1)
    assert fast["count"] == part["count"] > 30 and fast["skipped"] == part["skipped"] == 0
    assert same_bits(replays.compact_arrays(fast), numpy_compact(part, _core.num_actions()))


@pytest.mark.parametrize("replicar_rows", [False, True])
def test_a_kept_recording_gives_the_same_compact_samples(tmp_path: Path, replicar_rows: bool) -> None:
    arrays = random_arrays(7, rows=300, per_side=2, gaps=0.1)
    players = [{"index": k, "team": k % 2} for k in range(4)]
    file = tmp_path / "kept.rlpr.zst"
    file.write_bytes(replays.compress(replays.recording(arrays, players).data))
    settings = settings_of(2, replicar_rows)
    full = _core.demonstrations(str(file), settings, 0.1)
    compact = _core.demonstrations(str(file), settings, 0.1, compact=True)
    assert compact["count"] == full["count"] > 20
    assert same_bits(replays.compact_arrays(compact), numpy_compact(full, _core.num_actions()))


def test_without_a_kickoff_the_core_gives_no_samples_and_bad_columns_are_refused() -> None:
    arrays = random_arrays(0, rows=200, per_side=1)
    arrays["car_is_on_ground"][:] = 0  # no car on its spot on the ground: no kickoff
    players = [{"index": 0, "team": 0}, {"index": 1, "team": 1}]
    columns = replays.core_columns(arrays, players)
    none = _core.replay_samples(*columns, "x", settings_of(1), 0.1)
    assert (none["kickoffs"], none["count"]) == (0, 0) and none["ticks"] > 0
    assert replays.compact_arrays(none)[0].shape == (0, none["obs_size"])
    with pytest.raises(ValueError, match="car values hold"):
        _core.replay_samples(columns[0], *columns[1:3], columns[3][:-4], *columns[4:], "x", settings_of(1))
    arrays["sim_tick"] = arrays["sim_tick"] - 10_000  # negative tick numbers
    with pytest.raises(ValueError, match="tick number"):
        _core.replay_samples(*replays.core_columns(arrays, players), "x", settings_of(1))
    del arrays["car_jump_ticks"]
    with pytest.raises(ValueError, match="no jump_ticks column"):
        replays.core_columns(arrays, players)


def test_compact_samples_keep_the_mask_exactly_and_the_label_to_rounding() -> None:
    rng = np.random.default_rng(0)
    actions, count, obs_size = _core.num_actions(), 50, 12
    mask = rng.random((count, actions)) < 0.6
    mask[:, 0] = True
    label = np.where(mask, rng.random((count, actions)) ** 30, 0.0)  # a few large shares, many tiny ones
    label /= label.sum(axis=1, keepdims=True)
    part = {
        "count": count,
        "obs_size": obs_size,
        "obs": rng.normal(size=(count, obs_size)).astype("<f4").tobytes(),
        "mask": mask.astype(np.uint8).tobytes(),
        "label": label.astype("<f4").tobytes(),
    }
    obs, packed, index, share = numpy_compact(part, actions)
    assert obs.dtype == np.float16 and packed.shape == (count, (actions + 7) // 8)
    rows = torch.arange(count)
    tensors = {
        "obs": torch.from_numpy(obs),
        "mask": torch.from_numpy(packed),
        "label_index": torch.from_numpy(index),
        "label_share": torch.from_numpy(share),
    }
    got_obs, got_mask, got_label = pretrain._unpacked(tensors, rows, actions, torch.device("cpu"))
    assert torch.equal(got_mask, torch.from_numpy(mask)), "the mask exactly"
    want = torch.from_numpy(np.frombuffer(part["obs"], "<f4").reshape(count, obs_size).copy())
    torch.testing.assert_close(got_obs, want, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_label.sum(1), torch.ones(count), rtol=2e-3, atol=2e-3)
    assert torch.equal(torch.from_numpy(index[:, 0]).long(), torch.from_numpy(label.argmax(1))), (
        "first: argmax"
    )
    top = np.argsort(-label, axis=1)[:, : replays.LABEL_ENTRIES]
    expected = np.zeros_like(label)
    np.put_along_axis(expected, top, np.take_along_axis(label, top, axis=1), axis=1)
    expected /= expected.sum(axis=1, keepdims=True)
    torch.testing.assert_close(got_label, torch.from_numpy(expected).float(), rtol=2e-3, atol=2e-3)


def stand_in_lean_tool(folder: Path) -> Path:
    """nitro-replicar's stand-in, as `stand_in_tool` is replicar's: it copies the prepared files as
    nitro-replicar writes them (`lean_file`) and logs to the same calls.log."""
    lean = folder / "lean"
    lean.mkdir(exist_ok=True)
    for name in ("one", "two"):
        lean_file(lean / f"{name}.lean", *replays.read_replicar(folder / f"{name}.parquet"))
    tool = stand_in_tool(lean)
    tool.write_text(
        tool.read_text(encoding="utf-8")
        .replace(".parquet", ".lean")
        .replace(f"{lean}/calls.log", f"{folder}/calls.log"),
        encoding="utf-8",
    )
    return tool


def tools_and_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: list[str]) -> Path:
    """The stand-in replicar and a folder of replays named `names` (one named 'broken' fails, 'two' is a
    2v2); the project's replays folder moved into `tmp_path`."""
    tools = tmp_path / "tools"
    tools.mkdir()
    replicar_file(tools / "one.parquet")
    replicar_file(tools / "two.parquet", per_side=2)
    monkeypatch.setattr(replays, "replicar_tool", lambda: stand_in_tool(tools))
    monkeypatch.setattr(replays, "nitro_replicar_tool", lambda: stand_in_lean_tool(tools))
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    (tmp_path / "meshes" / "soccar").mkdir(parents=True)
    return replay_folder(tmp_path, names)


def large(tmp_path: Path, demos: Path, **changes: Any) -> NitroConfig:
    config = parse_toml(SETTINGS, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    chosen = PretrainConfig(
        enabled=True,
        replays=(demos,),
        limit=4,
        passes=2,
        batch_replays=2,
        keep_batches=2,
        workers=2,
        minibatch=32,  # a few hundred decisions in all
    )
    paths = dataclasses.replace(config.paths, collision_meshes=tmp_path / "meshes")
    return dataclasses.replace(config, paths=paths, pretrain=dataclasses.replace(chosen, **changes))


@SHELL
def test_one_replay_is_made_ready_in_a_worker_or_says_why_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "broken", "two"])
    config = large(tmp_path, demos)
    settings = pretrain.copy_settings(config, replicar_rows=True)  # as the large mode's jobs
    work = tmp_path / "work"
    work.mkdir()

    def job(name: str) -> replays.Job:
        return replays.Job(
            0,
            demos / f"{name}.replay",
            (1,),
            replays.replicar_tool(),
            tmp_path,
            work,
            tmp_path / "none",
            1,
            settings,
            0.1,
        )

    ready = replays.prepare(job("a"))
    assert ready.usable and ready.size == 1 and ready.kickoffs >= 1 and ready.count > 50
    loaded = pretrain.load([_recording_of(tmp_path)], config, rows="replicar")
    assert ready.count == len(loaded), "as many samples as the recording gives"
    torch.testing.assert_close(torch.tensor(ready.obs).float(), loaded.obs, rtol=1e-3, atol=1e-3)
    first = torch.from_numpy(ready.label_index[:, 0].astype(np.int64))
    assert torch.equal(first, loaded.label.argmax(1)), "the same first choices"
    assert not list(work.iterdir()), "replicar's file is removed"
    broken = replays.prepare(job("broken"))
    assert not broken.usable and broken.problem == "could not decode the replay"
    two = replays.prepare(job("two"))
    assert two.size == 2 and two.problem is None and not two.usable, "another team size: read, not used"


def lean_file(path: Path, arrays: dict[str, Any], players: list[dict[str, Any]]) -> None:
    """`read_replicar`'s arrays as nitro-replicar writes them (crates/nitro_replicar/src/main.rs's layout)."""
    rows, cars = arrays["car_position"].shape[:2]
    names = sorted(set(arrays["car_status"].ravel()) - {""})
    header = {"players": players, "rows": rows, "cars": cars, "status_names": names}
    text = json.dumps(header).encode()
    ball = np.concatenate(
        [arrays[f"ball_{n}"] for n in ("position", "velocity", "angular_velocity", "rotation")], 1
    )
    widths = replays._LEAN_FLOATS
    floats = np.concatenate(
        [arrays[f"car_{n}"].reshape(rows, cars, w).astype("<f4") for n, w in widths], axis=2
    )
    codes = np.array([[names.index(v) if v else 255 for v in row] for row in arrays["car_status"]], np.uint8)
    small = np.concatenate(
        [codes.view(np.int8)[:, :, None]]
        + [arrays[f"car_{n}"].astype(np.int8)[:, :, None] for n in replays._LEAN_SMALL[1:]],
        axis=2,
    )
    parts = [arrays["sim_tick"].astype("<i8"), ball.astype("<f4"), floats, small, arrays["car_jump_ticks"]]
    head = b"NRPL" + struct.pack("<II", 1, len(text)) + text
    path.write_bytes(
        head + b"".join(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes() for a in parts)
    )


@SHELL
def test_nitro_replicars_file_gives_the_arrays_and_samples_replicars_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a"])
    arrays, players = replays.read_replicar(tmp_path / "tools" / "one.parquet")
    lean_file(tmp_path / "one.lean", arrays, players)
    read, read_players = replays.read_lean(tmp_path / "one.lean")
    assert read_players == players and sorted(read) == sorted(arrays)
    for name, array in arrays.items():
        assert read[name].dtype == array.dtype and read[name].flags.writeable, name
        np.testing.assert_array_equal(read[name], array, err_msg=name)
    # The cars in the header's player order, as read_replicar picks car_<index>'s columns.
    swapped = [{**p, "index": 1 - k} for k, p in enumerate(players)]
    lean_file(tmp_path / "swapped.lean", arrays, swapped)
    read, _ = replays.read_lean(tmp_path / "swapped.lean")
    np.testing.assert_array_equal(read["car_position"], arrays["car_position"][:, ::-1])
    (tmp_path / "cut.lean").write_bytes((tmp_path / "one.lean").read_bytes()[:-1])
    with pytest.raises(ValueError, match="wrong size"):
        replays.read_lean(tmp_path / "cut.lean")
    # A job with nitro-replicar (its stand-in copies the file) makes the same samples.
    tool = tmp_path / "lean-tool" / "replicar"
    tool.parent.mkdir()
    tool.write_text(f'#!/bin/sh\ncp "{tmp_path / "one.lean"}" "$4"\n', encoding="utf-8")
    tool.chmod(0o755)
    config = large(tmp_path, demos)
    settings = pretrain.copy_settings(config, replicar_rows=True)
    work = tmp_path / "work"
    work.mkdir()
    made = [
        replays.prepare(
            replays.Job(
                0, demos / "a.replay", (1,), t, tmp_path, work, tmp_path / "none", 1, settings, 0.1, lean
            )
        )
        for t, lean in ((replays.replicar_tool(), False), (tool, True))
    ]
    assert made[0].usable and made[0].count == made[1].count and made[1].problem is None
    for name in ("obs", "mask", "label_index", "label_share"):
        np.testing.assert_array_equal(getattr(made[0], name), getattr(made[1], name), err_msg=name)
    assert not list(work.iterdir()), "the converted file is removed"


def _recording_of(tmp_path: Path) -> Path:
    file = tmp_path / "one.rlpr"
    file.write_bytes(replays.recording(*replays.read_replicar(tmp_path / "tools" / "one.parquet")).data)
    return file


@SHELL
def test_the_copy_learns_batch_by_batch_and_carries_on_after_a_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["a", "b", "two", "c", "broken", "d", "e", "f", "g", "h"]
    demos = tools_and_replays(tmp_path, monkeypatch, names)
    config = large(tmp_path, demos, passes=10)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    heard: list[str] = []

    class Stop(Exception):
        pass

    def stop_after_round_one(line: str) -> None:
        heard.append(line)
        if line.startswith("  round 1:"):
            raise Stop

    with pytest.raises(Stop):
        pretrain.start_source(config, folder, stop_after_round_one)
    work = folder.root / pretrain.COPY
    progress = torch.load(work / pretrain.PROGRESS, weights_only=True)
    assert progress["rounds"] == 1 and progress["used"] == 2 and len(progress["held"]) == 2
    assert progress["other_sizes"] == 1 and progress["failed"] == {"could not decode the replay": 1}
    assert (tmp_path / "replays" / "sizes.txt").read_text().count("two 2") == 1, "remembered for later"

    heard.clear()
    source = pretrain.start_source(config, folder, heard.append)
    assert source is not None and heard[0].startswith(
        "[pretrain], large replay mode: carrying on after round 1"
    )
    details = source.state["pretrained"]
    assert details["replays"] == 4 and details["large"]["rounds"] == 2 and details["held_out_replays"] == 2
    assert [e["epoch"] for e in details["epochs"]] == [1, 2]
    assert details["epochs"][1]["held_loss"] < details["epochs"][0]["held_loss"], "learning across rounds"
    assert not any(line.startswith("  round 1:") for line in heard), "round 1 is not done again"
    assert not list((tmp_path / "replays").glob("1v1/*")), "no recording is written in this mode"
    assert source.state["learner"]["model"]["policy.0.weight"].shape[1] == progress["obs_size"]

    # The finished copy is reused; other batch settings make it again from the start.
    assert pretrain.start_source(config, folder, heard.append) is not None
    again: list[str] = []
    other = large(tmp_path, demos, passes=10, keep_batches=1)
    pretrain.start_source(other, folder, again.append)
    assert again[0].startswith("The [pretrain] settings or the networks changed")
    assert any("large replay mode: up to 4" in line for line in again)


@SHELL
def test_too_few_replays_for_the_large_mode_is_said_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "two"])
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "few", tmp_path / "s.toml")
    with pytest.raises(
        pretrain.PretrainError, match=r"only 1 of the 2 replays .* leave pretrain.batch_replays"
    ):
        pretrain.start_source(large(tmp_path, demos), folder, print)


def test_a_round_learns_from_every_kept_batch() -> None:
    torch.manual_seed(0)
    actions = _core.num_actions()

    def batch(rows: int, action: int) -> dict[str, torch.Tensor]:
        mask = np.ones((rows, actions), dtype=bool)
        index = np.zeros((rows, replays.LABEL_ENTRIES), dtype=np.uint8)
        index[:, 0] = action
        share = np.zeros((rows, replays.LABEL_ENTRIES), dtype=np.float16)
        share[:, 0] = 1.0
        obs = np.full((rows, 4), float(action) / actions, dtype=np.float16)
        return {
            "obs": torch.from_numpy(obs),
            "mask": torch.from_numpy(np.packbits(mask, axis=1, bitorder="little")),
            "label_index": torch.from_numpy(index),
            "label_share": torch.from_numpy(share),
        }

    config = parse_toml(SETTINGS, NitroConfig, file=Path("s.toml"), base_dir=Path.cwd())
    from nitro.model import build_networks

    model = build_networks(config, 4, actions)
    params = [p for name, p in model.named_parameters() if not name.startswith("value.")]
    optimizer = torch.optim.Adam(params, lr=1e-2)
    batches = [batch(300, 7), batch(500, 40)]
    for seed in range(20):
        pretrain._learn_round(model, optimizer, batches, 1, 128, torch.device("cpu"), seed)
    for packed, action in zip(batches, (7, 40), strict=True):
        assert pretrain._measure_packed(model, packed, torch.device("cpu"))[1] == 1.0, action


def test_the_large_mode_settings() -> None:
    assert PretrainConfig().batch_replays == 0, "off: every replay at once, as before"
    assert pretrain.held_replays(30) == 2 and pretrain.held_replays(10_000) == 50
    assert pretrain.workers(0) == max(1, min(16, (os.cpu_count() or 2) // 2)) and pretrain.workers(3) == 3


MESHES = Path(__file__).parents[1] / "collision_meshes"


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@SHELL
def test_nitro_train_with_the_large_mode_starts_from_its_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from nitro.cli import main

    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d", "e"])
    file = tmp_path / "run.toml"
    file.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n'
        + SETTINGS
        + "[env]\nnum_arenas = 4\nthreads = 2\n[ppo]\nsteps_per_iteration = 160\nminibatches = 2\n"
        + f'[pretrain]\nenabled = true\nreplays = "{demos.as_posix()}"\nlimit = 3\npasses = 2\n'
        + "batch_replays = 2\nworkers = 2\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    command = ["train", str(file), "--name", "big", "--runs", str(runs), "--steps", "1", "--then-train"]
    assert main(command) == 0
    printed = capsys.readouterr().out
    assert "large replay mode: up to 3 1v1 replays of the 5" in printed
    assert "  round 1: 2 of 3 replays learned" in printed and "  round 2: 3 of 3 replays learned" in printed
    assert "Training 'big'" in printed, "--then-train: training straight after the copy"
    folder = RunFolder(runs / "big")
    state = folder.load_latest()
    assert (
        state is not None and state["started_from"] == pretrain.COPY_LABEL and state["started_from_replays"]
    )
    assert state["agent_steps"] > 0
    copy = torch.load(folder.root / pretrain.COPY / "state.pt", weights_only=True)
    assert copy["pretrained"]["replays"] == 3 and copy["pretrained"]["large"]["rounds"] == 2


@SHELL
def test_several_folders_of_several_sizes_are_copied_in_turn_into_one_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A padded observation copies 1v1s and 2v2s from a folder of each, taken in turn, into an attention
    network with a shared trunk."""
    ones = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d"])
    twos = ones.parent / "twos"
    twos.mkdir()
    for age, name in enumerate(["two-1", "two-2", "two-3", "two-4"]):
        (twos / f"{name}.replay").write_bytes(b"")
        os.utime(twos / f"{name}.replay", (2_000_000 - age, 2_000_000 - age))
    assert [p.stem for p in replays.newest_in_turn([ones, twos])][:4] == ["a", "two-1", "b", "two-2"]
    text = SETTINGS.replace("[model]\npolicy = [32]\nvalue = [32]\n", "") + (
        '[env]\nteam_size = 2\nobs_team_size = 2\n[model]\nencoder = "attention"\nshared = [24]\n'
        "policy = [32]\nvalue = [32]\nattention_width = 16\nattention_heads = 2\n"
    )
    config = parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    chosen = PretrainConfig(
        enabled=True,
        replays=(ones, twos),
        team_sizes=(1, 2),
        limit=6,
        passes=2,
        batch_replays=2,
        workers=2,
        minibatch=32,
    )
    paths = dataclasses.replace(config.paths, collision_meshes=tmp_path / "meshes")
    config = dataclasses.replace(config, paths=paths, pretrain=chosen)
    (tmp_path / "s.toml").write_text(text, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "mixed", tmp_path / "s.toml")
    heard: list[str] = []
    shown: list[str] = []
    monkeypatch.setattr(pretrain, "STATUS_SECONDS", 0.0)  # a status line at every chance
    source = pretrain.start_source(config, folder, heard.append, shown.append)
    assert source is not None and "1v1/2v2 replays" in heard[0]
    assert shown[0].startswith("Looking through the replays in")
    learning = [line for line in shown if "now learning round" in line]
    assert learning and all("replays learned" in line and " left (done about " in line for line in learning)
    assert not any("now" in line for line in heard), "status lines go to `status`, the rounds to `say`"
    details = source.state["pretrained"]
    assert details["large"]["by_size"] == {"1": 4, "2": 4} and details["large"]["other_team_sizes"] == 0
    assert details["replays"] == 6 and details["held_out_replays"] == 2
    model = source.state["learner"]["model"]
    assert any(k.startswith("shared.0.") for k in model), "the encoder sits in the shared trunk"
    sizes = (tmp_path / "replays" / "sizes.txt").read_text()
    assert "a 1" in sizes and "two-1 2" in sizes


@SHELL
def test_a_copy_carries_on_another_runs_copy_with_new_replays_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pretrain.continue_from starts the copy from another run's copy (made before any training);
    new_only leaves out every replay a copy on this PC made ready before."""
    demos = tools_and_replays(tmp_path, monkeypatch, [f"r{k}" for k in range(12)])
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    runs = tmp_path / "runs"
    made = RunFolder.create(runs, "first", tmp_path / "s.toml")
    first = pretrain.start_source(large(tmp_path, demos), made, str)
    assert first is not None and first.state["pretrained"]["replays"] == 4  # and 2 held back: 6 made ready
    noted = {line.split()[0] for line in (tmp_path / "replays" / "sizes.txt").read_text().splitlines()}
    calls_log = tmp_path / "tools" / "calls.log"
    calls_log.unlink()
    second = RunFolder.create(runs, "second", tmp_path / "s.toml")
    config = large(tmp_path, demos, continue_from="first", new_only=True)
    measure, started = pretrain._measure_packed, []

    def first_measure(model: Any, packed: Any, device: Any) -> tuple[float, float]:
        if not started:
            started.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return measure(model, packed, device)

    monkeypatch.setattr(pretrain, "_measure_packed", first_measure)
    heard: list[str] = []
    source = pretrain.start_source(config, second, heard.append)
    assert source is not None
    start = next(line for line in heard if line.startswith("[pretrain], large replay mode"))
    assert "Left out as made ready before: 6." in start and "copy of 'first' (4 replays" in start
    weights = first.state["learner"]["model"]
    assert all(torch.equal(started[0][k], weights[k]) for k in weights), "it starts from the first copy"
    assert any(line.startswith("  before these replays (the copy of 'first')") for line in heard)
    details = source.state["pretrained"]
    assert details["continued_from"] == {"run": "first", "replays": 4} and details["replays"] == 4
    assert details["large"]["used_before"] == 6 and details["before"]["held_loss"] > 0
    again = {Path(line).stem for line in calls_log.read_text().splitlines()}
    assert len(noted) == 6 and not noted & again, "the second copy converts none the first made ready"
    assert {"continue_from": "first", "new_only": True}.items() <= pretrain._made_with(config).items()
    assert "continue_from" not in pretrain._made_with(large(tmp_path, demos)), "unchanged when unset"

    third = RunFolder.create(runs, "third", tmp_path / "s.toml")

    def refused(**changes: Any) -> str:
        with pytest.raises(pretrain.PretrainError) as err:
            pretrain.start_source(large(tmp_path, demos, **changes), third, str)
        return str(err.value)

    assert "'missing' has no copy of replays" in refused(continue_from="missing")
    assert "names this run itself" in refused(continue_from="third")
    shorter = large(tmp_path, demos, continue_from="first")
    other_env = dataclasses.replace(shorter.env, tick_skip=8)
    other_model = dataclasses.replace(shorter.model, policy=(16,))
    for wrong, part in (
        (dataclasses.replace(shorter, env=other_env), "env.tick_skip"),
        (dataclasses.replace(shorter, model=other_model), "model.policy"),
    ):
        with pytest.raises(pretrain.PretrainError, match=part):
            pretrain.start_source(wrong, third, str)


@SHELL
def test_a_carried_on_copy_stopped_mid_way_carries_on_and_a_misfit_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop during round 2 keeps the plan and the first "before"; a copy whose networks do
    not fit names the misfit."""
    demos = tools_and_replays(tmp_path, monkeypatch, [f"r{k}" for k in range(14)])
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    runs = tmp_path / "runs"
    pretrain.start_source(large(tmp_path, demos), RunFolder.create(runs, "first", tmp_path / "s.toml"), str)
    second = RunFolder.create(runs, "second", tmp_path / "s.toml")
    config = large(tmp_path, demos, continue_from="first", new_only=True, limit=6)
    learn, rounds = pretrain._learn_round, []

    def stop_in_round_two(*args: Any, **kwargs: Any) -> float:
        rounds.append(1)
        if len(rounds) == 2:
            raise KeyboardInterrupt  # the PC turned off
        return learn(*args, **kwargs)

    monkeypatch.setattr(pretrain, "_learn_round", stop_in_round_two)
    with pytest.raises(KeyboardInterrupt):
        pretrain.start_source(config, second, str)
    saved = torch.load(second.root / pretrain.COPY / pretrain.PROGRESS, weights_only=True)
    assert saved["rounds"] == 1 and saved["before"] is not None
    monkeypatch.setattr(pretrain, "_learn_round", learn)
    heard: list[str] = []
    source = pretrain.start_source(config, second, heard.append)
    assert source is not None and any("carrying on after round 1" in line for line in heard)
    assert not any(line.startswith("  before these replays") for line in heard), "measured once"
    details = source.state["pretrained"]
    assert details["before"] == saved["before"] and details["replays"] == 6
    assert details["continued_from"] == {"run": "first", "replays": 4}

    state_file = runs / "first" / pretrain.COPY / "state.pt"
    state = torch.load(state_file, weights_only=True)
    first_layer = state["learner"]["model"]["policy.0.weight"]
    state["learner"]["model"]["policy.0.weight"] = torch.zeros(first_layer.shape[0], first_layer.shape[1] + 3)
    torch.save(state, state_file)  # as if made with another observation
    third = RunFolder.create(runs, "third", tmp_path / "s.toml")
    with pytest.raises(pretrain.PretrainError, match=r"size mismatch for policy\.0\.weight"):
        pretrain.start_source(large(tmp_path, demos, continue_from="first"), third, str)


@SHELL
def test_the_copy_is_charted_and_reported_round_by_round_and_nitro_wandb_sends_nothing_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, charts: dict[str, Any]
) -> None:
    """With [wandb] on, each round goes to W&B against the replays learned in all (a carried-on
    copy's before it, at round 0); with run.console = "report", each round prints its block."""
    from nitro.cli import main

    demos = tools_and_replays(tmp_path, monkeypatch, [f"r{k}" for k in range(12)])
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    runs = tmp_path / "runs"

    def charted(config: NitroConfig) -> NitroConfig:
        run = dataclasses.replace(config.run, console="report")
        return dataclasses.replace(config, run=run, wandb=WandbConfig(enabled=True, project="t"))

    shown: list[str] = []
    first = RunFolder.create(runs, "first", tmp_path / "s.toml")
    pretrain.start_source(charted(large(tmp_path, demos)), first, str, shown.append)
    assert [v["copy/replays_in_all"] for v in charts["log"]] == [2.0, 4.0]
    assert any(line.startswith("---- first | copying replays | round 1 of about 2 ") for line in shown)
    charts["log"].clear()
    second = RunFolder.create(runs, "second", tmp_path / "s.toml")
    config = charted(large(tmp_path, demos, continue_from="first", new_only=True))
    pretrain.start_source(config, second, str, shown.append)
    logged = [(v["copy/round"], v["copy/replays_in_all"]) for v in charts["log"]]
    assert logged == [(0.0, 4.0), (1.0, 6.0), (2.0, 8.0)], "round 0: the first copy, before these replays"
    assert "copy/train_loss" in charts["log"][1] and "copy/train_loss" not in charts["log"][0]
    rounds, passes = pretrain.copy_for_charts(second.root)
    assert [(r["round"], r["replays_in_all"]) for r in rounds] == [(0, 4), (1, 6), (2, 8)] and passes == []
    charts["log"].clear()
    assert main(["wandb", "second", "--runs", str(runs)]) == 0
    assert charts["log"] == [], "everything was sent while copying"


def test_nitro_wandb_sends_the_passes_of_a_copy_made_by_nitro_pretrain(
    tmp_path: Path, charts: dict[str, Any]
) -> None:
    from nitro.cli import main

    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "cloned", tmp_path / "s.toml")
    epochs = [
        {"epoch": k, "train_loss": 2.0 / k, "held_loss": 2.1 / k, "held_agreement": 0.3} for k in (1, 2)
    ]
    (folder.root / "pretrain.json").write_text(json.dumps({"epochs": epochs}), encoding="utf-8")
    assert main(["wandb", "cloned", "--runs", str(tmp_path / "runs")]) == 0
    assert [v["clone/pass"] for v in charts["log"]] == [1.0, 2.0]
    assert ("clone/held_loss", {"step_metric": "clone/pass"}) in charts["define"]


def test_continue_from_and_new_only_need_the_large_mode_and_a_run_name(tmp_path: Path) -> None:
    config = parse_toml(SETTINGS, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)

    def problems(**changes: Any) -> list[str]:
        chosen = PretrainConfig(**{"enabled": True, "batch_replays": 10, **changes})
        return [k for k, _ in dataclasses.replace(config, pretrain=chosen).problems() if k.startswith("pre")]

    assert problems(continue_from="copied", new_only=True) == []
    assert problems(continue_from="a", new_only=True, batch_replays=0) == [
        "pretrain.continue_from",
        "pretrain.new_only",
    ]
    assert problems(continue_from="runs/a") == ["pretrain.continue_from"]


def test_team_sizes_need_the_large_mode_and_room_in_the_observation(tmp_path: Path) -> None:
    def problems(env: str, **pretrain_changes: Any) -> list[str]:
        config = parse_toml(SETTINGS + env, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
        chosen = PretrainConfig(**{"enabled": True, "batch_replays": 10, **pretrain_changes})
        return [
            m
            for k, m in dataclasses.replace(config, pretrain=chosen).problems()
            if k == "pretrain.team_sizes"
        ]

    assert problems("[env]\nobs_team_size = 3\n", team_sizes=(1, 2, 3)) == []
    assert problems("", team_sizes=(1,)) == [], "the run's own size needs no room"
    assert "padded observation" in problems("", team_sizes=(1, 2))[0]
    assert "does not fit" in problems("[env]\nobs_team_size = 2\n", team_sizes=(1, 3))[0]
    assert "batch_replays" in problems("[env]\nobs_team_size = 2\n", team_sizes=(1, 2), batch_replays=0)[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_a_round_learns_in_fp16_on_the_graphics_card() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    actions, obs_size, rows = _core.num_actions(), 20, 256
    config = parse_toml(SETTINGS, NitroConfig, file=Path("s.toml"), base_dir=Path())
    model = pretrain.build_networks(config, obs_size, actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = torch.randint(0, actions, (rows,))
    batch = {
        "obs": torch.randn(rows, obs_size).half(),
        "mask": torch.full((rows, (actions + 7) // 8), 255, dtype=torch.uint8),
        "label_index": torch.stack(
            [best] + [torch.zeros(rows, dtype=torch.long)] * (replays.LABEL_ENTRIES - 1), 1
        ).to(torch.uint8),
        "label_share": torch.tensor([1.0] + [0.0] * (replays.LABEL_ENTRIES - 1)).half().repeat(rows, 1),
    }
    scaler = torch.amp.GradScaler("cuda")
    first = pretrain._learn_round(model, optimizer, [batch], 1, 64, device, 0, scaler)
    for _ in range(20):
        last = pretrain._learn_round(model, optimizer, [batch], 1, 64, device, 0, scaler)
    assert last < first, "it learns in FP16"


@SHELL
def test_a_stop_after_the_last_round_an_unreadable_save_and_a_stale_progress_are_handled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["a", "b", "c", "d", "e", "f"]
    demos = tools_and_replays(tmp_path, monkeypatch, names)
    config = large(tmp_path, demos)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "stops", tmp_path / "s.toml")
    work = folder.root / pretrain.COPY

    class Stop(Exception):
        pass

    def stop_at(round_line: str) -> Any:
        def say(line: str) -> None:
            if line.startswith(round_line):
                raise Stop

        return say

    with pytest.raises(Stop):  # after the last round's save, before the copy is kept
        pretrain.start_source(config, folder, stop_at("  round 2:"))
    heard: list[str] = []
    source = pretrain.start_source(config, folder, heard.append)
    assert source is not None and "carrying on after round 2" in heard[0]
    details = source.state["pretrained"]
    assert details["replays"] == 4 and details["large"]["rounds"] == 2
    assert 0 < details["large"]["seconds_per_replay"] < 60, "counted over both starts"
    assert not (work / pretrain.PROGRESS).exists() and not (work / pretrain.HELD).exists(), (
        "removed when kept"
    )
    assert not (tmp_path / "replays" / "converted").exists(), "replicar's folder goes when empty"

    # A progress cut off by a power cut: said, and the copy made again (no traceback).
    other = large(tmp_path, demos, passes=3)
    with pytest.raises(Stop):
        pretrain.start_source(other, folder, stop_at("  round 1:"))
    raw = (work / pretrain.PROGRESS).read_bytes()
    (work / pretrain.PROGRESS).write_bytes(raw[: len(raw) // 2])
    heard.clear()
    assert pretrain.start_source(other, folder, heard.append) is not None
    assert any("progress.pt could not be read" in line for line in heard)
    assert any("large replay mode: up to 4" in line for line in heard), "from the start"

    # A stopped copy of other settings never meets this copy's held-back samples.
    third = large(tmp_path, demos, passes=4)
    with pytest.raises(Stop):
        pretrain.start_source(third, folder, stop_at("  round 1:"))
    with pytest.raises(Stop):  # a fourth setting saves its held-back samples, then stops before a round
        pretrain.start_source(large(tmp_path, demos, passes=5), folder, stop_at("[pretrain], large"))
    heard.clear()
    assert pretrain.start_source(third, folder, heard.append) is not None
    assert not any("carrying on" in line for line in heard), "its progress went with the other start"


@SHELL
def test_failure_reasons_hold_no_replay_name_or_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["broken-0123456789abcdef"])
    tool = tmp_path / "loud"
    tool.write_text(
        '#!/bin/sh\necho "error: $2: No such file at byte 123456" >&2\nexit 2\n', encoding="utf-8"
    )
    tool.chmod(0o755)
    replay = demos / "broken-0123456789abcdef.replay"
    reason = replays.convert(tool, replay, tmp_path / "out.parquet", tmp_path)
    assert reason is not None and "0123456789abcdef" not in reason and str(demos) not in reason
    assert reason.startswith("error: <replay>")


def test_a_copy_of_several_sizes_serves_any_team_size_and_named_folders_must_exist(tmp_path: Path) -> None:
    text = SETTINGS + "[env]\nteam_size = 2\nobs_team_size = 3\n"
    config = parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    chosen = PretrainConfig(enabled=True, replays=(tmp_path,), batch_replays=10, team_sizes=(1, 2, 3))
    config = dataclasses.replace(config, pretrain=chosen)
    made = pretrain._made_with(config)
    other = pretrain._made_with(dataclasses.replace(config, env=dataclasses.replace(config.env, team_size=3)))
    assert made != other and pretrain._same_copy(made, other), "team_size free before the first save"
    room = dataclasses.replace(config.env, obs_team_size=4)
    assert not pretrain._same_copy(made, pretrain._made_with(dataclasses.replace(config, env=room)))
    one = dataclasses.replace(config, pretrain=dataclasses.replace(chosen, team_sizes=()))
    one_three = dataclasses.replace(one, env=dataclasses.replace(one.env, team_size=3))
    assert not pretrain._same_copy(pretrain._made_with(one), pretrain._made_with(one_three)), "one size: no"
    missing = dataclasses.replace(chosen, replays=(tmp_path, tmp_path / "3v3"))
    with pytest.raises(pretrain.PretrainError, match="is not a folder"):
        pretrain._sources(missing)


def _packed_batch(rows: int, obs_size: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    actions = _core.num_actions()
    best = torch.randint(0, actions, (rows,), generator=generator)
    return {
        "obs": torch.randn(rows, obs_size, generator=generator).half(),
        "mask": torch.full((rows, (actions + 7) // 8), 255, dtype=torch.uint8),
        "label_index": torch.stack(
            [best] + [torch.zeros(rows, dtype=torch.long)] * (replays.LABEL_ENTRIES - 1), 1
        ).to(torch.uint8),
        "label_share": torch.tensor([1.0] + [0.0] * (replays.LABEL_ENTRIES - 1)).half().repeat(rows, 1),
    }


def test_the_learning_rate_falls_along_a_cosine_to_its_end_share() -> None:
    """lr_end_share 1 keeps the rate (as before); below 1 it falls smoothly to that share."""
    assert [pretrain.lr_at(3e-4, 1.0, d) for d in (0.0, 0.5, 1.0)] == [3e-4] * 3
    falling = [pretrain.lr_at(1.0, 0.1, d / 10) for d in range(11)]
    assert falling[0] == pytest.approx(1.0) and falling[-1] == pytest.approx(0.1)
    assert falling[5] == pytest.approx(0.55) and all(a > b for a, b in itertools.pairwise(falling))
    assert pretrain.lr_at(1.0, 0.1, 2.0) == pytest.approx(0.1), "past the end it stays at the end"


def test_a_round_sets_the_learning_rate_before_each_step() -> None:
    config = parse_toml(SETTINGS, NitroConfig, file=Path("s.toml"), base_dir=Path())
    model = pretrain.build_networks(config, 20, _core.num_actions())
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    seen: list[float] = []

    def rate(step: int, steps: int) -> float:
        seen.append(step / steps)
        return 1e-3 * (1 - step / steps)

    pretrain._learn_round(
        model, optimizer, [_packed_batch(256, 20, 0)], 2, 64, torch.device("cpu"), 0, lr_for=rate
    )
    assert seen == [k / 8 for k in range(8)], "2 passes of 4 steps, the rate set before each"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 / 8)


@SHELL
def test_the_held_back_replays_are_measured_every_few_rounds_and_after_the_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """measure_every 2 over 3 rounds measures round 2 and the last; lr and lr_end_share reach the
    optimizer; the console says when a round was not measured."""
    names = ["a", "b", "c", "d", "e", "f", "g", "h"]  # 2 held back, 6 learned: 3 rounds of 2
    demos = tools_and_replays(tmp_path, monkeypatch, names)
    config = large(tmp_path, demos, limit=6, measure_every=2, lr=1e-3, lr_end_share=0.1)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    rates: list[tuple[float, float]] = []  # each round's first and last rate
    learn = pretrain._learn_round

    def noting(*args: Any, **kwargs: Any) -> float:
        rate = kwargs["lr_for"] if "lr_for" in kwargs else args[9]
        steps = 2 * 8  # passes x steps a pass, at most: only the ends are read
        rates.append((rate(0, steps), rate(steps, steps)))
        return learn(*args, **kwargs)

    monkeypatch.setattr(pretrain, "_learn_round", noting)
    heard: list[str] = []
    source = pretrain.start_source(config, folder, heard.append)
    assert source is not None
    epochs = source.state["pretrained"]["epochs"]
    assert [e["held_loss"] is not None for e in epochs] == [False, True, True]
    assert any("measured every 2 rounds" in line for line in heard if line.startswith("  round 1:"))
    assert rates[0][0] == pytest.approx(1e-3) and rates[-1][1] == pytest.approx(1e-4)
    assert [round(a, 9) for a, _ in rates[1:]] == [round(b, 9) for _, b in rates[:-1]], "one curve"


@SHELL
def test_replays_running_out_after_a_round_between_measurements_still_measure_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["a", "b", "c", "d", "e", "f"]  # 2 held back, 4 learned: 2 rounds, then no more replays
    demos = tools_and_replays(tmp_path, monkeypatch, names)
    config = large(tmp_path, demos, limit=10, measure_every=5)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    heard: list[str] = []
    source = pretrain.start_source(config, folder, heard.append)
    assert source is not None
    epochs = source.state["pretrained"]["epochs"]
    assert epochs[-1]["held_loss"] is not None and epochs[0]["held_loss"] is None
    assert any(line.startswith("  after round") for line in heard)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_rounds_held_on_the_graphics_card_learn_exactly_as_rounds_held_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picking each step's rows on the card changes nothing but the speed."""
    device = torch.device("cuda")
    config = parse_toml(SETTINGS, NitroConfig, file=Path("s.toml"), base_dir=Path())
    batches = [_packed_batch(300, 20, 1), _packed_batch(200, 20, 2)]
    weights = []
    for on_card in (False, True):
        if not on_card:
            monkeypatch.setattr(pretrain, "_resident", lambda packed, device: packed)
        else:
            monkeypatch.undo()
        torch.manual_seed(0)
        model = pretrain.build_networks(config, 20, _core.num_actions()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        for seed in range(3):
            pretrain._learn_round(model, optimizer, batches, 2, 64, device, seed, scaler)
        weights.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        assert pretrain._resident(batches[0], device)["obs"].device.type == ("cuda" if on_card else "cpu")
    for name, value in weights[0].items():
        assert torch.equal(value, weights[1][name]), name


@SHELL
def test_the_copy_writes_how_far_it_is_for_the_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """live.json as the copy goes (after round 1: 2 of 4 learned) and when it is done."""
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d", "e", "f"])
    config = large(tmp_path, demos)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    work = folder.root / pretrain.COPY
    seen: list[dict[str, Any] | None] = []

    def hear(line: str) -> None:
        if line.startswith("  round 1:"):
            seen.append(pretrain.read_live(work))

    assert pretrain.start_source(config, folder, hear) is not None
    during = seen[0]
    assert during is not None and not during["finished"] and (during["learned"], during["limit"]) == (2, 4)
    assert during["doing"] == "learned round 1" and [r["round"] for r in during["rounds"]] == [1]
    done = pretrain.read_live(work)
    assert done is not None and done["finished"] and done["learned"] == 4 and len(done["rounds"]) == 2
    assert done["seconds_left"] is None and done["skill"] is None and done["games"] == "1v1"


def _fake_matches(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> list[tuple[str, str, int]]:
    """nitro.eval.play as a stand-in: Blue wins 8 goals to 2 (10 draws) in every match, or fails."""
    from nitro import eval as matches

    played: list[tuple[str, str, int]] = []

    def play(blue: Any, orange: Any, games: int, seed: int, team_size: int | None = None) -> Any:
        if fail:
            raise matches.EvalError("no arena meshes")
        played.append((blue.name, orange.name, games))
        assert set(blue.state["learner"]["model"]) == set(orange.state["learner"]["model"])
        return matches.MatchResult(20, 8, 2, 10, 5.0, 60.0)

    monkeypatch.setattr(matches, "play", play)
    return played


@SHELL
def test_skill_checks_play_where_the_copy_started_and_the_last_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With skill_every 1 over 3 rounds, round 1 plays the start, rounds 2 and 3 the start and the last
    check; only those two versions stay on disk while copying, none once the copy is kept; the ratings rise
    with wins and the copy keeps them."""
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d", "e", "f", "g", "h"])
    config = large(tmp_path, demos, limit=6, skill_every=1, skill_points=20)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    work = folder.root / pretrain.COPY
    played = _fake_matches(monkeypatch)
    kept: list[set[str]] = []
    heard: list[str] = []

    def hear(line: str) -> None:
        heard.append(line)
        if line.startswith("  round "):
            kept.append({p.stem for p in (work / pretrain.VERSIONS).glob("*.pt")})

    source = pretrain.start_source(config, folder, hear)
    assert source is not None
    against = [name for _, name, _ in played]
    assert against == ["round 0", "round 0", "round 1", "round 0", "round 2"]
    assert all(blue == "this copy" and points == 20 for blue, _, points in played)
    assert kept == [{"0", "1"}, {"0", "2"}, {"0", "3"}]
    assert not list((work / pretrain.VERSIONS).glob("*.pt")), "removed once the copy is kept"
    skill = source.state["pretrained"]["skill"]
    mus = [v["mu"] for v in skill["versions"]]
    assert [v["round"] for v in skill["versions"]] == [0, 1, 2, 3] and mus[0] == 25.0
    assert mus[1] > mus[0] and mus[3] > mus[1], "winning 8 goals to 2 raises the rating"
    assert [(m["round"], m["against"], m["won"], m["lost"]) for m in skill["matches"]][:2] == [
        (1, 0, 8, 2),
        (2, 0, 8, 2),
    ]
    assert any(
        "skill check after round 3: against round 0's copy in 20 points it took 80% of the 10 goals" in h
        for h in heard
    )
    live = pretrain.read_live(work)
    assert live is not None and live["skill"]["versions"][-1]["round"] == 3


@SHELL
def test_a_skill_check_that_cannot_play_never_stops_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d", "e", "f"])
    config = large(tmp_path, demos, skill_every=1)
    (tmp_path / "s.toml").write_text(SETTINGS, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    _fake_matches(monkeypatch, fail=True)
    heard: list[str] = []
    source = pretrain.start_source(config, folder, heard.append)
    assert source is not None and source.state["pretrained"]["replays"] == 4
    assert sum("not played: no arena meshes" in h for h in heard) == 2


def test_skill_checks_need_the_large_mode_and_opponents(tmp_path: Path) -> None:
    from nitro.config.loader import ConfigError

    text = SETTINGS + "[env]\nopponents = false\n[pretrain]\nenabled = true\nskill_every = 5\n"
    with pytest.raises(ConfigError) as refused:
        parse_toml(text, NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path)
    assert "needs a game with opponents" in str(refused.value)
    assert "needs pretrain.batch_replays" in str(refused.value)


@SHELL
def test_conversions_are_kept_so_no_later_bot_converts_them_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy keeps every replay it converts, any game size, in the store; a later copy by another bot
    makes the same samples from them without converting, and the convert-only pass converts only the rest."""
    from nitro.cli import main
    from test_replays import converted_calls

    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    names = ["a", "two", "b", "broken", "c", "d"]
    demos = tools_and_replays(tmp_path, monkeypatch, names)
    tools = tmp_path / "tools"
    file = tmp_path / "run.toml"
    file.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n'
        + SETTINGS
        + f'[pretrain]\nenabled = true\nreplays = "{demos.as_posix()}"\nlimit = 3\npasses = 2\n'
        + "batch_replays = 2\nworkers = 2\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    assert main(["train", str(file), "--name", "first", "--runs", str(runs)]) == 0
    first = set(converted_calls(tools))
    kept = first - {"broken"}
    assert {"a", "two"} <= kept, first
    counts = replays.store_summary()["counts"]
    assert counts == {"1v1": len(kept - {"two"}), "2v2": 1}, "another size is kept too, for later bots"
    assert main(["train", str(file), "--name", "second", "--runs", str(runs)]) == 0
    assert set(converted_calls(tools)) <= {"broken"}, (
        "the second bot converts nothing kept (a failure is tried again)"
    )
    copies = [
        torch.load(runs / n / pretrain.COPY / "state.pt", weights_only=True) for n in ("first", "second")
    ]
    assert copies[0]["pretrained"]["replays"] == copies[1]["pretrained"]["replays"] > 0
    for key, value in copies[0]["learner"]["model"].items():
        assert torch.equal(value, copies[1]["learner"]["model"][key]), (
            f"the same samples, the same copy: {key}"
        )
    # Converting only: what is not kept yet (the broken one is tried again; it has no size to remember).
    heard: list[str] = []
    done = replays.convert_to_store([demos], tmp_path / "meshes", workers=2, say=heard.append)
    rest = set(names) - kept
    assert set(converted_calls(tools)) == rest and done.todo == done.done == len(rest)
    assert dict(done.failed) == {"could not decode the replay": 1} and done.kept == len(rest) - 1
    assert replays.store_summary()["counts"] == {"1v1": 4, "2v2": 1}
    view = json.loads(replays.converting_file().read_text(encoding="utf-8"))
    assert view["state"] == "done" and view["found"] == 6 and view["kept"] == len(rest) - 1
    assert replays.convert_to_store([demos], tmp_path / "meshes", workers=2, say=heard.append).todo == 1


@SHELL
def test_a_second_convert_only_pass_is_refused_while_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two passes at once would convert the same replays twice and clear each other's files (seen in
    use)."""
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    demos = tools_and_replays(tmp_path, monkeypatch, ["a"])
    quiet: list[str] = []
    with (
        replays._one_pass(replays.store_folder() / "pass.lock"),
        pytest.raises(replays.ReplayError, match="already being converted"),
    ):
        replays.convert_to_store([demos], tmp_path / "meshes", workers=1, say=quiet.append)
    assert replays.convert_to_store([demos], tmp_path / "meshes", workers=1, say=quiet.append).kept == 1


@SHELL
def test_a_nearly_full_disk_keeps_nothing_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "two"])
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 2**62)
    heard: list[str] = []
    done = replays.convert_to_store([demos], tmp_path / "meshes", workers=1, say=heard.append)
    assert done.done == 2 and done.kept == 0 and done.disk_full
    assert any("GB free" in line for line in heard)
    assert replays.store_summary()["counts"] == {"1v1": 0, "2v2": 0}
