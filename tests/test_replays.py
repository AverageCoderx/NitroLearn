"""nitro.replays and tools/replays.py: replicar's files of players' replays become
recordings `nitro pretrain` reads. Built without the game or replicar itself: replicar-style Parquet files are
made here with replicar's column names (from its writer, replicar-format 2.0.4) and a stand-in for its
command-line tool."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from nitro import _core, pretrain, replays  # noqa: E402
from nitro.config.loader import parse_toml  # noqa: E402
from nitro.config.schema import NitroConfig  # noqa: E402

TOOLS = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location("replays_tool", TOOLS / "replays.py")
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["replays_tool"] = tool
spec.loader.exec_module(tool)

# RocketSim's RLPR layout: the header, then per tick each car's record and the ball's (each after a 4-byte
# id), a body record starting with the frame, position, rotation, velocity and spin.
_HEADER = struct.Struct("<4sBIII6fI")
_PHYS = struct.Struct("<I3f9f3f3f")
_BALL_SIZE = 332
_CAR_SIZES = {2: 584, 3: 744, 4: 864, 5: 872, 6: 588, 7: 596, 8: 600, 9: 604}


def read_rlpr(path: Path) -> list[tuple[int, list[tuple[float, ...]], float]]:
    """Each tick of an RLPR recording: its frame, the cars' positions and the ball's height."""
    data = path.read_bytes()
    magic, endian, version, info_size, cars, *_hitbox, count = _HEADER.unpack_from(data, 0)
    assert magic == b"RLPR" and endian == 0 and info_size == 28 and version in _CAR_SIZES
    offset, ticks = _HEADER.size, []
    for _ in range(count):
        positions = []
        for _car in range(cars):
            positions.append(_PHYS.unpack_from(data, offset + 4)[1:4])
            offset += 4 + _CAR_SIZES[version]
        ball = _PHYS.unpack_from(data, offset + 4)
        ticks.append((ball[0], positions, ball[3]))
        offset += 4 + _BALL_SIZE
    return ticks


SETTINGS = '[run]\ndevice = "cpu"\n[model]\npolicy = [32]\nvalue = [32]\n'
ROWS = 300
DEMOLISHED_ROW = 150


def replicar_file(path: Path, per_side: int = 1) -> None:
    """A match as replicar writes it: player 0 is Orange (to show the recording puts Blue first). Every car
    starts on its kickoff spot; after 10 ticks Blue drives straight on and Orange steers left. One Blue car is
    demolished for one row."""
    players = [
        {"index": k, "team": 1 - k % 2, "name": f"p{k}", "hitbox": "octane"} for k in range(2 * per_side)
    ]
    columns: dict[str, list[object]] = {"frame": [], "sim_tick": []}

    def put(name: str, value: object) -> None:
        columns.setdefault(name, []).append(value)

    for t in range(ROWS):
        put("frame", t // 4)
        put("sim_tick", 5000 + t)
        moving = t > 10
        for axis, value in zip("xyz", (0.0, 0.0, 93.15), strict=True):
            put(f"ball_position_{axis}", value)
        for name in ("velocity", "angular_velocity"):
            for axis in "xyz":
                put(f"ball_{name}_{axis}", 0.0)
        for axis, value in zip("xyzw", (0.0, 0.0, 0.0, 1.0), strict=True):
            put(f"ball_rotation_{axis}", value)
        for p in players:
            k, blue = p["index"], p["team"] == 0
            x = (-2048.0 if blue else 2048.0) + 300.0 * (k // 2)
            y = (-2560.0 + 7.5 * t * moving) if blue else 2560.0
            for axis, value in zip("xyz", (x, y, 17.0), strict=True):
                put(f"car_{k}_position_{axis}", value)
            for axis, value in zip("xyz", (0.0, 900.0 * moving if blue else 0.0, 0.0), strict=True):
                put(f"car_{k}_velocity_{axis}", value)
            for axis in "xyz":
                put(f"car_{k}_angular_velocity_{axis}", 0.0)
                put(f"car_{k}_flip_relative_torque_{axis}", 0.0)
            for axis, value in zip("xyzw", (0.0, 0.0, 0.0, 1.0), strict=True):
                put(f"car_{k}_rotation_{axis}", value)
            put(f"car_{k}_status", "present")
            for flag in ("has_jumped", "has_double_jumped", "has_flipped", "is_flipping", "is_jumping"):
                put(f"car_{k}_{flag}", False)
            put(f"car_{k}_is_on_ground", True)
            put(f"car_{k}_is_demoed", t == DEMOLISHED_ROW and k == 1)
            for wheel in range(4):
                put(f"car_{k}_wheel_{wheel}_contact", True)
            for name in ("flip_time", "air_time_since_jump"):
                put(f"car_{k}_{name}", 0.0)
            put(f"car_{k}_jump_ticks", t)
            put(f"car_{k}_boost", 33.0)
            for name, value in (("throttle", 1.0), ("steer", 0.0 if blue else -1.0)):
                put(f"car_{k}_controls_{name}", value)
            for name in ("pitch", "yaw", "roll"):
                put(f"car_{k}_controls_{name}", 0.0)
            for name in ("jump", "boost", "handbrake"):
                put(f"car_{k}_controls_{name}", False)
    table = pa.table(columns).replace_schema_metadata(
        {b"replicar": json.dumps({"format_version": 1, "players": players, "groups": ["state"]})}
    )
    pq.write_table(table, path)


def test_a_converted_replay_becomes_a_recording_pretraining_learns_from(tmp_path: Path) -> None:
    replicar_file(tmp_path / "match.parquet")
    arrays, players = replays.read_replicar(tmp_path / "match.parquet")
    assert arrays["car_position"].shape == (ROWS, 2, 3) and arrays["car_rotation"].shape == (ROWS, 2, 4)
    converted = replays.recording(arrays, players)
    assert converted.rows_left_out == 1 and converted.ticks == ROWS - 1
    assert converted.kickoffs >= 1
    file = tmp_path / "match.rlpr"
    file.write_bytes(converted.data)

    ticks = read_rlpr(file)
    assert len(ticks) == ROWS - 1
    assert sorted(t[2] for t in ticks)[len(ticks) // 2] > 10, "positions in uu, not Bullet units"
    assert ticks[0][1][0][1] < 0 < ticks[0][1][1][1], "Blue's car comes first"
    assert ticks[DEMOLISHED_ROW][0] - ticks[DEMOLISHED_ROW - 1][0] == 2, "the demolished row is a gap"

    config = parse_toml(SETTINGS, NitroConfig, file=tmp_path / "bc.toml", base_dir=tmp_path)
    samples = pretrain.load([file], config)
    steer = torch.tensor([_core.action_controls(int(a))[1] for a in samples.label.argmax(1)])
    assert torch.equal(steer.reshape(-1, 2), torch.tensor([[0.0, -1.0]]).expand(len(samples) // 2, 2))


def test_a_2v2_reads_with_its_team_size(tmp_path: Path) -> None:
    replicar_file(tmp_path / "two.parquet", per_side=2)
    arrays, players = replays.read_replicar(tmp_path / "two.parquet")
    assert replays.team_size(players) == 2 and arrays["car_position"].shape[1] == 4
    assert replays.recording(arrays, players).kickoffs >= 1


# The packing tick by tick that `replays.recording` replaced with NumPy, kept as the reference the
# new writer must match byte for byte.
def ref_rotation_rows(q: Any) -> list[float]:
    """A unit quaternion (x, y, z, w) as the matrix whose columns are forward, right and up, stored row by
    row (as RocketSim's recordings and ours store it; replicar's `rotation_matrix` is the same matrix)."""
    x, y, z, w = (float(v) for v in q)
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ]  # fmt: skip


def ref_phys(frame: int, pos: Any, rot: Any, vel: Any, ang_vel: Any) -> bytes:
    body = struct.pack(
        "<I3f9f3f3f", frame, *map(float, pos), *ref_rotation_rows(rot), *map(float, vel), *map(float, ang_vel)
    )
    return body + bytes(replays.PHYS_SIZE - len(body))  # no world contact, contact point or impulses


def ref_car(frame: int, a: dict[str, Any], row: int, p: int) -> bytes:
    import numpy as np

    def flag(name: str) -> bool:
        return bool(a[f"car_{name}"][row, p] == 1)

    def number(name: str) -> float:
        return float(a[f"car_{name}"][row, p])

    on_ground = flag("is_on_ground")
    has_jumped = flag("has_jumped")
    has_flip = on_ground or (
        not flag("has_flipped")
        and not flag("has_double_jumped")
        # RocketSim 0.3's rule, the next tick counted first, in the arrays' float32.
        and np.float32(number("air_time_since_jump")) + np.float32(replays.TICK_SECONDS)
        < replays.DOUBLE_JUMP_MAX_DELAY
    )
    # The port counts jump ticks on the ground too; the format's jump time is 0 until a jump (replicar's own
    # mapping to RocketSim's state does the same).
    jump_time = number("jump_ticks") / 120.0 if has_jumped else 0.0
    out = bytearray(
        ref_phys(
            frame,
            a["car_position"][row, p],
            a["car_rotation"][row, p],
            a["car_velocity"][row, p],
            a["car_angular_velocity"][row, p],
        )
    )
    out += struct.pack(
        "<3?xff3?x3ff?3x5f3?x",
        on_ground,
        flag("is_jumping"),
        flag("is_flipping"),
        jump_time,
        number("flip_time") if flag("has_flipped") else 0.0,  # the format's convention
        has_jumped,
        flag("has_double_jumped") or flag("has_flipped"),
        has_flip,
        *map(float, a["car_flip_relative_torque"][row, p]),
        min(max(number("boost") / 100.0, 0.0), 1.0),
        False,  # touching the ball: not read by cloning
        *(number(f"controls_{name}") for name in ("throttle", "steer", "pitch", "yaw", "roll")),
        *(flag(f"controls_{name}") for name in ("jump", "boost", "handbrake")),
    )
    up = ref_rotation_rows(a["car_rotation"][row, p])[2::3]
    for k in range(4):
        contact = (
            bool(a[f"car_wheel_{k}_contact"][row, p] == 1) if f"car_wheel_{k}_contact" in a else on_ground
        )
        out += bytes(8) + struct.pack("<?3x3f", contact, *(up if contact else (0.0, 0.0, 0.0))) + bytes(24)
    out += bytes(4)  # touching another car: unknown
    assert len(out) == replays.CAR_SIZE
    return bytes(out)


def ref_usable(a: dict[str, Any], row: int, players: list[int]) -> bool:
    if not all(math.isfinite(float(v)) for v in a["ball_position"][row]):
        return False
    for p in players:
        status = a["car_status"][row, p] if "car_status" in a else "present"
        demoed = "car_is_demoed" in a and a["car_is_demoed"][row, p] == 1
        if status in ("", "absent") or demoed or not math.isfinite(float(a["car_position"][row, p, 0])):
            return False
    return True


def ref_kickoff(a: dict[str, Any], row: int, players: list[int]) -> bool:
    """As `nitro_core::replay::is_kickoff`: the ball at rest in the centre, each car on the ground on its spot
    (the speed allowance included)."""
    bx, by, _ = (float(v) for v in a["ball_position"][row])
    if abs(bx) >= 5 or abs(by) >= 5 or math.hypot(*map(float, a["ball_velocity"][row])) >= 5:
        return False
    for p in players:
        y = abs(float(a["car_position"][row, p, 1]))
        if a["car_is_on_ground"][row, p] != 1 or math.hypot(*map(float, a["car_velocity"][row, p])) >= 200:
            return False
        if not any(abs(y - spot) < 60 for spot in (4608.0, 3840.0, 2560.0)):
            return False
    return True


def ref_recording(a: dict[str, Any], players: list[dict[str, Any]]) -> replays.Converted:
    """A replicar file's arrays (`replicar.read(...).arrays()`, a row per tick) and its header's players as an
    RLPR recording: Blue's cars first, every usable row (all cars present and not demolished) as one tick
    numbered by its simulation tick, so a left-out row breaks the run there."""
    order = [p["index"] for p in sorted(players, key=lambda p: (p["team"], p["index"]))]
    ticks = bytearray()
    count = kickoffs = left_out = 0
    frames = a["sim_tick"] if "sim_tick" in a else a["frame"]
    for row in range(len(frames)):
        if not ref_usable(a, row, order):
            left_out += 1
            continue
        frame = int(frames[row])
        for p in order:
            ticks += struct.pack("<I", replays.CAR_SIZE) + ref_car(frame, a, row, p)
        ticks += struct.pack("<I", replays.PHYS_SIZE) + ref_phys(
            frame,
            a["ball_position"][row],
            a["ball_rotation"][row],
            a["ball_velocity"][row],
            a["ball_angular_velocity"][row],
        )
        count += 1
        kickoffs += ref_kickoff(a, row, order)
    low = [
        (o - s / 2) * replays.UU_TO_BT
        for o, s in zip(replays.OCTANE_OFFSET, replays.OCTANE_SIZE, strict=True)
    ]
    high = [
        (o + s / 2) * replays.UU_TO_BT
        for o, s in zip(replays.OCTANE_OFFSET, replays.OCTANE_SIZE, strict=True)
    ]
    header = struct.pack("<4sBIII6fI", b"RLPR", 0, 6, 28, len(order), *low, *high, count)
    return replays.Converted(header + bytes(ticks), count, kickoffs, left_out)


def random_arrays(
    seed: int, rows: int = 240, per_side: int = 2, optional: bool = True, gaps: float = 1.0
) -> dict[str, Any]:
    """replicar-shaped arrays with every awkward value: unknown flags (-1), NaN positions and boost, absent
    and demolished cars, boost outside 0-100, air times around the flip window, a few kickoff rows; with
    `optional` false, the columns a reader may lack (status, demolished, wheel contacts) are left out.
    `gaps` scales how often a row cannot be used (a NaN position, an absent or demolished car)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cars = 2 * per_side

    def quats(*shape: int) -> Any:
        q = rng.normal(size=(*shape, 4)).astype(np.float32)
        return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)

    def flags(*shape: int) -> Any:
        return rng.choice([-1, 0, 1], size=shape, p=[0.05, 0.5, 0.45]).astype(np.int64)

    def floats(*shape: int, scale: float = 1000.0) -> Any:
        return (rng.normal(size=shape) * scale).astype(np.float32)

    a: dict[str, Any] = {"sim_tick": np.arange(rows, dtype=np.int64) + 7000}
    a["ball_position"] = floats(rows, 3)
    a["ball_position"][rng.random(rows) < 0.03 * gaps, 1] = np.nan
    a["ball_rotation"], a["ball_velocity"], a["ball_angular_velocity"] = (
        quats(rows),
        floats(rows, 3),
        floats(rows, 3, scale=3),
    )
    a["car_position"] = floats(rows, cars, 3)
    a["car_position"][rng.random((rows, cars)) < 0.02 * gaps, 0] = np.nan
    a["car_rotation"] = quats(rows, cars)
    a["car_velocity"], a["car_angular_velocity"] = floats(rows, cars, 3), floats(rows, cars, 3, scale=3)
    a["car_flip_relative_torque"] = floats(rows, cars, 3, scale=1)
    for name in (
        "is_on_ground",
        "has_jumped",
        "has_double_jumped",
        "has_flipped",
        "is_flipping",
        "is_jumping",
    ):
        a[f"car_{name}"] = flags(rows, cars)
    a["car_flip_time"] = floats(rows, cars, scale=0.5)
    a["car_air_time_since_jump"] = (1.25 + rng.normal(size=(rows, cars)) * 0.3).astype(np.float32)
    a["car_air_time_since_jump"][rng.random((rows, cars)) < 0.05] = np.nan
    a["car_jump_ticks"] = rng.integers(-1, 300, size=(rows, cars))
    a["car_boost"] = (rng.random((rows, cars)) * 140 - 20).astype(np.float32)
    a["car_boost"][rng.random((rows, cars)) < 0.03] = np.nan
    for name in ("throttle", "steer", "pitch", "yaw", "roll"):
        a[f"car_controls_{name}"] = rng.uniform(-1, 1, size=(rows, cars)).astype(np.float32)
    for name in ("jump", "boost", "handbrake"):
        a[f"car_controls_{name}"] = flags(rows, cars)
    if optional:
        a["car_status"] = rng.choice(
            ["present", "absent", ""], size=(rows, cars), p=[1 - 0.04 * gaps, 0.02 * gaps, 0.02 * gaps]
        ).astype(object)
        a["car_is_demoed"] = rng.choice(
            [-1, 0, 1], size=(rows, cars), p=[0.02, 0.98 - 0.02 * gaps, 0.02 * gaps]
        ).astype(np.int64)
        for k in range(4):
            a[f"car_wheel_{k}_contact"] = flags(rows, cars)
    # A few kickoff rows: the ball at rest in the centre, every car on the ground on its spot.
    for row in rng.choice(rows, size=6, replace=False):
        a["ball_position"][row] = (1.0, -2.0, 93.0)
        a["ball_velocity"][row] = 0.0
        spots = [4608.0, 3840.0, 2560.0]
        for c in range(cars):
            a["car_position"][row, c] = (100.0 * c, (-1) ** c * spots[c % 3], 17.0)
            a["car_velocity"][row, c] = 0.0
            a["car_is_on_ground"][row, c] = 1
            if optional:
                a["car_status"][row, c] = "present"
                a["car_is_demoed"][row, c] = 0
    return a


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("optional", [True, False])
def test_the_numpy_writer_gives_the_same_bytes_as_packing_tick_by_tick(seed: int, optional: bool) -> None:
    arrays = random_arrays(seed, per_side=1 + seed % 3, optional=optional)
    players = [{"index": k, "team": (k + seed) % 2} for k in range(arrays["car_position"].shape[1])]
    new, old = replays.recording(arrays, players), ref_recording(arrays, players)
    assert old.ticks > 100 and old.rows_left_out > 0 and old.kickoffs > 0, "the awkward cases are there"
    assert (new.ticks, new.kickoffs, new.rows_left_out) == (old.ticks, old.kickoffs, old.rows_left_out)
    assert new.data == old.data


def stand_in_tool(folder: Path) -> Path:
    """A `replicar convert <replay> -o <out> --meshes <m>` that copies a prepared file, keeps a log of what it
    was asked, and fails on a replay named 'broken'."""
    tool = folder / "replicar"
    tool.write_text(
        "#!/bin/sh\n"
        f'echo "$2" >> "{folder}/calls.log"\n'
        'case "$2" in\n'
        '  *broken*) echo "could not decode the replay" >&2; exit 3;;\n'
        f'  *two*) cp "{folder}/two.parquet" "$4";;\n'
        f'  *) cp "{folder}/one.parquet" "$4";;\n'
        "esac\n",
        encoding="utf-8",
    )
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return tool


def replay_folder(tmp_path: Path, names: list[str]) -> Path:
    """Empty .replay files (the stand-in tool never reads them), the first name the newest."""
    demos = tmp_path / "DemosEpic"
    demos.mkdir()
    for age, name in enumerate(names):
        replay = demos / f"{name}.replay"
        replay.write_bytes(b"")
        os.utime(replay, (1_000_000 - age, 1_000_000 - age))
    return demos


def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir()
    replicar_file(tools / "one.parquet")
    replicar_file(tools / "two.parquet", per_side=2)
    monkeypatch.setattr(replays, "replicar_tool", lambda: stand_in_tool(tools))
    (tmp_path / "meshes" / "soccar").mkdir(parents=True)
    return tools


def converted_calls(tools: Path) -> list[str]:
    log = tools / "calls.log"
    calls = [Path(c).stem for c in log.read_text().split()] if log.exists() else []
    log.unlink(missing_ok=True)
    return calls


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")
def test_the_newest_replays_are_converted_until_the_limit_and_every_size_is_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = stand_in(tmp_path, monkeypatch)
    demos = replay_folder(tmp_path, ["a_two", "b_broken", "c_one", "d_one", "e_one", "f_two"])
    out, meshes = tmp_path / "replays", tmp_path / "meshes"
    heard: list[str] = []
    report = replays.convert_replays([demos], 1, 2, meshes, out, heard.append)
    assert report.found == 6 and report.kept == 2
    assert [p.name for p in report.recordings] == ["c_one.rlpr.zst", "d_one.rlpr.zst"], "newest first"
    assert sorted(f.name for f in (out / "1v1").iterdir()) == ["c_one.rlpr.zst", "d_one.rlpr.zst"]
    assert report.team_sizes == {"2v2": 1, "1v1": 2} and report.failed == ["could not decode the replay"]
    assert not (out / "converted").exists(), "the converted files are removed once used"
    assert heard == [
        f"  converting replay {k} (at most 10 this start): {r} of 2 1v1 replays ready"
        for k, r in ((1, 0), (2, 0), (3, 0), (4, 1))
    ]
    assert report.converted == 4 and not report.stopped_early
    text = replays.describe(report, 1)
    assert "1v1 recordings kept: 2" in text and "could not decode the replay (x1)" in text
    assert "p0" not in text, "no player's name in the report"
    assert converted_calls(tools) == ["a_two", "b_broken", "c_one", "d_one"]

    again = replays.convert_replays([demos], 1, 3, meshes, out)
    assert again.kept == 3 and converted_calls(tools) == ["b_broken", "e_one"], (
        "kept and other-size replays are not redone; a failed one is tried again"
    )
    two = replays.convert_replays([demos], 2, 5, meshes, out)
    assert [p.name for p in two.recordings] == ["a_two.rlpr.zst", "f_two.rlpr.zst"]
    assert converted_calls(tools) == ["a_two", "b_broken", "f_two"], (
        "a 2v2 seen by a 1v1 run is converted again for a 2v2 run (only its size was kept); the 1v1 ones not"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")
def test_conversion_stops_after_five_times_the_limit_when_the_size_is_rare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = stand_in(tmp_path, monkeypatch)
    demos = replay_folder(tmp_path, [*(f"{k:02d}_two" for k in range(12)), "z_one"])
    heard: list[str] = []
    report = replays.convert_replays([demos], 1, 2, tmp_path / "meshes", tmp_path / "replays", heard.append)
    assert report.kept == 0 and report.stopped_early and report.converted == 10
    assert len(converted_calls(tools)) == 10 and len(heard) == 10, "a line for every conversion"
    assert "stopped at the conversion cap" in replays.describe(report, 1)
    later = replays.convert_replays([demos], 1, 2, tmp_path / "meshes", tmp_path / "replays")
    assert converted_calls(tools) == ["10_two", "11_two", "z_one"], "the next start goes on where it stopped"
    assert later.kept == 1 and not later.stopped_early


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")
def test_a_replay_without_a_kickoff_is_never_converted_again_and_duplicates_count_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = stand_in(tmp_path, monkeypatch)
    replicar_file(tools / "one.parquet")
    table = pq.read_table(tools / "one.parquet")
    moved_ball = pa.array([500.0] * table.num_rows)  # never at rest in the centre: no kickoff tick
    table = table.set_column(table.column_names.index("ball_position_x"), "ball_position_x", moved_ball)
    pq.write_table(
        table.replace_schema_metadata(pq.read_metadata(tools / "one.parquet").metadata), tools / "one.parquet"
    )
    demos = replay_folder(tmp_path, ["a_one"])
    steam = tmp_path / "Demos"
    steam.mkdir()
    (steam / "a_one.replay").write_bytes(b"")
    out = tmp_path / "replays"
    first = replays.convert_replays([demos, steam], 1, 5, tmp_path / "meshes", out)
    assert first.found == 1, "the same file name in both folders counts once"
    assert first.failed == ["no kickoff tick, so the teams cannot be told apart"]
    assert converted_calls(tools) == ["a_one"]
    for wanted in (1, 2):
        replays.convert_replays([demos], wanted, 5, tmp_path / "meshes", out)
    assert converted_calls(tools) == [], "remembered for every size"


def test_a_compressed_recording_reads_as_the_plain_one(tmp_path: Path) -> None:
    replicar_file(tmp_path / "match.parquet")
    data = replays.recording(*replays.read_replicar(tmp_path / "match.parquet")).data
    (tmp_path / "plain.rlpr").write_bytes(data)
    packed = replays.compress(data)
    assert len(packed) * 5 < len(data), "zstd shrinks a recording several times"
    (tmp_path / "packed.rlpr.zst").write_bytes(packed)
    config = parse_toml(SETTINGS, NitroConfig, file=tmp_path / "bc.toml", base_dir=tmp_path)
    plain, zipped = (pretrain.load([tmp_path / name], config) for name in ("plain.rlpr", "packed.rlpr.zst"))
    assert torch.equal(plain.obs, zipped.obs) and torch.equal(plain.label, zipped.label)


def test_missing_meshes_or_converter_are_said_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demos = replay_folder(tmp_path, ["a"])
    with pytest.raises(replays.ReplayError, match="meshes were not found"):
        replays.convert_replays([demos], 1, 1, tmp_path / "nowhere", tmp_path / "replays")
    monkeypatch.setattr(replays, "REPLICAR_ROOT", tmp_path / "no-replicar")
    monkeypatch.setattr(replays, "find_cargo", lambda: None)
    with pytest.raises(replays.ReplayError, match="cargo is needed"):
        replays.replicar_tool()
    assert replays.convert_replays([tmp_path / "empty"], 1, 1, tmp_path, tmp_path / "replays").found == 0


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")
def test_the_tools_next_step_reads_its_conversions_as_replicars_wherever_they_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nitro pretrain's --rows auto knows replicar's rows by the project's replays folder, so the
    tool's next step names them for an --out elsewhere too."""
    stand_in(tmp_path, monkeypatch)
    demos = replay_folder(tmp_path, ["a_one"])
    out = tmp_path / "elsewhere"
    argv = ["--from", str(demos), "--limit", "1", "--meshes", str(tmp_path / "meshes"), "--out", str(out)]
    assert tool.main(argv) == 0
    assert f"Next: nitro pretrain <settings> {out / '1v1'} --rows replicar" in capsys.readouterr().out


def test_the_cloning_check_measures_on_held_back_recordings_and_names_no_paths(tmp_path: Path) -> None:
    folder = tmp_path / "rlpr"
    folder.mkdir()
    replicar_file(tmp_path / "match.parquet")
    data = replays.recording(*replays.read_replicar(tmp_path / "match.parquet")).data
    for name in "abc":
        (folder / f"{name}.rlpr").write_bytes(data)
    settings = tmp_path / "bc.toml"
    settings.write_text(SETTINGS, encoding="utf-8")
    lines = tool.clone_check(settings, sorted(folder.iterdir()), epochs=2)
    assert lines[0].startswith("Cloning check (bc.toml, 2 passes)") and "from 3 recordings" in lines[0]
    assert "commonest action" in lines[1] and lines[-1].startswith("  pass 2:")
    report = replays.Report(looked_in=[tmp_path / "Users" / "someone" / "Documents" / "TAGame" / "DemosEpic"])
    assert "someone" not in replays.describe(report, 1), "the Windows user name stays out of the report"


def test_on_windows_the_converter_runs_without_a_window_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started from the dashboard, nitro has no console, so each converter run opened a window of its own
    (about 40 flashing windows)."""
    seen: list[dict[str, Any]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        seen.append(kwargs)
        return subprocess.CompletedProcess(command, 1, "", "error: not a replay")

    monkeypatch.setattr(replays.subprocess, "run", run)
    monkeypatch.setattr(replays.sys, "platform", "win32")
    assert replays.convert(tmp_path / "t", tmp_path / "a.replay", tmp_path / "a.lean", tmp_path) == (
        "error: not a replay"
    )
    assert seen[0]["creationflags"] == 0x08000000  # CREATE_NO_WINDOW
    monkeypatch.setattr(replays.sys, "platform", "linux")
    replays.convert(tmp_path / "t", tmp_path / "a.replay", tmp_path / "a.lean", tmp_path)
    assert "creationflags" not in seen[1]


# A match as replicar really writes it, for the corrected samples.
FIRST_TICK = 9000
DECISIONS = 19  # tick_skip 8: decision k sees the frame FIRST_TICK + 8k
JUMP_ROW, DODGE_ROW = 8 * 12 + 1, 8 * 14 + 1  # the rows where jump is pressed
LANDED_ROW = 8 * 15 + 5
TWICE_ROW = 20  # written twice, as replicar sometimes does
PREVIOUS_STEER = 9 + 34 + 1  # the native observation: the ball, the pads, then the previous action's controls


def blue_steer(k: int) -> float:
    """Blue's steering at decision k on the ground; Orange's is the opposite."""
    return {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.6, 5: 0.6, 6: -1.0, 7: -1.0, 8: -1.0, 9: -0.6, 10: -0.6}.get(k, 0.0)


def replicar_match() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A 1v1 as replicar writes it, player 0 Orange: a row holds the controls applied in the step after it, so
    decision k's controls (action delay 1) are the rows of the 8 frames after it sees; yaw is 0 on the ground
    and fitted in the air, where steer is the raw stick; is_jumping is set from the row after a press; a side
    dodge's way is in yaw with roll 0; one row is written twice. Blue turns right at full lock (decisions
    1-3) and part lock (4-5), then left (6-10); Orange the opposite way. Both jump at decision 12, hold a turn
    the fit reads as yaw only at 13, dodge to the side at 14 (Blue right, Orange left) and land before 16."""
    import numpy as np

    rows = 8 * DECISIONS + 2
    row = np.arange(rows)
    block = (row - 1) // 8  # the decision whose controls the row holds
    players = [{"index": 0, "team": 1, "name": "orange"}, {"index": 1, "team": 0, "name": "blue"}]
    side = np.array([-1.0, 1.0])  # Orange, Blue
    shape = (rows, 2)
    air = (row > JUMP_ROW + 1) & (row < LANDED_ROW)
    jumped = (row > JUMP_ROW) & (row < LANDED_ROW)
    flipped = (row > DODGE_ROW) & (row < LANDED_ROW)
    pressed = ((row >= JUMP_ROW) & (row < JUMP_ROW + 3)) | ((row >= DODGE_ROW) & (row < DODGE_ROW + 3))
    steer = np.array([blue_steer(int(k)) for k in block])[:, None] * side * ~air[:, None]
    yaw = np.zeros(shape)
    yaw[block == 13] = side  # the stick held into a turn: the fit reads yaw, the raw steer stays 0
    stick = (row >= DODGE_ROW) & (row < DODGE_ROW + 2)
    yaw[stick], steer[stick] = side, side  # the dodge's way: in yaw (and the raw stick), roll 0

    def per_car(values: Any, dtype: Any = np.float32) -> Any:
        return np.repeat(np.asarray(values)[:, None], 2, axis=1).astype(dtype)

    moving = row > 10
    a: dict[str, Any] = {"sim_tick": (FIRST_TICK + row).astype(np.int64)}
    a["ball_position"] = np.tile(np.array([0.0, 0.0, 93.15], np.float32), (rows, 1))
    a["ball_rotation"] = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (rows, 1))
    a["ball_velocity"] = np.zeros((rows, 3), np.float32)
    a["ball_angular_velocity"] = np.zeros((rows, 3), np.float32)
    position = np.zeros((rows, 2, 3), np.float32)
    position[:, 0] = (2048.0, 2560.0, 17.0)
    position[:, 1, 0], position[:, 1, 1], position[:, 1, 2] = -2048.0, -2560.0 + 7.5 * row * moving, 17.0
    a["car_position"] = position
    a["car_rotation"] = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (rows, 2, 1))
    velocity = np.zeros((rows, 2, 3), np.float32)
    velocity[:, 1, 1] = 900.0 * moving
    a["car_velocity"] = velocity
    a["car_angular_velocity"] = np.zeros((rows, 2, 3), np.float32)
    a["car_flip_relative_torque"] = np.zeros((rows, 2, 3), np.float32)
    a["car_status"] = np.full(shape, "present", dtype=object)
    a["car_is_demoed"] = np.zeros(shape, np.int64)
    a["car_is_on_ground"] = per_car(~air, np.int64)
    a["car_has_jumped"] = per_car(jumped, np.int64)
    a["car_has_double_jumped"] = np.zeros(shape, np.int64)
    a["car_has_flipped"] = per_car(flipped, np.int64)
    a["car_is_flipping"] = per_car(flipped, np.int64)
    a["car_is_jumping"] = per_car((row > JUMP_ROW) & (row <= JUMP_ROW + 3), np.int64)
    a["car_flip_time"] = per_car(np.where(flipped, (row - DODGE_ROW) / 120.0, 0.0))
    a["car_air_time_since_jump"] = per_car(np.where(jumped, (row - JUMP_ROW) / 120.0, 0.0))
    a["car_jump_ticks"] = per_car(np.where(jumped, np.minimum(row - JUMP_ROW, 24), 0), np.int64)
    a["car_boost"] = np.full(shape, 33.0, np.float32)
    a["car_controls_throttle"] = np.ones(shape, np.float32)
    a["car_controls_steer"] = steer.astype(np.float32)
    a["car_controls_pitch"] = np.zeros(shape, np.float32)
    a["car_controls_yaw"] = yaw.astype(np.float32)
    a["car_controls_roll"] = np.zeros(shape, np.float32)
    a["car_controls_jump"] = per_car(pressed, np.int64)
    a["car_controls_boost"] = np.zeros(shape, np.int64)
    a["car_controls_handbrake"] = np.zeros(shape, np.int64)
    order = np.insert(row, TWICE_ROW + 1, TWICE_ROW)
    return {name: values[order] for name, values in a.items()}, players


def match_config(tmp_path: Path) -> NitroConfig:
    """Tick skip 8 and the native observation, as the match's rows are laid out (DECISIONS,
    PREVIOUS_STEER)."""
    text = SETTINGS + '[env]\ntick_skip = 8\nobservation = "native"\n'
    return parse_toml(text, NitroConfig, file=tmp_path / "bc.toml", base_dir=tmp_path)


def by_decision(first: Any, obs: Any) -> tuple[list[list[list[float]]], Any]:
    """The first choices' controls and the observations' previous steer, [decision][car], Blue first."""
    controls = [[_core.action_controls(int(a)) for a in pair] for pair in first.reshape(-1, 2).tolist()]
    return controls, obs[:, PREVIOUS_STEER].reshape(-1, 2)


def match_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, list[list[list[float]]], Any]]:
    """The match's samples by both paths: its recording read by pretrain.load from the replays folder (as
    the converters leave it), and the large mode's straight from replicar's columns."""
    import numpy as np

    arrays, players = replicar_match()
    config = match_config(tmp_path)
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    file = tmp_path / "replays" / "1v1" / "match.rlpr.zst"
    file.parent.mkdir(parents=True)
    file.write_bytes(replays.compress(replays.recording(arrays, players).data))
    loaded = pretrain.load([file], config)
    settings = pretrain.copy_settings(config, replicar_rows=True)
    part = _core.replay_samples(*replays.core_columns(arrays, players), "the match", settings, 0.1)
    obs, _, index, _ = replays.compact_arrays(part)
    return [
        ("recording", *by_decision(loaded.label.argmax(1).numpy(), loaded.obs.numpy())),
        ("columns", *by_decision(index[:, 0], obs.astype(np.float32))),
    ]


STEER, YAW, ROLL, JUMP = 1, 3, 4, 5  # in a first choice's controls


def test_corrected_samples_read_replicars_rows_as_the_game_played_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both paths: turns both ways at full and part lock steer that way, the previous action follows,
    the jump press labels the decision before it, and the side dodge goes to its side."""
    paths = match_samples(tmp_path, monkeypatch)
    for path, controls, previous in paths:
        assert len(controls) == DECISIONS, path
        for k in range(1, 11):
            want = [blue_steer(k), -blue_steer(k)]
            assert [math.copysign(1, w) for w in want] == [math.copysign(1, c[STEER]) for c in controls[k]]
            assert all(c[STEER] != 0 for c in controls[k]), (path, k, "turns, not straight on")
        assert all(c[STEER] == 0 for k in (0, 11) for c in controls[k]), path
        assert previous[0].tolist() == [0.0, 0.0], path
        for k in range(1, DECISIONS):
            assert previous[k].tolist() == [c[STEER] for c in controls[k - 1]], (path, k)
        assert previous[2].tolist() == [1.0, -1.0] and previous[7].tolist() == [-1.0, 1.0], path
        assert [[c[JUMP] for c in controls[k]] for k in (11, 12, 13)] == [[0, 0], [1, 1], [0, 0]], path
        assert [c[YAW] for c in controls[13]] == [1.0, -1.0], (path, "the fitted yaw, not the raw stick")
        assert [(c[JUMP], c[ROLL], c[2]) for c in controls[14]] == [(1, 1.0, 0.0), (1, -1.0, 0.0)], path
    assert paths[0][1] == paths[1][1] and (paths[0][2] == paths[1][2]).all(), "both paths alike"


def test_where_a_recording_lies_says_what_its_rows_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pretrain.load's rows: in the replays folder a recording is replicar's; elsewhere one the bot or
    RocketSim recorded, unless rows says otherwise."""
    arrays, players = replicar_match()
    data = replays.compress(replays.recording(arrays, players).data)
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    inside, outside = tmp_path / "replays" / "1v1" / "m.rlpr.zst", tmp_path / "mine" / "m.rlpr.zst"
    for file in (inside, outside):
        file.parent.mkdir(parents=True)
        file.write_bytes(data)
    config = match_config(tmp_path)

    def jumps(file: Path, rows: str = "auto") -> list[int]:
        label = pretrain.load([file], config, rows=rows).label.argmax(1).reshape(-1, 2)[:, 0]
        return [k for k, a in enumerate(label.tolist()) if _core.action_controls(a)[JUMP]]

    assert jumps(inside) == jumps(outside, "replicar") == [12, 14]
    assert jumps(outside) == jumps(inside, "recorded") == [11, 13], "read as recorded: a decision early"
    with pytest.raises(pretrain.PretrainError, match="rows must be"):
        pretrain.load([inside], config, rows="replays")
