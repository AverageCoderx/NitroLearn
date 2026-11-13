"""Rocket League replays as recordings `nitro pretrain` reads.

The replays come from the game's own folders (Documents\\My Games\\Rocket League\\TAGame\\DemosEpic for
the Epic Games version, Demos for Steam) or any folder of .replay files. Rolv Arild's replicar (MIT; its
command-line tool is built once into .scratch\\replicar) re-simulates each one in RocketSim at 120 ticks a
second and fills in what a replay does not hold (the exact ticks of presses, the air controls); each of its
files becomes a compressed RLPR recording in replays\\<n>v<n>, the format `nitro pretrain` reads. Every
replay is converted once: replays\\sizes.txt remembers the team size of each one seen, so a replay of
another size is never converted again. Replays hold other players' names: replays\\ is ignored by git, and
the report holds counts only. `[pretrain]` in the settings runs all of this when a run starts;
tools/replays.py runs it alone. The large copy keeps each conversion in replays\\store, which
every later copy reads first; `nitro convert-replays` converts into it alone.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nitro import prebuilt
from nitro.doctor import ADD_RUST
from nitro.toolchain import cargo_env, find_cargo

ROOT = Path(__file__).resolve().parents[2]
REPLICAR_VERSION = "2.2.1"
REPLICAR_ROOT = ROOT / ".scratch" / "replicar"
NITRO_REPLICAR_SOURCE = ROOT / "crates" / "nitro_replicar"
NITRO_REPLICAR_ROOT = ROOT / ".scratch" / "nitro-replicar"
OUT = ROOT / "replays"

# RLPR version 6, as `nitro_core::rlpr` reads it: sizes in bytes.
PHYS_SIZE = 332
CAR_SIZE = 588
UU_TO_BT = 0.02
# Every car gets the Octane's RocketSim body: the format has one hitbox per recording; cloning never reads it.
OCTANE_SIZE = (120.507, 86.6994, 38.6591)
OCTANE_OFFSET = (13.8757, 0.0, 20.755)
# RocketSim: a car without a flip used and within this long after its jump may still flip.
DOUBLE_JUMP_MAX_DELAY = 1.25
TICK_SECONDS = 1.0 / 120.0


def documents() -> Path:
    """The user's Documents folder (wherever Windows keeps it, OneDrive included)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Guid(ctypes.Structure):
            _fields_ = [
                ("a", wintypes.DWORD),
                ("b", wintypes.WORD),
                ("c", wintypes.WORD),
                ("d", ctypes.c_ubyte * 8),
            ]

        # FOLDERID_Documents {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
        folder = Guid(
            0xFDD39AD0, 0x238F, 0x46AF, (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7)
        )
        path = ctypes.c_wchar_p()
        shell = ctypes.windll.shell32  # type: ignore[attr-defined, unused-ignore]  # Windows only
        if shell.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(path)) == 0:
            found = Path(path.value or "")
            ctypes.windll.ole32.CoTaskMemFree(path)  # type: ignore[attr-defined, unused-ignore]  # Windows only
            return found
    return Path.home() / "Documents"


def game_folders(base: Path) -> list[Path]:
    return [base / "My Games" / "Rocket League" / "TAGame" / name for name in ("DemosEpic", "Demos")]


def newest_in_turn(folders: list[Path]) -> list[Path]:
    """The .replay files of `folders`, each folder's newest first, taken from the folders in turn, so a plan
    over several folders (one per game size) mixes them; a file name in two folders counts once."""
    out: list[Path] = []
    seen: set[str] = set()
    for group in itertools.zip_longest(*(newest_replays([folder]) for folder in folders)):
        for replay in group:
            if replay is not None and replay.stem not in seen:
                seen.add(replay.stem)
                out.append(replay)
    return out


def newest_replays(folders: list[Path]) -> list[Path]:
    """The .replay files in `folders`, newest first; a file name in two folders (a player who moved from
    Steam to Epic) counts once."""
    found = [p for folder in folders if folder.is_dir() for p in folder.glob("*.replay")]
    unique: dict[str, Path] = {}
    for replay in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
        unique.setdefault(replay.stem, replay)
    return list(unique.values())


@dataclass
class Converted:
    data: bytes
    ticks: int
    kickoffs: int
    rows_left_out: int


@dataclass
class Report:
    looked_in: list[Path] = field(default_factory=list)
    found: int = 0
    kept: int = 0
    ticks: int = 0
    kickoffs: int = 0
    rows_left_out: int = 0
    team_sizes: Counter[str] = field(default_factory=Counter)
    failed: list[str] = field(default_factory=list)  # reasons only, never names
    recordings: list[Path] = field(default_factory=list)  # of the wanted size, newest replay first
    converted: int = 0  # replays run through replicar on this call
    stopped_early: bool = False  # the conversion cap ended the search before the folder was gone through


class ReplayError(RuntimeError):
    """The replays cannot be converted at all (no converter, no pyarrow, no meshes); the message says why."""


def _rotation_rows(q: Any) -> Any:
    """Unit quaternions (x, y, z, w; any leading shape) as the matrices whose columns are forward, right and
    up, each stored row by row (as RocketSim's recordings and ours store it; replicar's `rotation_matrix` is
    the same matrix): [..., 9], in float64."""
    import numpy as np

    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )  # fmt: skip


def _layouts(cars: int) -> tuple[Any, Any]:
    """RLPR version 6's physics record and one tick of `cars` cars (each record after its u32 size) as
    NumPy structured types, so a whole recording is written at once."""
    import numpy as np

    phys = np.dtype(
        {
            "names": ["frame", "pos", "rot", "vel", "ang_vel"],
            "formats": ["<u4", ("<f4", 3), ("<f4", 9), ("<f4", 3), ("<f4", 3)],
            "offsets": [0, 4, 16, 52, 64],
            "itemsize": PHYS_SIZE,
        }
    )
    names = ["phys", "on_ground", "is_jumping", "is_flipping", "jump_time", "flip_time", "has_jumped"]
    names += ["used_second_jump", "has_flip", "torque", "boost", "controls", "jump", "use_boost", "handbrake"]
    formats: list[Any] = [phys, "?", "?", "?", "<f4", "<f4", "?", "?", "?", ("<f4", 3), "<f4", ("<f4", 5)]
    formats += ["?", "?", "?"]
    offsets = [0, 332, 333, 334, 336, 340, 344, 345, 346, 348, 360, 368, 388, 389, 390]
    for k in range(4):  # each wheel: 8 bytes, contact, 3 bytes, the contact normal, 24 bytes
        names += [f"wheel_{k}", f"normal_{k}"]
        formats += ["?", ("<f4", 3)]
        offsets += [392 + 48 * k + 8, 392 + 48 * k + 12]
    car = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": CAR_SIZE})
    tick_names, tick_formats, tick_offsets = [], [], []
    for c in range(cars):
        tick_names += [f"size_{c}", f"car_{c}"]
        tick_formats += ["<u4", car]
        tick_offsets += [c * (4 + CAR_SIZE), c * (4 + CAR_SIZE) + 4]
    tick_names += ["ball_size", "ball"]
    tick_formats += ["<u4", phys]
    tick_offsets += [cars * (4 + CAR_SIZE), cars * (4 + CAR_SIZE) + 4]
    tick = np.dtype(
        {
            "names": tick_names,
            "formats": tick_formats,
            "offsets": tick_offsets,
            "itemsize": cars * (4 + CAR_SIZE) + 4 + PHYS_SIZE,
        }
    )
    return phys, tick


def _fill_phys(out: Any, frames: Any, pos: Any, rot: Any, vel: Any, ang_vel: Any) -> None:
    out["frame"] = frames
    out["pos"] = pos
    out["rot"] = _rotation_rows(rot)
    out["vel"] = vel
    out["ang_vel"] = ang_vel


def _flag(a: dict[str, Any], name: str, rows: Any, p: int) -> Any:
    return a[f"car_{name}"][rows, p] == 1


def _usable_rows(a: dict[str, Any], players: list[int]) -> Any:
    """Rows where the ball is known and every car present, not demolished and placed."""
    import numpy as np

    usable = np.isfinite(a["ball_position"]).all(axis=1)
    for p in players:
        if "car_status" in a:
            usable &= ~np.isin(a["car_status"][:, p], ["", "absent"])
        if "car_is_demoed" in a:
            usable &= a["car_is_demoed"][:, p] != 1
        usable &= np.isfinite(a["car_position"][:, p, 0])
    return usable


def _kickoff_rows(a: dict[str, Any], rows: Any, players: list[int]) -> Any:
    """As `nitro_core::replay::is_kickoff`: the ball at rest in the centre, each car on the ground on its spot
    (the speed allowance included)."""
    import numpy as np

    ball = a["ball_position"][rows].astype(np.float64)
    speed = np.sqrt((a["ball_velocity"][rows].astype(np.float64) ** 2).sum(axis=1))
    kickoff = (np.abs(ball[:, 0]) < 5) & (np.abs(ball[:, 1]) < 5) & (speed < 5)
    for p in players:
        y = np.abs(a["car_position"][rows, p, 1].astype(np.float64))
        car_speed = np.sqrt((a["car_velocity"][rows, p].astype(np.float64) ** 2).sum(axis=1))
        on_spot = np.zeros(len(y), dtype=bool)
        for spot in (4608.0, 3840.0, 2560.0):
            on_spot |= np.abs(y - spot) < 60
        kickoff &= (a["car_is_on_ground"][rows, p] == 1) & (car_speed < 200) & on_spot
    return kickoff


def recording(a: dict[str, Any], players: list[dict[str, Any]]) -> Converted:
    """A replicar file's arrays (`replicar.read(...).arrays()`, a row per tick) and its header's players as an
    RLPR recording: Blue's cars first, every usable row (all cars present and not demolished) as one tick
    numbered by its simulation tick, so a left-out row breaks the run there. Written with NumPy all at once
    (byte for byte what packing tick by tick gave, tests/test_replays.py)."""
    import numpy as np

    order = [p["index"] for p in sorted(players, key=lambda p: (p["team"], p["index"]))]
    frames_all = a["sim_tick"] if "sim_tick" in a else a["frame"]
    usable = _usable_rows(a, order)
    rows = np.flatnonzero(usable)
    frames = np.asarray(frames_all)[rows]
    if len(frames) and (frames.min() < 0 or frames.max() > 0xFFFFFFFF):
        raise ValueError("a tick number outside what a recording can hold")
    _, tick = _layouts(len(order))
    out = np.zeros(len(rows), dtype=tick)
    for c, p in enumerate(order):
        out[f"size_{c}"] = CAR_SIZE
        car = out[f"car_{c}"]
        _fill_phys(
            car["phys"],
            frames,
            a["car_position"][rows, p],
            a["car_rotation"][rows, p],
            a["car_velocity"][rows, p],
            a["car_angular_velocity"][rows, p],
        )
        on_ground = _flag(a, "is_on_ground", rows, p)
        has_jumped = _flag(a, "has_jumped", rows, p)
        flipped, double = _flag(a, "has_flipped", rows, p), _flag(a, "has_double_jumped", rows, p)
        air = a["car_air_time_since_jump"][rows, p]
        car["on_ground"] = on_ground
        car["is_jumping"] = _flag(a, "is_jumping", rows, p)
        car["is_flipping"] = _flag(a, "is_flipping", rows, p)
        # The port counts jump ticks on the ground too; the format's jump time is 0 until a jump (replicar's
        # own mapping to RocketSim's state does the same).
        ticks = a["car_jump_ticks"][rows, p].astype(np.float64)
        car["jump_time"] = np.where(has_jumped, ticks / 120.0, 0.0)
        # The format's flip time is 0 unless the car flipped (it tells a dodge from a double jump); replicar
        # 2.1+ counts it as RocketSim 0.3 does, also on the ground and after a double jump.
        car["flip_time"] = np.where(flipped, a["car_flip_time"][rows, p], 0.0)
        car["has_jumped"] = has_jumped
        car["used_second_jump"] = double | flipped
        # RocketSim 0.3's has_flip_or_jump: the next tick's time counted first.
        car["has_flip"] = on_ground | (~flipped & ~double & (air + TICK_SECONDS < DOUBLE_JUMP_MAX_DELAY))
        car["torque"] = a["car_flip_relative_torque"][rows, p]
        boost = a["car_boost"][rows, p].astype(np.float64) / 100.0
        boost = np.where(boost < 0.0, 0.0, boost)
        car["boost"] = np.where(boost > 1.0, 1.0, boost)  # touching the ball stays 0: not read by cloning
        car["controls"] = np.stack(
            [a[f"car_controls_{n}"][rows, p] for n in ("throttle", "steer", "pitch", "yaw", "roll")], axis=-1
        )
        car["jump"] = _flag(a, "controls_jump", rows, p)
        car["use_boost"] = _flag(a, "controls_boost", rows, p)
        car["handbrake"] = _flag(a, "controls_handbrake", rows, p)
        up = _rotation_rows(a["car_rotation"][rows, p])[:, 2::3]
        for k in range(4):
            name = f"car_wheel_{k}_contact"
            contact = a[name][rows, p] == 1 if name in a else on_ground
            car[f"wheel_{k}"] = contact
            car[f"normal_{k}"] = np.where(contact[:, None], up, 0.0)
    out["ball_size"] = PHYS_SIZE
    _fill_phys(
        out["ball"],
        frames,
        a["ball_position"][rows],
        a["ball_rotation"][rows],
        a["ball_velocity"][rows],
        a["ball_angular_velocity"][rows],
    )
    kickoffs = int(_kickoff_rows(a, rows, order).sum())
    low = [(o - s / 2) * UU_TO_BT for o, s in zip(OCTANE_OFFSET, OCTANE_SIZE, strict=True)]
    high = [(o + s / 2) * UU_TO_BT for o, s in zip(OCTANE_OFFSET, OCTANE_SIZE, strict=True)]
    header = struct.pack("<4sBIII6fI", b"RLPR", 0, 6, 28, len(order), *low, *high, len(rows))
    return Converted(header + out.tobytes(), len(rows), kickoffs, int(len(usable) - len(rows)))


_VECTORS = {"position": "xyz", "velocity": "xyz", "angular_velocity": "xyz", "rotation": "xyzw"}
_CAR_VECTORS = {**_VECTORS, "flip_relative_torque": "xyz"}
_CAR_VALUES = (
    "status",
    "is_on_ground",
    "has_jumped",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
    "is_jumping",
    "is_demoed",
    "flip_time",
    "air_time_since_jump",
    "jump_ticks",
    "boost",
    *(f"wheel_{k}_contact" for k in range(4)),
    *(
        f"controls_{name}"
        for name in ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
    ),
)


def _column(table: Any, name: str) -> Any:
    """A replicar column as NumPy, as replicar's own reader gives it: floats with NaN, integers and flags with
    -1 when unknown, names as strings ('' when unknown)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    array = table.column(name).combine_chunks()
    scale = (table.schema.field(name).metadata or {}).get(b"scale")
    if pa.types.is_dictionary(array.type):
        array = array.cast(pa.string())
    if pa.types.is_string(array.type):
        return np.array(pc.fill_null(array, "").to_pylist(), dtype=object)
    if scale is not None or pa.types.is_floating(array.type):
        values = array.cast(pa.float64()).to_numpy(zero_copy_only=False)
        return (values * float(scale or 1.0)).astype(np.float32)
    return pc.fill_null(array.cast(pa.int64()), -1).to_numpy(zero_copy_only=False)


def read_replicar(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The columns of a replicar file that a recording needs, shaped as replicar's reader shapes them
    (`car_position`: rows x players x 3), and its header's players. Reads it with pyarrow alone: replicar's
    own Python reader needs Python 3.12."""
    import json

    import numpy as np
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    metadata = file.metadata.metadata or {}
    if b"replicar" not in metadata:
        raise ValueError("not a replicar file (no replicar header)")
    header = json.loads(metadata[b"replicar"])
    players = header["players"]
    indices = [p["index"] for p in players]
    # Only the columns used: about half of a file's.
    wanted = {"sim_tick", "frame"} | {f"ball_{n}_{c}" for n, parts in _VECTORS.items() for c in parts}
    for p in indices:
        wanted |= {f"car_{p}_{n}_{c}" for n, parts in _CAR_VECTORS.items() for c in parts}
        wanted |= {f"car_{p}_{n}" for n in _CAR_VALUES}
    table = file.read(columns=[n for n in file.schema_arrow.names if n in wanted])
    names = set(table.column_names)
    out: dict[str, Any] = {"sim_tick": _column(table, "sim_tick" if "sim_tick" in names else "frame")}
    for name, parts in _VECTORS.items():
        out[f"ball_{name}"] = np.stack([_column(table, f"ball_{name}_{c}") for c in parts], axis=-1)
    for name, parts in _CAR_VECTORS.items():
        out[f"car_{name}"] = np.stack(
            [np.stack([_column(table, f"car_{p}_{name}_{c}") for c in parts], axis=-1) for p in indices],
            axis=1,
        )
    for name in _CAR_VALUES:
        if all(f"car_{p}_{name}" in names for p in indices):
            out[f"car_{name}"] = np.stack([_column(table, f"car_{p}_{name}") for p in indices], axis=1)
    # Player indices become positions in these arrays.
    return out, [{**p, "index": k} for k, p in enumerate(players)]


def team_size(players: list[dict[str, Any]]) -> int:
    sides = Counter(p["team"] for p in players)
    return max(sides.values(), default=0)


def _no_window() -> dict[str, Any]:
    """On Windows, no console window of its own for a program run here. Started from the dashboard, nitro has
    no console, so every converter run opened one and closed it again (about 40 at
    once with 10 converting)."""
    if sys.platform != "win32":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _our_output() -> dict[str, Any]:
    """A build's output into this process's own: the terminal, or a dashboard job's console.log."""
    try:
        sys.stdout.flush()
        return {"stdout": sys.stdout.fileno(), "stderr": subprocess.STDOUT}
    except (AttributeError, OSError, ValueError):
        return {}


# What to do when a converter is neither ready-made nor buildable here.
_GET_RUST = f"run the installer again (it fetches the ready-made converters), or {ADD_RUST}"


def replicar_tool() -> Path:
    """replicar's command-line tool: the release's ready-made one, else built from crates.io the first time
    (a few minutes)."""
    tool = REPLICAR_ROOT / "bin" / ("replicar.exe" if sys.platform == "win32" else "replicar")
    if not tool.exists():
        shipped = prebuilt.program("replicar", REPLICAR_VERSION, ROOT)
        if shipped is not None:
            return shipped
        cargo = find_cargo()
        if cargo is None:
            raise ReplayError(
                "Rust's cargo is needed once to build the replay converter (replicar), and this folder has "
                f"no ready-made one: {_GET_RUST}"
            )
        print(
            f"Building the replay converter, replicar {REPLICAR_VERSION} (once, a few minutes)...", flush=True
        )
        try:
            subprocess.run(
                [
                    str(cargo),
                    "install",
                    "replicar-cli",
                    "--version",
                    REPLICAR_VERSION,
                    "--locked",
                    "--root",
                    str(REPLICAR_ROOT),
                ],
                check=True,
                env=cargo_env(),
                **_no_window(),
                **_our_output(),
            )
        except subprocess.CalledProcessError as err:
            raise ReplayError(f"building replicar failed (cargo exit code {err.returncode})") from None
    return tool


def convert(tool: Path, replay: Path, parquet: Path, meshes: Path) -> str | None:
    """None when it worked, else the reason: the tool's last line (which never holds a player's name), without
    the replay's folder or name."""
    done: subprocess.CompletedProcess[str] = subprocess.run(
        [str(tool), "convert", str(replay), "-o", str(parquet), "--meshes", str(meshes)],
        capture_output=True,
        text=True,
        check=False,
        **_no_window(),
    )
    if done.returncode == 0 and parquet.exists():
        return None
    lines = (done.stderr or done.stdout).strip().splitlines()
    if not lines:
        return f"exit code {done.returncode}"
    line = lines[-1]
    names = {str(replay), replay.as_posix(), str(replay.parent), replay.parent.as_posix()}
    for spelling in sorted(
        names | ({replay.stem} if len(replay.stem) >= 8 else set()), key=len, reverse=True
    ):
        line = line.replace(spelling, "<replay>")
    return line[:200]


def _source_stamp() -> str:
    return prebuilt.nitro_replicar_source(NITRO_REPLICAR_SOURCE.parents[1])


def nitro_replicar_tool() -> Path:
    """NitroLearn's own build of replicar (crates/nitro_replicar): the release's ready-made one while its
    source is unchanged, else built the first time (a few minutes) and again when its source changes. Not
    for this processor alone (`target-cpu=native`): that changed the states in their last bits (fused
    multiply-adds); the point is replicar's states exactly."""
    tool = (
        NITRO_REPLICAR_ROOT / "bin" / ("nitro-replicar.exe" if sys.platform == "win32" else "nitro-replicar")
    )
    stamp = NITRO_REPLICAR_ROOT / "built-from.txt"
    wanted = _source_stamp()
    if tool.exists() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == wanted:
        return tool
    shipped = prebuilt.program("nitro-replicar", wanted, ROOT)
    if shipped is not None:
        return shipped
    cargo = find_cargo()
    if cargo is None:
        raise ReplayError(
            "Rust's cargo is needed once to build the replay converter (nitro-replicar), and this folder has "
            f"no ready-made one for its code: {_GET_RUST}"
        )
    print(
        f"Building the replay converter, nitro-replicar (replicar {REPLICAR_VERSION}; once, a few minutes)",
        flush=True,
    )
    command = [str(cargo), "install", "--path", str(NITRO_REPLICAR_SOURCE), "--locked", "--force"]
    try:
        subprocess.run(
            [*command, "--root", str(NITRO_REPLICAR_ROOT)],
            check=True,
            env=cargo_env(),
            **_no_window(),
            **_our_output(),
        )
    except subprocess.CalledProcessError as err:
        raise ReplayError(f"building nitro-replicar failed (cargo exit code {err.returncode})") from None
    stamp.write_text(wanted + "\n", encoding="utf-8")
    return tool


# nitro-replicar's file (crates/nitro_replicar/src/main.rs): per car, its floats and small values, in order.
_LEAN_FLOATS = (
    ("position", 3),
    ("velocity", 3),
    ("angular_velocity", 3),
    ("rotation", 4),
    ("flip_relative_torque", 3),
    ("flip_time", 1),
    ("air_time_since_jump", 1),
    ("boost", 1),
    *((f"controls_{name}", 1) for name in ("throttle", "steer", "pitch", "yaw", "roll")),
)
_LEAN_SMALL = (
    "status",
    "is_on_ground",
    "has_jumped",
    "has_double_jumped",
    "has_flipped",
    "is_flipping",
    "is_jumping",
    "is_demoed",
    *(f"wheel_{k}_contact" for k in range(4)),
    "controls_jump",
    "controls_boost",
    "controls_handbrake",
)


def read_lean(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A nitro-replicar file as `read_replicar` gives a replicar file: the same arrays (`tests` check it on
    real replays), from plain arrays instead of Parquet columns."""
    return lean_arrays(path.read_bytes())


def lean_arrays(data: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A nitro-replicar file's bytes as `read_lean` reads them."""
    import numpy as np

    if data[:4] != b"NRPL" or struct.unpack_from("<I", data, 4)[0] != 1:
        raise ValueError("not a nitro-replicar file of version 1")
    size = struct.unpack_from("<I", data, 8)[0]
    header = json.loads(data[12 : 12 + size])
    at = 12 + size
    rows, cars = int(header["rows"]), int(header["cars"])
    per_car = 4 * sum(w for _, w in _LEAN_FLOATS) + len(_LEAN_SMALL) + 8
    if len(data) != at + rows * (8 + 13 * 4 + cars * per_car):
        raise ValueError("a nitro-replicar file of the wrong size (cut off?)")

    def take(dtype: str, shape: tuple[int, ...]) -> Any:
        nonlocal at
        array = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape)), offset=at).reshape(shape)
        at += array.nbytes
        return array

    out: dict[str, Any] = {"sim_tick": take("<i8", (rows,)).copy()}
    ball = take("<f4", (rows, 13))
    for name, (begin, end) in zip(_VECTORS, ((0, 3), (3, 6), (6, 9), (9, 13)), strict=True):
        out[f"ball_{name}"] = ball[:, begin:end].copy()
    # The cars as read_replicar orders them: by the header's player index.
    order = [int(p["index"]) for p in header["players"]]
    floats = take("<f4", (rows, cars, sum(w for _, w in _LEAN_FLOATS)))[:, order]
    begin = 0
    for name, width in _LEAN_FLOATS:
        part = floats[:, :, begin : begin + width]
        out[f"car_{name}"] = (part if name in _CAR_VECTORS else part[:, :, 0]).copy()
        begin += width
    small = take("i1", (rows, cars, len(_LEAN_SMALL)))[:, order]
    names = np.array([*header["status_names"], ""], dtype=object)
    codes = small[:, :, 0].view(np.uint8).astype(np.int64)
    out["car_status"] = names[np.where(codes < len(names) - 1, codes, len(names) - 1)]
    for k, name in enumerate(_LEAN_SMALL[1:], start=1):
        out[f"car_{name}"] = small[:, :, k].astype(np.int64)
    out["car_jump_ticks"] = take("<i8", (rows, cars))[:, order].copy()
    return out, [{**p, "index": k} for k, p in enumerate(header["players"])]


def compress(data: bytes) -> bytes:
    """A recording as a .rlpr.zst file's bytes (the core reads zstd frames; pyarrow writes them)."""
    import pyarrow as pa

    return bytes(pa.compress(data, codec="zstd", asbytes=True))


NO_KICKOFF = "no-kickoff"  # a replay whose teams cannot be told apart: never converted again


# The converted replays: each replay as nitro-replicar converted it, compressed, in its game size's
# folder, so no copy and no bot converts a replay twice. It holds no bot's settings: every bot makes its own
# samples from it in a moment. A new replicar version starts a new folder.
def store_folder() -> Path:
    """The store: replays\\store\\replicar-<version>, its game sizes' folders inside."""
    return OUT / "store" / f"replicar-{REPLICAR_VERSION}"


STORED = ".nrpl.zst"  # 8 bytes (the file's size, little-endian) and then a zstd frame of the file
STORE_MIN_FREE = 20 * 2**30  # bytes the disk keeps free: below this, conversions are no longer kept


def stored(stem: str, size: int, store: Path | None = None) -> Path:
    return (store or store_folder()) / f"{size}v{size}" / f"{stem}{STORED}"


def find_stored(
    stem: str, sizes: tuple[int, ...] = (1, 2, 3, 4), store: Path | None = None
) -> tuple[Path, int] | None:
    """The kept conversion of the replay `stem` among the game sizes `sizes`, with its size: in `store`, or
    without one in the store."""
    for size in sizes:
        path = stored(stem, size, store)
        if path.is_file():
            return path, size
    return None


def keep(data: bytes, stem: str, size: int, store: Path | None = None) -> bool:
    """Keeps a nitro-replicar file's bytes in the store (whole or not at all, as threads write side by side);
    False when the disk is nearly full."""
    import threading

    import pyarrow as pa

    folder = (store or store_folder()) / f"{size}v{size}"
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder).free < STORE_MIN_FREE + len(data):
        return False
    target = stored(stem, size, store)
    partial = target.with_name(f"{target.name}.{threading.get_ident()}.partial")
    packed = pa.Codec("zstd", compression_level=1).compress(data, asbytes=True)
    partial.write_bytes(struct.pack("<Q", len(data)) + bytes(packed))
    partial.replace(target)
    return True


def read_stored(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A kept conversion as `read_lean` reads the file it was made from."""
    import pyarrow as pa

    data = path.read_bytes()
    raw = pa.Codec("zstd").decompress(data[8:], struct.unpack_from("<Q", data)[0], asbytes=True)
    return lean_arrays(bytes(raw))


def store_summary(store: Path | None = None) -> dict[str, Any]:
    """The kept conversions: how many of each game size and the bytes they take, and the disk's free bytes."""
    store = store or store_folder()
    counts: dict[str, int] = {}
    used = 0
    for folder in sorted(f for f in store.glob("*v*") if f.is_dir()) if store.is_dir() else []:
        n = 0
        for entry in os.scandir(folder):
            if entry.name.endswith(STORED):
                n += 1
                used += entry.stat().st_size
        counts[folder.name] = n
    probe = store if store.is_dir() else OUT if OUT.is_dir() else ROOT
    return {"counts": counts, "bytes": used, "free": shutil.disk_usage(probe).free}


def has_kickoff(a: dict[str, Any], players: list[dict[str, Any]]) -> bool:
    """Whether the teams can be told apart (a kickoff tick), as `recording` and the copy need."""
    order = [p["index"] for p in sorted(players, key=lambda p: (p["team"], p["index"]))]
    import numpy as np

    rows = np.flatnonzero(_usable_rows(a, order))
    return bool(_kickoff_rows(a, rows, order).any())


class Sizes:
    """replays\\sizes.txt: the team size of every replay converted so far, by file name, so none is
    converted twice; "no-kickoff" marks the ones that can never become a recording."""

    def __init__(self, out: Path) -> None:
        self.file = out / "sizes.txt"
        self.known: dict[str, str] = {}
        if self.file.exists():
            for line in self.file.read_text(encoding="utf-8").splitlines():
                stem, _, kind = line.rpartition(" ")
                if stem:
                    self.known[stem] = kind

    def note(self, stem: str, kind: str) -> None:
        if self.known.get(stem) != kind:
            self.known[stem] = kind
            with self.file.open("a", encoding="utf-8") as file:
                file.write(f"{stem} {kind}\n")

    def other_size(self, stem: str, wanted: int) -> bool:
        """Known not to give a recording of `wanted` players a side."""
        kind = self.known.get(stem)
        return kind is not None and kind != str(wanted)


def convert_replays(
    sources: list[Path],
    wanted: int,
    limit: int,
    meshes: Path,
    out: Path = OUT,
    say: Callable[[str], None] | None = None,
) -> Report:
    """Recordings of the newest `limit` replays of `wanted` players a side in `sources`, converting the
    ones not converted before (at most five times `limit` of them per call, so a rare size does not take
    hours; `stopped_early` says when that ended the search); `say` hears the progress."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ReplayError(
            "reading converted replays needs pyarrow: run the installer again, or "
            ".\\.venv\\Scripts\\python -m pip install pyarrow"
        ) from None
    report = Report(looked_in=sources)
    replays = newest_replays(sources)
    report.found = len(replays)
    if not replays:
        return report
    if not (meshes / "soccar").is_dir():
        raise ReplayError(f"the arena meshes were not found in {meshes} (paths.collision_meshes)")
    out.mkdir(parents=True, exist_ok=True)
    sizes = Sizes(out)
    folder = out / f"{wanted}v{wanted}"
    folder.mkdir(exist_ok=True)
    for stale in folder.glob("*.partial"):  # a write cut off by a power cut
        stale.unlink()
    parquet_dir = out / "converted"
    tool: Path | None = None
    for replay in replays:
        if report.kept >= limit:
            break
        if report.converted >= limit * 5:
            report.stopped_early = True
            break
        target = folder / f"{replay.stem}.rlpr.zst"
        if target.exists():  # converted before
            report.team_sizes[f"{wanted}v{wanted}"] += 1
            report.kept += 1
            report.recordings.append(target)
            continue
        if sizes.other_size(replay.stem, wanted):
            continue
        if tool is None:
            tool = replicar_tool()
            parquet_dir.mkdir(exist_ok=True)
        report.converted += 1
        if say:
            say(
                f"  converting replay {report.converted} (at most {limit * 5} this start): {report.kept} of "
                f"{limit} {wanted}v{wanted} replays ready"
            )
        parquet = parquet_dir / f"{replay.stem}.parquet"
        problem = convert(tool, replay, parquet, meshes)
        if problem:
            report.failed.append(problem)
            continue
        try:
            arrays, players = read_replicar(parquet)
        except (OSError, ValueError, KeyError) as err:
            report.failed.append(f"unreadable converted file: {type(err).__name__} {str(err)[:150]}")
            continue
        finally:
            parquet.unlink(missing_ok=True)  # about 8 MB each; the recording is what cloning reads
        size = team_size(players)
        report.team_sizes[f"{size}v{size}"] += 1
        if size != wanted:
            sizes.note(replay.stem, str(size))
            continue
        made = recording(arrays, players)
        if made.kickoffs == 0:
            sizes.note(replay.stem, NO_KICKOFF)
            report.failed.append("no kickoff tick, so the teams cannot be told apart")
            continue
        sizes.note(replay.stem, str(size))
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(compress(made.data))
        partial.replace(target)
        report.ticks += made.ticks
        report.kickoffs += made.kickoffs
        report.rows_left_out += made.rows_left_out
        report.kept += 1
        report.recordings.append(target)
    if parquet_dir.is_dir() and not any(parquet_dir.iterdir()):
        parquet_dir.rmdir()
    return report


def describe(report: Report, wanted: int) -> str:
    lines = [
        f"Replays to recordings, {time.strftime('%Y-%m-%d %H:%M')} (replicar {REPLICAR_VERSION})",
        # The last two parts only: the full path holds the Windows user name.
        f"Looked in: {', '.join(str(Path(*p.parts[-2:])) for p in report.looked_in)}",
        f"Replays found: {report.found:,}; {wanted}v{wanted} recordings kept: {report.kept}; converted now: "
        f"{report.converted}"
        + ("; stopped at the conversion cap (five times the limit)" if report.stopped_early else ""),
        f"Team sizes seen: {', '.join(f'{k} {v}' for k, v in sorted(report.team_sizes.items())) or 'none'}",
        f"New ticks written: {report.ticks:,} ({report.ticks / 120 / 60:.1f} minutes of play), kickoff ticks "
        f"{report.kickoffs:,}, rows left out (a car missing or demolished) {report.rows_left_out:,}",
        f"Not converted: {len(report.failed)}",
    ]
    lines += [f"  {reason} (x{n})" for reason, n in Counter(report.failed).most_common()]
    return "\n".join(lines)


# The large replay mode: replays are made ready a batch at a time, several at once, as compact
# samples held in memory only; no recording is written.
LABEL_ENTRIES = 16  # a label keeps its largest shares, as nitro_core::demos::LABEL_ENTRIES


@dataclass
class Prepared:
    """One replay made ready for copying: its samples, compact (observations in half precision, the mask's
    bits packed, the label's largest shares), or why not."""

    index: int  # its place in the plan
    stem: str
    size: int | None = None  # players a side, once read
    problem: str | None = None
    kickoffs: int = 0
    count: int = 0
    skipped: int = 0
    outside_window: int = 0  # samples left out by a kickoff window (replay_mix.kickoff_only_seconds)
    obs: Any = None  # [count, obs_size] float16
    mask: Any = None  # [count, actions / 8] uint8, the mask's bits, lowest first
    label_index: Any = None  # [count, LABEL_ENTRIES] uint8, the largest share first
    label_share: Any = None  # [count, LABEL_ENTRIES] float16, each row summing to 1
    seconds: float = 0.0  # spent making it ready
    kept_now: bool | None = None  # converted now: kept in the store, or not for want of disk space

    @property
    def usable(self) -> bool:
        return self.problem is None and self.count > 0


@dataclass(frozen=True)
class Job:
    """What a worker needs to make one replay ready."""

    index: int
    replay: Path
    wanted: tuple[int, ...]  # the game sizes copied
    tool: Path
    meshes: Path
    work: Path  # replicar's file goes here and is removed
    kept: Path  # this replay converted before (kept recordings or the store), used when it exists
    kept_size: int  # the game size of `kept`
    settings_json: str
    softness: float
    lean: bool = False  # `tool` is nitro-replicar (pretrain.converter = "nitro")
    store: Path | None = None  # where a nitro-replicar conversion is kept for later copies


# What `_core.replay_samples` takes per car (the layout in nitro_core::replicar): replicar's columns and their
# widths, then its flags.
_CORE_CAR_FLOATS = (
    ("position", 3),
    ("rotation", 4),
    ("velocity", 3),
    ("angular_velocity", 3),
    ("flip_relative_torque", 3),
    ("flip_time", 1),
    ("air_time_since_jump", 1),
    ("jump_ticks", 1),
    ("boost", 1),
    *((f"controls_{name}", 1) for name in ("throttle", "steer", "pitch", "yaw", "roll")),
)
_CORE_CAR_FLAGS = (
    "is_on_ground",
    "has_jumped",
    "has_flipped",
    "has_double_jumped",
    "is_jumping",
    "is_flipping",
    "controls_jump",
    "controls_boost",
    "controls_handbrake",
)


def core_columns(
    a: dict[str, Any], players: list[dict[str, Any]]
) -> tuple[int, bytes, bytes, bytes, bytes, bytes]:
    """A replicar file's arrays as `_core.replay_samples` takes them: the cars (Blue first, as `recording`),
    then every row's tick, whether it is usable, the cars' values and flags and the ball's, as bytes."""
    import numpy as np

    order = [p["index"] for p in sorted(players, key=lambda p: (p["team"], p["index"]))]
    for name in [n for n, _ in _CORE_CAR_FLOATS] + list(_CORE_CAR_FLAGS):
        if f"car_{name}" not in a:
            raise ValueError(f"the converted file has no {name} column")
    rows = len(a["ball_position"])
    floats = np.empty((rows, len(order), sum(w for _, w in _CORE_CAR_FLOATS)), dtype="<f4")
    flags = np.empty((rows, len(order), len(_CORE_CAR_FLAGS)), dtype=np.uint8)
    for c, p in enumerate(order):
        at = 0
        for name, width in _CORE_CAR_FLOATS:
            floats[:, c, at : at + width] = a[f"car_{name}"][:, p].reshape(rows, width)
            at += width
        for k, name in enumerate(_CORE_CAR_FLAGS):
            flags[:, c, k] = a[f"car_{name}"][:, p] == 1
    ball = np.concatenate(
        [a[f"ball_{name}"] for name in ("position", "rotation", "velocity", "angular_velocity")],
        axis=1,
        dtype="<f4",
    )
    frames = np.asarray(a["sim_tick"] if "sim_tick" in a else a["frame"], dtype="<i8")
    usable = _usable_rows(a, order).astype(np.uint8)
    return len(order), frames.tobytes(), usable.tobytes(), floats.tobytes(), flags.tobytes(), ball.tobytes()


def compact_arrays(part: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """The core's compact samples (`_core.replay_samples`, or `_core.demonstrations` with compact=True) as
    NumPy: observations float16, the packed mask, label indices and shares."""
    import numpy as np

    from nitro import _core

    count = int(part["count"])
    obs = np.frombuffer(part["obs"], dtype="<f2").reshape(count, int(part["obs_size"]))
    mask = np.frombuffer(part["mask"], dtype=np.uint8).reshape(count, (_core.num_actions() + 7) // 8)
    index = np.frombuffer(part["label_index"], dtype=np.uint8).reshape(count, LABEL_ENTRIES)
    share = np.frombuffer(part["label_share"], dtype="<f2").reshape(count, LABEL_ENTRIES)
    return obs, mask, index, share


def _reason(err: BaseException, job: Job) -> str:
    """Why a replay failed, grouped by reason in the copy's details: without its name or folders."""
    text = str(err).replace(str(job.kept), "its kept recording")
    names = {str(job.replay), str(job.replay.parent)} | (
        {job.replay.stem} if len(job.replay.stem) >= 8 else set()
    )
    for spelling in sorted(names, key=len, reverse=True):  # the longest first: the path before its folder
        text = text.replace(spelling, "<replay>")
    return f"{type(err).__name__}: {text[:200]}"


def prepare(job: Job) -> Prepared:
    """One replay as compact samples: converted with replicar (or read from a recording kept before), as
    `convert_replays` does, then turned into samples in the core; failures become `problem`."""
    from nitro import _core

    began = time.perf_counter()
    out = Prepared(job.index, job.replay.stem)
    try:
        if job.kept.is_file() and job.kept.name.endswith(STORED):
            out.size = job.kept_size
            arrays, players = read_stored(job.kept)
            part = _core.replay_samples(
                *core_columns(arrays, players), "the replay", job.settings_json, job.softness
            )
            del arrays
            if int(part["kickoffs"]) == 0:
                out.problem = NO_KICKOFF
                return out
        elif job.kept.is_file():
            out.size = job.kept_size
            part = _core.demonstrations(str(job.kept), job.settings_json, job.softness, compact=True)
        else:
            converted = job.work / f"{job.replay.stem}{'.lean' if job.lean else '.parquet'}"
            problem = convert(job.tool, job.replay, converted, job.meshes)
            if problem:
                out.problem = problem
                for left in (converted, converted.with_name(converted.name + ".partial")):
                    left.unlink(missing_ok=True)  # replicar's cut-off file
                return out
            try:
                data = converted.read_bytes() if job.lean else b""
                arrays, players = lean_arrays(data) if job.lean else read_replicar(converted)
            except (OSError, ValueError, KeyError) as err:
                out.problem = f"unreadable converted file: {type(err).__name__} {str(err)[:150]}"
                return out
            finally:
                converted.unlink(missing_ok=True)
            out.size = team_size(players)
            if job.store is not None and data and has_kickoff(arrays, players):
                out.kept_now = keep(data, job.replay.stem, out.size, job.store)  # every size, for later bots
            del data
            if out.size not in job.wanted:
                return out
            columns = core_columns(arrays, players)
            del arrays
            part = _core.replay_samples(*columns, "the replay", job.settings_json, job.softness)
            out.kickoffs = int(part["kickoffs"])
            if out.kickoffs == 0:
                out.problem = NO_KICKOFF
                return out
        out.count, out.skipped = int(part["count"]), int(part["skipped"])
        out.outside_window = int(part.get("outside_window", 0))
        out.obs, out.mask, out.label_index, out.label_share = compact_arrays(part)
    except Exception as err:
        out.problem = _reason(err, job)
    except BaseException as err:  # a panic in the core (pyo3's PanicException is not an Exception)
        if isinstance(err, KeyboardInterrupt | SystemExit):
            raise
        out.problem = _reason(err, job)
    finally:
        out.seconds = time.perf_counter() - began
    return out


# Converting only: replays into the store with nothing learned, every thread at work and at a lower
# priority than everything else on the PC. `nitro convert-replays` and the dashboard's Resources page run it.
def converting_file() -> Path:
    """A convert-only pass's progress, for the dashboard."""
    return OUT / "store" / "converting.json"


@dataclass
class Converting:
    """A convert-only pass so far."""

    found: int = 0  # replays in the folders
    todo: int = 0  # not converted before
    done: int = 0  # converted now, kept or not
    kept: int = 0  # kept in the store
    by_size: Counter[str] = field(default_factory=Counter)
    failed: Counter[str] = field(default_factory=Counter)
    disk_full: bool = False
    began: float = field(default_factory=time.time)

    def view(self, state: str) -> dict[str, Any]:
        minutes = max(time.time() - self.began, 1.0) / 60
        rate = self.done / minutes
        return {
            "state": state,
            "found": self.found,
            "todo": self.todo,
            "done": self.done,
            "kept": self.kept,
            "by_size": dict(self.by_size),
            "failed": sum(self.failed.values()),
            "reasons": dict(self.failed.most_common(5)),
            "disk_full": self.disk_full,
            "per_minute": round(rate, 1),
            "seconds_left": round((self.todo - self.done) / rate * 60) if rate > 0 else None,
            "updated": time.time(),
        }


def converted_before(stem: str, sizes: Sizes, store: Path | None = None) -> bool:
    """In the store, kept as a recording, or known to have no kickoff."""
    recorded = any((OUT / f"{n}v{n}" / f"{stem}.rlpr.zst").is_file() for n in (1, 2, 3, 4))
    return recorded or sizes.known.get(stem) == NO_KICKOFF or find_stored(stem, store=store) is not None


def lower_priority() -> None:
    """This process, and the converters it starts, below everything else on the PC."""
    if sys.platform == "win32":
        import ctypes

        below_normal = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS; child processes inherit it
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)  # type: ignore[attr-defined, unused-ignore]
    else:
        os.nice(10)


def _convert_one(
    tool: Path, replay: Path, work: Path, meshes: Path, store: Path
) -> tuple[str, str | None, bool]:
    """Converts one replay into the store: (its game size or why not, the reason it failed, kept)."""
    converted = work / f"{replay.stem}.lean"
    problem = convert(tool, replay, converted, meshes)
    if problem:
        converted.unlink(missing_ok=True)
        converted.with_name(converted.name + ".partial").unlink(missing_ok=True)
        return "", problem, False
    try:
        data = converted.read_bytes()
        arrays, players = lean_arrays(data)
    except (OSError, ValueError, KeyError) as err:
        return "", f"unreadable converted file: {type(err).__name__} {str(err)[:150]}", False
    finally:
        converted.unlink(missing_ok=True)
    size = team_size(players)
    if not has_kickoff(arrays, players):
        return NO_KICKOFF, None, False
    return str(size), None, keep(data, replay.stem, size, store)


@contextlib.contextmanager
def _one_pass(lock: Path) -> Iterator[None]:
    """Holds `lock` for a convert-only pass: a second pass at the same time would convert the same replays
    again and clear the first one's files. The system lets go of it if the process dies."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as file:
        try:
            if sys.platform == "win32":
                import msvcrt

                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined, unused-ignore]
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined, unused-ignore]
        except OSError as err:
            raise ReplayError(
                "replays are already being converted into the store (another `nitro convert-replays` or the "
                "dashboard's Converted replays card); let that pass finish or stop it first"
            ) from err
        yield


def convert_to_store(
    sources: list[Path],
    meshes: Path,
    limit: int = 0,
    workers: int = 0,
    store: Path | None = None,
    say: Callable[[str], None] = print,
    live: Path | None = None,
) -> Converting:
    """Converts the replays in `sources` (newest first, at most `limit`, 0 = all) that are not in the store
    yet, `workers` at once (0 = every thread of this PC), keeping each in the store and noting its size.
    Refused while another pass converts into the same store."""
    if not (meshes / "soccar").is_dir():
        raise ReplayError(f"the arena meshes were not found in {meshes} (paths.collision_meshes)")
    store = store or store_folder()
    with _one_pass(store / "pass.lock"):
        return _convert_to_store(sources, meshes, limit, workers, store, say, live)


def _convert_to_store(
    sources: list[Path],
    meshes: Path,
    limit: int,
    workers: int,
    store: Path,
    say: Callable[[str], None],
    live: Path | None,
) -> Converting:
    import concurrent.futures

    live = live or converting_file()
    status = Converting()
    found = newest_in_turn(sources) if len(sources) > 1 else newest_replays(sources)
    status.found = len(found)
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = Sizes(OUT)
    todo = [p for p in found if not converted_before(p.stem, sizes, store)]
    before = len(found) - len(todo)
    todo = todo[:limit] if limit else todo
    status.todo = len(todo)
    workers = workers or os.cpu_count() or 2

    def report(state: str) -> None:
        live.parent.mkdir(parents=True, exist_ok=True)
        partial = live.with_name(live.name + ".partial")
        partial.write_text(json.dumps(status.view(state)), encoding="utf-8")
        partial.replace(live)

    say(
        f"{status.found:,} replays found, {before:,} converted before or not usable; converting "
        f"{status.todo:,}, {workers} at once."
    )
    report("converting")
    if not todo:
        report("done")
        return status
    tool = nitro_replicar_tool()
    # Its own folder: pretraining converts into OUT / "converted" at the same time.
    work = OUT / "converted" / "store-pass"
    work.mkdir(parents=True, exist_ok=True)
    for left in [*work.glob("*.partial"), *work.glob("*.lean")]:
        left.unlink(missing_ok=True)  # a stopped pass's files
    last = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="convert") as pool:
        jobs = {pool.submit(_convert_one, tool, replay, work, meshes, store): replay for replay in todo}
        for future in concurrent.futures.as_completed(jobs):
            kind, problem, kept = future.result()
            status.done += 1
            if problem:
                status.failed[re.sub(r"\d{3,}", "#", problem)[:120]] += 1
            elif kind:
                sizes.note(jobs[future].stem, kind)
                if kind != NO_KICKOFF:
                    status.by_size[f"{kind}v{kind}"] += 1
                    status.kept += int(kept)
                    if not kept and not status.disk_full:
                        status.disk_full = True
                        gigabytes = STORE_MIN_FREE // 2**30
                        say(f"The disk has under {gigabytes} GB free: converted replays are not kept.")
            if time.time() - last >= 30 or status.done == status.todo:
                last = time.time()
                view = status.view("converting")
                left = view["seconds_left"]
                say(
                    f"{status.done:,} of {status.todo:,} converted ({view['per_minute']} a minute"
                    + (f", about {round(left / 60)} min left" if left is not None else "")
                    + f"); kept {status.kept:,}: "
                    + ", ".join(f"{k} {v:,}" for k, v in sorted(status.by_size.items()))
                )
                report("converting")
    report("done")
    return status
