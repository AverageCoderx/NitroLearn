"""Recordings for the match viewer: how frames are packed into the page's data, kept in files and read back.

A recording is a dictionary (the page's data): `format` (the frame layout's version), `fps`, `cars`, the
frames as 16-bit numbers (each column multiplied by its scale, then stored column after column as the
difference from the frame before, compressed and written as base64), the boost pads' places, the bot's
figures at every decision and, with an attention network, what its attention places are. A frame holds the
ball's place, velocity, rotation and spin, the score and the pads' waits, then for each car its place,
velocity, rotation, spin, boost, flags, controls, wheel contact, body type and respawn wait. A recording in
another format (one a newer version wrote) is refused when it is read.
"""

from __future__ import annotations

import base64
import gzip
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# The version of the frame layout below: it changes whenever the layout does.
FORMAT = 2
PADS = 34
# Numbers per frame before the cars, and per car.
HEAD = 18 + PADS
CAR = 30
# What each column is multiplied by before it is rounded to 16 bits.
# Head: marker, ball place (to a quarter unit), velocity, forward, up, spin (rad/s), score, pad waits (s).
_HEAD_SCALES = [1] + [4] * 3 + [1] * 3 + [10_000] * 6 + [5_000] * 3 + [1] * 2 + [100] * PADS
# Car: team, place, velocity, forward, up, spin, boost, flags, controls, state bits, handbrake, body, respawn
# (a car's spin can pass 8 rad/s on one axis for a moment: its scale leaves room up to 10.9).
_CAR_SCALES = (
    [1] + [4] * 3 + [1] * 3 + [10_000] * 6 + [3_000] * 3 + [100, 1] + [10_000] * 8 + [1, 10_000, 1, 100]
)
SUFFIX = ".nlrec"


def width(cars: int) -> int:
    return HEAD + cars * CAR


def scales(cars: int) -> list[int]:
    """The scale of every column of a frame."""
    return _HEAD_SCALES + _CAR_SCALES * cars


def quantize(frames: np.ndarray, cars: int) -> np.ndarray:
    """The frames ([frames, width], real numbers) as the 16-bit numbers the viewer reads; a value beyond
    16 bits is held at the limit."""
    scaled = np.rint(frames.astype(np.float64) * np.asarray(scales(cars), dtype=np.float64))
    return np.clip(scaled, -32768, 32767).astype("<i2")


def pack(quantized: np.ndarray) -> str:
    """16-bit frames as text: column after column, each number the difference from the frame before (most
    columns barely change, so this compresses well), deflated, base64."""
    deltas = quantized.astype(np.int32)
    deltas[1:] -= quantized[:-1].astype(np.int32)
    by_column = np.ascontiguousarray(deltas.astype("<i2").T)  # the difference wraps within 16 bits
    return base64.b64encode(zlib.compress(by_column.tobytes(), 6)).decode("ascii")


def unpack(text: str, columns: int) -> np.ndarray:
    """`pack` undone: [frames, columns] 16-bit numbers."""
    by_column = np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype="<i2").reshape(columns, -1)
    return np.cumsum(by_column.T.astype(np.int32), axis=0).astype("<i2")


def raw(quantized: np.ndarray) -> str:
    """16-bit frames as plain base64, frame after frame: how live frames reach the page."""
    return base64.b64encode(np.ascontiguousarray(quantized).tobytes()).decode("ascii")


def data(
    frames: np.ndarray,
    *,
    fps: float,
    pads: list[list[float]],
    decisions: list[list[Any]],
    decisions_per_second: float,
    title: str = "",
    subtitle: str = "",
    attention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The page data of a recording of `frames` ([frames, width], real numbers)."""
    cars = (frames.shape[1] - HEAD) // CAR
    if frames.shape[1] != width(cars):
        raise ValueError(f"{frames.shape[1]} numbers a frame do not fit format {FORMAT}")
    return {
        "title": title,
        "subtitle": subtitle,
        "format": FORMAT,
        "fps": fps,
        "head": HEAD,
        "car": CAR,
        "cars": cars,
        "scales": scales(cars),
        "packing": "delta-deflate",
        "frames": pack(quantize(frames, cars)),
        "pads": [[round(v, 1) for v in pad] for pad in pads],
        "decisions": decisions,
        "decisionsPerSecond": decisions_per_second,
        "attention": attention,
    }


def save(file: Path, recording: dict[str, Any]) -> None:
    """Writes a recording as a file the viewer can open again (compressed JSON)."""
    text = json.dumps(recording, separators=(",", ":"), allow_nan=False)
    partial = file.with_name(file.name + ".part")
    with gzip.open(partial, "wt", encoding="utf-8") as out:
        out.write(text)
    partial.replace(file)


def load(file: Path) -> dict[str, Any]:
    """Reads a recording: a file written by `save`, or a viewer page (its recording is inside it). Raises
    ValueError, with a message for the user, for a file that holds none or one in a format not read here."""
    if file.suffix != SUFFIX:
        return from_page(file.read_text(encoding="utf-8"))
    with gzip.open(file, "rt", encoding="utf-8") as text:
        recording = json.load(text)
    if not isinstance(recording, dict) or "frames" not in recording:
        raise ValueError(f"{file} holds no recording")
    return _read_here(recording)


def from_page(text: str) -> dict[str, Any]:
    """The recording inside a viewer page; ValueError as `load`."""
    start = text.find('{"title":')
    if start < 0:
        raise ValueError("this page holds no recording")
    recording, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(recording, dict) or "frames" not in recording:
        raise ValueError("this page holds no recording")
    return _read_here(recording)


def _read_here(recording: dict[str, Any]) -> dict[str, Any]:
    """`recording`, refused unless its frames are in the layout this version reads."""
    made = recording.get("format")
    if made == FORMAT:
        return recording
    if isinstance(made, int) and made > FORMAT:
        raise ValueError(
            f"this recording was made by a newer NitroLearn (format {made}); update NitroLearn to watch it"
        )
    raise ValueError("this recording is in a format this NitroLearn does not read")
