"""Practice starts picked from watched games (`nitro practice`): moments a bot should play again, kept in a
CSV in the run's folder and used as a `snapshots` start (`[[state_setters]] name = "snapshots"`) with a
weight of their own.

A moment is one car and the ball, with the car's jump and flip state, as Blue sees the field (an Orange car's
moment is turned around), in the 41 columns of `nitro.viewer.recording.MOMENT_COLUMNS`. It comes from a
recording made with `nitro watch --moments` (exactly as the game had it), or from a CSV the viewer saved
("Practice this moment", or the start editor). The run's settings get one `snapshots` block for the file;
the next time the run starts (resume), its games start from these moments as often as the weight says.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from nitro.runs import RunFolder
from nitro.viewer import recording as recordings

FILE = "practice_starts.csv"  # in the run's folder
COLUMNS = ["group", *recordings.MOMENT_COLUMNS]
DEFAULT_WEIGHT = 0.2
_BLOCK = re.compile(r"(?ms)^\[\[state_setters\]\]\s*\n(?:(?!^\[).)*")


class PracticeError(ValueError):
    """A moment cannot be added; the message says why in plain English."""


def number_text(value: float) -> str:
    """`value` as the shortest text that reads back as the same 32-bit number."""
    single = np.float32(value)
    return repr(float(single)) if single != 0 else "0"


def moment_line(group: str, values: list[float] | np.ndarray) -> list[str]:
    if len(values) != recordings.MOMENT:
        raise PracticeError(f"a moment has {recordings.MOMENT} numbers, not {len(values)}")
    if not all(math.isfinite(float(v)) for v in values):
        raise PracticeError("a moment holds a number that is not finite")
    return [group, *(number_text(float(v)) for v in values)]


def read_moments(text: str) -> list[tuple[str, list[float]]]:
    """The moments of a snapshot CSV (columns by name, the header first; a `group` column is optional)."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise PracticeError("the file is empty")
    header = [cell.strip() for cell in rows[0]]
    missing = [name for name in recordings.MOMENT_COLUMNS if name not in header]
    if missing:
        raise PracticeError(f"the file is not a moments file: no column {', '.join(missing[:3])}")
    where = [header.index(name) for name in recordings.MOMENT_COLUMNS]
    group = header.index("group") if "group" in header else None
    found = []
    for number, row in enumerate(rows[1:], start=2):
        try:
            values = [float(row[i]) for i in where]
        except (IndexError, ValueError):
            raise PracticeError(f"line {number} of the file is not a whole moment") from None
        label = row[group].strip() if group is not None and group < len(row) else "practice"
        moment_line(label, values)  # checks the numbers
        found.append((label or "practice", values))
    if not found:
        raise PracticeError("the file holds no moment")
    return found


def from_recording(recording: dict[str, Any], seconds: float, car: int) -> list[float]:
    """Car `car`'s moment at the decision nearest `seconds` into a recording made with `--moments`."""
    kept = recording.get("moments")
    if not kept:
        raise PracticeError(
            "this recording keeps no moments: record the game again with  nitro watch <bot> --moments"
        )
    cars = int(recording["cars"])
    if not 0 <= car < cars:
        raise PracticeError(f"the game has cars 0 to {cars - 1}, not {car}")
    values = recordings.unpack_floats(kept["values"]).reshape(-1, cars, recordings.MOMENT)
    frames = np.asarray([decision[0] for decision in recording["decisions"]], dtype=np.float64)
    if len(frames) != len(values):
        raise PracticeError("this recording's moments do not match its decisions")
    nearest = int(np.argmin(np.abs(frames / float(recording["fps"]) - seconds)))
    return [float(v) for v in values[nearest, car]]


def _settings_path(folder: RunFolder, file: Path) -> str:
    """How the run's settings name `file`: relative to the folder its settings came from (where relative
    paths in them point), or in full where no relative path reaches it (another drive)."""
    import json

    base = Path(json.loads(folder.info.read_text(encoding="utf-8"))["settings_dir"])
    try:
        return Path(os.path.relpath(file.resolve(), base.resolve())).as_posix()
    except ValueError:
        return file.resolve().as_posix()


def _with_block(text: str, path: str, weight: float | None) -> str:
    """The settings text with a `snapshots` block for `path`: added (weight `weight`, or the default) when
    there is none, its weight changed when `weight` is given."""
    quoted = re.escape(path)
    for block in _BLOCK.finditer(text):
        body = block.group(0)
        if re.search(r'(?m)^\s*name\s*=\s*"snapshots"', body) and re.search(
            rf'(?m)^\s*file\s*=\s*"{quoted}"', body
        ):
            if weight is None:
                return text
            if re.search(r"(?m)^\s*weight\s*=", body):
                changed = re.sub(r"(?m)^(\s*weight\s*=\s*)[^#\n]*", rf"\g<1>{weight:g}", body, count=1)
            else:
                changed = body.rstrip("\n") + f"\nweight = {weight:g}\n"
            return text[: block.start()] + changed + text[block.end() :]
    added = (
        "\n# Practice starts picked in the match viewer (nitro practice): games start from these moments\n"
        "# as often as the weight says, compared with the other starts' weights.\n"
        f'[[state_setters]]\nname = "snapshots"\nfile = "{path}"\n'
        f"weight = {DEFAULT_WEIGHT if weight is None else weight:g}\n"
    )
    return text.rstrip("\n") + "\n" + added


def add(folder: RunFolder, moments: list[tuple[str, list[float]]], weight: float | None = None) -> Path:
    """Adds `moments` to the run's practice starts and makes sure its settings use them; returns the CSV.
    Nothing changes when the settings would not load afterwards."""
    if weight is not None and not (math.isfinite(weight) and weight >= 0):
        raise PracticeError("the weight is a number of 0 or more")
    lines = [moment_line(group, values) for group, values in moments]
    file = folder.root / FILE
    settings = folder.settings.read_text(encoding="utf-8-sig")
    changed = _with_block(settings, _settings_path(folder, file), weight)
    existing = file.read_text(encoding="utf-8") if file.is_file() else ""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    if not existing:
        writer.writerow(COLUMNS)
    writer.writerows(lines)
    file.write_text(existing + out.getvalue(), encoding="utf-8")
    if changed != settings:
        folder.settings.write_text(changed, encoding="utf-8")
        try:
            folder.load_settings()
        except Exception:
            folder.settings.write_text(settings, encoding="utf-8")
            if existing:
                file.write_text(existing, encoding="utf-8")
            else:
                file.unlink()
            raise
    return file


def count(folder: RunFolder) -> int:
    file = folder.root / FILE
    return len(read_moments(file.read_text(encoding="utf-8"))) if file.is_file() else 0
