"""The dashboard opens fast with many bots: its first page and the Garage read each bot's metrics from the
file's two ends and a few small reads (a long file read whole took seconds a bot, 265 s for 155 bots), never
read a save with PyTorch, and show the same numbers as reading the whole file does."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nitro.dashboard import app, bots, garage
from nitro.dashboard.api import Api
from nitro.runs import RunFolder
from test_dashboard import ROOT, make_bot

HEADER = [
    "agent_steps",
    "iteration",
    "steps_per_second",
    "reward_per_step",
    *(f"paid_r{i}" for i in range(30)),
]


def rows(count: int, first: int = 1) -> str:
    """`count` metrics rows of HEADER's width, about 250 bytes each, the reward climbing."""
    cells = ",".join(f"{0.123456 * k:.6f}" for k in range(30))
    numbers = range(first, first + count)
    return "".join(f"{i * 1000},{i},{40000 + i % 7},{0.001 * i},{cells}\n" for i in numbers)


def old_newest(folder: RunFolder) -> dict[str, float]:
    """The newest row as the dashboard read it before: through the whole file (`bots.metrics`)."""
    m = bots.metrics(folder)
    if not m.last:
        return {}
    return {k: v for k, v in zip(m.header, map(bots.number, m.last), strict=True) if v is not None}


def old_reward_points(folder: RunFolder, points: int = garage.POINTS) -> list[float]:
    """The Garage's reward line as it was read before: rows picked evenly by number from the whole file."""
    m = bots.metrics(folder)
    if not m.count or "reward_per_step" not in m.columns:
        return []
    spread = min(points, m.count)
    picked = sorted({round(i * (m.count - 1) / max(1, spread - 1)) for i in range(spread)})
    found = [m.value("reward_per_step", row) for row in picked]
    return [value for value in found if value is not None]


class Reads:
    """Counts the bytes read from metrics files, and the files opened."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bytes = 0
        self.opened = 0
        opener = Path.open

        def counted(path: Path, *args: Any, **kwargs: Any) -> Any:
            handle = opener(path, *args, **kwargs)
            mode = args[0] if args else kwargs.get("mode", "r")
            if path.name != "metrics.csv" or mode != "rb":
                return handle
            self.opened += 1
            return _Counted(handle, self)

        monkeypatch.setattr(Path, "open", counted)


class _Counted(io.RawIOBase):
    def __init__(self, handle: Any, reads: Reads) -> None:
        self.handle, self.reads = handle, reads

    def read(self, size: int | None = -1) -> Any:
        data = self.handle.read(size)
        self.reads.bytes += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        return int(self.handle.seek(offset, whence))

    def close(self) -> None:
        self.handle.close()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "agent_steps,reward_per_step",  # no whole line yet
        "agent_steps,reward_per_step\n",  # the header only
        "agent_steps,reward_per_step\n100,0.5\n200,0.6\n300,0.",  # the newest line still being written
        "agent_steps,reward_per_step\r\n100,0.5\r\n200,0.6\r\n",  # Windows line ends
        "agent_steps,reward_per_step\n100,0.5\n200,0.6,9\n\n",  # newest lines of another width: passed over
        "agent_steps,reward_per_step\n100,0.5\n200,\n",  # an empty cell
        "agent_steps,reward_per_step\n100,0.5\n" + "1,2,3\n" * 20_000 + "300,0.7\n",  # a long run of others
        "agent_steps,reward_per_step\n" + "1,2,3\n" * 20_000,  # no row of the header's width at all
        ",".join(f"column_{i}" for i in range(5000))
        + "\n"
        + "1," * 4999
        + "1\n",  # a header wider than a read
    ],
)
def test_the_newest_row_from_the_files_ends_is_the_one_the_whole_file_gives(
    tmp_path: Path, text: str
) -> None:
    folder = make_bot(tmp_path / "runs", "edge")
    folder.metrics.write_bytes(text.encode("utf-8"))
    assert bots.newest(folder) == old_newest(folder)
    assert garage.reward_points(folder) == old_reward_points(folder)
    found = bots.ends(folder)
    assert bots.ends(folder) is found, "read again only when the file changed"


def test_a_long_runs_reward_line_is_spread_through_the_file_first_row_to_newest(tmp_path: Path) -> None:
    folder = make_bot(tmp_path / "runs", "long")
    count = 12_000
    folder.metrics.write_text(",".join(HEADER) + "\n" + rows(count) + "1,2\n", encoding="utf-8")
    assert folder.metrics.stat().st_size > 10 * garage.WHOLE_BYTES
    assert bots.newest(folder) == old_newest(folder)
    points, before = garage.reward_points(folder), old_reward_points(folder)
    assert len(points) == garage.POINTS == len(before)
    assert points[0] == before[0] == 0.001 and points[-1] == before[-1] == 0.001 * count, "first and newest"
    assert points == sorted(set(points)), "every point a different row, oldest first"
    # Rows of about one length: the rows picked by place in the file are near those picked by number (here
    # within 17 rows of 12,000), so the line looks the same.
    assert all(abs(a - b) <= 0.001 * count / 200 for a, b in zip(points, before, strict=True))
    with folder.metrics.open("a", encoding="utf-8") as out:
        out.write(rows(1, count + 1))
    assert garage.reward_points(folder)[-1] == 0.001 * (count + 1), "read again once the file grew"


def test_many_bots_open_without_reading_their_metrics_whole_or_loading_a_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    long_text = ",".join(HEADER) + "\n" + rows(12_000)  # 3 MB: what a long run's file reads like, smaller
    short_text = ",".join(HEADER) + "\n" + rows(30)
    made = []
    for i in range(24):
        folder = make_bot(runs, f"bot{i:02d}", saves=2 if i % 3 else 0)
        folder.metrics.write_text(long_text if i % 2 else short_text, encoding="utf-8")
        made.append(folder)
    make_bot(runs, "fresh")  # no metrics yet
    expected = {f.root.name: (old_newest(f), old_reward_points(f)) for f in made}
    bots._CACHE.clear()  # what the whole-file reads above kept

    def whole(folder: RunFolder) -> Any:
        raise AssertionError(f"the start-up read {folder.metrics} whole")

    monkeypatch.setattr(bots, "metrics", whole)
    reads = Reads(monkeypatch)
    api = Api(ROOT, runs)
    page = app.page(api)
    shown = api.garage()
    long_bytes = len(long_text.encode())
    line = len(rows(1, 12_000))
    per_bot = (
        2 * bots._CHUNK + garage.POINTS * 3 * line
    )  # the two ends, a few lines a point of the reward line
    assert reads.bytes <= 24 * per_bot + 12 * len(short_text.encode()), reads.bytes
    assert reads.bytes < long_bytes, "less than one long file in all"
    home = {b["name"]: b for b in api.home()["bots"]}
    assert len(home) == 25 and '"name": "bot00"' in page
    for name, (row, line) in expected.items():
        b = home[name]
        saves = 2 if int(name[3:]) % 3 else 0
        assert b["agent_steps"] == (int(row.get("agent_steps", 0)) or (saves * 1000))
        assert b["steps_per_second"] == row.get("steps_per_second")
        assert b["reward_per_step"] == row.get("reward_per_step")
        points = shown["bots"][name]["reward"]
        if int(name[3:]) % 2 == 0:
            assert points == line, "a short run's line is the one the whole file gives"
        else:
            assert len(points) == len(line) and points[0] == line[0] and points[-1] == line[-1]
    assert home["fresh"]["agent_steps"] == 0 and shown["bots"]["fresh"]["reward"] == []

    before = reads.bytes
    api.home()
    api.garage()
    assert reads.bytes == before, "a refresh with nothing new reads no metrics"
    with made[1].metrics.open("a", encoding="utf-8") as out:
        out.write(rows(1, 12_001))
    before = reads.bytes
    assert {b["name"]: b for b in api.home()["bots"]}["bot01"]["agent_steps"] == 12_001_000
    assert api.garage()["bots"]["bot01"]["reward"][-1] == 12.001
    assert reads.bytes - before <= per_bot, "a training bot's refresh reads a few small parts of its file"


def test_the_first_page_and_the_garage_never_load_pytorch(tmp_path: Path) -> None:
    from test_dashboard_bot import real_saves

    runs = tmp_path / "runs"
    folder = make_bot(runs, "saved")
    real_saves(folder, (1000,))
    folder.metrics.write_text(",".join(HEADER) + "\n" + rows(5), encoding="utf-8")
    code = (
        "import sys; from pathlib import Path; from nitro.dashboard import app; "
        "from nitro.dashboard.api import Api; "
        f"api = Api(Path({str(ROOT)!r}), Path({str(runs)!r})); page = app.page(api); g = api.garage(); "
        "b = api.home()['bots'][0]; print('torch' in sys.modules, b['agent_steps'], "
        "b['parameters']['total'] > 0, len(g['bots']['saved']['reward']))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "5000", "True", "5"], out.stdout + out.stderr
