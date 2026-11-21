"""Long-run housekeeping: the saves' disk cap (run.saves_disk_gb) and metrics.csv's size cap
(run.metrics_max_mb)."""

from __future__ import annotations

import csv
import dataclasses
import os
from pathlib import Path

import pytest

from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import bots
from nitro.metrics import MetricsLog
from nitro.runs import RunFolder

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))


def _save(
    folder: RunFolder, steps: int, size: int, keep: int = 100, milestone: int = 0, kept: bool = False
) -> None:
    folder.save_checkpoint({"agent_steps": steps}, steps, keep=keep, milestone_steps=milestone, kept=kept)
    (folder.checkpoints / f"{steps:012d}" / "pad.bin").write_bytes(b"\0" * size)


def _left(folder: RunFolder) -> list[int]:
    return [int(p.name) for p in folder.complete_checkpoints()]


def test_the_oldest_ordinary_saves_go_first_then_milestones_never_the_newest_or_kept(tmp_path: Path) -> None:
    folder = RunFolder(tmp_path / "bot")
    mb = 1_000_000
    _save(folder, 100, mb, kept=True)
    for steps in (200, 300, 1000, 1100, 1200, 2000, 2100):
        _save(folder, steps, mb)
    # milestones every 1000 steps: 100 (the first of 0-999), 1000, 2000
    deleted, over = folder.cap_saves(5.5 * mb / 1e9, milestone_steps=1000)
    assert _left(folder) == [100, 1000, 1200, 2000, 2100]
    assert len(deleted) == 3 and "the save at 200 steps" in deleted[0] and not over
    deleted, over = folder.cap_saves(3.5 * mb / 1e9, milestone_steps=1000)
    assert _left(folder) == [100, 2000, 2100]  # ordinary saves first, then the oldest milestone
    assert "milestone save at 1,000 steps" in deleted[-1]
    deleted, over = folder.cap_saves(0.5 * mb / 1e9, milestone_steps=1000)
    assert _left(folder) == [100, 2100]  # the kept one and the newest stay, however small the cap
    assert over


def test_no_cap_keeps_everything(tmp_path: Path) -> None:
    folder = RunFolder(tmp_path / "bot")
    for steps in (1, 2, 3):
        _save(folder, steps, 1000)
    assert folder.cap_saves(1.0, 0) == ([], False)
    assert _left(folder) == [1, 2, 3]


def _log(path: Path, rows: int) -> MetricsLog:
    log = MetricsLog(path, ["agent_steps", "iteration", "reward_per_step", "entropy"])
    for i in range(rows):
        log.write(
            {
                "agent_steps": 320 * (i + 1),
                "iteration": i + 1,
                "reward_per_step": i * 0.5,
                "entropy": 1 / (i + 1),
            }
        )
    return log


def _points(log: MetricsLog) -> dict[str, list[tuple[float, float]]]:
    return {column: sample.points for column, sample in log.samples.items()}


@pytest.mark.parametrize("rows", [600, 1025, 3000, 7777, 20_000])
def test_compaction_keeps_the_reports_points_and_rows_unchanged(tmp_path: Path, rows: int) -> None:
    path = tmp_path / "metrics.csv"
    log = _log(path, rows)
    before_points = _points(log)
    with path.open(encoding="utf-8") as file:
        before = list(csv.DictReader(file))
    size = path.stat().st_size
    assert not log.compact(size + 1)
    assert log.compact(size - 1)
    with path.open(encoding="utf-8") as file:
        after = list(csv.DictReader(file))
    assert after[0] == before[0] and after[-1] == before[-1]
    assert all(row in before for row in after)  # rows kept are unchanged
    assert len(after) == (rows + 1) // 2 + (rows % 2 == 0)
    assert path.stat().st_size < size * 0.55
    assert _points(log) == before_points  # the report draws exactly the same points
    log.write(
        {"agent_steps": 320 * (rows + 1), "iteration": rows + 1, "reward_per_step": 0.0, "entropy": 0.0}
    )
    log.close()
    with path.open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == len(after) + 1  # it goes on appending


@pytest.mark.parametrize("rows", [1200, 6000, 24_000])
def test_compaction_keeps_the_dashboards_chart_rows(tmp_path: Path, rows: int) -> None:
    folder = RunFolder(tmp_path / "bot")
    folder.root.mkdir()
    log = _log(folder.metrics, rows)
    with folder.metrics.open(encoding="utf-8") as file:
        every = [list(row.values()) for row in csv.DictReader(file)]
    before = bots.spread_rows(folder, 300)
    log.compact(1)
    log.close()
    after = bots.spread_rows(folder, 300)
    assert after[0] == before[0] and after[-1] == before[-1]
    # Every row the charts read after is a row of the file before, unchanged, and they are about as many.
    assert all(row in every for row in after)
    assert abs(len(after) - len(before)) <= len(before) // 10


SETTINGS = """
[run]
seed = 7
saves_disk_gb = 0.000001
metrics_max_mb = 0.001
[env]
num_arenas = 8
threads = 4
[model]
policy = [32, 32]
value = [32, 32]
[ppo]
steps_per_iteration = 320
minibatches = 2
"""


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_run_caps_its_saves_and_metrics_at_each_save(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nitro.training import TrainingRun

    file = tmp_path / "settings.toml"
    file.write_text(SETTINGS, encoding="utf-8")
    parsed = parse_toml(SETTINGS, NitroConfig, file=file, base_dir=tmp_path)
    config = dataclasses.replace(
        parsed,
        paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES),
        run=dataclasses.replace(parsed.run, device="cpu"),
    )
    run = TrainingRun(config, RunFolder.create(tmp_path, "bot", file))
    for _ in range(3):
        for _ in range(4):
            run.iteration()
        run.checkpoint()
    run.metrics.close()
    out = capsys.readouterr().out
    assert len(run.folder.complete_checkpoints()) == 1  # only the newest left under a tiny cap
    assert "Disk cap: deleted the milestone save at 1,280 steps" in out
    assert "metrics.csv was over 0.001 MB" in out
    with run.folder.metrics.open(encoding="utf-8") as metrics:
        steps = [int(float(row["agent_steps"])) for row in csv.DictReader(metrics)]
    assert steps[0] == 320 and steps[-1] == 3840 and len(steps) < 12


def test_a_compaction_that_cannot_swap_the_file_leaves_it_and_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows refuses to replace metrics.csv while another program holds it open (a spreadsheet): that
    must not stop training (with run.auto_restart, a crash at every save, started again for ever)."""
    from nitro import metrics

    path = tmp_path / "metrics.csv"
    log = _log(path, 600)
    before = path.read_bytes()

    def refused(partial: Path, target: Path) -> None:
        raise PermissionError(13, "The process cannot access the file", str(target))

    monkeypatch.setattr(metrics, "_swap", refused)
    assert not log.compact(1)
    assert path.read_bytes() == before and not (tmp_path / "metrics.csv.partial").exists()
    log.write({"agent_steps": 320 * 601, "iteration": 601, "reward_per_step": 0.0, "entropy": 0.0})
    log.close()
    with path.open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 601  # it goes on appending
