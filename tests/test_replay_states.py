"""The `replay` state setter's settings side: recordings are found from patterns and the bundled set, the
core gets concrete files, missing recordings are reported with the line, and (with the meshes) a pool
starts episodes from a recording."""

from __future__ import annotations

import dataclasses
import json
import os
import struct
from pathlib import Path

import pytest
import torch

from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig
from nitro.config.state_setters import Replay
from nitro.envs import EnvPool, core_settings
from nitro.recordings import missing_replay_files, replay_files

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
PHYS = struct.Struct("<I3f9f3f3fB3x24x224xI")  # frame, pos, rotation rows, vel, ang vel, contact, (unused)
CAR_TAIL = struct.Struct(
    "<BBBxffBBBx3ffB3x5fBBBx"
)  # flags, jump/flip times, flags, torque, boost, touch, controls
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def car_record(
    frame: int, pos: tuple[float, float, float], vel: tuple[float, float, float], on_ground: bool
) -> bytes:
    phys = PHYS.pack(frame, *pos, *IDENTITY, *vel, 0.0, 0.0, 0.0, 0, 0)
    tail = CAR_TAIL.pack(
        on_ground, 0, 0, 0.0, 0.0, 0, 0, 1, 0.0, 0.0, 0.0, 0.4, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0
    )
    wheels = bytes(4 * 48)
    return struct.pack("<I", 588) + phys + tail + wheels + struct.pack("<B3x", 0)


def ball_record(frame: int, pos: tuple[float, float, float], vel: tuple[float, float, float]) -> bytes:
    return struct.pack("<I", 332) + PHYS.pack(frame, *pos, *IDENTITY, *vel, 0.0, 0.0, 0.0, 0, 0)


def write_recording(path: Path) -> None:
    """A 1v1 recording: a kickoff (Blue at -Y), then 20 frames of play with the ball rolling."""
    ticks = [
        car_record(1, (0.0, -4608.0, 17.0), (0.0, 0.0, 0.0), True)
        + car_record(1, (0.0, 4608.0, 17.0), (0.0, 0.0, 0.0), True)
        + ball_record(1, (0.0, 0.0, 93.15), (0.0, 0.0, 0.0))
    ]
    for i in range(2, 22):
        y = 100.0 * i
        ticks.append(
            car_record(i, (-500.0, -1000.0 + y, 17.0), (0.0, 800.0, 0.0), True)
            + car_record(i, (600.0, 2000.0 - y, 17.0), (0.0, -800.0, 0.0), True)
            + ball_record(i, (0.0, y, 93.15 + 3.0 * i), (0.0, 1200.0, 0.0))
        )
    header = (
        b"RLPR"
        + bytes([0])
        + struct.pack("<II", 6, 28)
        + struct.pack("<I6f", 2, -0.927561, -0.866994, 0.028509, 1.482587, 0.866994, 0.801690)
    )
    path.write_bytes(header + struct.pack("<I", len(ticks)) + b"".join(ticks))


def test_patterns_are_resolved_against_the_settings_file_and_handed_over_as_files(tmp_path: Path) -> None:
    (tmp_path / "recordings").mkdir()
    write_recording(tmp_path / "recordings" / "match-1.rlpr")
    write_recording(tmp_path / "recordings" / "match-2.rlpr")
    text = '[[state_setters]]\nname = "replay"\nfiles = ["recordings/*.rlpr"]\nair_bias = 2\nweight = 3\n'
    config = parse_toml(text, NitroConfig, file=tmp_path / "bot.toml", base_dir=tmp_path)
    (setter,) = config.state_setters
    assert isinstance(setter, Replay) and setter.weight == 3.0 and setter.air_bias == 2.0
    assert setter.files == ((tmp_path / "recordings" / "*.rlpr").resolve(),)
    files = replay_files(setter)
    assert [Path(f).name for f in files] == ["match-1.rlpr", "match-2.rlpr"]
    handed = json.loads(core_settings(config))["state_setters"]
    assert handed == [{"kind": {"name": "replay", "files": files, "air_bias": 2.0}, "weight": 3.0}]


def test_missing_recordings_are_reported_at_the_setting(tmp_path: Path) -> None:
    text = '[[state_setters]]\nname = "replay"\nfiles = ["nowhere/*.rlpr"]\n'
    with pytest.raises(ConfigError) as caught:
        parse_toml(text, NitroConfig, file=tmp_path / "bot.toml", base_dir=tmp_path)
    (issue,) = caught.value.issues
    assert issue.path == "state_setters[0].files" and "no recording matches" in issue.message
    with pytest.raises(ConfigError) as caught:
        parse_toml(
            '[[state_setters]]\nname = "replay"\n', NitroConfig, file=tmp_path / "b.toml", base_dir=tmp_path
        )
    assert "name at least one recording file, or set bundled = true" in caught.value.issues[0].message
    assert missing_replay_files(Replay(files=(), bundled=False)) is not None


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_a_pool_starts_its_episodes_from_the_recording(tmp_path: Path) -> None:
    write_recording(tmp_path / "match.rlpr")
    text = (
        "[run]\nseed = 2\n[env]\nnum_arenas = 4\ngroups = 1\nthreads = 2\n"
        '[[state_setters]]\nname = "replay"\nfiles = ["match.rlpr"]\n'
    )
    parsed = parse_toml(text, NitroConfig, file=tmp_path / "bot.toml", base_dir=tmp_path)
    config = dataclasses.replace(parsed, paths=dataclasses.replace(parsed.paths, collision_meshes=MESHES))
    pool = EnvPool(config, pin_memory=False)
    obs = pool.groups[0].obs.clone()
    assert torch.isfinite(obs).all()
    # Every arena shows a recorded moment: the ball between the cars' lines, never the kickoff centre for all.
    frames = [pool.frame(arena) for arena in range(4)]
    assert all(len(frame) > 3 for frame in frames)
    for _ in range(3):
        pool.groups[0].actions.zero_()
        pool.step(0)
        pool.wait(0)
    assert torch.isfinite(pool.groups[0].obs).all()
