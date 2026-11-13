"""The console report (`run.console = "report"`): what a block says for a training run of any
kind (from scratch, resumed, started from another run) and for a round of the replay copy, in plain ASCII."""

from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from nitro import console
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.runs import RunFolder
from nitro.training import TrainingRun

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

ROW: dict[str, Any] = {
    "agent_steps": 1_234_567_890,
    "iteration": 4_321,
    "reward_per_step": 0.0123,
    "raw_touch_ball": 0.004,
    "raw_goal": -0.00001,
    "episode_seconds": 14.2,
    "entropy": 1.234,
    "policy_change": 0.0021,
    "clip_fraction": 0.041,
    "policy_lr": 3e-4,
    "explained_variance": 0.81,
    "value_loss": 0.0123,
    "touches_per_minute": 3.2,
    "in_air_share": 0.12,
    "speed_kmh": 62.0,
    "boost_held": 34.0,
    "supersonic_share": 0.09,
    "demolished_share": 0.01,
    "aerial_touches_per_minute": 0.4,
    "flicks_per_minute": 0.0,
    "rating_mu": 25.3,
    "pool_goal_share": 0.48,
    "pool_size": 12,
    "grad_scale": 65536.0,
    "skipped_updates": 0,
}


def facts(**changes: Any) -> console.Facts:
    base = console.Facts(
        name="my-bot",
        running_seconds=7_380,
        steps_per_second=88_316,
        collect_seconds=0.42,
        learn_seconds=0.16,
        reward_labels=["touch_ball", "goal", "not_in_the_row"],
        started_from="copied at 0 agent steps",
        transfer=True,
        next_save="in 12.3M agent steps (every 200M)",
        warnings=["the value predictions stopped improving"],
    )
    return dataclasses.replace(base, **changes)


def test_a_block_says_everything_a_row_has_in_plain_ascii() -> None:
    lines = console.iteration_report(ROW, facts())
    text = "\n".join(lines)
    assert lines[0].startswith("---- my-bot | iteration 4,321 | 1.23B agent steps | 2 h 3 min this session ")
    for expected in (
        " Speed      88,316 steps/s | an iteration: playing 0.42 s, learning 0.16 s",
        " Reward     0.0123 a step | episodes 14.2 s on average",
        "touch_ball 0.0040 | goal 0.0000",  # no "-0.0000"; a reward without a column is left out
        " Policy     entropy 1.234 | change 0.0021 | clipped 4.1% | learning rate 3.0e-04",
        " Value      prediction 0.81 (1 = perfect) | loss 0.0123",
        " Play       3.2 touches a minute | 12% in the air | 62 km/h | boost 34 | 9% supersonic | 1% "
        "demolished",
        " Mechanics  aerial touches 0.40",
        " Self-play  rating 25.3 | 48% of the goals against past versions | 12 past versions",
        " Started    from copied at 0 agent steps (transfer learning)\n",
        " FP16       loss scale 65,536",
        " Next save  in 12.3M agent steps (every 200M)",
        " Warning    the value predictions stopped improving",
    ):
        assert expected in text, expected
    assert "flicks" not in text and "not_in_the_row" not in text, "nothing is said that did not happen"
    text.encode("ascii")  # a Windows console or a job's captured output never fails on a character


def test_a_row_with_little_in_it_still_gives_a_block_and_rows_average() -> None:
    lines = console.iteration_report(
        {"agent_steps": 5_000, "iteration": 2}, facts(started_from="", warnings=[])
    )
    assert lines[0].startswith("---- my-bot | iteration 2 | 5,000 agent steps")
    assert " Policy     entropy - | change - | clipped -" in lines
    assert not any(line.startswith((" Started", " Self-play", " FP16", " Warning")) for line in lines)
    rows = [
        {"agent_steps": 100, "iteration": 1, "entropy": 1.0},
        {"agent_steps": 200, "iteration": 2, "entropy": ""},
    ]
    rows.append({"agent_steps": 300, "iteration": 3, "entropy": 0.5})
    assert console.averaged(rows) == {"agent_steps": 300, "iteration": 3, "entropy": 0.75}


def test_a_round_of_the_replay_copy_gives_its_block() -> None:
    lines = console.round_report(
        console.Round(
            name="big-copy",
            number=30,
            used=3_000,
            limit=20_000,
            decisions=100_000_000,
            train_loss=1.11,
            held_loss=1.109,
            agreement=0.656,
            held_decisions=1_676_620,
            held_replays=50,
            rate=78.0,
            learned_seconds=41,
            left="about 3 h 39 min (done about 02:03)",
            used_before=2_517,
            before={"held_loss": 1.117, "held_agreement": 0.657},
        )
    )
    assert lines == [
        "---- big-copy | copying replays | round 30 of about 200 " + "-" * 44,
        " Replays    3,000 of 20,000 learned (100,000,000 decisions) | 2,517 left out as used before",
        " Loss       this round 1.110 | held back 1.109 (lower is closer) | before these replays 1.117",
        " Agreement  65.6% of the players' first choices, on 1,676,620 decisions from 50 held-back replays "
        "(before 65.7%)",
        " Speed      78.0 replays made ready a minute | round learned in 41 s",
        " Time left  about 3 h 39 min (done about 02:03)",
    ]


SETTINGS = """
[run]
seed = 7
console = "report"
[env]
num_arenas = 4
threads = 2
[terminals]
no_touch_seconds = 2
[model]
policy = [16]
value = [16]
[ppo]
steps_per_iteration = 160
minibatches = 1
[[state_setters]]
name = "kickoff"
"""


@pytest.fixture
def ticking(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clock that moves 20 s at every look, so every iteration prints."""
    now = [0.0]

    def clock() -> float:
        now[0] += 20.0
        return now[0]

    monkeypatch.setattr("nitro.training.time.monotonic", clock)


def settings_file(tmp_path: Path, text: str) -> tuple[NitroConfig, Path]:
    file = tmp_path / f"settings-{len(list(tmp_path.glob('settings-*')))}.toml"
    file.write_text(text, encoding="utf-8")
    config = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)
    paths = dataclasses.replace(config.paths, collision_meshes=MESHES)
    return dataclasses.replace(config, paths=paths, run=dataclasses.replace(config.run, device="cpu")), file


@NEEDS_MESHES
@pytest.mark.usefixtures("ticking")
def test_training_from_scratch_resumed_and_started_from_another_run_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, file = settings_file(tmp_path, SETTINGS)
    first = TrainingRun(config, RunFolder.create(tmp_path / "runs", "first", file))
    first.run(threading.Event(), until_steps=480)
    out = capsys.readouterr().out
    assert "---- first | iteration 1 |" in out and " Speed " in out and " Policy     entropy" in out
    assert " Next save  in about" in out and " Started" not in out
    folder = RunFolder(tmp_path / "runs" / "first")
    resumed = TrainingRun(config, folder, folder.load_latest())
    resumed.run(threading.Event(), until_steps=resumed.agent_steps + 320)
    assert "---- first | iteration " in capsys.readouterr().out, "a resumed run reports too"
    other, file = settings_file(tmp_path, SETTINGS + '[start_from]\nrun = "first"\n')
    started = TrainingRun(other, RunFolder.create(tmp_path / "runs", "second", file))
    started.run(threading.Event(), until_steps=320)
    out = capsys.readouterr().out
    assert "---- second | iteration 1 |" in out and " Started    from first at " in out
    assert "(transfer learning)" in out
    line, file = settings_file(tmp_path, SETTINGS.replace('console = "report"', 'console = "line"'))
    plain = TrainingRun(line, RunFolder.create(tmp_path / "runs", "plain", file))
    plain.run(threading.Event(), until_steps=320)
    out = capsys.readouterr().out
    assert " steps/s | reward per step " in out and "----" not in out, "the one line, as before"


def test_kickoffs_get_one_line_averaged_by_their_counts() -> None:
    rows = [
        {"agent_steps": 100, "iteration": 1, "kickoff_touch_s_diagonal": 2.0, "kickoff_count_diagonal": 3,
         "kickoff_touch_s_offcenter": "", "kickoff_count_offcenter": 0, "kickoff_touch_s_back": 3.0,
         "kickoff_count_back": 1},
        {"agent_steps": 200, "iteration": 2, "kickoff_touch_s_diagonal": 3.0, "kickoff_count_diagonal": 1,
         "kickoff_touch_s_offcenter": "", "kickoff_count_offcenter": 0, "kickoff_touch_s_back": "",
         "kickoff_count_back": 0},
    ]  # fmt: skip
    row = console.averaged(rows)
    assert row["kickoff_touch_s_diagonal"] == 2.25 and row["kickoff_count_diagonal"] == 2.0
    assert row["kickoff_touch_s_offcenter"] == "" and row["kickoff_touch_s_back"] == 3.0
    lines = [line for line in console.iteration_report(row, facts()) if line.startswith(" Kickoffs")]
    expected = " Kickoffs   first touch diagonal 2.25 s | back 3.00 s (kickoffs an iteration: 2.0 / 0.5)"
    assert lines == [expected]
    assert not any(
        line.startswith(" Kickoffs") for line in console.iteration_report({"iteration": 1}, facts())
    )
