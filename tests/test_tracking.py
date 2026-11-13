"""Weights & Biases tracking: off by default, what is sent and under which names, resuming
the same W&B run without sending replayed iterations twice, and failures that must not stop training."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from nitro import tracking
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig, WandbConfig

ON = WandbConfig(enabled=True, project="nitro-tests", tags=("1v1",))


class FakeRun:
    def __init__(self, calls: dict[str, Any], **kwargs: Any) -> None:
        self.calls, self.id, self.url, self.dir = calls, kwargs.get("id") or "abc123", "https://w/abc123", "."
        calls["init"].append(kwargs)

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.calls["define"].append((name, kwargs))

    def log(self, values: dict[str, Any]) -> None:
        if self.calls.get("fail"):
            raise ConnectionError("offline")
        self.calls["log"].append(values)

    def finish(self) -> None:
        self.calls["finished"] += 1


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("WANDB_API_KEY", "a-key")  # logged in
    calls: dict[str, Any] = {"init": [], "define": [], "log": [], "finished": 0}
    module = types.ModuleType("wandb")
    module.init = lambda **kwargs: FakeRun(calls, **kwargs)  # type: ignore[attr-defined]
    module.Settings = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    return calls


def row(steps: int, **values: Any) -> dict[str, Any]:
    return {"agent_steps": steps, "iteration": steps // 100, "entropy": 0.5, **values}


def test_tracking_is_off_by_default_and_then_does_nothing(tmp_path: Path) -> None:
    assert not NitroConfig().wandb.enabled
    tracker = tracking.start(WandbConfig(), tmp_path, {}, [])
    tracker.log(row(100))
    tracker.finish()
    assert not (tmp_path / tracking.STATE_FILE).exists()


def test_every_column_is_sent_against_agent_steps_in_sections(
    tmp_path: Path, fake_wandb: dict[str, Any]
) -> None:
    settings = {"env.team_size": 1, "ppo.policy_lr": 1.5e-4}
    (tmp_path / "my-run").mkdir()
    tracker = tracking.start(ON, tmp_path / "my-run", settings, ["kickoff", "random"])
    init = fake_wandb["init"][0]
    assert init["project"] == "nitro-tests" and init["name"] == "my-run" and init["tags"] == ["1v1"]
    assert init["config"] == settings and init["id"] is None and init["mode"] == "online"
    assert init["settings"] == {"console": "off", "save_code": False, "init_timeout": 45}
    assert ("*", {"step_metric": "agent_steps"}) in fake_wandb["define"]
    tracker.log(
        row(
            100,
            steps_per_second=40_000,
            reward_per_step=0.1,
            raw_touch_ball=0.02,
            weight_norm_policy=3.5,
            rating_mu=26.0,
            goal_share_kickoff=0.5,
            episodes_random="",
            episode_seconds=12.0,
            touches_per_minute=4.0,
        )
    )
    assert fake_wandb["log"] == [
        {
            "agent_steps": 100,
            "learning/iteration": 1.0,
            "learning/entropy": 0.5,
            "speed/steps_per_second": 40_000.0,
            "rewards/reward_per_step": 0.1,
            "rewards/raw_touch_ball": 0.02,
            "learning/weight_norm_policy": 3.5,
            "selfplay/rating_mu": 26.0,
            "start_states/goal_share_kickoff": 0.5,
            "learning/episode_seconds": 12.0,
            "play/touches_per_minute": 4.0,
        }
    ]
    tracker.finish()
    assert fake_wandb["finished"] == 1


def test_chart_names_worked_out_once_send_what_every_row_sent_before(
    tmp_path: Path, fake_wandb: dict[str, Any]
) -> None:
    """The tracker keeps each column's chart name; every row, one with a column new to it too, goes to
    W&B as when the name was worked out for every value. The wall clock's speed is charted with the speed."""
    labels = ["kickoff", "random"]
    tracker = tracking.start(ON, tmp_path, {}, labels)
    rows = [
        row(100, steps_per_second=40_000, wall_steps_per_second="", goal_share_kickoff=0.5, raw_goal=0.1),
        row(200, steps_per_second=41_000, wall_steps_per_second=39_000, goal_share_kickoff="", raw_goal=0.2),
        row(300, wall_steps_per_second=40_000, rating_mu=26.0, raw_goal=float("nan")),  # a new column
    ]
    for each in rows:
        tracker.log(each)
    sent = [
        {"agent_steps": each["agent_steps"]}
        | {
            tracking.chart_name(column, labels): float(value)
            for column, value in each.items()
            if column != "agent_steps" and value not in ("", None) and math.isfinite(float(value))
        }
        for each in rows
    ]
    assert fake_wandb["log"] == sent
    assert sent[1]["speed/wall_steps_per_second"] == 39_000.0 and "selfplay/rating_mu" in sent[2]
    for column in ("pool_goal_share_even", "pool_games_even", "start_edge"):
        assert tracking.chart_name(column, labels) == f"selfplay/{column}"


def test_a_resumed_run_continues_the_same_wandb_run_without_sending_replays(
    tmp_path: Path, fake_wandb: dict[str, Any]
) -> None:
    first = tracking.start(ON, tmp_path, {}, [])
    for steps in (100, 200, 300):
        first.log(row(steps))
    first.finish()  # killed after 300; the run resumes from its save at 100
    assert json.loads((tmp_path / tracking.STATE_FILE).read_text()) == {
        "id": "abc123",
        "sent_until": 300,
        "url": "https://w/abc123",  # the dashboard's link to the charts
    }
    resumed = tracking.start(ON, tmp_path, {}, [])
    assert fake_wandb["init"][1]["id"] == "abc123" and fake_wandb["init"][1]["resume"] == "allow"
    for steps in (200, 300, 400):
        resumed.log(row(steps))
    assert [values["agent_steps"] for values in fake_wandb["log"]] == [100, 200, 300, 400]


def test_a_failure_while_sending_warns_once_and_training_goes_on(
    tmp_path: Path, fake_wandb: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    tracker = tracking.start(ON, tmp_path, {}, [])
    fake_wandb["fail"] = True
    tracker.log(row(100))
    tracker.log(row(200))
    assert sum("W&B stopped working" in r.message for r in caplog.records) == 1
    fake_wandb["fail"] = False
    tracker.log(row(300))
    assert fake_wandb["log"] == []  # stays off for the rest of the session


def test_a_wandb_that_cannot_start_stops_before_training_with_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType("wandb")

    def refuse(**kwargs: Any) -> None:
        raise RuntimeError("api_key not configured")

    module.init = refuse  # type: ignore[attr-defined]
    module.Settings = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    module.teardown = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "a-key")
    with pytest.raises(tracking.TrackingError, match="wandb login"):
        tracking.start(ON, tmp_path, {}, [])
    monkeypatch.setitem(sys.modules, "wandb", None)  # not installed
    with pytest.raises(tracking.TrackingError, match="pip install wandb"):
        tracking.start(ON, tmp_path, {}, [])


def test_an_empty_project_is_refused_only_when_tracking_is_on() -> None:
    text = '[wandb]\nproject = ""\n'
    assert not parse_toml(text, NitroConfig, file=Path("n.toml"), base_dir=Path.cwd()).wandb.enabled
    with pytest.raises(ConfigError, match="needs a name"):
        parse_toml(text + "enabled = true\n", NitroConfig, file=Path("n.toml"), base_dir=Path.cwd())


def test_the_real_wandb_package_records_a_run_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    monkeypatch.setenv("WANDB_SILENT", "true")
    settings = WandbConfig(enabled=True, mode="offline")
    tracker = tracking.start(settings, tmp_path, {"env.team_size": 1}, [])
    for steps in (100, 200):
        tracker.log(row(steps, reward_per_step=0.1))
    tracker.finish()
    state = json.loads((tmp_path / tracking.STATE_FILE).read_text())
    assert state["sent_until"] == 200
    runs = list((tmp_path / "wandb").glob(f"offline-run-*-{state['id']}"))
    assert len(runs) == 1 and any(runs[0].glob("*.wandb"))


def test_a_progress_note_that_cannot_be_written_never_stops_training(
    tmp_path: Path,
    fake_wandb: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A virus scanner holding wandb_run.json on Windows must not end the run."""
    tracker = tracking.start(ON, tmp_path, {}, [])

    def locked(path: Path, state: dict[str, Any]) -> None:
        raise PermissionError("the file is in use")

    monkeypatch.setattr(tracking, "_write_state", locked)
    tracker.log(row(100))
    tracker.log(row(200))
    assert [values["agent_steps"] for values in fake_wandb["log"]] == [100, 200]
    assert sum("Could not note W&B's progress" in r.message for r in caplog.records) == 1
    (tmp_path / tracking.STATE_FILE).write_text("{not json", encoding="utf-8")
    tracking.start(ON, tmp_path, {}, [])  # a damaged note starts a new W&B run
    assert fake_wandb["init"][-1]["id"] is None


def test_online_wandb_that_cannot_start_goes_offline_and_training_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No internet never stops training; W&B keeps its files for `nitro wandb`. The failed
    online start holds the run's id until W&B is torn down, and without a login W&B is not even asked online
    (it would wait for an answer in the console)."""
    calls: dict[str, Any] = {"init": [], "define": [], "log": [], "finished": 0, "teardown": 0}

    def init(**kwargs: Any) -> FakeRun:
        if kwargs["mode"] == "online":
            raise ConnectionError("no internet")
        if calls["teardown"] == 0:
            raise RuntimeError("run ID abc123 is in use")
        return FakeRun(calls, **kwargs)

    module = types.ModuleType("wandb")
    module.init = init  # type: ignore[attr-defined]
    module.Settings = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    module.teardown = lambda: calls.__setitem__("teardown", calls["teardown"] + 1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "a-key")
    (tmp_path / "my-run").mkdir()
    tracker = tracking.start(ON, tmp_path / "my-run", {}, [])
    assert calls["teardown"] == 1 and [c["mode"] for c in calls["init"]] == ["offline"]
    tracker.log(row(100))
    assert calls["log"][0]["agent_steps"] == 100
    assert any("nitro wandb my-run" in r.message for r in caplog.records)

    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setattr(tracking, "logged_in", lambda: False)
    asked: list[str] = []
    module.init = lambda **kwargs: asked.append(kwargs["mode"]) or FakeRun(calls, **kwargs)  # type: ignore[attr-defined]
    tracking.start(ON, tmp_path / "my-run", {}, [])
    assert asked == ["offline"], "not logged in: straight to offline, no question in the console"


def test_a_login_is_found_where_wandb_login_keeps_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    monkeypatch.setattr(tracking.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("NETRC", str(tmp_path / "none"))
    assert not tracking.logged_in()
    login = "machine api.wandb.ai\n  login user\n  password a-key\n"
    (tmp_path / "_netrc").write_text(login, encoding="utf-8")
    assert tracking.logged_in(), "Windows: %USERPROFILE%\\_netrc"


def test_the_copy_is_charted_round_by_round_once_and_beside_the_training(
    tmp_path: Path, fake_wandb: dict[str, Any]
) -> None:
    rounds = [
        {"round": 0, "replays": 0, "replays_in_all": 7_500, "held_loss": 1.117, "held_agreement": 0.657},
        {"round": 1, "replays": 100, "replays_in_all": 7_600, "held_loss": 1.11, "train_loss": None},
    ]
    copy = tracking.start_copy(ON, tmp_path, {})
    copy.rounds(rounds)
    copy.rounds(rounds)  # a carry-on sends its backlog again: nothing twice
    copy.finish()
    assert ("copy/held_loss", {"step_metric": "copy/replays_in_all"}) in fake_wandb["define"]
    assert [v["copy/round"] for v in fake_wandb["log"]] == [0.0, 1.0]
    assert fake_wandb["log"][1] == {
        "copy/replays": 100.0, "copy/replays_in_all": 7_600.0, "copy/held_loss": 1.11, "copy/round": 1.0
    }  # fmt: skip
    training = tracking.start(ON, tmp_path, {}, [])
    training.log(row(100))
    state = json.loads((tmp_path / tracking.STATE_FILE).read_text())
    assert state == {"id": "abc123", "copy_rounds": 1, "sent_until": 100, "url": "https://w/abc123"}, (
        "both notes kept"
    )


def test_nitro_wandb_sends_what_is_missing_in_one_session_and_finds_offline_parts(
    tmp_path: Path, fake_wandb: dict[str, Any]
) -> None:
    passes = [{"epoch": 1, "train_loss": 2.0, "held_loss": 2.1, "held_agreement": 0.4}]
    rounds = [{"round": 1, "replays": 100, "replays_in_all": 100, "held_loss": 1.2}]
    rows = [row(100), row(200)]
    assert tracking.send_all(ON, tmp_path, {}, [], rounds, passes, rows) == ("online", 1, 1, 2)
    assert len(fake_wandb["init"]) == 1 and fake_wandb["finished"] == 1
    sent = [sorted(v)[0] for v in fake_wandb["log"]]
    assert sent == ["copy/held_loss", "clone/held_agreement", "agent_steps", "agent_steps"]
    defined = dict(fake_wandb["define"])
    assert "*" not in defined, "no pattern that could also catch the copy's charts"
    assert defined["copy/held_loss"] == {"step_metric": "copy/replays_in_all"}
    assert defined["learning/entropy"] == {"step_metric": "agent_steps"}
    assert tracking.send_all(ON, tmp_path, {}, [], rounds, passes, rows) == ("online", 0, 0, 0)
    assert len(fake_wandb["log"]) == 4, "nothing twice"
    for name, synced in (("offline-run-1-a", False), ("offline-run-2-a", True), ("run-3-a", False)):
        (tmp_path / "wandb" / name).mkdir(parents=True)
        if synced:
            (tmp_path / "wandb" / name / "run-a.wandb.synced").write_text("")
    assert [p.name for p in tracking.offline_runs(tmp_path)] == ["offline-run-1-a"]
