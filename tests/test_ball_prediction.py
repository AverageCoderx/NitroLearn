"""Ball prediction in the observation, `env.ball_prediction`: its checks and refusals, the observation's size
and layout, the games, saves and the exported bot. The predictor itself is tested in the core
(crates/nitro_core/src/ball_prediction.rs, env.rs, obs.rs, demos.rs) and the bot's reading of RLBot's
prediction in crates/nitro_bot/tests/ball_prediction.rs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.cli import _observation_size
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig
from nitro.envs import EnvPool, core_settings, probe_config
from nitro.export import _readme, export_run, write_policy
from nitro.learner import Learner
from nitro.model import ActorCritic, build_networks, config_parts, entity_parts, observation_size
from nitro.runs import RunFolder, fixed_settings
from nitro.transfer import Source, problems

ROOT = Path(__file__).parents[1]
MESHES = ROOT / "collision_meshes"
FOUR = "ball_prediction = [0.5, 1, 2, 3]"


def parse(env: str, extra: str = "") -> NitroConfig:
    """Settings with `env` in [env]: the native observation at tick skip 8 unless it says otherwise."""
    for key, value in (("observation", '"native"'), ("tick_skip", "8")):
        if f"{key} =" not in env:
            env = f"{key} = {value}\n{env}"
    text = f'[paths]\ncollision_meshes = "{MESHES.as_posix()}"\n[env]\n{env}\n{extra}'
    return parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path.cwd())


def refusals(env: str, extra: str = "") -> dict[str, str]:
    with pytest.raises(ConfigError) as info:
        parse(env, extra)
    return {issue.path: issue.message for issue in info.value.issues}


def test_it_is_off_by_default_and_the_observation_is_unchanged() -> None:
    plain = parse("")
    assert plain.env.ball_prediction == ()
    assert "ball_prediction" not in json.loads(core_settings(plain))
    assert observation_size(plain) == 112
    on = parse(FOUR)
    assert on.env.ball_prediction == (0.5, 1.0, 2.0, 3.0)
    assert on.env.ball_prediction_steps() == (8, 15, 30, 45)
    assert json.loads(core_settings(on))["ball_prediction"] == [0.5, 1.0, 2.0, 3.0]
    # 12 numbers a time: position, velocity and both relative to the car.
    assert observation_size(on) == 112 + 48
    v2 = parse(f'observation = "nitro_v2"\n{FOUR}')
    assert observation_size(v2) == 148 + 48


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ("ball_prediction = [1, 0.5]", "must increase"),
        ("ball_prediction = [0.5, 0.52]", "rounds to [8, 8]"),
        ("ball_prediction = [0]", "above 0"),
        ("ball_prediction = [7]", "at most 6"),
        ("ball_prediction = [0.1, 0.2, 0.4, 0.8, 1, 2, 3, 4, 5]", "at most 8"),
        ("tick_skip = 7\nball_prediction = [5.99]", "beyond the 719 ticks"),
    ],
)
def test_times_it_cannot_work_with_are_refused(env: str, message: str) -> None:
    issues = refusals(env)
    found = " ".join(text for path, text in issues.items() if path.startswith("env.ball_prediction"))
    assert message in found, issues


def test_the_core_checks_the_times_too() -> None:
    settings = json.loads(core_settings(parse(FOUR)))
    settings["ball_prediction"] = [1.0, 0.5]  # refused before any meshes are loaded
    with pytest.raises(ValueError, match="must increase"):
        _core.EnvPool(json.dumps(settings))


def test_check_settings_shows_the_size_and_the_rounded_times() -> None:
    assert _observation_size(parse("")) == "Observation: 112 numbers."
    line = _observation_size(parse("ball_prediction = [0.5, 1.3]"))
    assert line == (
        "Observation: 136 numbers, 24 of them the ball prediction at 0.533, 1.33 s ahead (rounded to whole "
        "decisions)."
    )


def test_the_ball_token_of_an_attention_network_carries_the_prediction() -> None:
    assert entity_parts("nitro_v2") == (20, 62, 26)
    assert entity_parts("nitro_v2", 4) == (68, 110, 26)
    config = parse(
        f'team_size = 2\nobservation = "nitro_v2"\n{FOUR}',
        '[model]\nencoder = "attention_communication"\npolicy = [16]\nvalue = [16]',
    )
    assert config_parts(config) == (68, 110, 26)
    size = observation_size(config)
    model = build_networks(config, size, 90)
    assert model.logits(torch.zeros(3, size)).shape == (3, 90)


def test_games_with_ball_prediction_carry_it() -> None:
    if not (MESHES / "soccar").is_dir():
        pytest.skip("SKIPPED: no collision meshes")
    plain = probe_config(parse(""))
    on = probe_config(parse(FOUR))
    obs_a = EnvPool(plain, pin_memory=False).groups[0].obs
    obs_b = EnvPool(on, pin_memory=False).groups[0].obs
    assert (obs_a.shape[-1], obs_b.shape[-1]) == (112, 160)
    # The same games (the same seed): the ball, then its prediction, then the rest as before.
    assert torch.equal(obs_b[:, :9], obs_a[:, :9])
    assert torch.equal(obs_b[:, 9 + 48 :], obs_a[:, 9:])
    assert bool(obs_b[:, 9:57].abs().sum() > 0)


def test_saves_record_it_and_saves_from_before_read_as_off(tmp_path: Path) -> None:
    settings = tmp_path / "bot.toml"
    settings.write_text(f"[env]\n{FOUR}\n", encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "seer", settings)
    state = {"settings": fixed_settings(folder.load_settings())}
    assert state["settings"]["env.ball_prediction"] == (0.5, 1.0, 2.0, 3.0)
    folder.settings.write_text("[env]\nball_prediction = [0.5, 1]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"env\.ball_prediction: was \[0\.5, 1\.0, 2\.0, 3\.0\] when"):
        folder.load_settings(state)
    # A save from before the setting existed had no prediction.
    folder.settings.write_text("", encoding="utf-8")
    old = {"settings": fixed_settings(folder.load_settings())}
    del old["settings"]["env.ball_prediction"]
    assert folder.load_settings(old).env.ball_prediction == ()
    folder.settings.write_text(f"[env]\n{FOUR}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"env\.ball_prediction: was \[\] when"):
        folder.load_settings(old)


def test_a_bot_starts_from_another_only_with_the_same_prediction() -> None:
    plain, seer = parse(""), parse(FOUR)
    old = {"settings": fixed_settings(plain)}
    del old["settings"]["env.ball_prediction"]  # a save from before the setting existed
    assert problems(Source(old, "old"), plain) == []
    assert problems(Source(old, "old"), seer) == [
        "env.ball_prediction is [] there and [0.5, 1.0, 2.0, 3.0] here"
    ]
    assert problems(Source({"settings": fixed_settings(seer)}, "seer"), seer) == []


def test_the_exported_bot_carries_the_times_and_says_how_it_predicts(tmp_path: Path) -> None:
    config = dataclasses.replace(parse(FOUR), model=ModelConfig(policy=(8,), value=(8,)))
    path = tmp_path / "policy.nwt"
    write_policy(path, ActorCritic(160, 90, config.model), config, 160, 90, "run", 0)
    policy = _core.Policy(path)
    assert policy.header()["ball_prediction"] == [0.5, 1.0, 2.0, 3.0]
    assert policy.obs_size == 160
    plain = dataclasses.replace(parse(""), model=config.model)
    write_policy(path, ActorCritic(112, 90, plain.model), plain, 112, 90, "run", 0)
    assert _core.Policy(path).header()["ball_prediction"] == []
    assert b"ball_prediction" not in path.read_bytes(), "files without it stay as they were"
    assert "ball prediction" not in _readme("x", "run", 1, plain, True)

    settings = tmp_path / "bot.toml"
    settings.write_text(f'[run]\ndevice = "cpu"\n[model]\npolicy = [8]\nvalue = [8]\n[env]\n{FOUR}\n')
    folder = RunFolder.create(tmp_path / "runs", "seer", settings)
    loaded = folder.load_settings()
    learner = Learner(ActorCritic(160, 90, loaded.model), loaded.ppo, 0.99, torch.device("cpu"), fp16=False)
    state = {
        "learner": learner.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": 1_000,
        "iterations": 1,
        "settings": fixed_settings(loaded),
    }
    folder.save_checkpoint(state, 1_000, keep=5, milestone_steps=0)
    out = tmp_path / "bots" / "seer"
    export_run(folder, out, name="Seer", build=False)
    assert _core.Policy(out / "policy.nwt").header()["ball_prediction"] == [0.5, 1.0, 2.0, 3.0]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "0.5, 1, 2 and 3 s" in readme and "--meshes" in readme and "differ slightly" in readme


def test_copied_recordings_carry_the_prediction_of_each_decisions_ball(tmp_path: Path) -> None:
    """Replay copying and replays in training make their samples in the core (`demonstrations`), the
    prediction too."""
    if not (MESHES / "soccar").is_dir():
        pytest.skip("SKIPPED: no collision meshes")
    from nitro import pretrain
    from test_pretrain import recording

    recording(tmp_path / "a.rlpr")
    config = dataclasses.replace(parse(FOUR), run=dataclasses.replace(parse("").run, device="cpu"))
    samples = pretrain.load([tmp_path / "a.rlpr"], config)
    assert samples.obs.shape[1] == 160
    # The recording's ball rests at the centre spot: predicted to stay there, at every time.
    resting = torch.tensor([0.0, 0.0, 93.15 / 2044.0])
    for k in range(4):
        at = 9 + 12 * k
        assert torch.allclose(samples.obs[:, at : at + 3], resting.expand(len(samples), 3), atol=1e-3)
        assert torch.allclose(samples.obs[:, at + 3 : at + 6], torch.zeros(len(samples), 3), atol=1e-3)
    plain = pretrain.load([tmp_path / "a.rlpr"], parse(""))
    assert torch.equal(samples.obs[:, :9], plain.obs[:, :9])
    assert torch.equal(samples.obs[:, 57:], plain.obs[:, 9:])
    assert pretrain.previous_action(config) == slice(9 + 48 + 34, 9 + 48 + 34 + 8)


def test_the_settings_round_each_time_exactly_as_the_core() -> None:
    """2.1 s at tick skip 8 is 31 decisions in 32-bit floats (as the core computes), 32 in Python's 64-bit."""
    from nitro.config.schema import decisions_ahead

    assert decisions_ahead(2.1, 8) == 31 == _core.ball_prediction_decisions([2.1], 8)[0]
    checked = 0
    for tick_skip in range(1, 33):
        for hundredths in range(1, 600):
            time = hundredths / 100
            try:
                (core,) = _core.ball_prediction_decisions([time], tick_skip)
            except ValueError:
                continue  # beyond the 719 ticks: the settings check refuses it too
            assert decisions_ahead(time, tick_skip) == core, (time, tick_skip)
            checked += 1
    assert checked > 15_000


def test_the_last_time_stays_within_rlbots_719_ticks() -> None:
    """The last time stays within the 719 ticks the bot can read from RLBot's prediction even when it comes a
    tick late, so 6 s (720 ticks) is refused."""
    assert parse("ball_prediction = [5.9]").env.ball_prediction_steps() == (89,)
    for env in ("ball_prediction = [6.0]", "tick_skip = 6\nball_prediction = [6.0]"):
        issues = refusals(env)
        assert "beyond the 719 ticks" in issues["env.ball_prediction"], issues
    with pytest.raises(ValueError, match="beyond the 719 ticks"):
        _core.ball_prediction_decisions([6.0], 8)
