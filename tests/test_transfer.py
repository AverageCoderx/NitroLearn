"""Transfer learning: finding the run to start from, refusing networks that do not fit,
taking all of it or only the policy, and the value warm-up that holds the policy still."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest
import torch

from nitro.config import load_config
from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig, StartFromConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout
from nitro.runs import RunFolder, all_settings, fixed_settings
from nitro.transfer import Source, TransferError, apply, find_source, problems, rhythm_notes

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
OBS, ACTIONS, STEPS, AGENTS = 4, 3, 6, 4


def rollout_for(model: ActorCritic, steps: int = STEPS, agents: int = AGENTS) -> Rollout:
    generator = torch.Generator().manual_seed(0)
    obs, actions = model.obs_size, model.num_actions
    r = Rollout.allocate(steps, agents, obs, actions, CPU)
    r.obs.copy_(torch.randn(steps, agents, obs, generator=generator))
    r.mask.copy_(torch.rand(steps, agents, actions, generator=generator) > 0.3)
    r.mask[..., 2] = True
    with torch.no_grad():
        logits, values = model(r.obs)
    log_probs = masked_log_probs(logits, r.mask)
    r.actions.copy_(torch.distributions.Categorical(logits=log_probs).sample())
    r.log_probs.copy_(log_probs.gather(2, r.actions[..., None]).squeeze(-1))
    r.values.copy_(values)
    r.rewards.copy_(torch.randn(steps, agents, generator=generator))
    r.terminal[4, :] = True
    return r


SMALL = ModelConfig(policy=(16,), value=(16,))


def config(env: str = "", start: str = "", model: ModelConfig = SMALL) -> NitroConfig:
    text = f"[env]\n{env}\n[start_from]\n{start}\n"
    parsed = parse_toml(text, NitroConfig, file=Path("nitro.toml"), base_dir=Path.cwd())
    return dataclasses.replace(parsed, model=model)


def learner(settings: NitroConfig, seed: int, obs: int = OBS, actions: int = ACTIONS) -> Learner:
    torch.manual_seed(seed)
    model = ActorCritic(obs, actions, settings.model)
    return Learner(model, settings.ppo, gamma=0.9, device=CPU, fp16=False)


def saved_run(
    tmp_path: Path, name: str, settings: NitroConfig, steps: tuple[int, ...], actions: int = ACTIONS
) -> list[Learner]:
    file = tmp_path / f"{name}.toml"
    file.write_text("", encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", name, file)
    made = []
    for k, step in enumerate(steps):
        trained = learner(settings, seed=10 + k, actions=actions)
        trained.reward_scaler.moments.count.fill_(1.0)
        trained.reward_scaler.moments.m2.fill_((3.0 + k) ** 2)
        state = {"learner": trained.state_dict(), "agent_steps": step, "settings": fixed_settings(settings)}
        folder.save_checkpoint(state, step, keep=5, milestone_steps=0)
        made.append(trained)
    return made


def test_the_run_is_found_by_name_or_path_and_its_save_by_steps(tmp_path: Path) -> None:
    saved_run(tmp_path, "one", config(), (100, 200))
    runs = tmp_path / "runs"
    assert find_source(StartFromConfig(), runs) is None
    newest = find_source(StartFromConfig(run="one"), runs)
    assert newest is not None and newest.label == "one at 200 agent steps"
    older = find_source(StartFromConfig(run="one", at=100), runs)
    assert older is not None and older.state["agent_steps"] == 100
    direct = find_source(StartFromConfig(run=str(runs / "one" / "checkpoints" / f"{100:012d}")), runs)
    assert direct is not None and direct.state["agent_steps"] == 100
    with pytest.raises(TransferError, match="no save at 150 agent steps"):
        find_source(StartFromConfig(run="one", at=150), runs)
    with pytest.raises(TransferError, match="no run or save called 'two'"):
        find_source(StartFromConfig(run="two"), runs)


def source_of(settings: NitroConfig, seed: int = 1) -> tuple[Source, Learner]:
    trained = learner(settings, seed)
    trained.reward_scaler.moments.count.fill_(1.0)
    trained.reward_scaler.moments.m2.fill_(49.0)
    state = {"learner": trained.state_dict(), "agent_steps": 5, "settings": fixed_settings(settings)}
    return Source(state, "src at 5 agent steps"), trained


def test_only_networks_that_fit_and_mean_the_same_are_taken() -> None:
    source, _ = source_of(config())
    assert problems(source, config()) == []
    wider = config(model=ModelConfig(policy=(32,), value=(16,)))
    assert any("model.policy is [16] there and [32] here" in p for p in problems(source, wider))
    assert any("env.team_size" in p for p in problems(source, config("team_size = 2")))
    assert any("env.observation" in p for p in problems(source, config('observation = "native"')))
    padded, _ = source_of(config("obs_team_size = 3"))
    assert problems(padded, config("obs_team_size = 3\nteam_size = 3")) == [], "a 1v1 bot starts a 3v3 run"
    assert any(
        "env.obs_team_size is 3 there and 2 here" in p for p in problems(padded, config("obs_team_size = 2"))
    )
    no_popart = dataclasses.replace(config(), ppo=PPOConfig(popart=False))
    assert any("ppo.popart" in p for p in problems(source, no_popart))
    policy_only = dataclasses.replace(no_popart, start_from=StartFromConfig(run="src", parts="policy"))
    assert problems(source, policy_only) == []
    with pytest.raises(TransferError, match=r"cannot start this run: model\.policy"):
        apply(source, learner(wider, 2), wider)


def test_all_takes_both_networks_and_the_scaling_policy_only_the_policy() -> None:
    settings = config(start='run = "src"')
    source, trained = source_of(settings)
    fresh = learner(settings, 2)
    apply(source, fresh, settings)
    for key, value in trained.model.state_dict().items():
        assert torch.equal(fresh.model.state_dict()[key], value), key
    assert float(fresh.reward_scaler.moments.std) == 7.0

    policy_only = config(start='run = "src"\nparts = "policy"')
    fresh = learner(policy_only, 2)
    untouched = {k: v.clone() for k, v in fresh.model.state_dict().items() if k.startswith("value.")}
    apply(source, fresh, policy_only)
    state = fresh.model.state_dict()
    assert all(
        torch.equal(state[k], v) for k, v in trained.model.state_dict().items() if k.startswith("policy.")
    )
    assert all(torch.equal(state[k], v) for k, v in untouched.items()), "the value network starts afresh"
    assert float(fresh.reward_scaler.moments.std) != 7.0


def test_a_frozen_policy_learns_nothing_while_the_value_network_does() -> None:
    settings = dataclasses.replace(config(), ppo=PPOConfig(epochs=2, minibatches=2))
    frozen = learner(settings, 3)
    frozen.policy_frozen = True
    policy = [p.detach().clone() for p in frozen.model.policy.parameters()]
    value = [p.detach().clone() for p in frozen.model.value.parameters()]
    rate = frozen.learning_rates["policy"]
    stats = frozen.learn(rollout_for(frozen.model))
    assert all(torch.equal(a, b) for a, b in zip(policy, frozen.model.policy.parameters(), strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(value, frozen.model.value.parameters(), strict=True))
    assert stats.value_only_steps == stats.gradient_steps == 4 and stats.policy_change == 0
    assert 0 < stats.entropy <= 1 and stats.losses_finite
    assert frozen.learning_rates["policy"] == rate


def test_start_from_has_no_warm_up_or_pull_back() -> None:
    """[start_from] takes no value warm-up and no pull back toward the starting policy: the settings are
    unknown."""
    for line in (
        "value_warmup_steps = 1000",
        "keep_close = 0.5",
        "keep_close_half_life = 5",
        "keep_close_floor = 1e-9",
    ):
        with pytest.raises(ConfigError, match="is not a known setting"):
            config(start=line)


@pytest.mark.parametrize("shared", [(), (32,)], ids=["separate", "shared"])
def test_a_value_warm_up_moves_only_the_value_networks_own_layers(shared: tuple[int, ...]) -> None:
    """With the policy held, its layers and a shared trunk keep every weight; the value network's own
    layers learn."""
    settings = config(model=dataclasses.replace(SMALL, shared=shared))
    warming = learner(settings, 3)
    warming.policy_frozen = True
    before = {k: v.clone() for k, v in warming.model.state_dict().items()}
    for _ in range(2):
        warming.learn(rollout_for(warming.model))
    after = warming.model.state_dict()
    held = [k for k in before if k.startswith(("shared.", "policy."))]
    assert held and all(torch.equal(before[k], after[k]) for k in held)
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("value."))


def test_a_bot_started_from_another_takes_its_activation(tmp_path: Path) -> None:
    """A bot started from one with other activations (here ReLU, against the default leaky ReLU) keeps
    the networks' own."""
    from nitro.cli import _check_source

    relu = dataclasses.replace(SMALL, activation="relu")
    saved_run(tmp_path, "older", config(model=relu), (100,), actions=90)
    runs = tmp_path / "runs"
    assert _check_source(config(start='run = "older"'), runs) == "relu"
    assert _check_source(config(start='run = "older"', model=relu), runs) is None, (
        "the same: nothing to change"
    )


def test_train_from_a_source_that_cannot_start_the_run_makes_no_run_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`nitro train --from` checks the source before it makes the run's folder: a refused start leaves
    nothing behind, so the same name can be used again once the settings are fixed."""
    from nitro.cli import _check_source, main

    saved_run(tmp_path, "older", config(), (100,), actions=90)
    runs = tmp_path / "runs"
    with pytest.raises(TransferError, match="no run or save called 'missing'"):
        _check_source(config(start='run = "missing"'), runs)
    with pytest.raises(TransferError, match=r"env\.team_size is 1 there and 2 here"):
        _check_source(config("team_size = 2", start='run = "older"'), runs)
    saved_run(tmp_path, "other-actions", config(), (100,))
    with pytest.raises(TransferError, match="chooses among 3 actions"):
        _check_source(config(start='run = "other-actions"'), runs)

    settings = tmp_path / "2v2.toml"
    settings.write_text("[env]\nteam_size = 2\n", encoding="utf-8")
    for source in ("older", "missing"):
        argv = ["train", str(settings), "--from", source, "--runs", str(runs), "--name", "bad-from"]
        assert main(argv) == 1
        assert "Cannot start: start_from" in capsys.readouterr().err
        assert not (runs / "bad-from").exists(), f"--from {source}: no run folder is left behind"


def test_a_tick_skip_change_gives_lambdas_same_seconds_and_the_settings_counted_in_steps() -> None:
    """GAE lambda counts decisions while the discounts are in seconds. A bot at tick skip 8 (lambda
    0.95, half-life 15 s) starting a bot at tick skip 4: the window halves unless lambda is 0.9747."""
    settings = dataclasses.replace(config("tick_skip = 8"), ppo=PPOConfig(discount_half_life=15.0))
    older = Source({"all_settings": all_settings(settings)}, "older at 5 agent steps")
    here = dataclasses.replace(config("tick_skip = 4"), ppo=PPOConfig(discount_half_life=15.0))
    kept = copy.deepcopy(older.state)
    notes = rhythm_notes(older, here)
    assert older.state == kept, "only words: the save is left as it was"
    assert len(notes) == 3
    assert "1.26 s there (0.95), 0.65 s here (0.95); the same seconds is 0.9747" in notes[0]
    assert notes[1].endswith(
        "keep the number they had there, so each now covers 0.5 times the game time: "
        "ppo.steps_per_iteration 50,000."
    )
    assert "in seconds and carry over unchanged" in notes[2]
    off = ("goal critic", "snapshot", "flat_rating", "vdag", "entropy")
    assert not any(name in note for note in notes for name in off), "settings not in use are not named"


def test_settings_already_set_for_the_new_tick_skip_are_said_to_be() -> None:
    """A bot at tick skip 8 (snapshot_steps 20M, vdag_reinit_steps 0.5B) moved to 4 with lambda 0.975 and
    both doubled: those two keep their game time and are not doubled again."""
    gco = load_config(GCO_STARTER)
    eight = dataclasses.replace(
        gco,
        env=dataclasses.replace(gco.env, tick_skip=8),
        ppo=dataclasses.replace(gco.ppo, gae_lambda=0.95, vdag_reinit_steps=5e8),
        selfplay=dataclasses.replace(gco.selfplay, snapshot_steps=20_000_000),
    )
    source = Source({"all_settings": all_settings(eight)}, "source at 5 agent steps")
    v2 = dataclasses.replace(
        gco,
        env=dataclasses.replace(gco.env, tick_skip=4),
        ppo=dataclasses.replace(gco.ppo, gae_lambda=0.975, vdag_reinit_steps=1e9),
        selfplay=dataclasses.replace(gco.selfplay, snapshot_steps=40_000_000),
        start_from=StartFromConfig(run="source"),
    )
    notes = rhythm_notes(source, v2)
    assert len(notes) == 6
    assert "1.26 s there (0.95), 1.26 s here (0.975)" in notes[0]
    assert "goal critic" in notes[1] and "1.32 s there (0.95), 1.32 s here (0.975)" in notes[1]
    assert notes[2].endswith(
        "keep the number they had there, so each now covers 0.5 times the game time: "
        "ppo.steps_per_iteration 200,000, selfplay.flat_rating_steps 500,000,000."
    )
    assert notes[3].endswith(
        "were changed from there: selfplay.snapshot_steps 20,000,000 there, 40,000,000 here (the same game "
        "time), ppo.vdag_reinit_steps 500,000,000 there, 1,000,000,000 here (the same game time)."
    )
    assert notes[4].endswith("ppo.entropy_target 0.5 is per decision, at 2 times the decisions a second.")

    tripled = dataclasses.replace(v2, selfplay=dataclasses.replace(v2.selfplay, snapshot_steps=60_000_000))
    notes = rhythm_notes(source, tripled)
    assert "selfplay.snapshot_steps 20,000,000 there, 60,000,000 here (1.5 times its game time)" in notes[3]
    # What only this run has counts from its start: no number there to compare with.
    without = dataclasses.replace(v2, ppo=dataclasses.replace(v2.ppo, vdag_beta=0.0))
    assert not any("vdag" in note for note in rhythm_notes(source, without)), "not in use here"
    alone = Source({"all_settings": {**source.state["all_settings"], "ppo.vdag_beta": 0.0}}, "source")
    assert "from this run's start" in rhythm_notes(alone, v2)[4], "not in use there: nothing to compare"


def test_no_tick_skip_note_when_it_is_the_same_or_the_save_does_not_record_it() -> None:
    same = Source({"all_settings": {"env.tick_skip": 4, "ppo.gae_lambda": 0.9}}, "src at 5 agent steps")
    assert rhythm_notes(same, config()) == []
    older, _ = source_of(config())  # a save without all_settings
    assert rhythm_notes(older, config("tick_skip = 8")) == []


def test_a_copy_of_replays_gives_its_policy_only(tmp_path: Path) -> None:
    """A clone (nitro pretrain) or a [pretrain] run's first save has a value network that never trained:
    starting from it takes the policy only, whatever start_from.parts asks for; once trained, all of it."""
    from nitro.transfer import policy_only_note

    settings = tmp_path / "s.toml"
    settings.write_text("", encoding="utf-8")
    runs = tmp_path / "runs"
    cloned = RunFolder.create(runs, "cloned", settings)
    cloned.save_checkpoint({"agent_steps": 0, "pretrained": {"replays": 3}}, 0, keep=1, milestone_steps=0)
    copied = RunFolder.create(runs, "copied", settings)
    copied.save_checkpoint({"agent_steps": 0, "started_from_replays": True}, 0, keep=1, milestone_steps=0)
    for name in ("cloned", "copied"):
        for parts in ("all", "matching", "policy"):
            source = find_source(StartFromConfig(run=name, parts=parts), runs)
            assert source is not None and (source.parts or parts) == "policy", (name, parts)
    assert "only its policy is taken" in policy_only_note(source)
    copied.save_checkpoint({"agent_steps": 640, "started_from_replays": True}, 640, keep=2, milestone_steps=0)
    trained = find_source(StartFromConfig(run="copied", parts="all"), runs)
    assert trained is not None and trained.parts == "", "trained since: start_from.parts decides"
