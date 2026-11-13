"""`model.encoder = "attention_communication"`: the other cars, the ball and the own car talk
to each other before the own car looks at them. The order of the cars within a side and the contents of
empty places change nothing; saves, copies and learning work as for the other networks; the RLBot bot runs it
exactly as PyTorch does. The FP16 and inference-graph test runs on a machine with a graphics card, the real
run where the
collision meshes are."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from nitro.config.loader import ConfigError
from nitro.config.schema import PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic, TalkingEncoder, build_networks, entity_parts, observation_layout
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout
from nitro.surgery import SurgeryError, widen_model
from test_attention import CPU, MESHES, NEEDS_GPU, answers, observations, settings

TALK = (
    'policy = [32]\nvalue = [32]\nencoder = "attention_communication"\nattention_width = 32\n'
    "attention_heads = 4\n"
)


def networks(
    tmp_path: Path, model: str = TALK, seed: int = 0
) -> tuple[ActorCritic, tuple[int, int, int, int]]:
    config = settings(tmp_path, model=model)
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(seed)
    return build_networks(config, own + (teammates + opponents) * block, 90).eval(), layout


@pytest.mark.parametrize("observation", ["native", "nitro_v2"])
@pytest.mark.parametrize("room", [2, 3])
def test_the_ball_and_the_own_car_are_where_the_core_puts_them(observation: str, room: int) -> None:
    from nitro import _core

    _, own, block, _, _, _ = _core.observation_layout_of(observation, room, True, room)
    ball, car_at, car = entity_parts(observation)
    own_extras, other_extras = {"native": (9, 6), "nitro_v2": (22, 12)}[observation]
    assert own == car_at + car + own_extras and car_at == ball + 34 + 8
    assert block - 1 == car + other_extras, "the own car reads like the other cars' first numbers"
    with pytest.raises(ValueError, match="native or nitro_v2"):
        entity_parts("unknown")


def test_the_networks_fit_and_each_has_its_own_encoder(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    assert isinstance(model.policy[0], TalkingEncoder) and isinstance(model.value[0], TalkingEncoder)
    assert model.policy[0] is not model.value[0]
    logits, values = answers(model, observations(layout, 5, (2, 3)))
    assert logits.shape == (5, 90) and values.shape == (5,)
    assert answers(model, observations(layout, 5, (2, 3)).view(5, 1, -1))[0].shape == (5, 1, 90)


def test_the_order_of_the_cars_within_a_side_changes_nothing(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    own, block, teammates, opponents = layout
    obs = observations(layout, 6, (2, 3))
    shuffled = obs.clone()
    cars = shuffled[:, own:].view(6, teammates + opponents, block)
    cars[:] = cars[:, [1, 0, 4, 2, 3]]
    for a, b in zip(answers(model, obs), answers(model, shuffled), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    crossed = obs.clone()
    crossed[:, own:].view(6, teammates + opponents, block)[:] = crossed[:, own:].view(
        6, teammates + opponents, block
    )[:, [2, 1, 0, 3, 4]]
    with torch.no_grad():
        assert not torch.allclose(model.policy[0](obs), model.policy[0](crossed), rtol=1e-3, atol=1e-4)


def test_empty_places_are_ignored_whatever_they_hold_and_every_game_size_runs(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    own, block, teammates, opponents = layout
    for filled in ((0, 1), (1, 2), (2, 3), (0, 0)):
        obs = observations(layout, 4, filled)
        noisy = obs.clone()
        cars = noisy[:, own:].view(4, teammates + opponents, block)
        empty = cars[..., -1] == 0
        cars[empty] = torch.randn(int(empty.sum()), block) * 50
        cars[..., -1] = torch.where(empty, 0.0, cars[..., -1])
        clean, dirty = answers(model, obs), answers(model, noisy)
        assert all(t.isfinite().all() for t in clean), filled
        for a, b in zip(clean, dirty, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_the_ball_and_the_own_car_reach_the_answer(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    encoder = model.policy[0]
    assert isinstance(encoder, TalkingEncoder)
    ball, car_at, car = encoder.parts
    obs = observations(layout, 4, (1, 1))
    with torch.no_grad():
        before = encoder(obs)
        for part in (slice(0, ball), slice(car_at, car_at + car)):
            moved = obs.clone()
            moved[:, part] += 1.0
            assert not torch.allclose(encoder(moved), before), part


def test_who_listens_to_whom_is_kept_for_the_viewer(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    encoder = model.policy[0]
    assert isinstance(encoder, TalkingEncoder)
    _, _, teammates, opponents = layout
    tokens = 3 + teammates + opponents
    assert encoder.attention_labels() == ["nothing", "ball", "itself"] + ["teammate"] * 2 + ["opponent"] * 3
    encoder.keep_weights = True
    answers(model, observations(layout, 4, (1, 1)))
    looked, talked = encoder.attention_weights, encoder.talk_weights
    assert looked is not None and talked is not None
    assert looked.shape == (4, encoder.heads, tokens) and talked.shape == (4, encoder.heads, tokens, tokens)
    torch.testing.assert_close(looked.sum(-1), torch.ones(4, encoder.heads))
    torch.testing.assert_close(talked.sum(-1), torch.ones(4, encoder.heads, tokens))
    empty = [4, 6, 7]  # the second teammate place and the last two opponent places
    assert torch.all(looked[..., empty] == 0) and torch.all(talked[..., empty] == 0), "nobody listens to them"


def test_saves_and_copies_answer_exactly_the_same(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    obs = observations(layout, 8, (2, 3))
    other, _ = networks(tmp_path, seed=7)
    other.load_state_dict(model.state_dict())
    copy = model.fresh_copy()
    copy.load_state_dict(model.state_dict())
    frozen = model.fresh_copy()
    frozen.load_policy_state_dict(model.policy_state_dict())
    for twin in (other, copy):
        for a, b in zip(answers(model, obs), answers(twin, obs), strict=True):
            assert torch.equal(a, b)
    assert torch.equal(frozen.logits(obs), model.logits(obs))


@pytest.mark.parametrize("lean", [False, True], ids=["", "lean"])
def test_it_learns_and_every_encoder_parameter_moves(tmp_path: Path, lean: bool) -> None:
    model, layout = networks(tmp_path, TALK + ("lean_attention = true\n" if lean else ""))
    model.train()
    steps, agents = 6, 8
    r = Rollout.allocate(steps, agents, model.obs_size, 90, CPU)
    r.obs.copy_(observations(layout, steps * agents, (1, 2)).view(steps, agents, -1))
    r.mask.fill_(True)
    with torch.no_grad():
        logits, values = model(r.obs.view(steps * agents, -1))
    log_probs = masked_log_probs(logits, r.mask.view(steps * agents, -1))
    r.actions.copy_(torch.distributions.Categorical(logits=log_probs).sample().view(steps, agents))
    r.log_probs.copy_(log_probs.gather(1, r.actions.view(-1, 1)).view(steps, agents))
    r.values.copy_(values.view(steps, agents))
    r.rewards.copy_(torch.randn(steps, agents))
    before = [p.detach().clone() for p in model.policy[0].parameters()]
    ppo = PPOConfig(epochs=2, minibatches=2, encoder_lr_scale=2.0)
    learner = Learner(model, ppo, 0.9, CPU, fp16=False)
    assert len(learner.optimizer.param_groups) == 4, "the encoders' rate applies to this encoder too"
    stats = learner.learn(r)
    assert stats.params_finite and stats.losses_finite and stats.skipped_updates == 0
    encoder = list(model.policy[0].named_parameters())
    # The key biases have no effect on any output (one bias shared by every key shifts all scores alike):
    # with lean_attention the looks' ones get no gradient, the others a rounding-sized one that
    # cannot move them.
    keys = {name for name, p in encoder if name.endswith(".k.bias")}
    looks = {name for name in keys if name.startswith("listen.")}
    assert len(keys) == 4 and len(looks) == 2
    assert all((p.grad is None) == (lean and name in looks) for name, p in encoder if name in keys)
    assert all(
        not torch.equal(a, b) for a, (name, b) in zip(before, encoder, strict=True) if name not in keys
    )
    assert len(model.fresh_parameters()) == len(list(model.parameters()))


def test_the_rlbot_bot_runs_it_exactly_as_pytorch_does(tmp_path: Path) -> None:
    from nitro import _core
    from nitro.export import write_policy

    for model_text, skip in ((TALK, False), (TALK + "attention_skip = true\nattention_blocks = 1\n", True)):
        config = settings(tmp_path, model=model_text)
        model, layout = networks(tmp_path, model_text, seed=4)
        with torch.no_grad():  # LayerNorms off their identity start and the null token off zero, so the
            for p in model.parameters():  # order of every weight in the bot file is tested
                p.add_(torch.randn_like(p) * 0.3)
        path = tmp_path / "policy.nwt"
        write_policy(path, model, config, model.obs_size, 90, "talk-test", 0)
        bot = _core.Policy(path)
        assert bot.header()["format"] == 4, "the format names the encoder"
        for filled in ((2, 3), (0, 1), (1, 0), (0, 0)):
            obs = observations(layout, 6, filled)
            with torch.no_grad():
                expected = model.logits(obs)
            flat = bot.logits_batch(obs.flatten().tolist(), 6)
            torch.testing.assert_close(torch.tensor(flat).view(6, 90), expected, rtol=1e-5, atol=1e-5)
        assert model.policy[0].skip is skip


def test_the_settings_ask_what_attention_asks_and_it_is_fixed_for_a_run(tmp_path: Path) -> None:
    from nitro.runs import fixed_settings
    from nitro.transfer import Source, problems

    assert observation_layout(settings(tmp_path, env="", model=TALK)) == (83, 30, 0, 1), "its own room"
    with pytest.raises(ConfigError, match="must divide"):
        settings(tmp_path, model=TALK.replace("attention_heads = 4", "attention_heads = 5"))
    assert any("1v1 bot" in note for note in settings(tmp_path, model=TALK).advice())
    two = settings(tmp_path, env="team_size = 2\nobs_team_size = 3\n", model=TALK + "attention_fast = true\n")
    assert any("attention_fast does nothing" in note for note in two.advice())
    source = Source({"settings": fixed_settings(settings(tmp_path))}, "attention")
    assert problems(source, settings(tmp_path, model=TALK)) == [
        'model.encoder is "attention" there and "attention_communication" here'
    ], "an attention save is not a talking one: a different design under a different name"
    model, _ = networks(tmp_path)
    with pytest.raises(SurgeryError, match="attention"):
        widen_model(model, 2)


@NEEDS_GPU
def test_fp16_stays_close_to_fp32_and_the_inference_graph_captures_it(tmp_path: Path) -> None:
    from nitro.envs import GroupTensors
    from nitro.inference import GroupInference

    model, layout = networks(tmp_path)
    model = model.cuda()
    obs = observations(layout, 64, (2, 3)).cuda()
    full = answers(model, obs)
    with torch.autocast("cuda", dtype=torch.float16):
        half = answers(model, obs)
    for a, b in zip(full, half, strict=True):
        torch.testing.assert_close(a, b.float(), rtol=0.05, atol=0.05)
    tensors = GroupTensors.allocate(64, model.obs_size, 90, pin_memory=True)
    tensors.obs.copy_(obs.cpu())
    tensors.mask.fill_(True)
    engine = GroupInference(model, tensors, torch.device("cuda"), fp16=True, cuda_graph=True)
    engine.run()
    engine.wait()
    assert ((tensors.actions >= 0) & (tensors.actions < 90)).all()


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@pytest.mark.parametrize("observation", ["native", "nitro_v2"])
def test_a_real_2v2_run_trains_and_the_own_car_reads_as_its_teammate_sees_it(
    tmp_path: Path, observation: str
) -> None:
    from nitro.export import model_from_state
    from nitro.health import learning_problems
    from nitro.training import build

    env = f'team_size = 2\nobs_team_size = 2\nnum_arenas = 8\nthreads = 2\nobservation = "{observation}"\n'
    config = settings(tmp_path, env=env, model=TALK)
    config = dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES),
        ppo=dataclasses.replace(config.ppo, steps_per_iteration=640, minibatches=2),
    )
    setup = build(config)
    layout = observation_layout(config)
    own, block, _, _ = layout
    _, car_at, car = entity_parts(config.env.observation)
    obs = setup.trainer.envs.groups[0].obs
    owns = obs[:, car_at : car_at + car]
    for row in obs[:8]:
        for place in range(3):  # a teammate and two opponents, all present in a 2v2 game
            start = own + place * block
            assert row[start + block - 1] == 1.0
            seen = row[start : start + car]
            if place == 0:  # a teammate is seen in the team's own view: as it sees itself
                assert any(torch.equal(seen, mine) for mine in owns), "the own car reads as seen"
    reports = [setup.trainer.iteration() for _ in range(2)]
    assert learning_problems([r.learn for r in reports]) == []
    state = {"learner": setup.trainer.learner.state_dict()}
    rebuilt, _, _ = model_from_state(state, config)
    sample = observations(layout, 4, (1, 2))
    with torch.no_grad():
        expected = setup.model.cpu()(sample)
    for a, b in zip(expected, answers(rebuilt, sample), strict=True):
        assert torch.equal(a, b)
