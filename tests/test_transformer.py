"""`model.encoder = "transformer"`: the own car, the ball, every other car and optionally every boost pad as
tokens, standard pre-norm transformer blocks over them, read out from the own car's token or a summary
token. Masks, the order of the cars, saves and copies, learning, the settings and the RLBot bot (bot file
format 6) are checked here; FP16 and the inference graph need a graphics card, a real run the collision
meshes."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import NitroConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic, TransformerEncoder, build_networks, observation_layout
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout
from nitro.runs import fixed_settings
from nitro.surgery import SurgeryError, widen_model
from test_attention import CPU, MESHES, NEEDS_GPU, answers, observations

TRANSFORMER = (
    'policy = [32]\nvalue = [32]\nencoder = "transformer"\ntransformer_width = 32\ntransformer_heads = 4\n'
    "transformer_ff = 64\n"
)
WITH_PADS = TRANSFORMER + 'transformer_pads = true\ntransformer_readout = "summary"\n'
VARIANTS = {
    "own-car": TRANSFORMER,
    "pads-summary": WITH_PADS,
    "skip": TRANSFORMER + "attention_skip = true\n",
    "shared": TRANSFORMER + "shared = [32]\n",
    "lean": TRANSFORMER + "lean_attention = true\n",
    "lean-pads-summary": WITH_PADS + "lean_attention = true\n",
}


def settings(tmp_path: Path, model: str = TRANSFORMER, env: str = "team_size = 3\n") -> NitroConfig:
    """With the native observation unless `env` names one: its layout is the one these tests spell out."""
    native = "" if "observation" in env else 'observation = "native"\n'
    return parse_toml(
        f"[env]\n{native}{env}[model]\n{model}", NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path
    )


def networks(
    tmp_path: Path, model: str = TRANSFORMER, seed: int = 0, env: str = "team_size = 3\n"
) -> tuple[ActorCritic, tuple[int, int, int, int]]:
    config = settings(tmp_path, model, env)
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(seed)
    return build_networks(config, own + (teammates + opponents) * block, 90).eval(), layout


def moved(model: ActorCritic) -> ActorCritic:
    """Every weight off its start, so a swapped or misplaced one shows."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model


def first(model: ActorCritic) -> TransformerEncoder:
    encoder = model.shared[0] if len(model.shared) else model.policy[0]
    assert isinstance(encoder, TransformerEncoder)
    return encoder


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_networks_fit_and_every_part_of_the_observation_reaches_a_token(
    tmp_path: Path, variant: str
) -> None:
    model, layout = networks(tmp_path, VARIANTS[variant])
    assert layout == (83, 30, 2, 3), "room for the run's own 3 a side without env.obs_team_size"
    encoder = first(model)
    if variant != "shared":
        assert isinstance(model.value[0], TransformerEncoder) and model.value[0] is not encoder
    logits, values = answers(model, observations(layout, 5, (2, 3)))
    assert logits.shape == (5, 90) and values.shape == (5,)
    assert answers(model, observations(layout, 5, (2, 3)).view(5, 1, -1))[0].shape == (5, 1, 90)
    pads = encoder.pads
    rest = encoder.rest.tolist()
    expected = [i for i in range(9, 83) if not 51 <= i < 74 and not (pads and i < 43)]
    assert rest == expected, "the own part minus the ball, the own car and (as tokens) the pads"
    # Each part of the own part changes the answer: the ball, a pad, the previous action, the own car.
    moved(model)
    obs = observations(layout, 3, (1, 1))
    base = model.logits(obs)
    for index in (0, 20, 47, 60, 80):
        nudged = obs.clone()
        nudged[:, index] += 1.0
        assert not torch.allclose(model.logits(nudged), base, atol=1e-6), index


@pytest.mark.parametrize("variant", ["own-car", "pads-summary"])
def test_the_order_of_the_cars_within_a_side_and_empty_places_change_nothing(
    tmp_path: Path, variant: str
) -> None:
    model, layout = networks(tmp_path, VARIANTS[variant])
    moved(model)
    own, block, teammates, opponents = layout
    obs = observations(layout, 6, (2, 3))
    shuffled = obs.clone()
    cars = shuffled[:, own:].view(6, teammates + opponents, block)
    cars[:] = cars[:, [1, 0, 4, 2, 3]]
    for a, b in zip(answers(model, obs), answers(model, shuffled), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    crossed = obs.clone()
    crossed[:, own:].view(6, teammates + opponents, block)[:] = obs[:, own:].view(6, 5, block)[
        :, [2, 1, 0, 3, 4]
    ]
    assert not torch.allclose(model.logits(obs), model.logits(crossed), atol=1e-5), "the side matters"
    for filled in ((0, 1), (1, 2), (0, 0)):
        clean = observations(layout, 4, filled)
        noisy = clean.clone()
        places = noisy[:, own:].view(4, 5, block)
        empty = places[..., -1] == 0
        places[empty] = torch.randn(int(empty.sum()), block) * 50
        places[..., -1] = torch.where(empty, 0.0, places[..., -1])
        a, b = answers(model, clean), answers(model, noisy)
        assert all(t.isfinite().all() for t in a), filled
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", ["own-car", "pads-summary"])
def test_the_watch_path_by_hand_gives_the_same_answers_and_what_was_looked_at(
    tmp_path: Path, variant: str
) -> None:
    model, layout = networks(tmp_path, VARIANTS[variant])
    moved(model)
    encoder = first(model)
    obs = observations(layout, 5, (1, 2))
    fused = model.logits(obs)
    assert encoder.attention_weights is None, "kept only when asked"
    encoder.keep_weights = True
    by_hand = model.logits(obs)
    torch.testing.assert_close(by_hand, fused, rtol=1e-5, atol=1e-5)
    weights = encoder.attention_weights
    labels = encoder.attention_labels()
    assert weights is not None and weights.shape == (5, 4, len(labels))
    torch.testing.assert_close(weights.sum(-1), torch.ones(5, 4))
    summary = variant == "pads-summary"
    assert labels[: 2 + summary] == ["nothing", "itself", "ball"][1 - summary :]
    assert labels.count("teammate") == 2 and labels.count("opponent") == 3
    assert (labels[-1] == "pads") == summary, "the pads' shares added up into one place"
    second_teammate = labels.index("teammate") + 1
    empty = [second_teammate, len(labels) - 1 - summary]
    assert weights[..., empty].abs().max() == 0, "empty places get no attention"
    torch.use_deterministic_algorithms(True)
    try:
        encoder.keep_weights = False
        torch.testing.assert_close(model.logits(obs), fused, rtol=1e-5, atol=1e-5)
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_saves_and_copies_answer_exactly_the_same_and_learning_moves_every_weight(
    tmp_path: Path, variant: str
) -> None:
    model, layout = networks(tmp_path, VARIANTS[variant])
    obs = observations(layout, 8, (2, 3))
    other, _ = networks(tmp_path, VARIANTS[variant], seed=7)
    assert not torch.equal(answers(model, obs)[0], answers(other, obs)[0])
    other.load_state_dict(model.state_dict())
    copy = model.fresh_copy()
    copy.load_state_dict(model.state_dict())
    frozen = model.fresh_copy()
    frozen.load_policy_state_dict(model.policy_state_dict())
    for twin in (other, copy):
        for a, b in zip(answers(model, obs), answers(twin, obs), strict=True):
            assert torch.equal(a, b)
    assert torch.equal(frozen.logits(obs), model.logits(obs))

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
    encoder = first(model)
    before = [p.detach().clone() for p in encoder.parameters()]
    ppo = PPOConfig(epochs=2, minibatches=2, encoder_lr_scale=2.0)
    learner = Learner(model, ppo, 0.9, CPU, fp16=False)
    mine = [id(p) for p in encoder.parameters()]
    assert [scale for _, scale, params in learner.rate_groups if [id(p) for p in params] == mine] == [2.0]
    stats = learner.learn(r)
    assert stats.params_finite and stats.losses_finite and stats.skipped_updates == 0
    assert all(not torch.equal(a, b) for a, b in zip(before, encoder.parameters(), strict=True))
    assert len(model.fresh_parameters()) == len(list(model.parameters())), "fresh weights for every one"


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_rlbot_bot_runs_it_exactly_as_pytorch_does(tmp_path: Path, variant: str) -> None:
    from nitro import _core
    from nitro.export import write_policy

    config = settings(tmp_path, VARIANTS[variant])
    model, layout = networks(tmp_path, VARIANTS[variant])
    moved(model)
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, model.obs_size, 90, "transformer-test", 0)
    bot = _core.Policy(path)
    assert bot.header()["format"] == 6, "the format names the transformer"
    for filled in ((2, 3), (0, 1), (1, 0), (0, 0)):
        obs = observations(layout, 6, filled)
        with torch.no_grad():
            expected = model.logits(obs)
        flat = bot.logits_batch(obs.flatten().tolist(), 6)
        torch.testing.assert_close(torch.tensor(flat).view(6, 90), expected, rtol=1e-5, atol=1e-5)


def test_the_settings_check_the_sizes_and_fix_them_for_transformer_runs_only(tmp_path: Path) -> None:
    from nitro.runs import RunFolder
    from nitro.transfer import Source, problems

    with pytest.raises(ConfigError, match=r"must divide model\.transformer_width"):
        settings(tmp_path, TRANSFORMER.replace("transformer_heads = 4", "transformer_heads = 5"))
    attention_odd = TRANSFORMER + "attention_heads = 5\nattention_fast = true\n"
    config = settings(tmp_path, attention_odd)
    assert any("attention_fast does nothing with encoder = 'transformer'" in n for n in config.advice())

    def folder_with(name: str, model: str) -> RunFolder:
        file = tmp_path / f"{name}.toml"
        file.write_text(f"[env]\nteam_size = 3\n[model]\n{model}", encoding="utf-8")
        return RunFolder.create(tmp_path / "runs", name, file)

    folder = folder_with("t", TRANSFORMER)
    saved = {"settings": fixed_settings(folder.load_settings())}
    folder.settings.write_text(
        f"[env]\nteam_size = 3\n[model]\n{TRANSFORMER}attention_width = 64\n", encoding="utf-8"
    )
    assert folder.load_settings(saved).model.attention_width == 64, "the attention sizes stay free"
    folder.settings.write_text(f"[env]\nteam_size = 3\n[model]\n{WITH_PADS}", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"model\.transformer_pads"):
        folder.load_settings(saved)
    plain = "policy = [32]\nvalue = [32]\n"
    plain_folder = folder_with("p", plain)
    plain_saved = {"settings": fixed_settings(plain_folder.load_settings())}
    plain_folder.settings.write_text(
        f"[env]\nteam_size = 3\n[model]\n{plain}transformer_width = 64\n", encoding="utf-8"
    )
    assert plain_folder.load_settings(plain_saved).model.transformer_width == 64, "free for a plain bot"

    source = Source({"settings": fixed_settings(settings(tmp_path))}, "teacher")
    assert problems(source, settings(tmp_path, WITH_PADS)) == [
        "model.transformer_pads is false there and true here",
        'model.transformer_readout is "own_car" there and "summary" here',
    ]
    assert problems(source, settings(tmp_path)) == []
    attention = settings(tmp_path, 'policy = [32]\nvalue = [32]\nencoder = "attention"\n')
    assert problems(source, attention) == ['model.encoder is "transformer" there and "attention" here']
    model, _ = networks(tmp_path)
    with pytest.raises(SurgeryError, match="attention"):
        widen_model(model, 2)


@NEEDS_GPU
@pytest.mark.parametrize("variant", ["own-car", "pads-summary"])
def test_fp16_stays_close_to_fp32_and_the_inference_graph_captures_it(tmp_path: Path, variant: str) -> None:
    from nitro.envs import GroupTensors
    from nitro.inference import GroupInference

    model, layout = networks(tmp_path, VARIANTS[variant])
    model = model.cuda()
    obs = observations(layout, 64, (1, 2)).cuda()
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
@pytest.mark.parametrize("variant", ["own-car", "pads-summary"])
def test_a_real_run_trains_and_its_save_rebuilds_the_same_networks(tmp_path: Path, variant: str) -> None:
    from nitro.export import model_from_state
    from nitro.health import learning_problems
    from nitro.training import build

    config = settings(tmp_path, VARIANTS[variant], env="team_size = 2\nnum_arenas = 8\nthreads = 2\n")
    config = dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES),
        run=dataclasses.replace(config.run, device="cpu"),
        ppo=dataclasses.replace(config.ppo, steps_per_iteration=640, minibatches=2),
    )
    setup = build(config)
    reports = [setup.trainer.iteration() for _ in range(3)]
    assert learning_problems([r.learn for r in reports]) == []
    rebuilt, obs_size, _ = model_from_state({"learner": setup.trainer.learner.state_dict()}, config)
    layout = observation_layout(config)
    assert layout == (83, 30, 1, 2) and obs_size == 83 + 3 * 30
    obs = observations(layout, 4, (1, 2))
    with torch.no_grad():
        expected = setup.model.cpu()(obs)
    for a, b in zip(expected, answers(rebuilt, obs), strict=True):
        assert torch.equal(a, b)


def test_it_copies_recorded_games_and_starts_a_run_from_the_copy(tmp_path: Path) -> None:
    from nitro import pretrain
    from nitro.runs import RunFolder
    from nitro.transfer import apply, find_source
    from test_pretrain import recording

    text = f'[run]\ndevice = "cpu"\n[model]\n{TRANSFORMER}'
    file = tmp_path / "bc.toml"
    file.write_text(text, encoding="utf-8")
    config = parse_toml(text, NitroConfig, file=file, base_dir=tmp_path)  # 1v1, its own room
    for name in ("a", "b"):
        recording(tmp_path / f"{name}.rlpr")
    samples = pretrain.load(pretrain.recordings([tmp_path]), config)
    assert samples.obs.shape[1] == 110 + 39, "the own part and one opponent's place (nitro_v2)"
    model, history = pretrain.train(config, samples, samples, epochs=3, batch=64, lr=1e-3, device=CPU, seed=0)
    assert isinstance(model.policy[0], TransformerEncoder) and history[-1].train_loss < history[0].train_loss
    folder = RunFolder.create(tmp_path / "runs", "copied", file)
    pretrain.save(folder, model, config, {"samples": len(samples)})
    source = find_source(dataclasses.replace(config.start_from, run="copied"), tmp_path / "runs")
    assert source is not None
    fresh = Learner(build_networks(config, model.obs_size, 90), config.ppo, 0.99, CPU, fp16=False)
    apply(
        source,
        fresh,
        dataclasses.replace(config, start_from=dataclasses.replace(config.start_from, parts="policy")),
    )
    with torch.no_grad():
        assert torch.equal(fresh.model.logits(samples.obs[:16]), model.logits(samples.obs[:16]))
