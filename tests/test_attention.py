"""The attention encoder (`model.encoder = "attention"`): it reads a padded observation as
the own car and the other cars, so the order of the cars within a side and the contents of empty places
change nothing; it saves, copies and learns like the plain networks; what cannot take it yet refuses it."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from nitro.config.loader import ConfigError, parse_toml
from nitro.config.schema import ModelConfig, NitroConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic, EntityEncoder, build_networks, observation_layout
from nitro.ppo import masked_log_probs
from nitro.rollout import Rollout
from nitro.surgery import SurgeryError, widen_model

CPU = torch.device("cpu")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
SMALL = 'policy = [32]\nvalue = [32]\nencoder = "attention"\nattention_width = 32\nattention_heads = 4\n'


def settings(tmp_path: Path, env: str = "obs_team_size = 3\n", model: str = SMALL) -> NitroConfig:
    """With the native observation unless `env` names one: its layout is the one these tests spell out."""
    native = "" if "observation" in env else 'observation = "native"\n'
    return parse_toml(
        f"[env]\n{native}{env}[model]\n{model}", NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path
    )


def networks(tmp_path: Path, seed: int = 0) -> tuple[ActorCritic, tuple[int, int, int, int]]:
    config = settings(tmp_path)
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(seed)
    model = build_networks(config, own + (teammates + opponents) * block, 90)
    return model.eval(), layout


def observations(layout: tuple[int, int, int, int], rows: int, filled: tuple[int, int]) -> torch.Tensor:
    """Random observations with the first `filled` teammate and opponent places taken, the rest empty."""
    own, block, teammates, opponents = layout
    generator = torch.Generator().manual_seed(1)
    obs = torch.randn(rows, own + (teammates + opponents) * block, generator=generator)
    cars = obs[:, own:].view(rows, teammates + opponents, block)
    cars[..., -1] = 0.0
    for place in [*range(filled[0]), *(teammates + i for i in range(filled[1]))]:
        cars[:, place, -1] = 1.0
    empty = cars[..., -1] == 0
    cars[empty] = 0.0
    return obs


def answers(model: ActorCritic, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return model(obs)


def test_the_layout_and_the_networks_fit_the_padded_observation(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    assert layout == (83, 30, 2, 3)
    assert isinstance(model.policy[0], EntityEncoder) and isinstance(model.value[0], EntityEncoder)
    assert model.policy[0] is not model.value[0], "each network reads with its own encoder"
    logits, values = answers(model, observations(layout, 5, (2, 3)))
    assert logits.shape == (5, 90) and values.shape == (5,)
    assert answers(model, observations(layout, 5, (2, 3)).view(5, 1, -1))[0].shape == (5, 1, 90)


def test_the_order_of_the_cars_within_a_side_changes_nothing(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    own, block, teammates, opponents = layout
    obs = observations(layout, 6, (2, 3))
    shuffled = obs.clone()
    cars = shuffled[:, own:].view(6, teammates + opponents, block)
    cars[:] = cars[:, [1, 0, 4, 2, 3]]  # teammates swapped, opponents rotated
    for a, b in zip(answers(model, obs), answers(model, shuffled), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    crossed = obs.clone()
    crossed_cars = crossed[:, own:].view(6, teammates + opponents, block)
    crossed_cars[:] = crossed_cars[:, [2, 1, 0, 3, 4]]  # a teammate and an opponent trade places
    with torch.no_grad():
        read, read_crossed = model.policy[0](obs), model.policy[0](crossed)
    assert not torch.allclose(read, read_crossed, rtol=1e-3, atol=1e-4), "the side matters"


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
        assert all(t.isfinite().all() for t in clean), f"{filled}: no car to look at still works"
        for a, b in zip(clean, dirty, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_saves_and_copies_answer_exactly_the_same(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    obs = observations(layout, 8, (2, 3))
    other, _ = networks(tmp_path, seed=7)
    assert not torch.equal(answers(model, obs)[0], answers(other, obs)[0])
    other.load_state_dict(model.state_dict())
    copy = model.fresh_copy()
    copy.load_state_dict(model.state_dict())
    frozen = model.fresh_copy()
    frozen.load_policy_state_dict(model.policy_state_dict())  # the self-play pool's way
    for twin in (other, copy):
        for a, b in zip(answers(model, obs), answers(twin, obs), strict=True):
            assert torch.equal(a, b)
    assert torch.equal(frozen.logits(obs), model.logits(obs))


@pytest.mark.parametrize("lean", [False, True], ids=["", "lean"])
def test_it_learns_like_the_plain_networks_and_fresh_weights_reach_the_encoder(
    tmp_path: Path, lean: bool
) -> None:
    config = settings(tmp_path, model=SMALL + ("lean_attention = true\n" if lean else ""))
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, 90)
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
    encoder_before = [p.detach().clone() for p in model.policy[0].parameters()]
    learner = Learner(model, PPOConfig(epochs=2, minibatches=2), 0.9, CPU, fp16=False)
    stats = learner.learn(r)
    assert stats.params_finite and stats.losses_finite and stats.skipped_updates == 0
    encoder = list(model.policy[0].named_parameters())
    moved = {name: not torch.equal(a, b) for a, (name, b) in zip(encoder_before, encoder, strict=True)}
    # The key biases have no effect on any output (one bias shared by every key shifts all scores alike):
    # with lean_attention they get no gradient, without it a rounding-sized one too small to move them.
    keys = {name for name, p in encoder if name.endswith(".k.bias")}
    assert keys == {f"attention.{i}.k.bias" for i in range(2)}
    assert all(p.grad is None for name, p in encoder if name in keys) == lean
    assert all(m for name, m in moved.items() if name not in keys), "every other encoder parameter moved"
    pairs = model.fresh_parameters()
    assert len(pairs) == len(list(model.parameters())), "fresh weights for every parameter"


def test_the_settings_say_what_attention_needs(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must divide"):
        settings(tmp_path, model=SMALL.replace("attention_heads = 4", "attention_heads = 5"))
    # No room written: the run's own team size is the room (every car its own place), 1v1 included.
    for env, room in (("", 1), ("team_size = 2\n", 2), ("obs_team_size = 1\n", 1)):
        assert observation_layout(settings(tmp_path, env=env))[2:] == (room - 1, room), env


def test_attention_in_a_1v1_game_is_allowed_with_a_note(tmp_path: Path) -> None:
    assert any("1v1 bot" in note for note in settings(tmp_path).advice()), "team_size 1, room for 3 a side"
    assert settings(tmp_path, env="team_size = 2\nobs_team_size = 3\n").advice() == []
    assert NitroConfig().advice() == []


def test_a_saved_attention_network_rebuilds_whatever_the_rest_of_the_settings_say(tmp_path: Path) -> None:
    from nitro.config.state_setters import Replay

    config = settings(tmp_path)
    moved = Replay(files=(str(tmp_path / "moved" / "*.rlpr"),), bundled=False)  # recordings gone
    elsewhere = dataclasses.replace(config, state_setters=(moved,))
    assert observation_layout(elsewhere) == observation_layout(config) == (83, 30, 2, 3)


def test_attention_sizes_are_fixed_for_attention_runs_only(tmp_path: Path) -> None:
    from nitro.runs import RunFolder, fixed_settings
    from nitro.transfer import Source, problems

    def run(name: str, model: str) -> tuple[RunFolder, NitroConfig]:
        file = tmp_path / f"{name}.toml"
        file.write_text(
            f'[env]\nobservation = "native"\nobs_team_size = 3\n[model]\n{model}', encoding="utf-8"
        )
        return RunFolder.create(tmp_path / "runs", name, file), settings(tmp_path, model=model)

    plain = "policy = [32]\nvalue = [32]\n"
    folder, config = run("plain", plain + "attention_heads = 8\n")
    saved = {"settings": fixed_settings(settings(tmp_path, model=plain))}
    assert folder.load_settings(saved) == config, "a plain network ignores the attention sizes"
    folder, _ = run("attention", SMALL.replace("attention_heads = 4", "attention_heads = 8"))
    with pytest.raises(ConfigError, match=r"model\.attention_heads"):
        folder.load_settings({"settings": fixed_settings(settings(tmp_path))})
    source = Source({"settings": fixed_settings(settings(tmp_path))}, "teacher")
    other_heads = settings(tmp_path, model=SMALL.replace("attention_heads = 4", "attention_heads = 8"))
    assert problems(source, other_heads) == ["model.attention_heads is 4 there and 8 here"]
    assert problems(source, settings(tmp_path)) == []


def test_the_rlbot_bot_runs_an_attention_network_exactly_as_pytorch_does(tmp_path: Path) -> None:
    import json
    import struct

    from nitro import _core
    from nitro.export import write_policy

    config = settings(tmp_path)
    model, layout = networks(tmp_path)
    with torch.no_grad():  # LayerNorms off their identity start and the null car off zero, so the
        for p in model.parameters():  # order of every weight in the bot file is tested
            p.add_(torch.randn_like(p) * 0.3)
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, model.obs_size, 90, "attention-test", 0)
    bot = _core.Policy(path)
    assert bot.header()["format"] == 2
    for filled in ((2, 3), (0, 1), (1, 0), (0, 0)):
        obs = observations(layout, 6, filled)
        with torch.no_grad():
            expected = model.logits(obs)
        flat = bot.logits_batch(obs.flatten().tolist(), 6)
        torch.testing.assert_close(torch.tensor(flat).view(6, 90), expected, rtol=1e-5, atol=1e-5)
    plain = ActorCritic(model.obs_size, 90, ModelConfig(policy=(16,), value=(16,)))
    write_policy(path, plain, config, model.obs_size, 90, "plain", 0)
    raw = path.read_bytes()
    assert struct.unpack_from("<I", raw, 8)[0] == 1, "plain networks are format 1"
    assert json.loads(raw[16 : 16 + struct.unpack_from("<I", raw, 12)[0]])["format"] == 1


def test_what_cannot_take_an_attention_network_yet_refuses_it(tmp_path: Path) -> None:
    model, _ = networks(tmp_path)
    with pytest.raises(SurgeryError, match="attention"):
        widen_model(model, 2)
    with pytest.raises(ValueError, match="layout"):
        ActorCritic(233, 90, dataclasses.replace(ModelConfig(), encoder="attention"))


SKIP = SMALL + "attention_skip = true\n"


def test_the_skip_path_puts_the_own_part_beside_the_encoder_and_the_bot_runs_it(tmp_path: Path) -> None:
    from nitro import _core
    from nitro.export import write_policy

    config = settings(tmp_path, model=SKIP)
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(3)
    model = build_networks(config, own + (teammates + opponents) * block, 90).eval()
    encoder = model.policy[0]
    assert isinstance(encoder, EntityEncoder) and encoder.outputs == own + 32
    obs = observations(layout, 6, (1, 2))
    with torch.no_grad():
        out = encoder(obs)
        plain = EntityEncoder(layout, 32, 2, 4)
        plain.load_state_dict(encoder.state_dict())
        assert torch.equal(out[:, :own], obs[:, :own]), "the own part goes through as it is"
        assert torch.equal(out[:, own:], plain(obs)), "beside the same encoder's output"
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, model.obs_size, 90, "skip-test", 0)
    bot = _core.Policy(path)
    assert bot.header()["format"] == 3, "the format names the skip path"
    for filled in ((2, 3), (0, 0)):
        obs = observations(layout, 6, filled)
        with torch.no_grad():
            expected = model.logits(obs)
        flat = bot.logits_batch(obs.flatten().tolist(), 6)
        torch.testing.assert_close(torch.tensor(flat).view(6, 90), expected, rtol=1e-5, atol=1e-5)


def test_the_skip_path_is_fixed_for_a_run(tmp_path: Path) -> None:
    from nitro.runs import fixed_settings
    from nitro.transfer import Source, problems

    source = Source({"settings": fixed_settings(settings(tmp_path))}, "teacher")
    assert problems(source, settings(tmp_path, model=SKIP)) == [
        "model.attention_skip is false there and true here"
    ]
    plain = settings(tmp_path, model="policy = [32]\nvalue = [32]\nattention_skip = true\n")
    assert build_networks(plain, 233, 90).obs_size == 233, "a plain network ignores it"


def test_the_encoders_can_learn_at_their_own_rate_and_a_save_resumes_either_way(tmp_path: Path) -> None:
    model, layout = networks(tmp_path)
    ppo = PPOConfig(epochs=1, minibatches=1, policy_lr=1e-3, value_lr=2e-3)
    same = Learner(model, ppo, 0.9, CPU, fp16=False)
    assert [g["lr"] for g in same.optimizer.param_groups] == [1e-3, 2e-3], "one group per network"
    scaled = Learner(model, dataclasses.replace(ppo, encoder_lr_scale=3.0), 0.9, CPU, fp16=False)
    rates = [g["lr"] for g in scaled.optimizer.param_groups]
    assert rates == pytest.approx([3e-3, 1e-3, 6e-3, 2e-3]), "each encoder apart, at 3x its network's"
    encoder_params = len(list(model.policy[0].parameters()))
    assert len(scaled.optimizer.param_groups[0]["params"]) == encoder_params
    assert set(scaled.groups) == {"policy", "value"}, "clipping and the reported norms stay per network"

    steps, agents = 4, 4
    r = Rollout.allocate(steps, agents, model.obs_size, 90, CPU)
    r.obs.copy_(observations(layout, steps * agents, (1, 1)).view(steps, agents, -1))
    r.mask.fill_(True)
    r.rewards.copy_(torch.randn(steps, agents))
    scaled.learn(r)
    same.learn(r)
    for before, other in ((scaled, ppo), (same, dataclasses.replace(ppo, encoder_lr_scale=3.0))):
        state = before.state_dict()
        again = Learner(model, other, 0.9, CPU, fp16=False)
        again.load_state_dict(state)
        saved = list(state["optimizer"]["state"].values())
        loaded = [again.optimizer.state[p] for g in again.optimizer.param_groups for p in g["params"]]
        assert len(saved) == len(loaded) == len(list(model.parameters())), "every parameter's Adam state"
        for a, b in zip(saved, loaded, strict=True):
            assert torch.equal(a["exp_avg"], b["exp_avg"]) and torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])
        expected = [g["lr"] for g in Learner(model, other, 0.9, CPU, fp16=False).optimizer.param_groups]
        assert [g["lr"] for g in again.optimizer.param_groups] == expected, "the current settings' rates"


@NEEDS_GPU
@pytest.mark.parametrize("model_text", [SMALL, SKIP], ids=["encoder", "with-skip"])
def test_fp16_stays_close_to_fp32_and_the_inference_graph_captures_it(
    tmp_path: Path, model_text: str
) -> None:
    from nitro.envs import GroupTensors
    from nitro.inference import GroupInference

    config = settings(tmp_path, model=model_text)
    layout = observation_layout(config)
    own, block, teammates, opponents = layout
    torch.manual_seed(0)
    model = build_networks(config, own + (teammates + opponents) * block, 90).eval().cuda()
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


MESHES = Path(__file__).parents[1] / "collision_meshes"


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
@pytest.mark.parametrize("model_text", [SMALL, SMALL + "lean_attention = true\n"], ids=["", "lean"])
def test_a_real_run_with_attention_trains_and_its_save_rebuilds_the_same_networks(
    tmp_path: Path, model_text: str
) -> None:
    from nitro.export import model_from_state
    from nitro.health import learning_problems
    from nitro.training import build

    env = "team_size = 2\nobs_team_size = 2\nnum_arenas = 8\nthreads = 2\n"
    config = settings(tmp_path, env=env, model=model_text)
    config = dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES),
        ppo=dataclasses.replace(config.ppo, steps_per_iteration=640, minibatches=2),
    )
    setup = build(config)
    reports = [setup.trainer.iteration() for _ in range(2)]
    assert learning_problems([r.learn for r in reports]) == []
    state = {"learner": setup.trainer.learner.state_dict()}
    rebuilt, obs_size, _ = model_from_state(state, config)
    layout = observation_layout(config)
    obs = observations(layout, 4, (1, 2))
    assert obs.shape[1] == obs_size
    with torch.no_grad():
        expected = setup.model.cpu()(obs)
    for a, b in zip(expected, answers(rebuilt, obs), strict=True):
        assert torch.equal(a, b)
