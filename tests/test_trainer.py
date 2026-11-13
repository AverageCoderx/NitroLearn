from __future__ import annotations

import gc
import math
import weakref
from typing import Any

import pytest
import torch

from nitro.cartpole import CartPoleEnvs
from nitro.config.schema import ModelConfig, PPOConfig
from nitro.envs import GroupTensors
from nitro.health import learning_problems
from nitro.inference import GroupInference
from nitro.learner import Learner, LearnStats
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.trainer import Trainer

CPU = torch.device("cpu")
SMALL = ModelConfig(policy=(16,), value=(16,))
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
# (device, fp16): the eager CPU path and both CUDA-graph paths.
PATHS = [
    pytest.param("cpu", False, id="cpu-fp32"),
    pytest.param("cuda", False, id="cuda-fp32", marks=NEEDS_GPU),
    pytest.param("cuda", True, id="cuda-fp16", marks=NEEDS_GPU),
]


class CountingEnvs:
    """Fake environments whose every output is a known function of the step number `s`.

    Agents observe [s, row]; action 2 is allowed only on even steps; the reward is s; steps with
    s % 5 == 4 end in a goal (terminal) and other steps with s % 3 == 2 are truncated, with final
    observation [1000 + s, row]. Episode ids and step numbers follow, and every action received is kept.
    """

    def __init__(self, sizes: tuple[int, ...], pin_memory: bool = False) -> None:
        self.groups = tuple(GroupTensors.allocate(n, 2, 3, pin_memory=pin_memory) for n in sizes)
        self.counter = [0] * len(sizes)
        self.received: list[list[torch.Tensor]] = [[] for _ in sizes]
        start = 0
        for tensors, n in zip(self.groups, sizes, strict=True):
            tensors.obs[:, 1] = torch.arange(start, start + n, dtype=torch.float32)
            self._observe(tensors, 0)
            start += n

    @staticmethod
    def _observe(tensors: GroupTensors, s: int) -> None:
        tensors.obs[:, 0] = float(s)
        tensors.mask.fill_(True)
        tensors.mask[:, 2] = s % 2 == 0

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def step(self, group: int) -> None:
        tensors = self.groups[group]
        s = self.counter[group]
        assert tensors.mask.gather(1, tensors.actions.long()[:, None]).all(), "a forbidden action was taken"
        self.received[group].append(tensors.actions.clone())
        terminal = s % 5 == 4
        truncated = s % 3 == 2 and not terminal
        tensors.reward.fill_(float(s))
        tensors.terminal.fill_(terminal)
        tensors.truncated.fill_(truncated)
        tensors.final_obs[:, 0] = 1000.0 + s
        tensors.final_obs[:, 1] = tensors.obs[:, 1]
        if terminal or truncated:
            tensors.episode.add_(1)
            tensors.step.zero_()
        else:
            tensors.step.add_(1)
        self.counter[group] = s + 1
        self._observe(tensors, s + 1)

    def wait(self, group: int) -> None:
        pass


def trainer_for(
    envs: CountingEnvs | CartPoleEnvs,
    steps: int,
    ppo: PPOConfig,
    device: torch.device = CPU,
    fp16: bool = False,
) -> Trainer:
    torch.manual_seed(0)
    num_actions = envs.groups[0].mask.shape[1]
    model = ActorCritic(envs.groups[0].obs.shape[1], num_actions, SMALL).to(device)
    learner = Learner(model, ppo, gamma=0.9, device=device, fp16=fp16)
    return Trainer(envs, model, learner, steps, device, fp16=fp16, cuda_graphs=device.type == "cuda")


@pytest.mark.parametrize(("device", "fp16"), PATHS)
def test_rollout_records_every_field_in_place(device: str, fp16: bool) -> None:
    dev = torch.device(device)
    envs = CountingEnvs((2, 3), pin_memory=dev.type == "cuda")
    trainer = trainer_for(envs, 5 * 7, PPOConfig(epochs=1, minibatches=1), dev, fp16)
    assert trainer.steps == 7
    times = trainer.collect()
    assert times.env_wait >= 0 and times.collect > 0
    r = trainer.rollout
    s = torch.arange(7, device=dev)[:, None].expand(7, 5)
    rows = torch.arange(5.0, device=dev).expand(7, 5)
    terminal = s % 5 == 4
    truncated = (s % 3 == 2) & ~terminal
    torch.testing.assert_close(r.obs, torch.stack([s.float(), rows], dim=-1))
    assert torch.equal(r.mask[:, :, 2], s % 2 == 0) and r.mask[:, :, :2].all()
    torch.testing.assert_close(r.rewards, s.float())
    assert torch.equal(r.terminal, terminal) and torch.equal(r.truncated, truncated)
    received = torch.cat([torch.stack(group) for group in envs.received], dim=1).to(dev)
    assert torch.equal(r.actions, received.long())
    ended = (terminal | truncated).long()
    episode = torch.cat([torch.zeros(1, 5, dtype=torch.long, device=dev), ended.cumsum(0)[:-1]])
    assert torch.equal(r.episode, episode)
    assert (r.episode_step[1:] == torch.where(ended[:-1].bool(), 0, r.episode_step[:-1] + 1)).all()

    model = trainer.model
    tolerance = {"atol": 2e-3, "rtol": 2e-3} if fp16 else {}
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
        logits, values = model(r.obs)
        final = torch.stack([1000.0 + s.float(), rows], dim=-1)
        final_values = model.normalised_value(final)
        last = torch.stack([torch.full((5,), 7.0, device=dev), rows[0]], dim=-1)
        bootstrap = model.normalised_value(last)
    log_probs = masked_log_probs(logits, r.mask).gather(2, r.actions[..., None]).squeeze(-1)
    torch.testing.assert_close(r.log_probs, log_probs, **tolerance)
    torch.testing.assert_close(r.values, values, **tolerance)
    expected = torch.where(truncated, final_values, torch.zeros_like(final_values))
    torch.testing.assert_close(r.truncated_values, expected, **tolerance)
    torch.testing.assert_close(r.bootstrap_values, bootstrap, **tolerance)


@pytest.mark.parametrize(("device", "fp16"), PATHS)
def test_next_iteration_continues_where_the_last_one_stopped(device: str, fp16: bool) -> None:
    dev = torch.device(device)
    envs = CountingEnvs((2,), pin_memory=dev.type == "cuda")
    trainer = trainer_for(envs, 2 * 4, PPOConfig(epochs=1, minibatches=1), dev, fp16)
    trainer.iteration()
    trainer.iteration()
    expected = torch.arange(4.0, 8.0, device=dev)[:, None].expand(4, 2)
    torch.testing.assert_close(trainer.rollout.obs[:, :, 0], expected)
    torch.testing.assert_close(trainer.rollout.rewards, expected)


@pytest.mark.parametrize(("device", "fp16"), PATHS)
def test_training_with_action_masks_applies_finite_updates(device: str, fp16: bool) -> None:
    # Regression test for NaN gradients from masked entropy: the masked CartPole forbids most actions.
    dev = torch.device(device)
    envs = CartPoleEnvs(num_envs=16, num_groups=2, seed=1, pin_memory=dev.type == "cuda", masked=True)
    trainer = trainer_for(envs, 512, PPOConfig(epochs=2, minibatches=4), dev, fp16)
    reports = [trainer.iteration() for _ in range(4)]
    assert learning_problems([r.learn for r in reports]) == []
    if not fp16:
        assert all(r.learn.skipped_updates == 0 for r in reports)
    assert all(0 < r.learn.entropy <= 1 for r in reports)


@pytest.mark.parametrize(("device", "fp16"), PATHS)
@pytest.mark.parametrize(("epochs", "minibatches"), [(1, 1), (2, 4)], ids=["one-update", "eight-updates"])
def test_learning_check_passes_a_learning_run_and_fails_a_frozen_one(
    device: str, fp16: bool, epochs: int, minibatches: int
) -> None:
    # The check once required PPO's KL statistic to be non-zero; that statistic is taken before each
    # learning step, so with one update per iteration (one epoch, one minibatch) a run that learned still
    # failed.
    dev = torch.device(device)

    def stats(lr: float) -> list[LearnStats]:
        envs = CartPoleEnvs(num_envs=16, num_groups=2, seed=1, pin_memory=dev.type == "cuda", masked=True)
        ppo = PPOConfig(epochs=epochs, minibatches=minibatches, policy_lr=lr, value_lr=lr)
        trainer = trainer_for(envs, 512, ppo, dev, fp16)
        return [trainer.iteration().learn for _ in range(3)]

    learning = stats(1e-3)
    assert learning_problems(learning) == []
    assert all(s.policy_change > 0 for s in learning)
    frozen = learning_problems(stats(0.0))
    assert "the weights did not change in 3 of 3 iterations" in frozen
    assert any(p.startswith("the policy did not change") and "3 of 3" in p for p in frozen)


class NanRewardEnvs(CountingEnvs):
    """CountingEnvs whose third step pays a NaN reward, so that iteration's gradients are not finite."""

    def step(self, group: int) -> None:
        super().step(group)
        if self.counter[group] == 3:
            self.groups[group].reward.fill_(math.nan)


def test_a_skipped_fp32_update_keeps_the_weights_and_a_loss_scale_of_exactly_1() -> None:
    envs = NanRewardEnvs((2,))
    torch.manual_seed(0)
    model = ActorCritic(2, 3, SMALL)
    learner = Learner(model, PPOConfig(epochs=1, minibatches=1), gamma=0.9, device=CPU, fp16=False)
    trainer = Trainer(envs, model, learner, 8, CPU, fp16=False, cuda_graphs=False, on_problem="warn")
    stats = trainer.iteration().learn
    assert stats.skipped_updates == 1 and stats.params_finite
    assert stats.grad_scale == 1.0, "FP32 never rescales, even after a skipped update"


def test_cartpole_iterations_produce_finite_learning_statistics() -> None:
    envs = CartPoleEnvs(num_envs=8, num_groups=2, seed=1, pin_memory=False)
    trainer = trainer_for(envs, steps=256, ppo=PPOConfig(epochs=2, minibatches=4))
    reports = [trainer.iteration() for _ in range(3)]
    stats = reports[-1].learn
    assert stats.gradient_steps == 8 and stats.skipped_updates == 0
    assert learning_problems([r.learn for r in reports]) == []
    assert reports[-1].agent_steps == 256 and reports[-1].steps_per_second > 0


def test_a_restart_lets_the_old_environments_and_engines_go_before_building_the_new_ones() -> None:
    envs = CountingEnvs((3, 5))
    model = ActorCritic(2, 3, SMALL)
    learner = Learner(model, PPOConfig(steps_per_iteration=32, minibatches=2), 0.99, CPU, False)
    trainer = Trainer(envs, model, learner, 32, CPU, fp16=False, cuda_graphs=False)
    trainer.iteration()
    gone = [weakref.ref(envs), weakref.ref(envs.groups[0]), weakref.ref(trainer.engines[0])]
    del envs

    def build() -> CountingEnvs:
        assert all(ref() is None for ref in gone), "the old set must be gone before the new one is built"
        return CountingEnvs((3, 5))

    gc.disable()  # plain reference counting must be enough; `release_memory` collects on top of it
    try:
        trainer.restart(build)
    finally:
        gc.enable()
    trainer.iteration()
    with pytest.raises(ValueError, match="same groups"):
        trainer.restart(lambda: CountingEnvs((4, 4)))


@NEEDS_GPU
def test_restarts_do_not_grow_the_cards_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each save once grew the card's memory: rebuilds captured new CUDA graphs while the old ones were
    alive, and each capture warmed up on a new stream while cuBLAS keeps a workspace per stream. Earlier
    tests in the process may already have used PyTorch's whole stream pool, which hides the second cause
    from the memory figures, so the streams made are counted too."""
    cuda = torch.device("cuda")
    envs = CartPoleEnvs(64, 2, seed=1, pin_memory=True)
    model = ActorCritic(4, 2, SMALL).to(cuda)
    learner = Learner(model, PPOConfig(steps_per_iteration=256, minibatches=2), 0.99, cuda, False)
    trainer = Trainer(envs, model, learner, 256, cuda, fp16=False, cuda_graphs=True)
    trainer.iteration()
    made = []
    real_stream = torch.cuda.Stream

    def counted(*args: Any, **kwargs: Any) -> torch.cuda.Stream:
        if "stream_id" not in kwargs:  # `torch.cuda.current_stream()` wraps an existing stream this way
            made.append(args)
        return real_stream(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Stream", counted)
    reserved, allocated = [], []
    for i in range(6):
        trainer.restart(lambda i=i: CartPoleEnvs(64, 2, seed=2 + i, pin_memory=True))
        trainer.iteration()
        reserved.append(torch.cuda.memory_reserved())
        allocated.append(torch.cuda.memory_allocated())
    assert made == [], "a restart must not make new CUDA streams"
    assert max(reserved[2:]) <= reserved[1], f"reserved memory kept growing over restarts: {reserved}"
    assert max(allocated[2:]) <= allocated[1], f"allocated memory kept growing over restarts: {allocated}"


def test_too_many_minibatches_is_refused() -> None:
    envs = CartPoleEnvs(num_envs=2, num_groups=1, seed=0, pin_memory=False)
    with pytest.raises(ValueError, match="minibatches"):
        trainer_for(envs, steps=2, ppo=PPOConfig(minibatches=8))


@pytest.mark.parametrize(("device", "fp16"), PATHS[:2])
def test_sampled_actions_follow_the_masked_policy(device: str, fp16: bool) -> None:
    torch.manual_seed(0)
    dev = torch.device(device)
    model = ActorCritic(2, 4, SMALL).to(dev)
    with torch.no_grad():
        for p in model.policy.parameters():
            p.zero_()
        model.policy[-1].bias.copy_(torch.tensor([0.0, math.log(3.0), 9.0, 0.0]))
    tensors = GroupTensors.allocate(4096, 2, 4, pin_memory=dev.type == "cuda")
    tensors.mask.copy_(torch.tensor([True, True, False, True]).expand(4096, 4))
    engine = GroupInference(model, tensors, dev, fp16=fp16, cuda_graph=dev.type == "cuda")
    counts = torch.zeros(4)
    changed, previous = 0.0, None
    for _ in range(25):
        engine.run()
        engine.wait()
        actions = tensors.actions.long()
        counts += torch.bincount(actions, minlength=4).float()
        if previous is not None:
            changed += (actions != previous).float().mean().item() / 24
        previous = actions.clone()
    frequencies = counts / counts.sum()
    # Allowed logits 0, log 3, 0 give probabilities 0.2, 0.6, 0.2; the masked action (logit 9) never.
    torch.testing.assert_close(frequencies, torch.tensor([0.2, 0.6, 0.0, 0.2]), atol=0.01, rtol=0)
    # Fresh noise on every replay: an agent's action changes with probability 1 - (0.2² + 0.6² + 0.2²).
    assert abs(changed - 0.56) < 0.02, f"actions changed between replays {changed:.3f} of the time"
    expected_log_probs = torch.log(torch.tensor([0.2, 0.6, 1.0, 0.2], device=dev))[engine.actions]
    torch.testing.assert_close(engine.log_probs, expected_log_probs)


@pytest.mark.parametrize("device,fp16", PATHS[:2])  # FP32 on the CPU and, with a graphics card, on the card
def test_an_out_of_memory_error_halves_the_minibatch_and_carries_on(device: str, fp16: bool) -> None:
    dev = torch.device(device)
    pinned = dev.type == "cuda"  # a CUDA graph reads the games' tensors only from pinned memory
    envs = CountingEnvs((2, 3), pin_memory=pinned)
    trainer = trainer_for(envs, 5 * 8, PPOConfig(epochs=1, minibatches=1), dev, fp16)
    real_learn = trainer.learner.learn
    failures = {"left": 2}

    def flaky_learn(*args: object, **kwargs: object) -> LearnStats:
        if failures["left"]:
            failures["left"] -= 1
            # A real out-of-memory error strikes part way: some weights and statistics already moved.
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.add_(1.0)
            trainer.learner.reward_scaler.moments.count += 1000
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (pretend)")
        return real_learn(*args, **kwargs)  # type: ignore[arg-type]

    trainer.learner.learn = flaky_learn  # type: ignore[method-assign]
    count_before = trainer.learner.reward_scaler.moments.count
    report = trainer.iteration()
    assert trainer.learner.config.minibatches == 4, "halved twice"
    assert report.learn.gradient_steps == 4 and len(trainer.notices) == 2
    # The failed attempts were undone: the update equals a clean run with 4 minibatches from the start.
    reference = trainer_for(
        CountingEnvs((2, 3), pin_memory=pinned), 5 * 8, PPOConfig(epochs=1, minibatches=4), dev, fp16
    )
    reference.iteration()
    for ours, theirs in zip(trainer.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(ours, theirs, atol=1e-5)
    assert trainer.learner.reward_scaler.moments.count < count_before + 1000, "statistics counted once"
    assert "ppo.minibatches 1 -> 2" in trainer.notices[0] and "settings file" in trainer.notices[0]


def test_the_buffered_snapshot_keeps_the_state_as_it_was_and_reuses_its_buffers() -> None:
    """The out-of-memory snapshot copies into buffers made once; what it holds is the state when it was
    taken, a retry restores it exactly, and the next iteration's copy reuses the same buffers."""
    from nitro.trainer import Snapshot, on_cpu

    envs = CountingEnvs((2, 3))
    trainer = trainer_for(envs, 5 * 8, PPOConfig(epochs=1, minibatches=1))
    trainer.snapshot = Snapshot(buffered=True)
    learner = trainer.learner
    first = trainer.snapshot.take(learner.state_dict())
    weights = {key: value.clone() for key, value in learner.model.state_dict().items()}
    with torch.no_grad():
        for parameter in learner.model.parameters():
            parameter.add_(1.0)
    assert all(torch.equal(first["model"][key], value) for key, value in weights.items()), "as taken"
    learner.load_state_dict(on_cpu(first))
    assert all(torch.equal(learner.model.state_dict()[key], value) for key, value in weights.items())
    second = trainer.snapshot.take(learner.state_dict())
    assert all(second["model"][key] is first["model"][key] for key in weights), "the buffers are reused"
    real_learn = learner.learn
    failures = {"left": 1}

    def flaky_learn(*args: object, **kwargs: object) -> LearnStats:
        if failures["left"]:
            failures["left"] -= 1
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.add_(1.0)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (pretend)")
        return real_learn(*args, **kwargs)  # type: ignore[arg-type]

    learner.learn = flaky_learn  # type: ignore[method-assign]
    trainer.iteration()
    reference = trainer_for(CountingEnvs((2, 3)), 5 * 8, PPOConfig(epochs=1, minibatches=2))
    reference.iteration()
    for ours, theirs in zip(trainer.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(ours, theirs, atol=1e-5)


def test_an_out_of_memory_error_that_never_stops_is_raised() -> None:
    envs = CountingEnvs((2, 3))
    trainer = trainer_for(envs, 5 * 8, PPOConfig(epochs=1, minibatches=1))

    def always(*args: object, **kwargs: object) -> LearnStats:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (pretend)")

    trainer.learner.learn = always  # type: ignore[method-assign]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        trainer.iteration()
    assert trainer.learner.config.minibatches == 16


@pytest.mark.parametrize(
    "device,fp16", PATHS
)  # the card's (FP32, FP16 with graphed steps) with a graphics card
@pytest.mark.parametrize("optimizer", ["adam", "muon"])
@pytest.mark.parametrize("where", ["step", "clipping"])
def test_an_out_of_memory_error_after_unscaling_is_retried(
    monkeypatch: pytest.MonkeyPatch, device: str, fp16: bool, optimizer: str, where: str
) -> None:
    """An error after the loss scaler's unscale_ (clipping, the optimizer's step) left it mid-step and
    the retry stopped with "unscale_() has already been called". The
    loss scale ends as a clean run's, and on a card the next update replays the graphs the retry recorded."""
    dev = torch.device(device)
    pinned = dev.type == "cuda"

    def make() -> Trainer:
        envs = CountingEnvs((2, 3), pin_memory=pinned)
        return trainer_for(envs, 5 * 8, PPOConfig(epochs=1, minibatches=1, optimizer=optimizer), dev, fp16)

    trainer = make()
    trainer.iteration()  # a clean one first: the optimizer's state and running returns exist
    learner = trainer.learner
    failures = {"left": 1}

    def failing(real: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            if failures["left"]:
                failures["left"] -= 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (pretend)")
            return real(*args, **kwargs)

        return call

    if where == "step":
        monkeypatch.setattr(learner.optimizer, "step", failing(learner.optimizer.step))
    else:
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", failing(torch.nn.utils.clip_grad_norm_))
    trainer.iteration()
    assert not failures["left"], "the error was raised"
    assert learner.config.minibatches == 2 and len(trainer.notices) == 1
    # A clean run at 2 minibatches from the second update on; the failed attempt drew its minibatch order from
    # PyTorch's shared generator (learner.generator is None), so the clean one draws one too.
    reference = make()
    reference.iteration()
    reference.learner.config = PPOConfig(epochs=1, minibatches=2, optimizer=optimizer)
    real_learn = reference.learner.learn

    def learn(rollout: Any, *args: Any, **kwargs: Any) -> LearnStats:
        torch.randperm(rollout.steps * rollout.agents, device=dev)
        return real_learn(rollout, *args, **kwargs)

    reference.learner.learn = learn  # type: ignore[method-assign]
    reference.iteration()
    for ours, theirs in zip(trainer.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(ours, theirs, atol=1e-5)
    assert learner.scaler.state_dict() == reference.learner.scaler.state_dict(), "the loss scale as clean"
    if learner.graphed is not None:  # the next update replays the graphs the retry recorded
        recorded = dict(learner.graphed.graphs)
        assert recorded
        trainer.iteration()
        assert all(learner.graphed.graphs.get(key) is graph for key, graph in recorded.items())


def test_a_fresh_loss_scaler_keeps_the_fp16_scale_and_takes_whole_steps_again() -> None:
    """The scaler that replaces one left mid-step keeps its scale and growth count (FP16's settings)."""
    model = ActorCritic(2, 3, SMALL)
    learner = Learner(model, PPOConfig(), gamma=0.9, device=CPU, fp16=True)

    def backward() -> None:
        learner.optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([p.sum() for p in model.parameters()]).sum()
        learner.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        learner.scaler.unscale_(learner.optimizer)

    backward()
    learner.scaler.step(learner.optimizer)
    learner.scaler.update(1024.0)
    backward()
    learner.scaler.step(learner.optimizer)
    learner.scaler.update()
    backward()  # cut short here: no step, no update
    with pytest.raises(RuntimeError, match="unscale_"):
        backward()
    old = learner.scaler
    assert old._get_growth_tracker() == 1  # update(1024.0) set the scale, the next counted a step
    learner.fresh_loss_scaler()
    scaler = learner.scaler
    assert scaler is not old
    # Its tensors already made (graphed steps' capture puts back only those), and kept by the next step.
    scale, growth = scaler._scale, scaler._growth_tracker
    assert scale is not None and growth is not None
    assert float(scale) == 1024.0 and int(growth) == 1 and scaler.get_growth_interval() == 2000
    backward()
    scaler.step(learner.optimizer)
    scaler.update()
    assert scaler._scale is scale and scaler._growth_tracker is growth
    assert scaler.get_scale() == 1024.0 and scaler._get_growth_tracker() == 2


def test_an_out_of_memory_retry_keeps_the_running_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry restored the return statistics from the snapshot, which leaves out each agent's running
    return, so the retried update started every episode still going at 0. 7 steps an iteration: episodes
    run across iterations."""

    def run(fail: bool) -> list[tuple[torch.Tensor, ...]]:
        trainer = trainer_for(CountingEnvs((2, 3)), 5 * 7, PPOConfig(epochs=1, minibatches=1))
        learner = trainer.learner
        calls = {"n": 0}
        real = learner._advantages

        def advantages(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if fail and calls["n"] == 2:  # the second update, after its reward scaler ran
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (pretend)")
            return real(*args, **kwargs)

        monkeypatch.setattr(learner, "_advantages", advantages)
        found = []
        for _ in range(4):
            trainer.iteration()
            scaler = learner.reward_scaler
            assert scaler.running is not None
            moments = scaler.moments
            found.append(tuple(t.clone() for t in (scaler.running, moments.count, moments.mean, moments.m2)))
        assert len(trainer.notices) == (1 if fail else 0)
        return found

    clean, retried = run(False), run(True)
    for number, (ours, theirs) in enumerate(zip(retried, clean, strict=True)):
        for name, a, b in zip(("running", "count", "mean", "m2"), ours, theirs, strict=True):
            assert torch.equal(a, b), f"iteration {number + 1}: {name}"


class OrderCheckingEnvs(CountingEnvs):
    """CountingEnvs that refuse a wait without a step before it, or two steps without a wait between."""

    def __init__(self, sizes: tuple[int, ...]) -> None:
        super().__init__(sizes)
        self.in_flight = [False] * len(sizes)
        self.steps_taken = [0] * len(sizes)

    def step(self, group: int) -> None:
        assert not self.in_flight[group], f"group {group} stepped twice without a wait"
        super().step(group)
        self.in_flight[group] = True
        self.steps_taken[group] += 1

    def wait(self, group: int) -> None:
        assert self.in_flight[group], f"group {group} waited for before it stepped"
        self.in_flight[group] = False


@pytest.mark.parametrize("sizes", [(2, 3), (2,), (1, 1, 2)])
def test_pipelined_launches_collect_exactly_the_same_rollout(sizes: tuple[int, ...]) -> None:
    rollouts = []
    for pipelined in (False, True):
        envs = OrderCheckingEnvs(sizes)
        trainer = trainer_for(envs, steps=7, ppo=PPOConfig(minibatches=1))
        trainer.pipelined = pipelined
        trainer.collect()
        assert envs.steps_taken == [trainer.steps] * len(sizes)
        assert not any(envs.in_flight), "every group's last step was waited for"
        r = trainer.rollout
        rollouts.append((r.obs.clone(), r.actions.clone(), r.rewards.clone(), r.values.clone()))
    for plain, pipelined in zip(*rollouts, strict=True):
        assert torch.equal(plain, pipelined)
