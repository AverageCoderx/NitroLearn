"""`ppo.optimizer = "muon"`: Muon for the hidden layers' weight matrices, Adam for the rest.

The Newton-Schulz step is checked against torch's own (`torch.optim.Muon`'s, BF16) and against FP64; one
optimizer step against the update written out by hand and against `torch.optim.Muon` (whose
"match_rms_adamw" scaling is 0.2 x ours; to BF16's rounding); the Adam part against torch's Adam to the
bit; then the learner: which matrices take Muon, a skipped step, FP16 loss scaling, an exact resume, and a
switch of optimizer on a resume. On a graphics card (not here): fused Adam's in-kernel skip and graphed
steps."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.optim._muon import _zeropower_via_newtonschulz

from nitro.config.schema import ModelConfig, PPOConfig
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.muon import MOMENTUM, NS_COEFFICIENTS, MuonAdam, orthogonalised
from test_optimizer_options import ACTIONS, OBS, rollout_for

CPU = torch.device("cpu")
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
MODEL = ModelConfig(shared=(8, 8), policy=(16, 16), value=(12, 12), residual=True, activation="leaky_relu")


def by_hand_fp64(update: torch.Tensor) -> torch.Tensor:
    """The Newton-Schulz iteration written out in FP64."""
    a, b, c = NS_COEFFICIENTS
    x = update.double()
    tall = x.shape[0] > x.shape[1]
    x = x.T if tall else x
    x = x / x.norm()
    for _ in range(5):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


@pytest.mark.parametrize("shape", [(6, 10), (10, 6), (7, 7)])
def test_newton_schulz_matches_torch_and_fp64(shape: tuple[int, int]) -> None:
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    mine = orthogonalised(g, torch.float32)
    assert mine.shape == g.shape and mine.dtype == torch.float32
    torch.testing.assert_close(mine.double(), by_hand_fp64(g), rtol=0, atol=1e-5)
    reference = _zeropower_via_newtonschulz(g, NS_COEFFICIENTS, 5, 1e-7)
    torch.testing.assert_close(
        orthogonalised(g, torch.bfloat16).float(), reference.float(), rtol=0, atol=0.05
    )
    # Nearly orthogonal: every singular value between about 0.5 and 1.5 (Muon's quintic, by design).
    values = torch.linalg.svdvals(mine)
    assert float(values.min()) > 0.4 and float(values.max()) < 1.5


def test_one_step_is_the_update_written_out() -> None:
    torch.manual_seed(1)
    w = torch.nn.Parameter(torch.randn(5, 8))
    bias = torch.nn.Parameter(torch.randn(5))
    optimizer = MuonAdam([{"params": [w, bias], "lr": 0.01}], [w], eps=1e-5, fused=False)
    w0, buffer = w.detach().clone(), torch.zeros(5, 8)
    for _ in range(3):
        grad = torch.randn(5, 8)
        w.grad, bias.grad = grad.clone(), torch.randn(5)
        before = w.detach().clone()
        optimizer.step()
        buffer = MOMENTUM * buffer + grad
        step = orthogonalised(grad + MOMENTUM * buffer, torch.float32)
        expected = before - 0.01 * math.sqrt(8) * step
        torch.testing.assert_close(w.detach(), expected, rtol=0, atol=1e-6)
        assert torch.equal(optimizer.state[w]["momentum_buffer"], buffer)
    assert "exp_avg" in optimizer.state[bias] and "exp_avg" not in optimizer.state[w]
    # Each number moved by about the rate per step (0.6-0.9 lr on real gradients).
    rms = float((w.detach() - w0).pow(2).mean().sqrt()) / 3
    assert 0.003 < rms < 0.03


def test_a_step_matches_torch_muon() -> None:
    """torch.optim.Muon with 'match_rms_adamw' scales by 0.2 sqrt(max(rows, cols)): 5x the rate is ours.
    It keeps its momentum as a running average, a fixed multiple of ours, which the normalisation removes;
    both orthogonalise in BF16."""
    torch.manual_seed(2)
    mine, theirs = torch.nn.Parameter(torch.randn(6, 9)), torch.nn.Parameter(torch.randn(6, 9))
    with torch.no_grad():
        theirs.copy_(mine)
    ours = MuonAdam([{"params": [mine], "lr": 0.02}], [mine], eps=1e-5, fused=False, dtype=torch.bfloat16)
    reference = torch.optim.Muon([theirs], lr=0.1, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
    start = mine.detach().clone()
    for _ in range(4):
        grad = torch.randn(6, 9)
        mine.grad, theirs.grad = grad.clone(), grad.clone()
        ours.step()
        reference.step()
        # Five Newton-Schulz steps in BF16 (about 3 significant digits), each rounding in its own order:
        # within 5% of how far the weights have moved (2-3% measured).
        moved = float((mine.detach() - start).abs().max())
        assert float((mine.detach() - theirs.detach()).abs().max()) < 0.05 * moved


def test_everything_else_is_torchs_adam_to_the_bit() -> None:
    torch.manual_seed(3)
    params = [torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(3))]
    matrix = torch.nn.Parameter(torch.randn(4, 4))
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    mixed = MuonAdam(
        [{"params": [params[0], matrix, params[1]], "lr": 0.003}], [matrix], eps=1e-5, fused=False
    )
    adam = torch.optim.Adam([{"params": copies, "lr": 0.003}], eps=1e-5)
    for _ in range(5):
        grads = [torch.randn_like(p) for p in params]
        for p, c, g in zip(params, copies, grads, strict=True):
            p.grad, c.grad = g.clone(), g.clone()
        matrix.grad = torch.randn(4, 4)
        mixed.step()
        adam.step()
        assert all(torch.equal(p, c) for p, c in zip(params, copies, strict=True))
    assert [len(g["params"]) for g in mixed.param_groups] == [3], "the groups are whole again after a step"
    with pytest.raises(ValueError, match="matrices"):
        MuonAdam([{"params": params, "lr": 0.1}], [params[1]], eps=1e-5, fused=False)


def test_a_skipped_step_leaves_the_matrix_and_its_momentum() -> None:
    """With fused Adam the loss scaler hands `found_inf` to the step (on the card); here Muon's part alone."""
    torch.manual_seed(4)
    w = torch.nn.Parameter(torch.randn(3, 5))
    optimizer = MuonAdam([{"params": [w], "lr": 0.1}], [w], eps=1e-5, fused=False)
    w.grad = torch.full((3, 5), math.nan)
    kept = w.detach().clone()
    with torch.no_grad():
        optimizer._muon_step(w, 0.1, torch.ones(()), None)
    assert torch.equal(w.detach(), kept) and not optimizer.state[w]["momentum_buffer"].any()
    w.grad = torch.randn(3, 5)
    plain = copy.deepcopy(optimizer)
    plain_w = plain.param_groups[0]["params"][0]
    plain_w.grad = w.grad / 4.0
    with torch.no_grad():
        optimizer._muon_step(w, 0.1, torch.zeros(()), torch.full((), 4.0))  # scaled by 4, not yet unscaled
        plain._muon_step(plain_w, 0.1, None, None)
    assert torch.equal(w.detach(), plain_w.detach())


def learner(seed: int = 5, fp16: bool = False, **ppo: object) -> Learner:
    torch.manual_seed(seed)
    model = ActorCritic(OBS, ACTIONS, MODEL)
    config = PPOConfig(**({"epochs": 2, "minibatches": 2, "optimizer": "muon", "graphed_steps": False} | ppo))  # type: ignore[arg-type]
    return Learner(model, config, gamma=0.9, device=CPU, fp16=fp16)


def test_the_learner_takes_muon_for_the_hidden_matrices_only() -> None:
    learn = learner()
    model = learn.model
    assert isinstance(learn.optimizer, MuonAdam)
    shapes = sorted(tuple(p.shape) for p in model.muon_parameters())
    # Not the trunk's first layer (it reads the observation), nor the two output layers.
    assert shapes == sorted([(8, 8), (16, 8), (16, 16), (12, 8), (12, 12)])
    assert all(
        p is not model.policy[-1].weight and p is not model.value_output.weight
        for p in model.muon_parameters()
    )
    for seed in range(3):
        stats = learn.learn(rollout_for(learn.model, seed))
        assert stats.losses_finite and stats.skipped_updates == 0
    muon = {id(p) for p in model.muon_parameters()}
    for p, state in learn.optimizer.state.items():
        assert ("momentum_buffer" in state) == (id(p) in muon)
    # The same start with Adam alone learns other weights.
    pair = [learner(optimizer="adam"), learner()]
    for one in pair:
        torch.manual_seed(9)
        one.learn(rollout_for(one.model, 0))
    assert not torch.equal(pair[0].model.muon_parameters()[0], pair[1].model.muon_parameters()[0])


@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_a_resumed_muon_run_learns_exactly_as_one_that_never_stopped(fp16: bool) -> None:
    straight = learner(fp16=fp16)
    straight.learn(rollout_for(straight.model, 0))
    saved = copy.deepcopy(straight.state_dict())
    assert saved["optimizer_kind"] == "muon"
    torch.manual_seed(50)
    straight.learn(rollout_for(straight.model, 1))

    resumed = learner(seed=99, fp16=fp16)  # other starting weights: everything comes from the save
    resumed.load_state_dict(saved)
    torch.manual_seed(50)
    resumed.learn(rollout_for(resumed.model, 1))
    for a, b in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    for p, q in zip(straight.model.parameters(), resumed.model.parameters(), strict=True):
        for key, value in straight.optimizer.state[p].items():
            assert torch.equal(value, resumed.optimizer.state[q][key]), key
    if fp16:
        assert straight.scaler.get_scale() == resumed.scaler.get_scale() > 1


def test_switching_the_optimizer_on_a_resume_starts_it_afresh(caplog: pytest.LogCaptureFixture) -> None:
    adam = learner(optimizer="adam")
    adam.learn(rollout_for(adam.model, 0))
    saved = adam.state_dict()
    assert "optimizer_kind" not in saved, "an Adam save is as before"
    muon = learner(seed=7)
    muon.load_state_dict(saved)
    assert "starts afresh" in caplog.text
    assert not muon.optimizer.state
    assert all(
        torch.equal(a, b) for a, b in zip(adam.model.parameters(), muon.model.parameters(), strict=True)
    )
    muon.learn(rollout_for(muon.model, 1))
    back = learner(optimizer="adam")
    back.load_state_dict(muon.state_dict())
    assert not back.optimizer.state
    assert back.learn(rollout_for(back.model, 2)).losses_finite


@NEEDS_GPU
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_on_a_card_muon_learns_and_a_bad_step_is_skipped_inside_the_kernel(fp16: bool) -> None:
    """Fused Adam on the card: the loss scaler hands found_inf to the step, which skips Muon's part too."""
    from test_graphed_steps import learner as card_learner
    from test_graphed_steps import rollout

    learn = card_learner(torch.device("cuda"), fp16=fp16, graphed_steps=False, optimizer="muon")
    assert isinstance(learn.optimizer, MuonAdam) and learn.optimizer._step_supports_amp_scaling
    for seed in range(3):
        stats = learn.learn(rollout(learn.model, seed))
        assert stats.losses_finite and stats.skipped_updates == 0
    matrix = learn.model.muon_parameters()[0]
    kept = matrix.detach().clone()
    buffer = learn.optimizer.state[matrix]["momentum_buffer"].clone()
    learn.optimizer.zero_grad()
    for p in learn.model.parameters():
        p.grad = torch.full_like(p, math.inf)
    learn.scaler.unscale_(learn.optimizer)
    learn.scaler.step(learn.optimizer)
    learn.scaler.update()
    assert torch.equal(matrix.detach(), kept) and torch.equal(
        learn.optimizer.state[matrix]["momentum_buffer"], buffer
    )


@NEEDS_GPU
def test_graphed_muon_steps_give_the_eager_updates() -> None:
    """For lifting the settings refusal of graphed_steps with Muon: the same updates to the bit."""
    from test_graphed_steps import learner as graphed_learner
    from test_graphed_steps import rollout

    eager, graphed = (
        graphed_learner(torch.device("cuda"), microbatch=0, graphed_steps=on, optimizer="muon")
        for on in (False, True)
    )
    for seed in range(4):
        for learn in (eager, graphed):
            torch.manual_seed(100 + seed)
            learn.learn(rollout(learn.model, seed))
        for a, b in zip(eager.model.parameters(), graphed.model.parameters(), strict=True):
            assert torch.equal(a, b)
    assert graphed.graphed is not None and graphed.graphed.graphs, "the steps ran graphed"
