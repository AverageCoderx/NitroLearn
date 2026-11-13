"""ppo.muon_batched: the Muon matrices of one shape stepped together give the steps one at a time
gives, to rounding, with and without the loss scaler's skip, and the optimizer's saves are the same."""

from __future__ import annotations

import pytest
import torch

from nitro.muon import MuonAdam, orthogonalised, orthogonalised_stack


def matrices(seed: int) -> list[torch.nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    shapes = [(24, 24), (24, 24), (24, 24), (16, 40), (16, 40), (40, 16), (8, 8)]
    return [torch.nn.Parameter(torch.randn(*shape, generator=g)) for shape in shapes]


def optimizer(params: list[torch.nn.Parameter], batched: bool) -> MuonAdam:
    groups = [{"params": params[:4], "lr": 0.01}, {"params": params[4:], "lr": 0.02}]
    return MuonAdam(groups, params[:6], eps=1e-8, fused=False, batched=batched)


def muon_part(opt: MuonAdam, batched: bool, found_inf: torch.Tensor, grad_scale: torch.Tensor) -> None:
    if batched:
        opt._muon_steps(found_inf, grad_scale)
        return
    for group in opt.param_groups:
        for p in group["params"]:
            if id(p) in opt.muon:
                opt._muon_step(p, group["lr"], found_inf, grad_scale)


def stepped(batched: bool, skip: list[bool], scaled: bool) -> list[torch.Tensor]:
    params = matrices(0)
    opt = optimizer(params, batched)
    g = torch.Generator().manual_seed(1)
    for k, skipping in enumerate(skip):
        for p in params:
            p.grad = torch.randn(p.shape, generator=g) * 64
        if not scaled:
            opt.step()
        else:  # the Muon part as fused Adam's loss scaler hands it over (CPU Adam takes no found_inf)
            found_inf, grad_scale = torch.tensor(1.0 if skipping else 0.0), torch.tensor(64.0)
            with torch.no_grad():
                muon_part(opt, batched, found_inf, grad_scale)
        if k == 0:
            assert all("momentum_buffer" in opt.state[p] for p in params[:6])
    return [p.detach().clone() for p in params] + [
        opt.state[p]["momentum_buffer"].clone() for p in params[:6]
    ]


def test_a_stack_is_orthogonalised_as_each_matrix_alone() -> None:
    g = torch.Generator().manual_seed(3)
    for shape in [(5, 24, 24), (3, 16, 40), (3, 40, 16)]:
        stack = torch.randn(*shape, generator=g)
        together = orthogonalised_stack(stack, torch.float32)
        for k in range(shape[0]):
            torch.testing.assert_close(
                together[k], orthogonalised(stack[k], torch.float32), rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize("scaled", [False, True], ids=["plain", "loss scaler"])
def test_batched_steps_are_the_steps_one_at_a_time(scaled: bool) -> None:
    skips = [False, True, False, False] if scaled else [False] * 4
    for one, together in zip(stepped(False, skips, scaled), stepped(True, skips, scaled), strict=True):
        torch.testing.assert_close(together, one, rtol=1e-5, atol=1e-6)


def test_a_skipped_step_moves_nothing() -> None:
    before = matrices(0)
    after = stepped(True, [True], scaled=True)
    for p, q in zip(before, after[: len(before)], strict=True):
        assert torch.equal(p.detach(), q)


def test_the_saves_load_both_ways() -> None:
    params = matrices(0)
    one = optimizer(params, batched=False)
    for p in params:
        p.grad = torch.ones_like(p)
    one.step()
    together = optimizer(params, batched=True)
    together.load_state_dict(one.state_dict())
    assert together.batched and together.state_dict()["state"].keys() == one.state_dict()["state"].keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_on_a_card_batched_steps_are_the_steps_one_at_a_time_to_bf16_rounding() -> None:
    """On a card the Newton-Schulz products run in BF16 (`ns_dtype`): batched and one at a time are each
    within BF16's rounding of the FP32 answer (on a test card the batched one was the closer: 0.011
    against 0.037), and in FP32 they agree."""
    g = torch.Generator().manual_seed(5)
    stack = torch.randn(6, 600, 600, generator=g).cuda()
    together = orthogonalised_stack(stack, torch.bfloat16).float()
    exact = orthogonalised_stack(stack, torch.float32)
    for k in range(stack.shape[0]):
        alone = orthogonalised(stack[k], torch.bfloat16).float()
        reference = orthogonalised(stack[k], torch.float32)
        torch.testing.assert_close(exact[k], reference, rtol=1e-4, atol=1e-5)
        apart = (alone - reference).norm() / reference.norm()
        assert (together[k] - reference).norm() / reference.norm() < max(2 * apart.item(), 2e-2)
