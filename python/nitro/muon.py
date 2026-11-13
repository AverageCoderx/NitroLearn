"""Muon for the hidden layers' weight matrices, Adam for every other parameter (`ppo.optimizer = "muon"`).

Muon (Keller Jordan's optimizer, from its public description): each matrix keeps a momentum buffer
B = 0.95 B + G, steps along the Nesterov direction G + 0.95 B made nearly orthogonal by five Newton-Schulz
iterations (its singular values pushed toward 1), and moves by lr x sqrt(max(rows, cols)) times that (times
ppo.muon_lr_scale). Its numbers then move by 0.6-0.9 lr a step (measured on a long GCO run's save). Adam's
move 0.1-0.3 lr on PPO's noisy gradients, so at one rate Muon's move 3-5x Adam's on the shared trunk and the
value side; on the policy, where Adam's move about 0.4 lr, the two are roughly equal. Every Muon matrix feeds
a LayerNorm, so what changes the network is the step relative to the
matrix's size, which falls as the matrix grows (ppo.weight_decay holds it).

torch 2.14 has `torch.optim.Muon`, but not as needed here: its rate scalings are other ones, it takes
matrices only (a second optimizer beside Adam, while the learner's saves, rates, loss scaler and graphs
expect one), and it cannot skip a step inside the kernel as fused Adam does, so the loss scaler would wait for
the card every step. Here one optimizer is Adam (torch's own, the same maths and saves) for everything it
holds except the Muon matrices, which it steps itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch

MOMENTUM = 0.95
NS_STEPS = 5
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_EPS = 1e-7


def orthogonalised(update: torch.Tensor, dtype: torch.dtype, steps: int = NS_STEPS) -> torch.Tensor:
    """`update` (a matrix) with its singular values pushed toward 1 by `steps` quintic Newton-Schulz
    iterations in `dtype`: X <- a X + (b A + c A^2) X with A = X X^T, after X = update / |update|_F (so every
    singular value starts at most 1). Works on the side with fewer rows, so A is the smaller square."""
    a, b, c = NS_COEFFICIENTS
    tall = update.shape[0] > update.shape[1]
    x = update.T if tall else update
    x = (x.float() / x.float().norm().clamp_min(_EPS)).to(dtype)
    for _ in range(steps):
        gram = x @ x.T
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, poly, x, beta=a)
    return x.T if tall else x


def orthogonalised_stack(updates: torch.Tensor, dtype: torch.dtype, steps: int = NS_STEPS) -> torch.Tensor:
    """`orthogonalised` for a stack of matrices of one shape ([k, rows, cols]) at once (ppo.muon_batched): the
    same maths, each matrix normalised by its own size, the products batched (the same results to
    rounding: a batched product may sum in another order)."""
    a, b, c = NS_COEFFICIENTS
    tall = updates.shape[1] > updates.shape[2]
    x = updates.mT if tall else updates
    size = x.float().flatten(1).norm(dim=1).clamp_min(_EPS)[:, None, None]
    x = (x.float() / size).to(dtype)
    for _ in range(steps):
        gram = x @ x.mT
        poly = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, poly, x, beta=a)
    return x.mT if tall else x


def ns_dtype(device: torch.device) -> torch.dtype:
    """BF16 on graphics cards that compute it natively (Ampere and newer), as Muon's authors run it; FP32
    elsewhere (the processor, and older cards, which only emulate BF16)."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


class MuonAdam(torch.optim.Adam):
    """Adam over `params` (groups as torch.optim.Adam takes them), except the matrices in `muon`, which take
    Muon steps at their group's rate. Their state is a "momentum_buffer" and a "step" count (Adam's code
    wants one in every state; Muon does not use it); the rest is Adam's own state, so the saves load as an
    Adam's do. With fused Adam the loss scaler hands over `found_inf` and the step
    is skipped on the card (Muon's part too), never waiting for it."""

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        muon: Iterable[torch.Tensor],
        *,
        eps: float,
        fused: bool,
        dtype: torch.dtype | None = None,
        batched: bool = False,
        scale: float = 1.0,
    ) -> None:
        super().__init__(params, eps=eps, fused=fused)
        self.batched = batched
        self.scale = scale  # ppo.muon_lr_scale: on the Muon matrices' rate only
        matrices = list(muon)
        held = {id(p) for group in self.param_groups for p in group["params"]}
        if any(p.ndim != 2 or id(p) not in held for p in matrices):
            raise ValueError("Muon steps weight matrices the optimizer holds")
        self.muon = {id(p) for p in matrices}
        self.dtype = dtype

    def __getstate__(self) -> dict[str, Any]:
        """A copy (pickle, deepcopy) gets new parameters: the Muon matrices go by their places."""
        places = [
            (g, i)
            for g, group in enumerate(self.param_groups)
            for i, p in enumerate(group["params"])
            if id(p) in self.muon
        ]
        state: dict[str, Any] = super().__getstate__()
        return {
            **state,
            "dtype": self.dtype,
            "batched": self.batched,
            "scale": self.scale,
            "muon_places": places,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        places = state.pop("muon_places", None)
        state.setdefault("batched", getattr(self, "batched", False))  # a load keeps the setting
        state.setdefault("scale", getattr(self, "scale", 1.0))
        super().__setstate__(state)  # type: ignore[no-untyped-call]  # load_state_dict comes here too
        if places is not None:
            self.muon = {id(self.param_groups[g]["params"][i]) for g, i in places}

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        if closure is not None:
            raise ValueError("MuonAdam takes no closure")
        found_inf: torch.Tensor | None = getattr(self, "found_inf", None)
        grad_scale: torch.Tensor | None = getattr(self, "grad_scale", None)
        if self.batched:
            self._muon_steps(found_inf, grad_scale)
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    if id(p) in self.muon and p.grad is not None:
                        self._muon_step(p, group["lr"], found_inf, grad_scale)
        everything = [group["params"] for group in self.param_groups]
        try:  # Adam steps the rest: torch's own code, with the Muon matrices out of its groups
            for group in self.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in self.muon]
            super().step()
        finally:
            for group, params in zip(self.param_groups, everything, strict=True):
                group["params"] = params
        return None

    def _muon_step(
        self,
        p: torch.Tensor,
        lr: float,
        found_inf: torch.Tensor | None,
        grad_scale: torch.Tensor | None,
    ) -> None:
        assert p.grad is not None
        grad = p.grad if grad_scale is None else p.grad / grad_scale
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["step"] = torch.zeros((), device=p.device)
            state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        buffer: torch.Tensor = state["momentum_buffer"]
        moved = buffer * MOMENTUM + grad
        direction = orthogonalised(grad + MOMENTUM * moved, self.dtype or ns_dtype(p.device))
        rate = lr * self.scale * math.sqrt(max(p.shape))
        if found_inf is None:  # the loss scaler skips the whole step itself when a gradient is not finite
            state["step"].add_(1)
            buffer.copy_(moved)
            p.add_(direction.to(p.dtype), alpha=-rate)
            return
        skip = found_inf > 0  # on the card: kept as they were, NaN or not in the new values
        state["step"].add_(1 - skip.float())
        buffer.copy_(torch.where(skip, buffer, moved))
        p.copy_(torch.where(skip, p, torch.add(p, direction.to(p.dtype), alpha=-rate)))

    def _muon_steps(self, found_inf: torch.Tensor | None, grad_scale: torch.Tensor | None) -> None:
        """`_muon_step` for every Muon matrix, those of one shape, type and rate together (ppo.muon_batched):
        the element-wise parts as foreach calls, the Newton-Schulz products batched, so a step is a few
        dozen GPU jobs instead of about 30 per matrix."""
        together: dict[tuple[Any, ...], list[torch.Tensor]] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in self.muon and p.grad is not None:
                    together.setdefault((tuple(p.shape), p.dtype, p.device, group["lr"]), []).append(p)
        for (shape, _, device, lr), params in together.items():
            states = [self.state[p] for p in params]
            for p, state in zip(params, states, strict=True):
                if "momentum_buffer" not in state:
                    state["step"] = torch.zeros((), device=device)
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            grads = [p.grad for p in params if p.grad is not None]
            if grad_scale is not None:  # the stack divided once: foreach with a card scalar may wait
                grads = list((torch.stack(grads) / grad_scale).unbind(0))
            buffers = [state["momentum_buffer"] for state in states]
            moved = torch._foreach_mul(buffers, MOMENTUM)
            torch._foreach_add_(moved, grads)
            nesterov = torch._foreach_add(grads, torch._foreach_mul(moved, MOMENTUM))
            direction = orthogonalised_stack(torch.stack(nesterov), self.dtype or ns_dtype(device))
            rate = lr * self.scale * math.sqrt(max(shape))
            steps = [state["step"] for state in states]
            moves = list(direction.to(params[0].dtype).unbind(0))
            if found_inf is None:
                torch._foreach_add_(steps, 1)
                torch._foreach_copy_(buffers, moved)
                torch._foreach_add_(params, moves, alpha=-rate)
                continue
            skip = found_inf > 0
            torch._foreach_add_(steps, [1 - skip.float()] * len(steps))
            kept = torch.stack(buffers)
            torch._foreach_copy_(buffers, torch.where(skip, kept, torch.stack(moved)).unbind(0))
            current = torch.stack(params)
            stepped = torch.add(current, torch.stack(moves), alpha=-rate)
            torch._foreach_copy_(params, torch.where(skip, current, stepped).unbind(0))
