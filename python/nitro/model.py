"""Policy and value networks: multilayer perceptrons, optionally reading the observation through an
attention encoder first (`model.encoder`: "attention", "attention_communication" or "transformer").

An attention encoder's weights do not depend on how many cars it reads: one shared embedding for every
car, empty places masked. So a network trained with room for N players a side reads an observation with
room for any other number (`_car_places`), or exactly the cars of a game (`places`, as the RLBot bot does
beyond its room)."""

from __future__ import annotations

import dataclasses
import itertools
import math
import weakref
from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from nitro import small_attention
from nitro.config.schema import ModelConfig
from nitro.fused_norm import AppliedReLU, FusedLayerNormReLU

if TYPE_CHECKING:
    from nitro.config.schema import NitroConfig

# A padded observation's layout: the size of the own part (own car, ball, pads, previous action),
# the size of each other car's block (its last number 1 when a car fills the place, 0 when it is empty),
# and the teammate and opponent places, teammates first.
Layout = tuple[int, int, int, int]
# How many teammate and opponent blocks an observation holds.
Places = tuple[int, int]
# Where the ball and the own car sit in a padded observation's own part: ball size (it comes first), own car
# start and size (nitro_core::obs; `entity_parts`).
Parts = tuple[int, int, int]

# The activations `model.fused_norm` runs in its routine.
FUSED_ACTIVATIONS = ("relu", "leaky_relu")

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class _ReLU(nn.Module):
    """ReLU in place. With `to_half`, under FP16 autocast its result is passed on in FP16.

    The next linear layer would round it to FP16 anyway, and rounding before ReLU gives exactly the same
    forward numbers, so this only saves memory traffic: FP16 learning ~5% faster.
    The backward pass differs in one corner: an input in (0, 2^-25]
    rounds to 0 in FP16, so its gradient is dropped where FP32 would have passed it.
    Without `to_half` an FP16 input (a `_LayerNorm` in FP16 before it) leaves in FP32 after the ReLU,
    for an FP32 output layer; with `keep` (a shared trunk's last layer) it stays in FP16 into both heads,
    whose linear layers take FP16 anyway. Both give the bits FP32 throughout gave, gradients included: two
    heads' FP16 gradients add up in FP32 with one rounding either way, with less memory traffic.
    """

    def __init__(self, to_half: bool, keep: bool = False) -> None:
        super().__init__()
        self.to_half = to_half
        self.keep = keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device.type
        if (
            self.to_half
            and x.dtype == torch.float32
            and torch.is_autocast_enabled(device)
            and torch.get_autocast_dtype(device) == torch.float16
        ):
            x = x.half()
        elif not self.to_half and not self.keep and x.dtype == torch.float16:
            return torch.relu_(x).float()
        return torch.relu_(x)


class ResidualBlock(nn.Module):
    """`x + layer(x)` (`model.residual`): `branch` is one hidden layer as `mlp` makes it (linear,
    LayerNorm, activation), so the LayerNorm sits inside the branch after the linear, as in every plain
    layer, and nothing normalises the sum. The sum is FP32 under FP16 autocast: the stream that carries the
    blocks' outputs stays in full precision, the branch runs in the caller's (as the transformer's)."""

    def __init__(self, branch: nn.Sequential) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x.float() + self.branch(x)
        return result


def _hidden_layer(size: int, width: int, last: bool, trunk: bool, config: ModelConfig) -> list[nn.Module]:
    """One hidden layer: linear, LayerNorm (optional), activation. `last`: the stack's last hidden layer;
    `trunk`: the stack has no output layer of its own (a shared trunk)."""
    linear = nn.Linear(size, width)
    if config.init == "orthogonal":
        nn.init.orthogonal_(linear.weight, math.sqrt(2))
        nn.init.zeros_(linear.bias)
    if config.fused_norm and config.layer_norm and config.activation in FUSED_ACTIVATIONS:
        # One GPU routine for both; the slot keeps the layers numbered, so the saves alike.
        act = _ReLU(to_half=not last) if config.activation == "relu" else nn.LeakyReLU()
        fused = FusedLayerNormReLU(width, act, to_half=not last)
        return [linear, fused, AppliedReLU(fused.slope)]
    layers: list[nn.Module] = [linear]
    if config.layer_norm:
        layers.append(nn.LayerNorm(width))
    if config.activation == "relu":
        layers.append(_ReLU(to_half=not last, keep=last and trunk))
    else:
        layers.append(_ACTIVATIONS[config.activation]())
    return layers


def mlp(inputs: int, hidden: Sequence[int], outputs: int | None, config: ModelConfig) -> nn.Sequential:
    """Linear → LayerNorm (optional) → activation per hidden layer, then an optional output layer. With
    `model.residual` a hidden layer as wide as its input is a `ResidualBlock`.

    Only layers that feed the next hidden layer pass on FP16: the last one feeds an FP32 output layer, or
    (a shared trunk) two networks whose gradients must add up in FP32.
    """
    layers: list[nn.Module] = []
    size = inputs
    for i, width in enumerate(hidden):
        layer = _hidden_layer(size, width, i == len(hidden) - 1, outputs is None, config)
        if config.residual and width == size:
            layers.append(ResidualBlock(nn.Sequential(*layer)))
        else:
            layers += layer
        size = width
    if outputs is not None:
        layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


def _flat_layers(stack: nn.Sequential) -> list[nn.Module]:
    """A stack's layers in order, each residual block's branch spelled out."""
    out: list[nn.Module] = []
    for module in stack:
        out += list(module.branch) if isinstance(module, ResidualBlock) else [module]
    return out


def _output(head: nn.Sequential) -> nn.Linear:
    layer = head[-1]
    assert isinstance(layer, nn.Linear)
    return layer


def _run_head(head: nn.Sequential, features: torch.Tensor) -> torch.Tensor:
    """Hidden layers in the caller's precision; the output layer always in FP32."""
    hidden = head[:-1](features)
    with torch.autocast(features.device.type, enabled=False):
        result: torch.Tensor = _output(head)(hidden.float())
    return result


def _car_places(layout: Layout, width: int) -> Places:
    """The teammate and opponent blocks in an observation `width` numbers long: the trained places, or
    another room's (a padded observation with room for r a side has r - 1 teammate and r opponent places)."""
    own, block, teammates, opponents = layout
    blocks, rest = divmod(width - own, block)
    if not rest and blocks == teammates + opponents:
        return teammates, opponents
    if rest or blocks < 1 or blocks % 2 == 0:
        raise ValueError(
            f"an observation of {width} numbers is not an own part of {own} and a padded observation's "
            f"car blocks of {block}"
        )
    return (blocks - 1) // 2, (blocks + 1) // 2


def _sides(trained: torch.Tensor, layout: Layout, places: Places) -> torch.Tensor:
    """0 for each teammate block and 1 for each opponent block: `trained` (the buffer made for `layout`'s
    places) when `places` are those."""
    teammates, opponents = places
    if places == layout[2:]:
        return trained
    return (torch.arange(teammates + opponents, device=trained.device) >= teammates).long()


def _product_dtype(device_type: str, otherwise: torch.dtype) -> torch.dtype:
    """What a matrix product gives on `device_type`: autocast's dtype while it is on, else `otherwise`."""
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return otherwise


def _mask_bias(present: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[b, 1, tokens], added to the scores: 0 where a token may be looked at, -inf where not."""
    return torch.where(present, 0.0, -math.inf).to(dtype)[:, None]


# A one-query attention's weights with its LayerNorms folded in (`_fold_look`).
Folded = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _fold_look(
    query_norm: nn.LayerNorm,
    q: tuple[torch.Tensor, torch.Tensor],
    token_norm: nn.LayerNorm,
    k: torch.Tensor,
    v: tuple[torch.Tensor, torch.Tensor],
    out: tuple[torch.Tensor, torch.Tensor],
    heads: int,
) -> Folded:
    """The weights `_look_once` takes, from a query's and the tokens' LayerNorms and the q, k, v and output
    layers, in the weights' own precision whatever autocast does: W_q and b_q for the query normalised
    without scale and shift; W_k with the tokens' norm scale and 1/sqrt(head size), its bias dropped (it adds
    the same to every token's score, which the softmax ignores); W_v with the tokens' norm scale; and the
    output layer's bias with the values' bias carried through (the attention weights add up to 1)."""
    (wq, bq), (wv, bv), (wo, bo) = q, v, out
    scale = 1.0 / math.sqrt(wq.shape[0] // heads)
    with torch.autocast(wq.device.type, enabled=False):
        q_weight = wq * query_norm.weight
        q_bias = torch.addmv(bq, wq, query_norm.bias)
        k_weight = k * (token_norm.weight * scale)
        v_weight = wv * token_norm.weight
        out_bias = torch.addmv(bo, wo, torch.addmv(bv, wv, token_norm.bias))
    return q_weight, q_bias, k_weight, v_weight, out_bias


def _look_once(
    query: torch.Tensor,
    tokens: torch.Tensor,
    bias: torch.Tensor,
    folded: Folded,
    out_weight: torch.Tensor,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head attention of one query per row over tokens without making their keys and values
    (`model.lean_attention`). `query` [b, w] and `tokens` [b, t, w] are normalised without scale and
    shift (`tokens` in the products' dtype), `bias` the additive mask [b, 1, t], `folded` from `_fold_look`.
    Per head h: score_j = (W_k,h^T q_h) . n_j, and the output is W_out(W_v,h sum_j w_j n_j) plus the folded
    output bias. Returns the output [b, w] and the attention [b, heads, t]."""
    q_weight, q_bias, k_weight, v_weight, out_bias = folded
    b, w = query.shape
    d = w // heads
    q = F.linear(query, q_weight, q_bias).view(b, heads, d).transpose(0, 1)  # [h, b, d]
    keys = torch.bmm(q, k_weight.view(heads, d, w).to(q.dtype)).transpose(0, 1)  # [b, h, w]
    weights = torch.baddbmm(bias, keys, tokens.transpose(1, 2)).softmax(-1)
    mixed = torch.bmm(weights.to(tokens.dtype), tokens).transpose(0, 1)  # [h, b, w]
    values = torch.bmm(mixed, v_weight.view(heads, d, w).transpose(1, 2).to(mixed.dtype))  # [h, b, d]
    return F.linear(values.transpose(0, 1).reshape(b, w), out_weight, out_bias), weights


class _Look(nn.Module):
    """A module whose lean pass looks once (`_look_once`) with weights it folds from its own (`fold`)."""

    def fold(self) -> tuple[Folded, torch.Tensor]:
        """`_fold_look`'s folds and the output weight, from the weights as they are now."""
        raise NotImplementedError


class _AttentionStep(_Look):
    """One step of the own car's attention over the other cars (pre-norm, as EARL-Perceiver): masked
    multi-head attention, then a 4x feed-forward, each added back. The products follow autocast (FP16 under
    FP16 autocast, which would cast a `.float()` operand straight back); the softmax runs in FP32."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_query, self.norm_cars, self.norm_ff = (nn.LayerNorm(width) for _ in range(3))
        self.q, self.k, self.v, self.out = (nn.Linear(width, width) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))

    def forward(
        self, query: torch.Tensor, cars: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The query after this step, and the attention over the cars ([b, heads, cars])."""
        b, t, w = cars.shape
        h, d = self.heads, w // self.heads
        q = self.q(self.norm_query(query)).view(b, 1, h, d).transpose(1, 2)
        normed = self.norm_cars(cars)
        k = self.k(normed).view(b, t, h, d).transpose(1, 2)
        v = self.v(normed).view(b, t, h, d).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)  # [b, h, 1, t]
        weights = scores.masked_fill(~present[:, None, None, :], float("-inf")).softmax(-1)
        mixed = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b, 1, w)
        return self._finish(query, mixed), weights[:, :, 0]

    def forward_kv(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        null_kv: tuple[torch.Tensor, torch.Tensor],
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` with the cars' keys and values already made (`model.attention_fast`) and the null
        car's apart, so it is not copied into every observation's tokens."""
        b, t, w = k.shape
        h, d = self.heads, w // self.heads
        # q stays FP32: the null car's score uses it in FP32, and its two gradients then add up in FP32.
        q = self.q(self.norm_query(query)).view(b, 1, h, d).transpose(1, 2).float()
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        k_null, v_null = (x.view(1, h, 1, d) for x in null_kv)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)  # [b, h, 1, t]
        scores = scores.masked_fill(~present[:, None, None, :], float("-inf"))
        null = (q * k_null.float()).sum(-1, keepdim=True) / math.sqrt(d)  # [b, h, 1, 1]
        weights = torch.cat([null, scores], dim=-1).softmax(-1)
        mixed = weights[..., :1].to(v.dtype) * v_null.to(v.dtype) + weights[..., 1:].to(v.dtype) @ v
        return self._finish(query, mixed.transpose(1, 2).reshape(b, 1, w)), weights[:, :, 0]

    def forward_lean(
        self, query: torch.Tensor, tokens: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` for `model.lean_attention`: query [b, w], the tokens normalised once without this
        step's scale and shift, in the products' dtype, and the additive mask (`_look_once`). Returns the
        query after this step and the attention [b, heads, tokens]."""
        folded, out_weight = self.fold()
        normed = F.layer_norm(query, query.shape[-1:], eps=self.norm_query.eps)
        mixed, weights = _look_once(normed, tokens, bias, folded, out_weight, self.heads)
        query = query + mixed
        result: torch.Tensor = query + self.ff(self.norm_ff(query))
        return result, weights

    def fold(self) -> tuple[Folded, torch.Tensor]:
        folded = _fold_look(
            self.norm_query,
            (self.q.weight, self.q.bias),
            self.norm_cars,
            self.k.weight,
            (self.v.weight, self.v.bias),
            (self.out.weight, self.out.bias),
            self.heads,
        )
        return folded, self.out.weight

    def _finish(self, query: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        query = query + self.out(mixed)
        result: torch.Tensor = query + self.ff(self.norm_ff(query))
        return result


class EntityEncoder(nn.Module):
    """Reads a padded observation as the own car (the query) and the other cars (tokens): each car through
    one shared embedding plus a teammate/opponent embedding, empty places masked, and a learned null car
    always present so a game with no other car still has something to attend to. Returns one
    `width`-sized vector per observation, with `skip` the own part of the observation in front of it
    (`outputs` numbers); the order of the cars within a side does not matter, nor how many there are."""

    def __init__(
        self,
        layout: Layout,
        width: int,
        steps: int,
        heads: int,
        skip: bool = False,
        fast: bool = False,
        lean: bool = False,
    ) -> None:
        super().__init__()
        self.layout, self.width, self.steps_, self.heads, self.skip = layout, width, steps, heads, skip
        # `model.attention_fast` and `model.lean_attention` (which replaces both other paths).
        self.fast, self.lean = fast, lean
        # `nitro watch` sets it: each forward then keeps the last step's attention ([b, heads, 1 + cars],
        # the null car first) in `attention_weights`.
        self.keep_weights = False
        self.attention_weights: torch.Tensor | None = None
        self.outputs = width + (layout[0] if skip else 0)
        own, block, teammates, opponents = layout
        self.query = nn.Sequential(nn.Linear(own, width), nn.ReLU(), nn.Linear(width, width))
        self.car = nn.Sequential(nn.Linear(block - 1, width), nn.ReLU(), nn.Linear(width, width))
        # Small random sides, so a new network already tells teammates from opponents.
        self.sides = nn.Parameter(torch.randn(2, width) * 0.02)
        self.null = nn.Parameter(torch.zeros(width))
        self.attention = nn.ModuleList(_AttentionStep(width, heads) for _ in range(steps))
        self.register_buffer("side", torch.tensor([0] * teammates + [1] * opponents), persistent=False)

    def forward(self, observations: torch.Tensor, places: Places | None = None) -> torch.Tensor:
        """`places`: the teammate and opponent blocks after the own part, when they are neither the trained
        ones nor another room's padded observation's (one block per car present, as the bot beyond its
        room)."""
        own, block = self.layout[:2]
        teammates, opponents = places or _car_places(self.layout, observations.shape[-1])
        lead = observations.shape[:-1]  # any leading dimensions, as a linear layer takes
        obs = observations.reshape(-1, observations.shape[-1])
        b = obs.shape[0]
        cars = obs[:, own:].reshape(b, teammates + opponents, block)
        present = cars[..., -1] > 0.5
        side = _sides(self.side, self.layout, (teammates, opponents))  # type: ignore[arg-type]
        tokens = self.car(cars[..., :-1]) + self.sides[side]
        query = self.query(obs[:, :own]).unsqueeze(1)
        if self.lean:
            query, weights = self._attend_lean(query[:, 0], tokens, present)
        elif self.fast:
            query, weights = self._attend_fast(query, tokens, present)
        else:
            tokens = torch.cat([self.null.expand(b, 1, -1).to(tokens.dtype), tokens], dim=1)
            present = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=obs.device), present], dim=1)
            for step in self.attention:
                query, weights = step(query, tokens, present)
        if self.keep_weights:
            self.attention_weights = weights.detach().float().reshape(*lead, *weights.shape[1:])
        encoded = query.reshape(b, self.width)
        if self.skip:
            encoded = torch.cat([obs[:, :own].to(encoded.dtype), encoded], dim=1)
        result: torch.Tensor = encoded.reshape(*lead, self.outputs)
        return result

    def _attend_fast(
        self, query: torch.Tensor, tokens: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The steps with the same maths in fewer passes. The cars do not change between steps, so
        they are normalised once; each step's own normalisation scale and shift are folded into its key and
        value weights, every step's keys and values come from one matrix product, and the null car's keys
        and values are made once for the whole batch."""
        steps = [step for step in self.attention if isinstance(step, _AttentionStep)]
        eps = steps[0].norm_cars.eps
        weights, biases = [], []
        for step in steps:
            scale, shift = step.norm_cars.weight, step.norm_cars.bias
            for linear in (step.k, step.v):
                weights.append(linear.weight * scale)
                biases.append(linear.weight @ shift + linear.bias)
        weight, bias = torch.cat(weights), torch.cat(biases)
        parts = 2 * len(steps)
        kv = F.linear(F.layer_norm(tokens, (self.width,), eps=eps), weight, bias).chunk(parts, dim=-1)
        null = F.linear(F.layer_norm(self.null, (self.width,), eps=eps), weight, bias).chunk(parts)
        attention = torch.empty(0)
        for i, step in enumerate(steps):
            query, attention = step.forward_kv(
                query, kv[2 * i], kv[2 * i + 1], (null[2 * i], null[2 * i + 1]), present
            )
        return query, attention

    def _attend_lean(
        self, query: torch.Tensor, tokens: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The steps for `model.lean_attention`: the cars and the null car normalised once without
        any step's scale and shift, one additive mask, and each step's attention without keys and values
        (`_AttentionStep.forward_lean`)."""
        steps = [step for step in self.attention if isinstance(step, _AttentionStep)]
        b, w = query.shape
        eps = steps[0].norm_cars.eps
        dtype = _product_dtype(tokens.device.type, self.null.dtype)
        null = F.layer_norm(self.null, (w,), eps=eps).expand(b, 1, w).to(tokens.dtype)
        normed = torch.cat([null, F.layer_norm(tokens, (w,), eps=eps)], dim=1).to(dtype)
        bias = _mask_bias(F.pad(present, (1, 0), value=True), dtype)
        attention = torch.empty(0)
        for step in steps:
            query, attention = step.forward_lean(query, normed, bias)
        return query, attention

    def folded_layers(self) -> list[nn.Linear]:
        """The linear layers whose weights the forward folds itself rather than through the layer: the FP16
        inference copy keeps them in FP32, so its folds round as training's do."""
        steps = [step for step in self.attention if isinstance(step, _AttentionStep)]
        if self.lean:
            return [layer for step in steps for layer in (step.q, step.k, step.v, step.out)]
        if self.fast:
            return [layer for step in steps for layer in (step.k, step.v)]
        return []

    def attention_labels(self) -> list[str]:
        """What each place of `attention_weights` is: the null car, then the teammate and opponent places."""
        _, _, teammates, opponents = self.layout
        return ["nothing"] + ["teammate"] * teammates + ["opponent"] * opponents

    def export_weights(self) -> list[torch.Tensor]:
        """The weights in the order the bot's file holds them (crates/nitro_core/src/policy.rs)."""
        weights: list[torch.Tensor] = []
        for mlp_ in (self.query, self.car):
            for index in (0, 2):
                linear = mlp_[index]
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
        weights += [self.sides, self.null]
        for step in self.attention:
            assert isinstance(step, _AttentionStep)
            for norm in (step.norm_query, step.norm_cars, step.norm_ff):
                weights += [norm.weight, norm.bias]
            for linear in (step.q, step.k, step.v, step.out, step.ff[0], step.ff[2]):
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
        return weights

    def fresh_parameters(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every parameter paired with the same one of a freshly built encoder."""
        fresh = EntityEncoder(
            self.layout, self.width, self.steps_, self.heads, self.skip, self.fast, self.lean
        )
        fresh = fresh.to(self.null.device)
        return [(mine, new.detach()) for mine, new in zip(self.parameters(), fresh.parameters(), strict=True)]


class _TalkStep(nn.Module):
    """One round of the entities talking: each token looks at every present token, itself included
    (pre-norm, masked multi-head self-attention), then a 2x feed-forward (EARL's full-attention size), each
    added back. The products follow autocast (FP16 under FP16 autocast); the softmax runs in FP32."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm, self.norm_ff = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.k, self.v, self.out = (nn.Linear(width, width) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The tokens after this round, and who looked at whom ([b, heads, tokens, tokens])."""
        b, t, w = tokens.shape
        h, d = self.heads, w // self.heads
        normed = self.norm(tokens)
        q, k, v = (layer(normed).view(b, t, h, d).transpose(1, 2) for layer in (self.q, self.k, self.v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)  # [b, h, t, t]
        weights = scores.masked_fill(~present[:, None, None, :], float("-inf")).softmax(-1)
        mixed = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b, t, w)
        tokens = tokens + self.out(mixed)
        result: torch.Tensor = tokens + self.ff(self.norm_ff(tokens))
        return result, weights

    def forward_lean(self, tokens: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` for `model.lean_attention`: q, k and v in one product, and `bias` the additive
        mask ([b, 1, tokens]) made once for every round."""
        b, t, w = tokens.shape
        h, d = self.heads, w // self.heads
        weight = torch.cat([self.q.weight, self.k.weight, self.v.weight])
        both = torch.cat([self.q.bias, self.k.bias, self.v.bias])
        q, k, v = F.linear(self.norm(tokens), weight, both).view(b, t, 3, h, d).permute(2, 0, 3, 1, 4)
        weights = ((q @ k.transpose(-1, -2)) / math.sqrt(d) + bias[:, None]).softmax(-1)
        mixed = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(b, t, w)
        tokens = tokens + self.out(mixed)
        result: torch.Tensor = tokens + self.ff(self.norm_ff(tokens))
        return result, weights


class TalkingEncoder(nn.Module):
    """`model.encoder = "attention_communication"`. Tokens: a learned null token, the ball, the own
    car and every other car (the own car through the other cars' shared embedding with zeros for the
    numbers a car block adds: exact for its place relative to itself, but with nitro_v2 also its five
    numbers about the ball, which the query and the ball token carry instead; then a kind embedding of
    its own: teammate, opponent, self). Each block, first the tokens talk
    (self-attention, so each can take in how the others relate), then the own part's query looks at them as
    `EntityEncoder` does. Empty places are masked; the order of the cars within a side does not matter, nor
    how many there are. Everything happens inside one bot's network; nothing passes between bots."""

    def __init__(
        self,
        layout: Layout,
        parts: Parts,
        width: int,
        blocks: int,
        heads: int,
        skip: bool = False,
        lean: bool = False,
    ) -> None:
        super().__init__()
        self.layout, self.parts, self.width, self.blocks, self.heads, self.skip = (
            layout, parts, width, blocks, heads, skip
        )  # fmt: skip
        self.lean = lean  # `model.lean_attention`
        self.outputs = width + (layout[0] if skip else 0)
        own, block, teammates, opponents = layout
        ball, _, car = parts
        if car > block - 1:
            raise ValueError(f"an own car of {car} numbers does not fit a car block of {block - 1}")
        self.query = nn.Sequential(nn.Linear(own, width), nn.ReLU(), nn.Linear(width, width))
        self.car = nn.Sequential(nn.Linear(block - 1, width), nn.ReLU(), nn.Linear(width, width))
        self.ball = nn.Sequential(nn.Linear(ball, width), nn.ReLU(), nn.Linear(width, width))
        self.kinds = nn.Parameter(torch.randn(3, width) * 0.02)  # teammate, opponent, the own car
        self.null = nn.Parameter(torch.zeros(width))
        self.talk = nn.ModuleList(_TalkStep(width, heads) for _ in range(blocks))
        self.listen = nn.ModuleList(_AttentionStep(width, heads) for _ in range(blocks))
        self.register_buffer("kind", torch.tensor([0] * teammates + [1] * opponents), persistent=False)
        self.keep_weights = False
        self.attention_weights: torch.Tensor | None = None  # the query's last look ([b, heads, tokens])
        self.talk_weights: torch.Tensor | None = None  # the last round of talk ([b, heads, tokens, tokens])

    def forward(self, observations: torch.Tensor, places: Places | None = None) -> torch.Tensor:
        """`places`: as `EntityEncoder.forward`'s."""
        own, block = self.layout[:2]
        teammates, opponents = places or _car_places(self.layout, observations.shape[-1])
        ball_size, car_at, car = self.parts
        lead = observations.shape[:-1]
        obs = observations.reshape(-1, observations.shape[-1])
        b = obs.shape[0]
        cars = obs[:, own:].reshape(b, teammates + opponents, block)
        kind = _sides(self.kind, self.layout, (teammates, opponents))  # type: ignore[arg-type]
        me_numbers = F.pad(obs[:, car_at : car_at + car], (0, block - 1 - car))
        if self.lean:  # the own car and the others through the car embedding in one call
            kinds = torch.cat([kind.new_full((1,), 2), kind])
            others = self.car(torch.cat([me_numbers[:, None], cars[..., :-1]], dim=1)) + self.kinds[kinds]
            me = others[:, 0]
            others = others[:, 1:]
        else:
            others = self.car(cars[..., :-1]) + self.kinds[kind]
            me = self.car(me_numbers) + self.kinds[2]
        ball = self.ball(obs[:, :ball_size])
        tokens = torch.cat(
            [
                self.null.expand(b, 1, -1).to(others.dtype),
                ball[:, None].to(others.dtype),
                me[:, None].to(others.dtype),
                others,
            ],
            dim=1,
        )
        present = torch.cat(
            [torch.ones(b, 3, dtype=torch.bool, device=obs.device), cars[..., -1] > 0.5], dim=1
        )
        query = self.query(obs[:, :own]).unsqueeze(1)
        if self.lean:
            query, looked, talked = self._blocks_lean(query[:, 0], tokens, present)
        else:
            for talk, listen in zip(self.talk, self.listen, strict=True):
                tokens, talked = talk(tokens, present)
                query, looked = listen(query, tokens, present)
        if self.keep_weights:
            self.attention_weights = looked.detach().float().reshape(*lead, *looked.shape[1:])
            self.talk_weights = talked.detach().float().reshape(*lead, *talked.shape[1:])
        encoded = query.reshape(b, self.width)
        if self.skip:
            encoded = torch.cat([obs[:, :own].to(encoded.dtype), encoded], dim=1)
        result: torch.Tensor = encoded.reshape(*lead, self.outputs)
        return result

    def _blocks_lean(
        self, query: torch.Tensor, tokens: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The blocks for `model.lean_attention`: one additive mask for every round, the talk's q, k
        and v in one product, and each look without keys and values (`_AttentionStep.forward_lean`).
        Returns the query [b, w], its last look and the last round of talk."""
        dtype = _product_dtype(tokens.device.type, self.null.dtype)
        bias = _mask_bias(present, dtype)
        looked = talked = torch.empty(0)
        for talk, listen in zip(self.talk, self.listen, strict=True):
            assert isinstance(talk, _TalkStep) and isinstance(listen, _AttentionStep)
            tokens, talked = talk.forward_lean(tokens, bias)
            normed = F.layer_norm(tokens, (self.width,), eps=listen.norm_cars.eps).to(dtype)
            query, looked = listen.forward_lean(query, normed, bias)
        return query, looked, talked

    def folded_layers(self) -> list[nn.Linear]:
        """As `EntityEncoder.folded_layers`: with lean_attention, each look's q, k, v and output layers."""
        if not self.lean:
            return []
        return [
            layer
            for listen in self.listen
            if isinstance(listen, _AttentionStep)
            for layer in (listen.q, listen.k, listen.v, listen.out)
        ]

    def attention_labels(self) -> list[str]:
        """What each token is: the null token, the ball, the own car, the teammate and opponent places."""
        _, _, teammates, opponents = self.layout
        return ["nothing", "ball", "itself"] + ["teammate"] * teammates + ["opponent"] * opponents

    def export_weights(self) -> list[torch.Tensor]:
        """The weights in the order the bot's file holds them (crates/nitro_core/src/policy.rs, format 4)."""
        weights: list[torch.Tensor] = []
        for mlp_ in (self.query, self.car, self.ball):
            for index in (0, 2):
                linear = mlp_[index]
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
        weights += [self.kinds, self.null]
        for talk, listen in zip(self.talk, self.listen, strict=True):
            assert isinstance(talk, _TalkStep) and isinstance(listen, _AttentionStep)
            weights += [talk.norm.weight, talk.norm.bias]
            for linear in (talk.q, talk.k, talk.v, talk.out):
                weights += [linear.weight, linear.bias]
            weights += [talk.norm_ff.weight, talk.norm_ff.bias]
            for linear in (talk.ff[0], talk.ff[2]):
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
            for norm in (listen.norm_query, listen.norm_cars, listen.norm_ff):
                weights += [norm.weight, norm.bias]
            for linear in (listen.q, listen.k, listen.v, listen.out, listen.ff[0], listen.ff[2]):
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
        return weights

    def fresh_parameters(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every parameter paired with the same one of a freshly built encoder."""
        fresh = TalkingEncoder(
            self.layout, self.parts, self.width, self.blocks, self.heads, self.skip, self.lean
        )
        fresh = fresh.to(self.null.device)
        return [(mine, new.detach()) for mine, new in zip(self.parameters(), fresh.parameters(), strict=True)]


class _TransformerBlock(_Look):
    """One pre-norm transformer block: masked multi-head self-attention over every token (one q, k, v layer;
    PyTorch's fused attention), then a ReLU feed-forward, each added back."""

    def __init__(self, width: int, heads: int, ff: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm, self.norm_ff = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.ReLU(), nn.Linear(ff, width))

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor, by_hand: bool, small: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The tokens after this block; `mask` ([b, 1, 1, tokens]) is True where a token may be looked at.
        `by_hand`: the same attention as matrix products with the softmax in FP32, which also gives its
        weights ([b, heads, tokens, tokens]); otherwise `scaled_dot_product_attention`, whose fused kernels
        keep their own softmax in FP32, or with `small` (`model.small_attention`) NitroLearn's own
        routine for few tokens where it can run (its sums and softmax in FP32 too)."""
        b, t, w = tokens.shape
        h = self.heads
        qkv = self.qkv(self.norm(tokens))
        if small and not by_hand and small_attention.usable(qkv, h):
            mixed = small_attention.attention(qkv, mask.reshape(b, t), h)
            tokens = tokens + self.out(mixed)
            return tokens + self.ff(self.norm_ff(tokens)), None
        q, k, v = qkv.view(b, t, 3, h, w // h).permute(2, 0, 3, 1, 4)
        weights = None
        if by_hand:
            with torch.autocast(tokens.device.type, enabled=False):
                scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(w // h)
                weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            mixed = weights.to(v.dtype) @ v
        else:
            mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        tokens = tokens + self.out(mixed.transpose(1, 2).reshape(b, t, w))
        result: torch.Tensor = tokens + self.ff(self.norm_ff(tokens))
        return result, weights

    def forward_first(self, tokens: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` for the first token alone, as the last block of `model.lean_attention`, whose
        other tokens nobody reads: it looks at every token without making their keys and values
        (`_look_once`), and only it goes on through the output layer and the feed-forward. `bias` is the
        additive mask ([b, 1, tokens]). Returns the first token [b, w] and its
        attention [b, heads, tokens]."""
        w = tokens.shape[-1]
        folded, out_weight = self.fold()
        normed = F.layer_norm(tokens, (w,), eps=self.norm.eps)
        query = normed[:, 0]
        mixed, weights = _look_once(query, normed.to(bias.dtype), bias, folded, out_weight, self.heads)
        first = tokens[:, 0] + mixed
        result: torch.Tensor = first + self.ff(self.norm_ff(first))
        return result, weights

    def fold(self) -> tuple[Folded, torch.Tensor]:
        w = self.out.weight.shape[0]
        wq, wk, wv = self.qkv.weight.view(3, w, w)
        bq, _, bv = self.qkv.bias.view(3, w)
        folded = _fold_look(
            self.norm, (wq, bq), self.norm, wk, (wv, bv), (self.out.weight, self.out.bias), self.heads
        )
        return folded, self.out.weight


class TransformerEncoder(nn.Module):
    """`model.encoder = "transformer"`. Every number of the observation goes into one token: the own car (its
    car numbers through the other cars' shared embedding, with zeros where a car block holds numbers about
    the bot itself, plus the rest of the own part through an embedding of its own, plus the own-car kind),
    the ball, every other car (the shared embedding plus a teammate or opponent kind) and, with `pads`, each
    boost pad (its number through one embedding plus its place's). With `summary`, a learned summary token
    comes first. Then `layers` pre-norm transformer blocks over every token (empty places masked), and the
    output is the first token (the summary or the own car) after a final norm, with `skip` the own part in
    front of it. The order of the cars within a side does not matter, nor how many there are. The residual
    stream stays FP32 under FP16 autocast; the layers run in FP16."""

    def __init__(
        self,
        layout: Layout,
        parts: Parts,
        width: int,
        layers: int,
        heads: int,
        ff: int,
        pads: bool = False,
        summary: bool = False,
        skip: bool = False,
        lean: bool = False,
        small: bool = False,
    ) -> None:
        super().__init__()
        self.layout, self.parts, self.width, self.heads, self.skip = layout, parts, width, heads, skip
        self.layers, self.ff_width, self.pads = layers, ff, pads
        self.lean = lean  # `model.lean_attention`
        self.small = small  # `model.small_attention`
        self.outputs = width + (layout[0] if skip else 0)
        own, block, teammates, opponents = layout
        ball, car_at, car = parts
        if car > block - 1:
            raise ValueError(f"an own car of {car} numbers does not fit a car block of {block - 1}")
        pad_numbers = range(ball, ball + _PADS) if pads else range(0)
        rest = [i for i in range(ball, own) if not car_at <= i < car_at + car and i not in pad_numbers]
        self.register_buffer("rest", torch.tensor(rest), persistent=False)

        def embedding(inputs: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width))

        self.car = embedding(block - 1)
        self.ball = embedding(ball)
        self.own_rest = embedding(len(rest))
        if pads:
            self.pad = embedding(1)
            self.pad_places = nn.Parameter(torch.randn(_PADS, width) * 0.02)
        self.kinds = nn.Parameter(torch.randn(3, width) * 0.02)  # teammate, opponent, the own car
        self.summary = nn.Parameter(torch.zeros(width)) if summary else None
        self.blocks = nn.ModuleList(_TransformerBlock(width, heads, ff) for _ in range(layers))
        self.norm_out = nn.LayerNorm(width)
        self.register_buffer("kind", torch.tensor([0] * teammates + [1] * opponents), persistent=False)
        self.keep_weights = False
        # With keep_weights (`nitro watch`): what the read-out token looked at in the last block ([b, heads,
        # places], the pads' shares added up into one place).
        self.attention_weights: torch.Tensor | None = None

    def forward(self, observations: torch.Tensor, places: Places | None = None) -> torch.Tensor:
        """`places`: as `EntityEncoder.forward`'s."""
        own, block = self.layout[:2]
        teammates, opponents = places or _car_places(self.layout, observations.shape[-1])
        ball, car_at, car = self.parts
        lead = observations.shape[:-1]
        obs = observations.reshape(-1, observations.shape[-1])
        b = obs.shape[0]
        cars = obs[:, own:].reshape(b, teammates + opponents, block)
        me = F.pad(obs[:, car_at : car_at + car], (0, block - 1 - car))[:, None]
        embedded = self.car(torch.cat([me, cars[..., :-1]], dim=1)).float()  # the own car first
        kind = _sides(self.kind, self.layout, (teammates, opponents))  # type: ignore[arg-type]
        rest: torch.Tensor = self.rest  # type: ignore[assignment]
        mine = embedded[:, :1] + self.own_rest(obs[:, rest]).float()[:, None] + self.kinds[2]
        tokens = [mine, self.ball(obs[:, :ball]).float()[:, None], embedded[:, 1:] + self.kinds[kind]]
        present = [torch.ones(b, 2, dtype=torch.bool, device=obs.device), cars[..., -1] > 0.5]
        if self.summary is not None:
            tokens.insert(0, self.summary.expand(b, 1, -1))
            present.insert(0, torch.ones(b, 1, dtype=torch.bool, device=obs.device))
        if self.pads:
            tokens.append(self.pad(obs[:, ball : ball + _PADS, None]).float() + self.pad_places)
            present.append(torch.ones(b, _PADS, dtype=torch.bool, device=obs.device))
        x = torch.cat(tokens, dim=1)
        present_all = torch.cat(present, dim=1)
        mask = present_all[:, None, None, :]
        blocks = [block for block in self.blocks if isinstance(block, _TransformerBlock)]
        # The weights only when asked (the slower path), and the plain products in deterministic mode, whose
        # fused GPU kernels are not all repeatable. With lean_attention the last block gives the weights.
        by_hand = (self.keep_weights and not self.lean) or torch.are_deterministic_algorithms_enabled()
        weights = None
        for transformer_block in blocks[:-1] if self.lean else blocks:
            x, weights = transformer_block(x, mask, by_hand, self.small)
        looked: torch.Tensor | None
        if self.lean:
            bias = _mask_bias(present_all, _product_dtype(x.device.type, self.kinds.dtype))
            first, looked = blocks[-1].forward_first(x, bias)
        else:
            first = x[:, 0]
            looked = weights[:, :, 0] if weights is not None else None
        if self.keep_weights and looked is not None:
            if self.pads:
                looked = torch.cat([looked[..., :-_PADS], looked[..., -_PADS:].sum(-1, keepdim=True)], -1)
            self.attention_weights = looked.detach().float().reshape(*lead, *looked.shape[1:])
        encoded = self.norm_out(first)
        if self.skip:
            encoded = torch.cat([obs[:, :own].to(encoded.dtype), encoded], dim=1)
        result: torch.Tensor = encoded.reshape(*lead, self.outputs)
        return result

    def folded_layers(self) -> list[nn.Linear]:
        """As `EntityEncoder.folded_layers`: with lean_attention, the last block's q, k, v and out layers."""
        last = self.blocks[-1]
        return [last.qkv, last.out] if self.lean and isinstance(last, _TransformerBlock) else []

    def attention_labels(self) -> list[str]:
        """What each place of `attention_weights` is: the summary token ("nothing"), the own car, the ball,
        the teammate and opponent places, the boost pads."""
        _, _, teammates, opponents = self.layout
        return [
            *(["nothing"] if self.summary is not None else []),
            "itself",
            "ball",
            *["teammate"] * teammates,
            *["opponent"] * opponents,
            *(["pads"] if self.pads else []),
        ]

    def export_weights(self) -> list[torch.Tensor]:
        """The weights in the order the bot's file holds them (crates/nitro_core/src/policy.rs, format 6)."""
        weights: list[torch.Tensor] = []
        embeddings = [self.car, self.ball, self.own_rest, *([self.pad] if self.pads else [])]
        for embedding in embeddings:
            for index in (0, 2):
                linear = embedding[index]
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
        if self.pads:
            weights.append(self.pad_places)
        weights.append(self.kinds)
        if self.summary is not None:
            weights.append(self.summary)
        for transformer_block in self.blocks:
            assert isinstance(transformer_block, _TransformerBlock)
            linears = (transformer_block.ff[0], transformer_block.ff[2])
            weights += [transformer_block.norm.weight, transformer_block.norm.bias]
            for linear in (transformer_block.qkv, transformer_block.out):
                weights += [linear.weight, linear.bias]
            weights += [transformer_block.norm_ff.weight, transformer_block.norm_ff.bias]
            for linear in linears:
                assert isinstance(linear, nn.Linear)
                weights += [linear.weight, linear.bias]
        return [*weights, self.norm_out.weight, self.norm_out.bias]

    def fresh_parameters(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every parameter paired with the same one of a freshly built encoder."""
        fresh = TransformerEncoder(
            self.layout,
            self.parts,
            self.width,
            self.layers,
            self.heads,
            self.ff_width,
            self.pads,
            self.summary is not None,
            self.skip,
            self.lean,
            self.small,
        ).to(self.kinds.device)
        return [(mine, new.detach()) for mine, new in zip(self.parameters(), fresh.parameters(), strict=True)]


# The encoders that read the other cars as a set (`nitro.model`'s attention encoders).
ENCODERS = (EntityEncoder, TalkingEncoder, TransformerEncoder)

# Sizes in nitro_core::obs: the ball's part, the car's part, the pads and the previous action.
_BALL = {"native": 9, "nitro_v2": 20}
_CAR = {"native": 23, "nitro_v2": 26}
_PADS = 34
_PADS_AND_ACTION = _PADS + 8


def entity_parts(observation: str) -> Parts:
    """Where the ball and the own car sit in `observation`'s own part (the ball first, then the pads and the
    previous action, then the own car; nitro_core::obs). Native and nitro_v2 only, as padding is."""
    if observation not in _BALL:
        raise ValueError(
            f"an attention encoder needs the native or nitro_v2 observation, not {observation!r}"
        )
    ball = _BALL[observation]
    return ball, ball + _PADS_AND_ACTION, _CAR[observation]


def attention_encoder(model: ActorCritic) -> nn.Module | None:
    """The policy's attention encoder (the shared trunk's, when there is one), or None for a plain network.
    It has `keep_weights`, `attention_weights` ([..., heads, places]) and `attention_labels()` (`nitro
    watch`)."""
    path = model.shared if len(model.shared) else model.policy
    first = path[0] if len(path) else None
    return first if first is not None and hasattr(first, "attention_labels") else None


def observation_room(config: NitroConfig) -> int:
    """Players a side `config`'s observation has places for (0 = none, unpadded): env.obs_team_size,
    or with an attention encoder and 0 there, the run's own team size, since such an encoder reads each car
    from a place of its own with its "present" flag."""
    env = config.env
    return env.obs_team_size or (env.team_size if config.model.uses_attention else 0)


def observation_layout(config: NitroConfig) -> Layout:
    """The layout of the observation `config`'s games produce, from its observation settings alone (no
    games are built, and the rest of the settings is not checked: a save rebuilds wherever it is used)."""
    from nitro import _core

    env = config.env
    _, own, block, teammates, opponents, room = _core.observation_layout_of(
        env.observation, env.team_size, env.opponents, observation_room(config)
    )
    if room == 0:
        raise ValueError("an attention encoder needs places for the cars (env.obs_team_size)")
    return own, block, teammates, opponents


def play_settings(config: NitroConfig, team_size: int) -> NitroConfig:
    """`config` for games of `team_size` a side, to watch or evaluate (training stays at its size): a bot
    with an attention encoder reads any number of cars, so its observation gets room for the bigger of its
    own room and `team_size`; any other bot keeps its observation (`dashboard.bots.watch_sizes` says which
    sizes it can play)."""
    env = dataclasses.replace(config.env.one_size(), team_size=team_size)
    if config.model.uses_attention:
        env = dataclasses.replace(env, obs_team_size=play_room(config, team_size))
    return dataclasses.replace(config, env=env)


def play_room(config: NitroConfig, team_size: int) -> int:
    """The room `config`'s networks read games of `team_size` a side with: an attention encoder's room grows
    to the game's (it reads any number of cars); any other network keeps its own."""
    room = observation_room(config)
    return max(room, team_size) if config.model.uses_attention else room


def build_networks(config: NitroConfig, obs_size: int, num_actions: int) -> ActorCritic:
    """The networks `config` describes, for every place that builds them (training, the self-play pool,
    evaluation, export, pretraining, the viewer), so none can build another kind than a save holds."""
    layout = observation_layout(config) if config.model.uses_attention else None
    reads_ball = config.model.encoder in ("attention_communication", "transformer")
    parts = entity_parts(config.env.observation) if reads_ball else None
    return ActorCritic(obs_size, num_actions, config.model, layout, parts)


# How many numbers each value head gives (`model.value_heads`).
HEAD_OUTPUTS = {"main": 1, "goal": 1, "vdag1": 1, "vdag2": 1}


def parameter_counts(config: NitroConfig) -> dict[str, int]:
    """The parameters of the networks `config` describes (built on PyTorch's meta device, so no memory and
    no time): "total", per network ("shared", "policy", "value"), "playing" (what the bot runs: the shared
    trunk and the policy) and, with value heads, "critic_trunk" and "head_<name>" each."""
    from nitro import _core

    env = config.env
    obs_size, *_ = _core.observation_layout_of(
        env.observation, env.team_size, env.opponents, observation_room(config)
    )
    actions = _core.num_actions()
    with torch.device("meta"):
        model = build_networks(config, obs_size, actions)

    def count(params: Iterable[nn.Parameter]) -> int:
        return sum(p.numel() for p in params)

    groups = model.parameter_groups()
    counts = {"total": count(model.parameters())} | {name: count(group) for name, group in groups.items()}
    counts["playing"] = counts["policy"] + counts.get("shared", 0)
    if len(model.value_heads):
        counts["critic_trunk"] = count(model.value.parameters())
        counts |= {f"head_{name}": count(head.parameters()) for name, head in model.value_heads.items()}
    return counts


class ActorCritic(nn.Module):
    """A policy (action logits) and a value network, optionally on top of a shared trunk.

    The value head predicts *normalised* values; PopArt (nitro.ppo) holds the statistics. Both output
    layers run in FP32 even under FP16 autocast, so logits, log-probabilities and values keep full
    precision. With `model.value_heads` or `model.value_head_layers` `value` is a critic trunk
    without an output layer and `value_heads` holds one network per head on top of it ("main" is the value
    PPO learns); otherwise `value_heads` is empty and `value` ends in its own output layer.
    """

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        config: ModelConfig,
        layout: Layout | None = None,
        parts: Parts | None = None,
    ) -> None:
        """`layout`: the padded observation's (`observation_layout`), needed with an attention encoder;
        `parts`: where its ball and own car are (`entity_parts`), needed with 'attention_communication'."""
        super().__init__()
        self.config = config
        self.obs_size, self.num_actions, self.layout, self.parts = obs_size, num_actions, layout, parts
        apart = config.value_heads_apart
        value_outputs = None if apart else 1
        if config.uses_attention:
            if layout is None:
                raise ValueError("an attention encoder needs the observation's layout (build_networks)")
            own, block, teammates, opponents = layout
            if own + (teammates + opponents) * block != obs_size:
                raise ValueError(f"the layout {layout} does not add up to {obs_size} observation numbers")
            width = config.attention_width

            if config.encoder in ("attention_communication", "transformer") and parts is None:
                raise ValueError(f"{config.encoder} needs where the ball and own car are (build_networks)")

            def encoder() -> EntityEncoder | TalkingEncoder | TransformerEncoder:
                if config.encoder == "transformer":
                    assert parts is not None
                    return TransformerEncoder(
                        layout,
                        parts,
                        config.transformer_width,
                        config.transformer_layers,
                        config.transformer_heads,
                        config.transformer_ff,
                        config.transformer_pads,
                        config.transformer_readout == "summary",
                        config.attention_skip,
                        config.lean_attention,
                        config.small_attention,
                    )
                if config.encoder == "attention_communication":
                    assert parts is not None
                    return TalkingEncoder(
                        layout,
                        parts,
                        width,
                        config.attention_blocks,
                        config.attention_heads,
                        config.attention_skip,
                        config.lean_attention,
                    )
                return EntityEncoder(
                    layout,
                    width,
                    config.attention_blocks,
                    config.attention_heads,
                    config.attention_skip,
                    config.attention_fast,
                    config.lean_attention,
                )

            if config.shared:
                # One encoder and the shared layers serve both networks (a shared head).
                first = encoder()
                self.shared = nn.Sequential(first, *mlp(first.outputs, config.shared, None, config))
                self.policy = mlp(config.shared[-1], config.policy, num_actions, config)
                self.value = mlp(config.shared[-1], config.value, value_outputs, config)
                features = config.shared[-1]
            else:

                def network(hidden: Sequence[int], outputs: int | None) -> nn.Sequential:
                    first = encoder()
                    return nn.Sequential(first, *mlp(first.outputs, hidden, outputs, config))

                self.shared = nn.Sequential()
                self.policy = network(config.policy, num_actions)
                self.value = network(config.value, value_outputs)
                encoded = self.value[0]
                assert isinstance(encoded, ENCODERS)
                features = encoded.outputs
        else:
            self.shared = mlp(obs_size, config.shared, None, config)
            features = config.shared[-1] if config.shared else obs_size
            self.policy = mlp(features, config.policy, num_actions, config)
            self.value = mlp(features, config.value, value_outputs, config)
        # The value heads on the critic trunk; empty, so nothing in a save, without them.
        self.value_heads = nn.ModuleDict()
        if apart:
            trunk = config.value[-1] if config.value else features
            for name in config.value_heads:
                self.value_heads[name] = mlp(trunk, config.value_head_layers, HEAD_OUTPUTS[name], config)
        if config.init == "orthogonal":
            # Small policy outputs start the policy near uniform, so a new bot explores evenly.
            outputs = [(self.policy, 0.01), *((head, 1.0) for head in self._value_networks())]
            for head, gain in outputs:
                nn.init.orthogonal_(_output(head).weight, gain)
                nn.init.zeros_(_output(head).bias)

    def _value_networks(self) -> list[nn.Sequential]:
        """The networks that end in a value output layer: `value`, or each value head."""
        heads = [head for head in self.value_heads.values() if isinstance(head, nn.Sequential)]
        return heads if len(self.value_heads) else [self.value]

    @property
    def value_output(self) -> nn.Linear:
        """The main value's output layer (PopArt's)."""
        main = self.value_heads["main"] if len(self.value_heads) else self.value
        assert isinstance(main, nn.Sequential)
        return _output(main)

    def output_layers(self) -> list[nn.Linear]:
        """Every output layer (the policy's and each value's), which run in FP32."""
        return [_output(self.policy), *(_output(head) for head in self._value_networks())]

    def fresh_copy(self) -> ActorCritic:
        """Freshly initialised networks of exactly this kind and shape (a copy then loads the weights)."""
        return ActorCritic(self.obs_size, self.num_actions, self.config, self.layout, self.parts)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action logits, normalised value), both FP32."""
        features = self.shared(obs)
        return self.logits_from(features), self.value_from(features)

    def logits_from(self, features: torch.Tensor) -> torch.Tensor:
        """The policy's logits from the shared trunk's output (`shared(obs)`; the observation without one)."""
        return _run_head(self.policy, features)

    def value_from(self, features: torch.Tensor) -> torch.Tensor:
        """The normalised value from the shared trunk's output, as `logits_from` (the main head's)."""
        if len(self.value_heads):
            main = self.value_heads["main"]
            assert isinstance(main, nn.Sequential)
            return _run_head(main, self.value(features)).squeeze(-1)
        return _run_head(self.value, features).squeeze(-1)

    def head_values_from(
        self, features: torch.Tensor, names: Collection[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Every value head's output from the shared trunk's output, the critic trunk run once: name ->
        [...] for a head of one number ("main" is `value_from`'s), [..., n] for a head of n; FP32 output
        layers, the rest in the caller's precision. Without heads, {"main": value_from(features)}. `names`:
        only those heads (each head's numbers as with every head, to the bit)."""
        if not len(self.value_heads):
            return {"main": self.value_from(features)}
        trunk = self.value(features)
        out = {}
        for name, head in self.value_heads.items():
            if names is not None and name not in names:
                continue
            assert isinstance(head, nn.Sequential)
            values = _run_head(head, trunk)
            out[name] = values.squeeze(-1) if values.shape[-1] == 1 else values
        return out

    def head_values(self, obs: torch.Tensor, names: Collection[str] | None = None) -> dict[str, torch.Tensor]:
        """`head_values_from` for observations."""
        return self.head_values_from(self.shared(obs), names)

    def forward_heads(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The action logits and every value head's output (`head_values_from`), the trunk run once."""
        features = self.shared(obs)
        return self.logits_from(features), self.head_values_from(features)

    def normalised_value(self, obs: torch.Tensor, hold_trunk: bool = False) -> torch.Tensor:
        """`hold_trunk`: the shared trunk gives its features without gradients, so only the value network's
        own layers learn (a value warm-up with a shared trunk)."""
        if hold_trunk and len(self.shared):
            with torch.no_grad():
                features = self.shared(obs)
            return self.value_from(features)
        return self.value_from(self.shared(obs))

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Action logits only (a frozen opponent needs no value)."""
        return self.logits_from(self.shared(obs))

    def policy_state_dict(self) -> dict[str, torch.Tensor]:
        """The weights a frozen copy needs: the policy and any shared trunk, never the value network."""
        return {k: v for k, v in self.state_dict().items() if k.startswith(("shared.", "policy."))}

    def load_policy_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Copies policy (and trunk) weights in place, so tensors captured in a CUDA graph stay valid."""
        own = self.state_dict()
        missing = [k for k in own if k.startswith(("shared.", "policy.")) and k not in state]
        if missing:
            raise ValueError(f"the saved policy lacks {missing[0]}; its network shape differs")
        with torch.no_grad():
            for key, value in state.items():
                own[key].copy_(value)

    def fresh_parameters(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Every parameter paired with a freshly initialised tensor of the same shape, drawn as a new network
        would be; `reset_value_head` draws one value head's weights the same way."""
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        value_gain = None if len(self.value_heads) else 1.0  # a critic trunk has no output layer
        heads = [(self.shared, None), (self.policy, 0.01), (self.value, value_gain)]
        heads += [(head, 1.0) for head in self.value_heads.values() if isinstance(head, nn.Sequential)]
        for head, output_gain in heads:
            pairs += self._fresh_pairs(head, output_gain)
        return pairs

    def _fresh_pairs(
        self, head: nn.Sequential, output_gain: float | None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for module in _flat_layers(head):
            if isinstance(module, ENCODERS):
                pairs += module.fresh_parameters()
            elif isinstance(module, nn.Linear):
                weight = torch.empty_like(module.weight)
                bias = torch.empty_like(module.bias)
                gain = math.sqrt(2)
                if output_gain is not None and module is _output(head):
                    gain = output_gain
                if self.config.init == "orthogonal":
                    nn.init.orthogonal_(weight, gain)
                    nn.init.zeros_(bias)
                else:  # PyTorch's own Linear initialisation
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                    bound = 1 / math.sqrt(module.in_features)
                    nn.init.uniform_(bias, -bound, bound)
                pairs += [(module.weight, weight), (module.bias, bias)]
            elif isinstance(module, nn.LayerNorm):
                pairs.append((module.weight, torch.ones_like(module.weight)))
                pairs.append((module.bias, torch.zeros_like(module.bias)))
        return pairs

    @torch.no_grad()
    def reset_value_head(self, name: str) -> list[nn.Parameter]:
        """Draws the value head `name`'s weights afresh, as `fresh_parameters` would (the twin critics);
        returns its parameters."""
        head = self.value_heads[name]
        assert isinstance(head, nn.Sequential)
        for parameter, fresh in self._fresh_pairs(head, 1.0):
            parameter.copy_(fresh)
        return list(head.parameters())

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameters per network, for separate learning rates and gradient clipping; "value" holds the
        critic trunk and every value head."""
        value = [*self.value.parameters(), *self.value_heads.parameters()]
        groups = {"policy": list(self.policy.parameters()), "value": value}
        if len(self.shared) > 0:
            groups["shared"] = list(self.shared.parameters())
        return groups

    def muon_parameters(self) -> list[nn.Parameter]:
        """The weight matrices `ppo.optimizer = "muon"` moves with Muon: every hidden linear layer's of
        the shared trunk, the policy, the critic trunk and the value heads, except a layer that reads the
        observation itself (the first of a network with nothing in front of it), the output layers and the
        attention encoders (their weights stay with Adam)."""
        shared = len(self.shared) > 0
        stacks = [
            (self.shared, None, True),
            (self.policy, _output(self.policy), not shared),
            (self.value, None if len(self.value_heads) else _output(self.value), not shared),
        ]
        trunk = len(self.value) > 0
        stacks += [
            (head, _output(head), not shared and not trunk)
            for head in self.value_heads.values()
            if isinstance(head, nn.Sequential)
        ]
        found: list[nn.Parameter] = []
        for stack, output, reads_obs in stacks:
            layers = _flat_layers(stack)
            linears = [m for m in layers if isinstance(m, nn.Linear) and m is not output]
            if reads_obs and layers and isinstance(layers[0], nn.Linear):
                linears = linears[1:]  # it reads the observation
            found += [p for linear in linears for p in (linear.weight,) if isinstance(p, nn.Parameter)]
        return found

    def scale_free_pairs(self) -> list[tuple[str, nn.Parameter, nn.Parameter]]:
        """(network, weight, bias) of every linear layer a LayerNorm follows (plain or in a residual branch;
        each network's first layer too) in the shared trunk, the policy, the critic and the value heads
        ("value"). Scaling such a pair leaves the network's output unchanged (to the LayerNorm's eps):
        ppo.weight_decay and the turn/growth metrics. Not output layers or attention encoders."""
        stacks = [("shared", self.shared), ("policy", self.policy), ("value", self.value)]
        stacks += [("value", head) for head in self.value_heads.values() if isinstance(head, nn.Sequential)]
        found: list[tuple[str, nn.Parameter, nn.Parameter]] = []
        for name, stack in stacks:
            layers = _flat_layers(stack)
            for layer, after in itertools.pairwise(layers):
                if isinstance(layer, nn.Linear) and isinstance(after, nn.LayerNorm):
                    weight, bias = layer.weight, layer.bias
                    if isinstance(weight, nn.Parameter) and isinstance(bias, nn.Parameter):
                        found.append((name, weight, bias))
        return found


def fp16_inference_copy(model: ActorCritic) -> ActorCritic:
    """A copy of `model` for choosing actions under FP16 autocast that holds in FP16 the weights autocast
    would cast to FP16 on every pass anyway: its linear layers (not the FP32 output layers) and the FP16
    LayerNorms' weights. Its outputs are the model's to the bit; collection then replays no weight casts
    in each step's graph. `copy_inference_weights` brings it up to date with the model."""
    copy = model.fresh_copy().to(next(model.parameters()).device)
    copy.load_state_dict(model.state_dict())
    keep = {id(layer) for layer in copy.output_layers()}
    for module in copy.modules():
        if isinstance(module, nn.MultiheadAttention):  # its own maths, not autocast's linear
            keep.update(id(inner) for inner in module.modules())
        if isinstance(module, ENCODERS):  # weights its fast paths fold themselves
            keep.update(id(layer) for layer in module.folded_layers())
    for module in copy.modules():
        if isinstance(module, nn.Linear) and id(module) not in keep:
            module.half()
    copy.eval()
    for parameter in copy.parameters():
        parameter.requires_grad_(False)
    return copy


@dataclasses.dataclass(frozen=True)
class _WeightPairs:
    """What `copy_inference_weights` copies from `model` into one copy: every tensor of both, where their
    memory was when listed, and the (targets, sources) lists, one pair of lists per pair of dtypes."""

    model: weakref.ref[ActorCritic]
    tensors: tuple[torch.Tensor, ...]
    places: tuple[int, ...]
    lists: tuple[tuple[list[torch.Tensor], list[torch.Tensor]], ...]


# Per copy. Listing both networks' tensors (two state_dict walks) every collection took about as long as the
# copies themselves.
_WEIGHT_PAIRS: weakref.WeakKeyDictionary[ActorCritic, _WeightPairs] = weakref.WeakKeyDictionary()
# CUDA's one-call copy (FP32 into FP16) gives the per-tensor copy's bits: tests/test_collection_ends.py's
# card check, passed on a card. False copies tensor by tensor; on the processor one call
# is that loop.
_ONE_CALL_ON_CARDS = True


def _weight_pairs(copy: ActorCritic, model: ActorCritic) -> _WeightPairs:
    lists: dict[tuple[torch.dtype, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    tensors: list[torch.Tensor] = []
    targets, sources = copy.state_dict(keep_vars=True), model.state_dict(keep_vars=True)
    for (name, target), source in zip(targets.items(), sources.values(), strict=True):
        assert target.shape == source.shape, name
        mine, theirs = lists.setdefault((target.dtype, source.dtype), ([], []))
        mine.append(target)
        theirs.append(source)
        tensors += [target, source]
    places = tuple(t.data_ptr() for t in tensors)
    return _WeightPairs(weakref.ref(model), tuple(tensors), places, tuple(lists.values()))


@torch.no_grad()
def copy_inference_weights(copy: ActorCritic, model: ActorCritic) -> None:
    """Copies `model`'s weights into its `fp16_inference_copy` in place (rounded as autocast rounds them),
    so the tensors the collection graphs captured stay the same. From lists kept from the first call and
    listed again when either network's tensors moved; loads write into the same tensors.
    One copy call per pair of dtypes on the processor (and on a card with `_ONE_CALL_ON_CARDS`)."""
    pairs = _WEIGHT_PAIRS.get(copy)
    if (
        pairs is None
        or pairs.model() is not model
        or tuple(t.data_ptr() for t in pairs.tensors) != pairs.places
    ):
        pairs = _WEIGHT_PAIRS[copy] = _weight_pairs(copy, model)
    for targets, sources in pairs.lists:
        if _ONE_CALL_ON_CARDS or not targets[0].is_cuda:
            torch._foreach_copy_(targets, sources)
        else:
            for target, source in zip(targets, sources, strict=True):
                target.copy_(source)
