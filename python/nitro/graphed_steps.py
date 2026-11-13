"""Learning steps replayed as CUDA graphs (`ppo.graphed_steps`).

On a fast graphics card a small or medium network's learning step waits on the processor: every one of its
hundreds of kernels is launched from Python, and the card finishes each long before the next arrives (a 1v1
bot's 4x512 networks on a fast card: the card idle 155 of 193 ms of a learning update). Here each step's
minibatch is copied into
buffers of a fixed size and the whole step (`Learner._step`: forward, loss, backward, unscaling, clipping,
Adam and the loss scaler) replays as one CUDA graph, captured the first time a step of its kind runs. A
minibatch smaller than the buffers is padded with copies of its first sample weighted 0, so every mean and
spread is the real samples' (the same maths, rounded differently); a full one is the eager step exactly.
Kinds of step not covered (several cards unless run.cards_graphed_steps; a first step before Adam has its
state) run eagerly. On several cards each step is two graphs with the cards' sum of the gradients
between them, outside any graph.

The GCO bot's steps: Muon's step replays inside the step's (last) graph like Adam's (its state made
before any capture, the loss scaler's skip on the card); the extra value heads' targets are input buffers as
the advantages are; the entropy controller's bonus is a tensor filled before each replay; advantage_keep's
choice runs inside the graph on one card (a padded step's padding never kept, n and k from its real samples
on the card), and on several cards outside the graphs with the advantages' standardisation, since it gathers
every card's |A|.

ppo.value_share needs nothing here: the value side takes the first k rows of the buffers, k fixed by
their size, and a padded step weights them on the card to ceil(share x its real samples) (`_value_rows`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from nitro.inference import capturing, warmup_stream

if TYPE_CHECKING:
    from nitro.learner import Learner
    from nitro.multigpu import Cards

# The flat tensors' names for the extra value heads' targets: HEAD_TARGET + the head's name.
HEAD_TARGET = "head:"


class GraphedSteps:
    def __init__(self, learner: Learner) -> None:
        self.learner = learner
        self.device = learner.device
        # PyTorch captures Adam's step only when told it may; the default too, so a resume keeps it.
        learner.optimizer.defaults["capturable"] = True
        for group in learner.optimizer.param_groups:
            group["capturable"] = True
        self.capacity = 0
        self.storage: tuple[int, ...] = ()  # where the tensors a step changes lived when the graphs were made
        self.pool = torch.cuda.graph_pool_handle()
        # Per kind of step: what replays it, and its outputs (buffers each replay overwrites).
        self.graphs: dict[tuple[Any, ...], tuple[Callable[[], None], Any]] = {}
        self.entropy = torch.zeros((), device=self.device)  # ppo.entropy_target's bonus, refilled each step

    def takes(self) -> bool:
        """Whether the next step can run graphed: the optimizer has stepped (the first step makes Adam's and
        Muon's state, so no graph makes any; a weight that never gets a gradient, as lean attention's key
        biases, never has any; one whose state was dropped, as the twin critics' at a restart, gets it in the
        capture's eager run, put back to a fresh start), and one card, or several with
        run.cards_graphed_steps. Every card decides alike, so the cards' sums come in the same order;
        nothing of them is inside a graph, so each card captures the kinds of step its own data makes."""
        learner = self.learner
        cards = learner.cards
        return (cards is None or cards.graphed) and len(learner.optimizer.state) > 0

    def reserve(self, capacity: int, example: dict[str, torch.Tensor]) -> None:
        """Buffers for minibatches of up to `capacity` samples shaped as `example`'s (name -> a row's tensor
        of the whole update; HEAD_TARGET + a head's name: its targets); kept while the capacity stays (a
        new one drops every graph). Graphs read the
        weights, Adam's state and the loss scaler where they were captured: any of them moved since (a
        network sent to another device and back, an optimizer state loaded anew) drops them too."""
        storage = tuple(t.data_ptr() for t in self._state())
        if storage != self.storage:
            self._drop_graphs()
            self.storage = storage
        if capacity == self.capacity:
            return
        self._drop_graphs()
        self.capacity = capacity
        self.index = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.samples = torch.zeros(capacity, device=self.device)
        self.inputs = {
            name: torch.zeros((capacity, *t.shape[1:]), dtype=t.dtype, device=self.device)
            for name, t in example.items()
        }

    def _drop_graphs(self) -> None:
        """Lets every graph go; the next ones capture into a fresh memory pool. Tensors the dropped graphs
        made can outlive them (a step's outputs), and PyTorch refuses to capture
        into a pool whose graphs are gone while some of its memory is still held."""
        if self.graphs:
            self.graphs.clear()
            self.pool = torch.cuda.graph_pool_handle()

    def step(
        self,
        tensors: dict[str, torch.Tensor],
        batch: torch.Tensor,
        value_only: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """`Learner._step` on the samples `batch` of `tensors` (obs, mask, actions, old_log_probs, advantages,
        targets and the extra heads' targets; the update's flat tensors). Returns its outputs (buffers the
        next step overwrites)."""
        learner = self.learner
        m = batch.numel()
        padded = m < self.capacity
        self._fill(self.index, self.samples, batch, padded)
        for name, buffer in self.inputs.items():
            torch.index_select(tensors[name], 0, self.index, out=buffer)
        config = learner.config
        if config.entropy_target > 0:  # the controller's bonus moves between updates
            self.entropy.fill_(learner.entropy_coef)
        cards = learner.cards
        if cards is not None and not value_only and config.normalize_advantages:
            # Several cards: the advantages standardised over every card's step before it, outside the
            # graphs; the step itself then leaves them as they are (`apart`).
            advantages = self.inputs["advantages"]
            advantages.copy_(cards.standardised(advantages, self.samples if padded else None))
        if cards is not None and not value_only and 0 < config.advantage_keep < 1:
            # And filtered there (the all-gather of every card's |A|), on the real samples as
            # the eager step does, the padding left at 0 (the cards' step sizes came as the update began).
            advantages = self.inputs["advantages"]
            advantages[:m].copy_(learner._filtered(advantages[:m], None))
            advantages[m:].zero_()
        key = (
            value_only,
            padded,
            learner.policy_frozen,
            tuple(group["lr"] for group in learner.optimizer.param_groups),
        )
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(key, value_only, padded)
        replay, outputs = entry
        replay()
        return outputs  # type: ignore[no-any-return]

    @staticmethod
    def _fill(index: torch.Tensor, samples: torch.Tensor, taken: torch.Tensor, padded: bool) -> None:
        """`index` = `taken`, then copies of its first sample; `samples` 1 for the taken ones, 0 after."""
        n = taken.numel()
        index[:n].copy_(taken)
        if not padded:
            return
        index[n:].copy_(taken[:1].expand(index.numel() - n))
        samples.zero_()
        samples[:n].fill_(1.0)

    def _capture(
        self, key: tuple[Any, ...], value_only: bool, padded: bool
    ) -> tuple[Callable[[], None], Any]:
        learner = self.learner
        i = self.inputs
        cards = learner.cards
        heads = {name[len(HEAD_TARGET) :]: t for name, t in i.items() if name.startswith(HEAD_TARGET)}
        args = (
            i["obs"],
            i["mask"],
            i["actions"],
            i["old_log_probs"],
            i["advantages"],
            i["targets"],
            value_only,
            self.samples if padded else None,
            cards is not None,  # apart: the cards' sum between the halves, outside the graphs
            heads or None,
            self.entropy if learner.config.entropy_target > 0 else None,
        )

        def work() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            if cards is not None:  # the warm-up leaves the cards' sum out: one card may capture alone
                from nitro.learner import _finished  # nitro.learner imports this module

                parts = learner._step_parts(*args)
                measured, _, _ = next(parts)
                return _finished(parts, measured)
            return learner._step(*args)

        # One eager run first, as every capture needs (the kernels' set-up, the allocator), on the step's own
        # numbers; what it changes (weights, Adam, the loss scale) goes back as it was, so the step then
        # replayed is the only one taken.
        live = self._state()
        kept = [t.clone() for t in live]
        had = set(learner.optimizer.state)
        side = warmup_stream(self.device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            work()
        torch.cuda.current_stream().wait_stream(side)
        for tensor, value in zip(live, kept, strict=True):
            tensor.copy_(value)
        # A weight whose first gradient came in that run got Adam's state then: back to Adam's start.
        for p, state in learner.optimizer.state.items():
            if p not in had:
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        if cards is None:
            graph = torch.cuda.CUDAGraph()
            # Only this thread is held to the capture's rules (NCCL's watchdog may use the card).
            with capturing(graph, pool=self.pool, capture_error_mode="thread_local"):
                results: Any = work()
            replay: Callable[[], None] = graph.replay
        else:
            replay, results = self._capture_halves(args, cards)
        self.graphs[key] = (replay, results)
        return replay, results

    def _capture_halves(self, args: tuple[Any, ...], cards: Cards) -> tuple[Callable[[], None], Any]:
        """Several cards: the step's two halves (`Learner._step_parts`) as two graphs, the cards' sum
        of the gradients between their replays, outside any graph (summing inside a capture hung on two
        cards). The halves share the graphs' memory pool; the gradients and `measured` the first leaves are
        what the sum reads and writes back in place, and what the second reads. Still outside any graph, and
        in the same order on every card."""
        from nitro.learner import _finished  # nitro.learner imports this module

        learner = self.learner
        first, second = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(self.device)  # type: ignore[no-untyped-call]  # both halves on one stream
        parts = learner._step_parts(*args)
        with capturing(first, pool=self.pool, stream=stream, capture_error_mode="thread_local"):
            measured, count, _ = next(parts)
        grads = [p.grad for p in learner.model.parameters() if p.grad is not None]
        summed = torch.empty_like(measured)
        with capturing(second, pool=self.pool, stream=stream, capture_error_mode="thread_local"):
            outputs = _finished(parts, summed)

        def replay() -> None:
            first.replay()
            summed.copy_(cards.average_(grads, measured, count))
            second.replay()

        return replay, outputs

    def _state(self) -> list[torch.Tensor]:
        """Every tensor a step changes: the weights (the extra value heads' too), the optimizer's state
        (Adam's and Muon's), the loss scalers', and advantage_keep's sums over the update's steps."""
        learner = self.learner
        found = [p.data for p in learner.model.parameters()]
        for state in learner.optimizer.state.values():
            found += [v for v in state.values() if isinstance(v, torch.Tensor)]
        scaler = learner.scaler
        found += [t for t in (scaler._scale, scaler._growth_tracker) if t is not None]
        found.append(learner._kept)
        return found
