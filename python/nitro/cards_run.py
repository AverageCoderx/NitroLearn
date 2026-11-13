"""The processes of a run on several cards (`run.cards`).

Card 0 is the training run itself (`nitro.training.TrainingRun`) and alone writes saves, metrics, the
report and self-play's pool and ratings. Each other card is a helper process started
here: it builds its share of the games from card 0's state (written to a file at the start, as a save is),
then, before every iteration, takes what changed from card 0 (`Leader.before_iteration`) and plays and learns
its share. The learner's sums keep every card's networks identical; after every update the cards check it
(every run.cards_check_every updates and before each save).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from nitro import multigpu
from nitro.config.schema import NitroConfig
from nitro.health import TrainingProblem
from nitro.replay_mix import ReplayMix
from nitro.trainer import Envs

if TYPE_CHECKING:
    from nitro.training import TrainingRun

log = logging.getLogger("nitro")

FOLDER = "cards"  # in the run folder: the start file, the meeting file and each helper's log


def _cpus() -> list[int]:
    """The processor cores this process may run on (several cards train on Linux only)."""
    if sys.platform.startswith("linux"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


class Leader:
    """Card 0's side: starts the helpers at the first iteration, then tells them what changed before each."""

    def __init__(self, run: TrainingRun) -> None:
        self.run = run
        self.size = run.written.run.cards
        self.folder = run.folder.root / FOLDER
        self.helpers: list[multiprocessing.process.BaseProcess] = []
        self.cards: multigpu.Cards | None = None
        self.in_step = False  # inside an iteration: the helpers wait in the cards' sums, not for a message
        self.rebuilt: int | None = None  # the checkpoint whose games card 0 built since the last iteration
        self.unchecked = 0  # iterations since the networks were last checked (run.cards_check_every)

    def start(self) -> None:
        """Writes card 0's state, starts the helpers and joins them; returns once every card is ready."""
        run = self.run
        self.folder.mkdir(exist_ok=True)
        store = self.folder / "meeting"
        store.unlink(missing_ok=True)
        start = self.folder / "start.pt"
        torch.save({**run.state(), "checkpoint": run.checkpoint_index}, start)
        context = multiprocessing.get_context("spawn")
        for rank in range(1, self.size):
            helper = context.Process(
                target=helper_main,
                args=(rank, run.whole_config, run.folder.root, str(start), str(store)),
                name=f"nitro-card-{rank}",
                daemon=True,
            )
            helper.start()
            self.helpers.append(helper)
        if run.config.run.cards_pin_threads:  # after the helpers started: they take their own share
            multigpu.pin_threads(multigpu.card_cpus(0, self.size, _cpus()))
        log.info("Training on %d graphics cards: card 0 here, helpers' logs in %s.", self.size, self.folder)
        self.cards = multigpu.join(
            store,
            0,
            self.size,
            run.trainer.device,
            sum_dtype=multigpu.sum_dtype(run.config),
            graphed=run.config.run.cards_graphed_steps,
        )
        run.trainer.learner.cards = self.cards
        if run.replay_mix is not None:
            run.replay_mix.cards = self.cards
        self.rebuilt = None  # already in the start file

    @contextlib.contextmanager
    def iteration(self, agent_steps: int) -> Iterator[None]:
        """Around card 0's iteration: sends the helpers its changes first, checks the networks after (every
        run.cards_check_every iterations)."""
        with self._watched():
            check = self._send_changes(agent_steps)
            yield
            if check:
                self._check_weights()
        self.unchecked = 0 if check else self.unchecked + 1

    def check_before_save(self) -> None:
        """Checks the networks before a save when iterations since the last check left them unchecked
        (run.cards_check_every above 1): the helpers, waiting for card 0's next message, check too."""
        if self.cards is None or not self.unchecked:
            return
        with self._watched():
            self._check_helpers()
            self.cards.send({"stop": False, "check_only": True})
            self._check_weights()
        self.unchecked = 0

    @contextlib.contextmanager
    def _watched(self) -> Iterator[None]:
        """While the helpers are in the cards' sums with card 0: a helper that stopped shows as a plain
        problem naming its log."""
        self.in_step = True
        try:
            yield
        except RuntimeError as err:  # the cards' sums fail (or time out) when a card's process is gone
            if isinstance(err, TrainingProblem):
                raise
            for helper in self.helpers:
                helper.join(timeout=5)
            self._check_helpers()
            raise
        self.in_step = False

    def _send_changes(self, agent_steps: int) -> bool:
        """Sends the helpers what changed; returns whether the cards check their networks this iteration."""
        if self.cards is None:
            self.start()
        self._check_helpers()
        run = self.run
        learner = run.trainer.learner
        check = self.unchecked + 1 >= run.config.run.cards_check_every
        control: dict[str, Any] = {
            "stop": False,
            "agent_steps": agent_steps,
            "gamma": learner.gamma,
            "policy_frozen": learner.policy_frozen,
            "rebuilt": self.rebuilt,
            "selfplay": run.selfplay.state_dict() if self.rebuilt is not None else None,
            "check": check,
        }
        self.rebuilt = None
        assert self.cards is not None
        self.cards.send(control)
        return check

    def _check_weights(self) -> None:
        assert self.cards is not None
        if not self.cards.same_weights(self.run.model.parameters()):
            raise TrainingProblem(
                "the graphics cards' networks drifted apart (run.cards); this should not happen. "
                "Resume from the last save; if it happens again, train on one card (run.cards = 1) and "
                "report it"
            )

    def stop(self) -> None:
        """Lets the helpers finish; called once card 0 has saved, also after an error. Stopped inside an
        iteration (an error, a second Ctrl+C), the helpers wait in a sum card 0 never joins: they are
        ended instead."""
        clean = self.cards is not None and not self.in_step
        if clean and all(helper.is_alive() for helper in self.helpers):
            assert self.cards is not None
            with contextlib.suppress(RuntimeError):  # a helper that went away meanwhile is stopped below
                self.cards.send({"stop": True})
        if clean:  # before waiting for the helpers: NCCL's teardown waits for every card to leave
            multigpu.leave(clean=True)
        for helper in self.helpers:
            helper.join(timeout=60 if clean else 0)
            if helper.is_alive():
                helper.terminate()
                helper.join(timeout=10)
        if not clean:  # the helpers are gone: leave the cards' group too, so this process could join another
            multigpu.leave()
        self.cards = None
        for name in ("start.pt", "meeting"):  # the start file is as big as a save
            (self.folder / name).unlink(missing_ok=True)

    def _check_helpers(self) -> None:
        for rank, helper in enumerate(self.helpers, start=1):
            if not helper.is_alive():
                raise TrainingProblem(
                    f"graphics card {rank}'s process stopped (exit code {helper.exitcode}); its log is "
                    f"{self.folder / f'card{rank}.log'}"
                )


def helper_main(rank: int, config: NitroConfig, root: Path, start: str, store: str) -> None:
    """A helper card's process: its share of the games, learning in step with the others till card 0 stops."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C reaches every process; card 0 decides
    logging.basicConfig(
        filename=root / FOLDER / f"card{rank}.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        Helper(rank, config, root, start, store).run()
    except Exception:
        log.exception("Card %d stopped.", rank)
        multigpu.leave()
        raise
    multigpu.leave(clean=True)  # card 0 asked every card to stop


class Helper:
    def __init__(self, rank: int, whole: NitroConfig, root: Path, start: str, store: str) -> None:
        """`whole`: the run's settings as card 0 runs them, before they are shared out."""
        from nitro import training

        if whole.run.cards_pin_threads:  # before the games' threads start: they keep this share
            multigpu.pin_threads(multigpu.card_cpus(rank, whole.run.cards, _cpus()))
        self.cards = multigpu.join(
            Path(store),
            rank,
            whole.run.cards,
            training.resolve_device(whole.run),
            sum_dtype=multigpu.sum_dtype(whole),
            graphed=whole.run.cards_graphed_steps,
        )
        self.config = config = multigpu.share(whole, rank)  # this card's games, threads and seed
        state = torch.load(start, map_location=self.cards.device, weights_only=False)
        torch.manual_seed(config.run.seed)  # this card's own sampling noise
        state["rng_cpu"] = torch.get_rng_state()
        state["rng_cuda"] = (
            torch.cuda.get_rng_state()
            if self.cards.device.type == "cuda"
            else torch.empty(0, dtype=torch.uint8)
        )
        setup = training.build(config, state, root / "pool")
        self.setup = setup
        self.envs = setup.envs
        self.trainer = setup.trainer
        self.trainer.iterations = int(state.get("iterations", 0))  # numbered as card 0's
        self.selfplay = setup.selfplay
        if "selfplay" in state:
            self.selfplay.load_state_dict(state["selfplay"])
        self._count_learner_cars_only()
        self.trainer.learner.cards = self.cards
        # Copy steps on converted replays ([replay_mix]): from card 0's state, so their optimizer starts
        # as card 0's and every card applies the same averaged steps.
        self.replay_mix: ReplayMix | None = None
        if config.replay_mix.active:
            self.replay_mix = ReplayMix(
                config,
                setup.model,
                self.trainer.rollout.obs.shape[-1],
                self.trainer.device,
                self.trainer.fp16,
                state.get("replay_mix"),
            )
            self.replay_mix.cards = self.cards
            self.trainer.replay_mix = self.replay_mix
        log.info(
            "Card %d of %d: %d games on %s threads, %d steps an iteration.",
            self.cards.rank,
            self.cards.size,
            config.env.num_arenas,
            config.env.threads,
            config.ppo.steps_per_iteration,
        )

    def _count_learner_cars_only(self) -> None:
        for arena, counted in self.selfplay.counted_cars().items():
            self.envs.set_counted(arena, counted)

    def run(self) -> None:
        from nitro import training

        learner = self.trainer.learner
        while True:
            control = self.cards.send(None)
            assert isinstance(control, dict)
            if control["stop"]:
                if self.replay_mix is not None:
                    self.replay_mix.close()
                return
            if control.get("check_only"):  # card 0 saves next (Leader.check_before_save)
                self._check_weights()
                continue
            if control["rebuilt"] is not None:  # card 0 saved: new games, as there
                checkpoint = control["rebuilt"]

                def rebuild(checkpoint: int = checkpoint) -> Envs:
                    self.envs = training.build_envs(self.config, checkpoint, self.trainer.device)
                    return self.envs

                del self.envs
                self.trainer.restart(rebuild)
                self.selfplay.load_state_dict(control["selfplay"])  # the opponents card 0 drew
                self._count_learner_cars_only()
            learner.gamma = learner.reward_scaler.gamma = control["gamma"]
            learner.policy_frozen = control["policy_frozen"]
            learner.restart_twins(control["agent_steps"])  # with card 0, taking its fresh weights
            self.trainer.iteration()
            if control["check"]:  # before the statistics: card 0 checks right after its iteration
                self._check_weights()
            self.envs.take_stats()  # card 0 reports the games; these are let go

    def _check_weights(self) -> None:
        if not self.cards.same_weights(self.trainer.learner.model.parameters()):
            raise TrainingProblem("this card's networks drifted from card 0's")
