"""Simulated games, stepped by the Rust core, read and written through PyTorch CPU tensors."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import torch

from nitro import _core
from nitro.build_info import check_core
from nitro.config.rewards import AllOf, Custom, Reward, ZeroSum
from nitro.config.schema import NitroConfig
from nitro.config.state_setters import Replay, StateSetter
from nitro.model import observation_room
from nitro.recordings import replay_files


@dataclass(frozen=True)
class GroupTensors:
    """Everything one group of arenas reads and writes; row i always belongs to the same car."""

    obs: torch.Tensor  # [agents, obs_size] float32: what each car sees now
    mask: torch.Tensor  # [agents, num_actions] bool: actions allowed now
    reward: torch.Tensor  # [agents] float32: reward for the last step
    terminal: torch.Tensor  # [agents] bool: the last step ended the episode with a goal
    truncated: torch.Tensor  # [agents] bool: the episode was cut short; see final_obs
    episode: torch.Tensor  # [agents] int64: episode the observation in obs belongs to
    step: torch.Tensor  # [agents] int32: step number of that observation within its episode
    final_obs: torch.Tensor  # [agents, obs_size] float32: where a truncated episode ended
    actions: torch.Tensor  # [agents] int32: chosen by the caller before each step

    @classmethod
    def allocate(cls, agents: int, obs_size: int, num_actions: int, *, pin_memory: bool) -> GroupTensors:
        def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, pin_memory=pin_memory)

        return cls(
            obs=empty(agents, obs_size, dtype=torch.float32),
            mask=empty(agents, num_actions, dtype=torch.bool),
            reward=empty(agents, dtype=torch.float32),
            terminal=empty(agents, dtype=torch.bool),
            truncated=empty(agents, dtype=torch.bool),
            episode=empty(agents, dtype=torch.int64),
            step=empty(agents, dtype=torch.int32),
            final_obs=empty(agents, obs_size, dtype=torch.float32),
            actions=empty(agents, dtype=torch.int32),
        )


def tagged(item: Any) -> dict[str, Any]:
    """A reward or a state setter as the core names it: its name and its own settings."""
    if isinstance(item, Replay):
        # The core gets the concrete files; the patterns and the bundled flag are the Python side's.
        return {"name": item.name, "files": replay_files(item), "air_bias": item.air_bias}
    if isinstance(item, AllOf):
        return {"name": item.name, "parts": [tagged(part) for part in item.parts]}
    if isinstance(item, Custom):  # a reward of one's own: its name and its own settings
        return {"name": "custom", "kind": item.name, "settings": dict(item.settings)}
    params = dataclasses.asdict(item)
    del params["weight"]
    for kept_beside in ("zero_sum", "enabled"):
        params.pop(kept_beside, None)
    return params


def core_zero_sum(item: ZeroSum | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {"team_spirit": item.team_spirit, "opponent_scale": item.opponent_scale}


def core_reward(reward: Reward, *, weight: float | None = None) -> dict[str, Any]:
    """A `[[rewards]]` block as the core takes it."""
    return {
        "kind": tagged(reward),
        "weight": reward.weight if weight is None else weight,
        "zero_sum": core_zero_sum(reward.zero_sum),
    }


def core_setter(setter: StateSetter) -> dict[str, Any]:
    return {"kind": tagged(setter), "weight": setter.weight}


def core_settings(config: NitroConfig) -> str:
    """The JSON the Rust core expects (it re-checks everything)."""
    env = config.env
    setters = [core_setter(setter) for setter in config.state_setters]
    return json.dumps(
        {
            "collision_meshes": str(config.paths.collision_meshes),
            "seed": config.run.seed,
            "team_size": env.team_size,
            "opponents": env.opponents,
            "num_arenas": env.num_arenas,
            "groups": env.groups,
            "threads": env.threads,
            "tick_skip": env.tick_skip,
            "action_delay": env.action_delay,
            "observation": env.observation,
            "obs_team_size": observation_room(config),
            "action_masks": "standard",
            "terminals": dataclasses.asdict(config.terminals),
            "rewards": [core_reward(reward) for reward in config.active_rewards],
            "state_setters": setters,
            # Several game sizes and shuffled car places, sent only when set.
            **({"team_sizes": list(env.team_sizes)} if env.team_sizes else {}),
            **({"team_size_shares": list(env.team_size_shares)} if env.team_size_shares else {}),
            **({"shuffle_slots": True} if env.shuffle_slots else {}),
        }
    )


def probe_config(config: NitroConfig, seed: int | None = None) -> NitroConfig:
    """`config` cut down to one arena on one thread, for a shape probe (`nitro eval`) or a single game to
    watch: without several game sizes with shuffled car places (the one game is team_size a side),
    which belong to training, not to a look at the bot."""
    env = dataclasses.replace(config.env.one_size(), num_arenas=1, groups=1, threads=1)
    run = config.run if seed is None else dataclasses.replace(config.run, seed=seed)
    return dataclasses.replace(config, env=env, run=run)


class EnvPool:
    """All simulated arenas, split into groups that are stepped independently.

    Usage per group: fill `groups[g].actions`, call `step(g)`, do other work, then `wait(g)`; only after
    `wait` may the group's tensors be read or written again. The tensors are pinned when CUDA is
    available, so copies to and from the GPU may be asynchronous (`non_blocking=True`): such copies must
    have finished (synchronise a CUDA event recorded after them) before `step(g)` is called, or the
    workers would overwrite observations still being copied and read actions not yet written.
    """

    def __init__(self, config: NitroConfig, *, pin_memory: bool | None = None) -> None:
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        check_core()  # an old core with new settings fails in confusing ways
        self._core = _core.EnvPool(core_settings(config))
        self.groups = tuple(
            GroupTensors.allocate(
                self._core.group_agents(group),
                self._core.obs_size,
                self._core.num_actions,
                pin_memory=pin_memory,
            )
            for group in range(self._core.num_groups)
        )
        for group, tensors in enumerate(self.groups):
            self._core.attach(group, tensors)
            self._core.observe(group)
        for group in range(self.num_groups):
            self._core.wait(group)
        self.arena_team_sizes: list[int] = self._core.arena_team_sizes  # each arena's size

    @property
    def num_groups(self) -> int:
        return self._core.num_groups

    @property
    def num_workers(self) -> int:
        return self._core.num_workers

    @property
    def obs_layout(self) -> _core.ObsLayout:
        return self._core.obs_layout

    @property
    def reward_labels(self) -> list[str]:
        return self._core.reward_labels

    def step(self, group: int) -> None:
        """Starts one step of `group` using `groups[group].actions`; returns immediately."""
        self._core.step(group)

    def wait(self, group: int) -> None:
        """Blocks until `group` has finished its step."""
        self._core.wait(group)

    def card_start(self, group: int) -> tuple[int, int]:
        """The addresses of a C function and its argument that start one step of `group` as `step` does,
        for a CUDA host node to call (`env.card_starts_games`). Keep this pool alive while anything may
        call it; wait for the card before `wait`ing for the group."""
        return self._core.card_start(group)

    def take_stats(self) -> _core.EnvStats:
        """Reward and timing totals since the previous call."""
        return self._core.take_stats()

    def set_counted(self, arena: int, counted: list[bool]) -> None:
        """Which of `arena`'s cars (Blue, Orange, Blue, Orange...) count in `take_stats`; self-play leaves
        its frozen opponents out so the logged rewards are the learner's alone."""
        self._core.set_counted(arena, counted)

    @property
    def num_actions(self) -> int:
        return self._core.num_actions

    def frame(self, arena: int) -> list[float]:
        """Arena `arena`'s current state for viewing (layout: `_core.EnvPool.frame`)."""
        return self._core.frame(arena)

    def start_capture(self, arena: int, every_ticks: int) -> list[list[float]]:
        """Starts writing viewer frames for `arena` every `every_ticks` ticks; returns the boost pads
        as [x, y, z, big]. Frame layout: `nitro_core::capture`."""
        return self._core.start_capture(arena, every_ticks)

    def take_capture(self, arena: int) -> list[float]:
        """The viewer frames `arena` wrote since the last call, flat."""
        return self._core.take_capture(arena)
