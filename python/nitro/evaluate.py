"""Plays a policy without learning and returns what each car did (practice-task evaluation)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import torch

from nitro import _core
from nitro.config.schema import NitroConfig
from nitro.envs import EnvPool
from nitro.ppo import likeliest, masked_log_probs, normalized_entropy
from nitro.selector import Player, Saved, distinct_views

SAMPLED = (1.0, 1.0)  # both sides sampled as in training: evaluate's default pick


def other_rows(
    rows: int, cars: int, bot_cars: Sequence[int] | None, device: torch.device | str
) -> torch.Tensor:
    """True for the rows of the second side: Orange (cars in each arena alternate Blue, Orange, ...), or,
    with `bot_cars`, every car not in it."""
    car = torch.arange(rows, device=device) % cars
    if bot_cars is None:
        return car % 2 == 1
    return ~torch.isin(car, torch.tensor(list(bot_cars), device=device))


def team_logits(
    model: torch.nn.Module,
    opponent: torch.nn.Module | None,
    obs: torch.Tensor,
    cars: int,
    bot_cars: Sequence[int] | None = None,
) -> torch.Tensor:
    """Action logits for `obs` (`cars` rows per arena): `model` drives Blue and `opponent` (if any) Orange;
    with `bot_cars` (car indices within an arena) `model` drives those cars only and `opponent` every other
    one, teammates included."""
    logits: torch.Tensor = model(obs)[0]
    if opponent is None:
        return logits
    others = other_rows(obs.shape[0], cars, bot_cars, obs.device)
    return torch.where(others[:, None], opponent(obs)[0], logits)


def pick_rule(temperature: float) -> str:
    """A side's pick rule in words."""
    if temperature == 0:
        return "likeliest (as the RLBot bot)"
    if temperature == 1:
        return "sampled (as training)"
    return f"temperature {temperature:g}"


def check_pick(pick: tuple[float, float]) -> None:
    for temperature in pick:
        if not (math.isfinite(temperature) and temperature >= 0):
            raise ValueError(f"a pick temperature must be 0 or above and finite (got {temperature})")


def apply_pick(
    logits: torch.Tensor,
    mask: torch.Tensor,
    chosen: torch.Tensor,
    side_rows: torch.Tensor,
    pick: tuple[float, float],
    generator: torch.Generator | None,
) -> torch.Tensor:
    """`chosen` (sampled at temperature 1) with each side whose temperature is not 1 picking by its own
    rule instead: 0 = likeliest, else drawn from softmax(masked logits / T) with `generator`, a stream of its
    own, so a side at 1 keeps exactly the draws it has today. `side_rows`: True for the second side's rows."""
    picked: dict[float, torch.Tensor] = {}
    for side, temperature in enumerate(pick):
        if temperature == 1:
            continue
        if temperature not in picked:
            if temperature == 0:
                picked[temperature] = likeliest(logits, mask)
            else:
                masked = torch.where(mask, logits.float(), -math.inf)
                # Shifted by the row's largest allowed logit first, so a small T cannot overflow.
                scaled = (masked - masked.amax(1, keepdim=True)) / temperature
                probs = torch.softmax(scaled, 1)
                picked[temperature] = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        rows = side_rows if side == 1 else ~side_rows
        chosen = torch.where(rows, picked[temperature], chosen)
    return chosen


@torch.no_grad()
def evaluate(
    model: torch.nn.Module | Saved,
    config: NitroConfig,
    seed: int,
    episodes_per_arena: int,
    device: torch.device,
    fp16: bool,
    opponent: torch.nn.Module | Saved | None = None,
    bot_cars: Sequence[int] | None = None,
    decide_every: tuple[int, int] = (1, 1),
    pick: tuple[float, float] = SAMPLED,
    entropy: list[float] | None = None,
    scripted: Sequence[tuple[int, str]] = (),
) -> list[_core.EpisodeRecord]:
    """Plays `config`'s environments, seeded with `seed`, with the policy sampled as in training (by
    default).

    Runs until every car of every arena has finished `episodes_per_arena` episodes and returns those
    first episodes, ordered by arena and car. Taking the same number from every arena keeps quick
    episodes from being over-represented. The same `seed` gives the same start states and the same
    sampling noise, so two policies can be evaluated on identical situations. With `opponent`, that
    policy drives the Orange cars and `model` the Blue ones; with `bot_cars` as well, `model` drives only
    those cars of each arena and `opponent` all the others (`nitro skills` scores one car alone).
    `decide_every` (Blue, Orange; with
    `opponent`): each side chooses only every that many steps of its episode and holds its last choice in
    between, so bots trained at different tick skips meet in one game run at the smaller one.
    `pick` (Blue, Orange; or `bot_cars` and the others): each side's temperature, 1 = sampled as in training,
    0 = its likeliest action (as nitro_bot plays), else softmax(logits / T) (`apply_pick`). With
    `entropy`, each side's mean normalised entropy over the decisions it made (every step played) is
    appended to it, Blue's then Orange's (NaN for a side without cars).
    Each (team, name) of `scripted` gives that team's cars (0 Blue, 1 Orange) in every arena to a scripted
    benchmark opponent (`nitro.gauntlet`); the actions chosen for them are ignored. `model` and `opponent`
    may each be a saved multi-model bot, whose selector and models then play their cars (sampled, or with
    pick 0 the selector's likeliest choice).
    """
    if bot_cars is not None and opponent is None:
        raise ValueError("bot_cars needs an opponent policy for the other cars")
    if opponent is not None and bot_cars is None and not config.env.opponents:
        raise ValueError("an opponent policy needs an opposing team (env.opponents = true)")
    if decide_every != (1, 1) and (opponent is None or bot_cars is not None or min(decide_every) < 1):
        raise ValueError("decide_every needs an opponent policy on Orange and steps of at least 1")
    check_pick(pick)
    if pick[0] != pick[1] and opponent is None:
        raise ValueError("without an opponent policy one pick rule plays every car: give both sides the same")
    if config.terminals.max_episode_seconds <= 0:
        raise ValueError(
            "evaluation needs terminals.max_episode_seconds above 0, or an episode may never end"
        )
    bundles = [side for side in (model, opponent) if isinstance(side, Saved)]
    if bundles and (decide_every != (1, 1) or scripted or any(t not in (0, 1) for t in pick)):
        raise ValueError(
            "a multi-model bot plays at its own tick skip, sampled or likeliest, against bots, not scripted "
            "players"
        )
    pool_views = distinct_views(bundles)
    # A skill is measured under the game's rules and on the same start states whatever training used: one
    # game size, team_size.
    envs = EnvPool(
        dataclasses.replace(
            config, run=dataclasses.replace(config.run, seed=seed), env=config.env.one_size()
        ),
        pin_memory=device.type == "cuda",
        views=pool_views,
    )
    if bundles:
        return _evaluate_multi(
            model, opponent, config, envs, pool_views, seed, episodes_per_arena, device, fp16, bot_cars,
            pick, entropy,
        )  # fmt: skip
    assert not isinstance(model, Saved) and not isinstance(opponent, Saved)
    for arena in range(config.env.num_arenas):
        for team, name in scripted:
            envs.set_scripted(arena, team, name)
    cars = sum(tensors.obs.shape[0] for tensors in envs.groups) // config.env.num_arenas
    generator = torch.Generator(device=device).manual_seed(seed)
    # Temperatures between 0 and 1 (or above) draw from a second stream, not the T=1 side's.
    picker = None
    if any(t not in (0, 1) for t in pick):
        picker = torch.Generator(device=device).manual_seed(seed ^ 0x9C1C)
    spread = None  # with `entropy`: (Blue, Orange) x (sum, count)
    if entropy is not None:
        spread = torch.zeros(2, 2, dtype=torch.float64, device=device)
    finished: dict[tuple[int, int], list[_core.EpisodeRecord]] = {}
    players = [model] if opponent is None else [model, opponent]
    for player in players:
        player.eval()
    try:
        while len(finished) < config.env.num_arenas * cars or any(
            len(records) < episodes_per_arena for records in finished.values()
        ):
            for group, tensors in enumerate(envs.groups):
                with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                    logits = team_logits(model, opponent, tensors.obs.to(device), cars, bot_cars)
                mask = tensors.mask.to(device)
                log_probs = masked_log_probs(logits, mask)
                probs = log_probs.exp()
                chosen = torch.multinomial(probs, 1, generator=generator).squeeze(1)
                if pick != SAMPLED:
                    others = other_rows(chosen.shape[0], cars, bot_cars, chosen.device)
                    chosen = apply_pick(logits, mask, chosen, others, pick, picker)
                decides = None
                if decide_every != (1, 1):
                    # A side decides on its own rhythm from each episode's first step, holding its last
                    # action in between (as its bot program would for tick_skip ticks).
                    rows = torch.arange(chosen.shape[0], device=chosen.device)
                    every = torch.where(rows % cars % 2 == 1, decide_every[1], decide_every[0])
                    decides = tensors.step.to(chosen.device) % every == 0
                    chosen = torch.where(decides, chosen, tensors.actions.to(chosen.device, chosen.dtype))
                if spread is not None:
                    others = other_rows(chosen.shape[0], cars, bot_cars, chosen.device)
                    row_spread = normalized_entropy(log_probs, mask).double()
                    for side, side_rows in enumerate((~others, others)):
                        made = side_rows if decides is None else side_rows & decides
                        spread[side, 0] += torch.where(made, row_spread, 0.0).sum()
                        spread[side, 1] += made.sum()
                tensors.actions.copy_(chosen)
                envs.step(group)
                envs.wait(group)
            for record in envs.take_stats().episodes:
                finished.setdefault((record.env, record.car), []).append(record)
    finally:
        for player in players:
            player.train()
    if entropy is not None and spread is not None:
        entropy.extend(total / count if count else math.nan for total, count in spread.tolist())
    return [record for key in sorted(finished) for record in finished[key][:episodes_per_arena]]


def _evaluate_multi(
    model: torch.nn.Module | Saved,
    opponent: torch.nn.Module | Saved | None,
    config: NitroConfig,
    envs: EnvPool,
    pool_views: list[dict[str, Any]],
    seed: int,
    episodes_per_arena: int,
    device: torch.device,
    fp16: bool,
    bot_cars: Sequence[int] | None,
    pick: tuple[float, float],
    entropy: list[float] | None,
) -> list[_core.EpisodeRecord]:
    """`evaluate` with a multi-model bot on at least one side: each side decides for every car and plays its
    own; a multi-model side's selector state follows every car it drives."""
    cars = sum(tensors.obs.shape[0] for tensors in envs.groups) // config.env.num_arenas
    generator = torch.Generator(device=device).manual_seed(seed)
    players: list[Any] = []
    for index, side in enumerate((model, opponent)):
        if isinstance(side, Saved):
            players.append(
                Player(side, config, envs, pool_views, device, fp16, seed + index, greedy=pick[index] == 0)
            )
        else:
            if side is not None:
                side.eval()
            players.append(side)
    spread = torch.zeros(2, 2, dtype=torch.float64, device=device)  # (Blue, Orange) x (sum, count)
    finished: dict[tuple[int, int], list[_core.EpisodeRecord]] = {}
    try:
        while len(finished) < config.env.num_arenas * cars or any(
            len(records) < episodes_per_arena for records in finished.values()
        ):
            for group, tensors in enumerate(envs.groups):
                chosen: list[torch.Tensor | None] = []
                spreads: list[torch.Tensor | None] = []
                for player, temperature in zip(players, pick, strict=True):
                    if player is None:
                        chosen.append(None)
                        spreads.append(None)
                    elif isinstance(player, Player):
                        decision = player.decide(group)
                        chosen.append(decision.actions)
                        spreads.append(decision.randomness.double())
                    else:
                        with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                            logits = player(tensors.obs.to(device))[0]
                        mask = tensors.mask.to(device)
                        log_probs = masked_log_probs(logits, mask)
                        if temperature == 0:
                            chosen.append(likeliest(logits, mask))
                        else:
                            chosen.append(
                                torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(1)
                            )
                        spreads.append(normalized_entropy(log_probs, mask).double())
                actions = chosen[0]
                assert actions is not None
                others = other_rows(actions.shape[0], cars, bot_cars, device)
                if chosen[1] is not None:
                    actions = torch.where(others, chosen[1], actions)
                for team, (rows, row_spread) in enumerate(zip((~others, others), spreads, strict=True)):
                    if row_spread is not None:
                        spread[team, 0] += torch.where(rows, row_spread, 0.0).sum()
                        spread[team, 1] += rows.sum()
                tensors.actions.copy_(actions.to(torch.int32))
                envs.step(group)
                envs.wait(group)
                for player in players:
                    if isinstance(player, Player):
                        player.after_step(group)
            for record in envs.take_stats().episodes:
                finished.setdefault((record.env, record.car), []).append(record)
    finally:
        for player in players:
            if isinstance(player, torch.nn.Module):
                player.train()
    if entropy is not None:
        entropy.extend(total / count if count else math.nan for total, count in spread.tolist())
    return [record for key in sorted(finished) for record in finished[key][:episodes_per_arena]]
