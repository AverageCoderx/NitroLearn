"""pretrain.prev_action_dropout: the copy learns some decisions with no previous action."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from nitro import _core, pretrain, replays
from nitro.config.loader import load_toml
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings
from nitro.model import build_networks

ROOT = Path(__file__).resolve().parents[1]


def _batch(rows: int, obs_size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    actions = _core.num_actions()
    index = torch.zeros(rows, replays.LABEL_ENTRIES, dtype=torch.uint8)
    index[:, 0] = torch.randint(0, actions, (rows,), generator=generator).to(torch.uint8)
    share = torch.zeros(rows, replays.LABEL_ENTRIES, dtype=torch.float16)
    share[:, 0] = 1.0
    return {
        "obs": (torch.rand(rows, obs_size, generator=generator) + 0.5).half(),  # no zeros anywhere
        "mask": torch.full((rows, (actions + 7) // 8), 255, dtype=torch.uint8),
        "label_index": index,
        "label_share": share,
    }


def test_the_previous_action_is_where_the_observations_put_it() -> None:
    config = load_toml(ROOT / "configs/presets/2v2_starter.toml", NitroConfig)
    assert config.pretrain.prev_action_dropout == 0.0, "off unless chosen"

    def at(observation: str) -> slice:
        return pretrain.previous_action(
            dataclasses.replace(config, env=dataclasses.replace(config.env, observation=observation))
        )

    assert (at("nitro_v2"), at("native")) == (slice(54, 62), slice(43, 51))


def test_a_round_blanks_the_previous_action_on_its_share_only() -> None:
    config = load_toml(ROOT / "configs/presets/1v1_starter.toml", NitroConfig)
    obs_size, place = 80, slice(54, 62)
    seen: list[torch.Tensor] = []

    def learn(blank: tuple[slice, float] | None) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = build_networks(config, obs_size, _core.num_actions())
        logits = model.logits
        model.logits = lambda obs: (seen.append(obs.detach().clone()), logits(obs))[1]  # type: ignore[method-assign]
        params = [p for name, p in model.named_parameters() if not name.startswith("value.")]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        pretrain._learn_round(
            model, optimizer, [_batch(512, obs_size)], 1, 128, torch.device("cpu"), 7, blank=blank
        )
        return {k: v.clone() for k, v in model.state_dict().items()}

    plain = learn(None)
    assert all(bool((o[:, place] != 0).all()) for o in seen), "nothing blanked without the setting"
    seen.clear()
    assert all(torch.equal(plain[k], v) for k, v in learn((place, 0.0)).items()), (
        "a share of 0 changes nothing"
    )
    seen.clear()
    learn((place, 0.4))
    blanked = torch.cat([(o[:, place] == 0).all(1) for o in seen]).float().mean()
    assert 0.3 < blanked < 0.5, blanked
    rest = torch.cat([torch.cat([o[:, : place.start], o[:, place.stop :]], 1) for o in seen])
    assert bool((rest != 0).all()), "only the previous action is blanked"


def test_the_plain_copy_blanks_the_previous_action_too_and_is_unchanged_at_zero() -> None:
    """pretrain.train (the copy without batch_replays, and nitro pretrain) as the large copy does."""
    config = load_toml(ROOT / "configs/presets/1v1_starter.toml", NitroConfig)
    place = pretrain.previous_action(config)
    obs_size = _core.observation_layout(core_settings(config))[0]
    batch = _batch(512, obs_size)  # the starter's observation
    actions = _core.num_actions()
    label = torch.zeros(512, actions).scatter_(1, batch["label_index"][:, :1].long(), 1.0)
    samples = pretrain.Samples(
        batch["obs"].float(), torch.ones(512, actions, dtype=torch.bool), label, torch.zeros(512).long()
    )
    seen: list[torch.Tensor] = []
    logits = pretrain.ActorCritic.logits

    def watched(model: pretrain.ActorCritic, obs: torch.Tensor) -> torch.Tensor:
        if model.training:
            seen.append(obs.detach().clone())
        return logits(model, obs)

    def copy(share: float) -> dict[str, torch.Tensor]:
        chosen = dataclasses.replace(config.pretrain, prev_action_dropout=share)
        model, _ = pretrain.train(
            dataclasses.replace(config, pretrain=chosen),
            samples,
            samples.take(torch.arange(0)),
            epochs=2,
            batch=128,
            lr=1e-3,
            device=torch.device("cpu"),
            seed=5,
        )
        return model.state_dict()

    pretrain.ActorCritic.logits = watched  # type: ignore[method-assign]
    try:
        plain = copy(0.0)
        assert all(bool((o[:, place] != 0).all()) for o in seen), "nothing blanked at 0"
        seen.clear()
        blanked = copy(0.4)
        share = torch.cat([(o[:, place] == 0).all(1) for o in seen]).float().mean()
        assert 0.3 < share < 0.5, share
        rest = torch.cat([torch.cat([o[:, : place.start], o[:, place.stop :]], 1) for o in seen])
        assert bool((rest != 0).all()), "only the previous action is blanked"
        assert any(not torch.equal(plain[k], v) for k, v in blanked.items())
        assert all(torch.equal(plain[k], v) for k, v in copy(0.0).items()), "the same copy at 0"
        assert bool((samples.obs[:, place] != 0).all()), "the samples themselves are untouched"
    finally:
        pretrain.ActorCritic.logits = logits  # type: ignore[method-assign]
