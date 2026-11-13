"""Attention networks without a room written, playing any game size. An attention encoder's weights do not
depend on the
number of cars: with env.obs_team_size = 0 the run's own team size is its room, and at play time (the RLBot
bot, nitro watch, nitro eval, the dashboard) it reads any number of cars. Where it trained, everything
answers bit for bit as before: the reference here is a frozen copy of the encoders' forward pass from
before the change."""

from __future__ import annotations

import dataclasses
import json
import os
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from nitro import _core
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings
from nitro.eval import EvalError, Side, match_sides, same_game
from nitro.export import write_policy
from nitro.model import (
    ActorCritic,
    EntityEncoder,
    TalkingEncoder,
    _run_head,
    build_networks,
    observation_layout,
    observation_room,
    play_settings,
)
from nitro.runs import RunFolder

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
SIZES = "policy = [16]\nvalue = [16]\nattention_width = 16\nattention_heads = 2\n"
KINDS = {
    "attention": SIZES + 'encoder = "attention"\n',
    "attention_fast": SIZES + 'encoder = "attention"\nattention_fast = true\n',
    "attention_skip": SIZES + 'encoder = "attention"\nattention_skip = true\n',
    "talk": SIZES + 'encoder = "attention_communication"\n',
    "talk_skip": SIZES + 'encoder = "attention_communication"\nattention_skip = true\n',
    "shared": SIZES + 'encoder = "attention"\nshared = [16]\n',
    "transformer": (
        'policy = [16]\nvalue = [16]\nencoder = "transformer"\ntransformer_width = 16\n'
        "transformer_heads = 2\ntransformer_ff = 32\n"
    ),
    "transformer_pads_summary": (
        'policy = [16]\nvalue = [16]\nencoder = "transformer"\ntransformer_width = 16\n'
        "transformer_heads = 2\ntransformer_ff = 32\ntransformer_pads = true\n"
        'transformer_readout = "summary"\nattention_skip = true\n'
    ),
}
# model.lean_attention: the same maths to rounding, so in every size as the usual paths.
KINDS.update({f"{kind}_lean": KINDS[kind] + "lean_attention = true\n" for kind in list(KINDS)})
OLD_KINDS = [kind for kind in KINDS if not kind.startswith("transformer") and not kind.endswith("_lean")]
LEAN = [
    "attention_lean",
    "attention_skip_lean",
    "talk_lean",
    "transformer_lean",
    "transformer_pads_summary_lean",
]


def settings(tmp_path: Path, env: str, model: str) -> NitroConfig:
    return parse_toml(
        f"[env]\n{env}[model]\n{model}", NitroConfig, file=tmp_path / "s.toml", base_dir=tmp_path
    )


def networks(config: NitroConfig, seed: int = 0) -> ActorCritic:
    own, block, teammates, opponents = observation_layout(config)
    torch.manual_seed(seed)
    model = build_networks(config, own + (teammates + opponents) * block, 90).eval()
    with torch.no_grad():  # every weight off its start (norms, null car, embeddings), as a trained bot's
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model


def observations(own: int, block: int, filled: tuple[int, int], places: tuple[int, int], rows: int = 6):
    """Random observations: the first `filled` of `places` teammate and opponent places taken."""
    generator = torch.Generator().manual_seed(1)
    obs = torch.randn(rows, own + sum(places) * block, generator=generator)
    cars = obs[:, own:].view(rows, sum(places), block)
    cars[..., -1] = 0.0
    for place in [*range(filled[0]), *(places[0] + i for i in range(filled[1]))]:
        cars[:, place, -1] = 1.0
    cars[cars[..., -1] == 0] = 0.0
    return obs


def encoder_of(model: ActorCritic) -> torch.nn.Module:
    return model.shared[0] if len(model.shared) else model.policy[0]


def logits(model: ActorCritic, obs: torch.Tensor, places: tuple[int, int] | None = None) -> torch.Tensor:
    """The policy's logits with the encoder told how many cars `obs` holds."""
    with torch.no_grad():
        features = encoder_of(model)(obs, places)
        if len(model.shared):
            return _run_head(model.policy, model.shared[1:](features))
        return _run_head(model.policy[1:], features)


# The encoders' forward pass as it was before they read any number of cars (commit c5444ae), kept frozen as
# the reference that the trained places answer bit for bit as before.
def before_entity(encoder: EntityEncoder, observations: torch.Tensor) -> torch.Tensor:
    own, block, teammates, opponents = encoder.layout
    lead = observations.shape[:-1]
    obs = observations.reshape(-1, observations.shape[-1])
    b = obs.shape[0]
    cars = obs[:, own:].reshape(b, teammates + opponents, block)
    present = cars[..., -1] > 0.5
    side: torch.Tensor = encoder.side  # type: ignore[assignment]
    tokens = encoder.car(cars[..., :-1]) + encoder.sides[side]
    query = encoder.query(obs[:, :own]).unsqueeze(1)
    if encoder.fast:
        query, _ = encoder._attend_fast(query, tokens, present)
    else:
        tokens = torch.cat([encoder.null.expand(b, 1, -1).to(tokens.dtype), tokens], dim=1)
        present = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=obs.device), present], dim=1)
        for step in encoder.attention:
            query, _ = step(query, tokens, present)
    encoded = query.reshape(b, encoder.width)
    if encoder.skip:
        encoded = torch.cat([obs[:, :own].to(encoded.dtype), encoded], dim=1)
    return encoded.reshape(*lead, encoder.outputs)


def before_talking(encoder: TalkingEncoder, observations: torch.Tensor) -> torch.Tensor:
    own, block, teammates, opponents = encoder.layout
    ball_size, car_at, car = encoder.parts
    lead = observations.shape[:-1]
    obs = observations.reshape(-1, observations.shape[-1])
    b = obs.shape[0]
    cars = obs[:, own:].reshape(b, teammates + opponents, block)
    kind: torch.Tensor = encoder.kind  # type: ignore[assignment]
    others = encoder.car(cars[..., :-1]) + encoder.kinds[kind]
    me_numbers = torch.nn.functional.pad(obs[:, car_at : car_at + car], (0, block - 1 - car))
    me = encoder.car(me_numbers) + encoder.kinds[2]
    ball = encoder.ball(obs[:, :ball_size])
    tokens = torch.cat(
        [
            encoder.null.expand(b, 1, -1).to(others.dtype),
            ball[:, None].to(others.dtype),
            me[:, None].to(others.dtype),
            others,
        ],
        dim=1,
    )
    present = torch.cat([torch.ones(b, 3, dtype=torch.bool, device=obs.device), cars[..., -1] > 0.5], dim=1)
    query = encoder.query(obs[:, :own]).unsqueeze(1)
    for talk, listen in zip(encoder.talk, encoder.listen, strict=True):
        tokens, _ = talk(tokens, present)
        query, _ = listen(query, tokens, present)
    encoded = query.reshape(b, encoder.width)
    if encoder.skip:
        encoded = torch.cat([obs[:, :own].to(encoded.dtype), encoded], dim=1)
    return encoded.reshape(*lead, encoder.outputs)


@pytest.mark.parametrize("kind", OLD_KINDS)
@pytest.mark.parametrize(("observation", "room"), [("native", 3), ("nitro_v2", 2)])
def test_at_the_trained_places_the_older_encoders_answer_bit_for_bit_as_before(
    tmp_path: Path, kind: str, observation: str, room: int
) -> None:
    config = settings(tmp_path, f'observation = "{observation}"\nobs_team_size = {room}\n', KINDS[kind])
    model = networks(config)
    own, block, teammates, opponents = observation_layout(config)
    encoder = encoder_of(model)
    before = before_talking if isinstance(encoder, TalkingEncoder) else before_entity
    for filled in ((teammates, opponents), (0, 1), (0, 0)):
        obs = observations(own, block, filled, (teammates, opponents))
        with torch.no_grad():
            reference = before(encoder, obs)  # type: ignore[arg-type]
            assert torch.equal(encoder(obs), reference), filled
            policy, value = model(obs)
            if len(model.shared):
                trunk = model.shared[1:](reference)
                expected = (_run_head(model.policy, trunk), _run_head(model.value, trunk).squeeze(-1))
            else:
                value_encoded = before(model.value[0], obs)  # type: ignore[arg-type]
                expected = (
                    _run_head(model.policy[1:], reference),
                    _run_head(model.value[1:], value_encoded).squeeze(-1),
                )
        assert torch.equal(policy, expected[0]) and torch.equal(value, expected[1]), filled


@pytest.mark.parametrize("kind", ["attention", "talk", "transformer"])
def test_no_room_written_means_the_runs_own_team_size(tmp_path: Path, kind: str) -> None:
    unwritten = settings(tmp_path, "team_size = 2\n", KINDS[kind])
    written = settings(tmp_path, "team_size = 2\nobs_team_size = 2\n", KINDS[kind])
    assert observation_room(unwritten) == 2 and observation_layout(unwritten) == observation_layout(written)
    assert json.loads(core_settings(unwritten))["obs_team_size"] == 2, "the core builds that observation"
    a, b = networks(unwritten), networks(written)
    own, block, teammates, opponents = observation_layout(written)
    obs = observations(own, block, (1, 2), (teammates, opponents))
    with torch.no_grad():
        assert all(torch.equal(x, y) for x, y in zip(a(obs), b(obs), strict=True))
    path = tmp_path / "policy.nwt"
    write_policy(path, a, unwritten, a.obs_size, 90, "run", 0)
    assert _core.Policy(path).header()["obs_team_size"] == 2, "the bot builds the same observation"
    plain = settings(tmp_path, "team_size = 2\n", "policy = [16]\nvalue = [16]\n")
    assert observation_room(plain) == 0 and json.loads(core_settings(plain))["obs_team_size"] == 0


@pytest.mark.parametrize(
    "kind", ["attention", "attention_fast", "talk", "transformer", "transformer_pads_summary", *LEAN]
)
def test_a_network_reads_another_rooms_observation_and_any_number_of_cars(tmp_path: Path, kind: str) -> None:
    config = settings(tmp_path, "team_size = 2\n", KINDS[kind])
    model = networks(config)
    own, block, teammates, opponents = observation_layout(config)
    assert (teammates, opponents) == (1, 2)
    game = observations(own, block, (1, 2), (1, 2))  # 2v2 in its own room
    # The same game with room for 4 a side: the cars in their places, the rest empty.
    wide = torch.zeros(game.shape[0], own + 7 * block)
    wide[:, : own + block] = game[:, : own + block]
    wide[:, own + 3 * block : own + 5 * block] = game[:, own + block :]
    torch.testing.assert_close(logits(model, wide), logits(model, game), rtol=1e-5, atol=1e-5)
    # Exactly the cars of the game, however many: 4v2 (uneven), 12v12, a car alone.
    for places in ((3, 2), (11, 12), (0, 0)):
        obs = observations(own, block, places, places, rows=4)
        answer = logits(model, obs, places)
        assert answer.shape == (4, 90) and answer.isfinite().all(), places
    obs = observations(own, block, (3, 2), (3, 2), rows=4)
    cars = obs[:, own:].view(4, 5, block)
    shuffled = obs.clone()
    shuffled[:, own:].view(4, 5, block)[:] = cars[:, [2, 0, 1, 4, 3]]  # within each side
    torch.testing.assert_close(
        logits(model, shuffled, (3, 2)), logits(model, obs, (3, 2)), rtol=1e-5, atol=1e-5
    )
    crossed = obs.clone()
    crossed[:, own:].view(4, 5, block)[:] = cars[:, [3, 1, 2, 0, 4]]  # a teammate and an opponent trade
    assert not torch.allclose(logits(model, crossed, (3, 2)), logits(model, obs, (3, 2)), atol=1e-6)
    with pytest.raises(ValueError, match="padded observation's car blocks"):
        encoder_of(model)(torch.zeros(2, own + 4 * block))  # an even number of blocks is no room's


@pytest.mark.parametrize(
    "kind", ["attention", "attention_skip", "talk", "transformer", "transformer_pads_summary", *LEAN]
)
def test_the_rlbot_bot_reads_any_number_of_cars_as_pytorch_does(tmp_path: Path, kind: str) -> None:
    config = settings(tmp_path, "team_size = 2\n", KINDS[kind])
    model = networks(config)
    own, block, _, _ = observation_layout(config)
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, model.obs_size, 90, "any-size", 0)
    bot = _core.Policy(path)
    assert bot.reads_any_cars
    for places in ((1, 2), (0, 0), (3, 2), (11, 12)):
        obs = observations(own, block, places, places, rows=3)
        expected = logits(model, obs, places)
        got = torch.tensor([bot.logits_cars(row.tolist(), *places) for row in obs])
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="car blocks"):
        bot.logits_cars([0.0] * (own + 1), 0, 0)


def test_watch_sizes_and_the_settings_a_game_is_played_with(tmp_path: Path) -> None:
    from nitro.dashboard.bots import watch_sizes

    alone = settings(tmp_path, "", KINDS["attention"])  # 1v1, its own room
    padded = settings(tmp_path, "team_size = 2\nobs_team_size = 3\n", "policy = [16]\nvalue = [16]\n")
    plain = settings(tmp_path, "team_size = 2\n", "policy = [16]\nvalue = [16]\n")
    assert watch_sizes(alone) == [1, 2, 3, 4], "an attention bot plays every size the simulator plays"
    assert watch_sizes(padded) == [1, 2, 3] and watch_sizes(plain) == [2]
    big = play_settings(alone, 3)
    assert (big.env.team_size, observation_room(big)) == (3, 3)
    assert observation_layout(play_settings(alone, 1)) == observation_layout(alone), "its own size as trained"
    room_three = settings(tmp_path, "team_size = 2\nobs_team_size = 3\n", KINDS["transformer"])
    assert observation_room(play_settings(room_three, 1)) == 3, "within its room, the room it trained with"
    assert observation_room(play_settings(padded, 1)) == 3 and play_settings(padded, 1).env.team_size == 1

    def side(config: NitroConfig, name: str) -> Side:
        return Side(name, 0, {}, config)

    two = settings(tmp_path, "team_size = 2\n", KINDS["talk"])
    with pytest.raises(EvalError, match="choose the match's size with --team-size"):
        match_sides(side(alone, "a"), side(two, "b"))  # trained 1v1 and 2v2: no size is picked for them
    blue, orange = match_sides(side(alone, "a"), side(two, "b"), team_size=1)
    assert same_game(blue, orange) == [], "two attention bots meet whatever rooms they trained in"
    assert (blue.config.env.team_size, observation_room(blue.config), observation_room(orange.config)) == (
        1,
        2,
        2,
    )
    roomy = settings(tmp_path, "team_size = 2\nobs_team_size = 3\n", KINDS["attention"])
    blue, orange = match_sides(side(two, "b"), side(roomy, "r"))
    assert same_game(blue, orange) == [] and observation_room(blue.config) == 3, "same size, other rooms"
    blue, orange = match_sides(side(alone, "a"), side(two, "b"), team_size=4)
    assert observation_room(blue.config) == observation_room(orange.config) == 4
    with pytest.raises(EvalError, match="cannot play 3v3"):
        match_sides(side(alone, "a"), side(plain, "c"), team_size=3)
    assert same_game(*match_sides(side(alone, "a"), side(plain, "c"))) == ["the game size: 1v1 against 2v2"]
    # An attention bot reads a plain bot's game at the same size: with or without room for more cars.
    plain_one = settings(tmp_path, "", "policy = [16]\nvalue = [16]\n")
    assert same_game(*match_sides(side(alone, "a"), side(plain_one, "p"))) == []
    assert same_game(*match_sides(side(plain_one, "p"), side(alone, "a"))) == []
    padded_one = settings(tmp_path, "obs_team_size = 3\n", "policy = [16]\nvalue = [16]\n")
    blue, orange = match_sides(side(alone, "a"), side(padded_one, "q"))
    assert same_game(blue, orange) == [] and observation_room(blue.config) == 3, (
        "it takes the plain one's room"
    )


def test_watch_takes_an_attention_bot_beyond_its_size(tmp_path: Path) -> None:
    from nitro.cli import _watch_settings
    from nitro.runs import fixed_settings

    file = tmp_path / "bot.toml"
    file.write_text(f"[model]\n{KINDS['transformer']}", encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "bot", file)
    state = {"settings": fixed_settings(folder.load_settings())}
    config = _watch_settings(folder, state, Namespace(team_size=4))
    assert (config.env.team_size, observation_room(config)) == (4, 4)


def tiny_game(config: NitroConfig, **env: object) -> NitroConfig:
    return dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES),
        run=dataclasses.replace(config.run, device="cpu"),
        env=dataclasses.replace(config.env, num_arenas=2, threads=2, **env),
    )


@NEEDS_MESHES
@pytest.mark.parametrize("kind", ["attention", "transformer_pads_summary"])
def test_a_1v1_attention_bot_is_watched_and_evaluated_at_3v3(tmp_path: Path, kind: str) -> None:
    from nitro.eval import play
    from nitro.training import build
    from nitro.watch import record

    config = tiny_game(settings(tmp_path, "", KINDS[kind]))  # 1v1 without a room written
    setup = build(config)
    state = {"learner": {"model": setup.model.state_dict()}, "agent_steps": 0}
    recording = record(play_settings(config, 3), state, seconds=1.0, seed=3, fps=30)
    assert recording.cars == 6 and recording.attention is not None
    labels = recording.attention["labels"]
    assert labels.count("teammate") == 2 and labels.count("opponent") == 3
    for decision in recording.decisions:
        for car in decision[1:]:
            # Each share is rounded to 0.01 in the recording, so the sum is within half of that per share.
            assert len(car[4]) == len(labels) and abs(sum(car[4]) - 1) <= 0.005 * len(labels) + 1e-9
    timed = dataclasses.replace(
        config, terminals=dataclasses.replace(config.terminals, max_episode_seconds=1.0)
    )
    side = Side("bot", 0, state, timed)
    result = play(side, side, games=2, seed=1, team_size=3)
    assert result.points >= 2


@NEEDS_MESHES
@pytest.mark.parametrize("attention_blue", [True, False], ids=["attention-blue", "plain-blue"])
def test_an_attention_bot_plays_a_plain_bot_of_its_size(tmp_path: Path, attention_blue: bool) -> None:
    """A plain bot's observation has no "present" flags; the attention bot reads it with every car present,
    which is what a padded observation of the same game shows it, so the two can play each other."""
    from nitro.eval import _Flagged, _player, play
    from nitro.training import build

    timed = {"max_episode_seconds": 1.0}
    attention = tiny_game(settings(tmp_path, "", KINDS["attention"]))
    plain = tiny_game(settings(tmp_path, "", "policy = [16]\nvalue = [16]\n"))
    attention, plain = (
        dataclasses.replace(c, terminals=dataclasses.replace(c.terminals, **timed))
        for c in (attention, plain)
    )
    sides = [
        Side(name, 0, {"learner": {"model": build(config).model.state_dict()}, "agent_steps": 0}, config)
        for name, config in (("att", attention), ("plain", plain))
    ]
    blue, orange = sides if attention_blue else sides[::-1]
    result = play(blue, orange, games=2, seed=1)
    assert result.points >= 2
    # The attention network gets the flags only when the game's observation has none.
    model = _player(sides[0], plain, 148, 90, torch.device("cpu"))
    assert isinstance(model, _Flagged)
    obs = torch.randn(3, 148)
    flagged = torch.cat([obs, torch.ones(3, 1)], dim=1)  # 1v1 nitro_v2: one car block, its flag last
    torch.testing.assert_close(model(obs)[0], model.inner(flagged)[0])


# A 2v2 bot with attention trains on at 3v3 and plays every size. Each kind as a 2v2 bot of the nitro_v2
# observation with no room written (its room is 2).
TWOS = {kind: KINDS[kind] for kind in ("attention", "talk", "transformer")}
TWO_V_TWO = 'team_size = 2\nobservation = "nitro_v2"\n'
PLAIN = "policy = [16]\nvalue = [16]\n"


def _start_problems(saved: dict[str, object], config: NitroConfig) -> list[str]:
    """Why networks saved with the fixed settings `saved` cannot start a run of `config` (start_from)."""
    from nitro.transfer import Source, problems

    return problems(Source({"settings": saved}, "the old bot", parts="all"), config)


def test_an_attention_bot_resumes_and_starts_at_another_game_size_and_room(tmp_path: Path) -> None:
    from nitro.config.loader import ConfigError
    from nitro.runs import fixed_settings

    for kind, model in TWOS.items():
        file = tmp_path / f"{kind}.toml"
        file.write_text(f"[env]\n{TWO_V_TWO}[model]\n{model}", encoding="utf-8")
        folder = RunFolder.create(tmp_path / "runs", kind, file)
        saved = fixed_settings(folder.load_settings())
        games = (
            "team_size = 3\n",
            "team_size = 3\nobs_team_size = 3\n",
            "obs_team_size = 4\n",
            "opponents = false\n",
        )
        for env in games:
            text = f'[env]\nobservation = "nitro_v2"\n{env}[model]\n{model}'
            folder.settings.write_text(text, encoding="utf-8")
            config = folder.load_settings({"settings": saved})  # a resume takes it
            assert _start_problems(saved, config) == [], (kind, env)  # and so does start_from
        other = KINDS["transformer" if kind != "transformer" else "attention"]
        folder.settings.write_text(f"[env]\n{TWO_V_TWO}[model]\n{other}", encoding="utf-8")
        with pytest.raises(ConfigError, match=r"model\.encoder: was"):
            folder.load_settings({"settings": saved})
    # A plain network still keeps its game, and a padded one its room.
    plain = settings(tmp_path, "team_size = 2\n", PLAIN)
    bigger = settings(tmp_path, "team_size = 3\n", PLAIN)
    assert any("env.team_size" in why for why in _start_problems(fixed_settings(plain), bigger))
    padded = settings(tmp_path, "team_size = 2\nobs_team_size = 3\n", PLAIN)
    roomier = settings(tmp_path, "team_size = 2\nobs_team_size = 4\n", PLAIN)
    differences = _start_problems(fixed_settings(padded), roomier)
    assert differences == ["env.obs_team_size is 3 there and 4 here"]


def test_the_dashboard_lets_a_saved_bot_change_only_the_game_size_its_networks_follow(tmp_path: Path) -> None:
    from nitro.dashboard import describe

    def unlocked(config: NitroConfig) -> set[str]:
        fixed: dict[str, bool] = {}
        stack = describe.describe(config, "", locked=True)
        while stack:
            section = stack.pop()
            fixed |= {item["path"]: item["fixed"] for item in section["items"]}
            stack += section["sections"]
        return {path for path in ("env.team_size", "env.opponents", "env.obs_team_size") if not fixed[path]}

    every = {"env.team_size", "env.opponents", "env.obs_team_size"}
    assert unlocked(settings(tmp_path, TWO_V_TWO, KINDS["transformer"])) == every
    padded = settings(tmp_path, "team_size = 2\nobs_team_size = 3\n", PLAIN)
    assert unlocked(padded) == {"env.team_size", "env.opponents"}
    assert unlocked(settings(tmp_path, "team_size = 2\n", PLAIN)) == set()


def _probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), -1)


def _played(config: NitroConfig, steps: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Every car's observation and mask over `steps` steps of random play of `config`'s games."""
    from nitro.envs import EnvPool

    pool = EnvPool(config, pin_memory=False)
    generator = torch.Generator().manual_seed(5)
    obs, masks = [], []
    for _ in range(steps):
        for g, group in enumerate(pool.groups):
            obs.append(group.obs.clone())
            masks.append(group.mask.clone())
            group.actions[:] = torch.multinomial(group.mask.float(), 1, generator=generator)[:, 0].int()
            pool.step(g)
            pool.wait(g)
    return torch.cat(obs), torch.cat(masks)


def _with_room(config: NitroConfig, room: int) -> NitroConfig:
    return dataclasses.replace(config, env=dataclasses.replace(config.env, obs_team_size=room))


@NEEDS_MESHES
@pytest.mark.parametrize("kind", list(TWOS))
def test_a_2v2_attention_bot_trains_on_at_3v3_and_its_bot_plays_every_size_as_pytorch(
    tmp_path: Path, kind: str
) -> None:
    from nitro.training import TrainingRun

    text = (
        f'[run]\ndevice = "cpu"\n[env]\n{TWO_V_TWO}num_arenas = 2\nthreads = 2\n[model]\n{TWOS[kind]}'
        "[ppo]\nsteps_per_iteration = 256\nminibatches = 2\n"
    )
    file = tmp_path / "bot.toml"
    file.write_text(text, encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "bot", file)
    config = tiny_game(folder.load_settings())
    first = TrainingRun(config, folder)
    first.iteration()
    first.checkpoint(continue_after=False)
    first.metrics.close()
    model = first.trainer.learner.model.eval()
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, model.obs_size, 90, "bot", 0)
    bot = _core.Policy(path)
    assert bot.reads_any_cars and bot.header()["obs_team_size"] == 2

    # The exported bot (nitro_core's CPU forward, the RLBot bot's) answers as PyTorch in real games of every
    # size, as nitro eval and watch play them: room 2 up to 2v2, one place per car beyond.
    for size in (1, 2, 3):
        obs, mask = _played(tiny_game(play_settings(config, size)))
        with torch.no_grad():
            expected = _probs(model.logits(obs), mask)
        if size <= 2:
            rows = torch.tensor(bot.logits_batch(obs.flatten().tolist(), len(obs))).view(len(obs), -1)
            assert torch.equal(rows, torch.tensor([bot.logits(row.tolist()) for row in obs]))
        else:
            rows = torch.tensor([bot.logits_cars(row.tolist(), 2, 3) for row in obs])
        torch.testing.assert_close(_probs(rows, mask), expected, rtol=0, atol=1e-4)

    # Padding costs nothing: the same 2v2 games read with room for 3 give the bot the same bits, so the same
    # actions; PyTorch's kernels may add up in another order there (within 1e-6), choosing the same actions.
    two, mask = _played(tiny_game(config))
    three, mask_three = _played(tiny_game(_with_room(config, 3)))
    assert torch.equal(mask, mask_three) and three.shape[1] > two.shape[1]
    padded = [bot.logits_cars(row.tolist(), 2, 3) for row in three]
    assert [bot.logits(row.tolist()) for row in two] == padded
    with torch.no_grad():
        a, b = _probs(model.logits(two), mask), _probs(model.logits(three), mask)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert torch.equal(a.argmax(-1), b.argmax(-1))

    # Stop, set team_size = 3 in the run's copy, resume: the same networks train on at 3v3.
    folder.settings.write_text(text.replace("team_size = 2", "team_size = 3"), encoding="utf-8")
    state = folder.load_latest()
    bigger = tiny_game(folder.load_settings(state))
    assert (bigger.env.team_size, observation_room(bigger)) == (3, 3)
    resumed = TrainingRun(bigger, folder, state)
    assert resumed.envs.groups[0].obs.shape[0] == 3 * first.envs.groups[0].obs.shape[0] // 2  # 6 cars a game
    row = resumed.iteration()
    assert all(torch.isfinite(torch.tensor(float(row[key]))) for key in ("entropy", "value_loss"))
    resumed.metrics.close()
