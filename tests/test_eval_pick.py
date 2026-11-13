"""How a side picks its actions in evaluation: sampled as in training (the default, unchanged),
likeliest as the RLBot bot plays, or a temperature; and the commands' flags that choose it."""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from nitro import _core
from nitro.cli import build_parser, main
from nitro.config import default_config
from nitro.config.schema import NitroConfig, PPOConfig
from nitro.eval import EvalError, load_side, pick_line, play, randomness_line
from nitro.evaluate import apply_pick, evaluate, likeliest, pick_rule
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.runs import RunFolder, fixed_settings
from nitro.skills import skills_file

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
needs_meshes = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
ACTIONS = 90


def logits_and_mask(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, ACTIONS, generator=generator) * 3
    mask = torch.rand(rows, ACTIONS, generator=generator) > 0.4
    mask[:, 7] = True
    mask[0] = False
    mask[0, 11] = True  # a row with one allowed action
    return logits, mask


def sampled(logits: torch.Tensor, mask: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Generator]:
    """Today's draw, as evaluate makes it."""
    generator = torch.Generator().manual_seed(seed)
    probs = masked_log_probs(logits, mask).exp()
    return torch.multinomial(probs, 1, generator=generator).squeeze(1), generator


def test_sampled_on_both_sides_is_today_and_touches_no_generator() -> None:
    logits, mask = logits_and_mask(64, 1)
    chosen, generator = sampled(logits, mask, seed=3)
    state = generator.get_state()
    second = torch.Generator().manual_seed(5)
    second_state = second.get_state()
    orange = torch.arange(64) % 2 == 1
    assert torch.equal(apply_pick(logits, mask, chosen, orange, (1.0, 1.0), second), chosen)
    assert torch.equal(generator.get_state(), state), "the training stream is never used by the pick"
    assert torch.equal(second.get_state(), second_state)
    apply_pick(logits, mask, chosen, orange, (0.0, 1.0), second)
    assert torch.equal(second.get_state(), second_state), "likeliest draws nothing"
    apply_pick(logits, mask, chosen, orange, (0.5, 1.0), second)
    assert torch.equal(generator.get_state(), state)


def test_likeliest_is_the_masked_argmax_with_the_rust_bots_tie_rule() -> None:
    logits, mask = logits_and_mask(200, 2)
    picked = likeliest(logits, mask)
    assert mask[torch.arange(200), picked].all(), "never a forbidden action"
    assert picked[0] == 11, "a row with one allowed action takes it"
    assert torch.equal(picked, masked_log_probs(logits, mask).argmax(1))
    # Whole-number logits tie often: of equal ones nitro_bot's Policy::argmax takes the last.
    ties = torch.randint(0, 3, (200, ACTIONS), generator=torch.Generator().manual_seed(4)).float()
    tied = likeliest(ties, mask)
    for row in range(200):
        assert int(tied[row]) == _core.Policy.argmax(ties[row].tolist(), mask[row].int().tolist())
    assert (tied != torch.where(mask, ties, -math.inf).argmax(1)).any(), "the tie rule mattered here"


def test_a_temperature_draws_from_the_sharpened_policy() -> None:
    rows = 200_000
    logits, mask = logits_and_mask(2, 6)
    row_logits, row_mask = logits[1:2].expand(rows, -1), mask[1:2].expand(rows, -1)
    chosen = torch.zeros(rows, dtype=torch.long)
    blue = torch.zeros(rows, dtype=torch.bool)  # every row Blue
    picked = apply_pick(row_logits, row_mask, chosen, blue, (0.5, 1.0), torch.Generator().manual_seed(7))
    counts = torch.bincount(picked, minlength=ACTIONS).double() / rows
    expected = torch.softmax(torch.where(mask[1], logits[1].double(), -math.inf) / 0.5, 0)
    assert (counts - expected).abs().max() < 0.01
    assert counts[~mask[1]].sum() == 0, "never a forbidden action"
    tiny = apply_pick(logits, mask, chosen[:2], torch.zeros(2, dtype=torch.bool), (1e-30, 1.0), None)
    assert torch.equal(tiny, likeliest(logits, mask)), (
        "a tiny temperature neither overflows nor leaves argmax"
    )


def test_a_side_at_one_keeps_exactly_its_sampled_actions() -> None:
    logits, mask = logits_and_mask(64, 8)
    chosen, _ = sampled(logits, mask, seed=9)
    orange = torch.arange(64) % 2 == 1
    for pick in ((0.0, 1.0), (0.5, 1.0), (2.0, 1.0)):
        got = apply_pick(logits, mask, chosen, orange, pick, torch.Generator().manual_seed(1))
        assert torch.equal(got[orange], chosen[orange]), pick
        assert mask[torch.arange(64), got].all()
    got = apply_pick(logits, mask, chosen, orange, (1.0, 0.0), None)
    assert torch.equal(got[~orange], chosen[~orange])
    assert torch.equal(got[orange], likeliest(logits, mask)[orange])


def test_wrong_picks_are_refused(tmp_path: Path) -> None:
    config = default_config(tmp_path)
    model: Any = torch.nn.Identity()  # refused before anything plays
    cpu = torch.device("cpu")
    for bad in ((-1.0, -1.0), (math.nan, math.nan), (math.inf, math.inf)):
        with pytest.raises(ValueError, match="temperature"):
            evaluate(model, config, 0, 1, cpu, False, pick=bad)
    with pytest.raises(ValueError, match="one pick rule"):
        evaluate(model, config, 0, 1, cpu, False, pick=(0.0, 1.0))


def save(folder: RunFolder, agent_steps: int, seed: int) -> None:
    config = folder.load_settings()
    torch.manual_seed(seed)
    model = ActorCritic(148, ACTIONS, config.model)
    state = {
        "learner": Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False).state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": agent_steps,
        "iterations": 1,
        "settings": fixed_settings(config),
    }
    folder.save_checkpoint(state, agent_steps, keep=5, milestone_steps=0)


def bot(tmp_path: Path) -> RunFolder:
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[paths]\ncollision_meshes = "{MESHES.resolve().as_posix()}"\n[run]\ndevice = "cpu"\n'
        "[env]\nnum_arenas = 4\ngroups = 2\nthreads = 2\n[model]\npolicy = [8]\nvalue = [8]\n",
        encoding="utf-8",
    )
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    save(folder, 500, seed=1)
    save(folder, 1000, seed=2)
    return folder


def test_the_commands_take_the_pick_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["eval", "x"])
    assert (args.blue_temperature, args.orange_temperature, args.as_bot) == (None, None, False)
    args = parser.parse_args(["eval", "x", "--blue-temperature", "0.5", "--orange-temperature", "0"])
    assert (args.blue_temperature, args.orange_temperature) == (0.5, 0.0)
    assert parser.parse_args(["eval", "x", "--as-bot"]).as_bot
    assert parser.parse_args(["skills", "x"]).temperature == 1.0
    assert parser.parse_args(["skills", "x", "--temperature", "0"]).temperature == 0.0
    args = parser.parse_args(["ab", "a.toml", "b.toml", "--task", "ball_touch", "--also-likeliest"])
    assert (args.also_likeliest, args.likeliest_decides) == (True, False)
    assert parser.parse_args(
        ["ab", "a.toml", "b.toml", "--task", "shot", "--likeliest-decides"]
    ).likeliest_decides
    # Refusals, before anything plays.
    runs = str(tmp_path / "runs")
    assert main(["eval", "x", "--runs", runs, "--as-bot", "--blue-temperature", "1"]) == 1
    assert "leave out the temperatures" in capsys.readouterr().err
    assert main(["skills", "x", "--runs", runs, "--temperature", "-1"]) == 1
    assert "--temperature must be" in capsys.readouterr().err
    (tmp_path / "a.toml").write_text("", encoding="utf-8")
    a = str(tmp_path / "a.toml")
    assert main(["ab", a, a, "--speed", "--also-likeliest"]) == 1
    assert "add --task" in capsys.readouterr().err
    folder = bot(tmp_path)
    blue, orange = load_side(folder, 1000), load_side(folder, 500)
    for bad in (-0.5, math.nan, math.inf):
        with pytest.raises(EvalError, match="temperature"):
            play(blue, orange, games=4, seed=0, pick=(bad, 1.0))


def test_the_report_names_each_rule() -> None:
    assert pick_rule(1.0) == "sampled (as training)"
    assert pick_rule(0.0) == "likeliest (as the RLBot bot)"
    assert pick_rule(0.5) == "temperature 0.5"
    assert pick_line((0.0, 1.0)) == (
        "Picking actions: Blue likeliest (as the RLBot bot), Orange sampled (as training)."
    )
    assert randomness_line([0.514, 0.5]).endswith("Blue 0.51, Orange 0.50.")
    assert [skills_file(t) for t in (1.0, 0.0, 0.3)] == [
        "skills.csv",
        "skills_likeliest.csv",
        "skills_temperature_0.3.csv",
    ]


def fake_rows(setup: str, seed: int, also_likeliest: bool) -> list[dict[str, Any]]:
    """Sampled: B is better (lower); picking the likeliest: B is worse."""
    sampled_value = (3.0 if setup == "A" else 2.0) + 0.02 * seed
    likeliest_value = (2.0 if setup == "A" else 3.0) + 0.02 * seed
    rows: list[dict[str, Any]] = [
        {"stage": "untrained", "eval_seed": 1_000_001, "time_to_touch": 9.0},
        {"stage": "trained", "eval_seed": 1_000_001, "time_to_touch": sampled_value},
        {"stage": "trained", "eval_seed": 1_000_002, "time_to_touch": sampled_value},
    ]
    if also_likeliest:
        rows.append({"stage": "trained_likeliest", "eval_seed": 1_000_001, "time_to_touch": likeliest_value})
    common = {"agent_steps": 100, "episodes": 8, "touch_rate": 1.0, "train_minutes": 0.1}
    return [row | common | {"learned": True, "problems": ""} for row in rows]


def test_ab_scores_the_likeliest_row_only_when_asked_and_decides_as_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nitro.ab as ab
    from nitro.tasks import TASKS

    asked: list[bool] = []

    def practise(
        task: object, setup_seed: tuple[str, int], steps: int, also_likeliest: bool, checkpoints: list[int]
    ) -> Any:
        assert checkpoints == [], "no learning curve asked"
        asked.append(also_likeliest)
        return fake_rows(*setup_seed, also_likeliest)

    monkeypatch.setattr(ab, "practice_config", lambda task, setup, seed, with_rewards: (setup, seed))
    monkeypatch.setattr(ab, "practise", practise)
    task, settings, seeds = TASKS["ball_touch"], {"A": "A", "B": "B"}, [0, 1, 2, 3]

    def compare(name: str, **flags: bool) -> tuple[list[str], Path]:
        experiment = ab.Experiment(tmp_path / name, {})
        return ab.compare_on_task(experiment, task, settings, seeds, 100, **flags), experiment.folder  # type: ignore[arg-type]

    off, folder = compare("off")
    assert not any(asked), "off: practise is asked for nothing new"
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    assert set(plan) == {"task", "steps", "seeds", "eval_seeds", "commit", "code"}, "off: the plan of before"
    with (folder / "runs.csv").open(encoding="utf-8") as file:
        assert {row["stage"] for row in csv.DictReader(file)} == {"untrained", "trained"}
    assert any("B IS BETTER" in line for line in off)
    assert not any(line.startswith("Judged") for line in off)

    also, folder = compare("also", also_likeliest=True)
    assert json.loads((folder / "plan.json").read_text(encoding="utf-8"))["also_likeliest"] is True
    assert any("B IS BETTER" in line for line in also), "the sampled score still decides"
    assert any(
        line.startswith("Judged sampled as in training") and "A 2.03s, B 3.03s" in line for line in also
    )

    decided, folder = compare("decides", likeliest_decides=True)
    assert json.loads((folder / "plan.json").read_text(encoding="utf-8"))["decides"] == "trained_likeliest"
    assert any("B IS WORSE" in line for line in decided)
    assert any(line.startswith("Judged picking the likeliest action") for line in decided)
    # A comparison started one way cannot carry on another, a flag added or left out.
    for name, flags, change in (
        ("off", {"also_likeliest": True}, "also_likeliest: not set -> True"),
        ("also", {}, "also_likeliest: True -> not set"),
        ("also", {"likeliest_decides": True}, "decides: not set -> trained_likeliest"),
        ("decides", {}, "decides: trained_likeliest -> not set"),
        ("decides", {"also_likeliest": True}, "decides: trained_likeliest -> not set"),
    ):
        asked.clear()
        with pytest.raises(ValueError, match=change):
            compare(name, **flags)
        assert not asked, "refused before any run"
    again, _ = compare("decides", likeliest_decides=True)
    assert again == decided, "the same command carries on"


# With the collision meshes: real games.


def small(task: str, arenas: int = 8) -> NitroConfig:
    from nitro.tasks import TASKS

    config = TASKS[task].settings()
    return dataclasses.replace(
        config,
        paths=dataclasses.replace(config.paths, collision_meshes=MESHES),
        run=dataclasses.replace(config.run, device="cpu"),
        env=dataclasses.replace(config.env, num_arenas=arenas, threads=2, tick_skip=8),  # half the decisions
        ppo=PPOConfig(steps_per_iteration=arenas * 16, minibatches=2),
    )


def played(
    monkeypatch: pytest.MonkeyPatch, config: NitroConfig, model: Any, opponent: Any, **kwargs: Any
) -> tuple[list[Any], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """evaluate's records and every step's (observation, mask, actions) as the games received them."""
    import nitro.evaluate as evaluate_module

    seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    real_step = evaluate_module.EnvPool.step

    def recording_step(self: Any, group: int) -> None:
        tensors = self.groups[group]
        seen.append((tensors.obs.clone(), tensors.mask.clone(), tensors.actions.clone()))
        real_step(self, group)

    monkeypatch.setattr(evaluate_module.EnvPool, "step", recording_step)
    records = evaluate(model, config, 11, 1, torch.device("cpu"), False, opponent, **kwargs)
    monkeypatch.undo()
    return records, seen


def outcome(record: Any) -> tuple[Any, ...]:
    fields = ("env", "car", "goal", "seconds", "first_touch", "boost_collected", "ball_y_after_touch")
    return tuple(
        "nan" if isinstance(v, float) and math.isnan(v) else v for v in (getattr(record, f) for f in fields)
    )


@needs_meshes
def test_the_default_plays_exactly_todays_sampled_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard for the default: every action evaluate sends is today's draw, one multinomial per step
    from the seed's generator on the masked policy, Orange from the opponent's network."""
    from nitro.training import build

    config = small("kickoff")
    model = build(config).model
    torch.manual_seed(3)
    opponent = copy.deepcopy(model)
    for parameter in opponent.parameters():
        parameter.data.add_(torch.randn_like(parameter) * 0.5)
    records, seen = played(monkeypatch, config, model, opponent)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for obs, mask, actions in seen:
            orange = (torch.arange(obs.shape[0]) % 2 == 1)[:, None]
            logits = torch.where(orange, opponent(obs)[0], model(obs)[0])
            probs = masked_log_probs(logits, mask).exp()
            assert torch.equal(actions, torch.multinomial(probs, 1, generator=generator).squeeze(1))
    explicit, _ = played(monkeypatch, config, model, opponent, pick=(1.0, 1.0))
    entropy: list[float] = []
    measured, _ = played(monkeypatch, config, model, opponent, entropy=entropy)
    assert [outcome(r) for r in explicit] == [outcome(r) for r in records]
    assert [outcome(r) for r in measured] == [outcome(r) for r in records], "measuring changes nothing"
    assert len(entropy) == 2 and all(0 < value <= 1 for value in entropy)


@needs_meshes
def test_likeliest_blue_against_sampled_orange_in_real_games(monkeypatch: pytest.MonkeyPatch) -> None:
    from nitro.training import build

    config = small("kickoff")
    model = build(config).model
    opponent = copy.deepcopy(model)
    _, seen = played(monkeypatch, config, model, opponent, pick=(0.0, 1.0))
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for obs, mask, actions in seen:
            orange = torch.arange(obs.shape[0]) % 2 == 1
            logits = model(obs)[0]
            draw = torch.multinomial(masked_log_probs(logits, mask).exp(), 1, generator=generator).squeeze(1)
            assert torch.equal(actions[orange], draw[orange]), "Orange keeps today's draws"
            assert torch.equal(actions[~orange], likeliest(logits, mask)[~orange])
    # Blue at 0.5 draws from its own stream, so Orange's draws are still today's, step after step.
    records, seen = played(monkeypatch, config, model, opponent, pick=(0.5, 1.0))
    generator = torch.Generator().manual_seed(11)
    picker = torch.Generator().manual_seed(11 ^ 0x9C1C)
    with torch.no_grad():
        for obs, mask, actions in seen:
            orange = torch.arange(obs.shape[0]) % 2 == 1
            logits = model(obs)[0]
            draw = torch.multinomial(masked_log_probs(logits, mask).exp(), 1, generator=generator).squeeze(1)
            masked = torch.where(mask, logits, -math.inf)
            sharp = torch.softmax((masked - masked.amax(1, keepdim=True)) / 0.5, 1)
            blue = torch.multinomial(sharp, 1, generator=picker).squeeze(1)
            assert torch.equal(actions[orange], draw[orange]), "Orange keeps today's draws"
            assert torch.equal(actions[~orange], blue[~orange])
    assert len(seen) > 1
    again, _ = played(monkeypatch, config, model, opponent, pick=(0.5, 1.0))
    assert [outcome(r) for r in records] == [outcome(r) for r in again], "a temperature repeats exactly"


@needs_meshes
def test_a_tiny_match_as_the_bots_play(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bot(tmp_path)
    runs = str(tmp_path / "runs")
    assert main(["eval", "my-bot", "--runs", runs, "--games", "4", "--max-seconds", "2", "--as-bot"]) == 0
    out = capsys.readouterr().out
    assert "Picking actions: Blue likeliest (as the RLBot bot), Orange likeliest (as the RLBot bot)." in out
    assert "4 points from kickoff" in out and "How random each policy was" in out
    assert main(["eval", "my-bot", "--runs", runs, "--games", "4", "--max-seconds", "2"]) == 0
    assert "Blue sampled (as training), Orange sampled (as training)." in capsys.readouterr().out
    assert (
        main(
            [
                "skills",
                "my-bot",
                "--runs",
                runs,
                "--tasks",
                "ball_touch",
                "--games",
                "4",
                "--temperature",
                "0",
            ]
        )
        == 0
    )
    assert "Picking actions: likeliest" in capsys.readouterr().out
    assert (tmp_path / "runs" / "my-bot" / "skills_likeliest.csv").is_file()
    assert not (tmp_path / "runs" / "my-bot" / "skills.csv").exists(), "the sampled series stays apart"


@needs_meshes
def test_a_practice_run_adds_the_likeliest_row_only_when_asked() -> None:
    from nitro.tasks import TASKS, practise

    task = TASKS["ball_touch"]
    config = small("ball_touch")
    off = practise(task, config, 128)
    on = practise(task, config, 128, also_likeliest=True)
    assert [r["stage"] for r in off] == ["untrained", "trained", "trained"]
    assert [r["stage"] for r in on] == ["untrained", "trained", "trained", "trained_likeliest"]

    def same(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k != "train_minutes"}

    assert [same(r) for r in on[:3]] == [same(r) for r in off], "the rows of before are untouched"
    assert on[3]["eval_seed"] == 1_000_001 and on[3]["agent_steps"] == on[1]["agent_steps"]
    kickoff = practise(TASKS["kickoff"], small("kickoff"), 128, also_likeliest=True)
    assert kickoff[-1]["stage"] == "trained_likeliest", "against the untrained opponent too"
