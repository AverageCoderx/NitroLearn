"""Every preset in configs/presets loads, passes the settings check and means what its name says; the GCO
starter also trains."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from nitro.cli import main
from nitro.config import load_config
from nitro.runs import RunFolder
from nitro.training import TrainingRun

ROOT = Path(__file__).parents[1]
PRESETS = sorted((ROOT / "configs" / "presets").glob("*.toml"))
# The starters, one per game size; the GCO starter is a design of its own.
STARTERS = [p for p in PRESETS if re.fullmatch(r"(\d)v\1_starter", p.stem)]
GCO_STARTER = ROOT / "configs" / "presets" / "gco_1v1_starter.toml"
MESHES = ROOT / "collision_meshes"


@pytest.mark.parametrize("preset", STARTERS, ids=[p.stem for p in STARTERS])
def test_preset_loads_without_problems(preset: Path) -> None:
    config = load_config(preset)
    assert config.problems() == []
    team_size = int(preset.stem[0])
    assert config.env.team_size == team_size and config.env.opponents
    assert config.env.num_arenas * 2 * team_size >= 2000, "about the 1v1 preset's number of cars"
    assert not config.selfplay.enabled and config.run.precision == "fp32"


def test_the_presets_exist() -> None:
    assert [p.stem for p in PRESETS] == ["1v1_starter", "2v2_starter", "3v3_starter", "gco_1v1_starter"]
    assert PRESETS[:3] == STARTERS


@pytest.mark.parametrize("preset", STARTERS[1:], ids=[p.stem for p in STARTERS[1:]])
def test_the_attention_lines_of_the_team_presets_load_with_the_recommended_sizes(
    preset: Path, tmp_path: Path
) -> None:
    lines = preset.read_text(encoding="utf-8").splitlines()
    wanted = ("# obs_team_size", '# encoder = "attention"', "# attention_")
    uncommented = [line[2:] if line.startswith(wanted) else line for line in lines]
    copy = tmp_path / preset.name
    copy.write_text("\n".join(uncommented), encoding="utf-8")
    config = load_config(copy)
    model = config.model
    assert (model.encoder, model.attention_blocks, model.attention_heads, model.attention_width) == (
        "attention",
        2,
        4,
        128,
    )
    assert config.env.obs_team_size == 3 and config.advice() == []
    assert 'never use encoder = "attention"' in STARTERS[0].read_text(encoding="utf-8")
    talking = ("# obs_team_size", '# encoder = "attention_communication"')
    copy.write_text(
        "\n".join(line[2:] if line.startswith(talking) else line for line in lines), encoding="utf-8"
    )
    assert load_config(copy).model.encoder == "attention_communication", "its own line, instead"


SETTING_LINE = re.compile(r"^# (\[\[?[a-z_]+\]\]?\s*(#.*)?$|[a-z][a-z0-9_]* = )")


def commented_blocks(lines: list[str]) -> list[list[int]]:
    """The commented-out examples: runs of lines like `# [section]` or `# name = value`, with the indented
    comment lines that continue them."""
    blocks: list[list[int]] = []
    for i, line in enumerate(lines):
        if SETTING_LINE.match(line):
            if blocks and blocks[-1][-1] == i - 1 - _continued(lines, i):
                blocks[-1].append(i)
            else:
                blocks.append([i])
    return blocks


def _continued(lines: list[str], i: int) -> int:
    """How many indented comment lines come right before line i."""
    n = 0
    while i - 1 - n >= 0 and lines[i - 1 - n].startswith(" ") and lines[i - 1 - n].lstrip().startswith("#"):
        n += 1
    return n


# Examples that replace a line rather than add one, and what each needs alongside.
INSTEAD: tuple[str, ...] = ()
NEEDS = {
    '# encoder = "attention"': "# obs_team_size",
    '# encoder = "attention_communication"': "# obs_team_size",
}


@pytest.mark.parametrize("preset", STARTERS, ids=[p.stem for p in STARTERS])
def test_every_commented_example_loads_once_uncommented(preset: Path, tmp_path: Path) -> None:
    """Every optional feature is in each preset, commented out, and turns on by
    removing the # in front of its lines."""
    lines = preset.read_text(encoding="utf-8").splitlines()
    blocks = commented_blocks(lines)
    sections = {lines[b[0]].split("]")[0].strip("# [") for b in blocks if lines[b[0]].startswith("# [")}
    assert {"pretrain", "replay_mix", "start_from", "wandb"} <= sections
    assert not {"actions", "opponents", "selector", "reference"} & sections
    tried = 0
    for block in blocks:
        chosen = [i for i in block if not lines[i].startswith(INSTEAD)]
        if not chosen:
            continue
        for line, needed in NEEDS.items():
            if any(lines[i].startswith(line) for i in chosen):
                chosen += [i for i, other in enumerate(lines) if other.startswith(needed)]
        copy = [line[2:] if i in chosen else line for i, line in enumerate(lines)]
        file = tmp_path / preset.name
        file.write_text("\n".join(copy), encoding="utf-8")
        load_config(file)  # raises ConfigError, naming the setting, when an example is wrong
        tried += 1
    assert tried >= 9


@pytest.mark.parametrize("preset", PRESETS, ids=[p.stem for p in PRESETS])
def test_a_preset_copied_anywhere_finds_nitrolearns_meshes(preset: Path, tmp_path: Path) -> None:
    """The presets write no mesh path: copied to my-settings or anywhere else, the settings point at
    NitroLearn's own collision_meshes folder, unless the copy's folder has meshes of its own."""
    assert not re.search(r"^collision_meshes\s*=", preset.read_text(encoding="utf-8"), re.MULTILINE)
    own = Path(__file__).parents[1] / "collision_meshes"
    copy = tmp_path / "my-settings" / preset.name
    copy.parent.mkdir()
    copy.write_text(preset.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_config(copy).paths.collision_meshes == own.resolve()
    (copy.parent / "collision_meshes" / "soccar").mkdir(parents=True)
    assert load_config(copy).paths.collision_meshes == (copy.parent / "collision_meshes").resolve()


def test_the_gco_starter_has_every_part_of_the_design_on_and_passes_the_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_config(GCO_STARTER)
    assert config.problems() == [] and config.advice() == []
    assert config.env.team_size == 1 and config.env.opponents
    model, ppo = config.model, config.ppo
    assert model.residual and model.layer_norm and model.shared
    assert model.value_heads == ("main", "goal", "vdag1", "vdag2") and model.value_head_layers
    assert ppo.optimizer == "muon"
    assert ppo.signals_on() == ["goal_critic_weight", "vdag_beta", "advantage_keep", "entropy_target"]
    assert [(r.name, r.weight) for r in config.rewards] == [("goal", 1.0)], "goals are the only reward"
    assert config.selfplay.enabled
    meshes = (MESHES / "soccar").is_dir()  # without them the check says so
    assert main(["check-settings", str(GCO_STARTER)]) == (0 if meshes else 1)
    assert "no problems found" in capsys.readouterr().out


@pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")
def test_the_gco_starter_trains_two_iterations_on_the_processor(tmp_path: Path) -> None:
    """The starter as it is (residual trunks, the four value heads, Muon, every learning signal, self-play),
    with fewer games and steps and its networks 32 wide (the same layers) so two iterations fit a test on
    the processor."""
    config = load_config(GCO_STARTER)
    model = config.model

    def narrow(layers: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(32 for _ in layers)

    config = dataclasses.replace(
        config,
        run=dataclasses.replace(config.run, device="cpu", precision="fp32"),
        env=dataclasses.replace(config.env, num_arenas=8, threads=4),
        model=dataclasses.replace(
            model,
            shared=narrow(model.shared),
            policy=narrow(model.policy),
            value=narrow(model.value),
            value_head_layers=narrow(model.value_head_layers),
        ),
        ppo=dataclasses.replace(config.ppo, steps_per_iteration=640),
        selfplay=dataclasses.replace(config.selfplay, active_opponents=1),
    )
    assert config.problems() == []
    run = TrainingRun(config, RunFolder.create(tmp_path / "runs", "gco", GCO_STARTER), None)
    rows = [run.iteration() for _ in range(2)]
    run.metrics.close()
    assert [row["iteration"] for row in rows] == [1, 2]
    for row in rows:
        for column in ("goal_loss", "vdag_loss", "advantage_kept", "entropy_coef"):
            assert row[column] == row[column], f"{column} is NaN"
