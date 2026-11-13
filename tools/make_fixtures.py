"""Saved runs and exported bots from a released version, kept as test fixtures.

Every released version adds its own: a tiny real run per kind of save (native and nitro_v2 observations,
a padded observation, an attention network, self-play on, FP16 when a graphics card is present), two
iterations and a save, copied into tests/fixtures/saves/<label>-<kind>/ with the bot exported from it and
what its networks answer on a fixed set of observations. `tests/test_compatibility.py` then holds every
later version to them: the save loads, resumes and exports, and the networks and the exported bot answer
exactly the same.
A bot trained on a released version is never scrapped by an update.

Needs the collision meshes (it trains), and a graphics card for the FP16 kind:

    .\\.venv\\Scripts\\python -I tools\\make_fixtures.py --label v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "saves"
OBS_ROWS = 8  # observations each fixture's networks are asked about

SETTINGS = """[paths]
collision_meshes = {meshes}
[run]
seed = 7
precision = "{precision}"
checkpoint_minutes = 60
[env]
num_arenas = 8
threads = 2
observation = "{observation}"
obs_team_size = {obs_team_size}
[terminals]
no_touch_seconds = 2
[model]
policy = [32, 32]
value = [32, 32]
{model_extra}[ppo]
steps_per_iteration = 640
minibatches = 2
[selfplay]
enabled = {selfplay}
active_opponents = 1
[[rewards]]
name = "velocity_player_to_ball"
weight = 1
[[rewards]]
name = "strong_touch"
weight = 10
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""

KINDS: dict[str, dict[str, object]] = {
    "native": {},
    "nitro_v2": {"observation": "nitro_v2"},
    "padded": {"obs_team_size": 3},
    "attention": {
        "obs_team_size": 3,
        "model_extra": 'encoder = "attention"\nattention_width = 16\nattention_heads = 2\n',
    },
    "selfplay": {"selfplay": "true"},
    "fp16": {"precision": "fp16"},
}


def settings_text(meshes: Path, kind: str) -> str:
    values: dict[str, object] = {
        "meshes": json.dumps(str(meshes)),
        "precision": "fp32",
        "observation": "native",
        "obs_team_size": 0,
        "selfplay": "false",
        "model_extra": "",
    }
    values |= KINDS[kind]
    return SETTINGS.format(**values)


def answers(state: dict[str, object], config: object, folder: Path) -> dict[str, object]:
    """What the saved networks (on the CPU, FP32) and the exported bot answer on fixed observations."""
    from nitro import _core
    from nitro.export import model_from_state

    model, obs_size, num_actions = model_from_state(state, config)  # type: ignore[arg-type]
    obs = torch.randn(OBS_ROWS, obs_size, generator=torch.Generator().manual_seed(0)) * 2
    with torch.no_grad():
        logits, values = model(obs)
    policy = _core.Policy(folder / "bot" / "policy.nwt")
    bot = policy.logits_batch(obs.flatten().tolist(), OBS_ROWS)
    return {
        "obs": obs.tolist(),
        "logits": logits.tolist(),
        "values": values.tolist(),
        "bot_logits": [bot[r * num_actions : (r + 1) * num_actions] for r in range(OBS_ROWS)],
    }


def make(label: str, kind: str, meshes: Path, work: Path) -> Path:
    from nitro.config import load_config
    from nitro.export import export_run
    from nitro.runs import RunFolder
    from nitro.training import TrainingRun

    file = work / f"{kind}.toml"
    file.write_text(settings_text(meshes, kind), encoding="utf-8")
    folder = RunFolder.create(work / "runs", f"{label}-{kind}", file)
    run = TrainingRun(load_config(file), folder)
    for _ in range(2):
        run.iteration()
    run.checkpoint(continue_after=False)
    run.metrics.close()
    out = FIXTURES / f"{label}-{kind}"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(folder.root, out, ignore=shutil.ignore_patterns("report.html", "train.log", "*.partial"))
    # Nothing of this PC's paths goes into git: the settings (and the save's record of them) name the
    # meshes relative to the run, and a test points the run at its own copy.
    (out / "run.json").write_text(json.dumps({"settings_dir": "."}), encoding="utf-8")
    for saved in (out / "checkpoints").glob("*/state.pt"):
        state = torch.load(saved, map_location="cpu", weights_only=False)
        state["all_settings"] = {
            key: "collision_meshes" if value == str(meshes) else value
            for key, value in state["all_settings"].items()
        }
        assert str(work) not in json.dumps(state["all_settings"], default=str), "a path of this PC is left"
        torch.save(state, saved)
    text = (out / "settings.toml").read_text(encoding="utf-8")
    (out / "settings.toml").write_text(
        text.replace(
            f"collision_meshes = {json.dumps(str(meshes))}", 'collision_meshes = "collision_meshes"'
        ),
        encoding="utf-8",
    )
    fixture = RunFolder(out)
    export_run(fixture, out / "bot", None, build=False)
    state = fixture.load_latest()
    assert state is not None
    expected = answers(state, fixture.load_settings(state), out)
    (out / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", required=True, help="the release these saves stand for, e.g. v1")
    parser.add_argument(
        "--meshes", type=Path, default=Path(os.environ.get("NITRO_TEST_MESHES", ROOT / "collision_meshes"))
    )
    args = parser.parse_args(argv)
    if not (args.meshes / "soccar").is_dir():
        print(f"The collision meshes are not at {args.meshes}; pass --meshes.")
        return 1
    kinds = [k for k in KINDS if k != "fp16" or torch.cuda.is_available()]
    with tempfile.TemporaryDirectory() as work:
        for kind in kinds:
            print(
                f"{args.label}-{kind}: {make(args.label, kind, args.meshes.resolve(), Path(work))}",
                flush=True,
            )
    if "fp16" not in kinds:
        print("No graphics card: the FP16 fixture was left out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
