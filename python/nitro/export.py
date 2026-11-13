"""`nitro export`: turns a run's newest save into a bot folder that RLBot v5 can run.

The folder holds the policy as a plain weights file (`policy.nwt`, read by the Rust bot without PyTorch),
RLBot's `bot.toml` and `loadout.toml`, a README, the notices that travel with the bot (`NOTICES.md`), and the
bot program `nitro_bot` (`nitro_bot.exe` on Windows) when it could be built. Only the policy network is
exported; the value network and the learner's state stay in the checkpoint. The file format is documented
in `crates/nitro_core/src/policy.rs`.

The bot program is built with this computer's folders replaced by short names in what the compiler writes
into it (`--remap-path-prefix`), so a shared bot folder names no folder of the computer that made it; a
program that still holds the home folder's path is not copied into the folder.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nitro import _core, prebuilt, toolchain
from nitro.config import NitroConfig
from nitro.doctor import ADD_RUST
from nitro.fused_norm import AppliedReLU
from nitro.model import (
    ActorCritic,
    EntityEncoder,
    ResidualBlock,
    TalkingEncoder,
    TransformerEncoder,
    _ReLU,
    build_networks,
    observation_layout,
    observation_room,
)
from nitro.notices import FILE as NOTICES_FILE
from nitro.notices import notices
from nitro.runs import RunFolder, actions_problem

MAGIC = b"NITROPOL"
FORMAT_VERSION = 1  # plain networks; with an attention encoder, format 2
ATTENTION_FORMAT = 2
SKIP_FORMAT = 3  # an attention encoder with the own part passed beside it (model.attention_skip)
TALK_FORMAT = 4  # encoder = 'attention_communication'
TRANSFORMER_FORMAT = 6  # encoder = 'transformer'
RESIDUAL_FORMAT = 8  # residual blocks (model.residual)
BOT_CRATE = "nitro_bot"
PROGRAM = f"{BOT_CRATE}.exe" if sys.platform == "win32" else BOT_CRATE  # the bot program on this computer
# Cargo's folders, which source paths in a program's panic messages would show.
_BUILD_FOLDERS = (b".cargo", b"registry/src", b"registry\\src", b"git/checkouts", b"git\\checkouts")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
# The game a policy was trained for, written into its file for the bot.
_GAME_SETTINGS = ("observation", "obs_team_size", "team_size", "opponents", "tick_skip", "action_delay")


class ExportError(RuntimeError):
    """The save cannot be exported; the message says why."""


def _layer_records(model: ActorCritic) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    """Every layer of the shared trunk and the policy head, in order, with its weights."""
    layers: list[dict[str, Any]] = []
    weights: list[torch.Tensor] = []
    for module in [*model.shared, *model.policy]:
        if isinstance(module, ResidualBlock):
            inner, inner_weights = _module_records(list(module.branch))
            layers += [{"kind": "residual_start"}, *inner, {"kind": "residual_add"}]
            weights += inner_weights
            continue
        inner, inner_weights = _module_records([module])
        layers += inner
        weights += inner_weights
    return layers, weights


def _module_records(modules: list[nn.Module]) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    """`_layer_records` for these layers (none a residual block)."""
    layers: list[dict[str, Any]] = []
    weights: list[torch.Tensor] = []
    for module in modules:
        if isinstance(module, TransformerEncoder):
            own, block, teammates, opponents = module.layout
            ball, car_at, car = module.parts
            layers.append(
                {
                    "kind": "transformer",
                    "own": own,
                    "block": block,
                    "teammates": teammates,
                    "opponents": opponents,
                    "ball": ball,
                    "car_at": car_at,
                    "car": car,
                    # The pads follow the ball (nitro_core::obs).
                    "pads_at": ball if module.pads else 0,
                    "pads": int(module.pad_places.shape[0]) if module.pads else 0,
                    "width": module.width,
                    "layers": module.layers,
                    "heads": module.heads,
                    "ff": module.ff_width,
                    "summary": module.summary is not None,
                    "skip": module.skip,
                }
            )
            weights += module.export_weights()
        elif isinstance(module, TalkingEncoder):
            own, block, teammates, opponents = module.layout
            ball, car_at, car = module.parts
            layers.append(
                {
                    "kind": "entity_talk",
                    "own": own,
                    "block": block,
                    "teammates": teammates,
                    "opponents": opponents,
                    "ball": ball,
                    "car_at": car_at,
                    "car": car,
                    "width": module.width,
                    "blocks": module.blocks,
                    "heads": module.heads,
                    "skip": module.skip,
                }
            )
            weights += module.export_weights()
        elif isinstance(module, EntityEncoder):
            own, block, teammates, opponents = module.layout
            layers.append(
                {
                    "kind": "entity_encoder",
                    "own": own,
                    "block": block,
                    "teammates": teammates,
                    "opponents": opponents,
                    "width": module.width,
                    "steps": module.steps_,
                    "heads": module.heads,
                    **({"skip": True} if module.skip else {}),
                }
            )
            weights += module.export_weights()
        elif isinstance(module, nn.Linear):
            layers.append({"kind": "linear", "inputs": module.in_features, "outputs": module.out_features})
            weights += [module.weight, module.bias]
        elif isinstance(module, nn.LayerNorm):  # _LayerNorm included
            layers.append({"kind": "layer_norm", "size": module.normalized_shape[0]})
            weights += [module.weight, module.bias]
        elif isinstance(module, AppliedReLU) and module.slope > 0:
            layers.append({"kind": "leaky_relu"})
        elif isinstance(module, _ReLU | nn.ReLU | AppliedReLU):
            layers.append({"kind": "relu"})
        elif isinstance(module, nn.LeakyReLU):
            layers.append({"kind": "leaky_relu"})
        elif isinstance(module, nn.Tanh):
            layers.append({"kind": "tanh"})
        else:
            raise ExportError(
                f"the bot cannot run a '{type(module).__name__}' layer; export supports model.activation = "
                '"relu", "leaky_relu" or "tanh"'
            )
    return layers, weights


def write_policy(
    path: Path,
    model: ActorCritic,
    config: NitroConfig,
    obs_size: int,
    num_actions: int,
    run: str,
    steps: int,
) -> None:
    """Writes `model`'s policy as a `policy.nwt` file (format in crates/nitro_core/src/policy.rs)."""
    layers, weights = _layer_records(model)
    encoders = [layer for layer in layers if layer["kind"] == "entity_encoder"]
    kinds = {layer["kind"] for layer in layers}
    version = (
        TRANSFORMER_FORMAT
        if "transformer" in kinds
        else TALK_FORMAT
        if "entity_talk" in kinds
        else (SKIP_FORMAT if any(layer.get("skip") for layer in encoders) else ATTENTION_FORMAT)
        if encoders
        else FORMAT_VERSION
    )
    if any(layer["kind"] == "residual_start" for layer in layers):
        version = RESIDUAL_FORMAT  # its reader takes every other format's layers too
    header = _header(version, config, obs_size, num_actions, run, steps)
    _write(path, {**header, "layers": layers}, weights)


def _header(
    version: int, config: NitroConfig, obs_size: int, num_actions: int, run: str, steps: int
) -> dict[str, Any]:
    return {
        "format": version,
        "run": run,
        "agent_steps": steps,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "game": {
            "observation": config.env.observation,
            "obs_team_size": observation_room(config),
            "action_masks": "standard",
            "team_size": config.env.team_size,
            "opponents": config.env.opponents,
            "tick_skip": config.env.tick_skip,
            "action_delay": config.env.action_delay,
        },
        "obs_size": obs_size,
        "num_actions": num_actions,
    }


def _write(path: Path, header: dict[str, Any], weights: list[torch.Tensor]) -> None:
    header_bytes = json.dumps(header).encode("utf-8")
    with torch.no_grad():
        flat = torch.cat([w.detach().float().cpu().reshape(-1) for w in weights]).contiguous()
    if not bool(flat.isfinite().all()):
        raise ExportError("the saved policy contains NaN or infinite weights; it cannot play")
    body = flat.numpy().astype("<f4").tobytes()
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<II", header["format"], len(header_bytes)))
        file.write(header_bytes)
        file.write(body)
    partial.replace(path)


def _bot_toml(name: str, agent_id: str, description: str, team_size: int) -> str:
    tags = '["1v1"]' if team_size == 1 else '["teamplay"]'
    return f"""#:schema https://rlbot.org/schemas/agent.json
# RLBot v5 bot settings, written by `nitro export`. RLBot reads this file to list and start the bot.

[settings]
name = "{name}"
agent_id = "{agent_id}"
loadout_file = "loadout.toml"
# The bot program and the policy file next to this file; RLBot runs the command from this folder.
run_command = "{BOT_CRATE}.exe policy.nwt"
run_command_linux = "./{BOT_CRATE} policy.nwt"
# One bot process drives every car RLBot gives it (needed for 2v2 and 3v3 bots).
hivemind = true

[details]
description = "{description}"
developer = "NitroLearn"
language = "rust"
tags = {tags}
"""


_LOADOUT = """# The car's look in RLBot matches (item ids as RLBot uses them; 0 = none). Written once by
# `nitro export` and then left alone; edit freely, it changes nothing about how the bot plays.
# RLBot reads the [blue_loadout] and [orange_loadout] tables and their [..paint] sub-tables.

[blue_loadout]
team_color_id = 27
custom_color_id = 75
car_id = 23            # Octane
decal_id = 0
wheels_id = 1565
boost_id = 35
antenna_id = 0
hat_id = 0
paint_finish_id = 1681
custom_finish_id = 1681
engine_audio_id = 0
trails_id = 3220
goal_explosion_id = 3018

[blue_loadout.paint]
car_paint_id = 12
decal_paint_id = 0
wheels_paint_id = 7
boost_paint_id = 7
antenna_paint_id = 0
hat_paint_id = 0
trails_paint_id = 2
goal_explosion_paint_id = 0

[orange_loadout]
team_color_id = 9
custom_color_id = 75
car_id = 23            # Octane
decal_id = 0
wheels_id = 1565
boost_id = 35
antenna_id = 0
hat_id = 0
paint_finish_id = 1681
custom_finish_id = 1681
engine_audio_id = 0
trails_id = 3220
goal_explosion_id = 3018

[orange_loadout.paint]
car_paint_id = 12
decal_paint_id = 0
wheels_paint_id = 14
boost_paint_id = 14
antenna_paint_id = 0
hat_paint_id = 0
trails_paint_id = 14
goal_explosion_paint_id = 0
"""


def _readme(
    name: str,
    run: str,
    steps: int,
    config: NitroConfig,
    built: bool,
    skipped: bool = False,
) -> str:
    if built:
        missing = ""
    elif skipped:
        missing = f"""
**The bot program is not here yet:** building it was skipped (`nitro export --no-build`). Run `nitro export`
again without `--no-build` to build `{PROGRAM}` and copy it here.
"""
    else:
        missing = f"""
**The bot program is missing:** `{PROGRAM}` could not be built on this computer (the message `nitro export`
printed says why). Run `nitro export` again once that is fixed (for example once Rust is installed): it
builds the program and copies it here.
"""
    sizes = config.env.sizes_label + (" side by side" if config.env.team_sizes else "")
    delay = config.env.action_delay
    return f"""# {name}

A Rocket League bot trained with NitroLearn (run `{run}`, {steps:,} agent steps; {sizes},
tick skip {config.env.tick_skip}, action delay {delay} tick{"" if delay == 1 else "s"}).

## Fair play
Play it offline only: RLBot matches, private matches against your own bots, and bot tournaments. Never use it
in online matches against real players: that is cheating, and it breaks Rocket League's terms of service.

## Playing it in RLBot v5
1. Install RLBot v5 (https://rlbot.org) and start it.
2. In RLBot, add this folder (or its parent) as a bot folder: the bot appears as **{name}**.
3. Put it in a match and start. The game must be running; RLBot starts the bot program itself.

## Files
| File | What it is |
|---|---|
| `bot.toml` | How RLBot starts the bot; rewritten by every export (`--bot-name` sets the name shown). |
| `policy.nwt` | The trained policy: the network's weights and the game settings it was trained with. |
| `loadout.toml` | The car's look (item ids). Cosmetic only. |
| `{PROGRAM}` | The bot program. It needs no Python and no graphics card. |
| `{NOTICES_FILE}` | The licence notices of the bot program and this folder: keep it with them. |

The bot program reads the game, builds the observation exactly as in training, runs the network and sends
the controls.

## Options
Add `--sample` to `run_command` in `bot.toml` to make the bot pick actions randomly in proportion to its
confidence, as it did during training. Without it, the bot always takes its most likely action.
If the game uses a sent control one tick late, add `--latency 1` so the bot sends each decision a tick
early and acts exactly as in training. By default the bot sends them on the training rhythm (`--latency 0`).
A bot with an attention network plays matches bigger than it trained for (one place per car, any size; how
well is not measured); add `--trained-size-only` to make it sit those out, as other bots do.
The bot answers every game packet at once and runs the network on its own thread until the decision is needed;
the cars of a 2v2 or 3v3 decide in one pass, split across 2 threads (`--threads 2`, the default; the same
decisions with any number). Use `--threads 1` on a machine with few free cores, and `--no-decider` to decide
on each car's own thread instead.
{missing}
Any RLBot bot folder works from any location: copy this whole folder wherever you keep your bots.
"""


@dataclass(frozen=True)
class Exported:
    folder: Path
    files: list[Path]
    built: bool  # the bot program was built or copied into the folder
    build_message: str  # why not, when it was not


def _cargo() -> str | None:
    found = toolchain.find_cargo()
    return str(found) if found else None


def _bot_program(project: Path) -> tuple[Path | None, str]:
    """The release's ready-made bot program when it was built from the code in `project` (no Rust needed),
    else one built with cargo (`_build_bot`)."""
    shipped = prebuilt.program(BOT_CRATE, prebuilt.bot_source(project), project)
    if shipped is not None:
        if found := folder_in(shipped.read_bytes(), project):
            return None, f"the ready-made bot program {shipped} names a folder of this computer ('{found}')"
        return shipped, ""
    return _build_bot(project)


def _build_bot(project: Path) -> tuple[Path | None, str]:
    """Builds the bot program with cargo (when Rust is installed), its build folders' paths replaced by
    short names (`build_env`), and returns it; None and why when it could not be built, or when it still
    names a folder of this computer."""
    target = Path(os.environ.get("CARGO_TARGET_DIR", project / "target"))  # where cargo puts the build
    built = target / "release" / PROGRAM
    cargo = _cargo()
    if cargo is None:
        # A program left from an earlier build may hold this computer's folders: only a fresh build is used.
        return None, (
            "this folder has no ready-made bot program for its code (the installer fetches it) and Rust "
            f"(cargo) was not found to build one. Run the installer again, or {ADD_RUST}."
        )
    print("Building the bot program with cargo (a few minutes the first time)...", flush=True)
    done = subprocess.run(
        [cargo, "build", "--release", "-p", BOT_CRATE],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        env=build_env(project, target),
    )
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-15:])
        return None, f"cargo build failed:\n{tail}"
    try:
        data = built.read_bytes()
    except OSError as err:
        return None, f"the built program {built} could not be read ({err.strerror})"
    if found := folder_in(data, project):
        return None, (
            f"the built program still names a folder of this computer ('{found}'), so it is not copied into "
            "a bot folder others may get. Check the Rust settings (RUSTFLAGS, .cargo/config.toml) for "
            "--remap-path-prefix flags, then export again."
        )
    return built, ""


def build_env(project: Path, target: Path | None = None) -> dict[str, str]:
    """The environment for building the bot program: the folders whose paths the compiler writes into it
    (in panic messages: the home folder, cargo's own and its crate sources, the project and the build
    folder) replaced by short names, the flags cargo would use anyway (`_rustflags`) kept."""
    env = toolchain.cargo_env(project)
    flags = _rustflags(project, env)
    cargo_home = Path(env.get("CARGO_HOME", Path.home() / ".cargo"))
    folders = [
        (Path.home(), "home"),
        (project.resolve(), "nitrolearn"),
        (cargo_home, "cargo"),
        (cargo_home / "registry" / "src", "crates"),
        (cargo_home / "git" / "checkouts", "crates"),
    ]
    if target is not None:
        folders.append((target.resolve(), "target"))
    # When several prefixes match a path, the compiler takes the last: the longest (most specific) last.
    for folder, short in sorted(folders, key=lambda pair: len(str(pair[0]))):
        flags.append(f"--remap-path-prefix={folder}={short}")
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    env.pop("RUSTFLAGS", None)
    return env


def folder_in(data: bytes, project: Path) -> str | None:
    """A folder of this computer the program `data` names (the home folder, the project's, or one of cargo's),
    or None. Case is ignored, as Windows ignores it."""
    text = data.lower()
    folders = [*_BUILD_FOLDERS, str(project.resolve()).encode().lower()]
    home = Path.home()
    if len(home.parts) > 1:  # not the root folder, which every program names
        folders += [str(home).encode().lower(), home.as_posix().encode().lower()]
    found = [folder for folder in folders if folder in text]
    return found[0].decode(errors="replace") if found else None


def _rustflags(project: Path, env: Mapping[str, str]) -> list[str]:
    """The flags cargo gives rustc for a build in `project`, from the first source that has any, as cargo
    reads them: CARGO_ENCODED_RUSTFLAGS, RUSTFLAGS, `target.<this computer's triple>.rustflags`, then
    `build.rustflags` (config files merged, their CARGO_TARGET_<TRIPLE>_RUSTFLAGS / CARGO_BUILD_RUSTFLAGS
    variables added). `target.'cfg(...)'.rustflags` entries are not read."""
    if "CARGO_ENCODED_RUSTFLAGS" in env:
        return [flag for flag in env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if flag]
    if "RUSTFLAGS" in env:
        return env["RUSTFLAGS"].split()
    configs = _cargo_configs(project, env)
    triple = _host_triple(project, env)
    keys: list[tuple[str, ...]] = [("build", "rustflags")]
    if triple is not None:
        keys.insert(0, ("target", triple, "rustflags"))
    for key in keys:
        variable = "CARGO_" + re.sub(r"[^A-Z0-9]", "_", "_".join(key).upper())
        found: list[str] | None = None
        for config in configs:  # the lowest precedence first: lists join, a string replaces
            value: Any = config
            for part in key:
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, list):
                found = [*(found or []), *map(str, value)]
            elif isinstance(value, str):
                found = value.split()
        if variable in env:
            found = [*(found or []), *env[variable].split()]
        if found is not None:
            return found
    return []


def _cargo_configs(project: Path, env: Mapping[str, str]) -> list[dict[str, Any]]:
    """Cargo's config files for a build in `project`, the lowest precedence first: CARGO_HOME's, then
    `.cargo/config.toml` in each folder from the root down to `project`."""
    cargo_home = Path(env.get("CARGO_HOME", Path.home() / ".cargo")).resolve()
    folders = [cargo_home, *(folder / ".cargo" for folder in reversed(project.resolve().parents))]
    folders.append(project.resolve() / ".cargo")
    configs = []
    for i, folder in enumerate(folders):
        if i > 0 and folder == cargo_home:
            continue  # read once, as the lowest
        # Cargo prefers the file without the extension when both exist.
        path = next((p for p in (folder / "config", folder / "config.toml") if p.is_file()), None)
        if path is None:
            continue
        try:
            configs.append(tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue  # cargo itself says what is wrong with it
    return configs


def _host_triple(project: Path, env: Mapping[str, str]) -> str | None:
    """The target triple rustc builds for here (`rustc -vV`), or None when rustc cannot say."""
    rustc = env.get("RUSTC") or shutil.which("rustc")
    if rustc is None:
        return None
    try:
        done = subprocess.run(
            [rustc, "-vV"], cwd=project, capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    hosts = [line[len("host: ") :].strip() for line in done.stdout.splitlines() if line.startswith("host: ")]
    return hosts[0] if hosts else None


def model_from_state(state: dict[str, Any], config: NitroConfig) -> tuple[ActorCritic, int, int]:
    """The networks a checkpoint holds, built to the sizes its weights have; with the observation size and
    the number of actions."""
    model_state = state["learner"]["model"]
    last = max(int(k.split(".")[1]) for k in model_state if k.startswith("policy.") and k.endswith(".weight"))
    num_actions = int(model_state[f"policy.{last}.weight"].shape[0])
    if config.model.uses_attention:
        own, block, teammates, opponents = observation_layout(config)
        obs_size = own + (teammates + opponents) * block
    else:
        # The first network's first weight matrix (a residual block's, when it keeps the width).
        first = "shared." if any(k.startswith("shared.") for k in model_state) else "policy."
        obs_size = int(
            next(v for k, v in model_state.items() if k.startswith(first) and v.ndim == 2).shape[1]
        )
    model = build_networks(config, obs_size, num_actions)
    model.load_state_dict(model_state)
    model.eval()
    return model, obs_size, num_actions


def export_run(
    folder: RunFolder, out: Path, name: str | None, project: Path | None = None, build: bool = True
) -> Exported:
    """Exports `folder`'s newest save into `out` (created or refreshed in place), building the bot program
    too unless `build` is false."""
    state = folder.load_latest()
    if state is None:
        raise ExportError(
            f"{folder.root.name} has no save yet: start training it first (the first save comes after a few "
            "minutes)."
        )
    config = folder.load_settings(state)
    # The game settings the weights were trained with come from the save itself: the run's settings file
    # may have been edited since (tick skip and action delay are not fixed for a resume).
    saved = state.get("all_settings", {})
    trained = {name: saved[f"env.{name}"] for name in _GAME_SETTINGS if f"env.{name}" in saved}
    config = dataclasses.replace(config, env=dataclasses.replace(config.env, **trained))
    table = _core.num_actions()
    if problem := actions_problem(state["learner"]["model"], f"'{folder.root.name}'"):
        raise ExportError(problem)
    model, obs_size, num_actions = model_from_state(state, config)
    if num_actions != table:
        raise ExportError(
            f"'{folder.root.name}' chooses among {num_actions} actions, but this version of NitroLearn has "
            f"{table}: it was made by another version and cannot be exported"
        )

    run = folder.root.name
    steps = int(state["agent_steps"])
    name = name or run
    agent_id = f"nitrolearn/{_SAFE_NAME.sub('-', run).strip('-').lower() or 'nitro-bot'}"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    write_policy(out / "policy.nwt", model, config, obs_size, num_actions, run, steps)
    files.append(out / "policy.nwt")
    description = f"Trained with NitroLearn: run {run}, {steps:,} agent steps"
    for file, text in (
        (out / "bot.toml", _bot_toml(name, agent_id, description, config.env.team_size)),
        (out / "loadout.toml", _LOADOUT),
    ):
        if file.name == "loadout.toml" and file.exists():
            continue  # the car's look may have been changed by hand; keep that version
        file.write_text(text, encoding="utf-8")
        files.append(file)

    project = project or Path(__file__).resolve().parents[2]
    exe, message = _bot_program(project) if build else (None, "building it was skipped")
    built = exe is not None
    if exe is not None:
        shutil.copy2(exe, out / exe.name)
        files.append(out / exe.name)
    (out / "README.md").write_text(_readme(name, run, steps, config, built, not build), encoding="utf-8")
    files.append(out / "README.md")
    (out / NOTICES_FILE).write_text(notices(project), encoding="utf-8")
    files.append(out / NOTICES_FILE)
    return Exported(out, files, built, message)
