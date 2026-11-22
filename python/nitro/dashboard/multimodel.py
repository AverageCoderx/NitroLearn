"""The dashboard's multi-model bots: which of the user's bots can be a model of a new
multi-model bot (each one's kind, size and any problem, in plain words, before anything trains), making one
from the form, and the live view of each model's share of the driving."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from nitro.config.loader import ConfigError
from nitro.config.schema import NitroConfig
from nitro.dashboard import bots, edits
from nitro.runs import RunFolder

# The selector's network sizes the form offers: its policy and value layers each.
SELECTOR_SIZES = {"small": [128, 128], "medium": [256, 256], "large": [512, 512, 512]}
# What a multi-model bot cannot do yet; the new bot's settings switch them off (its base bot may use them).
_OFF = {
    "exploiters.enabled": False,
    "replay_mix.enabled": False,
    "pretrain.enabled": False,
    "start_from.enabled": False,
    "env.team_sizes": None,
    "env.team_size_shares": None,
    "run.cards": 1,
}


def _kind(config: NitroConfig) -> str:
    encoder = config.model.encoder
    return {"mlp": "plain network", "transformer": "transformer"}.get(encoder, "attention network")


def _game(config: NitroConfig) -> str:
    env = config.env
    return f"{env.team_size}v{env.team_size}" if env.opponents else f"{env.team_size} car alone"


def describe_bot(folder: RunFolder) -> dict[str, Any]:
    """One of the user's bots as a possible model: its kind, size and game, read from its settings (no save
    is loaded, so the list comes quickly)."""
    from nitro.model import parameter_counts

    out: dict[str, Any] = {"name": folder.root.name, "problems": []}
    try:
        config = folder.load_settings()
    except (ConfigError, OSError, ValueError) as err:
        out["problems"].append(f"its settings cannot be read: {err}")
        return out
    out |= {
        "kind": _kind(config),
        "game": _game(config),
        "observation": config.env.observation,
        "tick_skip": config.env.tick_skip,
        "saved": bool(folder.complete_checkpoints()),
    }
    try:
        out["parameters"] = parameter_counts(config)["playing"]
    except (ValueError, RuntimeError):
        out["parameters"] = None
    if config.selector.specialists:
        out["problems"].append("it is a multi-model bot itself")
    if not out["saved"]:
        out["problems"].append("it has no save yet: train it first")
    return out


def _settings(folder: RunFolder) -> NitroConfig | None:
    try:
        return folder.load_settings()
    except (ConfigError, OSError, ValueError):
        return None


def candidates(runs: Path, base: str = "") -> dict[str, Any]:
    """Every bot in `runs` as a possible model, with the problems it would have as a model of a multi-model
    bot playing `base`'s game (a bot's name; empty: no game chosen yet, only the bot's own problems)."""
    from nitro.selector import problems_of, pseudo

    base_config = _settings(RunFolder(runs / base)) if base else None
    found = []
    for folder in bots.bot_folders(runs):
        item = describe_bot(folder)
        config = _settings(folder) if not item["problems"] or item.get("kind") else None
        if base_config is not None and config is not None and not config.selector.specialists:
            item["problems"] += problems_of(pseudo(folder.root.name, config), base_config)
        found.append(item)
    return {"bots": found, "sizes": SELECTOR_SIZES, "fields": fields(), "warmup": SELECTOR_WARMUP}


# The selector's warm-up the form suggests before the models learn, so its first random choices do not
# confuse the models. Not measured: a guess of a few minutes of a full bot's training on a fast PC.
SELECTOR_WARMUP = 20_000_000


def fields() -> list[dict[str, Any]]:
    """The [selector] settings as the form shows them: label, help, explanation and limits."""
    from typing import get_type_hints

    from nitro.config.schema import SelectorConfig
    from nitro.dashboard import describe

    hints = get_type_hints(SelectorConfig)
    return [
        describe.field_item(f"selector.{f.name}", f, hints[f.name], f.default)
        for f in dataclasses.fields(SelectorConfig)
    ]


def settings_text(runs: Path, form: dict[str, Any]) -> tuple[str, Path]:
    """A new multi-model bot's settings file from the form: the game, rewards and starts of the `base` bot,
    the chosen `models`, the selector's size and its settings; what a multi-model bot cannot do yet is
    switched off. Returns the text and the folder its relative paths are read from."""
    models = [str(name) for name in form.get("models", [])]
    if not 2 <= len(models) <= 8:
        raise ValueError("choose 2 to 8 models")
    base = str(form.get("base") or models[0])
    base_folder = RunFolder(runs / base)
    if not base_folder.info.is_file():
        raise ValueError(f"there is no bot called {base}")
    text = base_folder.settings.read_text(encoding="utf-8-sig")
    settings_dir = Path(json.loads(base_folder.info.read_text(encoding="utf-8"))["settings_dir"])
    size = SELECTOR_SIZES.get(str(form.get("selector_size", "small")), SELECTOR_SIZES["small"])
    changes: dict[str, Any] = {
        "selector.specialists": models,
        "selector.min_hold_seconds": float(form.get("min_hold_seconds", 0.5)),
        "selector.switch_cost": float(form.get("switch_cost", 0.0)),
        "selector.train_models": bool(form.get("train_models", False)),
        "selector.model_pull": float(form.get("model_pull", 0.1)),
        "selector.models_start_after": int(float(form.get("models_start_after", SELECTOR_WARMUP))),
        "selector.goal_heads": bool(form.get("goal_heads", False)),
        "selector.standardise_values": bool(form.get("standardise_values", False)),
        "model.policy": size,
        "model.value": size,
        "model.shared": [],
        "model.encoder": "attention" if form.get("attention") else "mlp",
        **_OFF,
    }
    shares = [float(s) for s in form.get("min_share", [])]
    changes["selector.min_share"] = shares if any(shares) else None
    return edits.written(text, changes), settings_dir


def create(runs: Path, name: str, form: dict[str, Any]) -> dict[str, Any]:
    """Makes the multi-model bot `name` from the form (not started), or says what is wrong with the form."""
    text, settings_dir = settings_text(runs, form)
    folder = RunFolder(runs / name)
    folder.checkpoints.mkdir(parents=True)
    edits.save(folder.settings, text)
    folder.info.write_text(json.dumps({"settings_dir": str(settings_dir.resolve())}), encoding="utf-8")
    try:
        config = folder.load_settings()
    except ConfigError as err:
        import shutil

        shutil.rmtree(folder.root)
        return {"errors": [str(err)]}
    return {"made": name, "advice": config.advice()}


def live(folder: RunFolder, config: NitroConfig) -> dict[str, Any] | None:
    """A multi-model bot's newest shares, targets, bonuses and switches a minute, for its Overview; None for
    any other bot."""
    if not config.selector.specialists:
        return None
    from nitro.runs import numbered
    from nitro.selector import _label

    newest = bots.newest(folder)
    # Training names each model after its run folder (a path in selector.specialists too): the metrics'
    # own columns say which, in the models' order; before any, the names as written.
    labels = [
        column.removeprefix("selector_share_")
        for column in bots.metrics(folder).header
        if column.startswith("selector_share_")
    ]
    if len(labels) != len(config.selector.specialists):
        labels = numbered([_label(name) for name in config.selector.specialists])
    targets = list(config.selector.min_share) or [0.0] * len(labels)
    return {
        "models": [
            {
                "name": name,
                "label": label,
                "share": newest.get(f"selector_share_{label}"),
                "target": targets[i] / 100 if i < len(targets) else 0.0,
                "bonus": newest.get(f"selector_bonus_{label}"),
            }
            for i, (name, label) in enumerate(zip(config.selector.specialists, labels, strict=True))
        ],
        "switches_per_minute": newest.get("selector_switches_per_minute"),
        "goal_accuracy": newest.get("selector_goal_accuracy"),
        "train_models": config.selector.train_models,
        "models_start_after": config.selector.models_start_after if config.selector.train_models else 0,
        "agent_steps": newest.get("agent_steps"),
    }
