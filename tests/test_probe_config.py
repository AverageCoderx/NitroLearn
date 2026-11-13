"""The one-arena probe `nitro eval` and `nitro watch` build: it must accept any settings a run trains
with."""

from __future__ import annotations

import json
from pathlib import Path

from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.envs import core_settings, probe_config

BASE = Path(Path.cwd().anchor) / "base"
MANY = """
[env]
num_arenas = 1024
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "random"
"""


def test_the_probe_drops_what_needs_many_arenas_and_keeps_the_game() -> None:
    config = parse_toml(MANY, NitroConfig, file=Path("nitro.toml"), base_dir=BASE)
    probe = probe_config(config, seed=7)
    settings = json.loads(core_settings(probe))
    assert (settings["num_arenas"], settings["groups"], settings["threads"]) == (1, 1, 1)
    assert settings["seed"] == 7 and len(settings["state_setters"]) == 2
    assert probe_config(config).run.seed == config.run.seed
    assert not probe.problems()
