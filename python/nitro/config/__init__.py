"""Loading and validating NitroLearn settings files."""

from __future__ import annotations

from pathlib import Path

from nitro.config.loader import ConfigError, load_toml, parse_toml
from nitro.config.schema import NitroConfig

__all__ = ["ConfigError", "NitroConfig", "default_config", "load_config"]


def load_config(file: Path) -> NitroConfig:
    """Reads and validates a settings file; raises ConfigError listing every problem."""
    return load_toml(file, NitroConfig)


def default_config(base_dir: Path) -> NitroConfig:
    """All defaults, with relative paths resolved against `base_dir` (used when no file is given)."""
    return parse_toml("", NitroConfig, file=Path("<defaults>"), base_dir=base_dir)
