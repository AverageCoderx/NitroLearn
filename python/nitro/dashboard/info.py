"""What the dashboard's page shows that belongs to no bot: the Settings tab's groups (`layout`) and the About
block of the App settings page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nitro import __version__, brand
from nitro.dashboard import layout

USE = "Free for non-commercial use."
LICENCE = "PolyForm Noncommercial License 1.0.0"  # pyproject.toml names it by its SPDX id
FAIR_PLAY = "For offline training and RLBot matches only. Never use a bot online against real players."
# The pieces that come inside NitroLearn's windows, each with its licence, by name.
BUNDLED = (
    ("Archivo and JetBrains Mono (the fonts)", "SIL Open Font License 1.1"),
    ("pywebview (the window)", "BSD 3-Clause License"),
    ("tomlkit (writes the settings files)", "MIT License"),
    ("three.js (the 3D match viewer)", "MIT License"),
)


class InfoApi:
    """Mixed into `Api`: every public method is callable from the page."""

    _project: Path

    def settings_layout(self) -> dict[str, Any]:
        """The Settings tab's groups in the order shown, and the settings its Basics group lists."""
        return {"groups": layout.groups(), "basics": list(layout.BASICS)}

    def about(self) -> dict[str, Any]:
        """The About block: the name, credit and version, how it may be used and under which licence, the
        bundled pieces, and NitroLearn's folder."""
        return {
            "name": brand.NAME,
            "credit": brand.CREDIT,
            "version": __version__,
            "use": USE,
            "licence": LICENCE,
            "fair_play": FAIR_PLAY,
            "bundled": [{"name": name, "licence": licence} for name, licence in BUNDLED],
            "folder": str(self._project),
        }
