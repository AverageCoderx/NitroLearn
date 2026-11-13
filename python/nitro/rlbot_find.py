"""Where RLBot is on Windows. NitroLearn's bots are RLBot v5 bots (the v5 socket interface, a v5 `bot.toml`),
so only RLBot v5 counts as RLBot here.

RLBot v5's installer (rlbot-v5-installer.msi, in the release tagged "installer" of RLBot's launcher) puts its
launcher in %LOCALAPPDATA%\\RLBot5\\bin\\launcher.exe, and the launcher downloads rlbotgui.exe and
RLBotServer.exe next to it. RLBot v4 (RLBotGUI, in %LOCALAPPDATA%\\RLBotGUIX or ...\\RLBotGUI with its own
Python) does not run NitroLearn's bots: it is found only to say so.

Standard library only: the installer (tools/install_steps.py) loads this file before NitroLearn is installed.
"""

from __future__ import annotations

import re
from pathlib import Path

V5_FOLDER = "RLBot5"
# RLBot v5's programs in RLBot5\bin, the one to open first: the launcher (what RLBot's own shortcuts open)
# keeps the other two up to date.
V5_PROGRAMS = ("launcher.exe", "rlbotgui.exe", "RLBotServer.exe")
WHERE_V5 = "%LOCALAPPDATA%\\RLBot5\\bin\\launcher.exe"


def _parts(program: Path | str) -> list[str]:
    """The path's folder and file names in lower case, with either kind of slash."""
    return [part.lower() for part in re.split(r"[\\/]+", str(program)) if part]


def is_v5(program: Path | str) -> bool:
    """Whether `program` is one of RLBot v5's programs (in an RLBot5 folder)."""
    parts = _parts(program)
    return len(parts) >= 2 and parts[-1].endswith(".exe") and V5_FOLDER.lower() in parts[:-1]


def is_older(program: Path | str) -> bool:
    """Whether `program` belongs to an older RLBot (v4: RLBotGUI, RLBotGUIX, its RLBot.exe), not v5."""
    return not is_v5(program) and any(part.startswith("rlbot") for part in _parts(program))


def v5_programs(local_app_data: Path | None) -> list[Path]:
    """RLBot v5's programs found under %LOCALAPPDATA% (or its Programs folder), the launcher first."""
    if local_app_data is None:
        return []
    found = []
    for name in V5_PROGRAMS:
        for base in (local_app_data, local_app_data / "Programs"):
            program = base / V5_FOLDER / "bin" / name
            if program.is_file():
                found.append(program)
    return found


def find_v5(local_app_data: Path | None) -> Path | None:
    programs = v5_programs(local_app_data)
    return programs[0] if programs else None


def find_older(local_app_data: Path | None) -> Path | None:
    """An RLBot v4 install (its program, else its folder), or None."""
    if local_app_data is None:
        return None
    for base in (local_app_data, local_app_data / "Programs"):
        for folder in sorted(base.glob("RLBot*")):
            if folder.name.lower() == V5_FOLDER.lower() or not folder.is_dir():
                continue
            for pattern in ("*.exe", "*/*.exe"):
                for program in sorted(folder.glob(pattern)):
                    if "rlbot" in program.name.lower():
                        return program
            return folder
    return None
