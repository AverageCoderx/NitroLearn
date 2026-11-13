"""Check this PC: the checks of `nitro doctor`, for the dashboard. The command runs as its own process, since
its graphics-card check loads PyTorch, which the dashboard's own process never does."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nitro.dashboard import winapp

COMMAND = "nitro doctor"  # what the page tells the user to type for the same checks
SECONDS = 300.0  # the first load of PyTorch from a cold disk can take minutes
_STATES = {"OK": "ok", "WARN": "warning", "FAIL": "problem"}


def run(project: Path, seconds: float = SECONDS) -> dict[str, Any]:
    """Runs `nitro doctor --json` in `project` and returns what the page shows (`read`)."""
    flags = 0
    if sys.platform == "win32":  # no console window flashing up
        flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined, unused-ignore]
    try:
        done = subprocess.run(
            [winapp.console_python(), "-m", "nitro", "doctor", "--json"],
            cwd=project,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=seconds,
            creationflags=flags,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"The checks did not finish in {seconds:.0f} seconds. Try `{COMMAND}` in a terminal."
        }
    return read(done.stdout, done.stderr)


def read(printed: str, complaints: str = "") -> dict[str, Any]:
    """What `nitro doctor --json` printed, as the page shows it: each check with its state ("ok", "warning"
    or "problem"), its message and how to fix it, the counts, and the report's last line."""
    lines = [line for line in printed.splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1])
        checks = [
            {
                "name": str(check["name"]),
                "state": _STATES[check["status"]],
                "message": str(check["detail"]),
                "fix": str(check.get("fix", "")),
            }
            for check in report["checks"]
        ]
    except (IndexError, ValueError, KeyError, TypeError):
        said = "\n".join((complaints.strip() or printed.strip()).splitlines()[-12:])
        return {"error": f"The checks stopped before they were done.\n{said}".strip()}
    return {
        "checks": checks,
        "problems": sum(check["state"] == "problem" for check in checks),
        "warnings": sum(check["state"] == "warning" for check in checks),
        "verdict": str(report.get("verdict", "")),
        "version": str(report.get("version", "")),
        "command": COMMAND,
    }
