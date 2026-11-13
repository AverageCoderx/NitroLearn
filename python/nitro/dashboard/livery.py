"""A bot's race number and livery: a two-digit number and two colours it is known by at a glance. Picked from
the bot's name, so they stay the same from one day to the next; kept in the bot's folder once changed."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from nitro.dashboard import edits

FILE = "livery.json"  # in the bot's folder, once its number or colours were changed
DARK, LIGHT = "#141413", "#F3F1EC"
# (main, contrast) pairs: none near the two teams' blue and orange, each easy to tell from the others.
PALETTE = (
    ("#FFD60A", "#141413"),  # yellow
    ("#C8321F", "#F3F1EC"),  # red
    ("#22B8A6", "#141413"),  # teal
    ("#0F6B3E", "#FFD60A"),  # racing green
    ("#7A3FD1", "#F3F1EC"),  # purple
    ("#E83F8C", "#141413"),  # pink
    ("#8C8A84", "#C8321F"),  # primer grey
    ("#B6E33C", "#141413"),  # lime
    ("#7D1F3A", "#FFD60A"),  # maroon
)
_COLOUR = re.compile(r"#[0-9a-fA-F]{6}")


class LiveryError(ValueError):
    """A number or colour a livery cannot take, in plain English."""


def luminance(colour: str) -> float:
    """The relative luminance of `#rrggbb` (WCAG 2)."""
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(one: str, other: str) -> float:
    """The contrast ratio of two colours, from 1 (the same) to 21 (black on white)."""
    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def ink(main: str) -> str:
    """The text colour that reads best on `main`: dark or light."""
    return DARK if contrast(DARK, main) >= contrast(LIGHT, main) else LIGHT


def pick(name: str) -> dict[str, str]:
    """The number and colours a bot of this name gets when none were chosen: always the same for a name."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    main, second = PALETTE[digest[0] % len(PALETTE)]
    number = int.from_bytes(digest[1:3], "big") % 99 + 1
    return {"number": f"{number:02d}", "main": main, "contrast": second, "ink": ink(main)}


def checked(number: Any, main: Any, second: Any) -> dict[str, str]:
    """A livery as chosen in the window, or LiveryError saying what is wrong with it."""
    text = str(number).strip() if isinstance(number, (str, int)) and not isinstance(number, bool) else ""
    if not re.fullmatch(r"[0-9]{1,2}", text):
        raise LiveryError("a race number is one or two digits, 0 to 99")
    for what, colour in (("main colour", main), ("second colour", second)):
        if not isinstance(colour, str) or not _COLOUR.fullmatch(colour):
            raise LiveryError(f"the {what} is written like #FFD60A (a # and six of 0-9 and A-F)")
    main, second = main.upper(), second.upper()
    if contrast(main, second) < 1.5:
        raise LiveryError("the two colours are too alike to tell apart: pick a second colour that stands out")
    return {"number": f"{int(text):02d}", "main": main, "contrast": second, "ink": ink(main)}


def read(folder: Path) -> dict[str, str]:
    """The livery of the bot in `folder`: the one kept there, else the one picked from its name."""
    try:
        kept = json.loads((folder / FILE).read_text(encoding="utf-8"))
        return checked(kept["number"], kept["main"], kept["contrast"])
    except (OSError, ValueError, KeyError, TypeError):
        return pick(folder.name)


def write(folder: Path, number: Any, main: Any, second: Any) -> dict[str, str]:
    """Keeps a chosen livery in the bot's folder and returns it. LiveryError when it cannot be taken."""
    livery = checked(number, main, second)
    kept = {key: livery[key] for key in ("number", "main", "contrast")}
    edits.save(folder / FILE, json.dumps(kept, indent=1))
    return livery
