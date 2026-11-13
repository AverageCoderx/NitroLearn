"""Makes NitroLearn-Setup.exe's pictures (installer/NitroLearn.iss) from the logo (brand/logo.svg):

    installer/wizard-<scale>.png        the dark side panel of the Welcome and Finish pages: the logo, two
                                        yellow stripes, "NitroLearn" and "by Raze" (164 x 314 at 100%)
    installer/wizard-small-<scale>.png  the logo at the top right of the other pages (55 x 55 at 100%)

for each display scale Inno Setup picks from (100% to 250%).

    python tools/make_wizard_images.py        (needs Pillow: pip install pillow)

The logo's shapes are drawn from the numbers in logo.svg (a 48-unit square), four times larger than wanted
and then scaled down, so the edges are smooth; the text uses the dashboard's own Archivo font.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "python" / "nitro" / "brand"
OUT = ROOT / "installer"
SCALES = (100, 125, 150, 200, 250)
PANEL = (164, 314)  # Inno Setup's modern wizard image at 100%
SMALL = 55
SUPER = 4  # drawn this many times larger, then scaled down
DARK = "#141413"
YELLOW = "#FFD60A"
OLIVE = "#6B5A12"
LIGHT = "#F3F1EC"


def logo_numbers(svg: str) -> dict[str, Any]:
    """The logo's colours and shapes, read from logo.svg so the pictures follow it."""
    paths = [
        (
            re.search(r'd="([^"]+)"', tag)[1],  # type: ignore[index]
            (re.search(r'fill="([^"]+)"', tag) or ["", ""])[1],
            (re.search(r'stroke="([^"]+)"', tag) or ["", ""])[1],
        )
        for tag in re.findall(r"<path [^>]*/>", svg)
    ]
    circle = re.search(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)" fill="([^"]+)"', svg)
    rect = re.search(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="([^"]+)"', svg
    )
    width = re.search(r'stroke-width="([\d.]+)"', svg)
    if len(paths) < 4 or circle is None or rect is None or width is None:
        raise SystemExit("logo.svg does not have the shapes this script draws")
    return {
        "back": paths[0][1],
        "track": paths[1][2],
        "gauge": paths[2][2],
        "needle": [float(n) for n in re.findall(r"[\d.]+", paths[3][0])],
        "needle_fill": paths[3][1],
        "hub": (float(circle[1]), float(circle[2]), float(circle[3]), circle[4]),
        "bar": (float(rect[1]), float(rect[2]), float(rect[3]), float(rect[4]), rect[5]),
        "stroke": float(width[1]),
        "gauge_end": [float(n) for n in re.findall(r"[\d.]+", paths[2][0])][-2:],
    }


def _angle(x: float, y: float, cx: float = 24, cy: float = 27) -> float:
    """The angle Pillow's arcs use (degrees clockwise from 3 o'clock) of a point on the gauge."""
    return math.degrees(math.atan2(y - cy, x - cx)) % 360


def draw_logo(draw: Any, left: float, top: float, size: float, numbers: dict[str, Any], back: bool) -> None:
    """The logo in a `size` square at (left, top); with `back`, its dark notched square too."""
    unit = size / 48

    def at(x: float, y: float) -> tuple[float, float]:
        return left + x * unit, top + y * unit

    if back:
        draw.polygon([at(0, 0), at(36, 0), at(48, 12), at(48, 48), at(0, 48)], fill=numbers["back"])
    box = [at(24 - 15, 27 - 15), at(24 + 15, 27 + 15)]
    width = round(numbers["stroke"] * unit)
    start = _angle(13.39, 37.61)
    draw.arc(box, start, start + 270, fill=numbers["track"], width=width)
    end_x, end_y = numbers["gauge_end"]
    draw.arc(box, start, start + (_angle(end_x, end_y) - start) % 360, fill=numbers["gauge"], width=width)
    needle = numbers["needle"]
    draw.polygon([at(needle[i], needle[i + 1]) for i in range(0, 6, 2)], fill=numbers["needle_fill"])
    cx, cy, r, hub = numbers["hub"]
    draw.ellipse([at(cx - r, cy - r), at(cx + r, cy + r)], fill=hub)
    x, y, w, h, bar = numbers["bar"]
    draw.rectangle([at(x, y), at(x + w, y + h)], fill=bar)


def panel(scale: int, numbers: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    k = scale / 100 * SUPER
    width, height = round(PANEL[0] * k), round(PANEL[1] * k)
    image = Image.new("RGB", (width, height), DARK)
    draw = ImageDraw.Draw(image)
    draw_logo(draw, 14 * k, 34 * k, 62 * k, numbers, back=False)
    for offset, colour in ((0, YELLOW), (24, OLIVE)):  # two stripes rising to the right
        y = (182 + offset) * k
        draw.polygon(
            [(8 * k, y), (PANEL[0] * k, y - 84 * k), (PANEL[0] * k, y - 68 * k), (16 * k, y + 15 * k)],
            fill=colour,
        )
    font = BRAND / "fonts" / "Archivo.woff2"
    name = ImageFont.truetype(str(font), round(25 * k))
    by = ImageFont.truetype(str(font), round(13 * k))
    draw.text((15 * k, 252 * k), "NitroLearn", font=name, fill=LIGHT, anchor="ls")
    draw.text((16 * k, 276 * k), "by Raze", font=by, fill=YELLOW, anchor="ls")
    return image.resize(
        (round(PANEL[0] * scale / 100), round(PANEL[1] * scale / 100)), Image.Resampling.LANCZOS
    )


def small(scale: int, numbers: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    n = round(SMALL * scale / 100)
    image = Image.new("RGBA", (n * SUPER, n * SUPER), (0, 0, 0, 0))
    draw_logo(ImageDraw.Draw(image), 0, 0, n * SUPER, numbers, back=True)
    return image.resize((n, n), Image.Resampling.LANCZOS)


def main() -> int:
    numbers = logo_numbers((BRAND / "logo.svg").read_text(encoding="utf-8"))
    for scale in SCALES:
        for name, image in ((f"wizard-{scale}.png", panel(scale, numbers)), (f"wizard-small-{scale}.png",
                            small(scale, numbers))):  # fmt: skip
            image.save(OUT / name, optimize=True)
            print(OUT / name, image.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
