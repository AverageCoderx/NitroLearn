"""NitroLearn's look, shared by the dashboard and the match viewer: the logo, the credit line and the fonts
(Archivo and JetBrains Mono, SIL Open Font License 1.1: `fonts/LICENSES.txt`), embedded into a page so it
works offline."""

from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
NAME = "NitroLearn"
CREDIT = "by Raze"
# family, file, style, weights, widths (a variable font covers the whole range)
_FACES = (
    ("Archivo", "Archivo.woff2", "normal", "100 900", "62% 125%"),
    ("Archivo", "Archivo-Italic.woff2", "italic", "100 900", "62% 125%"),
    ("JetBrains Mono", "JetBrainsMono.woff2", "normal", "400 700", ""),
)


def logo() -> str:
    """The mark as SVG text (a boost gauge at full on a cut-corner plate)."""
    return (HERE / "logo.svg").read_text(encoding="utf-8").strip()


def font_faces() -> str:
    """The @font-face rules of the brand fonts, with the font files inside them."""
    rules = []
    for family, file, style, weights, widths in _FACES:
        data = base64.b64encode((FONTS / file).read_bytes()).decode("ascii")
        stretch = f"font-stretch: {widths}; " if widths else ""
        rules.append(
            f"@font-face {{ font-family: '{family}'; font-style: {style}; font-weight: {weights}; {stretch}"
            f"font-display: swap; src: url(data:font/woff2;base64,{data}) format('woff2'); }}"
        )
    return "\n".join(rules)
