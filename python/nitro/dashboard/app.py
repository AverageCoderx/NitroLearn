"""`nitro dashboard`: the dashboard in its own window (pywebview, on Windows' own Edge engine). The page is
handed to the window as text and talks to Python through pywebview's bridge: nothing listens on a port."""

from __future__ import annotations

import html
import json
from pathlib import Path

from nitro import brand
from nitro.dashboard import winapp
from nitro.dashboard.api import Api

WEB = Path(__file__).with_name("web")
TITLE = f"{brand.NAME} {brand.CREDIT}"
BACKGROUND = "#171614"  # the dark theme's page colour (web/00_tokens.css), shown until the page has painted
_STYLE_MARK = "/*STYLE*/"
_SCRIPT_MARK = "/*SCRIPT*/"
_DATA_MARK = "/*BOOT*/null"
_FONTS_MARK = "/*FONTS*/"
_LOGO_MARK = "<!--LOGO-->"
_NAME_MARK = "<!--NAME-->"
_CREDIT_MARK = "<!--CREDIT-->"


def style() -> str:
    """The page's style: the parts in `web/`, joined in name order."""
    return _joined("css")


def script() -> str:
    """The page's script: the parts in `web/`, joined in name order into one script (they share names)."""
    return _joined("js")


def page_source() -> str:
    """The page as its files make it, before the fonts and the first data are put in: `web/page.html` with
    the brand's logo, name and credit where they are marked, and the style and the script inside."""
    shell = (WEB / "page.html").read_text(encoding="utf-8")
    for mark, text in ((_LOGO_MARK, brand.logo()), (_NAME_MARK, brand.NAME), (_CREDIT_MARK, brand.CREDIT)):
        shell = shell.replace(mark, text if mark == _LOGO_MARK else html.escape(text))
    return shell.replace(_STYLE_MARK, style(), 1).replace(_SCRIPT_MARK, script(), 1)


def page(api: Api) -> str:
    """The dashboard page with its fonts inside (it works offline) and what it shows first (so it paints at
    once, before the bridge is up)."""
    boot = json.dumps(api.home(), allow_nan=False).replace("</", "<\\/")
    return page_source().replace(_FONTS_MARK, brand.font_faces(), 1).replace(_DATA_MARK, boot, 1)


def _joined(kind: str) -> str:
    return "\n".join(part.read_text(encoding="utf-8") for part in sorted(WEB.glob(f"[0-9][0-9]_*.{kind}")))


def run(project: Path, runs: Path, debug: bool = False) -> int:
    try:
        import webview
    except ImportError:
        print(
            "The dashboard needs pywebview. Install it with:  "
            ".venv\\Scripts\\python -m pip install pywebview tomlkit"
        )
        return 1
    # One dashboard per runs folder (two would act on the same bots): a second start brings the open one
    # to the front instead.
    instance = winapp.claim(runs)
    if instance is None:
        print("The dashboard is already open: it is brought to the front.")
        winapp.summon(runs)
        return 0
    try:
        winapp.set_app_id()  # before the window exists, so the taskbar shows NitroLearn, not Python
        api = Api(project, runs)
        window = webview.create_window(
            TITLE,
            html=page(api),
            js_api=api,
            width=1440,
            height=920,
            min_size=(900, 640),
            background_color=BACKGROUND,
            text_select=True,  # the console, banners and errors can be selected and copied
        )
        api._window = window
        winapp.dress(window, instance)
        winapp.start(webview, debug)
    finally:
        instance.release()
    return 0
