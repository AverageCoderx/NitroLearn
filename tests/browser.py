"""A headless Chromium for the tests that open a page, through Playwright's own browser. A test that needs
one skips, saying what is missing, where Playwright or its Chromium is not installed."""

from __future__ import annotations

from typing import Any

import pytest

INSTALL = "install them with: python -m pip install playwright && python -m playwright install chromium"


def playwright_or_skip() -> Any:
    """Playwright's `sync_api` module, or the test is skipped."""
    return pytest.importorskip(
        "playwright.sync_api", reason=f"no Playwright to open the page with ({INSTALL})"
    )


def open_chromium(playwright: Any) -> Any:
    """A headless Chromium started by `playwright` (what `sync_playwright()` gives), or the test is skipped.
    It uses no proxy: the pages under test are handed over as text and fetch nothing."""
    try:
        return playwright.chromium.launch(args=["--no-proxy-server"])
    except Exception as err:
        try:  # an installed Google Chrome does as well
            return playwright.chromium.launch(channel="chrome", args=["--no-proxy-server"])
        except Exception:
            reason = (str(err).strip().splitlines() or [type(err).__name__])[0]
            pytest.skip(f"no Chromium to open the page in ({reason}; {INSTALL})")


def open_bot(tab: Any, name: str = "") -> None:
    """In the dashboard's page: goes to the Garage and opens the bot called `name` (the first listed when
    none is named)."""
    tab.click(".nav [data-go=home]")
    tab.click(f"#main [data-bot='{name}']" if name else "#main [data-bot] >> nth=0")
