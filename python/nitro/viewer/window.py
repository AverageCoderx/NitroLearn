"""The match viewer in its own window (pywebview, on the system's own web engine), for a live game.

The page is a file opened by its `file://` address and asks for what was played through the window's
bridge, a call from the page into this process. Nothing is served: no HTTP server, no socket, no port.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nitro.viewer import feed


class NoWindow(Exception):
    """The system could not make a window; the message says why."""


class Bridge:
    """What the page may call. pywebview offers the page every public name of this object, so the store
    stays private."""

    def __init__(self, chunks: feed.Chunks) -> None:
        self._chunks = chunks

    def viewer_poll(self, cursor: int = 0) -> dict[str, Any]:
        """What the game has played since `cursor` (`feed.Chunks.after`)."""
        return self._chunks.after(int(cursor or 0))


def available() -> bool:
    """Whether a window can be made here (pywebview is installed)."""
    return importlib.util.find_spec("webview") is not None


def show(page: Path, title: str, chunks: feed.Chunks, closed: Callable[[], None] | None = None) -> None:
    """Opens the viewer page `page` (a file) in a window fed from `chunks` and returns once the window is
    closed; `closed` is called when it closes. Must be called on the main thread. Raises NoWindow when the
    system cannot make one."""
    import webview
    from webview.errors import WebViewException

    window = webview.create_window(
        title,
        # A file address: pywebview starts its own local web server for a plain path, never for this.
        url=page.resolve().as_uri(),
        js_api=Bridge(chunks),
        width=1440,
        height=900,
        min_size=(800, 520),
        background_color="#141413",
    )
    if window is not None and closed is not None:
        window.events.closed += closed
    try:
        webview.start(http_server=False)
    except WebViewException as err:
        raise NoWindow(str(err)) from err
