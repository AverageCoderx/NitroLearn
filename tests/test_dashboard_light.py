"""A light, fast dashboard: a bot's metrics are found by the core and read row by row (the same numbers as a
whole read), one bot's rows are kept at a time, the page is sent only the parts that changed, and a tab click
keeps the page's frame."""

from __future__ import annotations

import csv
import math
import random
from array import array
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.dashboard import bots
from nitro.dashboard.api import Api
from test_dashboard import ROOT, make_bot

COLUMNS = ["agent_steps", "iteration", "steps_per_second", "reward_per_step", "raw_a", "paid_a", "entropy"]


def write_rows(file: Path, rows: int, seed: int = 0) -> None:
    pick = random.Random(seed)
    lines = [",".join(COLUMNS)]
    for i in range(1, rows + 1):
        cells = [str(i * 1000), str(i), str(41000 + i % 7)]
        cells += [repr(pick.random()) if pick.random() > 0.05 else "" for _ in COLUMNS[3:]]
        lines.append(",".join(cells))
    lines.insert(rows // 2, '1,"2,3",4')  # rows csv refuses (a kill, a quoted comma): never counted
    lines.insert(rows // 3, "5,6")
    file.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bots, "_CACHE", {})
    monkeypatch.setattr(bots, "_SETTINGS", {})
    monkeypatch.setattr(bots, "_PARSED", None)


def test_every_number_is_the_whole_reads_and_the_charts_keep_single_precision(tmp_path: Path) -> None:
    folder = make_bot(tmp_path / "runs", "b")
    write_rows(folder.metrics, 20_000)
    with folder.metrics.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f)][1:]
    rows = [r for r in rows if len(r) == len(COLUMNS)]
    m = bots.metrics(folder)
    assert m.count == len(rows) and m.last == rows[-1]
    charted = m.rows_from(0)
    assert charted[-1] == m.count - 1 and len(charted) < m.count, "older rows thinned for the charts"
    for r in range(0, m.count, 37):
        for c, column in enumerate(COLUMNS):
            got, want = m.value(column, r), number(rows[r][c])
            if column == "iteration":  # never read by the dashboard
                want = None
            elif column == "entropy":  # a chart column: kept in single precision, on its rows only
                want = None if r not in set(charted) or want is None else array("f", (want,))[0]
            assert got == want, (column, r)


def test_one_bots_rows_are_kept_at_a_time(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    one, two = make_bot(runs, "one"), make_bot(runs, "two")
    write_rows(one.metrics, 9000)
    write_rows(two.metrics, 9000, seed=1)
    kept = bots.metrics(one)
    first = [kept.value("reward_per_step", r) for r in range(0, kept.count, 3)]
    assert 0 < len(kept.parsed) <= bots._ROWS_KEPT
    other = bots.metrics(two)
    other.value("reward_per_step", 5)
    assert not kept.parsed, "the other bot's rows are let go"
    assert [kept.value("reward_per_step", r) for r in range(0, kept.count, 3)] == first, (
        "and read again alike"
    )


def test_calls_on_their_own_threads_read_and_count_alike(tmp_path: Path) -> None:
    """The window runs each call in a thread of its own: reads of two bots' rows while one file grows give the
    written values (a prefetch let go of rows another call was about to read: a KeyError), and the grown
    file's rows are counted once."""
    import sys
    import threading
    import time

    runs = tmp_path / "runs"
    folders = [make_bot(runs, "one"), make_bot(runs, "two")]
    width = 6
    for folder in folders:
        lines = [",".join(f"c{i}" for i in range(width))]
        lines += [",".join(str(r * 1.5 + i) for i in range(width)) for r in range(9000)]
        folder.metrics.write_text("\n".join(lines) + "\n")
    errors: list[object] = []
    stop = threading.Event()

    def read(seed: int) -> None:
        pick = random.Random(seed)
        try:
            while not stop.is_set():
                m = bots.metrics(folders[pick.randrange(2)])
                for _ in range(40):
                    row = pick.randrange(m.count)
                    if m.numbers(row)[2] != row * 1.5 + 2:
                        errors.append(("value", row))
        except Exception as err:  # any failure is the test's
            errors.append(err)

    written = 9000

    def grow() -> None:
        nonlocal written
        while not stop.is_set():
            with folders[0].metrics.open("a") as handle:
                for _ in range(5):
                    handle.write(",".join(str(written * 1.5 + i) for i in range(width)) + "\n")
                    written += 1
            bots.metrics(folders[0])

    threads = [threading.Thread(target=read, args=(k,)) for k in range(4)] + [threading.Thread(target=grow)]
    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # threads switch often, so a short window between two steps is reached
    try:
        for t in threads:
            t.start()
        time.sleep(3)
    finally:
        stop.set()
        for t in threads:
            t.join()
        sys.setswitchinterval(switch)
    assert errors == []
    assert bots.metrics(folders[0]).count == written


def test_parts_the_page_holds_are_not_sent_again_until_they_change(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "b")
    make_bot(runs, "other")
    api = Api(ROOT, runs)
    first = api.bot("b")
    assert first["same"] == [] and set(first["digests"]) == {"settings"}
    again = api.bot("b", first["digests"])
    assert again["same"] == ["settings"] and "settings" not in again
    text = folder.settings.read_text(encoding="utf-8")
    folder.settings.write_text(text.replace("policy_lr = 3e-4", "policy_lr = 2e-4"), encoding="utf-8")
    changed = api.bot("b", first["digests"])
    assert "settings" in changed and changed["settings"] != first["settings"], "a changed file is sent anew"
    kinds = api.reward_kinds()
    assert kinds["catalog"] and set(kinds["digests"]) == {"catalog", "blocks", "help", "guide"}
    other = api.rewards("other", 60.0, kinds["digests"])
    assert sorted(other["same"]) == ["blocks", "catalog", "guide", "help"], (
        "the kinds are the same for any bot"
    )
    assert "catalog" not in other and other["rewards"]


def test_a_tab_click_keeps_the_bots_head_and_a_refresh_is_sent_only_what_changed(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    write_rows(make_bot(runs, "b").metrics, 300)
    api = Api(ROOT, runs)
    asked: list[tuple[str, list[Any]]] = []

    def py(method: str, args: list[Any]) -> Any:
        asked.append((method, args))
        return getattr(api, method)(*args)

    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1300, "height": 900})
            errors: list[str] = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", py)
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            open_bot(tab, "b")
            tab.wait_for_selector("#bothead")
            tab.evaluate("document.querySelector('#bothead').__mark = 1")
            tab.click(".tabs [data-tab=settings]")
            tab.click(".tabs [data-tab=rewards]")
            tab.wait_for_selector("#rw-list section")
            assert tab.evaluate("document.querySelector('#bothead').__mark") == 1, (
                "the head was not drawn anew"
            )
            asked.clear()
            tab.evaluate("refreshBot()")
            tab.wait_for_function("!S.refreshing")
            sent = [args for method, args in asked if method == "bot"]
            assert sent and list(sent[-1][1]) == ["settings"], "the page says what it holds"
            assert tab.evaluate("S.data.settings.length") > 0, "and keeps it"
            assert any(method == "reward_kinds" for method, _ in [*asked]) or tab.evaluate("!!S.rwKinds")
            assert errors == []
        finally:
            browser.close()


def test_a_bots_page_loads_no_pytorch(tmp_path: Path) -> None:
    """A bot's page with a replay copy (its progress file written) and charts: PyTorch alone is 0.5-2 GB."""
    import subprocess
    import sys

    runs = tmp_path / "runs"
    folder = make_bot(runs, "b", saves=1)
    write_rows(folder.metrics, 300)
    (folder.root / "pretrained").mkdir()
    (folder.root / "pretrained" / "live.json").write_text('{"learned": 1, "limit": 2}', encoding="utf-8")
    code = (
        "import sys; from pathlib import Path; from nitro.dashboard.api import Api; "
        f"api = Api(Path({str(ROOT)!r}), Path({str(runs)!r})); "
        "out = [api.bot('b'), api.bot_charts('b'), api.bot_tools('b'), api.rewards('b')]; "
        "print(out[0]['copy']['learned'], 'torch' in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.split() == ["1", "False"], done.stderr
