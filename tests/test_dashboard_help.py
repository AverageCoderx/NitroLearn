"""The Help page in the dashboard's window: the handbook opens and reads there (links between pages, to a
heading and to the web), the report for Discord is copied or shown, and a bot's Coach tab shows its plan and
hints."""

from __future__ import annotations

from pathlib import Path

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.dashboard.api import Api
from test_coach import healthy
from test_dashboard import ROOT, make_bot
from test_dashboard_look import opened


def test_the_handbook_reads_in_the_window_and_the_report_is_scrubbed(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    folder = make_bot(runs, "coached", saves=1)
    rows = [healthy(i, 120) for i in range(120)]
    for row in rows[100:]:
        row["entropy"] = 0.04
    header = list(rows[0])
    folder.metrics.write_text(
        ",".join(header) + "\n" + "".join(",".join(str(r[c]) for c in header) + "\n" for r in rows),
        encoding="utf-8",
    )
    api = Api(ROOT, runs)
    opened_urls: list[str] = []
    api.open_url = lambda url: opened_urls.append(url) or {"opened": True}  # type: ignore[method-assign]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            tab.click("#band .nav [data-go=help]")
            tab.wait_for_selector("#guide-reader h2 >> text=The NitroLearn handbook")
            titles = tab.locator(".guide-list .guide-nav").all_text_contents()
            assert titles[:3] == ["The handbook", "Getting started", "Your first bot"]
            assert "The training coach" in titles and "Troubleshooting" in titles
            assert "Join the Vanta Discord server" in tab.text_content("#main"), "the support part stays"

            tab.click(".guide-list >> text=The training coach")
            tab.wait_for_selector("#guide-reader h3 >> text=The plan: four stages")
            assert tab.locator("#guide-reader ol li").count() == 4, "a numbered list"
            assert tab.locator("#guide-reader code").count() > 0, "code spans"
            tab.click("#guide-reader [data-href='rewards.md']")
            tab.wait_for_selector("#guide-reader h2 >> text=Rewards")
            tab.click("#guide-reader [data-href='../your_own_rewards.md']")
            tab.wait_for_selector("#guide-reader h2 >> text=Your own rewards, in Rust")
            tab.click(".guide-list >> text=Rewards")
            tab.click("#guide-reader [data-href='../rewards.md']")
            tab.wait_for_selector("#guide-reader h2 >> text=The rewards that come with NitroLearn")
            assert tab.locator("#guide-reader table").count() > 0, "tables"

            tab.click(".guide-list >> text=Troubleshooting")
            tab.wait_for_selector("#guide-reader h3 >> text=Training")
            tab.click("#guide-reader [data-href='rewards.md#farming-paid-without-playing']")
            tab.wait_for_selector("#md-farming-paid-without-playing")
            tab.click(".guide-list >> text=Troubleshooting")
            tab.wait_for_selector("#guide-reader h3 >> text=Installing")
            tab.click("#guide-reader [data-href='https://discord.gg/bYP3kYtryg'] >> nth=0")
            tab.wait_for_function("true")
            assert opened_urls[-1:] == ["https://discord.gg/bYP3kYtryg"], "a web page opens in the browser"

            tab.select_option("[data-help-bot]", "coached")
            tab.click("[data-act=help-report-show]")
            shown = tab.wait_for_selector("pre >> text=NitroLearn report").text_content()
            assert "**Bot: coached**" in shown and "Check this PC" in shown and "not run yet" in shown
            assert str(ROOT) not in shown and str(tmp_path) not in shown
            tab.click("[data-act=help-report]")
            tab.wait_for_selector(".toast >> text=report")

            open_bot(tab, "coached")
            tab.click(".tabs [data-tab=coach]")
            tab.wait_for_selector("#coach-page >> text=Run a skill check")
            page = tab.text_content("#coach-page")
            assert "Basics" in page and "Shooting" in page and "Air play" in page
            assert "It has become almost certain of every choice" in page and "ppo.entropy" in page
            assert "Suggestions only" in page
            tab.click("#coach-page [data-act=guide-open]")
            tab.wait_for_selector("#guide-reader h2 >> text=The training coach")
            assert errors == []
        finally:
            browser.close()
