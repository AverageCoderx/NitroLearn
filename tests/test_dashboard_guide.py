"""The dashboard's "See more" explanations: one for every setting, shown in the page."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.config.loader import parse_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import describe, guide, reward_guide
from nitro.dashboard.api import Api
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]


def _paths(sections: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for section in sections:
        out |= {item["path"] for item in section["items"]} | _paths(section.get("sections", []))
    return out


def test_every_setting_has_a_see_more_text_in_its_parts() -> None:
    paths = _paths(describe.describe(NitroConfig(), "", locked=False))
    assert paths <= set(guide.MORE), sorted(paths - set(guide.MORE))
    for key, text in guide.MORE.items():
        assert text.startswith("What it does:"), key
        assert 40 <= len(text.split()) <= 320, key
    assert {"reward.weight", "reward.zero_sum", "start.weight", "start.random"} <= set(guide.MORE)


def test_the_explanations_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    RunFolder.create(tmp_path / "runs", "b", ROOT / "configs" / "presets" / "2v2_starter.toml")
    api = Api(ROOT, tmp_path / "runs")
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1400, "height": 900})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            open_bot(tab)
            tab.click("[data-tab=settings]")
            tab.fill("#settings-search", "discount half life")
            tab.click("[data-item='ppo.discount_half_life'] [data-help]")
            box = tab.wait_for_selector("[data-item='ppo.discount_half_life'] .guide")
            assert "What it does:" in box.inner_text() and "Watch out for:" in box.inner_text()
            tab.click("[data-tab=rewards]")
            tab.click("[data-act=rw-guide]")
            assert "Team spirit" in tab.inner_text(".modal")
            tab.click("[data-close]")
            tab.click("[data-tab=starts]")
            tab.click("[data-act=st-guide]")
            assert "Weight (share of games)" in tab.inner_text(".modal")
            tab.click("[data-close]")
            tab.click("[data-tab=rewards]")
            tab.click("[data-rwabout] >> nth=0")
            assert "What it does:" in tab.inner_text("[data-reward] .guide >> nth=0")
        finally:
            browser.close()
    assert errors == []


EXAMPLE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("module", [guide, reward_guide], ids=["settings", "rewards"])
def test_every_explanation_has_a_nerd_text_whose_examples_load(module: Any, tmp_path: Path) -> None:
    """Every See more text has a Nerd explanation (the same keys), each with at least one example between
    lines of three backticks; every settings-file example is a settings file the loader accepts."""
    assert set(module.NERD) == set(module.MORE), sorted(set(module.NERD) ^ set(module.MORE))
    for key, text in module.NERD.items():
        assert text.strip() and text.count("```") % 2 == 0, key
        examples = EXAMPLE.findall(text)
        assert examples, f"{key}: no example"
        for kind, example in examples:
            if kind == "toml":
                file = tmp_path / "example.toml"
                file.write_text(example, encoding="utf-8")
                parse_toml(example, NitroConfig, file=file, base_dir=tmp_path)  # raises, naming the line
