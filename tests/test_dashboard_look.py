"""The dashboard's Garage look and shell: the colours' contrast, each bot's race number and livery, the top
band with the brand's logo and credit, the Open RLBot help and the Check this PC page."""

from __future__ import annotations

import colorsys
import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest

from browser import open_bot, open_chromium, playwright_or_skip
from nitro import brand
from nitro.cli import main
from nitro.dashboard import app, checks, livery
from nitro.dashboard.api import Api
from test_dashboard import ROOT, make_bot

TOKENS = app.WEB / "00_tokens.css"
# The two teams' colours in the dark and the light theme: no bot's livery may be taken for a team.
TEAM_COLOURS = ("#5B97FF", "#1B58D6", "#FF8A2B", "#C24E00")
DOCTOR_JSON = {
    "version": "1.0.0",
    "verdict": "1 problem must be fixed before training.",
    "checks": [
        {"name": "Python", "status": "OK", "detail": "3.11.9", "fix": ""},
        {"name": "GPU", "status": "WARN", "detail": "PyTorch cannot see an NVIDIA GPU", "fix": "Update it."},
        {"name": "Collision meshes", "status": "FAIL", "detail": "not found", "fix": "Copy them."},
    ],
}


def themes() -> dict[str, dict[str, str]]:
    """The colours of each theme in 00_tokens.css, by name: the light theme changes some of the dark one's."""
    text = TOKENS.read_text(encoding="utf-8")

    def colours(opening: str) -> dict[str, str]:
        return dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", text.split(opening)[1].split("}")[0]))

    dark = {**colours("\n:root {"), **colours(':root[data-theme="dark"] {')}
    return {"dark": dark, "light": {**dark, **colours(':root[data-theme="light"] {')}}


def hue(colour: str) -> tuple[float, float]:
    """A colour's hue in degrees and its saturation (0 to 1)."""
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s


def opened(browser: Any, api: Api, errors: list[str], **page_options: Any) -> Any:
    """The dashboard's page in a new tab, talking to `api`."""
    tab = browser.new_page(**page_options)
    tab.on("pageerror", lambda e: errors.append(str(e)))
    tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
    tab.add_init_script("window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});")
    tab.goto("about:blank")
    tab.set_content(app.page(api))
    return tab


def test_text_reads_on_every_surface_in_both_themes() -> None:
    pairs = [
        *(("text", surface) for surface in ("bg", "panel", "raised", "block")),
        *(("dim", surface) for surface in ("bg", "panel", "raised", "block")),
        *(("good-text", surface) for surface in ("bg", "panel", "raised")),
        *(("bad-text", surface) for surface in ("bg", "panel", "raised")),
        *(("link", surface) for surface in ("bg", "panel", "raised")),
        ("on-yellow", "yellow"),
        ("on-strong", "strong"),
        ("on-good", "good"),
        ("on-bad", "bad"),
        ("band-text", "band"),
        ("band-dim", "band"),
        ("band-text", "band-raised"),
        ("yellow", "band"),
    ]
    for theme, colours in themes().items():
        assert colours["yellow"] == "#FFD60A" and colours["on-yellow"] == "#141413", "one yellow in both"
        for text, surface in pairs:
            ratio = livery.contrast(colours[text], colours[surface])
            assert ratio >= 4.5, f"{theme}: --{text} on --{surface} is only {ratio:.2f} to 1"
        for series in ("blue", "orange", "good"):  # chart lines, on the page and on panels
            for surface in ("bg", "panel"):
                assert livery.contrast(colours[series], colours[surface]) >= 3, (theme, series, surface)
        assert livery.contrast(colours["field-rule"], colours["panel"]) >= 3, "a field's edge shows"


def test_a_bots_livery_is_picked_from_its_name_and_stays_the_same() -> None:
    assert 8 <= len(livery.PALETTE) <= 10
    for main_colour, second in livery.PALETTE:
        assert livery.contrast(livery.ink(main_colour), main_colour) >= 4.5, (
            f"its number reads on {main_colour}"
        )
        assert livery.contrast(main_colour, second) >= 1.5
        for colour in (main_colour, second):
            degrees, saturation = hue(colour)
            for team in TEAM_COLOURS:
                away = abs((degrees - hue(team)[0] + 180) % 360 - 180)
                assert colour != team and (saturation < 0.15 or away >= 15), f"{colour} looks like a team's"
    for one, other in itertools.combinations([pair[0] for pair in livery.PALETTE], 2):
        channels = zip(*(tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in (one, other)), strict=True)
        assert sum((a - b) ** 2 for a, b in channels) ** 0.5 > 80, f"{one} and {other} are too alike"
    picked = livery.pick("striker-1v1")
    assert picked == livery.pick("striker-1v1") and set(picked) == {"number", "main", "contrast", "ink"}
    assert re.fullmatch(r"\d\d", picked["number"]) and picked["number"] != "00"
    assert (picked["main"], picked["contrast"]) in livery.PALETTE and picked["ink"] == livery.ink(
        picked["main"]
    )
    names = [f"bot-{i}" for i in range(200)]
    assert len({livery.pick(name)["main"] for name in names}) == len(livery.PALETTE), "every pair is used"
    assert len({livery.pick(name)["number"] for name in names}) > 60, "the numbers spread"


def test_a_livery_is_kept_with_the_bot_once_changed_and_checked_first(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = make_bot(runs, "painted")
    api = Api(ROOT, runs)
    assert api.home()["bots"][0]["livery"] == livery.pick("painted") == api.bot("painted")["livery"]
    assert not (folder.root / livery.FILE).exists(), "nothing is written until it is changed"
    changed = api.set_livery("painted", "7", "#0f6b3e", "#FFD60A")
    assert changed == {"livery": {"number": "07", "main": "#0F6B3E", "contrast": "#FFD60A", "ink": "#F3F1EC"}}
    assert json.loads((folder.root / livery.FILE).read_text(encoding="utf-8")) == {
        "number": "07",
        "main": "#0F6B3E",
        "contrast": "#FFD60A",
    }
    assert Api(ROOT, runs).home()["bots"][0]["livery"] == changed["livery"], (
        "kept when the window opens again"
    )
    for number, main_colour, second, said in (
        ("100", "#0F6B3E", "#FFD60A", "one or two digits"),
        ("x", "#0F6B3E", "#FFD60A", "one or two digits"),
        (True, "#0F6B3E", "#FFD60A", "one or two digits"),
        ("7", "green", "#FFD60A", "main colour"),
        ("7", "#0F6B3E", "#FFD60", "second colour"),
        ("7", "#0F6B3E", None, "second colour"),
        ("7", "#0F6B3E", "#0F6B3F", "too alike"),
    ):
        assert said in api.set_livery("painted", number, main_colour, second)["error"]
    assert "no bot called ghost" in api.set_livery("ghost", "7", "#0F6B3E", "#FFD60A")["error"]
    assert api.bot("painted")["livery"] == changed["livery"], "a refused change changes nothing"
    (folder.root / livery.FILE).write_text('{"number": "7", "main": 5}', encoding="utf-8")
    assert api.bot("painted")["livery"] == livery.pick("painted"), "a broken file: picked from its name again"


def test_check_this_pc_reads_what_nitro_doctor_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shown = checks.read("a warning line first\n" + json.dumps(DOCTOR_JSON) + "\n")
    assert [(c["name"], c["state"]) for c in shown["checks"]] == [
        ("Python", "ok"),
        ("GPU", "warning"),
        ("Collision meshes", "problem"),
    ]
    assert shown["checks"][2] == {
        "name": "Collision meshes",
        "state": "problem",
        "message": "not found",
        "fix": "Copy them.",
    }
    assert (shown["problems"], shown["warnings"], shown["command"]) == (1, 1, "nitro doctor")
    assert shown["verdict"] == "1 problem must be fixed before training."
    for printed in ("", "Traceback (most recent call last):\n  boom", '{"checks": [{"name": "x"}]}'):
        assert "stopped before they were done" in checks.read(printed, "it broke")["error"]
    assert "it broke" in checks.read("", "it broke")["error"]

    main(["doctor", "--json"])  # exit code 1 only with a failed check: here it is the output that counts
    report = json.loads(capsys.readouterr().out)
    names = {check["name"] for check in report["checks"]}
    assert {"Python", "Graphics card", "Rust core", "Disk space"} <= names
    assert all(check["status"] in ("OK", "WARN", "FAIL") for check in report["checks"]) and report["verdict"]

    real = Api(ROOT, tmp_path / "runs").check_pc()  # the command itself, as its own process
    assert "error" not in real, real
    assert [check["name"] for check in real["checks"]] == [check["name"] for check in report["checks"]]
    assert all(check["state"] in ("ok", "warning", "problem") for check in real["checks"])
    assert all(check["fix"] for check in real["checks"] if check["state"] != "ok"), "each one says how to fix"


def test_the_page_is_one_text_with_the_brand_inside() -> None:
    source = app.page_source()
    assert app.TITLE == "NitroLearn by Raze" and f"<title>{app.TITLE}</title>" in source
    assert source.count(brand.logo()) == 1 and "<!--" not in source.split("<body>")[1], "every mark filled"
    assert f'<span class="by">{brand.CREDIT}</span>' in source
    assert (
        "For offline training and RLBot matches only. Never use a bot online against real players." in source
    )
    assert f"{brand.NAME} {brand.CREDIT} · free for non-commercial use" in source
    page = source.replace("/*FONTS*/", brand.font_faces())
    fetched = re.findall(
        r"""(?:src|href)\s*=\s*["']?(?:https?:)?//|url\(\s*["']?(?:https?:)?//|@import""", page
    )
    assert fetched == [], "the page fetches nothing: no network, no files beside it"
    style = app.style()
    assert not re.search(r"border-radius:\s*[1-9]", style), "square corners"
    assert not re.search(r"(?:text-shadow|backdrop-filter|filter:\s*(?:blur|drop-shadow))", style), (
        "no glow or glass"
    )
    for shadow in re.findall(r"box-shadow:\s*([^;}]+)", style):
        assert re.fullmatch(r"0 0 0 \d+px #[0-9A-Fa-f]{6}", shadow.strip()), f"a drop shadow: {shadow}"


def test_the_band_the_plates_and_the_theme_switch_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown", saves=1)
    make_bot(runs, "other")
    api = Api(ROOT, runs)
    api.set_livery("shown", 7, "#0F6B3E", "#FFD60A")
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, viewport={"width": 1440, "height": 920})
            assert tab.title() == "NitroLearn by Raze"
            assert (
                tab.locator("#band .lockup svg").count() == 1 and tab.text_content("#band .by") == "by Raze"
            )
            assert tab.locator("#band .nav button").all_text_contents() == [
                *("Garage", "Resources", "Check this PC", "App settings", "Help"),
            ]
            assert tab.get_attribute("#band .nav [data-go=home]", "aria-current") == "page"
            # Help and ideas: support, bug reports and ideas on the Vanta Discord server, said plainly.
            tab.click("#band .nav [data-go=help]")
            tab.wait_for_selector("text=Join the Vanta Discord server")
            helps = tab.text_content("#main")
            reasons = ("Vanta Discord server", "Get help", "Report a bug", "Suggest an idea")
            assert all(w in helps for w in reasons)
            assert "https://discord.gg/bYP3kYtryg" in helps
            tab.click("#band .nav [data-go=home]")
            tab.wait_for_selector(".botcard[data-bot=shown]")
            assert "Never use a bot online against real players" in tab.text_content("#foot")
            band = tab.evaluate("getComputedStyle(document.querySelector('#band')).backgroundColor")
            plate = tab.locator(".botcard[data-bot=shown] .plate")
            assert plate.text_content() == "07"
            assert plate.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(15, 107, 62)"
            assert plate.evaluate("el => getComputedStyle(el).color") == "rgb(243, 241, 236)"
            other = livery.pick("other")
            assert tab.text_content(".botcard[data-bot=other] .plate") == other["number"]
            assert tab.evaluate("plate({}, 40)").count("--size:40px") == 1, "a bot without one still draws"

            open_bot(tab, "shown")
            tab.wait_for_selector("#bothead .plate >> text=07")
            assert tab.text_content("#bothead .crumbs") == "Garage / shown"
            assert tab.get_attribute("#band .nav [data-go=home]", "aria-current") == "page", (
                "still the Garage"
            )
            tab.click("#bothead .crumbs [data-go=home]")
            tab.wait_for_selector(".botcard[data-bot=shown]")

            assert tab.get_attribute("html", "data-theme") == "dark"
            tab.click("#theme-switch")
            tab.wait_for_function("document.documentElement.dataset.theme === 'light'")
            assert api.home()["prefs"]["theme"] == "light", "remembered"
            assert tab.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(233, 232, 228)"
            assert tab.evaluate("getComputedStyle(document.querySelector('#band')).backgroundColor") != (
                "rgb(233, 232, 228)"
            ), "the band stays black"
            assert band == "rgb(12, 12, 11)"
            tab.click("#band .nav [data-go=prefs]")
            tab.click("[data-seg=theme] [data-v=dark]")
            tab.wait_for_function("document.documentElement.dataset.theme === 'dark'")
            assert tab.get_attribute("#band .nav [data-go=prefs]", "aria-current") == "page"

            rounded = tab.evaluate(
                "[...document.querySelectorAll('*')].filter(el => { const s = getComputedStyle(el); "
                "return parseFloat(s.borderTopLeftRadius) > 0 || s.boxShadow !== 'none'; }).length"
            )
            assert rounded == 0, "no rounded corner and no shadow on the page"
            tab.set_viewport_size({"width": 900, "height": 640})
            assert tab.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (
                "the band wraps at the smallest window, nothing runs off the side"
            )
            reopened = opened(browser, api, errors)
            assert reopened.get_attribute("html", "data-theme") == "dark", "the stored look opens"
        finally:
            browser.close()
    assert errors == []


def test_open_rlbot_explains_itself_until_rlbot_is_chosen(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown")
    api = Api(ROOT, runs)
    program = tmp_path / "RLBot.exe"
    program.write_bytes(b"")
    asked: list[str] = []
    started: list[str] = []

    def choose() -> dict[str, Any]:  # in place of the window's file dialog and of starting the program
        asked.append("where")
        api.set_pref("rlbot_program", str(program))
        return {"opened": str(program)}

    def start() -> dict[str, Any]:
        chosen = api.home()["prefs"].get("rlbot_program", "")
        if not chosen:
            return {"need_program": True}
        started.append(chosen)
        return {"opened": chosen}

    api.choose_rlbot = choose  # type: ignore[method-assign]
    api.open_rlbot = start  # type: ignore[method-assign]
    assert Api(ROOT, runs).open_rlbot() == {"need_program": True}, "nothing chosen yet: it asks"
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#band [data-go=rlbot]")
            tab.wait_for_selector(".modal >> text=Open RLBot")
            said = tab.text_content(".modal")
            assert "real Rocket League match" in said and "offline" in said, "what RLBot is for"
            assert "where RLBot's program is" in said and "rlbot.org" in said
            assert tab.locator(".modal a").count() == 0, (
                "its site is text, not a link the window would follow"
            )
            tab.click(".modal [data-close]")
            assert asked == [] and not tab.is_visible(".modal")
            tab.click("#band [data-go=rlbot]")
            tab.click(".modal [data-act=rlbot-choose-close]")
            tab.wait_for_selector(".toast >> text=RLBot is remembered")
            assert asked == ["where"] and not tab.is_visible(".modal")
            tab.click("#band [data-go=rlbot]")
            tab.wait_for_selector(".toast >> text=Opening RLBot")
            assert started == [str(program)] and not tab.is_visible(".modal"), "chosen: it just opens"
            tab.click("#band .nav [data-go=prefs]")
            tab.wait_for_selector(f"#main >> text={program.name}")
        finally:
            browser.close()
    assert errors == []


def test_the_check_this_pc_page_lists_each_check_with_its_fix(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown")
    api = Api(ROOT, runs)
    runs_of_it: list[int] = []

    def check() -> dict[str, Any]:
        runs_of_it.append(1)
        report = dict(DOCTOR_JSON)
        if len(runs_of_it) > 1:  # fixed by the second look
            report["checks"] = [{**c, "status": "OK", "fix": ""} for c in DOCTOR_JSON["checks"]]  # type: ignore[union-attr]
            report["verdict"] = "All required checks passed."
        return checks.read(json.dumps(report))

    api.check_pc = check  # type: ignore[method-assign]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#band .nav [data-go=doctor]")
            tab.wait_for_selector(".check-row >> nth=2")
            assert tab.get_attribute("#band .nav [data-go=doctor]", "aria-current") == "page"
            rows = tab.locator(".check-row")
            assert [rows.nth(i).get_attribute("data-state") for i in range(3)] == ["ok", "warning", "problem"]
            assert [rows.nth(i).locator(".tag").text_content() for i in range(3)] == [
                "OK",
                "Warning",
                "Problem",
            ]
            assert (
                "Arena files (collision meshes)" in rows.nth(2).text_content()
                and "not found" in rows.nth(2).text_content()
            )
            assert "Copy them." in rows.nth(2).locator(".fix").text_content()
            assert rows.nth(0).locator(".fix").count() == 0, "nothing to fix where all is well"
            assert "1 problem must be fixed before training." in tab.text_content("#main .banner.bad")
            assert tab.text_content("#main .command") == "nitro doctor"
            tab.click("[data-act=doctor-run]")
            tab.wait_for_selector("#main .banner >> text=All required checks passed.")
            assert tab.locator(".check-row[data-state=ok]").count() == 3 and len(runs_of_it) == 2
            tab.click("#band .nav [data-go=home]")
            tab.click("#band .nav [data-go=doctor]")
            tab.wait_for_selector(".check-row >> nth=2")
            assert len(runs_of_it) == 2, "coming back shows the last look; the button looks again"
        finally:
            browser.close()
    assert errors == []


def test_no_two_parts_of_the_page_declare_the_same_name() -> None:
    """The parts are joined into one script: a function or constant declared twice at the top level would
    silently replace the other part's."""
    import re
    from collections import defaultdict

    from nitro.dashboard import app

    declared: defaultdict[str, list[str]] = defaultdict(list)
    for part in sorted((app.WEB).glob("[0-9][0-9]_*.js")):
        for line in part.read_text(encoding="utf-8").splitlines():
            found = re.match(
                r"(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)",
                line,
            )
            if found:
                declared[found.group(1) or found.group(2)].append(part.name)
    twice = {name: parts for name, parts in declared.items() if len(parts) > 1}
    assert twice == {}
