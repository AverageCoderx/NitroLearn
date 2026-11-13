"""The dashboard's Garage page (the bot on track, the getting-set-up checklist, the bays), the new-bot wizard
and the search box in the top band."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from browser import open_chromium, playwright_or_skip
from nitro.config.loader import load_toml
from nitro.config.schema import NitroConfig
from nitro.dashboard import app, bots, checks, garage, livery
from nitro.dashboard.api import Api
from nitro.runs import RunFolder
from test_dashboard import ROOT, TINY, items, make_bot

READY = {
    "version": "1.0.0",
    "verdict": "All required checks passed.",
    "checks": [
        {"name": "Python", "status": "OK", "detail": "3.11.9", "fix": ""},
        {"name": "GPU", "status": "WARN", "detail": "PyTorch cannot see an NVIDIA GPU", "fix": "Update it."},
        {"name": "Collision meshes", "status": "OK", "detail": "16/16 soccar meshes", "fix": ""},
    ],
}
# Whether any two of the elements `selector` finds lie on top of each other (by more than a pixel).
OVERLAP = """selector => { const boxes = [...document.querySelectorAll(selector)].filter(e => e.offsetParent)
  .map(e => e.getBoundingClientRect()).filter(r => r.width > 0 && r.height > 0);
  return boxes.some((a, i) => boxes.some((b, j) => j > i && a.left < b.right - 1 && b.left < a.right - 1
    && a.top < b.bottom - 1 && b.top < a.bottom - 1)); }"""
FITS = "document.documentElement.scrollWidth <= window.innerWidth"


def report(meshes: str = "OK") -> dict[str, Any]:
    """What Check this PC answers, with the arena files found or not."""
    data = json.loads(json.dumps(READY))
    data["checks"][2]["status"] = meshes
    if meshes != "OK":
        data["checks"][2] |= {"detail": "not found", "fix": "Copy them."}
        data["verdict"] = "1 problem must be fixed before training."
    return checks.read(json.dumps(data))


def write_metrics(folder: RunFolder, rows: int) -> None:
    lines = [f"{i * 1000},{i},{41000 + i},{0.01 * i}" for i in range(1, rows + 1)]
    folder.metrics.write_text(
        "agent_steps,iteration,steps_per_second,reward_per_step\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


def mark_running(folder: RunFolder) -> None:
    """Marks the bot as training, as its trainer does: this test's own process stands in for it."""
    marker = {"pid": os.getpid(), "started": time.time() - 600}
    (folder.root / bots.RUNNING).write_text(json.dumps(marker), encoding="utf-8")


def quiet(api: Api, meshes: str = "OK") -> Api:
    """`api` with the calls that start programs or open windows answered here."""
    api.check_pc = lambda: report(meshes)  # type: ignore[method-assign]
    api.job = lambda name, kind: {"state": "done"}  # type: ignore[method-assign]
    return api


def opened(browser: Any, api: Api, errors: list[str], calls: list[str] | None = None, **options: Any) -> Any:
    """The dashboard's page in a new tab, talking to `api`; `calls` collects the names of what it asks."""

    def answer(method: str, args: list[Any]) -> Any:
        if calls is not None:
            calls.append(method)
        return getattr(api, method)(*args)

    tab = browser.new_page(**{"viewport": {"width": 1440, "height": 920}, **options})
    tab.on("pageerror", lambda e: errors.append(str(e)))
    tab.expose_function("__py", answer)
    tab.add_init_script("window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});")
    tab.goto("about:blank")
    tab.set_content(app.page(api))
    tab.wait_for_selector("#g-list > *")
    return tab


def refresh(tab: Any) -> None:
    """What the page's 5-second refresh does on the Garage page, now."""
    tab.evaluate("loadHome().then(() => renderHome(true)).then(() => loadGarage())")


def settings_text(runs: Path, name: str) -> str:
    return RunFolder(runs / name).settings.read_text(encoding="utf-8")


# ---------- the calls behind the pages ----------
def test_the_garage_call_gives_each_bots_newest_save_reward_so_far_and_copy_progress(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    long = make_bot(runs, "long", saves=2)
    write_metrics(long, 1000)
    short = make_bot(runs, "short")
    write_metrics(short, 3)
    copier = make_bot(runs, "copier")
    (copier.root / "pretrained").mkdir()
    (copier.root / "pretrained" / "live.json").write_text('{"learned": 620, "limit": 1000, "rounds": []}')
    make_bot(runs, "fresh")
    api = Api(ROOT, runs)
    shown = api.garage()
    assert set(shown["bots"]) == {"long", "short", "copier", "fresh"} and shown["converting"] is False
    points = shown["bots"]["long"]["reward"]
    assert len(points) == garage.POINTS and points[0] == 0.01 and points[-1] == 10.0, (
        "a few points over the whole run, the newest always in"
    )
    assert points == sorted(points) and shown["bots"]["short"]["reward"] == [0.01, 0.02, 0.03]
    newest = (long.checkpoints / f"{2000:012d}" / "state.pt").stat().st_mtime
    assert shown["bots"]["long"]["last_save"] == newest and shown["bots"]["fresh"]["last_save"] is None
    assert shown["bots"]["fresh"]["reward"] == [] and shown["bots"]["fresh"]["copy"] is None
    assert shown["bots"]["copier"]["copy"] == {"learned": 620, "limit": 1000}
    with long.metrics.open("a", encoding="utf-8") as handle:
        handle.write("1001000,1001,41000,12.5\n")
    assert api.garage()["bots"]["long"]["reward"][-1] == 12.5, "read on from where the last read stopped"
    code = (
        "import sys; from pathlib import Path; from nitro.dashboard.api import Api; "
        f"Api(Path({str(ROOT)!r}), Path({str(runs)!r})).garage(); print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", "the Garage page's call never loads PyTorch"


def test_a_livery_is_picked_and_checked_before_its_bot_exists(tmp_path: Path) -> None:
    api = Api(ROOT, tmp_path / "runs")
    picked = api.livery_pick("sky-rocket")
    assert picked["livery"] == livery.pick("sky-rocket"), "what the bot will get by itself"
    assert [tuple(pair) for pair in picked["palette"]] == list(livery.PALETTE)
    assert api.livery_check("7", "#0f6b3e", "#FFD60A") == {
        "livery": {"number": "07", "main": "#0F6B3E", "contrast": "#FFD60A", "ink": "#F3F1EC"}
    }
    assert "one or two digits" in api.livery_check("", "#0F6B3E", "#FFD60A")["error"]
    assert "too alike" in api.livery_check("7", "#0F6B3E", "#0F6B3F")["error"]
    assert not (tmp_path / "runs").exists(), "nothing is written"


def test_brain_sizes_are_counted_as_the_trainer_builds_them_without_pytorch_in_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nitro.model import parameter_counts

    api = Api(ROOT, tmp_path / "runs")
    medium = {"model.policy": [512, 512, 512], "model.value": [512, 512, 512]}
    sizes = api.brain_sizes("2v2_starter", {"now": {}, "medium": medium, "odd": {"model.policy": "x"}})[
        "sizes"
    ]
    preset = load_toml(ROOT / "configs" / "presets" / "2v2_starter.toml", NitroConfig)
    assert sizes["now"] == parameter_counts(preset), "the trainer's own count"
    assert sizes["medium"]["total"] > 3 * sizes["now"]["total"] and sizes["medium"]["policy"] > 600_000
    assert "not a list of numbers" in sizes["odd"]["error"], "values the form refuses are not counted"
    assert "error" in api.brain_sizes("no-such-preset", {"now": {}})

    def no_process(*args: Any) -> dict[str, Any]:
        raise AssertionError("counted before: no process is started for it again")

    monkeypatch.setattr(garage, "count_parameters", no_process)
    assert api.brain_sizes("2v2_starter", {"again": medium})["sizes"]["again"] == sizes["medium"]


def test_what_a_new_bots_first_save_fixes_is_listed_for_the_review(tmp_path: Path) -> None:
    api = Api(ROOT, tmp_path / "runs")
    plain = {item["path"]: item for item in api.fixed_after_first_save("1v1_starter", {})["fixed"]}
    assert {"env.team_size", "env.observation", "model.encoder", "model.policy", "model.value"} <= set(plain)
    assert plain["model.policy"] == {
        "path": "model.policy",
        "label": "Policy",
        "value": [256, 256, 256],
        "title": "Policy network layers",
        "words": "3 layers of 256",
    }
    said = {item["title"]: item["words"] for item in plain.values()}
    assert said["Precision"] == "full (FP32)" and said["Kind of network (encoder)"] == "plain"
    assert said["Shared layers"] == "none" and said["Players per team"] == "1 (1v1)"
    assert not {"env.obs_team_size", "model.value_heads", "model.value_head_layers", "model.residual"} & set(
        plain
    ), "what adds nothing for a newcomer is left out"
    assert "ppo.popart" not in plain and "env.opponents" not in plain, "nor a default that means little"
    assert "model.attention_width" not in plain and "ppo.policy_lr" not in plain, (
        "not the sizes of an encoder it does not use, nor what can change any time"
    )
    changed = {"model.encoder": "attention", "model.policy": "1024, 1024"}
    attention = {item["path"]: item for item in api.fixed_after_first_save("2v2_starter", changed)["fixed"]}
    assert attention["model.policy"]["value"] == [1024, 1024] and "model.attention_width" in attention
    assert attention["run.precision"]["value"] == "fp16", "what the form adjusts for big networks shows too"
    assert "env.team_size" not in attention, "an attention network's game size can change later"
    assert attention["model.policy"]["words"] == "2 layers of 1,024"
    assert attention["model.attention_skip"]["words"] == "off", "the sizes of the encoder in use always show"
    residual = api.fixed_after_first_save("1v1_starter", {"model.residual": True})["fixed"]
    assert {"path": "model.residual", "words": "on"}.items() <= next(
        item for item in residual if item["path"] == "model.residual"
    ).items(), "a value that differs from the default shows"
    assert api.fixed_after_first_save("1v1_starter", {"env.tick_skip": "x"}) == {"fixed": []}
    assert "error" in api.fixed_after_first_save("no-such-preset", {})


# ---------- the Garage page ----------
def test_the_bot_that_trains_is_on_top_with_its_figures_its_buttons_and_its_car(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    racer = make_bot(runs, "racer", saves=3)
    write_metrics(racer, 50)
    mark_running(racer)
    copier = make_bot(runs, "copier")
    (copier.root / "pretrained").mkdir()
    (copier.root / "pretrained" / "live.json").write_text('{"learned": 620, "limit": 1000}')
    mark_running(copier)
    parked = make_bot(runs, "parked", saves=1)
    write_metrics(parked, 20)
    api = quiet(Api(ROOT, runs))
    api.set_livery("racer", 7, "#0F6B3E", "#FFD60A")
    assert {b["name"]: b["status"] for b in api.home()["bots"]} == {
        "racer": "training",
        "copier": "copying",
        "parked": "stopped",
    }
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.wait_for_selector(".ontrack[data-hero=racer] #gh-stats >> text=since the last save")
            hero = tab.locator(".ontrack")
            assert hero.locator(".plate").text_content() == "07" and hero.locator(
                ".tag.good"
            ).text_content() == ("On track")
            assert "Training · 1v1 · Plain network" in hero.locator("#gh-id").text_content()
            figures = hero.locator("#gh-stats .g-fig").all_text_contents()
            assert figures == ["50.0Ksteps", "41ksteps a second", "3saves", "under 1 minsince the last save"]
            car = hero.locator("svg.livery-car")
            assert (
                "number 07" in car.get_attribute("aria-label") and car.locator("text").text_content() == "07"
            )
            assert (
                car.locator("[fill='#0F6B3E']").count() >= 1 and car.locator("[fill='#FFD60A']").count() >= 3
            )
            assert tab.locator(".botcard[data-bot=racer]").count() == 0, "the bot on top is not also a row"
            assert tab.locator(".botcard").count() == 2
            also = tab.text_content("#gh-also")
            assert "Also at work" in also and "copier learning from replays 62%" in also
            assert "Learning from replays 62%" in tab.text_content(".botcard[data-bot=copier] .state")

            tab.click("#gh-watch")
            tab.wait_for_selector(".modal >> text=Watch racer play")
            tab.click(".modal [data-close]")
            tab.click("#gh-stop")
            tab.wait_for_selector(".toast >> text=Saving and stopping racer")
            assert (racer.root / bots.STOP_REQUEST).is_file(), (
                "Stop and save asked the trainer to save and stop"
            )

            # A refresh draws only what changed: the keyboard stays where it was, other rows stay as drawn.
            tab.focus("#g-new")
            tab.evaluate("document.querySelector('.botcard[data-bot=parked] .bay').kept = true")
            tab.evaluate("document.querySelector('#gh-stop').kept = true")
            with racer.metrics.open("a", encoding="utf-8") as handle:
                handle.write("90000,51,43000,0.9\n")
            refresh(tab)
            tab.wait_for_selector("#gh-stats >> text=90.0K")
            assert tab.evaluate("document.activeElement.id") == "g-new"
            assert tab.evaluate("document.querySelector('.botcard[data-bot=parked] .bay').kept") is True
            assert tab.evaluate("document.querySelector('#gh-stop').kept") is True, (
                "its buttons were not redrawn"
            )

            tab.click("#gh-name")
            tab.wait_for_selector("h1:has-text('racer')")
            tab.click(".nav [data-go=home]")
            tab.wait_for_selector(".ontrack[data-hero=racer]")
            (racer.root / bots.RUNNING).unlink()
            refresh(tab)
            tab.wait_for_selector(".ontrack[data-hero=copier]")
            hero_text = tab.text_content(".ontrack")
            assert "Learning from replays 62%" in hero_text and "620replays copied" in hero_text
            assert tab.locator(".botcard[data-bot=racer]").count() == 1, "stopped: back in its bay"
            (copier.root / bots.RUNNING).unlink()
            refresh(tab)
            tab.wait_for_selector(".pitlane >> text=Nothing is training right now")
            assert tab.locator(".ontrack").count() == 0 and tab.locator(".botcard").count() == 3
        finally:
            browser.close()
    assert errors == []


def test_the_checklist_follows_what_is_set_up_and_can_be_dismissed_for_good(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown")
    api = Api(ROOT, runs)
    looks: list[str] = []
    found = ["FAIL"]

    def check() -> dict[str, Any]:
        looks.append(found[0])
        return report(found[0])

    def choose() -> dict[str, Any]:  # in place of the window's file dialog and of starting the program
        program = tmp_path / "RLBot5" / "bin" / "launcher.exe"
        program.parent.mkdir(parents=True, exist_ok=True)
        program.touch()
        api.set_pref("rlbot_program", str(program))
        return {"opened": str(program)}

    api.check_pc = check  # type: ignore[method-assign]
    api.choose_rlbot = choose  # type: ignore[method-assign]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            assert "Checking this PC" in tab.text_content(".setup"), "until the checks have answered"
            tab.wait_for_selector(".setup >> text=Getting set up · 2 of 4")
            steps = tab.locator(".setup .setup-step")
            assert steps.all_text_contents() == [
                *("This PC is ready", "Arena files not found", "First bot made"),
                "Tell NitroLearn where RLBot is",
            ]
            assert (
                steps.nth(0).locator(".setup-tick svg").count() == 1
                and steps.nth(3).locator(".setup-tick.todo").count() == 1
            )
            assert steps.nth(1).locator(".setup-tick.hazard").count() == 1, (
                "a problem is marked as waiting for the user"
            )
            assert tab.text_content("#setup-next") == "See how to fix it"
            tab.click("#setup-arena")
            tab.wait_for_selector(".check-row[data-state=problem] >> text=Collision meshes")
            assert looks == ["FAIL"], "the Check this PC page shows the answer the Garage already had"

            found[0] = "OK"
            tab.click("[data-act=doctor-run]")
            tab.wait_for_selector("#main .banner >> text=All required checks passed.")
            tab.click(".nav [data-go=home]")
            tab.wait_for_selector(".setup >> text=Getting set up · 3 of 4")
            assert "Arena files found" in tab.text_content(".setup")
            assert tab.text_content("#setup-next") == "Choose RLBot…"
            tab.click("#setup-next")
            tab.wait_for_selector(".setup >> text=All set")
            assert tab.locator(".setup .setup-tick.todo").count() == 0
            tab.click("#setup-dismiss")
            tab.wait_for_selector(".setup", state="detached")
            assert api.home()["prefs"]["setup_done"] is True, "kept with the other preferences"

            again = opened(browser, api, errors)
            again.wait_for_timeout(1500)
            assert again.locator(".setup").count() == 0 and looks == ["FAIL", "OK"], (
                "dismissed for good: no checklist, and no checks run for it"
            )
        finally:
            browser.close()
    assert errors == []


def test_each_bay_has_the_one_action_that_fits_its_bot(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "fresh")
    trained = make_bot(runs, "trained", saves=2)
    write_metrics(trained, 30)
    waiting = make_bot(runs, "waiting", saves=1)
    write_metrics(waiting, 1)
    make_bot(runs, "broken", TINY.replace("seed = 3", 'seed = "three"'))
    api = quiet(Api(ROOT, runs))
    api.stage("waiting", "ppo.policy_lr", "0.0002")
    started: list[str] = []
    api.start = lambda name: started.append(name) or {"started": True}  # type: ignore[method-assign, func-returns-value]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            assert tab.text_content("#g-list h1") == "In the garage"
            assert (
                tab.locator("#g-list [data-act=unpack]").count() == 1 and tab.locator("#g-new").count() == 1
            )
            buttons = {
                name: tab.locator(f".botcard[data-bot={name}] > .btn")
                for name in ("fresh", "trained", "waiting", "broken")
            }
            assert {name: button.text_content() for name, button in buttons.items()} == {
                "fresh": "Start training",
                "trained": "Carry on",
                "waiting": "See changes",
                "broken": "Fix settings",
            }
            assert tab.locator(".botcard > .btn").count() == 4, "one a row"
            assert "primary" in buttons["fresh"].get_attribute("class"), (
                "the next step of a new bot is the loud one"
            )
            assert "primary" not in buttons["trained"].get_attribute("class")
            row = tab.locator(".botcard[data-bot=trained]")
            assert row.locator(".tag").all_text_contents() == ["Stopped"]
            assert row.locator(".g-fig").all_text_contents() == ["30.0Ksteps", "2saves"]
            tab.wait_for_selector(".botcard[data-bot=trained] svg.g-spark")
            assert "Last save just now" in row.locator(".trend").text_content()
            assert tab.locator(".botcard[data-bot=fresh] svg.g-spark").count() == 0, "no reward yet, no line"
            assert tab.locator(".botcard[data-bot=fresh] .trend").text_content() == "No save yet"
            assert tab.locator(".botcard[data-bot=waiting] .tag.hot").text_content() == "1 waiting for Apply"
            assert tab.locator(".botcard[data-bot=broken] .tag.bad").text_content() == "Settings problem"

            buttons["trained"].click()
            tab.wait_for_selector(".toast >> text=Starting trained")
            assert started == ["trained"]
            buttons["broken"].click()
            tab.wait_for_selector("text=The settings file has a mistake")
            assert tab.get_attribute("[role=tab][data-tab=settings]", "aria-selected") == "true"
            tab.click(".nav [data-go=home]")
            tab.click(".botcard[data-bot=waiting] > .btn")
            tab.wait_for_selector("#pending:not([hidden])")
            assert tab.text_content("h1") == "waiting"
            tab.click(".nav [data-go=home]")
            tab.click(".botcard[data-bot=fresh]")
            tab.wait_for_selector("h1:has-text('fresh')")
            assert started == ["trained"], "a click on the row opens the bot; only its button starts it"
        finally:
            browser.close()
    assert errors == []


def test_an_empty_garage_says_what_nitrolearn_does_and_offers_the_first_bot(tmp_path: Path) -> None:
    """Once the welcome is hidden (tests/test_dashboard_welcome.py), an empty garage still says it."""
    sync_api = playwright_or_skip()
    api = quiet(Api(ROOT, tmp_path / "runs"))
    api.set_pref("setup_done", True)
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            first = tab.text_content(".firstrun")
            assert "Your garage is empty" in first and "trains Rocket League bots on this PC" in first
            assert tab.locator(".botcard").count() == 0 and tab.locator(".ontrack").count() == 0
            assert tab.locator(".pitlane").count() == 0, "nothing to say about training yet"
            assert tab.locator(".firstrun [data-act=unpack]").count() == 1
            assert tab.locator(".setup").count() == 0 and tab.locator(".welcome").count() == 0
            assert tab.locator("[data-act=newbot]").count() == 1, "one button makes it"
            tab.click("#g-first")
            tab.wait_for_selector("#wizard [data-kind]")
            assert tab.locator("[data-kind=from-bot]").count() == 0, "no bot to start from yet"
        finally:
            browser.close()
    assert errors == []


# ---------- the new-bot wizard ----------
def through_the_wizard(tab: Any, name: str, kind: str = "", brain: str = "") -> None:
    """From the Garage to the wizard's brain-size step, with `name` typed and `kind` and `brain` picked."""
    tab.click(".nav [data-go=home]")
    tab.click("[data-act=newbot] >> nth=0")
    tab.wait_for_selector("#wizard [data-kind]")
    if kind:
        tab.click(f"[data-kind='{kind}']")
    tab.click("[data-act=wiz-next]")
    tab.fill("#wz-name", name)
    tab.click("[data-act=wiz-next]")
    tab.wait_for_selector("[data-brain]")
    if brain:
        tab.click(f"[data-brain={brain}]")


def make_it(tab: Any, name: str) -> None:
    tab.click("[data-act=wiz-next]")
    tab.click("[data-act=wiz-make]")
    tab.wait_for_selector(f"h1:has-text('{name}')")


def test_the_wizards_name_starts_filled_in_and_its_brain_step_does_not_jump(tmp_path: Path) -> None:
    """The name step starts with a real name no bot has (selected, so typing replaces it), not a placeholder
    that looks filled in; Next keeps it. On the brain-size step, Next stays where it is while the parameters
    are counted and once they are shown."""
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "my-bot")
    api = quiet(Api(ROOT, runs))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("[data-act=newbot] >> nth=0")
            tab.wait_for_selector("#wizard [data-kind]")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("#wz-name")
            assert tab.input_value("#wz-name") == "my-bot-2", "my-bot is taken"
            assert tab.get_attribute("#wz-name", "placeholder") is None
            picked = "[document.activeElement.selectionStart, document.activeElement.selectionEnd]"
            assert tab.evaluate(picked) == [0, len("my-bot-2")], "selected: typing replaces it"
            # Where Next is, each time it moves, from the brain step's first drawing on.
            tab.evaluate("""window.__tops = [];
                (function look() {
                  const next = document.getElementById("wz-next");
                  if (next && document.querySelector("[data-brain]")) {
                    const top = Math.round(next.getBoundingClientRect().top);
                    if (window.__tops[window.__tops.length - 1] !== top) window.__tops.push(top);
                  }
                  requestAnimationFrame(look);
                })();""")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("[data-count=small] >> text=366k parameters")
            tab.wait_for_selector("#wz-count >> text=These networks")
            tab.wait_for_timeout(300)
            assert len(tab.evaluate("window.__tops")) == 1, tab.evaluate("window.__tops")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("[data-act=wiz-make]")
            assert "my-bot-2" in tab.text_content(".wiz-made")
        finally:
            browser.close()
    assert errors == []


def test_the_wizard_makes_the_bot_the_form_made_for_default_and_for_custom_choices(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    api = quiet(Api(ROOT, runs))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            assert tab.locator("#band .nav [data-go=home]").get_attribute("aria-current") == "page"
            tab.click("[data-act=newbot] >> nth=0")
            tab.wait_for_selector("#wizard [data-kind]")
            assert tab.get_attribute("#band .nav [data-go=home]", "aria-current") == "page", (
                "part of the Garage"
            )
            kinds = tab.locator("[data-kind]")
            presets = [p.stem for p in sorted((ROOT / "configs" / "presets").glob("*.toml"))]
            assert [kinds.nth(i).get_attribute("data-kind") for i in range(kinds.count())] == [
                *(f"preset:{name}" for name in presets),
                "replays",
            ], "every preset the API lists, then learning from replays"
            assert tab.get_attribute("[data-kind='preset:1v1_starter']", "aria-pressed") == "true"
            assert tab.text_content("[data-kind='preset:1v1_starter'] .tag") == "Recommended"
            assert tab.locator("[data-kind] svg.mini").count() == kinds.count(), "each with its field picture"
            steps = tab.locator(".wiz-steps button")
            assert steps.all_text_contents() == ["1Kind of bot", "2Name and number", "3Brain size", "4Review"]
            assert [steps.nth(i).is_disabled() for i in range(4)] == [False, True, True, True]

            through_the_wizard(tab, "paged")
            assert tab.get_attribute("[data-brain=small]", "aria-pressed") == "true", "the starter's own size"
            tab.wait_for_selector("[data-count=small] >> text=366k parameters")
            assert tab.text_content("[data-count=large]") == "6.7M parameters"
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("[data-act=wiz-make]")
            review = tab.text_content("#wizard")
            assert "1v1 starter preset" in review and "Small: policy 3 layers of 256" in review
            assert "366k parameters" in review and "Everything else as in the starting settings." in review
            fixed = tab.locator(".wiz-fixed .chip").all_text_contents()
            assert "Policy network layers: 3 layers of 256" in fixed
            assert "Kind of network (encoder): plain" in fixed and "Precision: full (FP32)" in fixed
            assert (
                "Shared layers: none" in fixed
                and "What the bot sees (observation): detailed (nitro_v2)" in fixed
            )
            assert not [
                chip
                for chip in fixed
                if "Obs team size" in chip or "Residual" in chip or "Value head" in chip
            ]
            assert "Cannot be changed after its first save" in tab.text_content(".wiz-fixed")
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("h1:has-text('paged')")
            assert tab.get_attribute("[data-tab=rewards]", "aria-selected") == "true", "its rewards come next"
            assert api.create_bot("plain", "1v1_starter", {})["made"] == "plain"
            assert settings_text(runs, "paged") == settings_text(runs, "plain"), "the same bot as before"
            assert not RunFolder(runs / "paged").complete_checkpoints() and not bots.running(
                RunFolder(runs / "paged")
            )
            assert not (runs / "paged" / livery.FILE).exists(), (
                "the automatic number and colours need no file"
            )

            through_the_wizard(tab, "sized", brain="medium")
            assert tab.get_attribute("[data-brain=medium]", "aria-pressed") == "true"
            make_it(tab, "sized")
            api.create_bot(
                "sized-ref", "1v1_starter", {"model.policy": "512, 512, 512", "model.value": "512, 512, 512"}
            )
            assert settings_text(runs, "sized") == settings_text(runs, "sized-ref")

            through_the_wizard(tab, "tuned", kind="preset:2v2_starter", brain="custom")
            groups = tab.locator(".wiz-group h2").all_text_contents()
            assert groups == ["How it plays", "Its brain", "How it learns"], "every field has its group"
            asked = [
                tab.locator("[data-wiz]").nth(i).get_attribute("data-wiz")
                for i in range(tab.locator("[data-wiz]").count())
            ]
            form = api.wizard("2v2_starter")
            hidden = {"model.attention_width", "model.attention_blocks", "model.attention_heads"}
            hidden |= {
                "model.transformer_layers",
                "model.transformer_width",
                "model.transformer_heads",
                "model.transformer_ff",
            }
            assert asked == [
                i["path"] for step in form["steps"] for i in step["items"] if i["path"] not in hidden
            ]
            assert tab.locator(".wiz-item .help").count() >= len(asked) - 1, "each with its help text"
            tab.fill("#wz-env-tick_skip", "8")
            tab.keyboard.press("Tab")
            assert "(15.0 decisions a second)" in tab.text_content("#wz-rate")
            tab.select_option("#wz-model-encoder", "attention")
            tab.wait_for_selector("#wz-model-attention_width")
            tab.fill("#wz-model-policy", "512, 512")
            tab.fill("#wz-ppo-policy_lr", "abc")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector(".wiz-item .err >> text=is not a number")
            assert tab.evaluate("document.activeElement.id") == "wz-ppo-policy_lr", "said beside its field"
            tab.fill("#wz-ppo-policy_lr", "0.0002")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("[data-act=wiz-make]")
            review = tab.text_content(".wiz-review")
            assert "Tick skip" in review and "(starting settings: 4)" in review and "attention" in review
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("h1:has-text('tuned')")
            values = {"env.tick_skip": "8", "model.encoder": "attention", "model.policy": "512, 512"}
            api.create_bot("tuned-ref", "2v2_starter", values | {"ppo.policy_lr": "0.0002"})
            assert settings_text(runs, "tuned") == settings_text(runs, "tuned-ref")
            made = RunFolder(runs / "tuned").load_settings()
            assert (made.env.team_size, made.env.tick_skip, made.env.action_delay) == (2, 8, 1)
            assert (
                made.model.encoder == "attention"
                and made.model.attention_fast
                and made.ppo.policy_lr == 0.0002
            )
        finally:
            browser.close()
    assert errors == []


GCO = "gco_1v1_starter"
GCO_FULL = {
    "model.shared": [1152, 1152, 1152],
    "model.policy": [768, 768, 768],
    "model.value": [1280, 1280, 1280],
    "model.value_head_layers": [1280, 1280],
}


def checked_count(capsys: pytest.CaptureFixture[str], folder: RunFolder) -> int:
    """The total parameters `nitro check-settings` prints for a bot's settings file, which it must accept."""
    from nitro.cli import main

    meshes = (ROOT / "collision_meshes" / "soccar").is_dir()  # without them the check says so
    capsys.readouterr()
    assert main(["check-settings", str(folder.settings)]) == (0 if meshes else 1)
    out = capsys.readouterr().out
    assert not meshes or "no problems found" in out
    line = next(line for line in out.splitlines() if line.startswith("Networks: "))
    return int(line.split()[1].replace(",", ""))


def test_the_gco_starter_has_its_own_card_and_brain_sizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nitro.model import parameter_counts

    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    presets = api.home()["presets"]
    assert [p["name"] for p in presets] == ["1v1_starter", "2v2_starter", "3v3_starter", GCO], (
        "after the plain starters"
    )
    card = presets[-1]
    assert card["title"] == "GCO bot" and card["tags"] == ["Experimental"]
    assert card["about"] == "Untested design: learns from goals scored and conceded only. Slower to start."
    assert not {"title", "about", "tags", "brains"} & set(presets[0]), "plain presets keep the plain card"
    brains = {b["key"]: b for b in card["brains"]}
    assert list(brains) == ["small", "full"] and brains["small"]["values"] == {}
    assert brains["full"]["values"] == GCO_FULL
    asked = {p: [i["path"] for s in api.wizard(p)["steps"] for i in s["items"]] for p in (GCO, "1v1_starter")}
    assert "model.value_head_layers" in asked[GCO], "the form asks a file that has value heads"
    assert "model.value_head_layers" not in asked["1v1_starter"], "and only such a file"

    sizes = api.brain_sizes(GCO, {"small": {}, "full": GCO_FULL})["sizes"]
    preset = load_toml(ROOT / "configs" / "presets" / f"{GCO}.toml", NitroConfig)
    assert sizes["small"] == parameter_counts(preset), "the starter as written is the small size"
    assert 22_000_000 < sizes["full"]["total"] < 24_000_000, "about 23M, as the preset says"
    assert 4_500_000 < sizes["small"]["total"] < 5_500_000, "about 5M"
    for key in ("full", "small"):
        assert set(sizes[key]) >= {
            "shared",
            "critic_trunk",
            "head_main",
            "head_goal",
            "head_vdag1",
            "head_vdag2",
        }
    shares = [sizes[key]["playing"] / sizes[key]["total"] for key in ("full", "small")]
    assert abs(shares[0] - shares[1]) < 0.01, "the small one keeps the full one's shares"

    for key, values in (("small", {}), ("full", GCO_FULL)):
        assert api.create_bot(f"gco-{key}", GCO, values)["made"] == f"gco-{key}"
        folder = RunFolder(runs / f"gco-{key}")
        made = folder.load_settings()
        assert made.model.residual and made.model.value_heads == preset.model.value_heads
        assert made.ppo == preset.ppo and made.rewards == preset.rewards, "only the sizes change"
        assert checked_count(capsys, folder) == sizes[key]["total"]


def test_the_wizard_makes_a_gco_bot_in_either_of_its_sizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    api = quiet(Api(ROOT, runs))
    full = api.brain_sizes(GCO, {"full": GCO_FULL})["sizes"]["full"]["total"]
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("[data-act=newbot] >> nth=0")
            tab.wait_for_selector("#wizard [data-kind]")
            kinds = tab.locator("[data-kind]")
            assert [kinds.nth(i).get_attribute("data-kind") for i in range(4)] == [
                "preset:1v1_starter",
                "preset:2v2_starter",
                "preset:3v3_starter",
                f"preset:{GCO}",
            ]
            gco = f"[data-kind='preset:{GCO}']"
            assert tab.text_content(f"{gco} b") == "GCO bot"
            assert tab.text_content(f"{gco} .about") == (
                "Untested design: learns from goals scored and conceded only. Slower to start."
            )
            assert tab.get_attribute(f"{gco} .tag", "class") == "tag", "the plain tag, not the yellow one"
            assert tab.text_content(f"{gco} .tag") == "Experimental"
            assert tab.get_attribute("[data-kind='preset:1v1_starter'] .tag", "class") == "tag hot"
            assert tab.text_content("[data-kind='preset:2v2_starter'] b") == "2v2 starter", "as before"

            through_the_wizard(tab, "gco-small", kind=f"preset:{GCO}")
            brains = tab.locator("[data-brain]")
            assert [brains.nth(i).get_attribute("data-brain") for i in range(brains.count())] == [
                "small",
                "full",
                "custom",
            ], "its own sizes, then Custom"
            assert tab.get_attribute("[data-brain=small]", "aria-pressed") == "true", "the preset as written"
            assert tab.text_content("[data-brain=full] .layers") == (
                "trunk 1,152 · policy 768 · value 1,280 · heads 1,280"
            )
            assert (
                tab.text_content("[data-brain=small] .layers")
                == "trunk 520 · policy 352 · value 600 · heads 600"
            )
            tab.wait_for_selector("[data-count=small] >> text=5.0M parameters")
            assert tab.text_content("[data-count=full]") == "23M parameters"
            assert "four value heads" in tab.text_content(".wiz-lede")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("[data-act=wiz-make]")
            tab.wait_for_selector(".wiz-review >> text=5.0M parameters")
            review = tab.text_content(".wiz-review")
            assert "Small: trunk 520 · policy 352 · value 600 · heads 600 · 5.0M parameters" in review
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("h1:has-text('gco-small')")
            api.create_bot("gco-small-ref", GCO, {})
            assert settings_text(runs, "gco-small") == settings_text(runs, "gco-small-ref")

            through_the_wizard(tab, "gco-full", kind=f"preset:{GCO}", brain="small")
            tab.click("[data-brain=full]")
            make_it(tab, "gco-full")
            api.create_bot("gco-full-ref", GCO, GCO_FULL)
            assert settings_text(runs, "gco-full") == settings_text(runs, "gco-full-ref")
        finally:
            browser.close()
    assert errors == []
    small = api.brain_sizes(GCO, {"small": {}})["sizes"]["small"]["total"]
    assert checked_count(capsys, RunFolder(runs / "gco-small")) == small
    assert checked_count(capsys, RunFolder(runs / "gco-full")) == full


def test_the_wizards_name_number_and_colours_are_checked_and_kept(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "taken")
    api = quiet(Api(ROOT, runs))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#g-new")
            tab.wait_for_selector("#wizard [data-kind]")
            tab.keyboard.press("Enter")  # the chosen kind has the keyboard: Enter moves on
            tab.wait_for_selector("#wz-name")
            assert tab.evaluate("document.activeElement.id") == "wz-name"
            for typed, said in (
                ("", "Give the bot a name first."),
                ("has space", "A bot's name uses letters, digits"),
                ("con", "cannot name a folder on Windows"),
                ("taken", "There is already a bot called taken."),
            ):
                tab.fill("#wz-name", typed)
                tab.keyboard.press("Enter")
                tab.wait_for_selector(f"#wz-name-err >> text={said}")
                assert (
                    tab.evaluate("document.activeElement.id") == "wz-name"
                    and tab.locator(".toast").count() == 0
                )
                assert tab.get_attribute("#wz-name", "aria-invalid") == "true"
            tab.fill("#wz-name", "sky-rocket")
            assert tab.text_content("#wz-name-err") == "", "gone as soon as the name is typed anew"
            auto = livery.pick("sky-rocket")
            tab.wait_for_selector(f"#wz-preview .plate >> text={auto['number']}")
            assert (
                tab.input_value("#wz-number") == auto["number"]
                and tab.input_value("#wz-main") == auto["main"].lower()
            )
            assert "sky-rocket" in tab.text_content(
                "#wz-preview"
            ) and "Picked from its name" in tab.text_content("#wz-preview")
            assert tab.locator("#wz-preview svg.livery-car text").text_content() == auto["number"]
            assert tab.locator("[data-swatch]").count() == len(livery.PALETTE) and tab.is_hidden("#wz-auto")

            tab.fill("#wz-number", "x")
            tab.wait_for_selector("#wz-livery-err >> text=A race number is one or two digits")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_function("document.activeElement.id === 'wz-number'")
            assert tab.is_visible("#wz-name"), "still on its step, the keyboard on the field to put right"
            tab.fill("#wz-number", "7")
            tab.wait_for_selector("#wz-preview .plate >> text=07")
            assert tab.text_content("#wz-livery-err") == ""
            pair = next(i for i, colours in enumerate(livery.PALETTE) if colours[0] != auto["main"])
            tab.click(f"[data-swatch='{pair}']")
            main, second = livery.PALETTE[pair]
            tab.wait_for_function(
                f"document.querySelector('[data-swatch=\"{pair}\"]').getAttribute('aria-pressed') === 'true'"
            )
            assert (
                tab.input_value("#wz-main") == main.lower()
                and tab.input_value("#wz-second") == second.lower()
            )
            assert tab.locator(f"#wz-preview svg.livery-car [fill='{main}']").count() >= 1, "the car follows"
            assert "Your number and colours" in tab.text_content("#wz-preview") and tab.is_visible("#wz-auto")

            tab.click("#wz-auto")
            tab.wait_for_selector(f"#wz-preview .plate >> text={auto['number']}")
            assert tab.input_value("#wz-number") == auto["number"] and tab.is_hidden("#wz-auto")
            tab.fill("#wz-number", "7")
            tab.click(f"[data-swatch='{pair}']")
            tab.wait_for_selector("#wz-preview .plate >> text=07")

            # The steps seen so far can be clicked; what was typed stays.
            tab.keyboard.press("Tab")
            tab.click("#wz-step-0")
            tab.wait_for_selector("[data-kind]")
            assert tab.is_disabled("#wz-step-2") and not tab.is_disabled("#wz-step-1")
            tab.click("#wz-step-1")
            assert tab.input_value("#wz-name") == "sky-rocket" and tab.input_value("#wz-number") == "07"
            tab.click("#wz-name")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("[data-brain]")
            tab.click("[data-act=wiz-back]")
            assert tab.input_value("#wz-name") == "sky-rocket"
            tab.click("#wz-step-2")
            tab.wait_for_selector("[data-brain]")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector(".wiz-made .plate >> text=07")
            assert "your number and colours" in tab.text_content(".wiz-made")
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("#bothead .plate >> text=07")
            kept = json.loads((runs / "sky-rocket" / livery.FILE).read_text(encoding="utf-8"))
            assert kept == {"number": "07", "main": main, "contrast": second}
            assert api.bot("sky-rocket")["livery"]["main"] == main

            tab.click(".nav [data-go=home]")
            tab.click("#g-new")
            tab.wait_for_selector("[data-kind]")
            tab.click("[data-act=wiz-cancel]")
            tab.wait_for_selector(".botcard[data-bot=sky-rocket]")
            assert tab.locator("#wizard").count() == 0 and sorted(b["name"] for b in api.home()["bots"]) == [
                *("sky-rocket", "taken"),
            ], "Cancel makes nothing"
        finally:
            browser.close()
    assert errors == []


def test_the_wizard_starts_a_bot_from_another_one_or_from_replays_first(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "parent", saves=2)
    make_bot(runs, "unsaved")
    api = quiet(Api(ROOT, runs))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click("#g-new")
            tab.click("[data-kind=from-bot]")
            tab.select_option("#wz-from", "unsaved")
            assert tab.is_disabled("#wz-transfer") and "it has no save" in tab.text_content(".wiz-more")
            tab.select_option("#wz-from", "parent")
            assert tab.is_checked("#wz-transfer") and "at 2,000 steps" in tab.text_content(".wiz-more")
            tab.click("[data-act=wiz-next]")
            assert tab.input_value("#wz-name") == "parent-2", (
                "a name to start from, as New bot from this one gave"
            )
            tab.fill("#wz-name", "child")
            tab.click("[data-act=wiz-next]")
            tab.wait_for_selector("#wizard >> text=child gets parent's networks' kind and sizes")
            assert tab.locator("[data-brain]").count() == 0, "its brain is the other bot's"
            tab.click("[data-act=wiz-next]")
            assert "parent's trained networks (transfer learning)" in tab.text_content(".wiz-review")
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("h1:has-text('child')")
            assert tab.get_attribute("[role=tab][data-tab=settings]", "aria-selected") == "true"
            child = RunFolder(runs / "child").load_settings()
            assert (child.start_from.source, child.start_from.at, child.start_from.enabled) == (
                "parent",
                2000,
                True,
            )
            assert api.copy_bot("parent", "child-ref", True)["made"] == "child-ref"
            assert settings_text(runs, "child") == settings_text(runs, "child-ref"), (
                "as New bot from this one"
            )

            tab.click(".nav [data-go=home]")
            tab.click("#g-new")
            tab.click("[data-kind=replays]")
            assert tab.locator("[data-seg=wzsize] button").all_text_contents() == ["1v1", "2v2", "3v3"]
            tab.click("[data-seg=wzsize] [data-v='2v2_starter']")
            tab.click("[data-act=wiz-next]")
            tab.fill("#wz-name", "copycat")
            tab.keyboard.press("Enter")
            tab.wait_for_selector("[data-brain]")
            tab.click("[data-act=wiz-next]")
            assert "From your replays" in tab.text_content(".wiz-review")
            tab.click("[data-act=wiz-make]")
            tab.wait_for_selector("h1:has-text('copycat')")
            tab.wait_for_selector("#pending:not([hidden])")
            assert tab.get_attribute("[role=tab][data-tab=settings]", "aria-selected") == "true"
            waiting = api.bot("copycat")["pending"]
            assert waiting["pretrain.enabled"] is True, "switched on as its switch does: waiting for Apply"
            made = RunFolder(runs / "copycat").load_settings()
            assert made.env.team_size == 2 and not made.pretrain.enabled, "nothing is written before Apply"
        finally:
            browser.close()
    assert errors == []


# ---------- the search ----------
def test_the_search_finds_bots_pages_settings_and_actions_by_keyboard_and_mouse(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "striker", saves=1)
    make_bot(runs, "keeper")
    api = quiet(Api(ROOT, runs))
    started: list[str] = []
    api.start = lambda name: started.append(name) or {"started": True}  # type: ignore[method-assign, func-returns-value]
    errors: list[str] = []
    calls: list[str] = []
    rows = "#finder-list .finder-row"
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors, calls)
            assert tab.get_attribute("#finder-q", "placeholder") == "Search or run anything"
            assert tab.evaluate(
                "(() => { const b = id => document.querySelector(id).getBoundingClientRect(); return "
                "b('.nav').right <= b('#finder').left && b('#finder').right <= b('#theme-switch').left; })()"
            ), "between the navigation and the theme switch"
            assert tab.is_hidden("#finder-list")
            tab.keyboard.press("Control+k")
            assert tab.evaluate("document.activeElement.id") == "finder-q" and tab.is_visible("#finder-list")
            groups = tab.locator(".finder-group").all_text_contents()
            assert groups == ["Bots", "Pages", "Actions"], "what can be done now, before a word is typed"
            tab.keyboard.press("Escape")
            assert tab.is_hidden("#finder-list") and tab.evaluate("document.activeElement.id") != "finder-q"
            tab.keyboard.press("/")
            assert tab.evaluate("document.activeElement.id") == "finder-q"

            tab.wait_for_timeout(1500)  # the Garage's own calls are done
            calls.clear()
            tab.keyboard.type("stri")
            assert tab.locator(rows).all_text_contents() == ["strikerStopped · 1v1"]
            assert tab.locator(".finder-group").all_text_contents() == ["Bots"]
            assert calls == [], "typing asks Python nothing"
            tab.keyboard.press("Enter")
            tab.wait_for_selector("h1:has-text('striker')")
            assert tab.input_value("#finder-q") == "" and tab.is_hidden("#finder-list")

            # A setting of the open bot: by its name in the file, by its label, by a word of its help.
            tab.wait_for_function("S.data && S.data.settings && S.data.settings.length > 0")
            every = len(items(api.bot("striker")["settings"]))
            assert tab.evaluate("settingsIndex().length") == every > 100, "every setting is indexed"
            assert tab.evaluate("settingsIndex() === settingsIndex()"), "once a bot, not at every key"
            tab.keyboard.press("Control+k")
            calls.clear()
            tab.keyboard.type("ppo.policy_lr")
            first = tab.locator(rows).first
            assert first.locator(".what").text_content() == "Learning rate", "by its plain title"
            assert first.locator(".hint.mono").text_content() == "ppo.policy_lr"
            assert tab.locator(".finder-group").all_text_contents() == ["Settings"] and calls == []
            tab.keyboard.press("Enter")
            tab.wait_for_selector("#f-ppo-policy_lr")
            assert tab.get_attribute("[role=tab][data-tab=settings]", "aria-selected") == "true"
            tab.wait_for_selector("[data-item='ppo.policy_lr'].set-flash")
            assert tab.evaluate("document.activeElement.id") == "f-ppo-policy_lr"
            assert tab.evaluate(
                "(() => { const r = document.querySelector('#f-ppo-policy_lr').getBoundingClientRect(); "
                "return r.top >= 0 && r.bottom <= window.innerHeight; })()"
            ), "scrolled into view"
            assert tab.evaluate("S.searchTarget") is None

            tab.keyboard.press("Control+k")
            tab.keyboard.type("physics ticks")
            found = tab.locator(rows).first
            assert found.locator(".what").text_content() == "Physics ticks per decision"
            assert found.locator(".where").text_content() == "Env"
            tab.fill("#finder-q", "tick skip")
            assert tab.locator(rows).first.locator(".hint").text_content() == "env.tick_skip"
            assert tab.locator(".finder-group").all_text_contents() == ["Settings"]
            tab.fill("#finder-q", "zzzz-nothing")
            assert tab.locator(rows).count() == 0 and "Nothing matches" in tab.text_content("#finder-list")

            # Pages: the four of the band, and the open bot's tabs.
            tab.fill("#finder-q", "console")
            assert tab.locator(rows).first.text_content() == "striker · Console"
            tab.keyboard.press("Enter")
            tab.wait_for_selector("#consoleText")
            assert tab.get_attribute("[data-tab=console]", "aria-selected") == "true"

            # Actions: arrows move, Enter runs; the terminal command is the hint.
            tab.keyboard.press("Control+k")
            tab.keyboard.type("training")
            names = tab.locator(rows + " .what").all_text_contents()
            assert "Carry on training" in names and "Stop and save" not in names, "only what makes sense now"
            start = names.index("Carry on training")
            assert tab.locator(rows).nth(start).locator(".hint").text_content() == "nitro resume striker"
            assert tab.locator(rows).nth(0).get_attribute("aria-selected") == "true"
            for _ in range(start):
                tab.keyboard.press("ArrowDown")
            assert tab.locator(rows + "[aria-selected=true] .what").text_content() == "Carry on training"
            tab.keyboard.press("ArrowUp")
            tab.keyboard.press("ArrowDown")
            assert tab.get_attribute("#finder-q", "aria-activedescendant") == f"finder-{start}"
            tab.keyboard.press("Enter")
            tab.wait_for_selector(".toast >> text=Starting striker")
            assert started == ["striker"]

            tab.keyboard.press("Control+k")
            tab.keyboard.type("doctor")
            row = tab.locator(rows, has_text="Check this PC").last
            assert row.locator(".hint.mono").text_content() == "nitro doctor"
            tab.fill("#finder-q", "resources")
            tab.locator(rows, has_text="Resources").first.click()  # the mouse works too
            tab.wait_for_selector("#main >> text=Your Rust files")
            assert tab.get_attribute("#band .nav [data-go=resources]", "aria-current") == "page"
            tab.click("#finder-q")
            tab.keyboard.type("light look")
            tab.keyboard.press("Enter")
            tab.wait_for_function("document.documentElement.dataset.theme === 'light'")
            tab.click("#finder-q")
            tab.keyboard.type("new bot")
            assert (
                tab.locator(rows).first.locator(".hint").text_content()
                == "nitro new <settings file> --name <name>"
            )
            tab.keyboard.press("Enter")
            tab.wait_for_selector("#wizard [data-kind]")
            tab.click("#finder-q")
            assert "Watch it play" not in tab.text_content("#finder-list"), (
                "no bot is open: nothing of a bot's"
            )
            tab.click(".wiz-steps .cap")
            assert tab.is_hidden("#finder-list"), "a click elsewhere closes it"
            tab.keyboard.press("Control+k")
            ids = tab.evaluate("COMMANDS.map(c => c.id)")
            assert len(set(ids)) == len(ids) >= 15 and tab.evaluate(
                "COMMANDS.every(c => c.label && typeof c.run === 'function')"
            ), "the registry other parts add their tools to"
        finally:
            browser.close()
    assert errors == []


def test_the_search_stays_fast_with_two_hundred_settings(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    make_bot(runs, "shown", saves=1)
    api = quiet(Api(ROOT, runs))
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = opened(browser, api, errors)
            tab.click(".botcard[data-bot=shown] .bay")
            tab.wait_for_function("S.data && S.data.settings && S.data.settings.length > 0")
            listed = tab.evaluate(
                "(() => { const all = settingsIndex(); while (all.length < 200) all.push(...all.slice(0, 9));"
                " return settingsIndex().length; })()"
            )
            assert listed >= 200
            took = tab.evaluate(
                "(() => { const t = performance.now(); let n = 0; for (let i = 0; i < 200; i++) "
                "n += finderResults('learning rate ' + 'abcdefgh'[i % 8]).length; "
                "return (performance.now() - t) / 200; })()"
            )
            assert took < 5, f"a search over every setting took {took:.2f} ms"
        finally:
            browser.close()
    assert errors == []


# ---------- the smallest window ----------
def test_the_garage_the_wizard_and_the_search_fit_the_smallest_window(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    runs = tmp_path / "runs"
    racer = make_bot(runs, "a-rather-long-bot-name-1v1", saves=3)
    write_metrics(racer, 40)
    mark_running(racer)
    parked = make_bot(runs, "parked", saves=1)
    write_metrics(parked, 40)
    make_bot(runs, "broken", TINY.replace("seed = 3", 'seed = "three"'))
    make_bot(runs, "fresh")
    api = quiet(Api(ROOT, runs), meshes="FAIL")
    api.stage("parked", "ppo.policy_lr", "0.0002")
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            for theme in ("dark", "light"):
                api.set_pref("theme", theme)
                tab = opened(browser, api, errors, viewport={"width": 900, "height": 640})
                tab.wait_for_selector(".setup .setup-tick.hazard")
                tab.wait_for_selector(".ontrack #gh-stats .g-fig")
                assert tab.evaluate(FITS), f"{theme}: the Garage runs off the side"
                assert tab.evaluate("document.querySelector('#band').getBoundingClientRect().height") < 70, (
                    "the band stays one line with the search box in it"
                )
                for group in ("#band > *", ".ontrack-stats .g-fig, .ontrack-actions .btn", ".setup > *"):
                    assert not tab.evaluate(OVERLAP, group), f"{theme}: {group} overlap"
                for name in ("parked", "broken", "fresh"):
                    cells = f".botcard[data-bot={name}] .bay > *, .botcard[data-bot={name}] > .btn"
                    assert not tab.evaluate(OVERLAP, cells), f"{theme}: {name}'s row overlaps"
                assert tab.is_visible("#gh-stop") and tab.is_visible("#gh-watch")

                tab.click("#finder-q")
                tab.keyboard.type("a")
                tab.wait_for_selector("#finder-list .finder-row")
                box = tab.locator("#finder-list").bounding_box()
                field = tab.locator("#finder-q").bounding_box()
                assert box is not None and box["x"] >= 0 and box["x"] + box["width"] <= 900
                assert field is not None and field["width"] >= 240, "the box opens when it has the keyboard"
                assert tab.evaluate(FITS)
                tab.keyboard.press("Escape")
                tab.keyboard.press("Escape")

                tab.click("#g-new")
                tab.wait_for_selector("#wizard [data-kind]")
                assert tab.evaluate(FITS) and not tab.evaluate(OVERLAP, ".wiz-kinds > *, .wiz-steps button")
                assert tab.is_visible("#wz-next"), "Next stays in reach"
                tab.click("#wz-next")
                tab.fill("#wz-name", "small-window")
                tab.wait_for_selector("#wz-preview svg.livery-car")
                assert tab.evaluate(FITS) and not tab.evaluate(OVERLAP, ".wiz-fields > *, .wiz-preview")
                tab.click("#wz-next")
                tab.click("[data-brain=custom]")
                tab.wait_for_selector(".wiz-item")
                assert tab.evaluate(FITS) and not tab.evaluate(OVERLAP, ".wiz-brains > *")
                tab.click("#wz-next")
                tab.wait_for_selector("#wz-make")
                assert tab.evaluate(FITS) and not tab.evaluate(OVERLAP, ".wiz-nav > button")
                tab.close()
        finally:
            browser.close()
    assert errors == []
