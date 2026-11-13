"""The Rewards tab's folders, order and search, and a reward added only once: two blocks of the same reward
in a settings file each pay (the second as `<name>_2`), so the page and the API refuse a second copy, and a
file that has one already is warned about with a one-click fix through Apply. Folders and order are a view
only, kept in the bot's folder (reward_layout.json), never in its settings file."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.dashboard import reward_layout, rewards
from nitro.dashboard.api import Api
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]
FIVE = (
    '[[rewards]]\nname = "goal"\nweight = 10\n\n[[rewards]]\nname = "touch_ball"\n\n'
    '[[rewards]]\nname = "strong_touch"\nweight = 60\n\n[[rewards]]\nname = "air"\nweight = 0.25\n\n'
    '[[rewards]]\nname = "pickup_boost"\nweight = 10\n'
)
TWICE = (
    '[[rewards]]\nname = "touch_ball"\nweight = 10\n\n[[rewards]]\nname = "goal"\n\n'
    '[[rewards]]\nname = "touch_ball"\nweight = 3\n'
)


def make_bot(tmp_path: Path, name: str, text: str) -> RunFolder:
    settings = tmp_path / f"{name}.toml"
    settings.write_text(text, encoding="utf-8")
    return RunFolder.create(tmp_path / "runs", name, settings)


def entries(api: Api, name: str) -> list[dict[str, Any]]:
    return [{"from": e["from"], "view": e["view"]} for e in api.rewards(name)["rewards"]]


def kept_layout(folder: RunFolder) -> Any:
    return json.loads((folder.root / reward_layout.FILE).read_text(encoding="utf-8"))


def test_two_copies_of_a_reward_are_two_rewards_so_the_api_adds_none(tmp_path: Path) -> None:
    folder = make_bot(tmp_path, "d", TWICE)
    config = folder.load_settings()
    assert [(r.name, r.weight) for r in config.rewards] == [
        ("touch_ball", 10),
        ("goal", 1),
        ("touch_ball", 3),
    ]
    api = Api(ROOT, tmp_path / "runs")
    state = api.rewards("d")
    assert [e["label"] for e in state["rewards"]] == ["touch_ball", "goal", "touch_ball_2"], "each pays"
    assert state["duplicates"] == [{"name": "touch_ball", "indexes": [0, 2]}]

    listed = entries(api, "d")
    listed[1]["view"]["weight"] = 5
    assert not api.stage_rewards("d", listed).get("error"), "a file with a copy already can still be edited"
    copy = {"from": None, "view": dict(listed[1]["view"])}
    refused = api.stage_rewards("d", [*listed, copy])
    assert "'Goal' is already in this bot's rewards" in refused["error"]
    assert "error" in api.stage_rewards("d", [*listed, listed[1]]), "one block sent twice"
    assert api.rewards("d")["rewards"][1]["view"]["weight"] == 5, "the refused change left the waiting one"

    staged = api.stage_rewards("d", [listed[0], listed[1]])
    assert staged["duplicates"] == [] and "touch_ball: removed" in staged["changes"]
    assert api.apply("d")["applied"] == 2
    assert [r.name for r in folder.load_settings().rewards] == ["touch_ball", "goal"]

    group = {"name": "all_of", "parts": [{"name": "touch_ball"}, {"name": "air"}]}
    other = {"name": "all_of", "parts": [{"name": "touch_ball"}, {"name": "goal"}]}
    assert rewards.identity(group) != rewards.identity(other), "groups of other parts are other rewards"
    assert rewards.identity({**group, "parts": group["parts"][::-1]}) == rewards.identity(group)
    assert rewards.identity({"name": "all_of", "parts": []}) is None, "a group without parts yet"
    base = entries(api, "d")
    groups = [{"from": None, "view": rewards.view(view)} for view in (group, other)]
    assert not api.stage_rewards("d", base + groups).get("error")
    same = {"from": None, "view": rewards.view(group)}
    assert "'All of' is already" in api.stage_rewards("d", base + groups + [same])["error"]


def test_the_layout_file_keeps_known_rewards_once_and_refuses_bad_folders(tmp_path: Path) -> None:
    folder = make_bot(tmp_path, "l", FIVE)
    written = folder.settings.read_text(encoding="utf-8")
    api = Api(ROOT, tmp_path / "runs")
    assert api.rewards("l")["layout"] == {"folders": [], "loose": []}
    sent = {
        "folders": [
            {
                "id": "f1",
                "name": "  Ball   play ",
                "collapsed": True,
                "rewards": ["touch_ball", "nope", "goal"],
            },
            {"id": "f1", "name": "Boost", "rewards": ["goal", "pickup_boost"]},
        ],
        "loose": ["air", "touch_ball"],
    }
    kept = api.set_reward_layout("l", sent)["layout"]
    assert kept == {
        "folders": [
            {"id": "f1", "name": "Ball play", "collapsed": True, "rewards": ["touch_ball", "goal"]},
            {"id": "f2", "name": "Boost", "collapsed": False, "rewards": ["pickup_boost"]},
        ],
        "loose": ["air"],
    }
    assert kept_layout(folder) == kept and api.rewards("l")["layout"] == kept
    assert (
        "already a folder called"
        in api.set_reward_layout("l", {"folders": [{"name": "a"}, {"name": "A"}]})["error"]
    )
    assert "needs a name" in api.set_reward_layout("l", {"folders": [{"name": " "}]})["error"]
    assert kept_layout(folder) == kept, "a refused layout changes nothing"
    assert folder.settings.read_text(encoding="utf-8") == written, "the settings file is never touched"
    assert reward_layout.keys(["all_of", "goal", "all_of"]) == ["all_of", "goal", "all_of#2"]

    listed = entries(api, "l")
    api.stage_rewards("l", [e for e in listed if e["view"]["name"] != "goal"])
    assert api.rewards("l")["layout"]["folders"][0]["rewards"] == ["touch_ball", "goal"], "until Apply"
    api.apply("l")
    assert api.rewards("l")["layout"]["folders"][0]["rewards"] == ["touch_ball"]
    assert kept_layout(folder)["folders"][0]["rewards"] == ["touch_ball"], "gone from the file too"


@contextmanager
def dashboard(api: Api, height: int = 920) -> Iterator[tuple[Any, list[str], Callable[[], None]]]:
    """The dashboard's page in a headless Chromium bridged to `api`, the page's script errors, and what loads
    the page again."""
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page(viewport={"width": 1440, "height": height})
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )

            def load() -> None:
                tab.goto("about:blank")
                tab.set_content(page(api))

            load()
            yield tab, errors, load
        finally:
            browser.close()


def open_rewards(tab: Any) -> None:
    open_bot(tab)
    tab.click("[data-tab=rewards]")
    tab.wait_for_selector("#rw-list .reward")


def test_the_page_adds_a_reward_once_and_removes_a_copy_already_there(tmp_path: Path) -> None:
    folder = make_bot(tmp_path, "d", TWICE)
    api = Api(ROOT, tmp_path / "runs")
    with dashboard(api) as (tab, errors, _):
        open_rewards(tab)
        banner = tab.inner_text("#rw-dups")
        assert "more than once" in banner and "Touch ball" in banner and "2 copies" in banner
        assert tab.locator("[data-reward='2'] .tag.bad").inner_text().lower() == "extra copy"
        assert tab.locator("[data-reward='0'] .tag.bad").count() == 0, "the first copy is kept"

        tab.click("[data-act=rw-add]")
        tab.fill("#rw-cat-q", "goal")
        item = tab.locator("#rw-catalog [data-rwjump=goal]")
        assert (
            "Added" in (item.text_content() or "")
            and tab.locator("#rw-catalog [data-rwadd=goal]").count() == 0
        )
        assert tab.locator("#rw-catalog mark.hit").count() > 0, "the match is marked"
        item.click()
        tab.wait_for_selector("#modal", state="hidden")
        tab.wait_for_selector("[data-reward='1'].flash")
        assert tab.locator("[data-rwkey=goal]").count() == 1, "no copy added"
        assert not api.rewards("d")["pending"]

        copy = "{from: null, view: JSON.parse(JSON.stringify(S.rw.rewards[1].view))}"
        tab.evaluate(f"S.rw.rewards.push({copy}); stageRewards()")
        tab.wait_for_selector(".toast.bad >> text=already in this bot's rewards")
        assert not api.rewards("d")["pending"], "the API refused what the page sent"
        tab.evaluate("S.rw = null; renderRewards(document.querySelector('#tab'))")
        tab.wait_for_selector("#rw-dups >> text=Remove the extra copy")

        tab.click("#rw-dups >> text=Remove the extra copy")
        tab.wait_for_selector("#pending:not([hidden])")
        tab.wait_for_function("!document.querySelector('#rw-dups .banner')")
        assert folder.settings.read_text(encoding="utf-8") == TWICE, "waits for Apply"
        tab.click("[data-act=apply]")
        tab.click("[data-act=apply-go]")
        tab.wait_for_selector("#pending", state="hidden")
        assert [(r.name, r.weight) for r in folder.load_settings().rewards] == [
            ("touch_ball", 10),
            ("goal", 1),
        ]
    assert errors == []


def write_metrics(folder: RunFolder, labels: tuple[str, ...]) -> None:
    with folder.metrics.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(
            ["agent_steps", "iteration", "steps_per_second", *(f"paid_abs_{label}" for label in labels)]
        )
        for i in range(1, 60):
            out.writerow([i * 1000, i, 1000.0, *(0.2 for _ in labels)])


def test_folders_moves_and_the_search_in_the_page(tmp_path: Path) -> None:
    folder = make_bot(tmp_path, "f", FIVE)
    write_metrics(folder, ("goal", "touch_ball", "strong_touch", "air", "pickup_boost"))
    written = folder.settings.read_text(encoding="utf-8")
    api = Api(ROOT, tmp_path / "runs")

    def ball() -> list[str]:
        return list(kept_layout(folder)["folders"][0]["rewards"])

    def shown(where: str) -> list[str]:
        return list(tab.eval_on_selector_all(f"{where} .reward", "els => els.map(e => e.dataset.rwkey)"))

    with dashboard(api, height=1800) as (tab, errors, reload):
        open_rewards(tab)
        tab.click("[data-act=rw-folder-new]")
        tab.fill("#rw-folder-name", "Ball")
        tab.click("[data-act=rw-folder-go]")
        tab.wait_for_selector("[data-rwfolder=f1] >> text=Empty")
        assert kept_layout(folder)["folders"] == [
            {"id": "f1", "name": "Ball", "collapsed": False, "rewards": []}
        ]

        tab.click("[data-rwmove=touch_ball]")
        tab.click(".rw-menu [data-to=f1]")
        tab.wait_for_selector("[data-rwfolder=f1] [data-rwkey=touch_ball]")
        assert ball() == ["touch_ball"]
        assert tab.evaluate("document.activeElement.dataset.rwgrip") == "touch_ball", "the focus stays on it"

        tab.drag_and_drop("[data-rwgrip=strong_touch]", "[data-rwfolder=f1] .rw-folder-head")
        tab.wait_for_function("document.querySelectorAll('[data-rwfolder=f1] .reward').length === 2")
        assert ball() == ["touch_ball", "strong_touch"] == shown("[data-rwfolder=f1]")
        tab.drag_and_drop(
            "[data-rwgrip=strong_touch]", "[data-rwkey=touch_ball]", target_position={"x": 300, "y": 4}
        )
        tab.wait_for_function(
            "document.querySelector('[data-rwfolder=f1] .reward').dataset.rwkey === 'strong_touch'"
        )
        assert ball() == ["strong_touch", "touch_ball"]
        tab.drag_and_drop(
            "[data-rwgrip=pickup_boost]", "[data-rwkey=goal]", target_position={"x": 300, "y": 4}
        )
        tab.wait_for_function("document.querySelector('.rw-loose .reward').dataset.rwkey === 'pickup_boost'")
        assert kept_layout(folder)["loose"] == ["pickup_boost", "goal", "air"]

        tab.focus("[data-rwgrip=touch_ball]")
        tab.keyboard.press("ArrowUp")
        tab.wait_for_function(
            "document.querySelector('[data-rwfolder=f1] .reward').dataset.rwkey === 'touch_ball'"
        )
        assert ball() == ["touch_ball", "strong_touch"]
        tab.keyboard.press("ArrowUp")  # the top already: stays
        tab.click("[data-rwmove=strong_touch]")
        assert tab.locator(".rw-menu [data-rwmv=down]").is_disabled(), "the last of its folder"
        tab.click(".rw-menu [data-rwmv=up]")
        tab.wait_for_function(
            "document.querySelector('[data-rwfolder=f1] .reward').dataset.rwkey === 'strong_touch'"
        )
        tab.focus("[data-rwgrip=strong_touch]")
        tab.keyboard.press("ArrowDown")
        tab.wait_for_function(
            "document.querySelector('[data-rwfolder=f1] .reward').dataset.rwkey === 'touch_ball'"
        )

        tab.click("[data-rwfolderrename=f1]")
        tab.fill("#rw-folder-name", "Ball control")
        tab.keyboard.press("Enter")
        tab.wait_for_selector("[data-rwfolder=f1] .rw-fold >> text=Ball control")
        tab.click("[data-act=rw-folder-new]")
        tab.fill("#rw-folder-name", "ball CONTROL")
        tab.click("[data-act=rw-folder-go]")
        tab.wait_for_selector(".toast.bad >> text=already a folder called")
        tab.click("[data-close]")

        tab.click("[data-rwmove=pickup_boost]")
        tab.click(".rw-menu [data-rwmv=new]")
        tab.fill("#rw-folder-name", "Boost")
        tab.click("[data-act=rw-folder-go]")
        tab.wait_for_selector("[data-rwfolder=f2] [data-rwkey=pickup_boost]")
        assert "Ball control" in (tab.text_content("#rw-aside .share:has-text('Touch ball') .rw-in") or ""), (
            "its tag"
        )

        tab.click("[data-rwfold=f1]")
        tab.wait_for_selector("[data-rwfolder=f1].closed")
        assert tab.locator("[data-rwfolder=f1] .reward").count() == 0
        assert kept_layout(folder)["folders"][0]["collapsed"] is True

        reload()
        open_rewards(tab)
        assert tab.locator("[data-rwfolder=f1].closed").count() == 1, "still closed after a reload"
        assert "2 rewards" in (tab.text_content("[data-rwfolder=f1] .rw-fold") or "")
        assert shown("[data-rwfolder=f2]") == ["pickup_boost"] and shown(".rw-loose") == ["goal", "air"]

        tab.fill("#rw-search", "touch")
        tab.wait_for_selector("[data-rwfolder=f1]:not(.closed) [data-rwkey=touch_ball]")
        assert shown("#rw-list") == ["touch_ball", "strong_touch"], "the other folder and the rest hide"
        assert tab.locator("[data-rwfolder=f2]").count() == 0
        assert tab.inner_text("[data-rwkey=touch_ball] .title mark.hit") == "Touch"
        tab.press("#rw-search", "Escape")
        tab.wait_for_selector("[data-rwfolder=f1].closed")
        assert tab.input_value("#rw-search") == "" and shown("#rw-list") == ["pickup_boost", "goal", "air"]
        tab.fill("#rw-search", "boost")  # the folder's name: all of it
        tab.wait_for_selector("[data-rwfolder=f2] mark.hit")

        tab.fill("#rw-search", "")
        tab.click("[data-rwfolderdel=f2]")
        tab.wait_for_function("!document.querySelector('[data-rwfolder=f2]')")
        assert [f["name"] for f in kept_layout(folder)["folders"]] == ["Ball control"]
        assert shown(".rw-loose") == ["goal", "air", "pickup_boost"], "back out of any folder"
        assert folder.settings.read_text(encoding="utf-8") == written, "folders never touch the settings"

        tab.click("[data-act=rw-add]")
        tab.fill("#rw-cat-q", "demo")
        tab.click("[data-rwadd=demo]")
        tab.wait_for_function(
            "[...document.querySelectorAll('.rw-loose .reward')].pop().dataset.rwkey === 'demo'"
        )
        tab.click("[data-rwfold=f1]")
        tab.click("[data-rwkey=strong_touch] [data-rwdel]")
        tab.wait_for_function("!document.querySelector('[data-rwkey=strong_touch]')")
        tab.click("[data-act=apply]")
        tab.click("[data-act=apply-go]")
        tab.wait_for_selector("#pending", state="hidden")
        tab.wait_for_selector("[data-rwfolder=f1] [data-rwkey=touch_ball]")
        assert ball() == ["touch_ball"], "a reward removed from the settings leaves its folder"
        assert [r.name for r in folder.load_settings().rewards] == [
            "goal",
            "touch_ball",
            "air",
            "pickup_boost",
            "demo",
        ]
    assert errors == []
