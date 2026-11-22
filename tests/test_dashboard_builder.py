"""The Rewards tab's reward builder: a built reward read from and written back into a
settings file (its terms as [[rewards.terms]] blocks, comments kept), the blocks and their help for the
editor, its title as its name in the training's numbers, and the editor in the page."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from browser import open_chromium, playwright_or_skip
from nitro.config import built
from nitro.config.loader import parse_toml
from nitro.config.rewards import Built
from nitro.config.schema import NitroConfig
from nitro.dashboard import rewards
from nitro.dashboard.api import Api
from nitro.dashboard.edits import problems
from nitro.runs import RunFolder
from test_dashboard_rewards import write_metrics

ROOT = Path(__file__).parents[1]
PRESET = ROOT / "configs" / "presets" / "2v2_starter.toml"

TEXT = """# My bot
[[rewards]]
name = "air"
weight = 1

[[rewards]]
name = "built"
title = "my_touch"
weight = 2
# What it pays for

# The high touch
[[rewards.terms]]
when = ["in_air", { ball_above = 300 }] # high up
measure = "touch"
weight = 1

[[rewards.terms]]
measure = ["air_time", "ball_height"]
combine = "min"

# How games start
[[state_setters]]
name = "kickoff"
weight = 1
"""


def entries_of(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"from": i, "view": copy.deepcopy(view)} for i, view in enumerate(views)]


def config_of(text: str) -> NitroConfig:
    return parse_toml(text, NitroConfig, file=PRESET, base_dir=PRESET.parent)


def test_a_built_reward_is_read_and_written_back_term_by_term_keeping_its_comments() -> None:
    views, _ = rewards.read(TEXT)
    view = views[1]
    assert view["known"] and view["title"] == "my_touch" and view["params"] == {}
    assert view["terms"][1] == {"measure": ["air_time", "ball_height"], "combine": "min"}
    assert rewards.written(TEXT, entries_of(views)) == TEXT, "nothing changed, nothing written"

    entries = entries_of(views)
    edited = entries[1]["view"]
    edited["terms"][0]["when"][1] = {"ball_above": "400"}  # as the page sends a number typed in
    edited["zero_sum"] = {"team_spirit": "0.5", "opponent_scale": 1}
    edited["terms"].append({"when": [{"not": "kickoff"}], "measure": "goal", "weight": "5"})
    entries.append({"from": None, "view": {**rewards.view({"name": "built"}), "title": "chase",
                                           "terms": [{"measure": "speed_toward_ball"}]}})  # fmt: skip
    after = rewards.written(TEXT, entries)
    for kept in ("# What it pays for", "# The high touch", "# high up", "# How games start"):
        assert kept in after
    assert 'when = ["in_air", { ball_above = 400 }] # high up' in after
    assert after.index("zero_sum") < after.index("# What it pays for"), "its own lines stay above its terms"
    assert after.index('measure = "goal"') < after.index('title = "chase"') < after.index("# How games start")
    assert problems(after, PRESET, PRESET.parent) == []
    config = config_of(after)
    first, second = (r for r in config.rewards if isinstance(r, Built))
    assert first.title == "my_touch" and len(first.terms) == 3 and first.terms[2]["weight"] == 5
    assert first.zero_sum is not None and first.zero_sum.team_spirit == 0.5
    assert second.title == "chase" and second.every_step
    assert [s.name for s in config.state_setters] == ["kickoff"]

    again = entries_of(rewards.read(after)[0])
    del again[1]["view"]["terms"][1]
    fewer = rewards.written(after, again)
    assert "air_time" not in fewer and "# The high touch" in fewer and config_of(fewer)


def test_built_rewards_are_named_by_their_titles_in_labels_keys_and_changes() -> None:
    views, _ = rewards.read(TEXT)
    other = {**copy.deepcopy(views[1]), "title": "other"}
    assert rewards.labels([views[0], views[1], other]) == ["air", "my_touch", "other"]
    assert rewards.duplicates([views[1], other]) == [] and len(rewards.duplicates([views[1], views[1]])) == 1
    entries = entries_of(views)
    entries[1]["view"]["title"] = "touch_up_high"
    entries[1]["view"]["terms"][0]["weight"] = 3
    assert rewards.changes(views, entries) == [
        "my_touch: renamed to touch_up_high",
        "touch_up_high: terms changed (2 terms)",
    ]


def test_the_catalog_and_the_blocks_for_the_editor(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    RunFolder.create(runs, "r", PRESET)
    state = Api(ROOT, runs).rewards("r")
    entry = next(k for k in state["catalog"] if k["name"] == "built")
    assert entry["title"] == "Build your own (no code)" and entry["repeatable"] and entry["params"] == []
    blocks = state["blocks"]
    assert [b["name"] for b in blocks["flags"]] == list(built.FLAGS)
    assert [b["name"] for b in blocks["numbered"]] == list(built.NUMBERED)
    assert {b["name"]: b["every_step"] for b in blocks["measures"]} == {
        name: every for name, (_, every) in built.MEASURES.items()
    }
    assert next(b for b in blocks["flags"] if b["name"] == "touching_ball")["event"]
    assert set(blocks["term_keys"]) == set(built.TERM_KEYS)


def test_what_the_editor_stages_is_checked_applied_and_shown_by_its_title(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "r", PRESET)
    api = Api(ROOT, runs)
    entries = [{"from": e["from"], "view": e["view"]} for e in api.rewards("r")["rewards"]]
    view = {**rewards.view({"name": "built"}), "title": "my_reward", "weight": "10",
            "terms": [{"when": ["in_air", {"ball_above": "300"}], "measure": "touch"}]}  # fmt: skip
    entries.append({"from": None, "view": view})
    staged = api.stage_rewards("r", entries)
    assert staged["pending"] and not any(e["problems"] for e in staged["rewards"]), staged["rewards"][-1]
    assert staged["rewards"][-1]["label"] == "my_reward"
    assert staged["changes"][-1] == "my_reward: added, weight 10"
    view["terms"][0]["measure"] = "speed"
    bad = api.stage_rewards("r", entries)["rewards"][-1]["problems"]
    assert any("no measure called 'speed'" in p for p in bad), bad
    view["terms"][0]["measure"] = "touch"
    api.stage_rewards("r", entries)
    assert api.apply("r")["applied"] == 1
    reward = folder.load_settings().rewards[-1]
    assert isinstance(reward, Built) and reward.terms[0]["when"][1] == {"ball_above": 300}
    write_metrics(folder, [{"air": 0.2, "my_reward": 0.8}] * 100)
    shares = api.rewards("r")["shares"]["rewards"]
    assert shares[0]["label"] == "my_reward" and shares[0]["share"] > 0.7


def test_a_reward_is_built_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "shown", PRESET)
    api = Api(ROOT, runs)
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
            tab.click(".botcard")
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("text=What each reward pays")
            count = len(api.rewards("shown")["rewards"])
            tab.click("[data-act=rw-add]")
            tab.fill("#rw-cat-q", "build")
            tab.click("[data-rwadd=built]")
            i = count
            tab.wait_for_selector(f"#bd-{i}-title")
            assert tab.input_value(f"#bd-{i}-title") == "my_reward"
            tab.fill(f"#bd-{i}-title", "high_touch_built")
            tab.keyboard.press("Enter")
            tab.wait_for_selector(".reward .title >> text=High touch built")
            tab.select_option(f"#bd-{i}-0-cond", "ball_above")
            tab.wait_for_selector(f"#bd-{i}-0-c1")
            tab.fill(f"#bd-{i}-0-c1", "400")
            tab.keyboard.press("Enter")
            tab.select_option(f"#bd-{i}-0-measure", "ball_height")
            tab.wait_for_selector(f"#bd-{i}-0-combine")
            tab.click(f"[data-bdx='addterm:{i}']")
            tab.wait_for_selector(f"#bd-{i}-1-weight")
            tab.fill(f"#bd-{i}-1-weight", "0.5")
            tab.keyboard.press("Enter")
            tab.click(f"[data-bdx='not:{i}:0:0']")
            tab.wait_for_selector(".bd-cond.bd-not")
            tab.wait_for_function(
                "() => !document.querySelector('#pending').hidden && "
                "!document.querySelector('.rw-problems[role=alert]')"
            )
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
        finally:
            browser.close()
    assert errors == []
    reward = folder.load_settings().rewards[-1]
    assert isinstance(reward, Built) and reward.title == "high_touch_built"
    assert json.loads(json.dumps(reward.terms)) == [
        {"when": [{"not": "in_air"}, {"ball_above": 400}], "measure": ["touch", "ball_height"]},
        {"measure": "one", "weight": 0.5},
    ]


# ---------- reward builder 2: formulas, variables, live checks, templates, a recorded game ----------

ADVANCED = """# My bot
[[rewards]]
name = "built"
title = "my_chain"
weight = 2

# Touches in a row
[[rewards.vars]]
name = "chain" # counts
next = "if(me.on_ground, 0, if(me.touch, chain + 1, chain))"

# The touch
[[rewards.terms]]
when = ["in_air", "me.touch"] # in the air
value = "chain * ball.z / 2044"
cap_episode = 5

# How games start
[[state_setters]]
name = "kickoff"
weight = 1
"""


def test_variables_and_formulas_are_read_and_written_back_keeping_their_comments() -> None:
    views, _ = rewards.read(ADVANCED)
    view = views[0]
    assert view["vars"] == [{"name": "chain", "next": "if(me.on_ground, 0, if(me.touch, chain + 1, chain))"}]
    assert view["terms"][0]["value"] == "chain * ball.z / 2044" and "measure" not in view["terms"][0]
    assert rewards.written(ADVANCED, entries_of(views)) == ADVANCED, "nothing changed, nothing written"
    entries = entries_of(views)
    edited = entries[0]["view"]
    edited["vars"][0]["start"] = "1"
    edited["vars"].append({"name": "air", "start": 0, "next": "if(me.on_ground, 0, air + dt)"})
    edited["terms"][0]["value"] = "chain * min(air, 2)"
    edited["terms"][0]["cap_episode"] = ""
    edited["terms"].append({"value": "clamp(ball.y * me.team / 5120, -1, 1)", "every_step": True})
    after = rewards.written(ADVANCED, entries)
    for kept in ("# Touches in a row", "# counts", "# The touch", "# in the air", "# How games start"):
        assert kept in after
    assert "cap_episode" not in after and 'value = "chain * min(air, 2)"' in after
    reward = config_of(after).rewards[0]
    assert isinstance(reward, Built) and [v["name"] for v in reward.vars] == ["chain", "air"]
    assert reward.vars[0]["start"] == 1 and reward.terms[1]["every_step"] is True
    entries[0]["view"]["vars"] = []
    assert "rewards.vars" not in rewards.written(after, entries), "the last variable removed, no block left"


def test_the_live_check_names_the_cores_problem_at_its_place() -> None:
    api = Api(ROOT, ROOT / "runs")
    view = rewards.read(ADVANCED)[0][0]
    fine = api.check_built(view)
    assert fine["problems"] == [] and fine["every_step"] is False and fine["memory"] > 0
    view["terms"][0]["value"] = "chain * ball.zz"
    assert api.check_built(view)["problems"] == [
        {"where": "terms[0].value", "message": "'ball' has no field 'zz' (at 14)"}
    ]
    view["terms"][0]["value"] = "1"
    view["vars"][0]["next"] = "chian + 1"
    assert api.check_built(view)["problems"] == [
        {"where": "vars[0]", "message": "there is nothing called 'chian' (at 1)"}
    ]
    view["vars"] = []
    view["terms"][0]["when"] = ["me.touch and"]
    assert api.check_built(view)["problems"][0]["where"] == "terms[0].when[0]"
    view["terms"] = [{"value": "ball.z / 2044"}]
    assert api.check_built(view) == {"problems": [], "every_step": True, "memory": 0}
    view["title"] = "Bad Title"
    assert api.check_built(view)["problems"][0]["where"] == "title"


def test_every_template_is_listed_and_passes_the_settings_check_once_added() -> None:
    listed = rewards.blocks()["templates"]
    file = json.loads(rewards.TEMPLATES.read_text(encoding="utf-8"))
    assert [t["title"] for t in listed] == [t["title"] for t in file] and len(listed) >= 10
    assert {t["title"]: t["copies"] for t in listed}["aerial_air_time"] == "aerial_touch_airtime"
    assert {t["title"]: t["copies"] for t in listed}["touch_then_goal"] == "", "a test-only copy is not named"
    base = PRESET.read_text(encoding="utf-8")
    for template in listed:
        view = {**rewards.view({"name": "built"}), "title": template["title"],
                "vars": template["vars"], "terms": template["terms"]}  # fmt: skip
        assert rewards.built_problems(view)["problems"] == [], template["title"]
        reward = config_of(rewards.written(base, [{"from": None, "view": view}])).rewards[0]
        assert isinstance(reward, Built) and reward.title == template["title"]


def test_a_reward_is_tested_alone_on_a_recorded_game(tmp_path: Path, monkeypatch: Any) -> None:
    from nitro import recordings
    from nitro.cli import main
    from nitro.viewer import recording as layouts
    from test_watch import MESHES, saved_run

    view = {**rewards.view({"name": "built"}), "title": "my_touches", "terms": [{"measure": "touch"}]}
    folder_root = tmp_path / "runs" / "my-bot"
    api = Api(ROOT, tmp_path / "runs")
    started: list[list[str]] = []
    monkeypatch.setattr(type(api._jobs), "start", lambda self, name, kind, args, cwd: started.append(args))
    found = [p for p in recordings.bundled_recordings() if "nexto_1v1" in p.name]
    folder = saved_run(tmp_path, 500)
    assert folder.root == folder_root
    bad = {**view, "terms": [{"value": "me.zz"}]}
    assert "fix the reward first" in api.score_built("my-bot", bad, str(tmp_path / "x.rlpr"))["error"]
    assert "there is no file" in api.score_built("my-bot", view, str(tmp_path / "x.rlpr"))["error"]
    game = found[0] if found else tmp_path / "game.rlpr"
    if not found:
        game.touch()
    out = api.score_built("my-bot", view, str(game))
    assert out["started"] and started[-1][:2] == ["score-replay", "my-bot"]
    assert started[-1][-2:] == ["--rewards", out["rewards"]]
    tried = Path(out["rewards"]).read_text(encoding="utf-8")
    assert 'title = "my_touches"' in tried and "[[rewards.terms]]" in tried
    if not found or not (MESHES / "soccar").is_dir():
        print("SKIPPED: scoring for real needs the collision meshes and RocketSim's recordings")
        return
    # The job's command, run here: the game is paid by this reward alone.
    assert main([*started[-1][:-2], "--seconds", "60", "--rewards", out["rewards"]]) == 0
    data = layouts.load(folder.root / "replay_score.nlrec")
    assert data["rewards"]["labels"] == ["my_touches"]


def test_an_advanced_reward_is_built_in_the_page_with_formulas_a_variable_and_a_template(
    tmp_path: Path,
) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "shown", PRESET)
    api = Api(ROOT, runs)
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
            tab.click(".botcard")
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("text=What each reward pays")
            i = len(api.rewards("shown")["rewards"])
            tab.click("[data-act=rw-add]")
            tab.fill("#rw-cat-q", "build")
            tab.click("[data-rwadd=built]")
            tab.wait_for_selector(f"#bd-{i}-title")
            # The term pays a formula: a mistake is named under it as it is typed, and goes once fixed.
            tab.click(f"[data-bdx='mode:{i}:0:formula']")
            value = f"#bd-{i}-t-0-value"
            tab.wait_for_selector(value)
            message = f"[data-bdmsg='{i}:terms[0].value']"
            tab.fill(value, "ball.zz / 2044")
            tab.wait_for_function(
                f'document.querySelector("{message}").textContent.includes("no field \'zz\'")'
            )
            tab.fill(value, "")
            tab.click("[data-bdx='ref:0']")
            tab.click(value)
            tab.click("[data-bdins='ball.z']")
            tab.type(value, " / 2044")
            tab.wait_for_function(f"document.querySelector(\"{message}\").textContent === ''")
            assert tab.input_value(value) == "ball.z / 2044"
            tab.click(f"#bd-{i}-title")  # leaves the formula: staged
            # A variable, read by a formula condition.
            tab.click(f"[data-bdx='addvar:{i}']")
            tab.wait_for_selector(f"#bd-{i}-v0-name")
            tab.fill(f"#bd-{i}-v0-name", "air")
            tab.keyboard.press("Tab")
            tab.fill(f"#bd-{i}-v-0-next", "if(me.on_ground, 0, air + dt)")
            tab.click(f"#bd-{i}-title")
            tab.select_option(f"#bd-{i}-0-cond", "__formula")
            tab.wait_for_selector(f"#bd-{i}-t-0-when-1")
            tab.fill(f"#bd-{i}-t-0-when-1", "air > 0.5")
            tab.click(f"#bd-{i}-title")
            tab.check(f"#bd-{i}-0-once_per_episode")
            # A template.
            tab.click("[data-act=rw-add]")
            tab.fill("#rw-cat-q", "kickoff race")
            tab.click("[data-bdtpl=kickoff_race]")
            tab.wait_for_selector(f"#bd-{i + 1}-title")
            tab.wait_for_function(
                "() => !document.querySelector('#pending').hidden && "
                "!document.querySelector('.rw-problems[role=alert]') && "
                "![...document.querySelectorAll('.bd-msg.bad')].some(m => m.textContent)"
            )
            tab.click("[data-act=apply]")
            tab.click("[data-act=apply-go]")
            tab.wait_for_selector("#pending", state="hidden")
        finally:
            browser.close()
    assert errors == []
    built_rewards = [r for r in folder.load_settings().rewards if isinstance(r, Built)]
    mine, template = built_rewards[-2:]
    assert mine.vars == ({"name": "air", "next": "if(me.on_ground, 0, air + dt)"},)
    assert json.loads(json.dumps(mine.terms)) == [
        {"when": ["in_air", "air > 0.5"], "value": "ball.z / 2044", "once_per_episode": True}
    ]
    assert template.title == "kickoff_race" and template.terms[0]["value"].startswith("if(has_opp")


MIDDLE = """[[rewards]]
name = "built"
title = "my_touch"
weight = 2

# Air time
[[rewards.terms]]
when = ["in_air"]
measure = "one"
weight = 0.1

# The touch
[[rewards.terms]]
measure = "touch"

# Before the goal reward
[[rewards]]
name = "goal"
"""


def test_terms_removed_or_added_keep_each_comment_with_its_own_block() -> None:
    views, _ = rewards.read(MIDDLE)
    assert rewards.written(MIDDLE, entries_of(views)) == MIDDLE

    def edited(change: Any) -> str:
        entries = entries_of(views)
        change(entries[0]["view"]["terms"])
        after = rewards.written(MIDDLE, entries)
        assert config_of(after) and '# Before the goal reward\n[[rewards]]\nname = "goal"' in after, after
        return after

    first_gone = edited(lambda terms: terms.pop(0))
    assert (
        "# Air time" not in first_gone and '# The touch\n[[rewards.terms]]\nmeasure = "touch"' in first_gone
    )
    last_gone = edited(lambda terms: terms.pop())
    assert "# The touch" not in last_gone and "weight = 0.1" in last_gone
    added = edited(lambda terms: terms.append({"measure": "goal"}))
    assert (
        added.index('measure = "touch"') < added.index('measure = "goal"') < added.index("# Before the goal")
    )


TWO_VARS = ADVANCED.replace(
    "# The touch\n",
    '# Seconds up\n[[rewards.vars]]\nname = "air"\nnext = "if(me.on_ground, 0, air + dt)"\n\n# The touch\n',
).replace("chain * ball.z", "ball.z")  # either variable can go


def test_variables_removed_or_added_keep_each_comment_with_its_own_block() -> None:
    views, _ = rewards.read(TWO_VARS)
    assert rewards.written(TWO_VARS, entries_of(views)) == TWO_VARS

    def edited(change: Any) -> str:
        entries = entries_of(views)
        change(entries[0]["view"]["vars"])
        after = rewards.written(TWO_VARS, entries)
        assert config_of(after) and '# The touch\n[[rewards.terms]]\nwhen = ["in_air"' in after, after
        return after

    first_gone = edited(lambda variables: variables.pop(0))
    assert "# Touches in a row" not in first_gone and "# counts" not in first_gone
    assert '# Seconds up\n[[rewards.vars]]\nname = "air"' in first_gone
    last_gone = edited(lambda variables: variables.pop())
    assert "# Seconds up" not in last_gone and 'name = "chain" # counts' in last_gone
    added = edited(lambda variables: variables.append({"name": "up", "next": "me.z"}))
    assert added.index('name = "air"') < added.index('name = "up"') < added.index("# The touch")


# ---------- the screenshot pass's fixes (1.0.6) ----------


def test_a_repeated_title_names_the_title(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    RunFolder.create(runs, "shown", PRESET)
    api = Api(ROOT, runs)
    views = [e["view"] for e in api.rewards("shown")["rewards"]]
    built_view = {**rewards.view({"name": "built"}), "title": "twin", "terms": [{"measure": "touch"}]}
    entries = [{"from": k, "view": v} for k, v in enumerate(views)]
    entries += [{"from": None, "view": dict(built_view)}, {"from": None, "view": dict(built_view)}]
    out = api.stage_rewards("shown", entries)
    assert out["error"] == "Two built rewards are called 'twin': give each built reward a title of its own"
    assert "only once" not in out["error"]


def test_the_builder_page_says_each_thing_once_and_in_place(tmp_path: Path) -> None:
    """A repeated title: one message naming it, however often the reward is edited, and the same under the
    title; a condition added twice is refused with a word; the "not" switch is a button of its own; a
    mistake is pointed at under the formula; a correct formula says so; the summary names each term."""
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    runs = tmp_path / "runs"
    RunFolder.create(runs, "shown", PRESET)
    api = Api(ROOT, runs)
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
            tab.click(".botcard")
            tab.click("[data-tab=rewards]")
            tab.wait_for_selector("text=What each reward pays")
            i = len(api.rewards("shown")["rewards"])
            for _ in range(2):
                tab.click("[data-act=rw-add]")
                tab.fill("#rw-cat-q", "build")
                tab.click("[data-rwadd=built]")
            tab.wait_for_selector(f"#bd-{i + 1}-title")
            assert "An example to start from" in tab.inner_text(f"[data-bdterm='0'] >> nth={0}")
            for _ in range(3):
                tab.fill(f"#bd-{i + 1}-title", "my_reward")
                tab.press(f"#bd-{i + 1}-title", "Tab")
                tab.wait_for_timeout(200)
            toasts = tab.locator(".toast").all_inner_texts()
            assert (
                toasts.count(
                    "Two built rewards are called 'my_reward': give each built reward a title of its own"
                )
                == 1
            )
            assert not any("only once" in t for t in toasts)
            tab.wait_for_function(
                f"document.querySelector(\"[data-bdmsg='{i}:title']\").textContent.includes('my_reward')"
            )
            # A condition already there is not added again.
            tab.select_option(f"#bd-{i}-0-cond", "in_air")
            tab.wait_for_selector("text='In air' is already a condition of term 1.")
            assert tab.locator(f"[data-bdx^='delcond:{i}:0:']").count() == 1
            # "not" is its own switch.
            switch = f"[data-bdx='not:{i}:0:0']"
            assert tab.get_attribute(switch, "aria-pressed") == "false"
            tab.click(switch)
            assert tab.get_attribute(switch, "aria-pressed") == "true" and "NOT: IN AIR" in tab.inner_text(
                ".bd-condwrap >> nth=0"
            )
            # A mistake pointed at; a correct formula says so.
            tab.click(f"[data-bdx='mode:{i}:0:formula']")
            value = f"#bd-{i}-t-0-value"
            tab.wait_for_selector(value)
            tab.fill(value, "me.z +* 2")
            spot = f"[data-bdmsg='{i}:terms[0].value'] .bd-spot"
            tab.wait_for_selector(spot)
            assert tab.inner_text(spot).splitlines() == ["me.z +* 2", "      ^"]
            tab.fill(value, "me.z / 2044")
            tab.wait_for_function(
                f"document.querySelector(\"[data-bdmsg='{i}:terms[0].value']\").textContent === 'Looks good.'"
            )
            tab.click(f"[data-bdx='addterm:{i}']")
            tab.wait_for_selector(f"[data-bdx='delterm:{i}:1']")
            assert "Term 1: pays me.z / 2044" in tab.inner_text("#tab")
            assert not errors, errors
        finally:
            browser.close()
