"""The dashboard's resources library: files recognised by what they hold, refused
with a reason when NitroLearn cannot use them, kept as a list with notes, and offered where they fit
(settings files in the new-bot form, reward sets in the Rewards tab)."""

from __future__ import annotations

import shutil
from pathlib import Path

from browser import open_bot, open_chromium, playwright_or_skip
from nitro.dashboard import resources
from nitro.dashboard.api import Api
from nitro.runs import RunFolder

ROOT = Path(__file__).parents[1]
PRESETS = ROOT / "configs" / "presets"
REWARD_SET = (
    '# for aerials\n[[rewards]]\nname = "air"\nweight = 1\n\n[[rewards]]\nname = "touch_ball"\nweight = 10\n'
)
RUST = "pub struct Flip;\nimpl nitro_core::custom::Reward for Flip {\n    fn value(&self) -> f32 { 1.0 }\n}\n"


def project(tmp_path: Path) -> Path:
    """A project folder of its own (my-settings is written to), with the real presets."""
    root = tmp_path / "project"
    shutil.copytree(PRESETS, root / "configs" / "presets")
    return root


def files(tmp_path: Path) -> dict[str, Path]:
    folder = tmp_path / "files"
    folder.mkdir()
    made = {
        "settings.toml": (PRESETS / "1v1_starter.toml").read_text(encoding="utf-8"),
        "set.toml": REWARD_SET,
        "flip.rs": RUST,
        "obs.rs": "impl ObsBuilder for MyObs {}\n",
        "parser.rs": "impl crate::ActionParser for MyActions {}\n",
        "plain.rs": "fn main() {}\n",
        "their_reward.cpp": "class MyReward : public RLGC::Reward {};\n",
        "reward.py": "def reward(): return 1\n",
        "odd.toml": "colour = 1\n",
        "broken.toml": "[env]\nteam_size = 9\n",
        "notes.txt": "hello\n",
    }
    for name, text in made.items():
        (folder / name).write_text(text, encoding="utf-8")
    return {name: folder / name for name in made}


def test_files_are_recognised_by_what_they_hold_and_refused_with_a_reason(tmp_path: Path) -> None:
    f = files(tmp_path)
    kinds = {name: resources.detect(path) for name, path in f.items()}
    assert kinds["settings.toml"]["kind"] == "settings" and kinds["settings.toml"]["about"].startswith(
        "1v1, tick skip 4"
    )
    assert (
        kinds["set.toml"]["kind"] == "rewards" and kinds["set.toml"]["about"] == "2 rewards: air, touch_ball"
    )
    assert [kinds[n]["kind"] for n in ("flip.rs", "obs.rs", "parser.rs")] == [
        "rust_reward",
        "rust_observation",
        "rust_actions",
    ]
    assert kinds["broken.toml"]["kind"] == "settings" and "team_size" in kinds["broken.toml"]["problem"]
    for name, words in (
        ("plain.rs", "implements no Reward"),
        ("their_reward.cpp", "C++ Reward"),
        ("reward.py", "slow training"),
        ("odd.toml", "it has colour"),
        ("notes.txt", "takes .toml"),
    ):
        assert not kinds[name]["kind"] and words in kinds[name]["problem"], (name, kinds[name])
    assert "no longer there" in resources.detect(tmp_path / "gone.toml")["problem"]


def test_the_library_keeps_paths_and_notes_and_refuses_what_it_cannot_use(tmp_path: Path) -> None:
    f = files(tmp_path)
    api = Api(project(tmp_path), tmp_path / "runs")
    out = api.add_resource_paths([str(p) for p in f.values()])
    assert sorted(e["kind"] for e in out["added"]) == sorted(
        ["settings", "rewards", "rust_reward", "rust_observation", "rust_actions", "settings"]
    )
    assert len(out["refused"]) == 5 and any("reward.py" in r for r in out["refused"])
    again = api.add_resource_paths([str(f["set.toml"])])
    listed = api.resources()["files"]
    assert len(listed) == 6 and again["added"][0]["id"] in {e["id"] for e in listed}, "added once"
    ident = next(e["id"] for e in listed if e["kind"] == "rewards")
    assert api.set_library_note("file:" + ident, "aerials")["notes"] == {"file:" + ident: "aerials"}
    assert (
        api.set_library_note("reward:touch_ball", "early only")["notes"]["reward:touch_ball"] == "early only"
    )
    assert "error" in api.set_library_note("../escape", "x")
    assert [r["name"] for r in api.reward_set(ident)["rewards"]] == ["air", "touch_ball"]
    api.remove_resource(ident)
    assert ident not in {e["id"] for e in api.resources()["files"]} and f["set.toml"].is_file()
    assert "file:" + ident not in api.resources()["notes"]
    built_in = api.resources()
    assert len([k for k in built_in["rewards"] if not k.get("own")]) == 13
    assert "nitro_v2" in built_in["observation"]["options"]


def test_the_reward_sets_that_come_with_it_are_listed_first_and_stay(tmp_path: Path) -> None:
    """configs/reward-sets (the sets that come with NitroLearn) are in the library without being added."""
    root = project(tmp_path)
    (root / "configs" / "reward-sets").mkdir()
    (root / "configs" / "reward-sets" / "aerial_bot.toml").write_text(REWARD_SET, encoding="utf-8")
    moves_set = "# A trick bot's rewards.\n# group: moves\n" + REWARD_SET
    (root / "configs" / "reward-sets" / "trick_bot.toml").write_text(moves_set, encoding="utf-8")
    api = Api(root, tmp_path / "runs")
    api.add_resource_paths([str(files(tmp_path)["set.toml"])])
    listed = api.resources()["files"]
    shipped, trick, added = listed
    assert (shipped["id"], shipped["kind"], shipped["title"]) == ("set-aerial_bot", "rewards", "aerial bot")
    assert shipped["built_in"] and not added.get("built_in")
    assert trick["trick_move"] and not shipped["trick_move"], "a set whose header has `# group: moves`"
    assert [r["name"] for r in api.reward_set("set-aerial_bot")["rewards"]] == ["air", "touch_ball"]
    assert "stays on the list" in api.remove_resource("set-aerial_bot")["error"]
    api.set_library_note("file:set-aerial_bot", "for later")
    assert api.resources()["files"][0]["note"] == "for later"


def test_settings_files_start_new_bots_and_reward_sets_are_saved_to_my_settings(tmp_path: Path) -> None:
    f = files(tmp_path)
    root = project(tmp_path)
    (root / "my-settings").mkdir()
    shutil.copy(PRESETS / "3v3_starter.toml", root / "my-settings" / "mine.toml")
    runs = tmp_path / "runs"
    api = Api(root, runs)
    api.add_resource_paths([str(f["settings.toml"]), str(f["broken.toml"])])
    sources = api.settings_sources()["sources"]
    assert [s["group"] for s in sources] == ["preset"] * 4 + ["my-settings", "library"]
    assert sources[3]["id"] == "gco_1v1_starter" and "GCO learning signals" in sources[3]["about"]
    mine, added = sources[4], sources[5]
    assert mine["team_size"] == 3 and added["team_size"] == 1, "the broken file is not offered"
    assert api.wizard(mine["id"])["team_size"] == 3
    made = api.create_bot("mine-bot", mine["id"], {"ppo.policy_lr": "2e-4"})
    assert made["made"] == "mine-bot" and RunFolder(runs / "mine-bot").load_settings().env.team_size == 3
    assert "error" in api.wizard("file:" + str(f["odd.toml"])), "only offered files can start a bot"
    RunFolder.create(runs, "b", PRESETS / "2v2_starter.toml")
    entries = [{"from": e["from"], "view": e["view"]} for e in api.rewards("b")["rewards"]]
    saved = api.save_reward_set("b", "my favourite", entries[:3])
    file = Path(saved["saved"])
    assert (
        file == root / "my-settings" / "reward-sets" / "my_favourite.toml"
        and saved["entry"]["kind"] == "rewards"
    )
    text = file.read_text(encoding="utf-8")
    assert text.startswith("# Reward set 'my favourite', saved from b") and "\n\n\n" not in text
    assert [r["name"] for r in api.reward_set(saved["entry"]["id"])["rewards"]] == [
        "air",
        "face_ball",
        "velocity_player_to_ball",
    ]
    assert "already" in api.save_reward_set("b", "my favourite", entries)["error"]
    assert "error" in api.save_reward_set("b", "../../escape", entries)


def test_the_resources_page_and_reward_sets_work_in_the_page(tmp_path: Path) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.app import page

    f = files(tmp_path)
    runs = tmp_path / "runs"
    folder = RunFolder.create(runs, "shown", PRESETS / "2v2_starter.toml")
    api = Api(project(tmp_path), runs)
    api.add_resource_paths([str(f["set.toml"]), str(f["flip.rs"]), str(f["settings.toml"])])
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
            tab.click("[data-go=resources]")
            tab.wait_for_selector("text=Your Rust files")
            tab.click("[data-resnote='reward:goal']")
            tab.fill("#res-note-reward-goal", "big but rare")
            tab.click("[data-resnotesave='reward:goal']")
            tab.wait_for_selector(".notetext >> text=big but rare")
            assert api.resources()["notes"]["reward:goal"] == "big but rare"
            tab.fill("#res-search", "flip.rs")
            tab.wait_for_function("document.querySelectorAll('#main .setting').length === 1")
            open_bot(tab)
            tab.click("[data-tab=rewards]")
            tab.click("[data-act=rw-add]")
            heads = tab.locator("#rw-catalog .catalog-head").all_inner_texts()
            assert [h.split(" (")[0].split(" 1")[0] for h in heads[:3]] == [
                *("Built in", "Trick moves", "More rewards"),
            ]
            assert heads[1].endswith(" 17"), "the trick-move rewards in their own group"
            tab.click("[data-close]")
            tab.click("[data-act=rw-load]")
            tab.click("[data-rwset][data-how=add]")
            tab.wait_for_selector("#pending:not([hidden])")
            assert api.bot("shown")["reward_changes"] == [
                "touch_ball: added, weight 10",
            ], "air is in the bot already: a reward is added only once"
            assert "this bot has them already: Air" in tab.inner_text("#toasts")
            tab.click(".nav [data-go=home]")
            tab.click("[data-act=newbot]")
            tab.wait_for_selector("#wz-source option[value^='file:']", state="attached")
            assert folder.settings.is_file()
        finally:
            browser.close()
    assert errors == []
