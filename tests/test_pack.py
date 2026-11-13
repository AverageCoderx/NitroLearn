"""`nitro pack` / `nitro unpack`: a bot carried to another computer and back."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PureWindowsPath

import pytest
import torch

from nitro import pack
from nitro.cli import build_parser
from nitro.recordings import bundled_recordings
from nitro.runs import COPY_SO_FAR, RunFolder, all_settings, fixed_settings

MESHES = Path(os.environ.get("NITRO_TEST_MESHES", Path(__file__).parents[1] / "collision_meshes"))
NEEDS_MESHES = pytest.mark.skipif(not (MESHES / "soccar").is_dir(), reason=f"no collision meshes in {MESHES}")

BOT = """
[paths]
collision_meshes = "../meshes"
[env]
num_arenas = 4
threads = 2
[ppo]
steps_per_iteration = 320
minibatches = 2
[[state_setters]]
name = "kickoff"
[[state_setters]]
name = "replay"
files = ["../data/match.rlpr.zst"]
"""


def recording() -> bytes:
    """A real match's recording (the 1v1 one RocketSim ships), the data file the bot's starts read."""
    found = [path for path in bundled_recordings() if "1v1" in path.name]
    if not found:
        pytest.skip("RocketSim's recordings are not on this computer (nitro fetch-recordings)")
    return found[0].read_bytes()


def names_of_this_computer(archive: Path, *folders: Path) -> list[str]:
    """Every file of the pack `archive` (and every file inside its saves, which are zips) whose bytes name
    one of `folders`, with the folder named."""
    needles = {str(folder) for folder in folders} | {folder.as_posix() for folder in folders}
    found = []
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            data = zipped.read(name)
            parts = {name: data}
            if name.endswith(".pt") and data[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(data)) as save:
                    parts |= {f"{name}:{inner}": save.read(inner) for inner in save.namelist()}
            found += [f"{part}: {n}" for part, d in parts.items() for n in needles if n.encode() in d]
    return found


# The first computer, in a program of its own: the core loads one meshes folder per program, and this one
# trains with the tests' meshes as the second computer.
FIRST = """
import sys
from pathlib import Path, PureWindowsPath
from nitro.runs import RunFolder
from nitro.training import TrainingRun
pc = Path(sys.argv[1])
folder = RunFolder.create(pc / "runs", "bot", pc / "mine" / "bot.toml")
run = TrainingRun(folder.load_settings(), folder)
run.iteration()
run.checkpoint(continue_after=False)
run.metrics.close()
"""


@NEEDS_MESHES
def test_a_packed_bot_carries_on_training_on_another_computer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from nitro.training import TrainingRun

    pc, server = tmp_path / "pc", tmp_path / "server"
    shutil.copytree(MESHES, pc / "meshes")
    (pc / "data").mkdir()
    match = recording()
    (pc / "data" / "match.rlpr.zst").write_bytes(match)
    (pc / "mine").mkdir()
    (pc / "mine" / "bot.toml").write_text(BOT, encoding="utf-8")
    subprocess.run([sys.executable, "-c", FIRST, str(pc)], check=True, timeout=600)
    folder = RunFolder(pc / "runs" / "bot")
    saved = folder.load_latest()
    assert saved is not None

    (folder.root / "watch.html").write_text("a recorded game", encoding="utf-8")
    (folder.root / "live.html").write_text("a live window's page, the arena inside", encoding="utf-8")
    (folder.root / "console.log").write_text(f"$ nitro resume bot --runs {pc / 'runs'}\n", encoding="utf-8")
    (folder.root / "apply_restart.json").write_text('{"restarted": true}', encoding="utf-8")
    (folder.root / "watch.html.partial").write_text("half a page", encoding="utf-8")
    alone, _ = pack.pack(folder.root, tmp_path / "alone.nitropack.zip")
    with zipfile.ZipFile(alone) as zipped:
        assert not json.loads(zipped.read(pack.MANIFEST))["meshes"], "the game's arena files only when asked"
        assert not any(name.startswith("meshes/") for name in zipped.namelist())
    out, notes = pack.pack(folder.root, tmp_path / "bot.nitropack.zip", meshes=True)
    assert notes == []
    with zipfile.ZipFile(out) as zipped:
        manifest = json.loads(zipped.read(pack.MANIFEST))
        names = set(zipped.namelist())
        travelled = zipped.read("run/run.json").decode("utf-8")
    assert "run/watch.html" in names and "run/live.html" not in names, "a live page stays on this computer"
    logs = {"run/console.log", "run/train.log", "run/apply_restart.json", "run/watch.html.partial"}
    assert not logs & names, "the logs of what ran on this computer stay here"
    assert "settings_dir" not in manifest and "settings_dir" not in travelled, "no folder of this computer"
    assert str(pc) not in travelled and str(pc) not in json.dumps(manifest)
    newest = folder.complete_checkpoints()[-1].name
    assert manifest["saves"] == [newest] and f"run/checkpoints/{newest}/state.pt" in names
    assert manifest["files"] == [{"rel": "../data/match.rlpr.zst", "zip": "files/0/match.rlpr.zst"}]
    assert manifest["meshes"] and any(name.startswith("meshes/soccar/") for name in names)
    assert names_of_this_computer(out, tmp_path, Path.home(), pack.ROOT) == []
    shutil.rmtree(pc)  # another computer: nothing of the first one's paths exists there

    done = pack.unpack(out, server / "runs", project=server, meshes=MESHES)
    assert done.folder == (server / "runs" / "bot").resolve() and not done.rust, done.notes
    moved = RunFolder(done.folder)
    info = json.loads(moved.info.read_text(encoding="utf-8"))
    assert Path(info["settings_dir"]) == done.folder / pack.SETTINGS_FILES / "up" and "unpacked" in info
    assert (done.folder / pack.SETTINGS_FILES / "data" / "match.rlpr.zst").read_bytes() == match
    state = moved.load_latest()
    assert state is not None and state["agent_steps"] == saved["agent_steps"]
    config = moved.load_settings(state)
    assert config.paths.collision_meshes == MESHES, "this computer's meshes, not ../meshes from its files"
    caplog.set_level("WARNING", logger="nitro")
    again = TrainingRun(config, moved, state)
    assert "Setting changed" not in caplog.text, "the recording moved computer: no change"
    model = again.trainer.learner.model.state_dict()
    assert all(torch.equal(model[k].cpu(), v) for k, v in saved["learner"]["model"].items())
    row = again.iteration()
    assert int(row["agent_steps"]) > saved["agent_steps"]
    again.checkpoint(continue_after=False)
    again.metrics.close()

    with pytest.raises(pack.PackError, match="already a bot called 'bot'"):
        pack.unpack(out, server / "runs", project=server, meshes=MESHES)
    kept = pack.unpack(out, server / "runs", replace=True, project=server, meshes=MESHES)
    aside = list((server / "runs" / pack.REPLACED).iterdir())
    assert len(aside) == 1 and RunFolder(aside[0]).complete_checkpoints(), "the bot that was there is kept"
    assert any("kept in" in note for note in kept.notes)


def test_what_cannot_travel_is_refused(tmp_path: Path) -> None:
    full = tmp_path / "far" / "match.rlpr"
    full.parent.mkdir()
    full.write_text("x", encoding="utf-8")
    settings = tmp_path / "bot.toml"
    settings.write_text(
        f'[[state_setters]]\nname = "replay"\nfiles = ["{full.as_posix()}"]\n', encoding="utf-8"
    )
    folder = RunFolder.create(tmp_path / "runs", "bot", settings)
    with pytest.raises(pack.PackError, match="full path"):
        pack.pack(folder.root, tmp_path / "x.zip")
    settings.write_text("", encoding="utf-8")
    other = RunFolder.create(tmp_path / "runs", "other", settings)
    (other.root / "running.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    with pytest.raises(pack.PackError, match="is training"):
        pack.pack(other.root, tmp_path / "x.zip")
    (other.root / "running.json").unlink()
    _, notes = pack.pack(other.root, tmp_path / "other.zip", meshes=False)
    assert notes == []
    json_info = json.loads(other.info.read_text(encoding="utf-8"))
    other.info.write_text(json.dumps({**json_info, "start_from": {"run": "x"}}), encoding="utf-8")
    with pytest.raises(pack.PackError, match="no save yet"):
        pack.pack(other.root, tmp_path / "x.zip")
    # A save of a copy of replays still being made: the copy carries on only here, where its replays are.
    copying = other.checkpoints / f"{0:012d}"
    copying.mkdir()
    (copying / "state.pt").write_bytes(b"")
    (copying / COPY_SO_FAR).write_text(json.dumps({"learned": 200, "limit": 1000}), encoding="utf-8")
    with pytest.raises(pack.PackError, match="has not finished copying its replays"):
        pack.pack(other.root, tmp_path / "x.zip")


def test_a_replay_folder_is_named_as_a_folder_and_the_unpacked_bot_hears_where_it_will_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bot's replay_mix.replays names a folder, which a pack cannot carry; it said "no file" although
    the folder was there, and after unpacking the same setting points into the runs folder: nothing there."""
    pc = tmp_path / "pc"
    kickoffs = pc / "replays" / "kickoffs-1v1"
    kickoffs.mkdir(parents=True)
    (kickoffs / "a.nrpl.zst").write_bytes(b"x" * 16)
    settings = pc / "configs" / "my_bot" / "bot.toml"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        '[replay_mix]\nenabled = true\nreplays = ["../../replays/kickoffs-1v1"]\n', encoding="utf-8"
    )
    folder = RunFolder.create(pc / "runs", "mixer", settings)
    files, notes = pack.data_files(folder, folder.load_settings())
    assert files == [] and notes == [
        "replay_mix.replays = '../../replays/kickoffs-1v1' is a folder (1 file, 0.0 MB): packs carry files "
        "only; copy it to the other computer and point replay_mix.replays at it there"
    ]
    out, packed = pack.pack(folder.root, tmp_path / "packed.zip", meshes=False)
    assert packed == notes
    server = tmp_path / "server"
    done = pack.unpack(out, server / "runs", project=server, meshes=MESHES)
    where = (server / "runs" / "replays" / "kickoffs-1v1").resolve()
    assert RunFolder(done.folder).load_settings().replay_mix.replays == (where,)
    assert f"replay_mix will look for its replays in {where}, which is not here" in "\n".join(done.notes)
    assert notes[0] in done.notes, "the pack's own note comes along"


def _zip(path: Path, manifest: dict[str, object], **members: str) -> Path:
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(pack.MANIFEST, json.dumps(manifest))
        for name, text in members.items():
            zipped.writestr(name.replace("__", "/"), text)
    return path


def test_a_pack_that_would_write_outside_its_bot_or_is_newer_is_refused(tmp_path: Path) -> None:
    good = {"format": 1, "nitro_version": "0.0.1", "run": "bot", "files": [], "rust": []}
    runs = tmp_path / "runs"
    cases = [
        (_zip(tmp_path / "a.zip", good, **{"run__..__..__evil": "x"}), "never does"),
        (_zip(tmp_path / "b.zip", good, **{"elsewhere__x": "x"}), "never does"),
        (_zip(tmp_path / "c.zip", {**good, "files": [{"rel": "/etc/x", "zip": "files/0/x"}]}), "never does"),
        (_zip(tmp_path / "d.zip", {**good, "nitro_version": "99.0.0"}), "newer than this one"),
        (_zip(tmp_path / "e.zip", {**good, "run": "../up"}), "cannot name"),
    ]
    for archive, message in cases:
        with pytest.raises(pack.PackError, match=message):
            pack.unpack(archive, runs, project=tmp_path)
    assert not runs.exists() or not any(runs.iterdir()), "nothing was written"


def test_the_bots_own_rust_rewards_join_the_next_build_unless_one_of_that_name_differs(
    tmp_path: Path,
) -> None:
    from nitro import rebuild

    archive = _zip(tmp_path / "r.zip", {}, **{"rust__mine.rs": "fn a() {}", "rust__theirs.rs": "fn b() {}"})
    (tmp_path / "theirs.rs").write_text("fn other() {}", encoding="utf-8")
    rebuild.write_list(tmp_path, [tmp_path / "theirs.rs"])
    with zipfile.ZipFile(archive) as zipped:
        added, notes = pack._add_rust(zipped, ["rust/mine.rs", "rust/theirs.rs"], tmp_path, tmp_path / "copy")
    assert added == [(tmp_path / pack.RUST / "mine.rs").resolve()]
    assert (tmp_path / pack.RUST / "mine.rs").read_text() == "fn a() {}"
    assert [p.name for p in rebuild.listed(tmp_path)] == ["theirs.rs", "mine.rs"]
    assert (tmp_path / "copy" / "theirs.rs").is_file() and "theirs.rs" in notes[0]


def test_pack_and_unpack_are_commands() -> None:
    args = build_parser().parse_args(["pack", "bot", "--all-saves"])
    assert args.run.__name__ == "_pack" and args.all_saves
    args = build_parser().parse_args(["unpack", "bot.nitropack.zip", "--name", "b2", "--replace"])
    assert args.run.__name__ == "_unpack" and args.name == "b2" and args.replace


def test_the_dashboard_packs_a_bot_and_adds_a_packed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nitro.dashboard.api import Api
    from test_dashboard import ROOT, make_bot

    runs = tmp_path / "runs"
    folder = make_bot(runs, "shown", saves=1)
    api = Api(ROOT, runs)
    api._project = tmp_path  # its packs folder
    asked: list[tuple[object, ...]] = []
    monkeypatch.setattr(api._jobs, "start", lambda *args: asked.append(args))
    made = api.pack_bot("shown")
    assert made["started"] and Path(made["file"]).parent == tmp_path / "packs"
    assert Path(made["file"]).name.startswith("shown-") and made["file"].endswith(pack.SUFFIX)
    assert asked[0][1] == "pack" and asked[0][2][:2] == ["pack", "shown"]
    archive, _ = pack.pack(folder.root, tmp_path / "shown.nitropack.zip")
    assert api.unpack_bot(str(archive)) == {"exists": "shown", "path": str(archive), "free": "shown-2"}
    assert "at most 64 characters" in api.unpack_bot(str(archive), False, "x" * 65)["error"]
    beside = api.unpack_bot(str(archive), False, "shown-2")
    assert beside["unpacked"] == "shown-2" and not (runs / pack.REPLACED).exists(), "added beside it"
    assert api.unpack_bot(str(archive), False, "shown-2")["error"] == "there is already a bot called shown-2"
    added = api.unpack_bot(str(archive), True)
    assert added["unpacked"] == "shown" and len(list((runs / pack.REPLACED).iterdir())) == 1
    assert [bot["name"] for bot in api.home()["bots"]] == ["shown", "shown-2"]
    assert "--no-meshes" in asked[0][2], "the game's arena files stay out of a pack made here"


def test_a_file_that_only_moved_computer_is_no_changed_setting() -> None:
    from nitro.runs import changed_settings

    def setter(file: str) -> dict[str, object]:
        return {"state_setters": [{"name": "replay", "files": [file]}]}

    pc = r"C:\bots\recordings\m.rlpr"
    server = "/root/runs/bot/settings-files/recordings/m.rlpr"
    assert changed_settings(setter(pc), setter(server), moved=True) == []
    packed, saved_there = "m.rlpr", "settings-files/recordings/m.rlpr"  # as a pack and a save there name it
    assert changed_settings(setter(packed), setter(saved_there), moved=True) == []
    assert changed_settings(setter(pc), setter("/root/other.rlpr"), moved=True) == [
        rf"state_setters[0].files[0]: {pc} -> /root/other.rlpr"
    ]
    assert len(changed_settings(setter(pc), setter(server))) == 1, "only an unpacked bot's files move"


def _raw_zip(path: Path, manifest: dict[str, object], members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(pack.MANIFEST, json.dumps(manifest))
        for name, text in members.items():
            zipped.writestr(name, text)
    return path


def test_no_name_in_a_pack_can_reach_outside_the_bots_folders(tmp_path: Path) -> None:
    """Crafted packs: '//' and drive letters after a section, a section a pack never has, in a Rust file's
    name or a data file's path; each refused before anything is written."""
    good = {"format": 1, "nitro_version": "0.0.1", "run": "bot", "files": [], "rust": []}
    victim = tmp_path / "outside" / "victim.txt"
    full = victim.as_posix().lstrip("/")
    cases = [
        (good, {f"run//{full}": "x"}),
        (good, {"run/C:/Windows/Startup/x.bat": "x"}),
        (good, {"run/a\\..\\..\\x": "x"}),
        (good, {f"specialists/foo//{full}": "x"}),
        ({**good, "rust": ["rust/D:x.rs"]}, {"rust/D:x.rs": "x"}),
        (good, {"meshes/../x": "x"}),
        ({**good, "files": [{"rel": "a/../../../x", "zip": "files/0/x"}]}, {"files/0/x": "x"}),
        ({**good, "files": [{"rel": "C:/x", "zip": "files/0/x"}]}, {"files/0/x": "x"}),
    ]
    runs = tmp_path / "runs"
    for k, (manifest, members) in enumerate(cases):
        with pytest.raises(pack.PackError, match="never does"):
            pack.unpack(_raw_zip(tmp_path / f"{k}.zip", manifest, members), runs, project=tmp_path)
    assert not victim.exists() and not (tmp_path / "x").exists()
    assert not runs.exists() or not any(runs.iterdir()), "nothing was written"
    (tmp_path / "half.zip").write_bytes(b"PK half a download")
    with pytest.raises(pack.PackError, match="not a NitroLearn pack"):
        pack.unpack(tmp_path / "half.zip", runs, project=tmp_path)


def test_a_pack_leaves_the_logs_and_the_game_files_on_this_computer(tmp_path: Path) -> None:
    from test_dashboard import make_bot

    (tmp_path / "meshes" / "soccar").mkdir(parents=True)
    (tmp_path / "meshes" / "soccar" / "mesh_0.cmf").write_text("arena", encoding="utf-8")
    folder = make_bot(tmp_path / "runs", "bot", saves=1)
    text = folder.settings.read_text(encoding="utf-8")
    folder.settings.write_text(text.replace(MESHES.as_posix(), (tmp_path / "meshes").as_posix()))
    (folder.root / "console.log").write_text(f"$ nitro resume bot --runs {tmp_path}\n", encoding="utf-8")
    (folder.root / "train.log").write_text("INFO Networks run on cpu (FP32).\n", encoding="utf-8")
    (folder.root / "apply_restart.json").write_text('{"restarted": true}', encoding="utf-8")
    (folder.root / "report.html.partial").write_text("half a page", encoding="utf-8")
    (folder.root / "notes.json").write_text('{"bot": "my note"}', encoding="utf-8")
    out, _ = pack.pack(folder.root, tmp_path / "bot.nitropack.zip")
    with zipfile.ZipFile(out) as zipped:
        names = set(zipped.namelist())
        assert not json.loads(zipped.read(pack.MANIFEST))["meshes"], "the game's arena files only when asked"
    assert not any(name.startswith("meshes/") for name in names)
    logs = {"run/console.log", "run/train.log", "run/apply_restart.json", "run/report.html.partial"}
    assert not logs & names, "the logs of what ran on this computer stay here"
    assert {"run/notes.json", "run/settings.toml", "run/run.json"} <= names
    with_meshes, _ = pack.pack(folder.root, tmp_path / "meshes.nitropack.zip", meshes=True)
    with zipfile.ZipFile(with_meshes) as zipped:
        assert "meshes/soccar/mesh_0.cmf" in zipped.namelist()


def test_unpacking_never_overwrites_what_is_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nitro import rebuild
    from test_dashboard import make_bot

    (tmp_path / "pc" / "meshes" / "soccar").mkdir(parents=True)
    (tmp_path / "pc" / "meshes" / "soccar" / "mesh_0.cmf").write_text("arena", encoding="utf-8")
    folder = make_bot(tmp_path / "pc" / "runs", "bot", saves=1)
    text = folder.settings.read_text(encoding="utf-8")
    folder.settings.write_text(text.replace(MESHES.as_posix(), (tmp_path / "pc" / "meshes").as_posix()))
    (folder.root / pack.WANDB).write_text("{}", encoding="utf-8")
    out, _ = pack.pack(folder.root, tmp_path / "bot.nitropack.zip", meshes=True)
    with pytest.raises(pack.PackError, match="already exists"):
        pack.pack(folder.root, out)

    project, runs = tmp_path / "server", tmp_path / "server" / "runs"
    done = pack.unpack(out, runs, project=project)
    assert (project / "collision_meshes" / "soccar" / "mesh_0.cmf").read_text() == "arena", (
        "meshes came along"
    )
    assert (done.folder / pack.WANDB).is_file()
    copy = pack.unpack(out, runs, name="bot-2", project=project)
    assert not (copy.folder / pack.WANDB).exists(), "a copy under another name gets a W&B run of its own"
    (project / "collision_meshes" / "soccar" / "mesh_0.cmf").write_text("mine", encoding="utf-8")
    pack.unpack(out, runs, name="bot-3", project=project)
    assert (project / "collision_meshes" / "soccar" / "mesh_0.cmf").read_text() == "mine"

    (done.folder / "running.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    with pytest.raises(pack.PackError, match="training here"):
        pack.unpack(out, runs, replace=True, project=project)

    (project / pack.RUST).mkdir(parents=True)
    (project / pack.RUST / "mine.rs").write_text("fn mine() {}", encoding="utf-8")
    with zipfile.ZipFile(_raw_zip(tmp_path / "r.zip", {}, {"rust/mine.rs": "fn theirs() {}"})) as zipped:
        added, notes = pack._add_rust(zipped, ["rust/mine.rs"], project, tmp_path / "copy")
    assert added == [] and "mine.rs" in notes[0] and rebuild.listed(project) == []
    assert (project / pack.RUST / "mine.rs").read_text() == "fn mine() {}"


def test_what_pack_cannot_carry_is_refused_with_the_setting_named(tmp_path: Path) -> None:
    from test_dashboard import make_bot

    # The file `recordings\m.rlpr` names: one file of that name on Linux, recordings/m.rlpr on Windows, so
    # the settings load either way and pack decides.
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings\\m.rlpr").write_text("x", encoding="utf-8")
    settings = tmp_path / "bot.toml"
    settings.write_text(
        '[[state_setters]]\nname = "replay"\nfiles = ["recordings\\\\m.rlpr"]\n', encoding="utf-8"
    )
    folder = RunFolder.create(tmp_path / "runs", "bot", settings)
    with pytest.raises(pack.PackError, match="written with"):
        pack.pack(folder.root, tmp_path / "bot.zip")
    linked = make_bot(tmp_path / "runs", "linked", saves=1)
    elsewhere = tmp_path / "other-drive" / "checkpoints"
    elsewhere.parent.mkdir()
    linked.checkpoints.rename(elsewhere)
    try:
        linked.checkpoints.symlink_to(elsewhere, target_is_directory=True)
    except OSError as err:  # Windows without developer mode may not make links
        pytest.skip(f"the rest needs a link, which this machine cannot make ({err})")
    out, _ = pack.pack(linked.root, tmp_path / "linked.zip")
    assert any(name.endswith("/state.pt") for name in zipfile.ZipFile(out).namelist()), "a linked folder too"


def test_a_linked_recordings_folder_travels_as_the_settings_name_it_and_wandbs_folder_stays(
    tmp_path: Path,
) -> None:
    """Second review: a data file in a linked folder was packed under the link's target, so the
    unpacked settings did not find it. W&B's own folder (its files name this computer) is not packed; a
    session W&B kept offline is said to stay here."""
    from test_dashboard import make_bot

    shared = tmp_path / "shared-recordings"
    shared.mkdir()
    (shared / "m.rlpr").write_text("x", encoding="utf-8")
    mine = tmp_path / "mine"
    mine.mkdir()
    try:
        (mine / "recordings").symlink_to(shared, target_is_directory=True)
    except OSError as err:  # Windows without developer mode may not make links
        pytest.skip(f"this machine cannot make a link ({err})")
    folder = make_bot(tmp_path / "pc", "bot", saves=1)
    info = json.loads(folder.info.read_text(encoding="utf-8"))
    folder.info.write_text(json.dumps({**info, "settings_dir": str(mine)}), encoding="utf-8")
    text = folder.settings.read_text(encoding="utf-8")
    folder.settings.write_text(text + '\n[[state_setters]]\nname = "replay"\nfiles = ["recordings/m.rlpr"]\n')
    files, _ = pack.data_files(folder, folder.load_settings())
    assert [rel for _, rel in files] == ["recordings/m.rlpr"]
    wandb = folder.root / "wandb"
    (wandb / "run-1").mkdir(parents=True)
    (wandb / "run-1" / "run-1.wandb").write_bytes(b"history")
    (wandb / "latest-run").symlink_to(wandb / "run-1", target_is_directory=True)
    (wandb / "offline-run-2").mkdir()
    out, notes = pack.pack(folder.root, tmp_path / "bot.zip")
    names = zipfile.ZipFile(out).namelist()
    assert not any(n.startswith("run/wandb") for n in names)
    assert "1 W&B session kept offline stays on this computer" in notes[0] and "nitro wandb bot" in notes[0]
    done = pack.unpack(out, tmp_path / "server" / "runs", project=tmp_path / "server")
    assert RunFolder(done.folder).load_settings().state_setters, "the settings find the file there"


def test_a_failed_replace_changes_nothing_other_bots_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second review: Windows refuses to move a bot's folder while a program has a file open in it;
    unpack had already changed the build list by then."""
    from nitro import rebuild
    from test_dashboard import make_bot

    folder = make_bot(tmp_path / "pc", "bot", saves=1)
    plain, _ = pack.pack(folder.root, tmp_path / "plain.zip")
    archive = tmp_path / "bot.zip"
    with zipfile.ZipFile(plain) as source, zipfile.ZipFile(archive, "w") as zipped:
        for member in source.namelist():
            data = source.read(member)
            if member == pack.MANIFEST:
                data = json.dumps({**json.loads(data), "rust": ["rust/mine.rs"]}).encode()
            zipped.writestr(member, data)
        zipped.writestr("rust/mine.rs", "fn a() {}")
    project, runs = tmp_path / "server", tmp_path / "server" / "runs"
    pack.unpack(plain, runs, project=project)
    real = Path.rename

    def refuse(self: Path, target: Path) -> Path:
        if self == runs.resolve() / "bot":
            raise PermissionError("Access is denied")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", refuse)
    with pytest.raises(PermissionError):
        pack.unpack(archive, runs, replace=True, project=project)
    assert rebuild.listed(project) == [] and not (project / pack.RUST).exists()
    monkeypatch.setattr(Path, "rename", real)
    done = pack.unpack(archive, runs, replace=True, project=project)
    assert [p.name for p in rebuild.listed(project)] == ["mine.rs"] and done.rust


def test_a_pack_from_other_code_is_refused_when_a_setting_is_unknown_here(tmp_path: Path) -> None:
    """Second review: the version number stays 1.0.0 between releases, so packs carry the settings'
    fingerprint; a pack whose settings this NitroLearn cannot read is refused before anything moves."""
    from test_dashboard import make_bot

    folder = make_bot(tmp_path / "pc", "bot", saves=1)
    plain, _ = pack.pack(folder.root, tmp_path / "plain.zip")
    assert pack.manifest(plain)["settings_fingerprint"] == pack.settings_fingerprint()

    def remade(name: str, settings_extra: str) -> Path:
        out = tmp_path / name
        with zipfile.ZipFile(plain) as source, zipfile.ZipFile(out, "w") as zipped:
            for member in source.namelist():
                data = source.read(member)
                if member == pack.MANIFEST:
                    data = json.dumps(
                        {**json.loads(data), "settings_fingerprint": "other", "id": name}
                    ).encode()
                if member == "run/settings.toml":
                    data += settings_extra.encode()
                zipped.writestr(member, data)
        return out

    runs = tmp_path / "server" / "runs"
    with pytest.raises(pack.PackError, match="newer NitroLearn: 'future' is not a setting"):
        pack.unpack(remade("newer.zip", "\n[future]\nx = 1\n"), runs, project=tmp_path / "server")
    assert not runs.exists() or not any(runs.iterdir())
    done = pack.unpack(remade("other.zip", ""), runs, project=tmp_path / "server")
    assert any("different version of NitroLearn" in note for note in done.notes)
    assert pack.unpacked_here(tmp_path / "other.zip", runs)
    assert not pack.unpacked_here(plain, runs), "another pack of that bot"


def test_a_reward_whose_rust_file_is_gone_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second review: it still trains here, built in, but its source cannot travel."""
    import dataclasses

    from nitro.config.rewards import Custom, CustomKind
    from nitro.config.schema import NitroConfig
    from test_dashboard import make_bot

    gone = tmp_path / "Downloads" / "my_touch.rs"
    kind = CustomKind("my_touch", "", False, str(gone), ())
    monkeypatch.setattr(pack, "custom_kinds", lambda: {"my_touch": kind})
    config = dataclasses.replace(NitroConfig(), rewards=(Custom(name="my_touch"),))
    assert pack.missing_rust_files(config) == [("my_touch", gone)]
    folder = make_bot(tmp_path / "pc", "bot", saves=1)
    monkeypatch.setattr(RunFolder, "load_settings", lambda self: config)
    with pytest.raises(pack.PackError, match=r"'my_touch' is built from .*my_touch\.rs, which is not there"):
        pack.pack(folder.root, tmp_path / "bot.zip")


SETUP = Path(__file__).resolve().parents[1] / "tools" / "cloud_setup.sh"


def _stubs(folder: Path, **commands: str) -> Path:
    folder.mkdir()
    for name, body in commands.items():
        (folder / name).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        (folder / name).chmod(0o755)
    return folder


@pytest.mark.skipif(sys.platform == "win32", reason="the setup script is for Linux machines")
def test_the_setup_script_says_what_it_does_on_a_machine_without_rust(tmp_path: Path) -> None:
    """Second review: with no cargo, the Rust check ended the script without a word."""
    stubs = _stubs(
        tmp_path / "bin",
        **{name: "exit 0" for name in ("c++", "tmux", "unzip", "nano")},
        curl="echo 'no network in this test' >&2; exit 7",
    )
    path = f"{stubs}:/usr/bin:/bin"
    assert not any((Path(p) / "cargo").exists() for p in path.split(":")), "a machine without Rust"
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "PATH": path}
    done = subprocess.run(
        ["bash", str(SETUP), "--cpu"], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode != 0 and "== Installing Rust" in done.stdout, done.stdout + done.stderr
    assert "no network in this test" in done.stderr
    elsewhere = tmp_path / "here"
    elsewhere.mkdir()
    (elsewhere / "bot.nitropack.zip").write_bytes(b"x")
    found = subprocess.run(
        ["bash", str(SETUP), "--cpu", "bot.nitropack.zip"],
        env=env,
        cwd=elsewhere,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "No pack file" not in found.stderr, "a pack's path is from the folder one is in"
    missing = subprocess.run(
        ["bash", str(SETUP), "nothing.zip"],
        env=env,
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 1 and "No pack file at nothing.zip" in missing.stderr


def test_a_pack_names_no_folder_of_this_computer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file of a pack, the saves' recorded settings and what the copy of replays was made with
    included, holds no full path of this computer: a path inside the bot's folder, the folder its settings
    came from or NitroLearn's goes relative to it, any other by its name alone; run.json and pack.json
    travel without the folders an unpack sets. The unpacked bot loads its settings and its save."""
    from test_dashboard import make_bot

    pc = tmp_path / "pc"  # the bot's settings came from here
    pc.mkdir()
    demos, meshes = tmp_path / "Documents" / "Demos", tmp_path / "meshes"
    folder = make_bot(pc / "runs", "bot", saves=1)
    # NitroLearn's folder stands elsewhere, so the test's folders are outside it wherever the temporary
    # folder is (tools/env.ps1 puts it inside NitroLearn's).
    real_root = pack.ROOT
    monkeypatch.setattr(pack, "ROOT", tmp_path / "NitroLearn")
    monkeypatch.setattr("nitro.runs.PROJECT", tmp_path / "NitroLearn")
    text = folder.settings.read_text(encoding="utf-8").replace(MESHES.as_posix(), meshes.as_posix())
    folder.settings.write_text(
        f"# my replays: {demos}\n{text}\n[pretrain]\nreplays = ['{demos.as_posix()}']\n", encoding="utf-8"
    )
    info = json.loads(folder.info.read_text(encoding="utf-8"))
    start = {"run": str(pc / "runs" / "older"), "at": 0}
    folder.info.write_text(
        json.dumps({**info, "collision_meshes": str(meshes), "start_from": start}), encoding="utf-8"
    )
    config = folder.load_settings()
    made_with = {"replays": str(demos), "limit": 30}
    save = {
        "agent_steps": 1000,
        "learner": {"model": {"policy.0.weight": torch.ones(2, 2)}},
        "settings": fixed_settings(config),
        "all_settings": all_settings(config, folder.root),
        "pretrained": {"made_with": made_with, "recordings": [str(pc / "rec" / "a.rlpr")]},
    }
    torch.save(save, folder.checkpoints / f"{1000:012d}" / "state.pt")
    (folder.root / "pretrained").mkdir()
    torch.save({**save, "agent_steps": 0}, folder.root / "pretrained" / "state.pt")
    (folder.root / "pretrained" / "pretrain.json").write_text(json.dumps({"made_with": made_with}))
    (folder.root / "train.log").write_text(f"Progress: {folder.root / 'report.html'} from {Path.home()}\n")
    (folder.root / "pool").mkdir()
    torch.save({"policy.0.weight": torch.zeros(2)}, folder.root / "pool" / "1.pt")
    unchanged = (folder.root / "pool" / "1.pt").read_bytes()

    out, notes = pack.pack(folder.root, tmp_path / "bot.zip", meshes=False)
    assert notes == []
    assert names_of_this_computer(out, tmp_path, Path.home(), real_root) == []
    with zipfile.ZipFile(out) as zipped:
        carried = zipped.read("run/run.json").decode("utf-8") + zipped.read(pack.MANIFEST).decode("utf-8")
        assert zipped.read("run/pool/1.pt") == unchanged, "a file naming no folder travels as it is"
    assert "settings_dir" not in carried and "collision_meshes" not in carried
    done = pack.unpack(out, tmp_path / "server" / "runs", project=tmp_path / "server")
    moved = RunFolder(done.folder)
    info = json.loads(moved.info.read_text(encoding="utf-8"))
    assert (
        Path(info["settings_dir"]).is_relative_to(done.folder) and info["start_from"]["run"] == "runs/older"
    )
    state = moved.load_latest()
    assert state is not None and state["agent_steps"] == 1000
    assert torch.equal(state["learner"]["model"]["policy.0.weight"], torch.ones(2, 2))
    assert state["all_settings"]["paths.collision_meshes"] == "meshes", "by its name: outside every folder"
    assert state["all_settings"]["pretrain.replays"] == ["Demos"]
    recorded = {"made_with": {"replays": "Demos", "limit": 30}, "recordings": ["rec/a.rlpr"]}
    assert state["pretrained"] == recorded, "relative to where the settings came from, else by name"
    config = moved.load_settings(state)
    assert config.pretrain.replays == (Path(info["settings_dir"]) / "Demos",), "a name, read as relative"
    assert "# my replays: Demos" in moved.settings.read_text(encoding="utf-8")


def test_a_pack_names_no_folder_of_a_windows_computer() -> None:
    """The same with a Windows computer's folders (fed in here as text): a path written with either slash,
    with its backslashes doubled inside "...", in another letter case or as a share on a server is carried
    relative to the bot's folder, the folder its settings came from or NitroLearn's, else by its name; the
    settings file's full paths in every form TOML writes them, a comment's included."""
    import tomllib

    project = PureWindowsPath(r"D:\Games\NitroLearn")
    bot, mine = project / "runs" / "bot", project / "my-settings"
    here = pack._Here.made([bot, mine, project], PureWindowsPath(r"C:\Users\Sam"))

    assert here.path(r"D:\Games\NitroLearn\my-settings\rec\a.rlpr") == "rec/a.rlpr"
    assert here.path("d:/games/nitrolearn/runs/bot/report.html") == "report.html"
    assert here.path(r"D:\GAMES\NitroLearn\collision_meshes") == "collision_meshes"
    assert here.path(r"E:\Replays\Demos" + "\\") == "Demos"
    assert here.path(r"\\server\share\replays") == "replays"
    assert here.value({"recordings": [r"d:\Games\NitroLearn\my-settings\rec\a.rlpr"], "steps": 3}) == {
        "recordings": ["rec/a.rlpr"],
        "steps": 3,
    }
    log = (
        r"Progress: D:\Games\NitroLearn\runs\bot\report.html from C:\Users\Sam" + "\n"
        r"saved in d:/games/nitrolearn/runs/bot/checkpoints, read c:\users\sam\Documents\x.toml" + "\n"
        r'{"file": "D:\\Games\\NitroLearn\\my-settings\\a.toml"}'
    )
    assert here.text(log) == (
        r"Progress: bot\report.html from ~" + "\n"
        r"saved in bot/checkpoints, read ~\Documents\x.toml" + "\n"
        r'{"file": "my-settings\\a.toml"}'
    )
    unchanged = r"nothing of D:\Elsewhere here"
    assert here.text(unchanged) is unchanged

    settings = (
        r"# my replays: D:\Temp\pytest-of-sam\Demos" + "\n"
        r"# the meshes: d:/temp/pytest-of-sam/meshes" + "\n"
        "[paths]\n"
        r'collision_meshes = "D:\\Temp\\pytest-of-sam\\meshes"' + "\n"
        "[pretrain]\n"
        r"replays = ['D:/Temp/pytest-of-sam/Demos', "
        r'"d:\\games\\nitrolearn\\my-settings\\replays\\ranked", '
        r"'C:\Users\Sam\Documents\Old']" + "\n"
    )
    carried = pack._carried_settings(settings, here)
    assert carried.startswith("# my replays: Demos\n# the meshes: meshes\n"), carried
    assert tomllib.loads(carried) == {
        "paths": {"collision_meshes": "meshes"},
        "pretrain": {"replays": ["Demos", "replays/ranked", "Old"]},
    }
    for name in ("D:", "C:", "Temp", "pytest-of-sam", "Users", "Games", "\\"):
        assert name.lower() not in carried.lower(), name
