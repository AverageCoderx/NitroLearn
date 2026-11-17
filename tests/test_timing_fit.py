"""The replay conversion's timing fit: on for new bots and new conversions, off for bots made before it,
fitted conversions in a store of their own (never mixed with unfitted ones), replays converted before left
alone unless asked, and the copy's settings saying which it used. The fit itself is tested in
crates/nitro_replicar (cargo test, with the collision meshes)."""

from __future__ import annotations

import dataclasses
import json
import struct
import sys
from pathlib import Path

import pytest

from nitro import cli, pretrain, replay_mix, replays
from nitro.config import default_config
from nitro.config.schema import NitroConfig, PretrainConfig
from nitro.runs import NEW_DEFAULTS, RunFolder
from test_pretrain_large import SHELL, large, stand_in_lean_tool, tools_and_replays

ROOT = Path(__file__).resolve().parents[1]


def lean_bytes(fitted: bool) -> bytes:
    header: dict[str, object] = {"players": [], "rows": 0, "cars": 0, "status_names": []}
    if fitted:
        header["timing_fit"] = {"version": replays.TIMING_FIT_VERSION}
    text = json.dumps(header).encode()
    return b"NRPL" + struct.pack("<II", 1, len(text)) + text


def test_fitted_conversions_have_a_store_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replays, "OUT", tmp_path)
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    plain, fitted = replays.store_folder(), replays.store_folder(fit_timing=True)
    assert plain.name == f"replicar-{replays.REPLICAR_VERSION}"
    assert fitted.name == f"replicar-{replays.REPLICAR_VERSION}-timing-fit-1"
    assert fitted.parent == plain.parent
    assert replays.timing_fitted(lean_bytes(True)) and not replays.timing_fitted(lean_bytes(False))
    assert replays.keep(lean_bytes(False), "r", 1, plain) and replays.keep(lean_bytes(True), "r", 1, fitted)
    for data, store in ((lean_bytes(True), plain), (lean_bytes(False), fitted)):
        with pytest.raises(ValueError, match="other timing"):
            replays.keep(data, "s", 1, store)
    assert replays.find_stored("s", store=plain) is None and replays.find_stored("s", store=fitted) is None
    kept = replays.stored("r", 1, fitted)
    assert replays.read_stored(kept, fit_timing=True)[1] == []
    with pytest.raises(ValueError, match="other timing"):
        replays.read_stored(kept, fit_timing=False)
    with pytest.raises(ValueError, match="other timing"):
        replays.read_stored(replays.stored("r", 1, plain), fit_timing=True)


def test_replays_converted_before_are_left_alone_unless_reconverted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For the fitted store a recording or an unfitted conversion kept before counts as converted, unless
    the pass reconverts; for the unfitted store a fitted conversion never counts."""
    monkeypatch.setattr(replays, "OUT", tmp_path)
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    (tmp_path / "1v1").mkdir()
    (tmp_path / "1v1" / "r.rlpr.zst").write_bytes(b"")
    assert replays.keep(lean_bytes(False), "u", 1)
    assert replays.keep(lean_bytes(True), "f", 1, replays.store_folder(True))
    sizes = replays.Sizes(tmp_path)
    plain, fitted = replays.store_folder(), replays.store_folder(True)
    assert [replays.converted_before(s, sizes, plain) for s in ("r", "u", "f")] == [True, True, False]
    assert [replays.converted_before(s, sizes, fitted) for s in ("r", "u", "f")] == [True, True, True]
    again = [replays.converted_before(s, sizes, fitted, reconvert=True) for s in ("r", "u", "f")]
    assert again == [False, False, True], "reconverting: only a fitted conversion counts"


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")
def test_the_converter_is_asked_for_the_fit(tmp_path: Path) -> None:
    tool = tmp_path / "tool"
    tool.write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/args"\ntouch "$4"\n', encoding="utf-8")
    tool.chmod(0o755)
    for fit, flag in ((False, ""), (True, " --fit-timing")):
        assert replays.convert(tool, tmp_path / "a.replay", tmp_path / "a.lean", tmp_path, fit) is None
        args = (tmp_path / "args").read_text(encoding="utf-8").strip()
        assert args == f"convert {tmp_path / 'a.replay'} -o {tmp_path / 'a.lean'} --meshes {tmp_path}{flag}"


def test_the_setting_is_on_and_applies_to_the_large_mode_with_the_nitro_converter() -> None:
    assert PretrainConfig().fit_timing is True and default_config(ROOT).pretrain.fit_timing is True
    base = NitroConfig()

    def chosen(**changes: object) -> PretrainConfig:
        return dataclasses.replace(base.pretrain, enabled=True, **changes)

    assert chosen(batch_replays=2).fits_timing
    assert not chosen(batch_replays=2, converter="replicar").fits_timing
    assert not chosen().fits_timing and not chosen(batch_replays=2, fit_timing=False).fits_timing
    for pretraining in (chosen(), chosen(converter="replicar"), chosen(batch_replays=2)):
        config = dataclasses.replace(base, pretrain=pretraining)
        assert not [path for path, _ in config.problems() if path == "pretrain.fit_timing"], "never refused"


def test_a_copy_records_the_fit_only_when_on(tmp_path: Path) -> None:
    config = large(tmp_path, tmp_path, fit_timing=False)
    plain = pretrain._made_with(config)
    assert "fit_timing" not in plain, "copies made before the setting carry on as they are"
    fitted = pretrain._made_with(large(tmp_path, tmp_path))
    assert fitted == {**plain, "fit_timing": True}
    assert not pretrain._same_copy(plain, fitted), "turning it on starts the copy over"
    assert pretrain._made_with(large(tmp_path, tmp_path, converter="replicar")) == {
        **pretrain._made_with(large(tmp_path, tmp_path, converter="replicar", fit_timing=False))
    }, "the setting does nothing for replicar's own tool"


def _bot(runs: Path, name: str, text: str, info: dict[str, object] | None = None) -> RunFolder:
    """A bot made by this version (`info` None) or, with `info`, one whose run.json is `info`."""
    settings = runs.parent / f"{name}.toml"
    settings.write_text(text, encoding="utf-8")
    folder = RunFolder.create(runs, name, settings)
    if info is not None:
        folder.info.write_text(json.dumps({"settings_dir": str(runs.parent), **info}), encoding="utf-8")
    return folder


def test_new_bots_fit_and_bots_made_before_keep_their_replays(tmp_path: Path) -> None:
    """A new bot takes the fit; a bot made before it (its run.json lists no new defaults) reads the replays
    as before unless its settings file turns the fit on; the settings file always decides."""
    runs = tmp_path / "runs"
    preset = (ROOT / "configs" / "presets" / "1v1_starter.toml").read_text(encoding="utf-8")
    new = _bot(runs, "new", preset)
    assert json.loads(new.info.read_text(encoding="utf-8"))["new_defaults"] == list(NEW_DEFAULTS)
    assert new.load_settings().pretrain.fit_timing is True
    old = _bot(runs, "old", preset, info={})
    assert old.load_settings().pretrain.fit_timing is False
    on = _bot(runs, "old-on", preset + "\n[pretrain]\nfit_timing = true\n", info={})
    assert on.load_settings().pretrain.fit_timing is True
    off = _bot(runs, "new-off", preset + "\n[pretrain]\nfit_timing = false\n")
    assert off.load_settings().pretrain.fit_timing is False
    # The replays in training follow it: the old bot keeps the store it read.
    assert replay_mix.folders(old.load_settings())[0].parent == replays.store_folder(False)
    assert replay_mix.folders(new.load_settings())[0].parent == replays.store_folder(True)


def test_an_old_bots_copy_in_progress_carries_on(tmp_path: Path) -> None:
    """A large copy begun before the setting existed (made_with without it) is the same copy after the update:
    its bot's settings come out with the fit off."""
    runs = tmp_path / "runs"
    text = "[pretrain]\nenabled = true\nbatch_replays = 2\n"
    old = _bot(runs, "old", text, info={})
    begun = pretrain._made_with(old.load_settings())
    assert "fit_timing" not in begun
    assert pretrain._same_copy(begun, pretrain._made_with(old.load_settings()))
    new = _bot(runs, "new", text)
    assert not pretrain._same_copy(begun, pretrain._made_with(new.load_settings()))


def test_the_dashboard_makes_new_bots_with_the_new_defaults(tmp_path: Path) -> None:
    from nitro.dashboard.api import Api

    api = Api(ROOT, tmp_path / "runs")
    assert "made" in api.create_bot("fresh", "1v1_starter", {})
    fresh = RunFolder(tmp_path / "runs" / "fresh")
    assert fresh.load_settings().pretrain.fit_timing is True
    assert "made" in api.copy_bot("fresh", "twin", False)
    assert RunFolder(tmp_path / "runs" / "twin").load_settings().pretrain.fit_timing is True


def test_the_dashboard_starts_a_reconversion_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nitro.dashboard.api import Api

    monkeypatch.setattr(replays, "OUT", tmp_path / "project-replays")
    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    assert replays.keep(lean_bytes(False), "old", 2)
    api = Api(ROOT, tmp_path / "runs")
    view = api.converting()
    assert view["unfitted"] == {"2v2": 1} and view["store"]["counts"] == {}
    started: list[list[str]] = []
    monkeypatch.setattr(api._jobs, "start", lambda key, kind, args, folder: started.append(args))
    folder = tmp_path / "mine"
    folder.mkdir()
    assert api.convert_start([str(folder)])["started"]
    assert api.convert_start([str(folder)], 5, True)["started"]
    assert "--reconvert" not in started[0] and "--no-fit-timing" not in started[0]
    assert started[1][-3:] == ["--limit", "5", "--reconvert"]


def test_the_command_fits_by_default_and_reconverts_only_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli.build_parser()
    plain = parser.parse_args(["convert-replays"])
    assert (plain.fit_timing, plain.reconvert) == (True, False)
    assert parser.parse_args(["convert-replays", "--no-fit-timing"]).fit_timing is False
    assert parser.parse_args(["convert-replays", "--reconvert"]).reconvert is True
    assert cli.main(["convert-replays", str(tmp_path), "--reconvert", "--no-fit-timing"]) == 1
    assert "cannot go with --no-fit-timing" in capsys.readouterr().err


@SHELL
def test_converting_with_the_fit_fills_its_own_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_replays import converted_calls

    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "two", "b"])
    meshes, tools = tmp_path / "meshes", tmp_path / "tools"
    done = replays.convert_to_store([demos], meshes, workers=2, limit=2)
    assert done.kept == 2 and sorted(converted_calls(tools)) == ["a", "two"]
    done = replays.convert_to_store([demos], meshes, workers=2, fit_timing=True)
    assert done.kept == 1 and converted_calls(tools) == ["b"], "the replays converted before are left"
    done = replays.convert_to_store([demos], meshes, workers=2, fit_timing=True, reconvert=True)
    assert done.kept == 2 and sorted(converted_calls(tools)) == ["a", "two"], "reconverted when asked"
    plain = replays.store_summary(replays.store_folder())["counts"]
    fitted = replays.store_summary(replays.store_folder(fit_timing=True))["counts"]
    assert plain == {"1v1": 1, "2v2": 1} and fitted == {"1v1": 2, "2v2": 1}
    assert not replays.timing_fitted(_raw(replays.stored("a", 1)))
    for stem in ("a", "b"):
        assert replays.timing_fitted(_raw(replays.stored(stem, 1, replays.store_folder(fit_timing=True))))
    # A converter that does not fit (an old build) fills nothing of the fitted store.
    monkeypatch.setattr(replays, "nitro_replicar_tool", lambda: stand_in_lean_tool(tools, fits=False))
    store = tmp_path / "other" / replays.store_folder(fit_timing=True).name
    done = replays.convert_to_store([demos], meshes, workers=2, store=store, reconvert=True)
    assert done.kept == 0 and sum(done.failed.values()) == 3


def _raw(path: Path) -> bytes:
    import pyarrow as pa

    data = path.read_bytes()
    return bytes(pa.Codec("zstd").decompress(data[8:], struct.unpack_from("<Q", data)[0], asbytes=True))


@SHELL
@pytest.mark.parametrize("fit_timing", [True, False])
def test_a_fitted_copy_never_takes_a_recording_kept_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fit_timing: bool
) -> None:
    """A recording kept by an earlier version is never timing-fitted: a fitted copy converts that replay
    again (with the fit), an unfitted one takes the recording as before."""
    from test_pretrain_large import _recording_of
    from test_replays import converted_calls

    monkeypatch.setattr(replays, "STORE_MIN_FREE", 0)
    demos = tools_and_replays(tmp_path, monkeypatch, ["a", "b", "c", "d"])
    kept = replays.OUT / "1v1" / "a.rlpr.zst"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(replays.compress(_recording_of(tmp_path).read_bytes()))
    (tmp_path / "s.toml").write_text("", encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "copied", tmp_path / "s.toml")
    config = large(tmp_path, demos, fit_timing=fit_timing)
    assert pretrain.start_source(config, folder, lambda _: None) is not None
    calls = converted_calls(tmp_path / "tools")
    assert ("a" in calls) is fit_timing, calls
