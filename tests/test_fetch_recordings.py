"""RocketSim's five real-game recordings: downloaded once from the pinned commit, each checked against its
pinned hash, when asked (`nitro fetch-recordings`) or the first time a run or `nitro pretrain --rocketsim`
needs them; never while settings are checked. Stand-in downloads: no network."""

from __future__ import annotations

import hashlib
import urllib.error
from collections.abc import Callable
from pathlib import Path

import pytest

from nitro import cli
from nitro import recordings as rec
from nitro.cli import main
from nitro.config.state_setters import Replay

PROJECT = Path(__file__).resolve().parents[1]
Download = Callable[[str, Path], str]


@pytest.fixture
def stand_ins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Five small stand-in recordings with the pins set to them, the default folder in `tmp_path`, no cargo
    checkout and no waiting between tries."""
    files = {f"rec_{i}.rlpr.zst": f"recording {i}".encode() * (i + 1) for i in range(5)}
    pins = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items()}
    monkeypatch.setattr(rec, "RECORDINGS", pins)
    monkeypatch.setattr(rec, "FOLDER", tmp_path / "rocketsim_recordings")
    monkeypatch.setattr(rec, "rocketsim_checkout", lambda revision=rec.PINNED_REVISION: None)
    monkeypatch.setattr(rec.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(rec, "_from_release", lambda folder, missing: [])  # RocketSim's own copy
    return files


def _serving(files: dict[str, bytes], fetched: list[str]) -> Download:
    def download(url: str, target: Path) -> str:
        assert url.startswith(rec.URL)
        name = url.removeprefix(rec.URL)
        fetched.append(name)
        target.write_bytes(files[name])
        return hashlib.sha256(files[name]).hexdigest()

    return download


def _offline(url: str, target: Path) -> str:
    target.write_bytes(b"half a file")
    raise urllib.error.URLError("no internet")


def test_the_recordings_come_from_the_pinned_commit_by_hash() -> None:
    assert rec.URL == (
        "https://raw.githubusercontent.com/ZealanL/RocketSim/"
        "45722876705adf5dc673aa764e80382cfe8c3312/rocketsim_test/recordings/"
    )
    assert len(rec.RECORDINGS) == 5 and rec.total_mb() == 119
    for name, (size, sha256) in rec.RECORDINGS.items():
        assert name.endswith(".rlpr.zst") and size > 10_000_000 and len(sha256) == 64
    assert rec.FOLDER == PROJECT / "rocketsim_recordings"
    assert "/rocketsim_recordings/" in (PROJECT / ".gitignore").read_text(encoding="utf-8")


def test_fetching_downloads_only_whats_missing_and_checks_each_file(stand_ins: dict[str, bytes]) -> None:
    fetched: list[str] = []
    said: list[str] = []
    paths = rec.fetch_recordings(say=said.append, download=_serving(stand_ins, fetched))
    assert paths == [rec.FOLDER / name for name in stand_ins] and fetched == list(stand_ins)
    assert all(path.read_bytes() == stand_ins[path.name] for path in paths)
    assert "5 files" in said[0] and "MB" in said[0] and "github.com/ZealanL/RocketSim" in said[0]
    assert rec.bundled_recordings() == paths
    # Already there: nothing downloaded.
    rec.fetch_recordings(say=said.append, download=lambda url, target: pytest.fail("they are there"))
    assert "already" in said[-1]
    # One cut short: only that one again.
    paths[2].write_bytes(b"cut")
    assert rec.bundled_recordings() == []
    fetched.clear()
    rec.fetch_recordings(say=said.append, download=_serving(stand_ins, fetched))
    assert fetched == [paths[2].name] and "1 files" in said[-2]
    assert not list(rec.FOLDER.glob("*.part"))


def test_a_file_that_does_not_match_its_hash_is_not_kept(stand_ins: dict[str, bytes]) -> None:
    wrong = dict(stand_ins) | {"rec_1.rlpr.zst": b"something else entirely"}
    with pytest.raises(rec.RecordingFetchError, match=r"hash differs.*nitro fetch-recordings"):
        rec.fetch_recordings(say=lambda text: None, download=_serving(wrong, []))
    assert not (rec.FOLDER / "rec_1.rlpr.zst").exists() and not list(rec.FOLDER.glob("*.part"))
    assert rec.bundled_recordings() == []


def test_offline_the_fetch_fails_with_a_plain_message(stand_ins: dict[str, bytes]) -> None:
    tries: list[str] = []

    def offline(url: str, target: Path) -> str:
        tries.append(url)
        return _offline(url, target)

    with pytest.raises(rec.RecordingFetchError) as caught:
        rec.fetch_recordings(say=lambda text: None, download=offline)
    assert len(tries) == rec.ATTEMPTS and rec.TIMEOUT_SECONDS == 120
    text = str(caught.value)
    assert "could not download" in text and "internet connection" in text and "nitro fetch-recordings" in text
    assert not list(rec.FOLDER.glob("*.part"))


def test_downloaded_copies_come_before_a_cargo_checkout(
    tmp_path: Path, stand_ins: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    found = tmp_path / "checkout" / "rocketsim_test" / "recordings"
    found.mkdir(parents=True)
    for name, data in stand_ins.items():
        (found / name).write_bytes(data)
    monkeypatch.setattr(rec, "rocketsim_checkout", lambda revision=rec.PINNED_REVISION: tmp_path / "checkout")
    assert rec.bundled_recordings() == [found / name for name in stand_ins]
    rec.fetch_recordings(say=lambda text: None, download=_serving(stand_ins, []))
    assert rec.bundled_recordings() == [rec.FOLDER / name for name in stand_ins]


def test_a_run_downloads_them_the_first_time_but_checking_settings_never_does(
    stand_ins: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rec, "_download", lambda url, target: pytest.fail("checking downloads nothing"))
    setter = Replay(bundled=True)
    assert rec.missing_replay_files(setter) is None
    fetched: list[str] = []
    monkeypatch.setattr(rec, "_download", _serving(stand_ins, fetched))
    assert rec.replay_files(setter) == [str(rec.FOLDER / name) for name in stand_ins]
    assert fetched == list(stand_ins)
    assert "Downloading RocketSim's real-game recordings" in capsys.readouterr().out
    monkeypatch.setattr(rec, "_download", _offline)
    for path in rec.FOLDER.iterdir():
        path.unlink()
    with pytest.raises(rec.RecordingFetchError, match="nitro fetch-recordings"):
        rec.replay_files(setter)


def test_the_command_fetches_them_and_says_when_it_cannot(
    tmp_path: Path,
    stand_ins: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(rec, "_download", _serving(stand_ins, []))
    assert main(["fetch-recordings", "--dest", str(tmp_path / "here")]) == 0
    assert sorted(p.name for p in (tmp_path / "here").iterdir()) == sorted(stand_ins)
    monkeypatch.setattr(rec, "_download", _offline)
    assert main(["fetch-recordings", "--dest", str(tmp_path / "elsewhere")]) == 1
    err = capsys.readouterr().err
    assert (
        err.startswith("Fetching RocketSim's recordings failed: could not download")
        and "Traceback" not in err
    )


def test_a_run_that_cannot_download_them_stops_with_the_message(
    stand_ins: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def starts_a_run(args: object) -> int:
        rec.replay_files(Replay(bundled=True))
        return 0

    monkeypatch.setattr(rec, "_download", _offline)
    monkeypatch.setattr(cli, "_check_settings", starts_a_run)
    assert main(["check-settings", "bot.toml"]) == 1
    err = capsys.readouterr().err
    assert "RocketSim's recordings (bundled = true) are not on this computer: could not download" in err
    assert "nitro fetch-recordings" in err


def test_pretrain_rocketsim_offline_says_what_to_do(
    stand_ins: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rec, "_download", _offline)
    settings = str(PROJECT / "configs" / "presets" / "1v1_starter.toml")
    assert main(["pretrain", settings, "--rocketsim"]) == 1
    err = capsys.readouterr().err
    assert err.startswith(
        "Cannot pretrain: RocketSim's recordings are not on this computer: could not download"
    )
    assert "nitro fetch-recordings" in err
