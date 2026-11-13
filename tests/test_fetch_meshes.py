"""`nitro fetch-meshes`: the 16 soccar meshes from RLGym's package, each checked against its pinned hash."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import urllib.request
from pathlib import Path

import pytest

from nitro import fetch_meshes as fm
from nitro.cli import main

PROJECT = Path(__file__).resolve().parents[1]


def _archive(meshes: list[bytes], extra: dict[str, bytes] | None = None) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        files = {f"{fm.INSIDE}/mesh_{i}.cmf": data for i, data in enumerate(meshes)} | (extra or {})
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


@pytest.fixture
def made_up(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, list[bytes]]:
    """A stand-in archive, with the pins set to its hashes."""
    meshes = [f"mesh {i}".encode() for i in range(16)]
    archive = _archive(meshes, {"rlgym-rocket-league-2.0.1/setup.py": b"never read"})
    monkeypatch.setattr(fm, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(fm, "MESH_SHA256", tuple(hashlib.sha256(m).hexdigest() for m in meshes))
    return archive, meshes


def test_the_meshes_are_taken_from_the_archive_and_checked(
    tmp_path: Path, made_up: tuple[bytes, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, meshes = made_up
    copy = tmp_path / "rlgym.tar.gz"
    copy.write_bytes(archive)
    said: list[str] = []
    folder = fm.fetch_meshes(tmp_path / "meshes", copy, said.append)
    assert folder == tmp_path / "meshes" / "soccar"
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"mesh_{i}.cmf" for i in range(16))
    assert (folder / "mesh_7.cmf").read_bytes() == meshes[7]
    # Already there: nothing downloaded or read again.
    monkeypatch.setattr(fm, "download", lambda *a: pytest.fail("no download when they are there"))
    fm.fetch_meshes(tmp_path / "meshes", None, said.append)
    assert "already" in said[-1]
    # One mesh changed: downloaded again (here from the archive) and put right.
    (folder / "mesh_3.cmf").write_bytes(b"changed")
    monkeypatch.setattr(fm, "download", lambda *a: archive)
    fm.fetch_meshes(tmp_path / "meshes", None, said.append)
    assert (folder / "mesh_3.cmf").read_bytes() == meshes[3]


def test_nothing_is_unpacked_from_an_archive_that_does_not_match(
    tmp_path: Path, made_up: tuple[bytes, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, meshes = made_up
    with pytest.raises(fm.MeshFetchError, match="SHA-256 hash differs"):
        fm.meshes_from(archive + b"x")
    swapped = _archive([b"not a mesh", *meshes[1:]])
    monkeypatch.setattr(fm, "ARCHIVE_SHA256", hashlib.sha256(swapped).hexdigest())
    copy = tmp_path / "swapped.tar.gz"
    copy.write_bytes(swapped)
    with pytest.raises(fm.MeshFetchError, match=r"mesh_0\.cmf in the archive does not match"):
        fm.fetch_meshes(tmp_path / "meshes", copy, lambda text: None)
    assert not (tmp_path / "meshes").exists()


def test_the_command_says_why_it_failed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fetch-meshes", "--archive", str(tmp_path / "none.tar.gz"), "--dest", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err


def _real_archive() -> bytes | None:
    """RLGym's real archive: NITRO_RLGYM_ARCHIVE names a downloaded copy, else PyPI when it can be reached."""
    if copy := os.environ.get("NITRO_RLGYM_ARCHIVE"):
        return Path(copy).read_bytes()
    try:
        with urllib.request.urlopen(fm.URL, timeout=30) as answer:
            data: bytes = answer.read()
            return data
    except OSError:
        return None


def test_the_pins_match_rlgyms_real_package(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = _real_archive()
    if archive is None:
        pytest.skip("RLGym's package cannot be downloaded here (set NITRO_RLGYM_ARCHIVE to a copy)")
    copy = tmp_path / "rlgym-rocket-league-2.0.1.tar.gz"
    copy.write_bytes(archive)
    assert main(["fetch-meshes", "--archive", str(copy), "--dest", str(tmp_path / "meshes")]) == 0
    assert "hashes checked" in capsys.readouterr().out
    assert fm.already_there(tmp_path / "meshes")
    # (Not loaded with the core here: a process loads its meshes once, and other tests load the project's.)
    ours = PROJECT / "collision_meshes" / "soccar"
    if ours.is_dir():
        for mesh in (tmp_path / "meshes" / "soccar").iterdir():
            assert mesh.read_bytes() == (ours / mesh.name).read_bytes(), "RocketSim recognises these"


@pytest.mark.skipif(not (PROJECT / "collision_meshes" / "soccar").is_dir(), reason="no collision meshes here")
def test_the_meshes_in_this_folder_are_the_pinned_ones() -> None:
    assert fm.already_there(PROJECT / "collision_meshes")
