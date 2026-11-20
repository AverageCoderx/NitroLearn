"""`nitro get-replays`: downloading only the chunks asked for, resumably, checked, skipping what is already
here. A fake server in memory stands in for GitHub (nothing listens on a port)."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import random
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import IO, Any

import pytest

from browser import open_chromium, playwright_or_skip
from nitro import replay_dataset as ds
from nitro import replays


def pack(root: Path, work: Path, chunk_bytes: int = 2**20) -> None:
    """Chunks and a manifest as the published copy has them: best score first, chunks of up to
    `chunk_bytes`, stored zips."""
    work.mkdir(parents=True)
    chunks = []
    for size in ("1v1", "2v2"):
        folder = root / size
        scores = {
            r["id"]: r["score"]
            for r in map(json.loads, (folder / "metadata.json").read_text(encoding="utf-8").splitlines())
        }
        order = sorted(folder.glob("*.replay"), key=lambda p: (-scores[p.stem], p.stem))
        groups: list[list[Path]] = [[]]
        for path in order:
            if groups[-1] and sum(p.stat().st_size for p in groups[-1]) + path.stat().st_size > chunk_bytes:
                groups.append([])
            groups[-1].append(path)
        for number, group in enumerate(groups, 1):
            name = f"{size}-{number:04d}.zip"
            with zipfile.ZipFile(work / name, "w", zipfile.ZIP_STORED) as archive:
                for path in group:
                    archive.writestr(zipfile.ZipInfo(path.name, (1980, 1, 1, 0, 0, 0)), path.read_bytes())
            data = (work / name).read_bytes()
            chunks.append(
                {
                    "name": name,
                    "size": size,
                    "bytes": len(data),
                    "unpacked": sum(p.stat().st_size for p in group),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "replays": [[p.stem, scores[p.stem]] for p in group],
                }
            )
    manifest = {"format": ds.FORMAT, "credit": ds.CREDIT, "chunks": chunks}
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


URL = "https://example.invalid/releases/download/dataset-v1/manifest.json"


class Answer(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class Server:
    """Serves files by name with HTTP Range; `cut` ends the first answer of a file after that many bytes."""

    def __init__(self, files: dict[str, bytes], ranges: bool = True, cut: int = 0) -> None:
        self.files, self.ranges, self.cut = files, ranges, cut
        self.asked: list[tuple[str, str | None]] = []

    def __call__(self, request: urllib.request.Request, seconds: float) -> IO[bytes]:
        name = request.full_url.rsplit("/", 1)[1]
        wanted = request.get_header("Range")
        self.asked.append((name, wanted))
        if name not in self.files:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", email.message.Message(), None)
        data = self.files[name]
        if wanted and self.ranges:
            start = int(wanted.removeprefix("bytes=").rstrip("-"))
            body, status = data[start:], 206
        else:
            body, status = data, 200
        if self.cut and name.endswith(".zip") and sum(n == name for n, _ in self.asked) == 1:
            body = body[: self.cut]
        return Answer(body, status)


def dataset(root: Path, size: str = "1v1", count: int = 12, length: int = 300_000) -> dict[str, float]:
    """Replays of random bytes and a line-delimited metadata.json: id -> score."""
    folder = root / size
    folder.mkdir(parents=True)
    rng = random.Random(size)
    scores = {}
    for i in range(count):
        replay_id = f"{size}-{i:04d}-aaaa"
        (folder / f"{replay_id}.replay").write_bytes(rng.randbytes(length))
        scores[replay_id] = round(rng.random(), 4)
    lines = [json.dumps({"id": k, "score": v, "players": []}) for k, v in scores.items()]
    (folder / "metadata.json").write_text("\n".join(lines), encoding="utf-8")
    return scores


@pytest.fixture
def packed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Server, dict[str, float], Path]:
    """Twelve 1v1 replays (300 KB) and four 2v2 ones packed into 1 MB chunks, served."""
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    scores = dataset(tmp_path / "data")
    dataset(tmp_path / "data", "2v2", 4)
    work = tmp_path / "work"
    pack(tmp_path / "data", work)
    files = {p.name: p.read_bytes() for p in work.iterdir() if p.suffix in (".zip", ".json")}
    return Server(files), scores, work


def quiet(_: str) -> None:
    pass


def test_the_list_orders_the_replays_best_first(
    packed: tuple[Server, dict[str, float], Path], tmp_path: Path
) -> None:
    server, scores, _ = packed
    manifest = ds.read_manifest(server.files["manifest.json"])
    ones = manifest.of_size("1v1")
    assert [c.name for c in ones] == ["1v1-0001.zip", "1v1-0002.zip", "1v1-0003.zip", "1v1-0004.zip"]
    assert all(len(c.replays) == 3 for c in ones)  # three 300 KB replays fit in 1 MB
    order = [i for c in ones for i, _ in c.replays]
    assert order == sorted(scores, key=lambda k: -scores[k])
    assert manifest.available()["2v2"]["replays"] == 4 and manifest.credit == ds.CREDIT


def test_only_the_chunks_asked_for_come_down_best_first(
    packed: tuple[Server, dict[str, float], Path],
) -> None:
    server, scores, _ = packed
    got = ds.get_replays("1v1", 4, URL, server, quiet, convert=False)
    assert got.replays_new == 6 and got.chunks == 2  # 4 asked for: two chunks of three
    assert [n for n, _ in server.asked if n.endswith(".zip")] == ["1v1-0001.zip", "1v1-0002.zip"]
    folder = ds.size_folder("1v1")
    best = sorted(scores, key=lambda k: -scores[k])[:6]
    assert sorted(p.stem for p in folder.glob("*.replay")) == sorted(best)
    assert [p.stem for p in replays.newest_replays([folder])] == best, "newest first means best first"
    assert not list((ds.folder() / "downloads").iterdir()), "zips are deleted once unpacked"
    assert json.loads(ds.progress_file().read_text())["state"] == "downloaded"
    server.asked.clear()
    more = ds.get_replays("1v1", 4, URL, server, quiet, convert=False)
    assert [n for n, _ in server.asked if n.endswith(".zip")] == ["1v1-0003.zip", "1v1-0004.zip"]
    assert more.replays_new == 6
    server.asked.clear()
    assert ds.get_replays("1v1", 0, URL, server, quiet, convert=False).chunks == 0
    assert [n for n, _ in server.asked] == ["manifest.json"], "nothing left to fetch"


def test_a_replay_already_converted_is_not_fetched_again(
    packed: tuple[Server, dict[str, float], Path],
) -> None:
    server, _, _ = packed
    first = ds.read_manifest(server.files["manifest.json"]).of_size("1v1")[0]
    for replay_id, _ in first.replays:
        kept = replays.stored(replay_id, 1, replays.store_folder(True))
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_bytes(b"converted")
    ds.get_replays("1v1", 3, URL, server, quiet, convert=False)
    assert [n for n, _ in server.asked if n.endswith(".zip")] == ["1v1-0002.zip"]


def test_a_cut_connection_carries_on_with_a_range_request(
    packed: tuple[Server, dict[str, float], Path],
) -> None:
    server, _, _ = packed
    server.cut = 200_000
    waits: list[float] = []
    ds.get_replays("1v1", 1, URL, server, quiet, convert=False, wait=waits.append)
    zips = [(n, r) for n, r in server.asked if n.endswith(".zip")]
    assert zips == [("1v1-0001.zip", None), ("1v1-0001.zip", "bytes=200000-")] and len(waits) == 1
    assert len(list(ds.size_folder("1v1").glob("*.replay"))) == 3


def test_a_server_that_ignores_ranges_starts_the_file_again(
    packed: tuple[Server, dict[str, float], Path],
) -> None:
    server, _, _ = packed
    server.cut, server.ranges = 200_000, False
    ds.get_replays("1v1", 1, URL, server, quiet, convert=False, wait=lambda _: None)
    assert len(list(ds.size_folder("1v1").glob("*.replay"))) == 3


def test_a_damaged_chunk_is_refused_and_thrown_away(packed: tuple[Server, dict[str, float], Path]) -> None:
    server, _, _ = packed
    data = bytearray(server.files["1v1-0001.zip"])
    data[100] ^= 0xFF
    server.files["1v1-0001.zip"] = bytes(data)
    with pytest.raises(ds.DatasetError, match="arrived damaged"):
        ds.get_replays("1v1", 1, URL, server, quiet, convert=False)
    assert not list((ds.folder() / "downloads").glob("*.part"))
    assert json.loads(ds.progress_file().read_text())["state"] == "failed"


def test_too_little_disk_stops_before_downloading(packed: tuple[Server, dict[str, float], Path]) -> None:
    server, _, _ = packed
    got = ds.get_replays("1v1", 1, URL, server, quiet, convert=True, free_bytes=lambda _: 20 * 2**30)
    assert got.replays_new == 0 and not [n for n, _ in server.asked if n.endswith(".zip")]
    view = json.loads(ds.progress_file().read_text())
    assert view["state"] == "disk_full" and "free some space" in view["note"]
    ds.get_replays("1v1", 1, URL, server, quiet, convert=False, free_bytes=lambda _: 20 * 2**30)
    assert len(list(ds.size_folder("1v1").glob("*.replay"))) == 3, "without converting it fits"


def test_a_missing_chunk_and_an_offline_list_say_so(packed: tuple[Server, dict[str, float], Path]) -> None:
    server, _, _ = packed
    del server.files["1v1-0001.zip"]
    with pytest.raises(ds.DatasetError, match="not on the server"):
        ds.get_replays("1v1", 1, URL, server, quiet, convert=False)
    said: list[str] = []
    offline = Server({})
    with pytest.raises(ds.DatasetError, match="not on the server"):
        ds.get_replays("1v1", 1, URL, offline, said.append, convert=False)
    assert any("copy downloaded before" in s for s in said), "the kept list is used when offline"
    with pytest.raises(ds.DatasetError, match="no 3v3 replays"):
        ds.get_replays("3v3", 1, URL, server, quiet, convert=False)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"format": 2}, "format 2"),
        (
            {
                "chunks": [
                    {
                        "name": "../evil.zip",
                        "size": "1v1",
                        "bytes": 1,
                        "unpacked": 1,
                        "sha256": "0" * 64,
                        "replays": [["a", 1]],
                    }
                ]
            },
            "bad chunk",
        ),
        (
            {
                "chunks": [
                    {
                        "name": "a.zip",
                        "size": "1v1",
                        "bytes": 1,
                        "unpacked": 1,
                        "sha256": "0" * 64,
                        "replays": [["../../x", 1]],
                    }
                ]
            },
            "bad replay id",
        ),
        ({"chunks": "nope"}, "could not be read"),
    ],
)
def test_a_bad_manifest_is_refused(change: dict[str, Any], says: str) -> None:
    raw = {"format": 1, "credit": "c", "chunks": []} | change
    with pytest.raises(ds.DatasetError, match=says):
        ds.read_manifest(json.dumps(raw).encode())


def test_the_command_downloads_from_a_packed_folder_on_this_computer(
    packed: tuple[Server, dict[str, float], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    from nitro.cli import main

    _, _, work = packed
    assert main(["get-replays", "--from", str(work / "manifest.json"), "--list"]) == 0
    assert "1v1: 12 replays in 4 chunks" in capsys.readouterr().out
    assert main(["get-replays", "--from", str(work / "manifest.json"), "--count", "2", "--no-convert"]) == 0
    assert len(list(ds.size_folder("1v1").glob("*.replay"))) == 3
    assert main(["get-replays", "--from", str(work / "manifest.json"), "--meshes", str(work)]) == 1
    assert "arena meshes were not found" in capsys.readouterr().err


def test_the_resources_page_downloads_replays_and_shows_the_progress(
    packed: tuple[Server, dict[str, float], Path], tmp_path: Path
) -> None:
    """The Download replays card: what is here and published, how many, Download (which starts
    `nitro get-replays` with the size, count and converting asked for), and a download's progress."""
    sync_api = playwright_or_skip()
    from nitro.dashboard.api import Api
    from nitro.dashboard.app import page

    server, _, _ = packed
    ds.get_replays("1v1", 1, URL, server, quiet, convert=False)  # 3 here, the list kept
    api = Api(Path(__file__).parents[1], tmp_path / "runs")
    view = api.dataset()
    assert view["here"] == {"1v1": 3} and view["published"]["2v2"]["replays"] == 4
    assert view["progress"]["state"] == "downloaded" and view["state"] == "none"
    started: list[tuple[str, int, bool]] = []

    def start(size: str, count: int, convert: bool = True) -> dict[str, Any]:
        started.append((size, count, convert))
        return {"started": True}

    api.dataset_start = start  # type: ignore[method-assign]  # not the real download here
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page()
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click("[data-go=resources]")
            tab.wait_for_selector("#dataset >> text=2v2: 4 replays")
            tab.wait_for_selector("#dataset >> text=Rolv Arild")
            tab.select_option("#ds-size", "2v2")
            tab.fill("#ds-count", "2,000")
            tab.uncheck("#ds-convert")
            tab.click("[data-act=ds-start]")
            tab.wait_for_selector("#dataset >> text=Downloading 1v1: part 1 of 1")  # the last download's
            assert started == [("2v2", 2000, False)]
        finally:
            browser.close()
    assert errors == []


def test_starting_a_download_checks_what_it_is_given(
    packed: tuple[Server, dict[str, float], Path], tmp_path: Path
) -> None:
    from nitro.dashboard.api import Api

    api = Api(Path(__file__).parents[1], tmp_path / "runs")
    assert "game size" in api.dataset_start("5v5", 10)["error"]
    assert "whole number" in api.dataset_start("1v1", -1)["error"]


def test_a_replay_list_not_online_yet_says_so_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")
    nothing = Server({})
    with pytest.raises(ds.NotPublished, match="Replay downloads aren't available yet"):
        ds.fetch_manifest(ds.MANIFEST_URL, nothing, quiet)
    with pytest.raises(ds.DatasetError) as other:
        ds.fetch_manifest(URL, nothing, quiet)  # a list someone points it at: a plain error
    assert not isinstance(other.value, ds.NotPublished) and "404" in str(other.value)
    with pytest.raises(ds.NotPublished):
        ds.get_replays("1v1", 10, ds.MANIFEST_URL, nothing, quiet, convert=False)


def test_the_dashboard_says_downloads_are_not_available_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_api = playwright_or_skip()
    from nitro.dashboard.api import Api
    from nitro.dashboard.app import page

    monkeypatch.setattr(replays, "OUT", tmp_path / "replays")

    def not_yet(*_: Any, **__: Any) -> ds.Manifest:
        raise ds.NotPublished(ds.NOT_YET)

    monkeypatch.setattr(ds, "fetch_manifest", not_yet)
    api = Api(Path(__file__).parents[1], tmp_path / "runs")
    assert api.dataset_published() == {"unavailable": ds.NOT_YET}
    assert api.dataset_start("1v1", 10) == {"unavailable": ds.NOT_YET}
    assert api._jobs.get(".dataset", "dataset") is None, "nothing started"
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = open_chromium(p)
        try:
            tab = browser.new_page()
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.expose_function("__py", lambda method, args: getattr(api, method)(*args))
            tab.add_init_script(
                "window.__testApi = new Proxy({}, {get: (t, m) => (...a) => window.__py(m, a)});"
            )
            tab.goto("about:blank")
            tab.set_content(page(api))
            tab.click("[data-go=resources]")
            tab.click("[data-act=ds-start]")
            tab.wait_for_selector("#dataset >> text=Replay downloads aren't available yet")
            assert tab.locator("[data-act=ds-start]").count() == 0, "no Download button that cannot work"
        finally:
            browser.close()
    assert errors == []


def test_a_connection_that_never_brings_a_byte_gives_up_and_keeps_what_came(
    packed: tuple[Server, dict[str, float], Path],
) -> None:
    server, _, _ = packed
    real = server.files["1v1-0001.zip"]

    def broken(request: urllib.request.Request, seconds: float) -> IO[bytes]:
        if request.full_url.endswith(".zip"):
            if request.get_header("Range") is None:
                return Answer(real[:1000], 200)  # one cut answer, then nothing at all
            raise urllib.error.URLError("connection refused")
        return server(request, seconds)

    waits: list[float] = []
    with pytest.raises(ds.DatasetError, match=r"could not download 1v1-0001\.zip"):
        ds.get_replays("1v1", 1, URL, broken, quiet, convert=False, wait=waits.append)
    assert len(waits) == ds.ATTEMPTS
    assert (ds.folder() / "downloads" / "1v1-0001.zip.part").stat().st_size == 1000, "kept for next time"
