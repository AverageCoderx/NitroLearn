"""`nitro fetch-meshes`: the arena's collision meshes (the 16 soccar .cmf files, about 200 KB), without
Rocket League.

They come from the official RLGym package on PyPI (rlgym-rocket-league 2.0.1), whose source archive holds
rlgym/rocket_league/sim/collision_meshes/soccar/mesh_0..15.cmf. The archive and every mesh are checked
against pinned SHA-256 hashes; only those 16 files are read from it, nothing is installed or run. They are
Rocket League game data, not covered by the package's Apache-2.0 licence: they are downloaded to this
computer, never shipped with NitroLearn (git ignores them). Meshes dumped from your own game work as well.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

URL = (
    "https://files.pythonhosted.org/packages/99/21/b87ef7ebb0d76e654753349a4f343068d08e1a9eaffd9c77c0593b43668f/"
    "rlgym-rocket-league-2.0.1.tar.gz"
)
ARCHIVE_SHA256 = "3be8e9d2f1cfb6514e0e3d455fb33ff95e26eb7c399f819fff0258025cf4d501"
INSIDE = "rlgym-rocket-league-2.0.1/rlgym/rocket_league/sim/collision_meshes/soccar"
MESH_SHA256 = (
    "8764d43b87ba134cacf1be454d59d493630188e79c0d88639258b359bb511c4e",
    "9d42e5db7cde5c7be0783c256e15e819bd09351fcb2debd3205e3d055073c993",
    "762ec145db3c317d82314ad94af6f8b591e822c26860ed3c0ee106aba5591eee",
    "0db5c559287bcd255b99e6f5db341b874ec9a5fcf6137247cf16d913e178db5c",
    "92c3224da44f1a359140647845a44457df35541458b92485e7a3232bb596566a",
    "776c24b1231b4f7ab5437af68e4cc1cc4d5cd7875b8cb386a67487162c3941a3",
    "a2262bfa2bfacca3caf3a8334f02b9b52624056853cc33a1340a23c3a33804e6",
    "c887df022d6a2b33e78a3cf8f5dd65d9cd61f4d72009781d40c32091119a06f3",
    "3350567082d9a133f59f8471d941fde6cce20eb2b2bc26c2ab0566b7fb7a21e9",
    "4864911a7f0a3add929f41cc8f676cde416dcc3c07ab08a30742c9893e8a6ce8",
    "36cc44e14a0e500aeb6167a4d5fce449b218b3d941df3ca46f398d269b01af3d",
    "7f4468be0b0835a48996f92d1a2398f1290acbf1152d39e1c53c2701d1fb6ce3",
    "952a90a0ff27c2733d507e59bdca8600968276eddad9a41b47a689010e805118",
    "b441f782f07e81c2444ed7a5c4ed618c26ef3e1940ac2c63c362154620631750",
    "085f7533c53a73da664c129c3b1c93f5ce08ceb302aa5b993ff5fe95af5d6ae8",
    "21c1d1eb450f9e09ee8d7d55e1d41bf563518f9b2deb2ab5369c47139faa2e4a",
)
DEFAULT_DEST = Path(__file__).resolve().parents[2] / "collision_meshes"


class MeshFetchError(RuntimeError):
    """The meshes could not be fetched or did not match their pinned hashes."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def already_there(dest: Path) -> bool:
    """Whether `dest`/soccar already holds the 16 meshes, each matching its hash."""
    folder = dest / "soccar"
    try:
        return all(
            _sha256((folder / f"mesh_{i}.cmf").read_bytes()) == wanted for i, wanted in enumerate(MESH_SHA256)
        )
    except OSError:
        return False


def download(url: str = URL, attempts: int = 4) -> bytes:
    """The archive's bytes, retried a few times (a busy network)."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as answer:
                data: bytes = answer.read()
                return data
        except (urllib.error.URLError, OSError) as err:
            if attempt == attempts - 1:
                raise MeshFetchError(
                    f"could not download {url} ({err}). Check the internet connection and try again "
                    "(antivirus web shields and company proxies that scan HTTPS can also block it), or "
                    "download that file in a browser and run: nitro fetch-meshes --archive <the file>"
                ) from None
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def meshes_from(archive: bytes) -> list[bytes]:
    """The 16 soccar meshes inside the package's source archive, in order, each checked by hash."""
    if _sha256(archive) != ARCHIVE_SHA256:
        raise MeshFetchError(
            "the downloaded package is not the expected rlgym-rocket-league 2.0.1 archive (its SHA-256 "
            "hash differs), so nothing was unpacked. Try again later."
        )
    meshes = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for i, wanted in enumerate(MESH_SHA256):
            name = f"{INSIDE}/mesh_{i}.cmf"
            try:
                member = tar.extractfile(name)
            except KeyError:
                member = None
            if member is None:
                raise MeshFetchError(f"the archive has no {name}")
            data = member.read()
            if _sha256(data) != wanted:
                raise MeshFetchError(f"mesh_{i}.cmf in the archive does not match its pinned hash")
            meshes.append(data)
    return meshes


def fetch_meshes(
    dest: Path = DEFAULT_DEST, archive: Path | None = None, say: Callable[[str], None] = print
) -> Path:
    """Puts the 16 soccar meshes in `dest`/soccar (from `archive`, a downloaded copy of the package, else
    from PyPI) and returns that folder. Nothing is downloaded when they are already there."""
    folder = dest / "soccar"
    if already_there(dest):
        say(f"The 16 soccar meshes are already in {folder} (hashes checked).")
        return folder
    if archive is not None:
        try:
            data = archive.read_bytes()
        except OSError as err:
            raise MeshFetchError(f"cannot read {archive} ({err.strerror})") from None
    else:
        say(f"Downloading the arena meshes from RLGym's package on PyPI ({URL.rsplit('/', 1)[1]}, 80 KB)")
        data = download()
    meshes = meshes_from(data)
    folder.mkdir(parents=True, exist_ok=True)
    for i, mesh in enumerate(meshes):
        target = folder / f"mesh_{i}.cmf"
        part = target.with_suffix(".part")
        part.write_bytes(mesh)
        os.replace(part, target)
    say(f"Put the 16 soccar meshes in {folder} (hashes checked).")
    return folder
