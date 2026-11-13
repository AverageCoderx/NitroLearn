"""`nitro install-rlbot`: RLBot v5's own installer, found in RLBot's release, checked, then opened. No network
here: the release list, the download and Windows Installer are stand-ins. What runs only on Windows (msiexec
and RLBot's installer windows themselves) is not run."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from nitro import rlbot_install as rl
from nitro.cli import main
from nitro.dashboard.api import Api

ROOT = Path(__file__).resolve().parents[1]
MSI = rl.MSI_MAGIC + bytes(100_000) + b"Installer for the RLBot v5 launcher" + bytes(200_000)
OFFICIAL_MSI = rl.OFFICIAL + "installer/rlbot-v5-installer.msi"

# The shape of GitHub's answer for RLBot's launcher release tagged "installer" (trimmed).
RELEASE = {
    "tag_name": "installer",
    "assets": [
        {
            "name": "launcher.exe",
            "browser_download_url": rl.OFFICIAL + "installer/launcher.exe",
            "size": 9,
            "digest": "sha256:" + "0" * 64,
        },
        {
            "name": "rlbot-v5-installer.msi",
            "browser_download_url": OFFICIAL_MSI,
            "size": len(MSI),
            "digest": "sha256:" + hashlib.sha256(MSI).hexdigest(),
        },
    ],
}


def test_the_installer_is_taken_from_rlbots_own_release_only() -> None:
    found = rl.pick_installer(RELEASE)
    assert found == rl.Installer(OFFICIAL_MSI, hashlib.sha256(MSI).hexdigest(), len(MSI))
    elsewhere = {
        "assets": [{"name": "rlbot-v5-installer.msi", "browser_download_url": "https://example.com/x.msi"}]
    }
    assert rl.pick_installer(elsewhere) is None, "an .msi from anywhere but RLBot's releases is never used"
    assert rl.pick_installer({"assets": [{"name": "rlbot.msi", "browser_download_url": OFFICIAL_MSI}]}) == (
        rl.Installer(OFFICIAL_MSI)
    ), "no hash or size listed: checked as an .msi only"
    assert rl.find_installer(lambda: RELEASE).url == OFFICIAL_MSI

    def unreachable() -> dict[str, Any]:
        raise urllib.error.URLError("rate limited")

    assert rl.find_installer(unreachable) == rl.Installer(rl.FALLBACK_URL), "the fixed official link"
    assert rl.find_installer(lambda: {"assets": []}) == rl.Installer(rl.FALLBACK_URL)
    assert rl.FALLBACK_URL.startswith(rl.OFFICIAL) and rl.RELEASE_API.startswith("https://api.github.com/")


def test_only_a_whole_windows_installer_file_passes() -> None:
    listed = rl.pick_installer(RELEASE)
    assert listed is not None
    assert rl.msi_problem(MSI, listed) is None
    assert rl.msi_problem(MSI, rl.Installer(OFFICIAL_MSI)) is None
    assert "bytes" in str(rl.msi_problem(MSI[:-1], listed)), "cut short"
    assert "not the size" in str(rl.msi_problem(rl.MSI_MAGIC + bytes(10), rl.Installer(OFFICIAL_MSI)))
    assert "not a Windows Installer" in str(
        rl.msi_problem(b"<html>" + bytes(300_000), rl.Installer(OFFICIAL_MSI))
    )
    changed = rl.MSI_MAGIC + b"\x01" + MSI[9:]
    assert "SHA-256" in str(rl.msi_problem(changed, listed))
    # RLBot's v5 installer names RLBot v5 (an .msi of anything else, RLBot v4 included, is not run).
    other = rl.MSI_MAGIC + bytes(300_000)
    assert "not the installer of RLBot v5" in str(rl.msi_problem(other, rl.Installer(OFFICIAL_MSI)))
    older = {"assets": [{"name": "x.msi", "browser_download_url": rl.OFFICIAL + "v4/x.msi"}]}
    assert rl.pick_installer(older) is None, "only the release tagged installer (RLBot v5's)"


class _Slow(io.BytesIO):
    def __init__(self, data: bytes, pause: float) -> None:
        super().__init__(data)
        self.pause = pause

    def read(self, size: int | None = -1) -> bytes:
        time.sleep(self.pause)
        return super().read(size)


def test_a_stalled_or_failed_download_says_what_holds_downloads_back() -> None:
    assert rl.download("u", opener=lambda url, timeout: io.BytesIO(MSI)) == MSI
    with pytest.raises(
        rl.RLBotInstallError, match=r"stopped after 0\.001 minutes without finishing\. Check the"
    ):
        rl.download("u", minutes=0.001, opener=lambda url, timeout: _Slow(MSI, 0.02))

    def offline(url: str, timeout: float) -> Any:
        raise urllib.error.URLError("no route")

    with pytest.raises(rl.RLBotInstallError, match="could not be downloaded") as err:
        rl.download("u", opener=offline)
    assert str(err.value).endswith(rl.DOWNLOAD_HINT)


def test_the_hint_is_the_installers_own() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import install_steps
    finally:
        sys.path.remove(str(ROOT / "tools"))
    assert rl.DOWNLOAD_HINT == install_steps.DOWNLOAD_HINT


def _launcher(local: Path) -> Path:
    program = local / "RLBot5" / "bin" / "launcher.exe"
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_bytes(b"")
    return program


def test_install_opens_the_checked_installer_then_remembers_rlbot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs, tmp, local = tmp_path / "runs", tmp_path / ".tmp" / "rlbot", tmp_path / "Local"
    opened: list[tuple[Path, bytes]] = []

    def msiexec(msi: Path) -> int:
        opened.append((msi, msi.read_bytes()))
        _launcher(local)  # what RLBot's installer puts there
        return 0

    listed = rl.pick_installer(RELEASE)
    assert listed is not None
    code = rl.install(runs, tmp, True, local, lambda: listed, lambda url: MSI, msiexec)
    assert code == 0 and opened == [(tmp / "rlbot-v5-installer.msi", MSI)]
    assert not (tmp / "rlbot-v5-installer.msi").exists(), "the download is not kept"
    prefs = json.loads((runs / ".dashboard" / "prefs.json").read_text(encoding="utf-8"))
    assert prefs == {"theme": "dark", "rlbot_program": str(local / "RLBot5" / "bin" / "launcher.exe")}
    out = capsys.readouterr().out
    assert OFFICIAL_MSI in out and "RLBot v5 is installed" in out and "Apps list" in out
    # Found now: nothing is downloaded again.
    assert rl.install(runs, tmp, True, local, lambda: pytest.fail("looked up"), lambda url: b"", msiexec) == 0
    assert len(opened) == 1 and "installed already" in capsys.readouterr().out


def test_install_keeps_the_dashboards_own_choice_of_rlbot_v5(tmp_path: Path) -> None:
    runs, local = tmp_path / "runs", tmp_path / "Local"
    chosen = tmp_path / "elsewhere" / "RLBot5" / "bin" / "rlbotgui.exe"
    chosen.parent.mkdir(parents=True)
    chosen.write_bytes(b"")
    (runs / ".dashboard").mkdir(parents=True)
    prefs = runs / ".dashboard" / "prefs.json"
    prefs.write_text(json.dumps({"theme": "light", "rlbot_program": str(chosen)}))
    assert not rl.remember_program(runs, _launcher(local))
    assert json.loads(prefs.read_text())["rlbot_program"] == str(chosen)
    # A chosen RLBot v4 program is replaced by RLBot v5's launcher.
    older = tmp_path / "Local" / "RLBotGUIX" / "RLBotGUI.exe"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"")
    prefs.write_text(json.dumps({"theme": "light", "rlbot_program": str(older)}))
    assert rl.remember_program(runs, local / "RLBot5" / "bin" / "launcher.exe")
    assert json.loads(prefs.read_text())["rlbot_program"] == str(local / "RLBot5" / "bin" / "launcher.exe")
    (tmp_path / "Local" / "Programs").mkdir()
    _launcher(local / "Programs")
    assert len(rl.launchers(local)) == 2, "two launchers: the dashboard asks which one"


def test_install_runs_nothing_it_cannot_trust_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs, tmp, local = tmp_path / "runs", tmp_path / "tmp", tmp_path / "Local"
    ran: list[Path] = []

    def msiexec(msi: Path) -> int:
        ran.append(msi)
        return rl.MSI_CANCELLED

    listed = rl.pick_installer(RELEASE)
    assert listed is not None
    assert rl.install(runs, tmp, True, local, lambda: listed, lambda url: b"<html>", msiexec) == 1
    assert ran == [] and "Nothing from it was run" in capsys.readouterr().out

    def stalled(url: str) -> bytes:
        raise rl.RLBotInstallError("stopped. " + rl.DOWNLOAD_HINT)

    assert rl.install(runs, tmp, True, local, lambda: listed, stalled, msiexec) == 1 and ran == []
    # Cancelled in RLBot's own window: nothing found, and it says so.
    assert rl.install(runs, tmp, True, local, lambda: listed, lambda url: MSI, msiexec) == 1
    out = capsys.readouterr().out
    assert len(ran) == 1 and "cancelled" in out and "not found" in out
    assert not (runs / ".dashboard" / "prefs.json").exists()


def test_off_windows_it_only_says_where_to_look(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def never(*args: Any) -> Any:
        pytest.fail("nothing is looked up, downloaded or run")

    assert rl.install(tmp_path, tmp_path, False, tmp_path, never, never, never) == 1
    assert rl.status(tmp_path, False, tmp_path) == 1
    assert capsys.readouterr().out.count(rl.SITE) == 2


def test_status_says_where_rlbot_is(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    local = tmp_path / "Local"
    assert rl.status(tmp_path / "runs", True, local) == 1
    program = _launcher(local)
    assert rl.status(tmp_path / "runs", True, local) == 0
    assert str(program) in capsys.readouterr().out.splitlines()[-1]


def test_rlbot_v4_is_not_rlbot_v5(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """RLBot v4 (RLBotGUIX) does not run NitroLearn's bots: it is named, and RLBot v5 is still installed."""
    runs, tmp, local = tmp_path / "runs", tmp_path / "tmp", tmp_path / "Local"
    older = local / "RLBotGUIX" / "RLBotGUI.exe"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"")
    (runs / ".dashboard").mkdir(parents=True)
    (runs / ".dashboard" / "prefs.json").write_text(json.dumps({"rlbot_program": str(older)}))
    assert rl.status(runs, True, local) == 1
    out = capsys.readouterr().out
    assert f"RLBot v4 found at {older}; NitroLearn's bots need RLBot v5." in out and "v5: not found" in out
    listed = rl.pick_installer(RELEASE)
    assert listed is not None

    def msiexec(msi: Path) -> int:
        _launcher(local)
        return 0

    assert rl.install(runs, tmp, True, local, lambda: listed, lambda url: MSI, msiexec) == 0
    out = capsys.readouterr().out
    assert "installs next to it" in out and "RLBot v5 is installed" in out
    prefs = json.loads((runs / ".dashboard" / "prefs.json").read_text(encoding="utf-8"))
    assert prefs["rlbot_program"] == str(local / "RLBot5" / "bin" / "launcher.exe")


@pytest.mark.skipif(sys.platform == "win32", reason="checks the message off Windows")
def test_the_command_exists_and_needs_windows(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["install-rlbot", "--status"]) == 1
    assert main(["install-rlbot"]) == 1
    assert "for Windows" in capsys.readouterr().out


def test_the_dashboard_starts_it_as_a_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    monkeypatch.setattr("nitro.dashboard.api.sys.platform", "linux")
    assert "for Windows" in api.install_rlbot()["error"]
    monkeypatch.setattr("nitro.dashboard.api.sys.platform", "win32")
    started: list[tuple[str, str, list[str], Path]] = []
    api._jobs.start = lambda bot, kind, args, folder: started.append((bot, kind, args, folder))  # type: ignore[method-assign]
    assert api.install_rlbot() == {"started": True, "from": rl.OFFICIAL + "installer/"}
    assert started == [
        (".rlbot", "rlbot", ["install-rlbot", "--runs", str(runs)], runs / ".dashboard" / "rlbot")
    ]
    monkeypatch.setattr(rl, "local_app_data", lambda: tmp_path / "Local")
    assert api.install_rlbot_state() == {"state": "none", "text": "", "next": 0, "program": ""}
    program = _launcher(tmp_path / "Local")
    assert api.install_rlbot_state()["program"] == str(program)


def test_the_dashboard_opens_only_rlbot_v5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open RLBot: a remembered RLBot v4 program is forgotten (the dashboard asks again), and choosing one
    says that NitroLearn's bots need RLBot v5."""
    pytest.importorskip("webview")
    runs = tmp_path / "runs"
    api = Api(ROOT, runs)
    opened: list[Path] = []
    monkeypatch.setattr("nitro.dashboard.api._open", opened.append)
    older = tmp_path / "Local" / "RLBotGUIX" / "RLBotGUI.exe"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"")
    api.set_pref("rlbot_program", str(older))
    assert api.open_rlbot() == {"need_program": True} and opened == []
    assert api.home()["prefs"].get("rlbot_program") == ""

    class Window:
        def __init__(self, path: Path) -> None:
            self.path = path

        def create_file_dialog(self, *args: Any, **kwargs: Any) -> list[str]:
            return [str(self.path)]

    api._window = Window(older)
    answer = api.choose_rlbot()
    assert "RLBot v4" in answer["error"] and "RLBot5" in answer["error"] and opened == []
    launcher = _launcher(tmp_path / "Local")
    api._window = Window(launcher)
    assert api.choose_rlbot() == {"opened": str(launcher)} and opened == [launcher]
