from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nitro import __version__, _core, doctor, rlbot_find
from nitro.doctor import (
    CheckResult,
    DriverFacts,
    GpuFacts,
    RLBotFound,
    Status,
    core_result,
    disk_result,
    driver_result,
    find_rlbot,
    format_report,
    gpu_result,
    look_for_rlbot,
    meshes_result,
    parse_nvidia_smi,
    python_result,
    rlbot_result,
    run_checks,
    rust_toolchain_result,
    temp_space_result,
    vc_runtime_result,
    webview2_result,
)

PROJECT = Path(__file__).resolve().parents[1]
MESHES = PROJECT / "collision_meshes"

GOOD_GPU = GpuFacts(
    torch_version="2.14.1+cu126",
    torch_cuda="12.6",
    available=True,
    name="NVIDIA GeForce Test Card",
    capability=(7, 5),
    arch_list=("sm_70", "sm_75", "sm_80"),
    vram_gb=8.0,
)


def test_python_version_rule() -> None:
    assert python_result((3, 11, 9)).status is Status.OK
    assert python_result((3, 12, 0)).status is Status.OK
    old = python_result((3, 10, 14))
    assert old.status is Status.FAIL and "3.10.14" in old.detail


def test_gpu_rules() -> None:
    assert gpu_result(GOOD_GPU).status is Status.OK
    # No usable GPU is a warning: training still runs on the CPU (Linux and cloud machines).
    cpu_build = gpu_result(GpuFacts("2.14.1+cpu", None, False))
    assert cpu_build.status is Status.WARN and "build for the processor" in cpu_build.detail
    assert "processor" in cpu_build.fix
    no_device = gpu_result(GpuFacts("2.14.1+cu126", "12.6", False))
    assert no_device.status is Status.WARN and "driver" in no_device.fix
    unsupported = GpuFacts(**{**GOOD_GPU.__dict__, "arch_list": ("sm_80", "sm_90")})
    assert gpu_result(unsupported).status is Status.FAIL
    # A card of capability 8.9 runs PyTorch cu126's sm_86 code (no sm_89 in it)
    newer_minor = GpuFacts(
        **{**GOOD_GPU.__dict__, "capability": (8, 9), "arch_list": ("sm_80", "sm_86", "sm_90")}
    )
    assert gpu_result(newer_minor).status is Status.OK
    ptx = GpuFacts(**{**GOOD_GPU.__dict__, "capability": (12, 0), "arch_list": ("sm_90", "compute_90")})
    assert gpu_result(ptx).status is Status.OK
    newer_only = GpuFacts(**{**GOOD_GPU.__dict__, "capability": (8, 0), "arch_list": ("sm_86", "sm_90")})
    assert gpu_result(newer_only).status is Status.FAIL
    small = GpuFacts(**{**GOOD_GPU.__dict__, "vram_gb": 4.0})
    assert gpu_result(small).status is Status.WARN


def test_disk_and_rust_rules() -> None:
    assert disk_result(Path("D:/runs"), 100.0).status is Status.OK
    assert disk_result(Path("D:/runs"), 5.0).status is Status.WARN
    assert temp_space_result(Path("C:/Temp"), 50.0).status is Status.OK
    low_temp = temp_space_result(Path("C:/Temp"), 2.4)
    assert low_temp.status is Status.WARN and "TEMP and TMP" in low_temp.fix
    assert rust_toolchain_result(Path("C:/cargo.exe")).status is Status.OK
    assert rust_toolchain_result(None).status is Status.WARN


def test_core_version_must_match_package() -> None:
    assert core_result(__version__).status is Status.OK
    mismatch = core_result("0.0.0-other")
    assert mismatch.status is Status.FAIL and "Install NitroLearn.cmd" in mismatch.fix


def test_missing_mesh_folder_is_a_failure_with_a_fix(tmp_path: Path) -> None:
    result = meshes_result(tmp_path / "nothing_here")
    assert result.status is Status.FAIL
    assert "does not exist" in result.detail
    assert "nitro fetch-meshes" in result.fix


def test_unrecognised_mesh_files_are_a_failure(tmp_path: Path) -> None:
    (tmp_path / "soccar").mkdir()
    (tmp_path / "soccar" / "mesh_0.cmf").write_bytes(b"not a mesh")
    result = meshes_result(tmp_path)
    assert result.status is Status.FAIL


def test_report_lists_fixes_only_for_problems() -> None:
    results = [
        CheckResult("Python", Status.OK, "3.11.9", "never shown"),
        CheckResult("Disk space", Status.WARN, "5 GB free", "free up space"),
        CheckResult("GPU", Status.FAIL, "not found", "update driver"),
    ]
    text = format_report(results, "9.9.9")
    assert "never shown" not in text
    assert "fix: free up space" in text and "fix: update driver" in text
    assert text.endswith("1 problem must be fixed before training.")


@pytest.mark.skipif(not MESHES.is_dir(), reason="no collision meshes here (nitro fetch-meshes)")
def test_this_pc_passes_every_required_check() -> None:
    results = run_checks(MESHES, __version__)
    failures = [result for result in results if result.status is Status.FAIL]
    assert failures == []
    names = {result.name for result in results}
    assert {"Python", "Graphics card", "Rust core", "Collision meshes", "Disk space"} <= names


@pytest.mark.skipif(not MESHES.is_dir(), reason="no collision meshes here (nitro fetch-meshes)")
def test_core_reports_recognised_meshes() -> None:
    report = _core.load_collision_meshes(MESHES)
    assert report.is_complete()
    assert (report.soccar_files, report.soccar_recognized, report.soccar_expected) == (16, 16, 16)
    assert Path(report.folder) == MESHES.resolve()


SMI = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.16                 Driver Version: 572.16         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
"""


def test_the_driver_is_checked_against_the_pytorch_build() -> None:
    facts = parse_nvidia_smi(SMI)
    assert facts == DriverFacts("572.16", (12, 8))
    umd = SMI.replace("CUDA Version", "CUDA UMD Version").replace("12.8", "13.4")
    assert parse_nvidia_smi(umd) == DriverFacts("572.16", (13, 4))
    newer = "| NVIDIA-SMI 610.88     KMD Version: 610.88     CUDA UMD Version: 13.3     |\n"
    assert parse_nvidia_smi(newer) == DriverFacts("610.88", (13, 3)), "newer drivers' header"
    assert parse_nvidia_smi("No devices were found") is None
    assert driver_result(facts, "12.6").status is Status.OK
    old = driver_result(DriverFacts("551.0", (12, 4)), "12.6")
    assert old.status is Status.WARN and "Update the NVIDIA driver" in old.fix
    cpu = driver_result(facts, None)
    assert cpu.status is Status.WARN and "PyTorch build for the card" in cpu.fix
    missing = driver_result(None, None)
    assert missing.status is Status.WARN and "nvidia.com/drivers" in missing.fix


def test_the_windows_parts_say_where_to_get_them(tmp_path: Path) -> None:
    assert vc_runtime_result(True).status is Status.OK
    vc = vc_runtime_result(False)
    assert vc.status is Status.FAIL and "vc_redist.x64.exe" in vc.fix and "does not install it" in vc.fix
    assert webview2_result("131.0.2903.70").status is Status.OK
    webview = webview2_result(None)
    assert webview.status is Status.WARN and "Evergreen Bootstrapper" in webview.fix
    missing = rlbot_result(RLBotFound(None))
    assert missing.status is Status.WARN and "rlbot.org" in missing.fix
    # RLBot: the program chosen in the dashboard, else RLBot v5's among the user's programs.
    runs = tmp_path / "runs"
    (runs / ".dashboard").mkdir(parents=True)
    local = tmp_path / "AppData" / "Local"
    assert find_rlbot([runs], local) is None
    program = local / "Programs" / "RLBot5" / "bin" / "RLBotServer.exe"
    program.parent.mkdir(parents=True)
    program.write_bytes(b"")
    assert find_rlbot([runs], local) == program
    chosen = tmp_path / "elsewhere" / "RLBot5" / "bin" / "rlbotgui.exe"
    chosen.parent.mkdir(parents=True)
    chosen.write_bytes(b"")
    (runs / ".dashboard" / "prefs.json").write_text(json.dumps({"rlbot_program": str(chosen)}), "utf-8")
    assert find_rlbot([tmp_path / "none", runs], local) == chosen
    # RLBot v5's launcher (what its own shortcuts open) before its other programs.
    launcher = local / "Programs" / "RLBot5" / "bin" / "launcher.exe"
    launcher.write_bytes(b"")
    (runs / ".dashboard" / "prefs.json").unlink()
    assert find_rlbot([runs], local) == launcher


def test_only_rlbot_v5_counts_and_rlbot_v4_is_named(tmp_path: Path) -> None:
    """NitroLearn's bots are RLBot v5 bots: RLBot v4 (RLBotGUIX, its RLBotGUI.exe) is never taken for RLBot,
    a chosen v4 program is ignored, and the check says so."""
    runs, local = tmp_path / "runs", tmp_path / "Local"
    (runs / ".dashboard").mkdir(parents=True)
    older = local / "RLBotGUIX" / "RLBotGUI.exe"
    older.parent.mkdir(parents=True)
    older.write_bytes(b"")
    (runs / ".dashboard" / "prefs.json").write_text(json.dumps({"rlbot_program": str(older)}), "utf-8")
    found = look_for_rlbot([runs], local)
    assert found == RLBotFound(None, older, older) and find_rlbot([runs], local) is None
    result = rlbot_result(found)
    assert result.status is Status.WARN
    assert f"RLBot v4 found at {older}; NitroLearn's bots need RLBot v5" in result.detail
    assert "is not RLBot v5's and is not used" in result.detail and "next to RLBot v4" in result.fix
    # A v4 folder alone (no program in it) is named too.
    (runs / ".dashboard" / "prefs.json").unlink()
    older.unlink()
    assert look_for_rlbot([runs], local) == RLBotFound(None, older.parent)
    # With RLBot v5 there, v5 is RLBot.
    launcher = local / "RLBot5" / "bin" / "launcher.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"")
    assert rlbot_result(look_for_rlbot([runs], local)) == CheckResult("RLBot", Status.OK, str(launcher))
    # Which programs are RLBot v5's, by their folder (any slash, any letter case).
    assert rlbot_find.is_v5("C:\\Users\\me\\AppData\\Local\\RLBot5\\bin\\launcher.exe")
    assert rlbot_find.is_v5("C:/Users/me/AppData/Local/rlbot5/bin/RLBotServer.exe")
    for other in (
        "C:\\Users\\me\\AppData\\Local\\RLBotGUIX\\RLBotGUI.exe",
        "C:\\Users\\me\\AppData\\Local\\RLBotGUIX\\Python311\\Lib\\site-packages\\rlbot\\dll\\RLBot.exe",
        "C:\\Games\\RocketLeague.exe",
    ):
        assert not rlbot_find.is_v5(other), other
    assert rlbot_find.is_older("C:\\Users\\me\\AppData\\Local\\RLBotGUIX\\RLBotGUI.exe")
    assert not rlbot_find.is_older("C:\\Games\\RocketLeague.exe")


def test_on_windows_a_missing_runtime_is_named_before_pytorch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without msvcp140.dll PyTorch cannot load: the check says which part to install instead of crashing."""

    def no_torch() -> GpuFacts:
        raise OSError("[WinError 126] The specified module could not be found")

    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr(doctor, "vc_runtime_found", lambda: False)
    monkeypatch.setattr(doctor, "gather_gpu_facts", no_torch)
    monkeypatch.setattr(doctor, "gather_driver", lambda: DriverFacts("572.16", (12, 8)))
    monkeypatch.setattr(doctor, "webview2_version", lambda: None)
    monkeypatch.setattr(doctor, "look_for_rlbot", lambda runs, local: RLBotFound(None))
    monkeypatch.setattr(doctor, "core_result", lambda version: CheckResult("Rust core", Status.FAIL, "x"))
    results = {result.name: result for result in run_checks(Path("nowhere"), "1.0.0")}
    assert results["Visual C++ runtime"].status is Status.FAIL
    assert results["PyTorch"].status is Status.FAIL and "Visual C++" in results["PyTorch"].fix
    assert {"NVIDIA driver", "Dashboard window", "RLBot"} <= set(results)


def test_the_fixes_read_plainly_in_the_dashboard_too() -> None:
    """The fixes show on the dashboard's Check this PC page as they are: no command in backticks, no
    "GPU"/"CPU", links in full, and the disk check matches the README's "about 10 GB to install, plus room for
    your bots' saves"."""
    results = [
        doctor.gpu_result(doctor.GpuFacts("2.14.1+cpu", None, False)),
        doctor.gpu_result(doctor.GpuFacts("2.14.1+cu126", "12.6", False)),
        doctor.gpu_result(doctor.GpuFacts("2.14.1+cu126", "12.6", True, "Card", (8, 6), ("sm_80",), 2.0)),
        doctor.driver_result(None, None),
        doctor.driver_result(doctor.DriverFacts("1", (12, 0)), "12.6"),
        doctor.driver_result(doctor.DriverFacts("1", (12, 6)), None),
        doctor.vc_runtime_result(False),
        doctor.rlbot_result(doctor.RLBotFound(None)),
        doctor.rust_toolchain_result(None),
        doctor.disk_result(Path("/"), 5.0),
        doctor.temp_space_result(Path("/"), 5.0),
    ]
    for result in results:
        said = f"{result.name} {result.detail} {result.fix}"
        assert "`" not in said and not re.search(r"\b(GPU|CPU)\b", said), said
        for link in re.finditer(r"[\w.-]+\.(?:com|org)\b", said):
            assert said[: link.start()].endswith("https://"), said
    assert doctor.MIN_FREE_DISK_GB == 10
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    said = " ".join(readme.split())
    assert "About 10 GB of free disk space (about 2 GB without an NVIDIA card) to install, plus room" in said
