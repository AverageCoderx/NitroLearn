"""`nitro doctor`: checks that this PC is ready to train, and says how to fix anything that isn't."""

from __future__ import annotations

import enum
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nitro import nvidia_smi, rlbot_find, toolchain

MIN_PYTHON = (3, 11)
MIN_VRAM_GB = 6.0
MIN_FREE_DISK_GB = 10.0  # the install itself is about 10 GB; this is room for saves and logs
MIN_FREE_TEMP_GB = 10.0
PROJECT = Path(__file__).resolve().parents[2]
INSTALLER = (
    "Run the installer again (NitroLearn-Setup into the same folder, or Install NitroLearn.cmd in "
    "NitroLearn's folder; on Linux: bash install.sh)."
)
# How Rust gets into NitroLearn's folder (only own Rust rewards and building from the code need it).
ADD_RUST = (
    'run the installer again with its part "Tools to build your own Rust rewards" ticked (NitroLearn-Setup, '
    "or Install NitroLearn.cmd in NitroLearn's folder; on Linux: bash install.sh --rust)"
)
GPU = "Graphics card"  # the graphics card check's name
# Said after a fix: the dashboard's Check this PC and `nitro doctor` show the same text.
CHECK_AGAIN = "check this PC again."
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
WEBVIEW2_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"
# Edge WebView2 Runtime's id in EdgeUpdate's registry keys.
WEBVIEW2_ID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


class Status(enum.Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    fix: str = ""


def python_result(version: tuple[int, int, int]) -> CheckResult:
    text = ".".join(str(part) for part in version)
    if version[:2] >= MIN_PYTHON:
        return CheckResult("Python", Status.OK, text)
    return CheckResult(
        "Python",
        Status.FAIL,
        f"{text} is too old",
        "Start NitroLearn with nitro.cmd (nitro.sh on Linux), which uses its own Python. " + INSTALLER,
    )


@dataclass(frozen=True)
class GpuFacts:
    torch_version: str
    torch_cuda: str | None
    available: bool
    name: str = ""
    capability: tuple[int, int] = (0, 0)
    arch_list: tuple[str, ...] = ()
    vram_gb: float = 0.0


def runs_on(arch_list: Sequence[str], major: int, minor: int) -> bool:
    """Whether a PyTorch build with these CUDA targets has code for a card of this compute capability:
    machine code for the same major version and an equal or lower minor one (a card of
    capability 8.9 runs a build's sm_86), or PTX (compute_XY) for an equal or lower capability, which the
    driver compiles."""
    for arch in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+)(\d)[a-z]?", arch)
        if not m:
            continue
        built = (int(m[2]), int(m[3]))
        if m[1] == "sm" and built[0] == major and built[1] <= minor:
            return True
        if m[1] == "compute" and built <= (major, minor):
            return True
    return False


def gpu_result(facts: GpuFacts) -> CheckResult:
    reinstall = INSTALLER + " It picks the PyTorch build for this card."
    # Training works on the processor too (far slower), so a missing card is a warning, not a failure.
    cpu_only = "Training will run on the processor, which is many times slower. "
    if facts.torch_cuda is None:
        return CheckResult(
            GPU,
            Status.WARN,
            f"PyTorch {facts.torch_version} is the build for the processor",
            cpu_only + f"For an NVIDIA card: {reinstall}",
        )
    if not facts.available:
        return CheckResult(
            GPU,
            Status.WARN,
            "PyTorch cannot see an NVIDIA graphics card",
            cpu_only + "If this PC has an NVIDIA card: update its driver and restart, then " + CHECK_AGAIN,
        )
    major, minor = facts.capability
    detail = f"{facts.name}, {facts.vram_gb:.1f} GB, compute {major}.{minor}, PyTorch {facts.torch_version}"
    if not runs_on(facts.arch_list, major, minor):
        return CheckResult(
            GPU, Status.FAIL, f"{detail} - this PyTorch build has no code for this card", reinstall
        )
    if facts.vram_gb < MIN_VRAM_GB:
        return CheckResult(
            GPU,
            Status.WARN,
            f"{detail} - less than {MIN_VRAM_GB:g} GB",
            "Use smaller networks and batches.",
        )
    return CheckResult(GPU, Status.OK, detail)


def gather_gpu_facts() -> GpuFacts:
    import torch

    facts = GpuFacts(torch.__version__, torch.version.cuda, torch.cuda.is_available())
    if not facts.available:
        return facts
    props = torch.cuda.get_device_properties(0)
    return GpuFacts(
        torch_version=facts.torch_version,
        torch_cuda=facts.torch_cuda,
        available=True,
        name=props.name,
        capability=(props.major, props.minor),
        arch_list=tuple(torch.cuda.get_arch_list()),
        vram_gb=props.total_memory / 1024**3,
    )


def core_result(package_version: str) -> CheckResult:
    try:
        from nitro import _core
    except ImportError as err:
        return CheckResult("Rust core", Status.FAIL, f"not built ({err})", INSTALLER)
    if _core.__version__ != package_version:
        return CheckResult(
            "Rust core",
            Status.FAIL,
            f"version {_core.__version__} does not match the Python package {package_version}",
            INSTALLER,
        )
    from nitro.build_info import core_status

    current, sentence = core_status()
    if not current:
        return CheckResult("Rust core", Status.FAIL, "older than the code in this folder", INSTALLER)
    return CheckResult("Rust core", Status.OK, f"{_core.__version__}, {sentence}")


def meshes_result(folder: Path) -> CheckResult:
    from nitro import _core

    fix = (
        'Run "nitro fetch-meshes" (the installer does it too), or point [paths] collision_meshes in your '
        "settings file at a folder that contains a 'soccar' subfolder."
    )
    try:
        report = _core.load_collision_meshes(folder)
    except _core.MeshLoadError as err:
        return CheckResult("Collision meshes", Status.FAIL, str(err), fix)
    detail = f"{report.soccar_recognized}/{report.soccar_expected} soccar meshes in {report.folder}"
    if not report.is_complete():
        return CheckResult(
            "Collision meshes",
            Status.FAIL,
            f"{detail} ({report.soccar_files} files found)",
            'Some files are not genuine soccar meshes: run "nitro fetch-meshes --dest <that folder>" to put '
            "the right ones there.",
        )
    return CheckResult("Collision meshes", Status.OK, detail)


def rust_toolchain_result(cargo: Path | None) -> CheckResult:
    if cargo is not None:
        return CheckResult("Rust toolchain", Status.OK, str(cargo))
    return CheckResult(
        "Rust toolchain",
        Status.WARN,
        "cargo not found",
        f"Only needed for your own Rust rewards (or to build NitroLearn from its code): {ADD_RUST}.",
    )


def find_cargo() -> Path | None:
    """The cargo NitroLearn's builds use: the folder's own Rust first, else one on PATH (nitro.toolchain)."""
    return toolchain.find_cargo()


@dataclass(frozen=True)
class DriverFacts:
    version: str
    cuda: tuple[int, int] | None  # the newest CUDA the driver runs


def driver_result(driver: DriverFacts | None, torch_cuda: str | None) -> CheckResult:
    """The NVIDIA driver against the CUDA version PyTorch was built for."""
    if driver is None:
        return CheckResult(
            "NVIDIA driver",
            Status.WARN,
            "not found (nvidia-smi is missing or does not answer)",
            "Only for an NVIDIA card: install its driver from https://www.nvidia.com/drivers, restart the "
            "PC, then run the installer again (it then installs the PyTorch build for the card).",
        )
    runs = f"{driver.cuda[0]}.{driver.cuda[1]}" if driver.cuda else "an unknown CUDA version"
    detail = f"version {driver.version}, runs CUDA {runs}"
    if torch_cuda is None:
        return CheckResult(
            "NVIDIA driver",
            Status.WARN,
            f"{detail}, but PyTorch is the build for the processor",
            INSTALLER + " It installs the PyTorch build for the card.",
        )
    wanted = tuple(int(part) for part in torch_cuda.split(".")[:2])
    if driver.cuda is not None and driver.cuda < wanted:
        return CheckResult(
            "NVIDIA driver",
            Status.WARN,
            f"{detail}; this PyTorch is built for CUDA {torch_cuda}",
            "Update the NVIDIA driver (https://www.nvidia.com/drivers, or the NVIDIA app), restart the PC, "
            "then " + CHECK_AGAIN,
        )
    return CheckResult("NVIDIA driver", Status.OK, detail)


def parse_nvidia_smi(text: str) -> DriverFacts | None:
    """The driver's version and CUDA version from `nvidia-smi`'s table header (newer drivers name the
    version "KMD Version" and the CUDA version "CUDA UMD Version")."""
    version = re.search(r"(?:Driver|KMD) Version:\s*([0-9.]+)", text)
    if version is None:
        return None
    cuda = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", text)
    return DriverFacts(version[1], (int(cuda[1]), int(cuda[2])) if cuda else None)


def gather_driver() -> DriverFacts | None:
    smi = nvidia_smi.find()  # also where Windows' driver puts it, off the PATH
    if smi is None:
        return None
    try:
        done = subprocess.run([smi], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(done.stdout) if done.returncode == 0 else None


def vc_runtime_result(found: bool) -> CheckResult:
    """Windows: PyTorch needs Microsoft's Visual C++ runtime (msvcp140.dll), which it does not bring."""
    if found:
        return CheckResult("Visual C++ runtime", Status.OK, "found")
    return CheckResult(
        "Visual C++ runtime",
        Status.FAIL,
        "msvcp140.dll not found: PyTorch cannot load without it",
        f"Install Microsoft's Visual C++ Redistributable (x64) from {VC_REDIST_URL}, then {CHECK_AGAIN} It "
        "is a small part of Windows that many games install too; NitroLearn does not install it for you "
        "because it changes Windows itself.",
    )


def vc_runtime_found() -> bool:
    if sys.platform != "win32":
        return True
    import ctypes

    try:
        ctypes.WinDLL("msvcp140.dll")
    except OSError:
        return False
    return True


def webview2_result(version: str | None) -> CheckResult:
    """Windows: the dashboard's window is drawn by Edge WebView2 (without it, pywebview falls back to an old
    Internet Explorer engine the dashboard does not work in)."""
    if version:
        return CheckResult("Dashboard window", Status.OK, f"Edge WebView2 {version}")
    return CheckResult(
        "Dashboard window",
        Status.WARN,
        "Microsoft Edge WebView2 Runtime not found",
        "Only the dashboard needs it. Install the WebView2 Runtime (Evergreen Bootstrapper) from "
        f"{WEBVIEW2_URL}, then open the dashboard again. Windows 11 has it already; training and the "
        "commands work without it.",
    )


def webview2_version() -> str | None:
    """WebView2 Runtime's version from EdgeUpdate's registry keys (read only), as pywebview looks for it."""
    if sys.platform != "win32":
        return None
    import winreg

    places = (
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_ID}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_ID}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_ID}"),
    )
    for hive, key in places:
        try:
            with winreg.OpenKey(hive, key) as opened:
                value = str(winreg.QueryValueEx(opened, "pv")[0])
        except OSError:
            continue
        if value and value != "0.0.0.0":
            return value
    return None


@dataclass(frozen=True)
class RLBotFound:
    program: Path | None  # RLBot v5's program
    older: Path | None = None  # an RLBot v4 install: NitroLearn's bots do not run there
    not_v5_choice: Path | None = None  # the dashboard's chosen program, when it is not RLBot v5's


def rlbot_result(found: RLBotFound) -> CheckResult:
    get_v5 = (
        'run "nitro install-rlbot" (RLBot\'s own installer, from its official download) or get RLBot v5 from '
        "https://rlbot.org"
    )
    ignored = (
        f"; the program chosen in the dashboard, {found.not_v5_choice}, is not RLBot v5's and is not used"
        if found.not_v5_choice
        else ""
    )
    if found.program is not None:
        return CheckResult("RLBot", Status.OK, f"{found.program}{ignored}")
    if found.older is not None:
        return CheckResult(
            "RLBot",
            Status.WARN,
            f"RLBot v4 found at {found.older}; NitroLearn's bots need RLBot v5{ignored}",
            f"RLBot v5 installs next to RLBot v4 (which stays as it is): {get_v5}.",
        )
    if found.not_v5_choice is not None:
        return CheckResult(
            "RLBot",
            Status.WARN,
            f"RLBot v5 not found{ignored}",
            f"NitroLearn's bots need RLBot v5: {get_v5}.",
        )
    return CheckResult(
        "RLBot",
        Status.WARN,
        "not found yet",
        f"Only needed to play your bots in Rocket League: {get_v5}. NitroLearn finds it by itself.",
    )


def look_for_rlbot(runs_folders: Sequence[Path], local_app_data: Path | None) -> RLBotFound:
    """RLBot v5's program: the one chosen in the dashboard (Open RLBot) when it is RLBot v5's, else RLBot v5's
    launcher among the user's local programs; with an RLBot v4 install and a chosen program that is not
    RLBot v5's, which are not used (NitroLearn's bots are RLBot v5 bots)."""
    not_v5 = None
    for runs in runs_folders:
        try:
            prefs = json.loads((runs / ".dashboard" / "prefs.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        chosen = prefs.get("rlbot_program") if isinstance(prefs, dict) else None
        if not chosen or not Path(chosen).is_file():
            continue
        if rlbot_find.is_v5(chosen):
            return RLBotFound(Path(chosen), rlbot_find.find_older(local_app_data))
        not_v5 = not_v5 or Path(chosen)
    return RLBotFound(rlbot_find.find_v5(local_app_data), rlbot_find.find_older(local_app_data), not_v5)


def find_rlbot(runs_folders: Sequence[Path], local_app_data: Path | None) -> Path | None:
    """RLBot v5's program (see look_for_rlbot), or None."""
    return look_for_rlbot(runs_folders, local_app_data).program


def disk_result(folder: Path, free_gb: float) -> CheckResult:
    detail = f"{free_gb:.0f} GB free at {folder.anchor or folder}"
    if free_gb < MIN_FREE_DISK_GB:
        return CheckResult(
            "Disk space",
            Status.WARN,
            detail,
            f"Your bots' saves and logs need room: free up space to at least {MIN_FREE_DISK_GB:g} GB.",
        )
    return CheckResult("Disk space", Status.OK, detail)


def temp_space_result(folder: Path, free_gb: float) -> CheckResult:
    detail = f"{free_gb:.1f} GB free at {folder}"
    if free_gb < MIN_FREE_TEMP_GB:
        return CheckResult(
            "Temp folder",
            Status.WARN,
            detail,
            f"PyTorch and the installer write temporary files here; free up {MIN_FREE_TEMP_GB:g} GB on "
            f"{folder.anchor} or point the TEMP and TMP environment variables at a bigger drive.",
        )
    return CheckResult("Temp folder", Status.OK, detail)


def torch_result(err: BaseException, vc_found: bool) -> CheckResult:
    """PyTorch itself could not be loaded."""
    fix = "Install the Visual C++ runtime first (see its check)." if not vc_found else INSTALLER
    return CheckResult("PyTorch", Status.FAIL, f"cannot be loaded ({err})", fix)


def run_checks(meshes: Path, package_version: str) -> list[CheckResult]:
    major, minor, micro = sys.version_info[:3]
    results = [python_result((major, minor, micro))]
    windows = sys.platform == "win32"
    vc_found = vc_runtime_found()
    if windows:
        results.append(vc_runtime_result(vc_found))
    torch_cuda = None
    try:
        facts = gather_gpu_facts()
    except (ImportError, OSError) as err:
        results.append(torch_result(err, vc_found))
    else:
        torch_cuda = facts.torch_cuda
        results.append(gpu_result(facts))
    results.append(driver_result(gather_driver(), torch_cuda))
    core = core_result(package_version)
    results.append(core)
    if core.status is Status.OK:
        results.append(meshes_result(meshes))
    results.append(rust_toolchain_result(find_cargo()))
    if windows:
        results.append(webview2_result(webview2_version()))
        local = os.environ.get("LOCALAPPDATA")
        programs = Path(local) if local else None
        results.append(rlbot_result(look_for_rlbot([Path.cwd() / "runs", PROJECT / "runs"], programs)))
    here = Path.cwd()
    results.append(disk_result(here, shutil.disk_usage(here).free / 1024**3))
    temp = Path(tempfile.gettempdir())
    results.append(temp_space_result(temp, shutil.disk_usage(temp).free / 1024**3))
    return results


def format_report(results: Sequence[CheckResult], package_version: str) -> str:
    width = max(len(result.name) for result in results)
    lines = [f"NitroLearn {package_version} - system check"]
    for result in results:
        lines.append(f"  [{result.status.value:^4}] {result.name:<{width}}  {result.detail}")
        if result.fix and result.status is not Status.OK:
            lines.append(f"  {'':6} {'':<{width}}  fix: {result.fix}")
    lines.append(verdict(results))
    return "\n".join(lines)


def verdict(results: Sequence[CheckResult]) -> str:
    """The report's last line: how many problems must be fixed, or that all is well."""
    failures = sum(result.status is Status.FAIL for result in results)
    if failures:
        return f"{failures} problem{'s' if failures != 1 else ''} must be fixed before training."
    return "All required checks passed."


def as_data(results: Sequence[CheckResult], package_version: str) -> dict[str, Any]:
    """The report as plain data (`nitro doctor --json`), for a program to read: the dashboard's Check this
    PC page shows it. A check's fix is given only when it is not OK, as in the printed report."""
    return {
        "version": package_version,
        "checks": [
            {
                "name": result.name,
                "status": result.status.value,
                "detail": result.detail,
                "fix": result.fix if result.status is not Status.OK else "",
            }
            for result in results
        ],
        "verdict": verdict(results),
    }
