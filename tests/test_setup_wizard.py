"""NitroLearn-Setup.exe (installer/NitroLearn.iss, compiled by Inno Setup in the release workflow), checked
here without Windows. Its Components page passes exactly the installer's parts, its defaults are the
menu's, the checklist follows the installer's own steps, the uninstaller removes NitroLearn's folders but
asks before the user's own, and the script names no machine's paths. Compiling and running it need
Windows."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ISS = PROJECT / "installer" / "NitroLearn.iss"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


steps = _load("install_steps_for_wizard_test", PROJECT / "tools" / "install_steps.py")
wizard = _load("setup_wizard_under_test", PROJECT / "tools" / "setup_wizard.py")
images = _load("make_wizard_images_under_test", PROJECT / "tools" / "make_wizard_images.py")


def _iss() -> str:
    return ISS.read_text(encoding="utf-8")


def _section(name: str) -> list[str]:
    """The lines of one [Section] of the script, without comments and blank lines."""
    found = re.search(rf"^\[{name}\]\n(.*?)(?=^\[|\Z)", _iss(), re.MULTILINE | re.DOTALL)
    assert found is not None, name
    return [line for line in found[1].splitlines() if line.strip() and not line.startswith(";")]


def _setup() -> dict[str, str]:
    return dict(line.split("=", 1) for line in _section("Setup"))


def _components() -> dict[str, str]:
    """Each component's name and its whole line."""
    return {re.search(r'Name: "([^"]+)"', line)[1]: line for line in _section("Components")}  # type: ignore[index]


def test_the_components_are_the_installers_parts_and_reach_it_as_parts() -> None:
    parts = re.search(r'^#define Parts "([^"]+)"', _iss(), re.MULTILINE)
    assert parts is not None and parts[1] == ",".join(steps.PARTS)
    assert list(_components()) == ["main", *steps.PARTS]
    assert "Flags: fixed" in _components()["main"] and "Types: recommended custom" in _components()["main"]
    code = _iss()[_iss().index("[Code]") :]
    assert "-Parts \"' + SelectedParts + '\" -NoLaunch -Progress" in code
    assert "Result := 'none'" in code, "no part ticked still says so"
    # Every choice of ticks reaches install_steps.py as exactly those parts, the rest off.
    for count in range(len(steps.PARTS) + 1):
        for chosen in itertools.combinations(steps.PARTS, count):
            args = steps.parse(["--uv", "uv", "--parts", ",".join(chosen) or "none"])
            assert steps.switched(args) == {part: part in chosen for part in steps.PARTS}
    with pytest.raises(SystemExit):
        steps.parse(["--uv", "uv", "--parts", "gpu,teleport"])
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    assert "[AllowEmptyString()][string]$Parts" in ps1 and '$steps += "--parts=$Parts"' in ps1
    assert "[switch]$Progress" in ps1 and "$steps += '--progress'" in ps1
    assert '$steps += "--bundles=$ReleaseFiles"' in ps1 and "[string]$Bundles" in ps1


def test_the_recommended_ticks_are_the_menus_first_install_defaults() -> None:
    found = steps.Found(True, True, "NVIDIA GeForce RTX 3070", "", True, False, True, {})
    defaults = steps.effective(found, steps.default_wishes(found, {}))
    ticked = {name for name, line in _components().items() if "recommended" in line and name != "main"}
    assert ticked == {part for part, on in defaults.items() if on}


def test_rlbot_v5_installed_already_is_unticked_and_said_so() -> None:
    """The Components page looks for RLBot v5 as NitroLearn does (python/nitro/rlbot_find.py: RLBot5\\bin's
    programs under %LOCALAPPDATA% or its Programs folder; an older RLBot does not count). Found, its line
    is unticked, greyed and says it is installed, also after another type is chosen; the install then
    neither lists nor runs RLBot's installer."""
    finder = _load("rlbot_find_for_wizard_test", PROJECT / "python" / "nitro" / "rlbot_find.py")
    code = _iss()[_iss().index("[Code]") :]
    programs = re.search(r"RLBotPrograms = '([^']+)';", code)
    assert programs is not None and tuple(programs[1].split(",")) == finder.V5_PROGRAMS
    assert f"RLBotFolder = '{finder.V5_FOLDER}\\bin\\';" in code
    found = code[code.index("function RLBotV5Found") : code.index("procedure SettleItem")]
    assert "Bases[0] := ExpandConstant('{localappdata}');" in found
    assert "Bases[1] := ExpandConstant('{localappdata}\\Programs');" in found
    assert "FileExists(AddBackslash(Bases[I]) + RLBotFolder + Name)" in found
    order = list(_components())
    assert f"GpuItem = {order.index('gpu')};" in code and f"RLBotItem = {order.index('rlbot')};" in code
    settle = code[code.index("procedure SettleItem") : code.index("procedure ShowCard")]
    assert "Checked[Index] := False" in settle and "ItemEnabled[Index] := False" in settle
    assert "if RLBotInstalled then\n    SettleItem(RLBotItem, 'RLBot v5: already installed" in settle
    assert "RLBotInstalled := RLBotV5Found;" in code
    assert "WizardForm.TypesCombo.OnChange := @TypesChanged;" in code
    assert "TypesChangedBefore(Sender);" in code, "Inno Setup's own handling of the type comes first"
    assert "if WizardIsComponentSelected('rlbot') then\n    AddStep('rlbot'" in code


def test_the_checklist_follows_the_installers_own_steps() -> None:
    keys = re.findall(r"AddStep\('(\w+)'", _iss())
    assert keys == ["pc", "python", "torch", "torch", "core", "rust", "meshes", "recordings", "check",
                    "shortcuts", "shortcuts", "rlbot"]  # fmt: skip
    source = (PROJECT / "tools" / "install_steps.py").read_text(encoding="utf-8")
    ps1 = (PROJECT / "install.ps1").read_text(encoding="utf-8")
    for key in set(keys):
        assert f'step("{key}", "start")' in source or f"StepState '{key}' 'start'" in ps1, key
    assert "Copy(Line, 1, 8) = '@@nitro '" in _iss()
    wizard_only = re.search(r"^if \(\$Progress\) \{\n(.*?)^\}", ps1, re.MULTILINE | re.DOTALL)
    assert wizard_only is not None and "$env:UV_NO_PROGRESS = '1'" in wizard_only[1], "no uv progress bars"
    assert "$env:UV_NO_PROGRESS" not in ps1.replace(wizard_only[0], ""), "the console install keeps them"
    assert "Copy(Line, 1, 15) = 'SETUP STOPPED: '" in _iss(), "the plain stop message is shown"


def test_the_setup_is_per_user_named_for_the_website_and_versioned_from_cargo_toml() -> None:
    setup = _setup()
    assert setup["OutputBaseFilename"] == "NitroLearn-Setup"
    assert setup["PrivilegesRequired"] == "lowest"
    assert setup["DefaultDirName"] == r"{localappdata}\Programs\NitroLearn"
    assert setup["AppVersion"] == "{#AppVersion}" and "#ifndef AppVersion\n  #error" in _iss()
    assert setup["WizardStyle"] == "modern" and setup["DisableWelcomePage"] == "no"
    assert setup["CreateUninstallRegKey"] == "yes", "the Apps-list entry; one line to switch it off"
    assert setup["AppPublisher"] == "Raze" and setup["AppSupportURL"] == "https://discord.gg/bYP3kYtryg"
    assert ";SignTool=" in _iss() and "SignTool=" not in "".join(_section("Setup")), "signing: ready, off"
    workflow = (PROJECT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "choco install innosetup" in workflow and ".VersionInfo.ProductVersion" in workflow
    assert '"/DAppVersion=$version" installer\\NitroLearn.iss' in workflow
    assert "python tools/setup_wizard.py" in workflow
    assert "cp artifacts/bundles/* artifacts/setup/NitroLearn-Setup.exe release/" in workflow
    assert "python tools/bundles.py check release" in workflow


def test_the_script_names_no_machines_paths_and_its_files_exist() -> None:
    for number, line in enumerate(_iss().splitlines(), 1):
        if not line.lstrip().startswith(";"):
            assert not re.search(r"\b[A-Za-z]:\\", line), f"line {number}: an absolute path"
    setup = _setup()
    named = [*setup["WizardImageFile"].split(","), *setup["WizardSmallImageFile"].split(",")]
    for name in [*named, setup["SetupIconFile"]]:
        assert (ISS.parent / name.replace("\\", "/")).is_file(), name
    # Made by tools/setup_wizard.py before compiling.
    assert setup["LicenseFile"] == r"build\license.rtf"
    sources = [re.search(r'Source: "([^"]+)"', line)[1] for line in _section("Files")]  # type: ignore[index]
    assert sources == [r"build\app\*", r"build\uv\*"]
    sizes = {name: _png_size(ISS.parent / name) for name in named}
    assert sizes["wizard-100.png"] == images.PANEL and sizes["wizard-small-100.png"] == (55, 55)
    assert sizes["wizard-200.png"] == (328, 628)


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_uninstalling_removes_nitrolearns_folders_and_asks_before_the_users_own() -> None:
    deleted = {
        re.search(r'Name: "\{app\}\\([^"]+)"', line)[1]  # type: ignore[index]
        for line in _section("UninstallDelete")
    }
    listed = subprocess.run(["git", "ls-files"], cwd=PROJECT, capture_output=True, text=True, check=True)
    shipped = {path.split("/")[0] for path in listed.stdout.splitlines() if "/" in path}
    assert shipped <= deleted, f"an update can add files there: {sorted(shipped - deleted)}"
    assert {".venv", ".tools", ".cache", ".tmp", "collision_meshes", "rocketsim_recordings"} <= deleted
    assert "*" in deleted, "the files at the top of the folder"
    users = re.search(r"UserFolders = '([^']+)'", _iss())
    assert users is not None and tuple(users[1].split(",")) == wizard.USER_FOLDERS
    assert {"runs", "my-settings", "packs", "replays"} <= set(wizard.USER_FOLDERS)
    assert not set(wizard.USER_FOLDERS) & deleted, "the user's folders only when they say so"
    assert "MB_YESNO, IDYES) <> IDNO" in _iss(), "kept unless the answer is No (and when silent)"
    messages = dict(line.split("=", 1) for line in _section("Messages"))
    first = messages["ConfirmUninstall"]
    assert "%1" in first and "keep your bots" in first and not re.search(r"complete|all of its", first), (
        "the first question promises no full removal before the keep-my-bots question"
    )
    # A No uninstall kept nothing, so the last message names no kept bots; kept ones get their own message.
    assert "kept" not in messages["UninstalledAll"]
    assert "(CurUninstallStep = usDone) and KeepUserFiles" in _iss()
    assert "Your bots and settings are still in" in _iss()
    remove = " ".join(_section("UninstallRun"))
    assert '{app}\\tools\\app_shortcuts.ps1"" -Remove' in remove
    named = [line for line in _iss().splitlines() if "RLBot5" in line]
    assert named == ["  RLBotFolder = 'RLBot5\\bin\\';"], "RLBot's folder: only looked for, never touched"
    assert "FileExists(AddBackslash(Dir) + 'Install NitroLearn.cmd')" in _iss(), "only into its own folder"


def test_a_folder_an_uninstall_kept_bots_in_takes_nitrolearn_again() -> None:
    """Uninstalling with "keep my bots" leaves the user's folders and a note about them; installing into that
    folder again is accepted with no question, while any other folder with files in it is refused."""
    code = _iss()[_iss().index("[Code]") :]
    assert _setup()["DirExistsWarning"] == "no", "the script's own check is the only question"
    note = re.search(r"KeptNote = '([^']+)';", code)
    assert note is not None and note[1].endswith(".txt")
    assert _section("InstallDelete") == [f'Type: files; Name: "{{app}}\\{note[1]}"'], "gone once reinstalled"
    check = code[code.index("function NextButtonClick") : code.index("{ ---------- the Components page")]
    ours = "not FileExists(AddBackslash(Dir) + 'Install NitroLearn.cmd')"
    assert f"DirExists(Dir) and not HoldsOnlyKeptFiles(Dir) and {ours}" in check
    kept = code[code.index("function HoldsOnlyKeptFiles") : code.index("function NextButtonClick")]
    assert "Result := IsUserFolder(Found.Name)" in kept and "CompareText(Found.Name, KeptNote) = 0" in kept
    assert "Pos(',' + Lowercase(Name) + ',', ',' + UserFolders + ',')" in code, "any letter case"
    assert "(Found.Attributes and DirectoryAttribute) <> 0" in kept and "DirectoryAttribute = $10;" in code
    uninstall = code[code.index("procedure CurUninstallStepChanged") :]
    assert "if KeepUserFiles then\n      WriteKeptNote(App)" in uninstall
    writer = code[code.index("procedure WriteKeptNote") : code.index("procedure CurUninstallStepChanged")]
    assert "AddBackslash(App) + KeptNote" in writer
    assert "if Kept <> '' then" in writer, "only beside kept folders"


def test_the_finish_page_launches_nitrolearn_only_after_a_good_install() -> None:
    run = " ".join(_section("Run"))
    assert 'Description: "Launch NitroLearn"' in run and "postinstall" in run and "unchecked" not in run
    assert 'Filename: "{app}\\.venv\\Scripts\\pythonw.exe"' in run and "Check: InstallSucceeded" in run
    assert 'Parameters: "-m nitro.dashboard.winapp open"' in run, "windowless, as the Start menu shortcut"
    assert "{cmd}" not in run and "runminimized" not in run, "no console window behind the app"
    messages = dict(line.split("=", 1) for line in _section("Messages"))
    for key in ("FinishedLabel", "FinishedLabelNoIcons"):
        assert "keep your bots" in messages[key]
    assert "[name/ver]" in messages["WelcomeLabel2"]
    assert "Click Next" not in messages["WelcomeLabel2"], "Inno Setup adds its own ClickNext line"
    assert "(like D:)" in messages["DiskSpaceGBLabel"]


def test_the_space_needed_is_said_the_same_everywhere() -> None:
    """About 10 GB on disk with the recommended parts (a real install's folder: 9.6 GB) and about 2 GB with
    PyTorch for the processor, said alike by the setup's pages, its component sizes, the installer's menu and
    the README; downloads about 3.5 GB."""
    messages = dict(line.split("=", 1) for line in _section("Messages"))
    for key in ("WelcomeLabel2", "DiskSpaceGBLabel", "DiskSpaceMBLabel"):
        assert "about 10 GB" in messages[key] and "about 2 GB" in messages[key], key
        assert "[gb]" not in messages[key] and "[mb]" not in messages[key], "Inno Setup's figure: fixed parts"
    assert "about 3.5 GB" in messages["WelcomeLabel2"]
    sizes = {name: int(m[1]) if (m := re.search(r"ExtraDiskSpaceRequired: (\d+)", line)) else 0
             for name, line in _components().items()}  # fmt: skip
    gb = 1024**3
    recommended = sum(size for name, size in sizes.items() if "recommended" in _components()[name])
    assert 9 * gb <= recommended <= 10 * gb and 1.5 * gb <= sizes["main"] <= 2.5 * gb
    assert sizes["rlbot"] == 0, "RLBot installs itself outside the folder"
    assert sizes["rust"] == int(1.5 * gb) and sizes["recordings"] == 119 * 1024**2, "as the menu says"
    assert "200 MB" not in _iss() and "3 GB" not in _iss()
    found = steps.Found(True, True, "NVIDIA GeForce RTX 3070", "", True, False, True, {})
    menu = steps.part_text(found, {}, "gpu")
    assert "(9 GB on disk; off: for the processor, 1.5 GB)" in menu
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    assert menu.replace(found.card, "<your NVIDIA card>") in readme, "the README shows the menu's line"
    assert "About 10 GB of free disk space (about 2 GB without an NVIDIA card)" in readme
    assert "3.5 GB to download and 10 GB on disk" in readme and "about 9 GB or 1.5 GB" in readme


def test_the_licence_page_puts_fair_play_and_the_short_terms_first(tmp_path: Path) -> None:
    rtf = wizard.licence_rtf()
    assert rtf.isascii() and rtf.startswith(r"{\rtf1") and rtf.endswith("}")
    opened = len(re.findall(r"(?<!\\){", rtf))
    assert opened == len(re.findall(r"(?<!\\)}", rtf))
    order = [
        "Fair play: NitroLearn is not a cheating tool",
        "Never use a NitroLearn bot, or anything made with NitroLearn, in online matches",
        "Licence: free for non-commercial use",
        "Your bots are yours",
        "PolyForm Noncommercial License 1.0.0",
        "Additional permission",
    ]
    places = [rtf.index(text) for text in order]
    assert places == sorted(places)
    assert "**" not in rtf and "](" not in rtf


def test_uv_comes_from_its_pinned_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = tmp_path / "uv.whl"
    with zipfile.ZipFile(wheel, "w") as out:
        out.writestr("uv-0.11.32.data/scripts/uv.exe", b"uv")
        out.writestr("uv-0.11.32.data/scripts/uvx.exe", b"uvx")
        out.writestr("uv-0.11.32.dist-info/licenses/LICENSE-MIT", "MIT")
        out.writestr("uv-0.11.32.dist-info/licenses/LICENSE-APACHE", "Apache")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    real = wizard._bundles()
    pinned = SimpleNamespace(UV={"windows-x64": SimpleNamespace(url="u", asset="a", sha256=sha)},
                             sha256_file=real.sha256_file, download=real.download)  # fmt: skip
    monkeypatch.setattr(wizard, "_bundles", lambda: pinned)
    wizard.uv_files(tmp_path / "uv", wheel)
    assert sorted(p.name for p in (tmp_path / "uv").iterdir()) == ["LICENSE-APACHE", "LICENSE-MIT", "uv.exe"]
    pinned.UV["windows-x64"].sha256 = "0" * 64
    with pytest.raises(SystemExit, match="not the pinned uv wheel"):
        wizard.uv_files(tmp_path / "other", wheel)


def test_the_pictures_follow_the_logo() -> None:
    numbers = images.logo_numbers(
        (PROJECT / "python" / "nitro" / "brand" / "logo.svg").read_text(encoding="utf-8")
    )
    assert numbers["back"] == "#141413" and numbers["gauge"] == "#FFD60A" and numbers["track"] == "#4D4A44"
    assert numbers["stroke"] == 6 and len(numbers["needle"]) == 6 and numbers["hub"][:2] == (24, 27)


def test_no_nvidia_card_unticks_and_greys_the_graphics_card_part() -> None:
    """Without an NVIDIA card the part cannot be had: its line says so, unticked and greyed (so its size is
    not counted), also after another type is chosen, as RLBot v5 found is."""
    code = _iss()[_iss().index("[Code]") :]
    settle = code[code.index("procedure SettleFoundParts") : code.index("procedure ShowCard")]
    assert "if NoCard then\n    SettleItem(GpuItem, 'PyTorch for the graphics card: no NVIDIA card" in settle
    assert "NoCard := Card = '';" in code


def test_a_running_nitrolearn_is_stopped_first_and_leftovers_are_said() -> None:
    """Setup and the uninstaller look for programs running from NitroLearn's folder (a training run goes on
    in the background after the window closes) and ask to stop them first; an uninstall that could not
    delete NitroLearn's folders says so instead of only reporting success."""
    code = _iss()[_iss().index("[Code]") :]
    running = code[code.index("function RunningFrom") : code.index("function NitroLearnStopped")]
    assert "SELECT Name, ExecutablePath FROM Win32_Process" in running
    assert "Copy(Path, 1, Length(Prefix)) = Prefix" in running, "only programs inside the folder"
    assert "Pos('unins', Lowercase(Name)) <> 1" in running, "the uninstaller itself does not count"
    prepare = code[code.index("function PrepareToInstall") : code.index("function InitializeUninstall")]
    assert "NitroLearnStopped(WizardDirValue, 'installing')" in prepare
    assert "Result := NitroLearnStopped(ExpandConstant('{app}'), 'uninstalling');" in code
    stopped = code[code.index("function NitroLearnStopped") : code.index("function PrepareToInstall")]
    assert "MB_RETRYCANCEL" in stopped and "Stop and save" in stopped
    uninstall = code[code.index("procedure CurUninstallStepChanged") :]
    assert "Left := LeftBehind(App);" in uninstall and "Restart the PC" in uninstall


def test_the_finish_page_and_messages_never_promise_what_was_not_made() -> None:
    """No Start menu shortcut made (not chosen, or it failed): the Finished page names the folder's own and
    how to open the hidden folder; a stopped install says so. Messages give the folder's address to paste
    into Run (Windows hides AppData in its file windows) and the checklist says how long it takes."""
    code = _iss()[_iss().index("[Code]") :]
    finished = code[code.index("procedure CurPageChanged") : code.index("procedure CurStepChanged")]
    assert "(StepState('shortcuts') <> 'done')" in finished and "OpenFolderHint" in finished
    assert "not InstallSucceeded" in finished
    hint = code[code.index("function OpenFolderHint") : code.index("function StepState")]
    assert "Windows key and R" in hint and "ExpandConstant('{app}')" in hint
    run = code[code.index("procedure RunInstallSteps") : code.index("function InstallSucceeded")]
    assert run.count("OpenFolderHint") == 2 and "Shortcuts folder" not in run
    assert "about 20 minutes" in code[code.index("procedure ShowSteps") :]
