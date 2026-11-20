# Makes the "NitroLearn" shortcuts that open the dashboard: in the Start menu (Programs) and, with -Desktop,
# on the desktop, for this user only (no administrator rights, nothing in the registry). They start it with
# pythonw.exe (python -m nitro.dashboard.winapp open), so no console window opens, with the settings
# "Shortcuts\Open the dashboard.cmd" uses (tools, caches and temporary files inside the folder) and the app
# icon, and they carry the dashboard's AppUserModelID so the open window and a pinned taskbar icon are one.
#
# Each shortcut is recorded in .tools\shortcuts.json before it is made; -Remove deletes exactly those and
# the record (uninstalling: "Uninstall NitroLearn.cmd"). Running it again replaces its own shortcuts. A
# shortcut named NitroLearn that opens something else (made or changed by the user) is never replaced or
# deleted, and is not recorded; one left by this folder's earlier place (moved) is NitroLearn's.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\app_shortcuts.ps1 [-Desktop] [-Remove]
param([switch]$Desktop, [switch]$Remove)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Windowless = Join-Path $Root '.venv\Scripts\pythonw.exe'
$Record = Join-Path $Root '.tools\shortcuts.json'
$AppId = 'NitroLearn.Dashboard'  # the same as nitro.dashboard.winapp.APP_ID
$env:PYTHONIOENCODING = 'utf-8'  # paths with any letters print without an error
$Name = 'NitroLearn'
$OpenArguments = '-m nitro.dashboard.winapp open'  # the same as nitro.dashboard.winapp.OPEN_ARGUMENTS
$LeaveAlone = 3  # nitro.dashboard.winapp.LEAVE_ALONE

function Invoke-Claim([string]$Link, [bool]$OnDesktop) {
    # Whether to make $Link: nitro.dashboard.winapp decides from what Find-Links read (so the dashboard's
    # tests cover it) and records it when it is to be made.
    $more = @()
    if ($OnDesktop) { $more += '--desktop' }
    & $Python -m nitro.dashboard.winapp shortcuts --root $Root --claim $Link @more | Out-Host
    if ($LASTEXITCODE -eq 0) { return $true }
    if ($LASTEXITCODE -eq $LeaveAlone) { return $false }
    throw "Could not update $Record."
}

function Get-Recorded {
    if (-not (Test-Path -LiteralPath $Record)) { return @() }
    return @((Get-Content -Raw -Encoding UTF8 -LiteralPath $Record | ConvertFrom-Json).shortcuts | ForEach-Object { [string]$_ })
}

function Find-Links([string[]]$Links) {
    # What each shortcut in $Links that is there opens, as JSON for nitro.dashboard.winapp: {path: {target,
    # arguments}}. One that cannot be read is left out (it is treated as NitroLearn's, as before).
    $found = @{}
    try { Initialize-ShellLink } catch { return '{}' }
    foreach ($link in $Links) {
        if (-not $link -or -not $link.ToLower().EndsWith('.lnk') -or -not (Test-Path -LiteralPath $link -PathType Leaf)) { continue }
        try {
            $read = [NitroLearn.Shortcut]::Read($link)
            $found[$link] = @{ target = $read[0]; arguments = $read[1] }
        } catch { }
    }
    return (ConvertTo-Json -Compress -InputObject $found)
}

function Test-Elsewhere([string]$Link) {
    # Without NitroLearn's Python: whether $Link opens something other than this NitroLearn.
    try {
        Initialize-ShellLink
        $read = [NitroLearn.Shortcut]::Read($Link)
        if (-not $read[0]) { return $true }
        # nitro.keeper's start-up entry (run.auto_restart): nitro.cmd carry-on, from here or a moved folder.
        if ((Split-Path -Leaf $Link).StartsWith('NitroLearn carry on training') -and ($read[1] -split '\s+') -contains 'carry-on' -and ((-not (Test-Path -LiteralPath $read[0])) -or [IO.Path]::GetFullPath($read[0]).StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase))) { return $false }
        $same = [IO.Path]::GetFullPath($read[0]) -eq [IO.Path]::GetFullPath($Windowless)
        # NitroLearn's own from a folder that is not there any more (moved): as nitro.dashboard.winapp decides.
        if (-not $same -and $read[0].ToLower().EndsWith('\.venv\scripts\pythonw.exe') -and -not (Test-Path -LiteralPath $read[0])) { $same = $true }
    } catch { return $false }
    return -not ($same -and (($read[1] -split '\s+' | Where-Object { $_ }) -join ' ') -eq $OpenArguments)
}

function Remove-Recorded {
    # Deletes the recorded shortcuts and their record; returns $false when one could not be deleted.
    if (Test-Path -LiteralPath $Python) {
        & $Python -m nitro.dashboard.winapp shortcuts --root $Root --remove | Out-Host  # shown, not returned
        return ($LASTEXITCODE -eq 0)
    }
    # NitroLearn's Python is gone already: delete the recorded shortcuts here.
    if (-not (Test-Path -LiteralPath $Record)) { return $true }
    $kept = @()
    foreach ($path in Get-Recorded) {
        if (-not ([string]$path).ToLower().EndsWith('.lnk')) { continue }
        try {
            if ((Test-Path -LiteralPath $path) -and (Test-Elsewhere $path)) {
                Write-Host "Left $path in place: it was changed to open something else."
            } elseif (Test-Path -LiteralPath $path) {
                Remove-Item -LiteralPath $path -Force
                Write-Host "Removed $path"
            }
        } catch {
            Write-Host "Could not remove ${path}: delete it by hand." -ForegroundColor Yellow
            $kept += $path
        }
    }
    if ($kept.Count -eq 0) { Remove-Item -LiteralPath $Record -Force }
    return ($kept.Count -eq 0)
}

function Test-Plain([string]$Path) {
    # Only letters the C# compiler's file handling keeps: plain ASCII.
    return ($Path -and ($Path -notmatch '[^\x20-\x7E]'))
}

function Get-PlainTemp {
    # A temporary folder whose path is plain ASCII: TEMP itself, else its short (8.3) name, else the user's
    # own TEMP or its short name, else ProgramData (whose path is plain ASCII on every Windows).
    $fso = New-Object -ComObject Scripting.FileSystemObject
    foreach ($dir in @($env:TEMP, [Environment]::GetEnvironmentVariable('TEMP', 'User'))) {
        if (-not $dir) { continue }
        $dir = [Environment]::ExpandEnvironmentVariables($dir)
        if (Test-Plain $dir) { return $dir }
        try {
            $short = $fso.GetFolder($dir).ShortPath
            if (Test-Plain $short) { return $short }
        } catch { }
    }
    $fallback = Join-Path $env:ProgramData 'NitroLearn-temp'
    New-Item -ItemType Directory -Force -Path $fallback | Out-Null
    return $fallback
}

function Initialize-ShellLink {
    # Shortcuts are made and read through the shell's IShellLinkW, which keeps every letter of the paths.
    # WScript.Shell's shortcut object passes them through the ANSI code page, which turns letters it lacks
    # into look-alikes (a Polish L with a stroke into a plain L), so the shortcut pointed at a folder that
    # does not exist. The AppUserModelID is a property of the .lnk file, set through the same object's
    # IPropertyStore.
    if (-not ('NitroLearn.Shortcut' -as [type])) {
        # Add-Type compiles the C# below into TEMP, and the compiler cannot write to a folder whose path has
        # letters outside the ANSI code page (NitroLearn's .tmp in such a folder): it compiles in a plain-
        # letter temporary folder instead, and TEMP is put back afterwards.
        $saved = @($env:TEMP, $env:TMP)
        $env:TEMP = Get-PlainTemp
        $env:TMP = $env:TEMP
        try {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace NitroLearn {
    [ComImport, Guid("000214F9-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IShellLinkW {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder file, int size, IntPtr data, uint flags);
        void GetIDList(out IntPtr list);
        void SetIDList(IntPtr list);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder name, int size);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder dir, int size);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string dir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder args, int size);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string args);
        void GetHotkey(out short key);
        void SetHotkey(short key);
        void GetShowCmd(out int show);
        void SetShowCmd(int show);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int size, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr window, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string file);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PropertyKey { public Guid FormatId; public uint PropertyId; }

    // PROPVARIANT holding a string: 16 bytes on 32-bit Windows, 24 on 64-bit, the string at offset 8.
    [StructLayout(LayoutKind.Sequential)]
    public struct PropVariant {
        public ushort Type; public ushort Reserved1; public ushort Reserved2; public ushort Reserved3;
        public IntPtr Value; public IntPtr Unused;
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int GetAt(uint index, IntPtr key);
        [PreserveSig] int GetValue(ref PropertyKey key, IntPtr value);
        [PreserveSig] int SetValue(ref PropertyKey key, ref PropVariant value);
        [PreserveSig] int Commit();
    }

    public static class Shortcut {
        public static string Make(string path, string target, string arguments, string folder, string icon,
                                  string description, string appId) {
            object link = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("00021401-0000-0000-C000-000000000046")));
            try {
                IShellLinkW shell = (IShellLinkW)link;
                shell.SetPath(target);
                shell.SetArguments(arguments);
                shell.SetWorkingDirectory(folder);
                shell.SetIconLocation(icon, 0);
                shell.SetDescription(description);
                string problem = "";
                try {
                    SetAppId((IPropertyStore)link, appId);
                } catch (Exception error) {
                    problem = error.Message;
                }
                ((IPersistFile)link).Save(path, true);
                return problem;
            } finally {
                Marshal.ReleaseComObject(link);
            }
        }

        // What the shortcut at `path` opens: its target and its arguments.
        public static string[] Read(string path) {
            object link = Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("00021401-0000-0000-C000-000000000046")));
            try {
                ((IPersistFile)link).Load(path, 0);
                IShellLinkW shell = (IShellLinkW)link;
                StringBuilder target = new StringBuilder(32768);
                shell.GetPath(target, target.Capacity, IntPtr.Zero, 4);  // SLGP_RAWPATH: the path as it was set
                StringBuilder arguments = new StringBuilder(32768);
                shell.GetArguments(arguments, arguments.Capacity);
                return new string[] { target.ToString(), arguments.ToString() };
            } finally {
                Marshal.ReleaseComObject(link);
            }
        }

        static void SetAppId(IPropertyStore store, string appId) {
            PropertyKey key = new PropertyKey();
            key.FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");  // System.AppUserModel.ID
            key.PropertyId = 5;
            PropVariant value = new PropVariant();
            value.Type = 31;  // VT_LPWSTR
            value.Value = Marshal.StringToCoTaskMemUni(appId);
            try {
                Marshal.ThrowExceptionForHR(store.SetValue(ref key, ref value));
                Marshal.ThrowExceptionForHR(store.Commit());
            } finally {
                Marshal.FreeCoTaskMem(value.Value);
            }
        }
    }
}
'@
        } finally {
            if ($env:TEMP -eq (Join-Path $env:ProgramData 'NitroLearn-temp')) {
                Remove-Item -LiteralPath $env:TEMP -Recurse -Force -ErrorAction SilentlyContinue  # nothing left behind
            }
            $env:TEMP, $env:TMP = $saved
        }
    }
}

function New-Shortcut([string]$Link) {
    # Returns '' or why the AppUserModelID was not set.
    Initialize-ShellLink
    # pythonw.exe as the target: a program without a console window, which the taskbar can pin.
    $about = 'NitroLearn dashboard: make, train, watch and export Rocket League bots'
    return [NitroLearn.Shortcut]::Make($Link, $Windowless, $OpenArguments, $Root, $icon, $about, $AppId)
}

if ($Remove) {
    if (Test-Path -LiteralPath $Python) { $env:NITRO_SHORTCUTS_FOUND = Find-Links (Get-Recorded) }
    if (Remove-Recorded) { exit 0 }
    exit 1
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host 'NitroLearn is not installed yet: double-click "Install NitroLearn.cmd" in its folder.' -ForegroundColor Yellow
    exit 1
}
. (Join-Path $Root 'tools\env.ps1')

$places = @([Environment]::GetFolderPath('Programs'))
if ($Desktop) { $places += [Environment]::GetFolderPath('Desktop') }
$icon = Join-Path $Root 'python\nitro\brand\icon.ico'

$links = @($places | Where-Object { $_ } | ForEach-Object { Join-Path $_ "$Name.lnk" })
$env:NITRO_SHORTCUTS_FOUND = Find-Links (@(Get-Recorded) + $links)  # nitro.dashboard.winapp.FOUND_VARIABLE
$null = Remove-Recorded  # the shortcuts an earlier run made; the ones asked for now are made again
for ($i = 0; $i -lt $places.Count; $i++) {
    if (-not $places[$i]) { continue }
    $link = Join-Path $places[$i] "$Name.lnk"
    if (-not (Invoke-Claim $link ($i -gt 0))) {
        Write-Host "Kept $link as it is: a shortcut of that name that opens something else is already there." -ForegroundColor Yellow
        continue
    }
    $problem = New-Shortcut $link
    if ($problem) {
        Write-Host "The shortcut works, but Windows may show the open dashboard as a separate taskbar icon ($problem)." -ForegroundColor Yellow
    }
    Write-Host "Made $link"
}
exit 0
