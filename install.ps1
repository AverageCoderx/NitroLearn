<#
.SYNOPSIS
  Installs NitroLearn on Windows, entirely inside this folder. Double-click "Install NitroLearn.cmd" to run it.

.DESCRIPTION
  It downloads uv (a pinned version, checked by its SHA-256 hash) into .tools\uv, lets uv put its own
  Python 3.11 in .tools\python and make the private environment .venv, then tools\install_steps.py installs
  PyTorch for this PC's graphics card, NitroLearn (ready-made for this version when the release has it, else
  built from its code with Rust kept in .tools), the arena meshes, and runs "nitro doctor".
  Nothing is installed system-wide: no change to an existing Python, the PATH or the registry.
  To remove NitroLearn, delete this folder. Safe to run again (for example after updating).

  uv and Python come from NitroLearn's own release first (a folder of the release's files, else the release
  on GitHub), and only then from PyPI and python-build-standalone; each is checked against its pinned hash
  wherever it comes from (the same pins as tools\bundles.py).

  Before downloading, it shows a menu of the optional parts (graphics-card PyTorch, RLBot v5, the Start menu
  and desktop shortcuts, Rust for your own Rust rewards, developer tools, RocketSim's recordings): type a
  number to switch one on or off, Enter to install. Run again, it shows what is installed ticked and changes
  only what was switched. The switches below set parts without the menu; without a console to answer in,
  nothing is asked and the parts stay as they are (first install: no shortcuts, no RLBot).

.PARAMETER Torch
  A PyTorch build other than the one picked for the card: cpu, cu126, cu128 or cu130 (remembered).

.PARAMETER CPU
  PyTorch for the processor, not the graphics card (as -Torch cpu).

.PARAMETER Rust
  Also put Rust in this folder (needed only for your own Rust rewards).

.PARAMETER FromSource
  Build NitroLearn from its code even when ready-made files exist (needs Rust and Microsoft's C++ build tools).

.PARAMETER Dev
  Developer install: tests and code checkers (builds from the code).

.PARAMETER Recordings
  Download RocketSim's five game recordings now (119 MB) rather than the first time a feature needs them.

.PARAMETER NoLaunch
  Do not open the dashboard at the end.

.PARAMETER RLBot
  Also install RLBot v5 (needed to play bots in Rocket League): its official installer is downloaded from
  RLBot's own releases and opened, and RLBot installs itself in %LOCALAPPDATA%\RLBot5. It is a separate
  program with its own licence, removed from Windows' Apps list, not with NitroLearn. RLBot v4 (an older
  RLBot) does not run NitroLearn's bots and can stay installed next to it.

.PARAMETER NoRLBot
  Do not install RLBot v5.

.PARAMETER Desktop
  Shortcuts in the Start menu and on the desktop.

.PARAMETER NoShortcuts
  No shortcuts (removes the ones an earlier install made).

.PARAMETER Parts
  Exactly these optional parts, comma-separated, the rest off: gpu, rlbot, start_menu, desktop, rust, dev,
  recordings. NitroLearn-Setup.exe passes its Components page so.

.PARAMETER Bundles
  A folder of the release's files (default: release-files in this folder, when there is one): install from it
  without downloading, except PyTorch and the arena meshes.

.PARAMETER Progress
  Also print a line per step (for NitroLearn-Setup.exe's checklist).
#>
param(
    [ValidateSet('auto', 'cpu', 'cu126', 'cu128', 'cu130')][string]$Torch = '',
    [switch]$Rust,
    [switch]$FromSource,
    [switch]$Dev,
    [switch]$CPU,
    [switch]$Recordings,
    [switch]$NoLaunch,
    [switch]$RLBot,
    [switch]$NoRLBot,
    [switch]$Desktop,
    [switch]$NoShortcuts,
    [AllowEmptyString()][string]$Parts,
    [string]$Bundles = '',
    [switch]$Progress
)

# Programs are checked through their exit codes: "Stop" would make Windows PowerShell 5.1 abort whenever a
# program merely writes to its error output, so cmdlets that must not fail say -ErrorAction Stop themselves.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'  # Windows PowerShell's progress bar makes downloads many times slower
Set-Location $PSScriptRoot
# PyPI needs TLS 1.2, which Windows PowerShell 5.1 does not always offer by default.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Every tool, cache and temporary file stays inside this folder; see tools\env.ps1.
. (Join-Path $PSScriptRoot 'tools\env.ps1')

$UvVersion = '0.11.32'
$UvWheel = 'uv-0.11.32-py3-none-win_amd64.whl'
$UvUrl = "https://files.pythonhosted.org/packages/57/60/a645cca710448004268b7731c4dc6fc7d084a7c620fc73ecc2127cb61661/$UvWheel"
$UvSha256 = '855632ee1d2a8491986efa54c562b806cd32477761889a1e13624b519d98a45e'
$PythonVersion = '3.11'
$PythonAsset = 'python-3.11.15-windows-x64.tar.gz'
$PythonSha256 = 'a48c2dbe832319f61aa8557c9900caec70f7fed0cbee391a4c9ff9f98b50222d'
$PythonMirrorPath = '20260718/cpython-3.11.15+20260718-x86_64-pc-windows-msvc-install_only_stripped.tar.gz'
$Uv = Join-Path $PSScriptRoot '.tools\uv\uv.exe'
$Releases = if ($env:NITRO_RELEASES) { $env:NITRO_RELEASES } else { 'https://github.com/AverageCoderx/NitroLearn/releases/download/v{version}/' }

function Step([string]$Text) { Write-Host "`n==> $Text" -ForegroundColor Cyan }
function StepState([string]$Key, [string]$State) { if ($Progress) { Write-Host "@@nitro $Key $State" } }
function Fail([string]$Text) { Write-Host "`nSETUP STOPPED: $Text" -ForegroundColor Red; exit 1 }
# Said whenever a download fails or stalls (tools\install_steps.py says the same).
$DownloadHint = 'Check the internet connection. Antivirus web shields (HTTPS scanning) and company proxies often hold back or block downloads like these: allow this folder in your antivirus (or pause its web shield for the install), then run the installer again. What was already downloaded is kept.'

function Get-FreeGB {
    # Free space on this folder's drive in GB, or -1 when Windows cannot say (a network folder).
    try { return (New-Object System.IO.DriveInfo ([IO.Path]::GetPathRoot($PSScriptRoot))).AvailableFreeSpace / 1GB } catch { return -1 }
}
function Get-FailureHint {
    # A full disk, when there is almost no room left; otherwise what usually holds downloads back.
    $free = Get-FreeGB
    if ($free -ge 0 -and $free -lt 1) {
        return "The disk with NitroLearn's folder is full. Free up space (NitroLearn needs about 10 GB with PyTorch for the graphics card, about 2 GB without), or install NitroLearn on a drive with more room, then run the installer again. What was already downloaded is kept."
    }
    return $DownloadHint
}

function Invoke-Uv([int]$Minutes, [string]$What, [string[]]$Arguments) {
    # uv with its output in this window, nothing to read on its input, and a time limit: a step that has
    # not finished by then is stopped and the setup says why instead of waiting on.
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $Uv
    $start.Arguments = ($Arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $start.WorkingDirectory = $PSScriptRoot
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = $true
    $process = [System.Diagnostics.Process]::Start($start)
    $process.StandardInput.Close()
    if (-not $process.WaitForExit($Minutes * 60 * 1000)) {
        try { $process.Kill() } catch {}
        Fail "$What was stopped after $Minutes minutes without finishing. $(Get-FailureHint)"
    }
    if ($process.ExitCode -ne 0) { Fail "$What failed (see above). $(Get-FailureHint)" }
}

if (-not [Environment]::Is64BitOperatingSystem) { Fail 'NitroLearn needs 64-bit Windows.' }
# Places that break an install: a folder OneDrive syncs (it uploads the gigabytes NitroLearn writes, and can
# lock files or move them to the cloud while they are in use) and a very long path (Windows' 260-character
# limit, deep inside .venv). Said before anything is downloaded; a person can still go on.
$warned = $false
foreach ($synced in @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
    if ($synced -and "$PSScriptRoot\".StartsWith($synced.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "`nThis folder is inside OneDrive ($synced). OneDrive uploads the several GB NitroLearn writes and can lock or move files while they are in use, which breaks the install and training. Move the NitroLearn folder out of OneDrive (for example to C:\NitroLearn) and run the installer there." -ForegroundColor Yellow
        $warned = $true
        break
    }
}
if ($PSScriptRoot.Length -gt 80) {
    Write-Host "`nThis folder's path is long ($($PSScriptRoot.Length) characters). Windows can fail on very long file paths deep inside the install: move the folder somewhere short, like C:\NitroLearn, and run the installer there." -ForegroundColor Yellow
    $warned = $true
}
# Asked only of a person at the console: not of the setup wizard or an update (they pass -Parts).
if ($warned -and -not $PSBoundParameters.ContainsKey('Parts') -and -not [Console]::IsInputRedirected) {
    Read-Host 'Press Enter to install here anyway, or close this window to move the folder first' | Out-Null
}
# Room for uv and Python; tools\install_steps.py checks the room for the rest once the parts are chosen.
$free = Get-FreeGB
if ($free -ge 0 -and $free -lt 1) {
    Fail "Not enough free disk space: $([Math]::Round($free, 1)) GB free on $([IO.Path]::GetPathRoot($PSScriptRoot)). NitroLearn needs about 10 GB with PyTorch for the graphics card (about 2 GB without): free up space, or put NitroLearn on a drive with more room, then run the installer again."
}
if ($Progress) {
    # Read line by line by the setup wizard, which reads UTF-8. Its checklist shows the progress, so uv's
    # progress bars (a line per redraw in a classic console) are off.
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:UV_NO_PROGRESS = '1'
}

# The release of this folder's version (Cargo.toml), and a folder of its files when there is one.
$Version = ''
$found = Select-String -Path (Join-Path $PSScriptRoot 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($found) { $Version = $found.Matches[0].Groups[1].Value }
$ReleaseUrl = $Releases.Replace('{version}', $Version)
$ReleaseFiles = ''
if ($Bundles) {
    # A relative folder is taken from this folder (the script starts there).
    $ReleaseFiles = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Bundles)
} elseif (Test-Path (Join-Path $PSScriptRoot 'release-files') -PathType Container) {
    $ReleaseFiles = Join-Path $PSScriptRoot 'release-files'
}

function Get-Pinned([string]$Name, [string]$Sha256, [string]$File, [string]$Original) {
    # The release's copy of $Name (a folder of the release's files, else the release), else $Original;
    # kept only when it matches $Sha256. 'ok', 'missing', or 'mismatch' (what was found did not match).
    $result = 'missing'
    if ($ReleaseFiles) {
        $local = Join-Path $ReleaseFiles $Name
        if (Test-Path $local -PathType Leaf) {
            if ((Get-FileHash $local -Algorithm SHA256).Hash.ToLower() -eq $Sha256) {
                Copy-Item $local $File -Force
                return 'ok'
            }
            $result = 'mismatch'
        }
    }
    foreach ($from in @(($ReleaseUrl + $Name), $Original)) {
        if (-not $from) { continue }
        try {
            Invoke-WebRequest $from -OutFile $File -UseBasicParsing -TimeoutSec 600 -ErrorAction Stop
        } catch {
            continue
        }
        if ((Get-FileHash $File -Algorithm SHA256).Hash.ToLower() -eq $Sha256) { return 'ok' }
        Remove-Item $File -Force
        $result = 'mismatch'
    }
    return $result
}

# uv in .tools\uv: NitroLearn's release's copy of its package, else the package on PyPI (never Astral's
# install script, which edits the PATH).
$haveUv = $false
if (Test-Path $Uv) {
    $said = (& $Uv --version 2>$null) -join ' '
    $haveUv = $said -like "uv $UvVersion*"
}
if (-not $haveUv) {
    Step "Getting uv $UvVersion (28 MB)"
    New-Item -ItemType Directory -Force $env:TEMP -ErrorAction Stop | Out-Null
    $wheel = Join-Path $env:TEMP $UvWheel
    $got = Get-Pinned $UvWheel $UvSha256 $wheel $UvUrl
    if ($got -eq 'mismatch') { Fail 'The uv download did not match its SHA-256 hash: nothing from it was used. Try again later.' }
    if ($got -ne 'ok') { Fail "uv could not be downloaded from PyPI. $(Get-FailureHint)" }
    New-Item -ItemType Directory -Force (Split-Path $Uv -Parent) -ErrorAction Stop | Out-Null
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($wheel)
    try {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq "uv-$UvVersion.data/scripts/uv.exe" }
        if (-not $entry) { Fail 'The uv package does not hold uv.exe.' }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Uv, $true)
    } finally {
        $zip.Dispose()
    }
    Remove-Item $wheel -Force
}

StepState 'python' 'start'
Step "Python $PythonVersion (kept in .tools\python)"
# Python from NitroLearn's release's copy when uv has to download it: uv reads it from a folder laid out as
# python-build-standalone's releases, and checks it against the hash built into uv. Without a copy, uv
# downloads it from python-build-standalone itself.
$mirror = Join-Path $PSScriptRoot '.tmp\python-mirror'
if (-not (Test-Path (Join-Path $env:UV_PYTHON_INSTALL_DIR 'cpython-3.11*'))) {
    $copy = Join-Path $mirror ($PythonMirrorPath.Replace('/', '\'))
    New-Item -ItemType Directory -Force (Split-Path $copy -Parent) -ErrorAction Stop | Out-Null
    if ((Get-Pinned $PythonAsset $PythonSha256 $copy '') -eq 'ok') {
        $env:UV_PYTHON_INSTALL_MIRROR = ([Uri]$mirror).AbsoluteUri
    }
}
Invoke-Uv 15 "Installing Python $PythonVersion" @('python', 'install', $PythonVersion)
Remove-Item Env:\UV_PYTHON_INSTALL_MIRROR -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $mirror -ErrorAction SilentlyContinue

# A .venv that no longer runs (for example after the folder was moved), or that was made from a Python
# outside this folder, is made again.
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path (Join-Path $PSScriptRoot '.venv')) {
    $works = $false
    $config = Join-Path $PSScriptRoot '.venv\pyvenv.cfg'
    if ((Test-Path $venvPython) -and (Test-Path $config)) {
        # UTF-8, as uv writes it: Windows PowerShell would read it in the ANSI code page, which alters a folder
        # name with other letters, and the .venv would be made again on every run.
        $base = (Get-Content -Encoding UTF8 -LiteralPath $config | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1) -replace '^home\s*=\s*', ''
        & $venvPython -c 'import sys' 2>$null
        $works = ($LASTEXITCODE -eq 0) -and "$base".Trim().StartsWith($env:UV_PYTHON_INSTALL_DIR, [StringComparison]::OrdinalIgnoreCase)
    }
    if (-not $works) {
        Step 'The private environment .venv does not work here, or uses a Python outside this folder: making it again'
        Remove-Item -Recurse -Force (Join-Path $PSScriptRoot '.venv') -ErrorAction Stop
    }
}
if (-not (Test-Path $venvPython)) {
    Step 'Making the private environment .venv'
    Invoke-Uv 10 'Making .venv' @('venv', '--seed', '--python', $PythonVersion, '.venv')
}
StepState 'python' 'done'

$steps = @('-I', (Join-Path $PSScriptRoot 'tools\install_steps.py'), '--uv', $Uv)
if ($CPU) { $steps += '--cpu' } elseif ($Torch) { $steps += @('--torch', $Torch) }
if ($Rust) { $steps += '--rust' }
if ($FromSource) { $steps += '--from-source' }
if ($Dev) { $steps += '--dev' }
if ($Recordings) { $steps += '--recordings' }
if ($NoLaunch) { $steps += '--no-launch' }
if ($RLBot) { $steps += '--rlbot' }
if ($NoRLBot) { $steps += '--no-rlbot' }
if ($Desktop) { $steps += '--desktop' }
if ($NoShortcuts) { $steps += '--no-shortcuts' }
# --name=value: Windows PowerShell drops an empty argument ("no parts") when it starts a program.
if ($PSBoundParameters.ContainsKey('Parts')) { $steps += "--parts=$Parts" }
if ($ReleaseFiles) { $steps += "--bundles=$ReleaseFiles" }
if ($Progress) { $steps += '--progress' }
& $venvPython @steps
exit $LASTEXITCODE
