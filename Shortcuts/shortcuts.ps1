# The double-click shortcuts in this folder: each .cmd file runs this script with one
# action, asks its questions in plain English, then runs the `nitro` command for you.
# Your own settings files live in my-settings\ (never changed by updates); the presets stay as shipped.
#
# For tests: NITRO_SHORTCUTS_ANSWERS="1;my-run" answers the questions in order, and
# NITRO_SHORTCUTS_DRY_RUN=1 prints what would run or open instead of doing it.
param([Parameter(Mandatory = $true)][string]$Action)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$Runs = Join-Path $Root 'runs'
$Mine = Join-Path $Root 'my-settings'
$Presets = Join-Path $Root (Join-Path 'configs' 'presets')
$DryRun = $env:NITRO_SHORTCUTS_DRY_RUN -eq '1'
$script:Answers = $null
if ($null -ne $env:NITRO_SHORTCUTS_ANSWERS) { $script:Answers = [System.Collections.ArrayList]@($env:NITRO_SHORTCUTS_ANSWERS -split ';') }

function Ask([string]$Question) {
    if ($null -ne $script:Answers) {
        if ($script:Answers.Count -eq 0) { throw "no answer left for: $Question" }
        $answer = $script:Answers[0]
        $script:Answers.RemoveAt(0)
        Write-Host "$Question $answer"
        return $answer
    }
    return Read-Host $Question
}

function Stop-With([string]$Message) {
    Write-Host ''
    Write-Host $Message -ForegroundColor Yellow
    exit 1
}

function Invoke-Nitro([string[]]$Arguments) {
    # Called as a statement, never captured, so nitro writes straight to this window as it goes; its exit
    # code is left in $script:Code.
    $shown = ($Arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $script:Code = 0
    if ($DryRun) { Write-Host "would run: nitro $shown"; return }
    Write-Host ''
    Write-Host "Running: .\nitro $shown" -ForegroundColor DarkGray
    & (Join-Path $Root 'nitro.cmd') @Arguments
    $script:Code = $LASTEXITCODE
}

function Open-Thing([string]$Path) {
    if ($DryRun) { Write-Host "would open: $Path"; return }
    Invoke-Item $Path
}

function Choose([string]$Title, [object[]]$Items, [string]$Default = '') {
    # A numbered list; returns the chosen item (Enter picks $Default's number when given).
    Write-Host ''
    Write-Host $Title -ForegroundColor Cyan
    for ($i = 0; $i -lt $Items.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i + 1), $Items[$i].Label) }
    $hint = ''
    if ($Default) { $hint = " (Enter for $Default)" }
    while ($true) {
        $answer = (Ask "Type a number$hint and press Enter:").Trim()
        if ($answer -eq '' -and $Default) { $answer = $Default }
        $number = 0
        if ([int]::TryParse($answer, [ref]$number) -and $number -ge 1 -and $number -le $Items.Count) {
            return $Items[$number - 1]
        }
        Write-Host "Please type a number from 1 to $($Items.Count)."
        if ($null -ne $script:Answers -and $script:Answers.Count -eq 0) { throw 'no valid choice given' }
    }
}

function Pick-File([string]$Start) {
    # Windows' own file picker; a typed path where it is not available.
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.InitialDirectory = $Start
        $dialog.Filter = 'Settings files (*.toml)|*.toml'
        if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.FileName }
        Stop-With 'No file chosen.'
    } catch {
        return (Ask 'Type the full path of the settings file:').Trim('" ')
    }
}

function Pick-Settings([string]$Title) {
    $items = @()
    if (Test-Path $Mine) {
        foreach ($file in Get-ChildItem $Mine -Filter '*.toml' | Sort-Object LastWriteTime -Descending) {
            $items += [pscustomobject]@{ Label = "my-settings\$($file.Name)"; Path = $file.FullName }
        }
    }
    foreach ($file in Get-ChildItem $Presets -Filter '*.toml' | Sort-Object Name) {
        $items += [pscustomobject]@{ Label = "preset: $($file.BaseName) (as shipped)"; Path = $file.FullName }
    }
    $items += [pscustomobject]@{ Label = 'another settings file (pick it)'; Path = '' }
    $chosen = Choose $Title $items '1'
    if ($chosen.Path) { return $chosen.Path }
    return Pick-File $Root
}

function Pick-Run([string]$Title) {
    $found = @()
    if (Test-Path $Runs) {
        $found = @(Get-ChildItem $Runs -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'run.json') } |
            Sort-Object LastWriteTime -Descending)
    }
    if ($found.Count -eq 0) { Stop-With "There are no runs yet in $Runs. Start one with 'Train a new bot'." }
    $items = foreach ($run in $found) {
        $saves = @(Get-ChildItem (Join-Path $run.FullName 'checkpoints') -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notlike '*.partial' }).Count
        [pscustomobject]@{
            Label = "{0}   (last active {1:yyyy-MM-dd HH:mm}, {2} saves)" -f $run.Name, $run.LastWriteTime, $saves
            Name = $run.Name
        }
    }
    return (Choose $Title @($items) '1').Name
}

function Ask-Name([string]$Question, [string]$Default) {
    while ($true) {
        $name = (Ask "$Question (Enter for $Default):").Trim()
        if ($name -eq '') { $name = $Default }
        if ($name -match '^[A-Za-z0-9][A-Za-z0-9._-]*$') { return $name }
        Write-Host 'Use letters, numbers, dots, dashes and underscores only (no spaces).'
        if ($null -ne $script:Answers -and $script:Answers.Count -eq 0) { throw 'no valid name given' }
    }
}

function Edit-Settings([string]$Path) {
    Write-Host ''
    Write-Host "Opening $Path in Notepad. To turn a feature on, remove the # in front of its lines. Save and"
    Write-Host 'close Notepad when you are done; the file is checked for mistakes right after.'
    if ($DryRun) { Write-Host "would edit: $Path" } else { Start-Process notepad.exe -ArgumentList "`"$Path`"" -Wait }
    Invoke-Nitro @('check-settings', $Path)
}

function Open-Report-When-Ready([string]$Report) {
    # The progress page appears after the first few minutes of training; open it then, in the background.
    if ($DryRun) { Write-Host "would open when written: $Report"; return }
    [void](Start-Job -ArgumentList $Report -ScriptBlock {
        param($Path)
        for ($i = 0; $i -lt 360; $i++) {
            if (Test-Path $Path) { Start-Sleep -Seconds 2; Invoke-Item $Path; return }
            Start-Sleep -Seconds 5
        }
    })
}

switch ($Action) {
    'new-settings' {
        $preset = Choose 'Which game size should your settings start from?' @(
            foreach ($file in Get-ChildItem $Presets -Filter '*.toml' | Sort-Object Name) {
                [pscustomobject]@{ Label = $file.BaseName; Path = $file.FullName }
            }
        ) '1'
        $name = Ask-Name 'Name for your settings file' ("my-" + ($preset.Label -replace '_starter$', ''))
        $target = Join-Path $Mine "$name.toml"
        if (Test-Path $target) { Stop-With "my-settings\$name.toml already exists: use 'Edit a settings file' to change it." }
        New-Item -ItemType Directory -Force $Mine | Out-Null
        # The preset as it is, byte for byte: it names no mesh folder, so the copy finds them as the preset does.
        [System.IO.File]::WriteAllBytes($target, [System.IO.File]::ReadAllBytes($preset.Path))
        Write-Host "Made my-settings\$name.toml from the $($preset.Label) preset."
        Edit-Settings $target
    }
    'edit-settings' {
        if (-not (Test-Path $Mine) -or @(Get-ChildItem $Mine -Filter '*.toml').Count -eq 0) {
            Stop-With "You have no settings files of your own yet. Make one with 'New settings file' (the presets stay as shipped, so updates never clash with your changes)."
        }
        $items = foreach ($file in Get-ChildItem $Mine -Filter '*.toml' | Sort-Object LastWriteTime -Descending) {
            [pscustomobject]@{ Label = $file.Name; Path = $file.FullName }
        }
        Edit-Settings (Choose 'Which settings file?' @($items) '1').Path
    }
    'train' {
        $settings = Pick-Settings 'Which settings should the new bot train with?'
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($settings) -replace '_starter$', ''
        $name = Ask-Name 'Name for the new run' ("$stem-" + (Get-Date -Format 'yyyyMMdd-HHmm'))
        if (Test-Path (Join-Path $Runs $name)) {
            $again = (Ask "A run called $name already exists. Continue that one instead? (y/n):").Trim()
            if ($again -match '^[yY]') { Invoke-Nitro @('resume', $name); exit $script:Code }
            Stop-With 'Nothing started. Pick another name next time.'
        }
        Write-Host ''
        Write-Host "Training '$name'. The progress page opens in your browser after the first few minutes."
        Write-Host 'To stop and save: press Ctrl+C once in this window and wait for "Saved" (if Windows then asks'
        Write-Host '"Terminate batch job (Y/N)?", either answer is fine). Closing the window instead loses the'
        Write-Host 'training since the last save.'
        Open-Report-When-Ready (Join-Path (Join-Path $Runs $name) 'report.html')
        Invoke-Nitro @('train', $settings, '--name', $name)
        exit $script:Code
    }
    'resume' {
        $name = Pick-Run 'Which run should continue?'
        Write-Host 'To stop and save: press Ctrl+C once and wait for "Saved".'
        Open-Report-When-Ready (Join-Path (Join-Path $Runs $name) 'report.html')
        Invoke-Nitro @('resume', $name)
        exit $script:Code
    }
    'progress' {
        $name = Pick-Run 'Whose progress page?'
        $report = Join-Path (Join-Path $Runs $name) 'report.html'
        if (-not (Test-Path $report)) { Stop-With "$name has no progress page yet: it appears after the first few minutes of training." }
        Open-Thing $report
    }
    'watch' {
        $name = Pick-Run 'Watch which run''s newest bot play?'
        Invoke-Nitro @('watch', $name)
        exit $script:Code
    }
    'export' {
        $name = Pick-Run 'Which run''s newest bot should become an RLBot bot?'
        Invoke-Nitro @('export', $name)
        $folder = Join-Path (Join-Path $Root 'bots') $name
        if ($script:Code -eq 0 -and ($DryRun -or (Test-Path $folder))) { Open-Thing $folder }
        exit $script:Code
    }
    'doctor' {
        Invoke-Nitro @('doctor')
        exit $script:Code
    }
    'dashboard' {
        # The dashboard is a window of its own: started without a console, as the Start menu shortcut starts
        # it (nitro.dashboard.winapp open: tools, caches and temporary files inside this folder), so this
        # window can close at once.
        $pythonw = Join-Path $Root '.venv\Scripts\pythonw.exe'
        if (-not (Test-Path $pythonw)) { Stop-With 'NitroLearn is not installed yet: double-click "Install NitroLearn.cmd" in its folder.' }
        . (Join-Path $Root 'tools\env.ps1')
        if ($DryRun) { Write-Host 'would run: pythonw -m nitro.dashboard.winapp open'; exit 0 }
        Start-Process -FilePath $pythonw -ArgumentList @('-m', 'nitro.dashboard.winapp', 'open') -WorkingDirectory $Root
        exit 0
    }
    default {
        Stop-With "Unknown shortcut action '$Action'."
    }
}
