# Full test suite and code checks on Windows; tools/check.sh is the Linux version. Run from anywhere:
#     powershell -File tools\check.ps1
# Requires a developer install: .\install.ps1 -Dev
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
. (Join-Path $PSScriptRoot 'env.ps1')
$venv = Join-Path $root '.venv\Scripts'

$failed = @()
function Run([string]$Name, [scriptblock]$Command) {
    Write-Host "`n--- $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { $script:failed += $Name }
}

Run 'Rust formatting' { cargo fmt --all --check }
# The rewards that come with NitroLearn are compiled in through build.rs, which cargo fmt never sees.
Run 'Rust formatting (rewards library)' { rustfmt --edition 2024 --check (Get-ChildItem crates\nitro_core\src\custom\library\*.rs).FullName }
Run 'Rust lints' { cargo clippy --workspace --all-targets -- -D warnings }
Run 'Rust formatting (nitro-replicar)' { cargo fmt --manifest-path crates/nitro_replicar/Cargo.toml --check }
Run 'Rust lints (nitro-replicar)' { cargo clippy --manifest-path crates/nitro_replicar/Cargo.toml --locked -- -D warnings }
Run 'Rust tests' { cargo test --workspace }
Run 'Rebuild Python extension' { & "$venv\python.exe" -m pip install --editable '.[dev]' --quiet }
Run 'Python lints' { & "$venv\ruff.exe" check python tests benchmarks tools }
Run 'Python formatting' { & "$venv\ruff.exe" format --check python tests benchmarks tools }
Run 'Python types' { & "$venv\mypy.exe" }
# -rs: every skipped test with its reason, so a skip is never mistaken for a pass.
Run 'Python tests' { & "$venv\pytest.exe" -q -rs }

if ($failed.Count -gt 0) {
    Write-Host "`nFAILED: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll checks passed." -ForegroundColor Green
