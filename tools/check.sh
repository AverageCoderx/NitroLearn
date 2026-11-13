#!/usr/bin/env bash
# Full test suite and code checks on Linux or macOS; tools/check.ps1 is the
# Windows version. Run from anywhere:
#     bash tools/check.sh
# Needs the developer install: bash install.sh --dev (a .venv with torch and NitroLearn built from its code).
#
# Tests that need the collision meshes (Rocket League assets, never in git) or a GPU skip themselves; the
# counts are printed at the end so skipped tests are never mistaken for passed ones.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
# shellcheck disable=SC1091
. .venv/bin/activate

failed=()
run() {
    local name=$1
    shift
    printf '\n--- %s\n' "$name"
    "$@" || failed+=("$name")
}
# Every kind of "SKIPPED: ..." notice in a test log, with how often it came (a folder path left out).
skip_notes() {
    grep -oE 'SKIPPED: .*' "$1" | sed -E 's/ at \/.*//' | sort | uniq -c | sed 's/^ */    /'
}

run 'Rust formatting' cargo fmt --all --check
# The rewards that come with NitroLearn are compiled in through build.rs, which cargo fmt never sees.
run 'Rust formatting (rewards library)' bash -c 'rustfmt --edition 2024 --check crates/nitro_core/src/custom/library/*.rs'
run 'Rust lints' cargo clippy --workspace --all-targets -- -D warnings
# nitro-replicar, the replay converter, is a workspace of its own.
run 'Rust formatting (nitro-replicar)' cargo fmt --manifest-path crates/nitro_replicar/Cargo.toml --check
run 'Rust lints (nitro-replicar)' cargo clippy --manifest-path crates/nitro_replicar/Cargo.toml --locked -- -D warnings
# The test steps pipe through tee and grep: `pipefail` keeps the test runner's own exit status.
rust_log=$(mktemp)
run 'Rust tests' bash -o pipefail -c "cargo test --workspace -- --nocapture 2>&1 | tee '$rust_log' | grep -E '^test result|FAILED|panicked'"
rust_skipped=$(grep -o 'SKIPPED: no collision meshes' "$rust_log" | wc -l)
rust_passed=$(grep -E '^test result' "$rust_log" | grep -oE '[0-9]+ passed' | awk '{s+=$1} END {print s+0}')
rust_notes=$(skip_notes "$rust_log")
rm -f "$rust_log"
# A venv made by uv has no pip; uv installs into it instead.
if command -v uv > /dev/null; then
    run 'Rebuild Python extension' uv pip install --editable '.[dev]' --quiet --no-build-isolation
else
    run 'Rebuild Python extension' python -m pip install --editable '.[dev]' --quiet --no-build-isolation
fi
run 'Python lints' ruff check python tests benchmarks tools
run 'Python formatting' ruff format --check python tests benchmarks tools
run 'Python types' mypy
py_log=$(mktemp)
run 'Python tests' bash -o pipefail -c "pytest -q -rs 2>&1 | tee '$py_log' | grep -v '^SKIPPED'"
py_summary=$(grep -E '^[0-9]+ (passed|failed|error)' "$py_log" | tail -1)
if [ -z "$py_summary" ] || echo "$py_summary" | grep -qE 'failed|error'; then
    failed+=('Python tests (summary)')
fi
# pytest's skip reasons (-rs), kept for the end.
py_skips=$(grep '^SKIPPED' "$py_log" | sed 's/^/    /')
rm -f "$py_log"

printf '\nRust tests: %s passed in cargo'"'"'s count, of which %s only skipped (no collision meshes), so %s ran.\n' \
    "$rust_passed" "$rust_skipped" "$((rust_passed - rust_skipped))"
[ -n "$rust_notes" ] && printf 'Their SKIPPED notices:\n%s\n' "$rust_notes"
printf 'Python tests: %s\n' "${py_summary:-no summary line}"
[ -n "$py_skips" ] && printf 'Skipped Python tests:\n%s\n' "$py_skips"
if [ ${#failed[@]} -gt 0 ]; then
    printf '\nFAILED: %s\n' "${failed[*]}"
    exit 1
fi
printf '\nAll checks passed.\n'
