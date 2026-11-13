#!/usr/bin/env bash
# Runs the `nitro` command from NitroLearn's own Python (made by install.sh), with tools, caches and
# temporary files kept inside this folder (tools/env.sh):  ./nitro.sh dashboard   ./nitro.sh --help
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -x "$here/.venv/bin/nitro" ]; then
    echo "NitroLearn is not installed yet. Run:  bash install.sh" >&2
    exit 1
fi
# shellcheck source=tools/env.sh
. "$here/tools/env.sh"
exec "$here/.venv/bin/nitro" "$@"
