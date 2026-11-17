#!/usr/bin/env bash
# Builds make_fixture against the LibTorch inside the project's PyTorch (no separate LibTorch download).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
torch_dir="$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))')"
abi="$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
out="${1:-$here/make_fixture}"
g++ -O1 -std=c++20 -D_GLIBCXX_USE_CXX11_ABI="$abi" "$here/make_fixture.cpp" -o "$out" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" -ltorch -ltorch_cpu -lc10
echo "built $out"
