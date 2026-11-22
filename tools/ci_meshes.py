"""The collision meshes for the tests in CI, as `nitro fetch-meshes` gets them, without installing NitroLearn
first (the Rust checks run without it): python tools/ci_meshes.py"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_meshes", ROOT / "python" / "nitro" / "fetch_meshes.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.fetch_meshes(ROOT / "collision_meshes")
