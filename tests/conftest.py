import os

import pytest

# Deterministic mode (tests/test_training_runs.py) needs cuBLAS's fixed workspace layout, which cuBLAS reads
# when it first starts in a process, so it is set before any test touches the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _no_installed_rlbot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard counts an RLBot v5 installed in %LOCALAPPDATA% as found: the tests see none there unless
    they place one themselves, on any computer."""
    monkeypatch.setattr("nitro.rlbot_install.local_app_data", lambda: None)
