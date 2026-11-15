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


@pytest.fixture(autouse=True)
def _github_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard's page asks for a newer release when it opens: in the tests GitHub is never reached (the
    check finds it offline and says nothing), unless a test answers for it."""

    def offline() -> object:
        from nitro.update import UpdateError

        raise UpdateError("Could not check for updates: offline in the tests.")

    monkeypatch.setattr("nitro.dashboard.updating.ask_github", offline)
