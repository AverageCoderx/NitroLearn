"""`nitro export`: the policy file the Rust bot reads must reproduce the PyTorch policy."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

from nitro import _core, export, notices, prebuilt
from nitro.config import default_config
from nitro.config.schema import ModelConfig
from nitro.export import ExportError, export_run, write_policy
from nitro.learner import Learner
from nitro.model import ActorCritic
from nitro.ppo import masked_log_probs
from nitro.runs import RunFolder, fixed_settings

OBS, ACTIONS = 112, 90
PROJECT = Path(__file__).resolve().parents[1]
SHAPES = {
    "default": ModelConfig(),
    "shared_trunk": dataclasses.replace(ModelConfig(), shared=(64, 64), policy=(32,), value=(32,)),
    "no_layer_norm": dataclasses.replace(ModelConfig(), layer_norm=False, policy=(48, 48)),
    "leaky_relu": dataclasses.replace(ModelConfig(), activation="leaky_relu", policy=(40,)),
    "tanh_pytorch_init": dataclasses.replace(
        ModelConfig(), activation="tanh", init="pytorch", policy=(40, 24)
    ),
    "large": dataclasses.replace(ModelConfig(), policy=(1024, 1024), value=(16,)),
    # Residual blocks (each network's first layer plain, as it changes the width) and value heads.
    "residual": dataclasses.replace(
        ModelConfig(), residual=True, activation="leaky_relu", policy=(64, 64, 64), value=(48, 48)
    ),
    "residual_heads": dataclasses.replace(
        ModelConfig(),
        residual=True,
        activation="leaky_relu",
        shared=(112, 96, 96),  # the first block keeps the observation's width
        policy=(64, 64),
        value=(80, 80),
        value_head_layers=(80,),
        value_heads=("main", "goal", "vdag1"),
    ),
}


def _config(model: ModelConfig):  # type: ignore[no-untyped-def]
    config = default_config(Path.cwd())
    return dataclasses.replace(
        config, model=model, run=dataclasses.replace(config.run, device="cpu"), paths=config.paths
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_exported_policy_matches_pytorch(shape: str, tmp_path: Path) -> None:
    torch.manual_seed(3)
    config = _config(SHAPES[shape])
    model = ActorCritic(OBS, ACTIONS, config.model)
    # Trained weights are not tiny; scale them up so the logits are too and differences would show.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    model.eval()
    path = tmp_path / "policy.nwt"
    write_policy(path, model, config, OBS, ACTIONS, "test-run", 12_345)

    policy = _core.Policy(path)
    assert (policy.obs_size, policy.num_actions) == (OBS, ACTIONS)
    header = policy.header()
    assert header["run"] == "test-run" and header["agent_steps"] == 12_345
    assert header["tick_skip"] == config.env.tick_skip and header["action_delay"] == config.env.action_delay

    if SHAPES[shape].residual:
        assert header["format"] == 8, "residual blocks need bot file format 8"
    rows = 500
    obs = torch.randn(rows, OBS) * 0.7
    with torch.no_grad():
        expected = model(obs)[0]
    got = torch.tensor(policy.logits_batch(obs.reshape(-1).tolist(), rows)).reshape(rows, ACTIONS)
    assert torch.isfinite(got).all()
    worst = (got - expected).abs().max().item()
    assert worst <= 1e-4, f"max |delta logit| {worst}"
    # Masked argmax agrees with the PyTorch policy's most likely allowed action.
    mask = torch.rand(rows, ACTIONS) > 0.3
    mask[:, 0] = True
    expected_actions = masked_log_probs(expected, mask).argmax(1)
    for row in range(0, rows, 25):
        assert policy.argmax(got[row].tolist(), mask[row].int().tolist()) == int(expected_actions[row])


def test_unsupported_activation_is_refused(tmp_path: Path) -> None:
    config = _config(dataclasses.replace(ModelConfig(), activation="gelu", policy=(8,)))
    model = ActorCritic(OBS, ACTIONS, config.model)
    with pytest.raises(ExportError, match="relu"):
        write_policy(tmp_path / "policy.nwt", model, config, OBS, ACTIONS, "r", 0)


def _saved_run(tmp_path: Path) -> RunFolder:
    """A run named my-bot with one save, at 2,000,000 agent steps."""
    config = _config(dataclasses.replace(ModelConfig(), policy=(16,), value=(16,)))
    settings = tmp_path / "bot.toml"
    settings.write_text('[run]\ndevice = "cpu"\n[model]\npolicy = [16]\nvalue = [16]\n', encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "my-bot", settings)
    model = ActorCritic(OBS, ACTIONS, config.model)
    learner = Learner(model, config.ppo, 0.99, torch.device("cpu"), fp16=False)
    state = {
        "learner": learner.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.empty(0, dtype=torch.uint8),
        "checkpoint": 1,
        "agent_steps": 2_000_000,
        "iterations": 40,
        "settings": fixed_settings(config),
    }
    folder.save_checkpoint(state, 2_000_000, keep=5, milestone_steps=0)
    return folder


def test_export_run_writes_a_complete_bot_folder(tmp_path: Path) -> None:
    folder = _saved_run(tmp_path)
    out = tmp_path / "bots" / "my-bot"
    exported = export_run(folder, out, name="My Bot", build=False)
    names = sorted(f.name for f in exported.files)
    assert names == ["NOTICES.md", "README.md", "bot.toml", "loadout.toml", "policy.nwt"]
    assert not exported.built
    bot_toml = (out / "bot.toml").read_text(encoding="utf-8")
    assert 'name = "My Bot"' in bot_toml and 'agent_id = "nitrolearn/my-bot"' in bot_toml
    assert 'run_command = "nitro_bot.exe policy.nwt"' in bot_toml
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "2,000,000 agent steps" in readme and "action delay 1 tick)" in readme
    assert "building it was skipped (`nitro export --no-build`)" in readme
    assert "could not be built" not in readme
    program = "nitro_bot.exe" if sys.platform == "win32" else "nitro_bot"
    assert f"| `{program}` |" in readme
    assert "Never use it\nin online matches against real players" in readme, "fair play"
    table = [line for line in readme.splitlines() if line.startswith("|")]
    assert len(table) == 7 and all(line.endswith("|") for line in table), "every row on one line"
    assert "older versions" not in readme
    policy = _core.Policy(out / "policy.nwt")
    assert policy.header()["agent_steps"] == 2_000_000

    # Exporting again keeps an edited loadout and refreshes the rest.
    (out / "loadout.toml").write_text("# mine\n", encoding="utf-8")
    export_run(folder, out, name=None, build=False)
    assert (out / "loadout.toml").read_text(encoding="utf-8") == "# mine\n"
    assert 'name = "my-bot"' in (out / "bot.toml").read_text(encoding="utf-8")


def test_export_copies_the_release_s_ready_made_bot_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the release's bot program for this code in .tools/prebuilt, no Rust is needed to export."""
    folder = _saved_run(tmp_path)
    shipped = tmp_path / "prebuilt"
    shipped.mkdir()
    (shipped / export.PROGRAM).write_bytes(b"ready-made bot program")
    listed = {"programs": {"nitro_bot": prebuilt.bot_source(PROJECT)}}
    (shipped / prebuilt.LIST).write_text(json.dumps(listed), encoding="utf-8")
    monkeypatch.setattr(prebuilt, "FOLDER", shipped)
    monkeypatch.setattr(export, "_build_bot", lambda project: pytest.fail("no cargo build"))
    out = tmp_path / "bots" / "my-bot"
    exported = export_run(folder, out, name=None)
    assert exported.built and (out / export.PROGRAM).read_bytes() == b"ready-made bot program"
    assert "could not be built" not in (out / "README.md").read_text(encoding="utf-8")


def test_export_without_a_save_is_refused(tmp_path: Path) -> None:
    settings = tmp_path / "bot.toml"
    settings.write_text("[run]\n", encoding="utf-8")
    folder = RunFolder.create(tmp_path / "runs", "empty", settings)
    with pytest.raises(ExportError, match="no save yet"):
        export_run(folder, tmp_path / "out", None, build=False)


def test_every_bot_folder_carries_the_licence_notices(tmp_path: Path) -> None:
    """NOTICES.md: NitroLearn's Required Notice and licence, the bot folder's permission (BOTS-LICENSE.md),
    fair play, and the third-party crates linked into the bot program with their licences."""
    text = notices.notices(PROJECT)
    required = next(line for line in (PROJECT / "LICENSE").read_text(encoding="utf-8").splitlines())
    assert required.startswith("Required Notice:") and required in text
    assert notices.LICENCE_URL in text
    assert "### Bots you make are yours" in text and "#### Additional permission" in text
    assert "as long as you keep any notice files the export puts there" in text
    assert "](" not in text, "BOTS-LICENSE.md's links are plain words in a bot folder"
    assert "## Fair play" in text and "online matches against real players" in text
    if shutil.which("cargo") is None:
        pytest.skip("no Rust here, so the bot program's crates cannot be listed")
    third = notices.third_party(PROJECT)
    for crate in ("rocketsim 0.3.3", "rlbot 0.6.0", "serde_json", "zstd-sys"):
        assert f"- {crate}" in third, crate
    assert "Copyright (c) 2022 ZealanL; Copyright (c) 2025 VirxEC" in third
    assert "Zstandard, the C library it builds: Copyright (c) Meta Platforms" in third
    assert "### MIT License" in third and "### BSD 3-Clause License" in third
    for left_out in ("nitro_core", "pyo3", "serde_derive", "proc-macro2"):
        assert f"- {left_out} " not in third, f"{left_out} is not linked into the bot program"


def test_without_cargo_the_notices_name_the_licences_that_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    notices.third_party.cache_clear()
    monkeypatch.setattr(notices, "_run", lambda project, args: None)
    try:
        text = notices.third_party(Path("/nowhere"))
    finally:
        notices.third_party.cache_clear()
    assert "Cargo.lock" in text and "### MIT License" in text and "### zlib License" in text


def test_the_bot_program_is_built_with_this_computers_folders_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home" / "someone"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("CARGO_HOME", str(home / ".cargo"))
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=x86-64-v2")
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    project = tmp_path / "home" / "someone" / "NitroLearn"
    env = export.build_env(project, project / "target")
    assert "RUSTFLAGS" not in env
    flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[:2] == ["-C", "target-cpu=x86-64-v2"], "the flags cargo would use anyway are kept"
    remaps = [flag.split("=", 1)[1] for flag in flags[2:]]
    assert all(flag.startswith("--remap-path-prefix=") for flag in flags[2:])
    assert f"{home}=home" in remaps and f"{project.resolve()}=nitrolearn" in remaps
    assert f"{home / '.cargo' / 'registry' / 'src'}=crates" in remaps
    lengths = [len(remap.rsplit("=", 1)[0]) for remap in remaps]
    assert lengths == sorted(lengths), "the most specific folder last (the compiler takes the last match)"
    # A program that still names one of them is not copied into a bot folder.
    assert export.folder_in(b"...panicked at " + str(home).encode() + b"/x.rs", project) is not None
    assert export.folder_in(b"crates/index.crates.io-1/serde-1.0/src/de.rs", project) is None
    assert export.folder_in(b"/x/.cargo/registry/src/serde/de.rs", project) is not None


@pytest.mark.skipif(shutil.which("cargo") is None, reason="no Rust here: the bot program cannot be built")
def test_the_built_bot_program_names_no_folder_of_this_computer() -> None:
    """The real build, as `nitro export` runs it: the program holds no home, project or cargo folder."""
    exe, message = export._build_bot(PROJECT)
    assert exe is not None, message
    assert export.folder_in(exe.read_bytes(), PROJECT) is None
