"""Your own Rocket League replays as recordings `nitro pretrain` reads, without training.
`[pretrain]` in a settings file does this by itself when a run starts; this tool is for converting
ahead of time.

    .\\.venv\\Scripts\\python -I tools\\replays.py                       the newest 1v1 replays, up to 30
    .\\.venv\\Scripts\\python -I tools\\replays.py --team-size 2 --limit 50
    .\\.venv\\Scripts\\python -I tools\\replays.py --from my-replays
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nitro.replays import OUT, ROOT, ReplayError, convert_replays, describe, documents, game_folders


def clone_check(settings: Path, files: list[Path], epochs: int = 3) -> list[str]:
    """A short cloning run on the recordings, measured on whole recordings it does not learn from, against
    always choosing the commonest action: the labels are learnable when the first beats the second. A check
    of the data, not a bot to keep (nothing is saved)."""
    import torch

    from nitro import pretrain
    from nitro.config import load_config
    from nitro.training import resolve_device

    config = load_config(settings)
    notes: list[str] = []
    samples = pretrain.load(files, config, notes=notes, rows="replicar")  # this tool's conversions
    train_set, held = pretrain.split(samples, 0.2, config.run.seed)
    common = int(torch.bincount(train_set.label.argmax(1)).argmax())
    baseline = float((held.label.argmax(1) == common).float().mean())
    lines = [
        f"Cloning check ({settings.name}, {epochs} passes): {len(samples):,} samples from "
        f"{len(samples.files)} recordings, {len(held):,} held back; {len(notes)} recordings left out (other "
        "game or observation)",
        f"  always the commonest action agrees on {baseline:.1%} of the held-back samples",
    ]
    _, history = pretrain.train(
        config,
        train_set,
        held,
        epochs=epochs,
        batch=4096,
        lr=3e-4,
        device=resolve_device(config.run),
        seed=config.run.seed,
    )
    lines += [
        f"  pass {e.epoch}: loss {e.train_loss:.3f}, held back {e.held_loss:.3f}, first choice agrees "
        f"{e.held_agreement:.1%}"
        for e in history
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn your own replays into recordings for nitro pretrain.")
    parser.add_argument(
        "--from", dest="source", type=Path, help="a folder of .replay files (default: the game's)"
    )
    parser.add_argument("--team-size", type=int, default=1, help="1 for 1v1 (default), 2, 3")
    parser.add_argument("--limit", type=int, default=30, help="at most this many recordings (default: 30)")
    parser.add_argument("--meshes", type=Path, default=ROOT / "collision_meshes", help="RocketSim's meshes")
    parser.add_argument("--out", type=Path, default=OUT, help="where the recordings go (default: replays)")
    parser.add_argument("--report", type=Path, help="also save the report (counts only) here")
    parser.add_argument("--clone-check", type=Path, metavar="SETTINGS", help="then a short cloning check")
    args = parser.parse_args(argv)
    sources = [args.source] if args.source else game_folders(documents())
    try:
        report = convert_replays(sources, args.team_size, args.limit, args.meshes, args.out, print)
    except ReplayError as err:
        print(f"Cannot convert: {err}", file=sys.stderr)
        return 1
    text = describe(report, args.team_size)
    if args.clone_check and report.kept:
        text += "\n" + "\n".join(clone_check(args.clone_check, report.recordings))
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if report.found == 0:
        print(
            "No replays found: save some in the game (Match History, Save Replay) or pass --from.",
            file=sys.stderr,
        )
        return 1
    folder = args.out / f"{args.team_size}v{args.team_size}"
    # replicar's rows wherever --out puts them: nitro pretrain's auto knows them by the replays folder
    print(f"Recordings are in {folder}. Next: nitro pretrain <settings> {folder} --rows replicar")
    return 0 if report.kept else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
