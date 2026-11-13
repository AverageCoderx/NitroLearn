"""Pictures of the match viewer, drawn in headless Chromium (software graphics, so it runs anywhere).

    python tools/viewer_shots.py RECORDING OUT_DIR [--at 5 12] [--cameras broadcast ball]

RECORDING is a recording's page data as JSON, a recording file (.nlrec) or a replay page (watch.html).
One PNG is written per moment, camera and quality level; the browser's console errors are printed and
make the command fail. Needs the Python package `playwright` and a Chromium it can start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from nitro import viewer  # noqa: E402
from nitro.viewer import recording  # noqa: E402

GRAPHICS = [
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--no-proxy-server",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--meshes", type=Path, default=ROOT / "collision_meshes")
    parser.add_argument("--no-meshes", action="store_true", help="draw the generated arena")
    parser.add_argument("--at", type=float, nargs="+", default=[5.0], help="moments of the game, in seconds")
    parser.add_argument("--cameras", nargs="+", default=["broadcast"])
    parser.add_argument("--follow", type=int, default=0, help="the car the car cameras follow")
    parser.add_argument("--quality", nargs="+", default=["high"])
    parser.add_argument("--size", default="1600x900")
    parser.add_argument("--scale", type=float, default=1.0, help="device pixels per CSS pixel")
    parser.add_argument("--frames", type=int, default=12, help="frames drawn before each picture")
    parser.add_argument("--run", default="", help="JavaScript to run in the page before each picture")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    if args.recording.suffix == ".json":
        data = json.loads(args.recording.read_text(encoding="utf-8"))
    else:
        data = recording.load(args.recording)
    arena = None if args.no_meshes else viewer.arena_blob(args.meshes)
    args.out.mkdir(parents=True, exist_ok=True)
    page_file = args.out / "page.html"
    page_file.write_text(viewer.page(data, arena=arena), encoding="utf-8")
    width, height = (int(n) for n in args.size.split("x"))
    problems: list[str] = []
    with sync_playwright() as play:
        browser = play.chromium.launch(args=GRAPHICS)
        tab = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=args.scale)
        tab.on("pageerror", lambda error: problems.append(f"page error: {error}"))
        tab.on("console", lambda message: problems.append(message.text) if message.type == "error" else None)
        tab.goto(page_file.as_uri())
        try:
            tab.wait_for_function("typeof NV !== 'undefined' && NV.frames > 2", timeout=60_000)
        except Exception:
            for problem in problems:
                print(problem, file=sys.stderr)
            raise
        tab.evaluate("NV.player.playing = false")
        for quality in args.quality:
            tab.evaluate("(q) => NV.post.setQuality(q)", quality)
            for moment in args.at:
                for camera in args.cameras:
                    tab.evaluate(
                        "([t, mode, car]) => { NV.player.seek(t); NV.cams.set(mode, car); }",
                        [moment, camera, args.follow],
                    )
                    if args.run:
                        tab.evaluate(args.run)
                    drawn = tab.evaluate("NV.frames")
                    tab.wait_for_function("(n) => NV.frames >= n", arg=drawn + args.frames, timeout=120_000)
                    name = f"{camera}_{moment:g}s_{quality}.png"
                    tab.screenshot(path=str(args.out / name), timeout=180_000)
                    print(f"{name}  frame time {tab.evaluate('NV.frameMs'):.0f} ms")
        browser.close()
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
