"""Makes NitroLearn's app icon from the logo: `python/nitro/brand/icon.ico` (Windows: the window, the taskbar
and the shortcuts) and `python/nitro/brand/icon.png` (256 px, for the other systems).

    python tools/make_icon.py

Each size is drawn from the vector logo by Playwright's Chromium (so small sizes stay sharp), then packed
here with the standard library: the sizes under 256 px as 32-bit bitmaps with an alpha channel (what every
Windows component reads), 256 px as PNG (as Windows' own icons do).
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "python" / "nitro" / "brand"
SIZES = (16, 24, 32, 48, 64, 128, 256)
PNG_FROM = 256  # this size and larger are stored as PNG inside the .ico

# Draws the SVG at n x n pixels and returns its RGBA bytes (not premultiplied), row by row from the top.
_DRAW = """async ([svg, n]) => {
    const img = new Image();
    img.src = 'data:image/svg+xml;base64,' + btoa(svg);
    await img.decode();
    const canvas = document.createElement('canvas');
    canvas.width = n;
    canvas.height = n;
    const g = canvas.getContext('2d');
    g.imageSmoothingQuality = 'high';
    g.drawImage(img, 0, 0, n, n);
    return Array.from(g.getImageData(0, 0, n, n).data);
}"""


def render(svg: str, sizes: tuple[int, ...] = SIZES) -> dict[int, bytes]:
    """The logo drawn at each size, as RGBA bytes."""
    from playwright.sync_api import sync_playwright

    drawn: dict[int, bytes] = {}
    with sync_playwright() as play:
        browser = play.chromium.launch(args=["--no-proxy-server"])
        try:
            tab = browser.new_page()
            tab.set_content("<!doctype html><html><body></body></html>")
            for n in sizes:
                sized = svg.replace("<svg ", f'<svg width="{n}" height="{n}" ', 1)
                drawn[n] = bytes(tab.evaluate(_DRAW, [sized, n]))
        finally:
            browser.close()
    return drawn


def png(rgba: bytes, n: int) -> bytes:
    """An n x n RGBA image as a PNG file."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    row = n * 4
    raw = b"".join(b"\x00" + rgba[y * row : (y + 1) * row] for y in range(n))
    header = struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def dib(rgba: bytes, n: int) -> bytes:
    """An n x n RGBA image as an icon bitmap: a 32-bit BGRA image (rows from the bottom) and its 1-bit mask
    (set where the image is fully transparent), each row of the mask padded to 4 bytes."""
    row = n * 4
    pixels = bytearray()
    for y in range(n - 1, -1, -1):
        line = rgba[y * row : (y + 1) * row]
        for x in range(n):
            r, g, b, a = line[x * 4 : x * 4 + 4]
            pixels += bytes((b, g, r, a))
    stride = (n + 31) // 32 * 4
    mask = bytearray()
    for y in range(n - 1, -1, -1):
        bits = bytearray(stride)
        for x in range(n):
            if rgba[(y * n + x) * 4 + 3] == 0:
                bits[x // 8] |= 0x80 >> (x % 8)
        mask += bits
    header = struct.pack("<IiiHHIIiiII", 40, n, n * 2, 1, 32, 0, len(pixels) + len(mask), 0, 0, 0, 0)
    return header + bytes(pixels) + bytes(mask)


def ico(images: dict[int, bytes]) -> bytes:
    """The .ico file holding every size in `images` (RGBA bytes by size)."""
    sizes = sorted(images)
    blobs = [png(images[n], n) if n >= PNG_FROM else dib(images[n], n) for n in sizes]
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = len(head) + 16 * len(sizes)
    entries = b""
    for n, blob in zip(sizes, blobs, strict=True):
        side = 0 if n >= 256 else n  # 0 means 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    return head + entries + b"".join(blobs)


def main() -> int:
    svg = (BRAND / "logo.svg").read_text(encoding="utf-8").strip()
    images = render(svg)
    (BRAND / "icon.ico").write_bytes(ico(images))
    (BRAND / "icon.png").write_bytes(png(images[256], 256))
    print(f"wrote {BRAND / 'icon.ico'} ({', '.join(map(str, SIZES))} px) and {BRAND / 'icon.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
