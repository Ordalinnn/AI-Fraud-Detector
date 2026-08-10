"""Regenerates browser-extension/icons/*.png and static/icon.png from the
square master icon artwork.

Why this exists: browser-extension/icons/icon16.png, icon48.png, and
icon128.png (and static/icon.png, used by the PWA manifest) were all found
to be byte-for-byte identical 1254x1254 copies of the same master image —
none of them were ever actually resized to the dimensions their filenames
and the manifests (manifest.json / static/manifest.json) declare. Browsers
downscale oversized icons at render time regardless, but shipping a ~1MB
image for what's displayed as a 16x16 toolbar icon is wasteful, and
low-quality box-filtering at that ratio can look worse than a properly
downsampled asset.

Run from the repo root: python scripts/generate_icons.py
Re-run whenever the master icon artwork changes. The master is read from
MASTER_ICON before it's overwritten, so it's safe that one of the outputs
(static/icon.png) shares a path with a historical copy of the source.
"""
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_ICON = REPO_ROOT / "static" / "icon.png"

# (output path, size in pixels)
TARGETS = [
    (REPO_ROOT / "browser-extension" / "icons" / "icon16.png", 16),
    (REPO_ROOT / "browser-extension" / "icons" / "icon48.png", 48),
    (REPO_ROOT / "browser-extension" / "icons" / "icon128.png", 128),
    # Matches the largest size declared in static/manifest.json's icons list
    # (192x192 and 512x512 both currently point at this same file).
    (REPO_ROOT / "static" / "icon.png", 512),
]


def main():
    master = Image.open(MASTER_ICON).convert("RGBA")
    if master.width != master.height:
        raise SystemExit(f"expected a square master icon, got {master.size}")

    for path, size in TARGETS:
        resized = master.resize((size, size), Image.LANCZOS)
        path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(path, format="PNG", optimize=True)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({size}x{size}, {path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
