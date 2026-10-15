"""
Builds the offline message checker for phones (static/check/).

    python scripts/build_web_checker.py          # copy + regenerate
    python scripts/build_web_checker.py --check  # fail if out of date (tests)

The page runs the website's own model in the browser, exactly like the
browser extension, so it reuses the extension's files instead of a second
copy of the logic:

  browser-extension/{i18n,mistakes,normalize,scorer,ensemble,detector}.js
  browser-extension/ensemble_model.json         -> static/check/lib/
  static/icon.png                               -> static/check/icon.png

and writes the asset list + a content hash into static/check/sw.js, so a
changed model or page reaches phones that already cached the old one (the
browser only re-installs a service worker whose own bytes changed).

Streamlit serves ./static at /app/static/ (server.enableStaticServing), so
the page lives at /app/static/check/index.html on the site itself.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "browser-extension"
CHECK = ROOT / "static" / "check"
LIB = CHECK / "lib"

COPIES = {EXT / name: LIB / name for name in
          ["i18n.js", "mistakes.js", "normalize.js", "scorer.js", "ensemble.js", "detector.js", "ensemble_model.json"]}
COPIES[ROOT / "static" / "icon.png"] = CHECK / "icon.png"

SW = CHECK / "sw.js"
BEGIN, END = "// BEGIN GENERATED (scripts/build_web_checker.py)", "// END GENERATED"


ASSET_TYPES = {".html", ".js", ".css", ".json", ".png"}
BINARY = {".png"}


def assets() -> list[str]:
    """Every file the page needs, relative to static/check/ (sw.js itself
    excluded). Hidden files (.DS_Store) and other stray files are left out:
    one file the server does not have would make the whole offline install
    fail on phones. Not "./": Streamlit does not serve folders; sw.js answers
    a navigation to the folder with index.html instead."""
    return sorted(p.relative_to(CHECK).as_posix() for p in CHECK.rglob("*")
                  if p.is_file() and p != SW and p.suffix in ASSET_TYPES
                  and not any(part.startswith(".") for part in p.relative_to(CHECK).parts))


def version(files: list[str]) -> str:
    h = hashlib.sha256()
    for rel in files:
        data = (CHECK / rel).read_bytes()
        if Path(rel).suffix not in BINARY:
            data = data.replace(b"\r\n", b"\n")  # same hash on a Windows (CRLF) checkout
        h.update(rel.encode())
        h.update(data)
    return h.hexdigest()[:16]


def generated_block() -> str:
    files = assets()
    return (f"{BEGIN}\n"
            f"const VERSION = {json.dumps(version(files))};\n"
            f"const ASSETS = {json.dumps(files, indent=2)};\n"
            f"{END}")


def render_sw() -> str:
    text = SW.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"{SW}: generated block markers not found")
    return pattern.sub(lambda _: generated_block(), text)


def main(check: bool) -> int:
    stale = [str(dst.relative_to(ROOT)) for src, dst in COPIES.items()
             if not dst.exists() or dst.read_bytes() != src.read_bytes()]
    if check:
        if stale:
            print("out of date copies: " + ", ".join(stale))
            return 1
        if render_sw() != SW.read_text(encoding="utf-8"):
            print("static/check/sw.js asset list / version is out of date")
            return 1
        return 0
    LIB.mkdir(parents=True, exist_ok=True)
    for src, dst in COPIES.items():
        shutil.copyfile(src, dst)
    SW.write_text(render_sw(), encoding="utf-8")
    print(f"static/check: {len(assets())} files, version {version(assets())}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
