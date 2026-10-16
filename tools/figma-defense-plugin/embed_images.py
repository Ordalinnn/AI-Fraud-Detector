"""Embed img/*.jpg into code.js (between BEGIN IMAGES / END IMAGES).

A Figma development plugin is a single file and has no network access here,
so the screenshots travel inside code.js as base64. Run after changing a
screenshot; `--check` fails when code.js is out of date.
"""
import base64
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code.js"


def block() -> str:
    lines = ["const IMAGES = {"]
    for img in sorted((HERE / "img").glob("*.jpg")):
        lines.append(f'  {img.stem}: "{base64.b64encode(img.read_bytes()).decode()}",')
    lines.append("};")
    return "\n".join(lines)


def render(src: str) -> str:
    return re.sub(r"(// BEGIN IMAGES\n).*?(// END IMAGES)", lambda m: m.group(1) + block() + "\n" + m.group(2), src, flags=re.S)


if __name__ == "__main__":
    src = CODE.read_text(encoding="utf-8")
    out = render(src)
    if "--check" in sys.argv:
        sys.exit(0 if out == src else "code.js images are out of date: run embed_images.py")
    CODE.write_text(out, encoding="utf-8")
    print("embedded:", ", ".join(p.stem for p in sorted((HERE / "img").glob("*.jpg"))))
