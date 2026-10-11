"""The extension ZIP offered on the site must contain everything the
manifest and scripts load (content scripts, the on-device model) and none
of the dev-only files (node_modules/, tests/)."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "browser-extension"


def _app_constant(name):
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def _packaged(rel: str) -> bool:
    parts = Path(rel).parts
    return (not set(parts) & _app_constant("EXTENSION_EXCLUDED_DIRS")
            and parts[-1] not in _app_constant("EXTENSION_EXCLUDED_FILES"))


def test_everything_the_extension_loads_is_packaged():
    m = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    needed = {m["background"]["service_worker"], m["action"]["default_popup"]}
    for cs in m["content_scripts"]:
        needed |= set(cs["js"]) | set(cs.get("css", []))
    needed |= set(re.findall(r'importScripts\(([^)]*)\)', (EXT / "background.js").read_text())[0].replace('"', "").replace(" ", "").split(","))
    needed |= set(re.findall(r'<script src="([^"]+)"', (EXT / "popup.html").read_text()))
    needed.add("char_model.tsv")
    for rel in sorted(needed):
        assert (EXT / rel).is_file(), rel
        assert _packaged(rel), rel


def test_dev_files_are_not_packaged():
    assert not _packaged("node_modules/jsdom/package.json")
    assert not _packaged("tests/content.test.js")
    assert not _packaged("package-lock.json")


def test_extension_model_is_the_android_model():
    android = ROOT / "android" / "app" / "src" / "main" / "assets" / "char_model.tsv"
    assert (EXT / "char_model.tsv").read_bytes() == android.read_bytes()
