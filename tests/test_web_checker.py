"""The offline phone checker (static/check/) is up to date and complete.

browser-extension/tests/phone_check.test.js runs the page itself (jsdom);
these checks keep the files around it honest: the copies of the extension's
model and scripts, the service worker's file list and version, the web app
manifest, and the link the site gives to iPhone users.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "static" / "check"


def test_copies_and_service_worker_are_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web_checker.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr + "\nrun: python scripts/build_web_checker.py"


def test_service_worker_caches_every_file_the_page_loads():
    sw = (CHECK / "sw.js").read_text(encoding="utf-8")
    assets = json.loads(re.search(r"const ASSETS = (\[.*?\]);", sw, re.S).group(1))
    html = (CHECK / "index.html").read_text(encoding="utf-8")
    loaded = set(re.findall(r'(?:src|href)="([^"#?]+)"', html)) - {"../../../"}
    assert loaded, "no assets found in index.html"
    for rel in loaded:
        assert rel in assets, f"{rel} is loaded by index.html but not cached by sw.js"
        assert (CHECK / rel).is_file(), rel
    # Streamlit does not serve folders: a "./" entry would make install fail
    assert "./" not in assets and all((CHECK / a).is_file() for a in assets)
    assert "lib/ensemble_model.json" in assets


def test_manifest_scope_is_the_checker_folder():
    m = json.loads((CHECK / "manifest.json").read_text(encoding="utf-8"))
    assert m["start_url"] == "./index.html" and m["scope"] == "./"
    assert m["display"] == "standalone"
    assert all((CHECK / i["src"]).is_file() for i in m["icons"])


def test_checker_url_uses_the_streamlit_cloud_prefix_only_there(monkeypatch):
    import ast
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    src = "\n".join(ast.get_source_segment((ROOT / "app.py").read_text(encoding="utf-8"), n)
                    for n in tree.body
                    if (isinstance(n, ast.FunctionDef) and n.name == "checker_url")
                    or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CHECKER_PATH" for t in n.targets)))
    for base, want in [
        ("https://x.streamlit.app/", "https://x.streamlit.app/~/+/app/static/check/index.html"),
        ("http://localhost:8501/", "http://localhost:8501/app/static/check/index.html"),
    ]:
        ns = {"public_app_url": lambda b=base: b}
        exec(src, ns)
        assert ns["checker_url"]() == want
        assert ns["checker_url"]("🇰🇿 KZ") == want + "?lang=kz"
