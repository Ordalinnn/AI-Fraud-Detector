"""The Figma presentation plugin carries its screenshots inside code.js."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parent.parent / "tools" / "figma-defense-plugin"


def test_embedded_screenshots_are_up_to_date():
    r = subprocess.run([sys.executable, str(PLUGIN / "embed_images.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + "\nrun: python tools/figma-defense-plugin/embed_images.py"


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_plugin_code_parses():
    r = subprocess.run(["node", "--check", str(PLUGIN / "code.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
