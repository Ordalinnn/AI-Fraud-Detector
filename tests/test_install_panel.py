"""The phone/computer how-to panel renders in every language without errors."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from translations import LANG_OPTIONS

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("lang", LANG_OPTIONS)
def test_install_panel_renders(tmp_path, monkeypatch, lang):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["lang"] = lang
    at.session_state["show_install_instructions"] = True
    at.run()
    assert not at.exception, at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "AI-Fraud-Detector.apk" in md          # Android tab links the APK
    assert "#text=" in md                          # iPhone tab has the shortcut URL
    assert "app/static/check/index.html" in md     # ...pointing at the offline check page
    assert len(at.tabs) >= 3
