"""
Regression test: after a deploy that changes translations.py while the
server keeps running, the app must pick up the new keys instead of failing
with KeyError (seen in production with "share_title").
"""
import os
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

import translations

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_changed_translations_module_is_reloaded(tmp_path, monkeypatch):
    # The app reloads the real module in place; restore its namespace after
    # the test so other test files keep the objects they imported.
    saved = dict(vars(translations))
    try:
        _run(tmp_path, monkeypatch)
    finally:
        vars(translations).clear()
        vars(translations).update(saved)


def _run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # Simulate a server that imported an OLD translations.py: an in-memory
    # copy without the key, and a file that looks newer than what was loaded.
    translations.TEXT = {lang: {k: v for k, v in d.items() if k != "share_title"}
                         for lang, d in translations.TEXT.items()}
    translations._loaded_mtime = os.path.getmtime(translations.__file__) - 60

    at.session_state["show_install_instructions"] = True
    at.run()
    assert not at.exception, at.exception
    import sys
    assert "share_title" in sys.modules["translations"].TEXT["🇷🇺 RU"]
