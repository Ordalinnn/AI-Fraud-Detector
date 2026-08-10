"""
Integration tests for app.py's Streamlit-specific wiring: ensemble_predict(),
session-scoped history persistence, and shared feedback persistence.

Unlike test_fraud_logic.py, these deliberately DO drive the real app.py —
that's the point. ensemble_predict() and the history/feedback persistence
functions only exist in app.py (which calls st.set_page_config() and trains
models at module import time), so there's no way to exercise them without a
live Streamlit script context. streamlit.testing.v1.AppTest provides exactly
that: a headless script run that can set widget values, click buttons, and
inspect session_state and on-disk side effects, without a browser or a real
HTTP server.

Each test runs from an isolated tmp_path working directory (via the
isolated_cwd fixture) so the session_data/ and feedback.json files these
tests create never touch the real project directory.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _submit_message(at, text):
    at.text_area(key="main_input_text").set_value(text)
    submit = [b for b in at.button if "analysis_form" in b.key][0]
    submit.click().run()
    return at


def test_analysis_produces_a_result_without_exceptions(isolated_cwd):
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    assert not at.exception, at.exception

    _submit_message(at, "Srochno! Vasha karta zablokirovana, otpravte kod iz SMS http://secure-login.xyz")
    assert not at.exception, at.exception

    result = at.session_state["last_result"]
    assert 0.0 <= result["prob"] <= 1.0
    assert 0.0 <= result["boost"] <= 0.30


def test_analysis_creates_session_scoped_history_file(isolated_cwd):
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    _submit_message(at, "test message for history http://bad-site.xyz")
    assert not at.exception, at.exception

    session_id = at.session_state["session_id"]
    history_path = isolated_cwd / "session_data" / f"history_{session_id}.json"
    assert history_path.exists()
    entries = json.loads(history_path.read_text(encoding="utf-8"))
    assert entries[-1]["Text"] == "test message for history http://bad-site.xyz"


def test_two_sessions_do_not_share_history(isolated_cwd):
    at1 = AppTest.from_file(APP_PATH, default_timeout=90)
    at1.run()
    _submit_message(at1, "message only session one should ever see")
    assert not at1.exception, at1.exception

    at2 = AppTest.from_file(APP_PATH, default_timeout=90)
    at2.run()
    _submit_message(at2, "message only session two should ever see")
    assert not at2.exception, at2.exception

    sid1 = at1.session_state["session_id"]
    sid2 = at2.session_state["session_id"]
    assert sid1 != sid2

    history1 = json.loads((isolated_cwd / "session_data" / f"history_{sid1}.json").read_text(encoding="utf-8"))
    history2 = json.loads((isolated_cwd / "session_data" / f"history_{sid2}.json").read_text(encoding="utf-8"))
    assert "session one" in history1[-1]["Text"]
    assert "session two" in history2[-1]["Text"]
    assert "session two" not in json.dumps(history1)
    assert "session one" not in json.dumps(history2)


def test_feedback_button_appends_to_shared_feedback_file(isolated_cwd):
    at = AppTest.from_file(APP_PATH, default_timeout=90)
    at.run()
    _submit_message(at, "feedback wiring check message")
    assert not at.exception, at.exception

    fb_yes = [b for b in at.button if b.key == "fb_yes"][0]
    fb_yes.click().run()
    assert not at.exception, at.exception
    assert at.session_state["feedback_given"] is True

    feedback_path = isolated_cwd / "feedback.json"
    assert feedback_path.exists()
    entries = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert entries[-1]["Text"] == "feedback wiring check message"
    assert entries[-1]["UserSaysCorrect"] is True
