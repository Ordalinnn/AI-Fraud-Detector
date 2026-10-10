"""
"Share -> AI Fraud Detector" (Android Web Share Target) and the iPhone
Shortcut both open the app with the message in the URL query string. The
app must analyse it straight away and then clear it from the URL.
"""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(ROOT / "app.py")


def _open_with(tmp_path, monkeypatch, **params):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    for k, v in params.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def test_shared_scam_is_analysed_immediately(tmp_path, monkeypatch):
    msg = "Срочно! Ваша карта заблокирована. Отправьте код из SMS http://kaspi-secure.xyz"
    at = _open_with(tmp_path, monkeypatch, text=msg)
    result = at.session_state["last_result"]
    assert result["input_text"] == msg
    assert result["prob"] > 0.5
    # Kept in the URL so a reload (common on iPhone right after opening)
    # still shows the message instead of an empty box.
    assert at.query_params["text"] in (msg, [msg])  # AppTest returns a list


def test_shared_text_is_analysed_once_per_session_so_user_can_edit(tmp_path, monkeypatch):
    msg = "Срочно! Ваша карта заблокирована. Отправьте код из SMS http://kaspi-secure.xyz"
    at = _open_with(tmp_path, monkeypatch, text=msg)
    at.text_area(key="main_input_text").set_value("Привет, как дела?")
    # (the box is inside a form: newer Streamlit commits it on submit)
    [b for b in at.button if b.key and "analysis_form" in b.key][0].click().run()
    assert not at.exception, at.exception
    # the shared text must not overwrite what the user typed and analysed
    assert at.session_state["last_result"]["input_text"] == "Привет, как дела?"


def test_shared_ordinary_message_is_safe(tmp_path, monkeypatch):
    at = _open_with(tmp_path, monkeypatch, text="Привет, урок математики перенесли на пятницу, не забудь тетрадь.")
    assert at.session_state["last_result"]["prob"] < 0.5


def test_android_fields_are_joined_without_repeating_the_link(tmp_path, monkeypatch):
    at = _open_with(tmp_path, monkeypatch, title="Kaspi", text="Подтвердите вход http://kaspi-login.xyz",
                    url="http://kaspi-login.xyz")
    text = at.session_state["last_result"]["input_text"]
    assert text.count("http://kaspi-login.xyz") == 1
    assert text.startswith("Kaspi")


def test_no_share_params_means_no_automatic_analysis(tmp_path, monkeypatch):
    at = _open_with(tmp_path, monkeypatch)
    assert "last_result" not in at.session_state


def test_manifest_declares_share_target():
    m = json.loads((ROOT / "static" / "manifest.json").read_text(encoding="utf-8"))
    st_ = m["share_target"]
    assert st_["method"] == "GET"
    assert st_["params"]["text"] == "text"
    assert st_["action"].startswith(m["scope"])


def test_shortcut_without_text_explains_the_fix(tmp_path, monkeypatch):
    # "#text=" with nothing after it -> the page script turns it into ?text_missing=1
    at = _open_with(tmp_path, monkeypatch, text_missing="1")
    assert "last_result" not in at.session_state
    assert any("Получить буфер обмена" in w.value for w in at.warning)


def test_empty_text_param_explains_the_fix(tmp_path, monkeypatch):
    at = _open_with(tmp_path, monkeypatch, text="")
    assert "last_result" not in at.session_state
    assert any("Получить буфер обмена" in w.value for w in at.warning)
