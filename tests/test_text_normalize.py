"""text_normalize.py: undoing scammers' disguises before the models read text."""
import importlib.util
from pathlib import Path

import pytest

import fraud_eval as fe
from fraud_logic import extract_features
from text_normalize import normalize_text

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("raw, expected", [
    ("Ваша к@рта зaблoкирoвана, назовите к0д", "Ваша карта заблокирована, назовите код"),
    ("к о д ы из смс", "коды из смс"),
    ("к.о.д", "код"),
    ("к​о​д", "код"),
    ("ＳＭＳ", "SMS"),
    ("Your сodе is 1234", "Your code is 1234"),
    ("kаspi.kz", "kaspi.kz"),
])
def test_disguises_are_undone(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("text", [
    "а я в шоке", "т.е. завтра", "20000тг переведите", "WhatsAppқа жазыңыз", "iPhoneды", "Kaspiге",
    "user@mail.ru", "SMS-ті жіберіңіз", "Сіздің картаңыз бұғатталды", "Wi-Fi пароль 12345678",
    "Привет, урок перенесли на пятницу", "Meeting at 10:30, room B-204",
])
def test_ordinary_text_is_left_alone(text):
    assert normalize_text(text) == text


def test_idempotent_and_leaves_most_real_messages_untouched():
    texts = fe.load_dataset()["text"].tolist()
    changed = 0
    for t in texts:
        n = normalize_text(t)
        assert normalize_text(n) == n, t
        changed += n != t
    # the hand-written data has a few deliberately disguised scams; ordinary
    # text must almost never change
    assert changed / len(texts) < 0.05, changed


def test_keywords_are_counted_in_disguised_text_but_domains_stay_raw():
    plain, _ = extract_features("Срочно! Ваша карта заблокирована, назовите код")
    disguised, _ = extract_features("Cрoчнo! Ваша кaртa зaблoкирoвaнa, нaзoвитe к0д")
    assert disguised["urgent_count"] == plain["urgent_count"] > 0
    assert disguised["threat_count"] == plain["threat_count"]
    feats, domains = extract_features("Войдите: http://kаspi.kz/login")
    assert feats["homoglyph_domain"] == 1, domains


def test_golden_file_for_kotlin_and_js_is_up_to_date():
    spec = importlib.util.spec_from_file_location("wng", ROOT / "scripts" / "write_normalize_golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.GOLDEN.read_text(encoding="utf-8") == m.render(), "run python scripts/write_normalize_golden.py"


def test_is_disguised():
    from text_normalize import is_disguised
    for t in ["Ваша к@рта", "ＳＭＳ", "к о д ы", "к\u200bод", "к̲о̲д", "kаspi", "𝐤𝐨𝐝"]:
        assert is_disguised(t), t
    for t in ["Привет ❤️", "семья 👨‍👩‍👧", "площадь 50 м²", "№5", "WhatsAppқа", "Wi-Fi", "текст…", ""]:
        assert not is_disguised(t), t
