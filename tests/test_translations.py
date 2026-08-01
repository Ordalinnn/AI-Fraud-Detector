"""
Tests for translations.py. Like fraud_logic.py, this module has no
Streamlit dependency, so it's importable directly.

validate_translations() already runs at import time and would make the
import itself fail if the language dicts were out of sync — these tests
exist so a broken translation set shows up as a named, readable test
failure in CI output rather than "test collection failed" for an
unrelated-looking reason.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from translations import TEXT, LANG_OPTIONS, DEFAULT_LANG, get_translations, validate_translations


def test_all_languages_declared_match_lang_options():
    assert set(TEXT.keys()) == set(LANG_OPTIONS)


def test_default_lang_is_a_real_language():
    assert DEFAULT_LANG in TEXT


def test_every_language_has_the_same_keys():
    key_sets = [set(d.keys()) for d in TEXT.values()]
    first = key_sets[0]
    for keys in key_sets[1:]:
        assert keys == first


def test_no_language_has_empty_strings():
    for lang, strings in TEXT.items():
        for key, value in strings.items():
            assert value.strip() != "", f"{lang}.{key} is empty"


def test_validate_translations_passes_on_current_data():
    # Should not raise.
    validate_translations()


def test_validate_translations_catches_a_missing_key():
    original = dict(TEXT["🇬🇧 EN"])
    try:
        del TEXT["🇬🇧 EN"]["title"]
        try:
            validate_translations()
            assert False, "expected ValueError for a missing key"
        except ValueError as e:
            assert "🇬🇧 EN" in str(e)
            assert "title" in str(e)
    finally:
        TEXT["🇬🇧 EN"] = original


def test_get_translations_falls_back_for_unknown_language():
    assert get_translations("not-a-real-language") == TEXT[DEFAULT_LANG]


def test_get_translations_returns_requested_language():
    for lang in LANG_OPTIONS:
        assert get_translations(lang) is TEXT[lang]
