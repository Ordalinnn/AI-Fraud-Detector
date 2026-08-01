"""
Unit tests for the pure fraud-detection logic in fraud_logic.py.

These deliberately don't import app.py: app.py runs Streamlit calls
(st.set_page_config, components.html, model training) at module import
time, which requires a live Streamlit script context and would make these
tests slow/fragile. fraud_logic.py has zero such dependency, which is the
whole point of having split it out.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraud_logic import (
    extract_features,
    rule_boost,
    risk_level,
    highlight_text,
    brand_impersonation,
    domain_flags,
    get_domain,
    extract_urls,
)


# =========================
# extract_features
# =========================
def test_fraud_message_triggers_multiple_signals():
    features, domains = extract_features(
        "Срочно! Ваша карта заблокирована. Отправьте код из SMS и перейдите "
        "по ссылке http://secure-login.xyz"
    )
    assert features["has_link"] == 1
    assert features["urgent_count"] > 0
    assert features["secret_count"] > 0
    assert features["money_count"] > 0
    assert features["threat_count"] > 0
    assert domains == ["secure-login.xyz"]


def test_safe_message_triggers_no_fraud_signals():
    features, domains = extract_features("Привет, завтра урок математики в 9:00.")
    assert features["has_link"] == 0
    assert features["urgent_count"] == 0
    assert features["secret_count"] == 0
    assert features["threat_count"] == 0
    assert domains == []


def test_safe_message_mentioning_money_is_not_flagged_as_suspicious_domain():
    # A message can mention banking without any link/domain being involved.
    features, domains = extract_features("Зарплата поступила на карту как обычно")
    assert features["has_link"] == 0
    assert features["suspicious_domain"] == 0
    assert domains == []


def test_empty_text_does_not_crash():
    features, domains = extract_features("")
    assert features["word_count"] == 0
    assert features["avg_word_length"] == 0
    assert domains == []


# =========================
# brand impersonation / domain reputation
# =========================
def test_typosquat_domain_is_flagged_as_brand_impersonation():
    assert brand_impersonation("kaspi-login.xyz") == "kaspi.kz"


def test_real_brand_domain_is_not_flagged():
    assert brand_impersonation("kaspi.kz") is None


def test_real_brand_subdomain_is_not_flagged():
    assert brand_impersonation("secure.kaspi.kz") is None


def test_unrelated_domain_is_not_flagged():
    assert brand_impersonation("example.com") is None


def test_domain_flags_includes_critical_severity_for_impersonation():
    flags = domain_flags("kaspi-login.xyz")
    assert any(sev == "critical" for _, sev in flags)
    assert any("Impersonates" in label for label, _ in flags)


def test_get_domain_strips_protocol_and_path():
    assert get_domain("https://www.example.com/path?x=1") == "example.com"
    assert get_domain("http://secure-login.xyz") == "secure-login.xyz"


def test_extract_urls_finds_http_and_www_links():
    urls = extract_urls("visit http://a.com or www.b.com today")
    assert len(urls) == 2


# =========================
# rule_boost
# =========================
def _zero_features():
    features, _ = extract_features("")
    return features


def test_rule_boost_is_zero_for_no_signals():
    assert rule_boost(_zero_features()) == 0.0


def test_rule_boost_is_capped_at_point_three():
    # This text trips nearly every combo rule_boost checks for (link+secret,
    # urgent+money, secret+money, threat+link, pressure, suspicious domain/
    # zone, brand impersonation, multiple warnings, multiple URLs) — the
    # uncapped sum would be 1.00, well over the 0.30 ceiling.
    features, _ = extract_features(
        "СРОЧНО! Не говорите никому! Ваша карта заблокирована, штраф, отправьте "
        "код и пароль на безопасный счет http://kaspi-login.xyz "
        "http://another-link.top прямо сейчас!"
    )
    assert rule_boost(features) == 0.30


def test_rule_boost_increases_with_more_combined_signals():
    mild_features, _ = extract_features("отправьте код")
    strong_features, _ = extract_features(
        "срочно отправьте код и пароль на карту http://secure-login.xyz"
    )
    assert rule_boost(strong_features) > rule_boost(mild_features)


# =========================
# risk_level
# =========================
def test_risk_level_boundaries():
    assert risk_level(0.0)[0] == "low"
    assert risk_level(0.29)[0] == "low"
    assert risk_level(0.3)[0] == "mid"
    assert risk_level(0.59)[0] == "mid"
    assert risk_level(0.6)[0] == "high"
    assert risk_level(0.79)[0] == "high"
    assert risk_level(0.8)[0] == "critical"
    assert risk_level(1.0)[0] == "critical"


# =========================
# highlight_text (also a basic XSS-safety check)
# =========================
def test_highlight_text_wraps_known_trigger_word():
    result = highlight_text("отправьте код сейчас")
    assert '<span class="hl-secret">код</span>' in result


def test_highlight_text_wraps_links():
    result = highlight_text("click http://bad-site.xyz now")
    assert 'class="hl-link"' in result
    assert "bad-site.xyz" in result


def test_highlight_text_escapes_html_to_prevent_injection():
    result = highlight_text("<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_highlight_text_plain_safe_text_is_unwrapped():
    result = highlight_text("hello there")
    assert "<span" not in result
