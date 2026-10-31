"""The bank-notification data and the ForteBank brand check."""
import csv
from pathlib import Path

import fraud_logic
import fraud_model as fm

ROOT = Path(__file__).resolve().parent.parent


def test_training_file_is_reproducible_and_loaded():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "make_bank_notifications.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    rows = mk.build()
    with open(fm.BANK_TRAIN_PATH, encoding="utf-8") as f:
        saved = list(csv.DictReader(f))
    assert [r["text"] for r in rows] == [r["text"] for r in saved], "re-run scripts/make_bank_notifications.py"
    loaded = fm.load_bank_notifications()
    assert len(loaded) == len(rows) and {l for _, l in loaded} == {0, 1}


def test_bank_data_is_part_of_training():
    extra = fm.extra_training_examples()
    assert len(extra) == len(fm.load_real_training_sample()) + len(fm.load_bank_notifications())


def test_test_set_is_not_in_training():
    train = {t for t, _ in fm.load_bank_notifications()}
    test = fm.load_bank_notifications_test()
    assert not train & set(test["text"])
    assert set(test["label"]) == {0, 1}


def test_fortebank_lookalike_domains_flagged_real_ones_not():
    for d in ["forte-secure.top", "f0rte-otmena.site", "fortebank-kz.info"]:
        assert fraud_logic.brand_impersonation(d) == "forte.kz"
    for d in ["forte.kz", "support.forte.kz", "comfort-hotel.kz"]:
        assert fraud_logic.brand_impersonation(d) is None
