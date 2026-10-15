"""One CSV format for "the model was wrong" reports (site, extension,
Android) and the script that merges testers' files."""
import importlib.util
from datetime import date
from pathlib import Path

import mistakes

ROOT = Path(__file__).resolve().parent.parent


def _import_script():
    spec = importlib.util.spec_from_file_location("import_mistakes", ROOT / "scripts" / "import_mistakes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_row_has_corrected_label_and_template_columns():
    r = mistakes.make_row("Привет, урок перенесли", True, 0.71, "site", today=date(2026, 10, 12))
    assert r == {"text": "Привет, урок перенесли", "label": 0, "language": "ru", "scam_type": "",
                 "source": "site: model said fraud 71%", "date_collected": "2026-10-12"}
    header = (ROOT / "data" / "real" / "team_real_messages_TEMPLATE.csv").read_text(encoding="utf-8").splitlines()[0]
    assert mistakes.to_csv([]).strip() == header.strip()


def test_csv_same_as_extension_and_android():
    csv = mistakes.to_csv([
        {"text": 'Карта, "заблокирована"\nсрочно', "label": 1, "language": "ru", "scam_type": "",
         "source": "web.whatsapp.com", "date_collected": "2026-10-12"},
        {"text": "=HYPERLINK(1)", "label": 0, "language": "en", "scam_type": "", "source": "popup",
         "date_collected": "2026-10-12"},
    ])
    lines = csv.split("\n")
    assert lines[1] == '"Карта, ""заблокирована""'
    assert lines[2] == 'срочно",1,ru,,web.whatsapp.com,2026-10-12'
    assert lines[3] == "'=HYPERLINK(1),0,en,,popup,2026-10-12"


def test_import_merges_dedupes_drops_conflicts_and_training_texts(tmp_path):
    m = _import_script()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("﻿" + mistakes.to_csv([
        mistakes.make_row("Сіздің картаңыз бұғатталды, кодты айтыңыз", False, 0.3, "app"),
        mistakes.make_row("-50% скидка только сегодня", True, 0.8, "popup"),
        mistakes.make_row("спорное сообщение", True, 0.6, "site"),
        mistakes.make_row("training text", True, 0.6, "site"),
    ]), encoding="utf-8")
    b.write_text(mistakes.to_csv([
        mistakes.make_row("Сіздің картаңыз бұғатталды, кодты айтыңыз", False, 0.2, "web.whatsapp.com"),
        mistakes.make_row("спорное сообщение", False, 0.4, "app"),
    ]), encoding="utf-8")
    rows = m.read_rows(a) + m.read_rows(b)
    merged, conflicts, in_training = m.merge(rows, {"training text"})
    texts = {r["text"]: r for r in merged}
    assert set(texts) == {"Сіздің картаңыз бұғатталды, кодты айтыңыз", "-50% скидка только сегодня"}
    assert texts["Сіздің картаңыз бұғатталды, кодты айтыңыз"]["label"] == 1
    assert "(reported 2x)" in texts["Сіздің картаңыз бұғатталды, кодты айтыңыз"]["source"]
    assert texts["-50% скидка только сегодня"]["label"] == 0, "formula guard undone"
    assert conflicts == ["спорное сообщение"]
    assert in_training == 1


def test_csv_quotes_semicolons_and_tabs_like_the_extension():
    import csv as _csv
    import io as _io
    row = mistakes.make_row("Привет;=cmd|' /C calc'!A0\tx", False, 0.03, "site")
    out = mistakes.to_csv([row])
    assert out.split("\n")[1].startswith('"Привет;=cmd|')
    back = list(_csv.reader(_io.StringIO(out)))
    assert back[1][0] == "Привет;=cmd|' /C calc'!A0\tx"
