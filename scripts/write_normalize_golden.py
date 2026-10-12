"""
Writes android/app/src/test/resources/normalize_golden.tsv: raw text ->
text_normalize.normalize_text(raw), for the Kotlin (TextNormalizerTest) and
JavaScript (browser-extension/tests/normalize.test.js) ports to match.

    python scripts/write_normalize_golden.py

Inputs: hand-picked tricky cases + ~300 dataset messages, each also
disguised with the attacks from scripts/eval_obfuscation.py.
tests/test_text_normalize.py fails if the file is out of date.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_normalize import normalize_text  # noqa: E402

GOLDEN = ROOT / "android" / "app" / "src" / "test" / "resources" / "normalize_golden.tsv"

TRICKY = [
    "", " ", "Ваша к@рта зaблoкирoвана, назовите к0д из S M S", "к.о.д", "К О Д Ы", "к-о-д-ы", "к_о_д", "к*о*д",
    "а я в шоке", "т.е. завтра", "Ф.И.О", "20000тг переведите", "0тправьте код", "kаspi.kz", "http://kаspi-bank.kz/login",
    "WhatsAppқа жазыңыз", "iPhoneды", "Kaspiге", "Your сodе is 1234, do nоt shаre it", "ＳＭＳ ｋｏｄ", "𝐤𝐨𝐝 𝐢𝐳 𝐒𝐌𝐒",
    "к\u200bо\u200bд", "к\u00adо\u00adд", "к̲о̲д", "\ufeffСрочно", "user@mail.ru", "@kaspi_support", "2fa0", "x0x",
    "К A S P I", "с.р.о.ч.н.о", "Сіздің картаңыз бұғатталды", "ҚАЗАҚСТАН", "ok оk OK", "МОСКВА MOCKBA", "Ёлка й",
    "😀 к0д 😀", "a b c d e", "1 2 3 4", "ab cd ef gh", "в и к а", "x.y", "e-mail", "Wi-Fi", "SMS-ті жіберіңіз",
    "\t к о д ы \n", "Ｋａｓｐｉ Ｇｏｌｄ", "Ⅻ ①②", "ﬁnance", "Ｉ", "İstanbul", "ß straße",
]


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def inputs() -> list[str]:
    spec = importlib.util.spec_from_file_location("eval_obfuscation", ROOT / "scripts" / "eval_obfuscation.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    import fraud_eval as fe
    texts = fe.load_dataset()["text"].tolist()
    pick = random.Random(0).sample(texts, 300)
    out = list(TRICKY)
    for i, t in enumerate(pick):
        name = list(ev.ATTACKS)[i % len(ev.ATTACKS)]
        out.append(ev.ATTACKS[name](t))
    return out


def render() -> str:
    rows = ["# raw<TAB>normalized (text_normalize.py); \\t \\n \\r \\\\ escaped"]
    for t in inputs():
        rows.append(f"{esc(t)}\t{esc(normalize_text(t))}")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    GOLDEN.write_text(render(), encoding="utf-8")
    print(f"wrote {GOLDEN}")
