// Undoes scammers' disguises before the model reads a message:
// "к0д", "зaблoкирoвана" with Latin letters, "к о д ы", invisible
// characters, fullwidth "ＳＭＳ".
//
// Line-by-line port of text_normalize.py (the steps are explained there).
// tests/normalize.test.js checks it against normalize_golden.tsv, written
// from the Python version and shared with the Android app's test.

(function (root) {
  "use strict";

  const INVISIBLE = new Set([0x00ad, 0x180e, 0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0x2060, 0x2061,
    0x2062, 0x2063, 0x2064, 0xfeff]);
  const SPACED_SEPARATORS = new Set([" ", ".", "-", "_", "*"]);
  const LAT_TO_CYR = {
    a: "а", c: "с", e: "е", i: "і", o: "о", p: "р", x: "х", y: "у", k: "к",
    A: "А", B: "В", C: "С", E: "Е", H: "Н", I: "І", K: "К", M: "М", O: "О",
    P: "Р", T: "Т", X: "Х", Y: "У",
  };
  const CYR_TO_LAT = Object.fromEntries(Object.entries(LAT_TO_CYR).map(([k, v]) => [v, k]));

  const RE_LETTER = /^\p{L}$/u;
  const RE_DIGIT = /^\p{Nd}$/u;
  const RE_MARK = /^\p{Mn}$/u;

  const isLetter = (c) => RE_LETTER.test(c);
  const isDigit = (c) => RE_DIGIT.test(c);
  const isAlnum = (c) => isLetter(c) || isDigit(c);
  function isCyrillic(c) {
    const o = c.codePointAt(0);
    return o >= 0x0400 && o <= 0x052f;
  }
  function isLatin(c) {
    const o = c.codePointAt(0);
    return (o >= 0x41 && o <= 0x5a) || (o >= 0x61 && o <= 0x7a) || (o >= 0xc0 && o <= 0x24f && o !== 0xd7 && o !== 0xf7);
  }

  function joinSpaced(chars) {
    const out = [];
    const n = chars.length;
    let i = 0;
    while (i < n) {
      const c = chars[i];
      if (isLetter(c) && (i === 0 || !isAlnum(chars[i - 1])) && i + 2 < n && SPACED_SEPARATORS.has(chars[i + 1])) {
        const sep = chars[i + 1];
        const letters = [c];
        let j = i;
        while (j + 2 < n && chars[j + 1] === sep && isLetter(chars[j + 2]) && (j + 3 >= n || !isAlnum(chars[j + 3]))) {
          letters.push(chars[j + 2]);
          j += 2;
        }
        if (letters.length >= (sep === " " ? 4 : 3)) {
          out.push(...letters);
          i = j + 1;
          continue;
        }
      }
      out.push(c);
      i += 1;
    }
    return out;
  }

  const allLookalikes = (word, isScript, table) => word.every((c) => !isScript(c) || c in table);

  function fixWord(input, preferCyr) {
    let word = input;
    let cyr = word.filter(isCyrillic).length;
    let lat = word.filter(isLatin).length;
    if (cyr && lat) {
      const latOk = allLookalikes(word, isLatin, LAT_TO_CYR);
      const cyrOk = allLookalikes(word, isCyrillic, CYR_TO_LAT);
      if (latOk && (preferCyr || !cyrOk)) {
        word = word.map((c) => (isLatin(c) ? LAT_TO_CYR[c] || c : c));
        cyr += lat; lat = 0;
      } else if (cyrOk) {
        word = word.map((c) => (isCyrillic(c) ? CYR_TO_LAT[c] || c : c));
        lat += cyr; cyr = 0;
      }
    }
    if (!cyr && !lat) return word;
    const toCyr = cyr >= lat;
    const out = word.slice();
    const n = word.length;
    for (let k = 0; k < n; k++) {
      const c = word[k];
      if (c !== "0" && c !== "@") continue;
      const prev = k > 0 ? word[k - 1] : "";
      const next = k + 1 < n ? word[k + 1] : "";
      if (c === "0") {
        const touchesDigit = (prev && isDigit(prev)) || (next && isDigit(next));
        const touchesLetter = (prev && isLetter(prev)) || (next && isLetter(next));
        if (touchesLetter && !touchesDigit) out[k] = toCyr ? "о" : "o";
      } else if (toCyr && prev && next && isLetter(prev) && isLetter(next)) {
        out[k] = "а";
      }
    }
    return out;
  }

  function fixWords(chars) {
    const preferCyr = chars.filter(isCyrillic).length >= chars.filter(isLatin).length;
    const out = [];
    let word = [];
    for (const c of chars) {
      if (isLetter(c) || isDigit(c) || c === "@") {
        word.push(c);
      } else {
        if (word.length) { out.push(...fixWord(word, preferCyr)); word = []; }
        out.push(c);
      }
    }
    if (word.length) out.push(...fixWord(word, preferCyr));
    return out;
  }

  function normalizeText(text) {
    if (text == null) return "";
    let chars = Array.from(String(text).normalize("NFKC"));
    chars = chars.filter((c) => !INVISIBLE.has(c.codePointAt(0)) && !RE_MARK.test(c));
    chars = joinSpaced(chars);
    chars = fixWords(chars);
    return chars.join("");
  }

  // True if the message uses a disguise normalizeText undoes - a warning sign
  // of its own (port of text_normalize.is_disguised).
  function isDisguised(text) {
    if (!text) return false;
    const raw = Array.from(String(text));
    if (raw.some((c) => isLetter(c) && c.normalize("NFKC") !== c)) return true;
    const chars = Array.from(String(text).normalize("NFKC"));
    // invisible character / combining mark INSIDE a word; emoji legitimately
    // carry joiners and variation selectors ("❤️"), not between two letters
    for (let k = 1; k < chars.length - 1; k++) {
      const c = chars[k];
      if ((INVISIBLE.has(c.codePointAt(0)) || RE_MARK.test(c)) && isLetter(chars[k - 1]) && isLetter(chars[k + 1])) return true;
    }
    const stripped = chars.filter((c) => !INVISIBLE.has(c.codePointAt(0)) && !RE_MARK.test(c));
    const joined = joinSpaced(stripped);
    if (joined.join("") !== stripped.join("")) return true;
    return fixWords(joined).join("") !== joined.join("");
  }

  root.normalizeText = normalizeText;
  root.isDisguised = isDisguised;
  if (typeof module !== "undefined" && module.exports) module.exports = { normalizeText, isDisguised };
})(typeof self !== "undefined" ? self : globalThis);
