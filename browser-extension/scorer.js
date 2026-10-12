// On-device ML model: character n-grams + logistic regression.
//
// The same model as the Android app (android/.../FraudScorer.kt) and the
// strongest single member of the web app's ensemble (93.6% on the jury's
// 282-message test). Exported from scikit-learn by
// scripts/export_mobile_model.py into char_model.tsv; this file is a port
// of that script's `reference_proba`, and tests/scorer.test.js checks it
// against scikit-learn's own probabilities on ~400 texts.
//
// Text is processed as Unicode code points (like Python strings), so emoji
// and other characters outside the BMP produce the same n-grams as on the
// server. Classic script (no ES modules) so it loads in the MV3 service
// worker via importScripts, in the popup via <script>, and in Node tests.

(function (root) {
  "use strict";

  const normalizeText = root.normalizeText
    || (typeof require === "function" ? require("./normalize.js").normalizeText : null);

  // Python's str.split() whitespace set.
  function isPyWhitespace(cp) {
    switch (cp) {
      case 0x09: case 0x0a: case 0x0b: case 0x0c: case 0x0d:
      case 0x1c: case 0x1d: case 0x1e: case 0x1f: case 0x20:
      case 0x85: case 0xa0: case 0x1680: case 0x2028: case 0x2029:
      case 0x202f: case 0x205f: case 0x3000:
        return true;
      default:
        return cp >= 0x2000 && cp <= 0x200a;
    }
  }

  function words(text) {
    const out = [];
    let cur = "";
    for (const ch of String(text == null ? "" : text).toLowerCase()) {
      if (isPyWhitespace(ch.codePointAt(0))) {
        if (cur) { out.push(cur); cur = ""; }
      } else {
        cur += ch;
      }
    }
    if (cur) out.push(cur);
    return out;
  }

  class CharScorer {
    constructor(vocab, intercept, nMin, nMax, normalize) {
      this.vocab = vocab; // Map: n-gram -> [idf, coef]
      this.intercept = intercept;
      this.nMin = nMin;
      this.nMax = nMax;
      // model trained on de-disguised text (text_normalize.py): undo the
      // same disguises here before reading n-grams
      this.normalize = !!normalize;
    }

    prepare(text) {
      const t = String(text == null ? "" : text);
      if (!this.normalize) return t;
      if (!normalizeText) throw new Error("normalize.js not loaded");
      return normalizeText(t);
    }

    // scikit-learn's char_wb analyzer for one word.
    wordNgrams(word) {
      const cps = Array.from(" " + word + " ");
      const out = [];
      for (let n = this.nMin; n <= this.nMax; n++) {
        let offset = 0;
        out.push(cps.slice(offset, offset + n).join(""));
        while (offset + n < cps.length) {
          offset += 1;
          out.push(cps.slice(offset, offset + n).join(""));
        }
        if (offset === 0) break; // word shorter than n: counted once
      }
      return out;
    }

    ngrams(text) {
      return words(text).flatMap((w) => this.wordNgrams(w));
    }

    probability(text) {
      const counts = new Map();
      for (const g of this.ngrams(this.prepare(text))) {
        if (this.vocab.has(g)) counts.set(g, (counts.get(g) || 0) + 1);
      }
      let dot = 0;
      let sq = 0;
      for (const [g, c] of counts) {
        const [idf, coef] = this.vocab.get(g);
        const w = (1 + Math.log(c)) * idf;
        dot += w * coef;
        sq += w * w;
      }
      const z = this.intercept + (sq > 0 ? dot / Math.sqrt(sq) : 0);
      return 1 / (1 + Math.exp(-z));
    }

    // Probability + the words that pushed the score up the most.
    analyze(text, topWords = 3) {
      const scored = [...new Set(words(this.prepare(text)))].map((w) => {
        let s = 0;
        for (const g of this.wordNgrams(w)) {
          const v = this.vocab.get(g);
          if (v) s += v[0] * v[1];
        }
        return [w, s];
      });
      // shown to people: drop punctuation stuck to the word ("заблокирована,")
      const clean = (w) => w.replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "") || w;
      const suspiciousWords = [...new Set(scored.filter((x) => x[1] > 0)
        .sort((a, b) => b[1] - a[1]).map((x) => clean(x[0])))].slice(0, topWords);
      return { probability: this.probability(text), suspiciousWords };
    }

    static parse(tsv) {
      const vocab = new Map();
      let intercept = 0;
      let nMin = 2;
      let nMax = 5;
      let normalize = false;
      for (const line of tsv.split("\n")) {
        // Comments never contain a tab; every model line does (an n-gram
        // itself may start with "#", e.g. from the word "c#").
        if (!line || (line[0] === "#" && line.indexOf("\t") < 0)) continue;
        const parts = line.split("\t");
        if (parts[0] === "intercept" && parts.length === 2) intercept = parseFloat(parts[1]);
        else if (parts[0] === "normalize" && parts.length === 2) normalize = parts[1] === "text_normalize_v1";
        else if (parts[0] === "ngram_range" && parts.length === 3) {
          nMin = parseInt(parts[1], 10);
          nMax = parseInt(parts[2], 10);
        } else if (parts.length === 3) vocab.set(parts[0], [parseFloat(parts[1]), parseFloat(parts[2])]);
        else throw new Error("bad model line: " + line.slice(0, 60));
      }
      if (!vocab.size) throw new Error("empty model");
      return new CharScorer(vocab, intercept, nMin, nMax, normalize);
    }
  }

  root.CharScorer = CharScorer;
  if (typeof module !== "undefined" && module.exports) module.exports = { CharScorer, words, isPyWhitespace };
})(typeof self !== "undefined" ? self : globalThis);
