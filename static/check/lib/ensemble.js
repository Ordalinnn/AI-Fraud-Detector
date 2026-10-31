// The web app's full model, run on the device: the same 6 base models and
// the same learned combiner ("stacker") as fraud_model.FraudEnsemble.score().
//
//   lr     StandardScaler + logistic regression on the 25 rule features
//   rf     StandardScaler + random forest (50 trees) on the same features
//   gb     StandardScaler + gradient boosting (50 trees) on the same features
//   tfidf  word 1-2-gram TF-IDF + multinomial Naive Bayes
//   char   char_wb 2-5-gram TF-IDF + logistic regression (scorer.js)
//   mlp    the same TF-IDF vector, 5,000 kept n-grams -> a neural network
//          with one hidden layer (16 ReLU units, 8-bit first layer)
//   boost  the rule boost (detector.js ruleBoost, = fraud_logic.rule_boost)
//   final  logistic regression over the logits of the six + the boost
//
// Every number comes from ensemble_model.json, written by
// scripts/export_extension_model.py from the model the site trains.
// tests/ensemble.test.js checks every score against scikit-learn on
// ~4,500 texts (difference < 1e-6), so the site and the extension give the
// same probability for the same text.
//
// The rule features themselves are computed by detector.js
// (extractFeatures) and passed in, so this file has no word lists.

(function (root) {
  "use strict";

  const CharScorer = root.CharScorer
    || (typeof require === "function" ? require("./scorer.js").CharScorer : null);
  const normalizeText = root.normalizeText
    || (typeof require === "function" ? require("./normalize.js").normalizeText : null);

  const FORMAT = "fraud_ensemble_v1";
  // Python's regex \w for str patterns: letters, numbers, underscore.
  const WORD_CHAR = /[\p{L}\p{N}_]/u;

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  // fraud_model._logit: clipped so a 0 or 1 never becomes infinite
  function logit(p) {
    const q = Math.min(Math.max(p, 1e-4), 1 - 1e-4);
    return Math.log(q / (1 - q));
  }
  const camel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());

  // base64 -> signed bytes, in the browser, the MV3 service worker and Node
  function int8FromBase64(b64) {
    let bytes;
    if (typeof atob === "function") {
      const bin = atob(b64);
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else {
      bytes = Uint8Array.from(Buffer.from(b64, "base64"));
    }
    return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length);
  }

  // scikit-learn's default token_pattern (?u)\b\w\w+\b: every maximal run
  // of word characters that is at least 2 characters long.
  function wordTokens(text) {
    const out = [];
    let cur = [];
    for (const ch of text) {
      if (WORD_CHAR.test(ch)) {
        cur.push(ch);
      } else {
        if (cur.length >= 2) out.push(cur.join(""));
        cur = [];
      }
    }
    if (cur.length >= 2) out.push(cur.join(""));
    return out;
  }

  function wordNgrams(tokens, nMin, nMax) {
    const out = [];
    for (let n = nMin; n <= Math.min(nMax, tokens.length); n++) {
      for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
    }
    return out;
  }

  // sklearn trees compare float32 inputs with the stored thresholds
  function treeValue(t, xs) {
    let node = 0;
    while (t.left[node] !== -1) {
      node = Math.fround(xs[t.feature[node]]) <= t.threshold[node] ? t.left[node] : t.right[node];
    }
    return t.value[node];
  }

  function scaled(part, x) {
    return x.map((v, i) => (v - part.mean[i]) / part.scale[i]);
  }

  class EnsembleScorer {
    constructor(m) {
      if (m.format !== FORMAT) throw new Error("unknown model format: " + m.format);
      this.m = m;
      this.fingerprint = m.training_fingerprint;
      this.featureKeys = m.features.map(camel);
      this.tfidfVocab = new Map(Object.entries(m.tfidf.vocab));
      this.char = new CharScorer(new Map(Object.entries(m.char.vocab)), m.char.intercept,
        m.char.ngram_range[0], m.char.ngram_range[1], false);
      if (m.mlp) {
        const p = m.mlp;
        this.mlpRow = new Map(p.ngrams.map((g, i) => [g, i]));
        this.mlpW1 = int8FromBase64(p.w1_int8);
        if (this.mlpW1.length !== p.ngrams.length * p.hidden) throw new Error("bad mlp weights");
      }
    }

    static parse(json) {
      return new EnsembleScorer(typeof json === "string" ? JSON.parse(json) : json);
    }

    prepare(text) {
      const t = String(text == null ? "" : text);
      if (!this.m.normalize) return t;
      if (!normalizeText) throw new Error("normalize.js not loaded");
      return normalizeText(t);
    }

    vector(f) {
      return this.featureKeys.map((k) => {
        const v = f[k];
        if (typeof v !== "number") throw new Error("missing feature " + k);
        return v;
      });
    }

    lr(x) {
      const p = this.m.lr;
      const xs = scaled(p, x);
      let z = p.intercept;
      for (let i = 0; i < xs.length; i++) z += p.coef[i] * xs[i];
      return sigmoid(z);
    }

    rf(x) {
      const xs = scaled(this.m.rf, x);
      let s = 0;
      for (const t of this.m.rf.trees) s += treeValue(t, xs);
      return s / this.m.rf.trees.length;
    }

    gb(x) {
      const p = this.m.gb;
      const xs = scaled(p, x);
      let raw = p.init;
      for (const t of p.trees) raw += p.learning_rate * treeValue(t, xs);
      return sigmoid(raw);
    }

    tfidf(prepared) {
      const p = this.m.tfidf;
      const counts = new Map();
      for (const g of wordNgrams(wordTokens(prepared.toLowerCase()), p.ngram_range[0], p.ngram_range[1])) {
        if (this.tfidfVocab.has(g)) counts.set(g, (counts.get(g) || 0) + 1);
      }
      let dot = 0;
      let sq = 0;
      for (const [g, c] of counts) {
        const [idf, delta] = this.tfidfVocab.get(g);
        const w = c * idf;
        dot += w * delta;
        sq += w * w;
      }
      return sigmoid(p.prior + (sq > 0 ? dot / Math.sqrt(sq) : 0));
    }

    // Neural member. Its input is the char model's TF-IDF vector (same
    // n-grams, same idf, normalised over ALL of them, as scikit-learn does
    // before the 5,000 are kept); only the kept n-grams feed the network.
    mlp(prepared) {
      const p = this.m.mlp;
      const H = p.hidden;
      const counts = new Map();
      for (const g of this.char.ngrams(prepared)) {
        if (this.char.vocab.has(g)) counts.set(g, (counts.get(g) || 0) + 1);
      }
      let sq = 0;
      const kept = [];
      for (const [g, c] of counts) {
        const w = (1 + Math.log(c)) * this.char.vocab.get(g)[0];
        sq += w * w;
        const row = this.mlpRow.get(g);
        if (row !== undefined) kept.push([row, w]);
      }
      const norm = sq > 0 ? Math.sqrt(sq) : 1;
      const h = p.b1.slice();
      for (const [row, w] of kept) {
        const x = w / norm;
        const base = row * H;
        for (let j = 0; j < H; j++) h[j] += x * (this.mlpW1[base + j] * p.w1_scale[j]);
      }
      let z = p.b2;
      for (let j = 0; j < H; j++) if (h[j] > 0) z += h[j] * p.w2[j];
      return sigmoid(z);
    }

    // All stacker inputs + the final probability. `f` is
    // detector.js extractFeatures(text), `boost` its ruleBoost(f).
    scores(text, f, boost) {
      const x = this.vector(f);
      const prepared = this.prepare(text);
      const s = {
        lr: this.lr(x), rf: this.rf(x), gb: this.gb(x),
        tfidf: this.tfidf(prepared), char: this.char.probability(prepared), boost
      };
      if (this.m.mlp) s.mlp = this.mlp(prepared);
      const st = this.m.stacker;
      let z = st.intercept;
      st.inputs.forEach((k, i) => { z += st.coef[i] * (k === "boost" ? s[k] : logit(s[k])); });
      s.final = sigmoid(z);
      return s;
    }

    // Final probability + the words that pushed the character model up.
    analyze(text, f, boost, topWords = 3) {
      const s = this.scores(text, f, boost);
      const { suspiciousWords } = this.char.analyze(this.prepare(text), topWords);
      return { probability: s.final, suspiciousWords, scores: s };
    }
  }

  root.EnsembleScorer = EnsembleScorer;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { EnsembleScorer, wordTokens, wordNgrams };
  }
})(typeof self !== "undefined" ? self : globalThis);
