package kz.aifrauddetector

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * On-device fraud model: character n-grams + logistic regression, exported
 * from the web app by scripts/export_mobile_model.py.
 *
 * This is a line-by-line port of `reference_proba` in that script, which in
 * turn is checked against scikit-learn to 1e-9; FraudScorerTest checks this
 * class against probabilities computed by scikit-learn (golden.tsv). Text is
 * handled as Unicode code points, like Python strings, so emoji and other
 * characters outside the BMP give the same n-grams as on the server.
 *
 * Pure Kotlin, no Android classes: runs in JVM unit tests unchanged.
 */
class FraudScorer private constructor(
    private val idf: Map<String, Double>,
    private val coef: Map<String, Double>,
    private val intercept: Double,
    private val nMin: Int,
    private val nMax: Int,
) {
    data class Result(val probability: Double, val suspiciousWords: List<String>)

    fun probability(text: String): Double {
        val counts = HashMap<String, Int>()
        for (g in ngrams(text)) {
            if (idf.containsKey(g)) counts[g] = (counts[g] ?: 0) + 1
        }
        var dot = 0.0
        var sq = 0.0
        for ((g, c) in counts) {
            val w = (1.0 + ln(c.toDouble())) * idf.getValue(g)
            dot += w * coef.getValue(g)
            sq += w * w
        }
        val z = intercept + if (sq > 0) dot / sqrt(sq) else 0.0
        return 1.0 / (1.0 + exp(-z))
    }

    /** Probability plus the words that pushed the score up the most - a
     *  simple, honest explanation: each word's n-grams' weights summed. */
    fun analyze(text: String, topWords: Int = 3): Result {
        val scored = words(text).distinct().map { word ->
            word to wordNgrams(word).sumOf { g -> (coef[g] ?: 0.0) * (idf[g] ?: 0.0) }
        }
        val top = scored.filter { it.second > 0 }.sortedByDescending { it.second }.take(topWords).map { it.first }
        return Result(probability(text), top)
    }

    // ---------- tokenisation identical to scikit-learn's char_wb ----------

    private fun words(text: String): List<String> {
        val out = ArrayList<String>()
        val cur = StringBuilder()
        val lower = pyLower(text)
        var i = 0
        while (i < lower.length) {
            val cp = lower.codePointAt(i)
            if (isPyWhitespace(cp)) {
                if (cur.isNotEmpty()) { out.add(cur.toString()); cur.setLength(0) }
            } else {
                cur.appendCodePoint(cp)
            }
            i += Character.charCount(cp)
        }
        if (cur.isNotEmpty()) out.add(cur.toString())
        return out
    }

    private fun wordNgrams(word: String): List<String> {
        val cps = (" $word ").codePoints().toArray()
        val out = ArrayList<String>()
        for (n in nMin..nMax) {
            var offset = 0
            out.add(slice(cps, offset, n))
            while (offset + n < cps.size) {
                offset += 1
                out.add(slice(cps, offset, n))
            }
            if (offset == 0) break // word shorter than n: counted once
        }
        return out
    }

    fun ngrams(text: String): List<String> = words(text).flatMap { wordNgrams(it) }

    private fun slice(cps: IntArray, from: Int, n: Int): String {
        val end = minOf(from + n, cps.size)
        return String(cps, from, end - from)
    }

    companion object {
        /** Python's str.split() whitespace (see PY_WHITESPACE in the export script). */
        fun isPyWhitespace(cp: Int): Boolean = when (cp) {
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680,
            0x2028, 0x2029, 0x202F, 0x205F, 0x3000 -> true
            in 0x2000..0x200A -> true
            else -> false
        }

        /** Python str.lower(): per-code-point lowercase, with the one
         *  common multi-char case (İ -> i + combining dot) handled. */
        fun pyLower(s: String): String {
            val sb = StringBuilder(s.length)
            var i = 0
            while (i < s.length) {
                val cp = s.codePointAt(i)
                if (cp == 0x130) sb.append("i̇") else sb.appendCodePoint(Character.toLowerCase(cp))
                i += Character.charCount(cp)
            }
            return sb.toString()
        }

        fun load(input: InputStream): FraudScorer {
            val idf = HashMap<String, Double>(32768)
            val coef = HashMap<String, Double>(32768)
            var intercept = 0.0
            var nMin = 2
            var nMax = 5
            BufferedReader(InputStreamReader(input, Charsets.UTF_8)).useLines { lines ->
                for (line in lines) {
                    // Comments never contain a tab; every model line does (an n-gram
                    // itself may start with "# ", e.g. from the word "c#").
                    if (line.isEmpty() || (line.startsWith("#") && '\t' !in line)) continue
                    val parts = line.split('\t')
                    when {
                        parts[0] == "intercept" && parts.size == 2 -> intercept = parts[1].toDouble()
                        parts[0] == "ngram_range" && parts.size == 3 -> {
                            nMin = parts[1].toInt(); nMax = parts[2].toInt()
                        }
                        parts.size == 3 -> {
                            idf[parts[0]] = parts[1].toDouble()
                            coef[parts[0]] = parts[2].toDouble()
                        }
                        else -> error("bad model line: $line")
                    }
                }
            }
            require(idf.isNotEmpty()) { "empty model" }
            return FraudScorer(idf, coef, intercept, nMin, nMax)
        }
    }
}
