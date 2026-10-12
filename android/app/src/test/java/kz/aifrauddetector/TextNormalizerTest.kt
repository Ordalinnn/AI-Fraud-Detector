package kz.aifrauddetector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * TextNormalizer must turn every text into exactly what text_normalize.py
 * does (normalize_golden.tsv, written by scripts/write_normalize_golden.py,
 * shared with the browser extension's test).
 */
class TextNormalizerTest {

    @Test
    fun matchesPythonOnEveryGoldenText() {
        val lines = File("src/test/resources/normalize_golden.tsv").readLines(Charsets.UTF_8)
            .filter { it.isNotEmpty() && !it.startsWith("# ") }
        assertTrue("golden file too small", lines.size >= 300)
        val bad = ArrayList<String>()
        for (line in lines) {
            val tab = line.indexOf('\t')
            val raw = FraudScorerTest.unescape(line.substring(0, tab))
            val expected = FraudScorerTest.unescape(line.substring(tab + 1))
            val got = TextNormalizer.normalize(raw)
            if (got != expected) bad.add("$raw -> $got (expected $expected)")
        }
        assertTrue(bad.take(5).joinToString("\n"), bad.isEmpty())
    }

    @Test
    fun disguisedScamScoresLikeThePlainOne() {
        val scorer = File("src/main/assets/char_model.tsv").inputStream().use { FraudScorer.load(it) }
        val plain = scorer.probability("Ваша карта заблокирована, назовите код из SMS")
        val disguised = scorer.probability("Ваша к@рта зaблoкирoвана, назовите к0д из S​MS")
        assertEquals(plain, disguised, 1e-12)
        assertTrue(plain > 0.5)
    }
}
