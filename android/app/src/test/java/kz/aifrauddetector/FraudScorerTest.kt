package kz.aifrauddetector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * The on-device model must give the SAME probabilities as the scikit-learn
 * model it was exported from. golden.tsv (written by
 * scripts/export_mobile_model.py) holds ~400 texts - Kazakh, Russian,
 * English, emoji, odd whitespace - with sklearn's probability for each.
 */
class FraudScorerTest {

    companion object {
        lateinit var scorer: FraudScorer

        @BeforeClass @JvmStatic
        fun load() {
            scorer = File("src/main/assets/char_model.tsv").inputStream().use { FraudScorer.load(it) }
        }

        fun unescape(s: String): String {
            val sb = StringBuilder()
            var i = 0
            while (i < s.length) {
                val c = s[i]
                if (c == '\\' && i + 1 < s.length) {
                    when (s[i + 1]) {
                        't' -> sb.append('\t'); 'n' -> sb.append('\n'); 'r' -> sb.append('\r')
                        '\\' -> sb.append('\\'); else -> { sb.append(c); sb.append(s[i + 1]) }
                    }
                    i += 2
                } else { sb.append(c); i++ }
            }
            return sb.toString()
        }
    }

    @Test
    fun matchesScikitLearnOnGoldenTexts() {
        val lines = File("src/test/resources/golden.tsv").readLines(Charsets.UTF_8).filter { !it.startsWith("# ") }
        assertTrue("golden file too small", lines.size >= 300)
        var worst = 0.0
        var worstText = ""
        for (line in lines) {
            val tab = line.lastIndexOf('\t')
            val text = unescape(line.substring(0, tab))
            val expected = line.substring(tab + 1).toDouble()
            val diff = Math.abs(scorer.probability(text) - expected)
            if (diff > worst) { worst = diff; worstText = text }
        }
        assertTrue("max difference $worst on: $worstText", worst < 1e-6)
    }

    @Test
    fun obviousScamIsFlaggedAndOrdinaryMessageIsNot() {
        val scam = scorer.analyze("Срочно! Ваша карта заблокирована. Назовите код из SMS и перейдите по ссылке http://kaspi-login.xyz")
        val ok = scorer.analyze("Привет, урок математики перенесли на пятницу, не забудь тетрадь")
        assertTrue("scam p=${scam.probability}", scam.probability > 0.5)
        assertTrue("ok p=${ok.probability}", ok.probability < 0.5)
        assertTrue(scam.suspiciousWords.isNotEmpty())
    }

    @Test
    fun kazakhScamIsFlagged() {
        val p = scorer.probability("Сіздің картаңыз бұғатталды. Аударымды тоқтату үшін SMS кодты айтыңыз")
        assertTrue("p=$p", p > 0.5)
    }

    @Test
    fun emptyAndWhitespaceTextGiveTheInterceptProbability() {
        assertEquals(scorer.probability(""), scorer.probability(" \n\t  "), 0.0)
    }

    @Test
    fun ngramsMatchScikitLearnCharWb() {
        // sklearn: TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5)).build_analyzer()("Ab cd")
        assertEquals(listOf(" a", "ab", "b ", " ab", "ab ", " ab "), scorer.ngrams("Ab").filter { it.length <= 4 }.take(6))
        // a 1-letter word padded to 3 chars yields its 2- and 3-grams, then stops
        assertEquals(listOf(" x", "x ", " x "), scorer.ngrams("x"))
    }
}
