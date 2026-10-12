package kz.aifrauddetector

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * "The model was wrong" reports, kept only on this phone (Store). Exported
 * as CSV in the format of data/real/team_real_messages_TEMPLATE.csv so the
 * team can check / retrain the model on real messages
 * (scripts/import_mistakes.py). Same format as the browser extension's
 * mistakes.js. Pure Kotlin: tested in JVM unit tests.
 */
object Mistakes {
    data class Entry(val text: String, val label: Int, val language: String, val source: String, val date: String)

    const val MAX = 500
    private const val MAX_TEXT = 2000
    val CSV_COLUMNS = listOf("text", "label", "language", "scam_type", "source", "date_collected")
    private val KAZAKH_ONLY = "әғқңөұүһіӘҒҚҢӨҰҮҺІ".toSet()

    /** Same rule as fraud_eval.detect_language / mistakes.js. */
    fun detectLanguage(text: String): String {
        if (text.any { it in KAZAKH_ONLY }) return "kk"
        val cyr = text.count { it in 'а'..'я' || it in 'А'..'Я' || it == 'ё' || it == 'Ё' }
        val lat = text.count { it in 'a'..'z' || it in 'A'..'Z' }
        if (cyr == 0 && lat == 0) return "other"
        return if (cyr >= lat) "ru" else "en"
    }

    /** How a message is matched against the user's "not fraud" marks. */
    fun textKey(text: String): String = text.trim().take(MAX_TEXT)

    /** Texts the user marked "not fraud": no more warnings for them on this
     *  phone (until the marks are cleared). */
    fun notFraudSet(list: List<Entry>): Set<String> = list.filter { it.label == 0 }.map { textKey(it.text) }.toSet()

    /** The report means "the opposite is true": the label is the corrected one. */
    fun makeEntry(text: String, predictedFraud: Boolean, probability: Double, source: String, now: Date = Date()): Entry {
        val t = textKey(text)
        return Entry(
            text = t,
            label = if (predictedFraud) 0 else 1,
            language = detectLanguage(t),
            source = "$source: model said ${if (predictedFraud) "fraud" else "safe"} ${Math.round(probability * 100)}%",
            date = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(now),
        )
    }

    /** Newest first, one entry per text, capped. */
    fun addTo(list: List<Entry>, e: Entry): List<Entry> =
        if (e.text.isEmpty()) list else (listOf(e) + list.filter { it.text != e.text }).take(MAX)

    private fun cell(v: String): String {
        var s = v
        if (s.isNotEmpty() && s[0] in "=+-@\t\r") s = "'$s" // spreadsheet formula guard
        return if (s.any { it == '"' || it == ',' || it == '\n' || it == '\r' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
    }

    fun toCsv(list: List<Entry>): String {
        val sb = StringBuilder(CSV_COLUMNS.joinToString(",")).append('\n')
        for (e in list) {
            sb.append(listOf(e.text, e.label.toString(), e.language, "", e.source, e.date).joinToString(",") { cell(it) })
                .append('\n')
        }
        return sb.toString()
    }
}
