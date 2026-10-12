package kz.aifrauddetector

import java.text.Normalizer

/**
 * Undoes scammers' disguises before the model reads a message:
 * "к0д", "зaблoкирoвана" with Latin letters, "к о д ы", invisible
 * characters, fullwidth "ＳＭＳ".
 *
 * Line-by-line port of text_normalize.py (see the steps there).
 * TextNormalizerTest checks it against normalize_golden.tsv, written from
 * the Python version; the browser extension's normalize.js is checked
 * against the same file.
 */
object TextNormalizer {

    private val INVISIBLE = setOf(
        0x00AD, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF,
    )
    private val SPACED_SEPARATORS = setOf(' '.code, '.'.code, '-'.code, '_'.code, '*'.code)

    private val LAT_TO_CYR: Map<Int, Int> = mapOf(
        "a" to "а", "c" to "с", "e" to "е", "i" to "і", "o" to "о", "p" to "р", "x" to "х", "y" to "у", "k" to "к",
        "A" to "А", "B" to "В", "C" to "С", "E" to "Е", "H" to "Н", "I" to "І", "K" to "К", "M" to "М", "O" to "О",
        "P" to "Р", "T" to "Т", "X" to "Х", "Y" to "У",
    ).entries.associate { it.key.codePointAt(0) to it.value.codePointAt(0) }
    private val CYR_TO_LAT: Map<Int, Int> = LAT_TO_CYR.entries.associate { it.value to it.key }

    private const val ZERO = '0'.code
    private const val AT = '@'.code
    private const val CYR_O = 0x043E // о
    private const val LAT_O = 0x6F   // o
    private const val CYR_A = 0x0430 // а

    private fun isLetter(cp: Int) = Character.isLetter(cp)
    private fun isDigit(cp: Int) = Character.getType(cp) == Character.DECIMAL_DIGIT_NUMBER.toInt()
    private fun isCyrillic(cp: Int) = cp in 0x0400..0x052F
    private fun isLatin(cp: Int) = cp in 0x41..0x5A || cp in 0x61..0x7A || (cp in 0xC0..0x24F && cp != 0xD7 && cp != 0xF7)
    private fun isAlnum(cp: Int) = isLetter(cp) || isDigit(cp)

    fun normalize(text: String): String {
        var cps = Normalizer.normalize(text, Normalizer.Form.NFKC).codePoints().toArray().toList()
        cps = cps.filter { it !in INVISIBLE && Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
        cps = joinSpaced(cps)
        cps = fixWords(cps)
        val sb = StringBuilder()
        for (cp in cps) sb.appendCodePoint(cp)
        return sb.toString()
    }

    private fun joinSpaced(cps: List<Int>): List<Int> {
        val out = ArrayList<Int>(cps.size)
        val n = cps.size
        var i = 0
        while (i < n) {
            val c = cps[i]
            if (isLetter(c) && (i == 0 || !isAlnum(cps[i - 1])) && i + 2 < n && cps[i + 1] in SPACED_SEPARATORS) {
                val sep = cps[i + 1]
                val letters = arrayListOf(c)
                var j = i
                while (j + 2 < n && cps[j + 1] == sep && isLetter(cps[j + 2]) && (j + 3 >= n || !isAlnum(cps[j + 3]))) {
                    letters.add(cps[j + 2])
                    j += 2
                }
                if (letters.size >= (if (sep == ' '.code) 4 else 3)) {
                    out.addAll(letters)
                    i = j + 1
                    continue
                }
            }
            out.add(c)
            i += 1
        }
        return out
    }

    private fun allLookalikes(word: List<Int>, isScript: (Int) -> Boolean, table: Map<Int, Int>) =
        word.all { !isScript(it) || it in table }

    private fun fixWord(input: List<Int>, preferCyr: Boolean): List<Int> {
        var word = input
        var cyr = word.count { isCyrillic(it) }
        var lat = word.count { isLatin(it) }
        if (cyr > 0 && lat > 0) {
            val latOk = allLookalikes(word, ::isLatin, LAT_TO_CYR)
            val cyrOk = allLookalikes(word, ::isCyrillic, CYR_TO_LAT)
            if (latOk && (preferCyr || !cyrOk)) {
                word = word.map { if (isLatin(it)) LAT_TO_CYR[it] ?: it else it }
                cyr += lat; lat = 0
            } else if (cyrOk) {
                word = word.map { if (isCyrillic(it)) CYR_TO_LAT[it] ?: it else it }
                lat += cyr; cyr = 0
            }
        }
        if (cyr == 0 && lat == 0) return word
        val toCyr = cyr >= lat
        val out = word.toMutableList()
        val n = word.size
        for (k in word.indices) {
            val c = word[k]
            if (c != ZERO && c != AT) continue
            val prev = if (k > 0) word[k - 1] else -1
            val next = if (k + 1 < n) word[k + 1] else -1
            if (c == ZERO) {
                val touchesDigit = (prev >= 0 && isDigit(prev)) || (next >= 0 && isDigit(next))
                val touchesLetter = (prev >= 0 && isLetter(prev)) || (next >= 0 && isLetter(next))
                if (touchesLetter && !touchesDigit) out[k] = if (toCyr) CYR_O else LAT_O
            } else if (toCyr && prev >= 0 && next >= 0 && isLetter(prev) && isLetter(next)) {
                out[k] = CYR_A
            }
        }
        return out
    }

    private fun fixWords(cps: List<Int>): List<Int> {
        val preferCyr = cps.count { isCyrillic(it) } >= cps.count { isLatin(it) }
        val out = ArrayList<Int>(cps.size)
        val word = ArrayList<Int>()
        for (c in cps) {
            if (isLetter(c) || isDigit(c) || c == AT) {
                word.add(c)
            } else {
                if (word.isNotEmpty()) { out.addAll(fixWord(word, preferCyr)); word.clear() }
                out.add(c)
            }
        }
        if (word.isNotEmpty()) out.addAll(fixWord(word, preferCyr))
        return out
    }
}
